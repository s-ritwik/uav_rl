"""
Torch implementation of a PX4-like multicopter control pipeline.
This is adapted from Pegasus example 12 for batched Isaac Lab usage.
Quaternion convention in this file is [qx, qy, qz, qw].
"""

import math

import torch


def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _safe_norm(vec: torch.Tensor, eps: float, keepdim: bool = False) -> torch.Tensor:
    return torch.linalg.norm(vec, dim=-1, keepdim=keepdim).clamp_min(eps)


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / _safe_norm(q, eps=eps, keepdim=True)


def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    x, y, z, w = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def rot_matrix_to_quat(rotation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    original_shape = rotation.shape[:-2]
    rot = rotation.reshape(-1, 3, 3)

    m00 = rot[:, 0, 0]
    m01 = rot[:, 0, 1]
    m02 = rot[:, 0, 2]
    m10 = rot[:, 1, 0]
    m11 = rot[:, 1, 1]
    m12 = rot[:, 1, 2]
    m20 = rot[:, 2, 0]
    m21 = rot[:, 2, 1]
    m22 = rot[:, 2, 2]

    trace = m00 + m11 + m22
    quat = torch.zeros((rot.shape[0], 4), device=rot.device, dtype=rot.dtype)

    cond0 = trace > 0.0
    cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
    cond2 = (~cond0) & (~cond1) & (m11 > m22)
    cond3 = (~cond0) & (~cond1) & (~cond2)

    if cond0.any():
        s = torch.sqrt((trace[cond0] + 1.0).clamp_min(eps)) * 2.0
        quat[cond0, 3] = 0.25 * s
        quat[cond0, 0] = (m21[cond0] - m12[cond0]) / s
        quat[cond0, 1] = (m02[cond0] - m20[cond0]) / s
        quat[cond0, 2] = (m10[cond0] - m01[cond0]) / s

    if cond1.any():
        s = torch.sqrt((1.0 + m00[cond1] - m11[cond1] - m22[cond1]).clamp_min(eps)) * 2.0
        quat[cond1, 3] = (m21[cond1] - m12[cond1]) / s
        quat[cond1, 0] = 0.25 * s
        quat[cond1, 1] = (m01[cond1] + m10[cond1]) / s
        quat[cond1, 2] = (m02[cond1] + m20[cond1]) / s

    if cond2.any():
        s = torch.sqrt((1.0 + m11[cond2] - m00[cond2] - m22[cond2]).clamp_min(eps)) * 2.0
        quat[cond2, 3] = (m02[cond2] - m20[cond2]) / s
        quat[cond2, 0] = (m01[cond2] + m10[cond2]) / s
        quat[cond2, 1] = 0.25 * s
        quat[cond2, 2] = (m12[cond2] + m21[cond2]) / s

    if cond3.any():
        s = torch.sqrt((1.0 + m22[cond3] - m00[cond3] - m11[cond3]).clamp_min(eps)) * 2.0
        quat[cond3, 3] = (m10[cond3] - m01[cond3]) / s
        quat[cond3, 0] = (m02[cond3] + m20[cond3]) / s
        quat[cond3, 1] = (m12[cond3] + m21[cond3]) / s
        quat[cond3, 2] = 0.25 * s

    quat = quat_normalize(quat, eps=eps)
    return quat.reshape(*original_shape, 4)


def yaw_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    x, y, z, w = q.unbind(dim=-1)
    yaw_num = 2.0 * (w * z + x * y)
    yaw_den = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(yaw_num, yaw_den)


def vee(S: torch.Tensor) -> torch.Tensor:
    return torch.stack((-S[..., 1, 2], S[..., 0, 2], -S[..., 0, 1]), dim=-1)


class AccelYawRateToAttitude:
    def __init__(
        self,
        mass: float,
        gravity: float,
        max_tilt_deg: float,
        max_thrust: float,
        min_thrust: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self._eps = float(torch.finfo(dtype).eps)

        self.mass = _as_tensor(mass, device, dtype)
        self.gravity = _as_tensor(gravity, device, dtype)
        self.max_thrust = _as_tensor(max_thrust, device, dtype)
        self.min_thrust = _as_tensor(min_thrust, device, dtype)
        self._world_z = _as_tensor([0.0, 0.0, 1.0], device, dtype)
        self.max_tilt_rad = _as_tensor(math.radians(max_tilt_deg), device, dtype)
        self._cos_max_tilt = torch.cos(self.max_tilt_rad)
        self._sin_max_tilt = torch.sin(self.max_tilt_rad)

    def _limit_tilt(self, z_b_des: torch.Tensor) -> torch.Tensor:
        z_b_des = z_b_des.clone()
        mask = z_b_des[:, 2] < self._cos_max_tilt
        if mask.any():
            xy = z_b_des[mask, :2]
            xy_norm = _safe_norm(xy, self._eps, keepdim=True)
            xy = xy * (self._sin_max_tilt / xy_norm)
            z_b_des[mask, :2] = xy
            z_b_des[mask, 2] = self._cos_max_tilt
            z_b_des[mask] = z_b_des[mask] / _safe_norm(z_b_des[mask], self._eps, keepdim=True)
        return z_b_des

    def update(self, accel_sp: torch.Tensor, yaw_sp: torch.Tensor):
        force_sp = self.mass * (accel_sp + self.gravity * self._world_z)
        thrust_sp = _safe_norm(force_sp, self._eps)
        thrust_sp = torch.clamp(thrust_sp, min=self.min_thrust, max=self.max_thrust)

        z_b_des = force_sp / _safe_norm(force_sp, self._eps, keepdim=True)
        z_b_des = self._limit_tilt(z_b_des)

        x_c_des = torch.stack((torch.cos(yaw_sp), torch.sin(yaw_sp), torch.zeros_like(yaw_sp)), dim=-1)
        y_b_des = torch.cross(z_b_des, x_c_des, dim=-1)

        y_norm = _safe_norm(y_b_des, self._eps)
        singular = y_norm < self._eps
        if singular.any():
            y_b_des[singular] = torch.stack(
                (-torch.sin(yaw_sp[singular]), torch.cos(yaw_sp[singular]), torch.zeros_like(yaw_sp[singular])),
                dim=-1,
            )

        y_b_des = y_b_des / _safe_norm(y_b_des, self._eps, keepdim=True)
        x_b_des = torch.cross(y_b_des, z_b_des, dim=-1)
        x_b_des = x_b_des / _safe_norm(x_b_des, self._eps, keepdim=True)

        rotation_sp = torch.stack((x_b_des, y_b_des, z_b_des), dim=-1)
        attitude_sp = rot_matrix_to_quat(rotation_sp, eps=self._eps)
        return attitude_sp, rotation_sp, thrust_sp


class AttitudePController:
    def __init__(self, gains, max_rates, device: torch.device, dtype: torch.dtype):
        self.gains = _as_tensor(gains, device, dtype)
        self.max_rates = _as_tensor(max_rates, device, dtype)

    def update(self, attitude: torch.Tensor, rotation_sp: torch.Tensor, yaw_rate_sp: torch.Tensor):
        rotation = quat_to_rot_matrix(attitude)
        attitude_error = 0.5 * vee(torch.matmul(rotation_sp.transpose(-1, -2), rotation) - torch.matmul(rotation.transpose(-1, -2), rotation_sp))
        rates_sp = -(self.gains * attitude_error)
        rates_sp[:, 2] = rates_sp[:, 2] + yaw_rate_sp
        rates_sp = torch.clamp(rates_sp, min=-self.max_rates, max=self.max_rates)
        return rates_sp, attitude_error


class RatePIDController:
    def __init__(
        self,
        num_envs: int,
        p_gains,
        i_gains,
        d_gains,
        integrator_limit,
        torque_limit,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._eps = float(torch.finfo(dtype).eps)
        self.p_gains = _as_tensor(p_gains, device, dtype)
        self.i_gains = _as_tensor(i_gains, device, dtype)
        self.d_gains = _as_tensor(d_gains, device, dtype)
        self.integrator_limit = _as_tensor(integrator_limit, device, dtype)
        self.torque_limit = _as_tensor(torque_limit, device, dtype)

        self.integral = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.prev_rates = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.initialized = torch.zeros((num_envs,), device=device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.integral.zero_()
            self.prev_rates.zero_()
            self.initialized[:] = False
        else:
            self.integral[env_ids] = 0.0
            self.prev_rates[env_ids] = 0.0
            self.initialized[env_ids] = False

    def update(self, rates_sp: torch.Tensor, rates: torch.Tensor, dt: float):
        safe_dt = max(float(dt), self._eps)

        first_update = ~self.initialized
        if first_update.any():
            self.prev_rates[first_update] = rates[first_update]
            self.initialized[first_update] = True

        rates_dot = (rates - self.prev_rates) / safe_dt
        rate_error = rates_sp - rates

        self.integral = self.integral + rate_error * safe_dt
        self.integral = torch.clamp(self.integral, min=-self.integrator_limit, max=self.integrator_limit)

        torque_sp = self.p_gains * rate_error + self.i_gains * self.integral - self.d_gains * rates_dot
        torque_sp = torch.clamp(torque_sp, min=-self.torque_limit, max=self.torque_limit)

        self.prev_rates = rates
        return torque_sp, rate_error, rates_dot


class VelocityPIDController:
    def __init__(
        self,
        num_envs: int,
        p_gains,
        i_gains,
        d_gains,
        integrator_limit,
        accel_limit_xy: float,
        accel_limit_up: float,
        accel_limit_down: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._eps = float(torch.finfo(dtype).eps)
        self.p_gains = _as_tensor(p_gains, device, dtype)
        self.i_gains = _as_tensor(i_gains, device, dtype)
        self.d_gains = _as_tensor(d_gains, device, dtype)
        self.integrator_limit = _as_tensor(integrator_limit, device, dtype)
        self.accel_limit_xy = _as_tensor(accel_limit_xy, device, dtype)
        self.accel_limit_up = _as_tensor(accel_limit_up, device, dtype)
        self.accel_limit_down = _as_tensor(accel_limit_down, device, dtype)

        self.integral = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.prev_error = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.initialized = torch.zeros((num_envs,), device=device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.integral.zero_()
            self.prev_error.zero_()
            self.initialized[:] = False
        else:
            self.integral[env_ids] = 0.0
            self.prev_error[env_ids] = 0.0
            self.initialized[env_ids] = False

    def update(self, vel_sp: torch.Tensor, vel: torch.Tensor, dt: float, accel_ff: torch.Tensor | None = None):
        if accel_ff is None:
            accel_ff = torch.zeros_like(vel_sp)

        safe_dt = max(float(dt), self._eps)
        vel_error = vel_sp - vel

        first_update = ~self.initialized
        if first_update.any():
            self.prev_error[first_update] = vel_error[first_update]
            self.initialized[first_update] = True

        vel_error_dot = (vel_error - self.prev_error) / safe_dt
        self.integral = self.integral + vel_error * safe_dt
        self.integral = torch.clamp(self.integral, min=-self.integrator_limit, max=self.integrator_limit)

        accel_sp = accel_ff + self.p_gains * vel_error + self.i_gains * self.integral + self.d_gains * vel_error_dot

        accel_xy = accel_sp[:, :2]
        accel_xy_norm = _safe_norm(accel_xy, self._eps, keepdim=True)
        over_xy = accel_xy_norm.squeeze(-1) > self.accel_limit_xy
        if over_xy.any():
            accel_sp[over_xy, :2] = accel_xy[over_xy] * (self.accel_limit_xy / accel_xy_norm[over_xy])

        accel_sp[:, 2] = torch.clamp(accel_sp[:, 2], min=-self.accel_limit_down, max=self.accel_limit_up)

        self.prev_error = vel_error
        return accel_sp, vel_error, vel_error_dot


class HilActuatorMapper:
    def __init__(
        self,
        input_offset,
        input_scaling,
        zero_position_armed,
        control_min,
        control_max,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.input_offset = _as_tensor(input_offset, device, dtype)
        self.input_scaling = _as_tensor(input_scaling, device, dtype)
        self.zero_position_armed = _as_tensor(zero_position_armed, device, dtype)
        self.control_min = _as_tensor(control_min, device, dtype)
        self.control_max = _as_tensor(control_max, device, dtype)

    def motor_omega_to_hil_controls(self, rotor_omega: torch.Tensor) -> torch.Tensor:
        controls = (rotor_omega - self.zero_position_armed) / self.input_scaling - self.input_offset
        return torch.clamp(controls, min=self.control_min, max=self.control_max)

    def hil_controls_to_motor_omega(self, controls: torch.Tensor) -> torch.Tensor:
        controls = torch.clamp(controls, min=self.control_min, max=self.control_max)
        return (controls + self.input_offset) * self.input_scaling + self.zero_position_armed


class PX4LikeMulticopterCascade:
    def __init__(
        self,
        num_envs: int,
        mass: float,
        gravity: float,
        max_tilt_deg: float,
        thrust_limits,
        velocity_p_gains,
        velocity_i_gains,
        velocity_d_gains,
        velocity_integrator_limits,
        velocity_accel_limits,
        attitude_p_gains,
        rate_p_gains,
        rate_i_gains,
        rate_d_gains,
        rate_limits,
        rate_integrator_limits,
        torque_limits,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if len(velocity_accel_limits) == 2:
            accel_limit_xy = velocity_accel_limits[0]
            accel_limit_up = velocity_accel_limits[1]
            accel_limit_down = velocity_accel_limits[1]
        elif len(velocity_accel_limits) == 3:
            accel_limit_xy = velocity_accel_limits[0]
            accel_limit_up = velocity_accel_limits[1]
            accel_limit_down = velocity_accel_limits[2]
        else:
            raise ValueError("velocity_accel_limits must be (xy, z) or (xy, up, down)")

        self.velocity_controller = VelocityPIDController(
            num_envs=num_envs,
            p_gains=velocity_p_gains,
            i_gains=velocity_i_gains,
            d_gains=velocity_d_gains,
            integrator_limit=velocity_integrator_limits,
            accel_limit_xy=accel_limit_xy,
            accel_limit_up=accel_limit_up,
            accel_limit_down=accel_limit_down,
            device=device,
            dtype=dtype,
        )
        self.accel_to_attitude = AccelYawRateToAttitude(
            mass=mass,
            gravity=gravity,
            max_tilt_deg=max_tilt_deg,
            max_thrust=thrust_limits[1],
            min_thrust=thrust_limits[0],
            device=device,
            dtype=dtype,
        )
        self.attitude_controller = AttitudePController(
            gains=attitude_p_gains,
            max_rates=rate_limits,
            device=device,
            dtype=dtype,
        )
        self.rate_controller = RatePIDController(
            num_envs=num_envs,
            p_gains=rate_p_gains,
            i_gains=rate_i_gains,
            d_gains=rate_d_gains,
            integrator_limit=rate_integrator_limits,
            torque_limit=torque_limits,
            device=device,
            dtype=dtype,
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        self.velocity_controller.reset(env_ids)
        self.rate_controller.reset(env_ids)

    def step_from_acceleration(
        self,
        attitude: torch.Tensor,
        body_rates: torch.Tensor,
        accel_sp: torch.Tensor,
        yaw_sp: torch.Tensor,
        yaw_rate_sp: torch.Tensor,
        dt: float,
    ):
        attitude_sp, rotation_sp, thrust_sp = self.accel_to_attitude.update(accel_sp=accel_sp, yaw_sp=yaw_sp)
        rates_sp, attitude_error = self.attitude_controller.update(
            attitude=attitude, rotation_sp=rotation_sp, yaw_rate_sp=yaw_rate_sp
        )
        torque_sp, rates_error, rates_dot = self.rate_controller.update(rates_sp=rates_sp, rates=body_rates, dt=dt)
        return {
            "accel_sp": accel_sp,
            "attitude_sp": attitude_sp,
            "rotation_sp": rotation_sp,
            "thrust_sp": thrust_sp,
            "rates_sp": rates_sp,
            "torque_sp": torque_sp,
            "attitude_error": attitude_error,
            "rates_error": rates_error,
            "rates_dot": rates_dot,
            "velocity_error": None,
            "velocity_error_dot": None,
        }

    def step_from_velocity(
        self,
        attitude: torch.Tensor,
        body_rates: torch.Tensor,
        velocity: torch.Tensor,
        velocity_sp: torch.Tensor,
        yaw_sp: torch.Tensor,
        yaw_rate_sp: torch.Tensor,
        dt: float,
        accel_ff: torch.Tensor | None = None,
    ):
        accel_sp, vel_error, vel_error_dot = self.velocity_controller.update(
            vel_sp=velocity_sp, vel=velocity, dt=dt, accel_ff=accel_ff
        )
        outputs = self.step_from_acceleration(
            attitude=attitude,
            body_rates=body_rates,
            accel_sp=accel_sp,
            yaw_sp=yaw_sp,
            yaw_rate_sp=yaw_rate_sp,
            dt=dt,
        )
        outputs["velocity_error"] = vel_error
        outputs["velocity_error_dot"] = vel_error_dot
        return outputs


# Backward-compatible alias.
PX4LikePostAccelPipeline = PX4LikeMulticopterCascade
