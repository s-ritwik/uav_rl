"""
Controller helpers for the manager-based vanilla UAV task.
Based on Pegasus example 12, adapted for batched Isaac Lab execution.
"""

from __future__ import annotations

import torch

from .px4_like_pipeline import HilActuatorMapper, PX4LikeMulticopterCascade, yaw_from_quaternion


def quat_wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.stack((quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]), dim=-1)


class RotorAllocator:
    """Converts total thrust + body torques into per-rotor angular velocities."""

    def __init__(
        self,
        rotor_positions_b: torch.Tensor,
        rotor_constant,
        rolling_moment_coefficient,
        rot_dir,
        max_rotor_velocity,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype

        self.rotor_positions_b = rotor_positions_b.to(device=device, dtype=dtype)
        self.rotor_constant = torch.as_tensor(rotor_constant, device=device, dtype=dtype)
        self.rolling_moment_coefficient = torch.as_tensor(rolling_moment_coefficient, device=device, dtype=dtype)
        self.rot_dir = torch.as_tensor(rot_dir, device=device, dtype=dtype)
        self.max_rotor_velocity = torch.as_tensor(max_rotor_velocity, device=device, dtype=dtype)

        if self.rotor_positions_b.shape != (4, 3):
            raise ValueError("RotorAllocator expects rotor_positions_b shape (4, 3).")

        allocation = torch.zeros((4, 4), device=device, dtype=dtype)
        allocation[0, :] = self.rotor_constant
        allocation[1, :] = self.rotor_positions_b[:, 1] * self.rotor_constant
        allocation[2, :] = -self.rotor_positions_b[:, 0] * self.rotor_constant
        allocation[3, :] = self.rolling_moment_coefficient * self.rot_dir

        self._allocation = allocation
        self._allocation_inv = torch.linalg.pinv(allocation)
        self._max_rotor_vel_sq = torch.square(self.max_rotor_velocity)

    def force_torque_to_omega(self, thrust_sp: torch.Tensor, torque_sp: torch.Tensor) -> torch.Tensor:
        desired = torch.cat((thrust_sp.unsqueeze(-1), torque_sp), dim=-1)
        omega_sq = torch.matmul(desired, self._allocation_inv.T)
        omega_sq = torch.clamp(omega_sq, min=0.0)

        max_allowed_sq = torch.max(self._max_rotor_vel_sq)
        max_value = torch.max(omega_sq, dim=-1, keepdim=True).values
        normalize = torch.maximum(max_value / max_allowed_sq, torch.ones_like(max_value))
        omega_sq = omega_sq / normalize

        return torch.sqrt(omega_sq)


class RotorMotorModel:
    """Quadratic thrust model used in Pegasus (T = k * omega^2)."""

    def __init__(self, rotor_constant, rolling_moment_coefficient, rot_dir, drag_coefficients, device, dtype):
        self.rotor_constant = torch.as_tensor(rotor_constant, device=device, dtype=dtype)
        self.rolling_moment_coefficient = torch.as_tensor(rolling_moment_coefficient, device=device, dtype=dtype)
        self.rot_dir = torch.as_tensor(rot_dir, device=device, dtype=dtype)
        self.drag_coeff = torch.as_tensor(drag_coefficients, device=device, dtype=dtype)

    def omega_to_forces(self, omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        omega_sq = torch.square(omega)
        rotor_forces = omega_sq * self.rotor_constant
        rolling_moment = torch.sum(omega_sq * self.rolling_moment_coefficient * self.rot_dir, dim=-1)
        return rotor_forces, rolling_moment

    def body_drag(self, body_velocity: torch.Tensor) -> torch.Tensor:
        return -(body_velocity * self.drag_coeff)


class PX4LikeVelocityController:
    """
    Batch controller for policy actions [vx, vy, vz, yaw_rate].

    The internal pipeline is:
    velocity -> acceleration -> attitude -> rates -> body torque.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
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
        input_offset,
        input_scaling,
        zero_position_armed,
        control_range,
    ):
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        self._cascade = PX4LikeMulticopterCascade(
            num_envs=num_envs,
            mass=mass,
            gravity=gravity,
            max_tilt_deg=max_tilt_deg,
            thrust_limits=thrust_limits,
            velocity_p_gains=velocity_p_gains,
            velocity_i_gains=velocity_i_gains,
            velocity_d_gains=velocity_d_gains,
            velocity_integrator_limits=velocity_integrator_limits,
            velocity_accel_limits=velocity_accel_limits,
            attitude_p_gains=attitude_p_gains,
            rate_p_gains=rate_p_gains,
            rate_i_gains=rate_i_gains,
            rate_d_gains=rate_d_gains,
            rate_limits=rate_limits,
            rate_integrator_limits=rate_integrator_limits,
            torque_limits=torque_limits,
            device=device,
            dtype=dtype,
        )

        self.hil_mapper = HilActuatorMapper(
            input_offset=input_offset,
            input_scaling=input_scaling,
            zero_position_armed=zero_position_armed,
            control_min=control_range[0],
            control_max=control_range[1],
            device=device,
            dtype=dtype,
        )

        self._yaw_sp = torch.zeros((num_envs,), device=device, dtype=dtype)
        self._yaw_initialized = torch.zeros((num_envs,), device=device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | None = None):
        self._cascade.reset(env_ids)
        if env_ids is None:
            self._yaw_sp.zero_()
            self._yaw_initialized[:] = False
        else:
            self._yaw_sp[env_ids] = 0.0
            self._yaw_initialized[env_ids] = False

    def step_velocity_mode(
        self,
        attitude_xyzw: torch.Tensor,
        body_rates: torch.Tensor,
        velocity_w: torch.Tensor,
        velocity_sp: torch.Tensor,
        yaw_rate_sp: torch.Tensor,
        dt: float,
        accel_ff: torch.Tensor | None = None,
    ):
        need_init = ~self._yaw_initialized
        if need_init.any():
            self._yaw_sp[need_init] = yaw_from_quaternion(attitude_xyzw[need_init])
            self._yaw_initialized[need_init] = True

        self._yaw_sp = self._yaw_sp + yaw_rate_sp * float(dt)

        return self._cascade.step_from_velocity(
            attitude=attitude_xyzw,
            body_rates=body_rates,
            velocity=velocity_w,
            velocity_sp=velocity_sp,
            yaw_sp=self._yaw_sp,
            yaw_rate_sp=yaw_rate_sp,
            dt=dt,
            accel_ff=accel_ff,
        )
