from __future__ import annotations

from dataclasses import dataclass

import torch

from pegasus.simulator.logic.backends import Backend, BackendConfig

try:
    from .px4_like_controller import PX4LikeVelocityController
except ImportError:
    from px4_like_controller import PX4LikeVelocityController


def _normalize_quat_xyzw(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(eps)


def _quat_xyzw_conjugate(quat: torch.Tensor) -> torch.Tensor:
    xyz = -quat[..., :3]
    w = quat[..., 3:]
    return torch.cat((xyz, w), dim=-1)


def _quat_xyzw_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = lhs.unbind(dim=-1)
    x2, y2, z2, w2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_xyzw_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat_xyzw(quat)
    zeros = torch.zeros((*vec.shape[:-1], 1), device=vec.device, dtype=vec.dtype)
    vec_quat = torch.cat((vec, zeros), dim=-1)
    return _quat_xyzw_multiply(_quat_xyzw_multiply(quat, vec_quat), _quat_xyzw_conjugate(quat))[..., :3]


def _quat_xyzw_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _quat_xyzw_apply(_quat_xyzw_conjugate(_normalize_quat_xyzw(quat)), vec)


def _quat_xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.stack((quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2]), dim=-1)


@dataclass
class PolicyFlightBackendConfig(BackendConfig):
    """Configuration for Pegasus policy playback with the vanilla controller contract."""

    policy: object
    platform: object
    physics_hz: float = 250.0
    policy_hz: float = 25.0
    device: str = "cpu"
    action_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    action_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    velocity_limits: tuple[float, float, float] = (6.0, 6.0, 4.0)
    yaw_rate_limit: float = 3.0
    mass: float = 1.5
    gravity: float = 9.81
    max_tilt_deg: float = 50.0
    thrust_limits: tuple[float, float] = (0.0, 35.0)
    velocity_p_gains: tuple[float, float, float] = (4.0, 4.0, 6.5)
    velocity_i_gains: tuple[float, float, float] = (0.2, 0.2, 1.4)
    velocity_d_gains: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_integrator_limits: tuple[float, float, float] = (2.0, 2.0, 2.0)
    velocity_accel_limits: tuple[float, float, float] = (8.0, 6.0, 6.0)
    attitude_p_gains: tuple[float, float, float] = (6.0, 6.0, 3.0)
    rate_p_gains: tuple[float, float, float] = (0.20, 0.20, 0.10)
    rate_i_gains: tuple[float, float, float] = (0.10, 0.10, 0.08)
    rate_d_gains: tuple[float, float, float] = (0.004, 0.004, 0.002)
    rate_limits: tuple[float, float, float] = (3.5, 3.5, 2.5)
    rate_integrator_limits: tuple[float, float, float] = (1.0, 1.0, 0.8)
    torque_limits: tuple[float, float, float] = (0.6, 0.6, 0.25)
    input_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    input_scaling: tuple[float, float, float, float] = (1000.0, 1000.0, 1000.0, 1000.0)
    zero_position_armed: tuple[float, float, float, float] = (100.0, 100.0, 100.0, 100.0)
    control_range: tuple[float, float] = (0.0, 1.0)


class PolicyFlightBackend(Backend):
    """Pegasus backend that feeds a trained vanilla policy into the PX4-like controller."""

    def __init__(self, config: PolicyFlightBackendConfig):
        super().__init__(config=config)

        self.cfg = config
        self.device = torch.device(config.device)
        self.dtype = torch.float32

        ratio = float(config.physics_hz) / float(config.policy_hz)
        self.control_decimation = max(1, int(round(ratio)))
        if abs(ratio - self.control_decimation) > 1.0e-6:
            raise ValueError(
                f"physics_hz / policy_hz must be an integer ratio. Got {config.physics_hz} / {config.policy_hz}."
            )

        self._controller = PX4LikeVelocityController(
            num_envs=1,
            device=self.device,
            dtype=self.dtype,
            mass=config.mass,
            gravity=config.gravity,
            max_tilt_deg=config.max_tilt_deg,
            thrust_limits=config.thrust_limits,
            velocity_p_gains=config.velocity_p_gains,
            velocity_i_gains=config.velocity_i_gains,
            velocity_d_gains=config.velocity_d_gains,
            velocity_integrator_limits=config.velocity_integrator_limits,
            velocity_accel_limits=config.velocity_accel_limits,
            attitude_p_gains=config.attitude_p_gains,
            rate_p_gains=config.rate_p_gains,
            rate_i_gains=config.rate_i_gains,
            rate_d_gains=config.rate_d_gains,
            rate_limits=config.rate_limits,
            rate_integrator_limits=config.rate_integrator_limits,
            torque_limits=config.torque_limits,
            input_offset=config.input_offset,
            input_scaling=config.input_scaling,
            zero_position_armed=config.zero_position_armed,
            control_range=config.control_range,
        )

        self._action_scale = torch.tensor(config.action_scale, device=self.device, dtype=self.dtype).unsqueeze(0)
        self._action_offset = torch.tensor(config.action_offset, device=self.device, dtype=self.dtype).unsqueeze(0)
        self._velocity_limits = torch.tensor(config.velocity_limits, device=self.device, dtype=self.dtype)

        self._cached_motor_omega = torch.zeros((1, 4), device=self.device, dtype=self.dtype)
        self._raw_action = torch.zeros((1, 4), device=self.device, dtype=self.dtype)
        self._velocity_sp = torch.zeros((1, 3), device=self.device, dtype=self.dtype)
        self._yaw_rate_sp = torch.zeros((1,), device=self.device, dtype=self.dtype)
        self._received_first_state = False
        self._tick = 0
        self._state = None

    def start(self):
        self.reset()

    def stop(self):
        pass

    def reset(self):
        self._controller.reset()
        self._cached_motor_omega.zero_()
        self._raw_action.zero_()
        self._velocity_sp.zero_()
        self._yaw_rate_sp.zero_()
        self._received_first_state = False
        self._tick = 0
        self._state = None

    def update_sensor(self, sensor_type: str, data):
        del sensor_type, data

    def update_graphical_sensor(self, sensor_type: str, data):
        del sensor_type, data

    def update_state(self, state):
        self._state = state
        self._received_first_state = True

    def input_reference(self):
        return self._cached_motor_omega[0]

    def update(self, dt: float):
        if not self._received_first_state or self._state is None:
            return

        if self._tick % self.control_decimation == 0:
            obs = self._build_observation()
            with torch.inference_mode():
                self._raw_action = self.cfg.policy.act(obs).to(self.device, dtype=self.dtype)
            self._process_action()

        attitude_xyzw = self._tensor(self._state.attitude)
        body_rates = self._tensor(self._state.angular_velocity)
        velocity_w = self._tensor(self._state.linear_velocity)

        outputs = self._controller.step_velocity_mode(
            attitude_xyzw=attitude_xyzw.unsqueeze(0),
            body_rates=body_rates.unsqueeze(0),
            velocity_w=velocity_w.unsqueeze(0),
            velocity_sp=self._velocity_sp,
            yaw_rate_sp=self._yaw_rate_sp,
            dt=float(dt),
            accel_ff=None,
        )

        motor_omega = self.vehicle.force_and_torques_to_velocities(
            outputs["thrust_sp"][0],
            outputs["torque_sp"][0],
        )
        self._cached_motor_omega = torch.as_tensor(motor_omega, device=self.device, dtype=self.dtype).unsqueeze(0)
        self._tick += 1

    def _tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _process_action(self) -> None:
        processed = self._raw_action * self._action_scale + self._action_offset
        processed[:, :3] = torch.clamp(processed[:, :3], min=-self._velocity_limits, max=self._velocity_limits)
        processed[:, 3] = torch.clamp(
            processed[:, 3],
            min=-float(self.cfg.yaw_rate_limit),
            max=float(self.cfg.yaw_rate_limit),
        )
        self._velocity_sp[:] = processed[:, :3]
        self._yaw_rate_sp[:] = processed[:, 3]

    def _build_observation(self) -> torch.Tensor:
        platform_state = self.cfg.platform.current_state
        if platform_state is None:
            raise RuntimeError("Platform state is not initialized.")

        robot_pos_w = self._tensor(self._state.position).unsqueeze(0)
        robot_quat_xyzw = _normalize_quat_xyzw(self._tensor(self._state.attitude).unsqueeze(0))
        robot_lin_vel_w = self._tensor(self._state.linear_velocity).unsqueeze(0)
        robot_ang_vel_b = self._tensor(self._state.angular_velocity).unsqueeze(0)
        robot_ang_vel_w = _quat_xyzw_apply(robot_quat_xyzw, robot_ang_vel_b)

        platform_pos_w = self._tensor(platform_state.position).unsqueeze(0)
        platform_quat_xyzw = _normalize_quat_xyzw(self._tensor(platform_state.quat_xyzw).unsqueeze(0))
        platform_lin_vel_w = self._tensor(platform_state.linear_velocity).unsqueeze(0)
        platform_ang_vel_w = self._tensor(platform_state.angular_velocity).unsqueeze(0)

        rel_pos = _quat_xyzw_apply_inverse(platform_quat_xyzw, robot_pos_w - platform_pos_w)
        rel_lin_vel = _quat_xyzw_apply_inverse(platform_quat_xyzw, robot_lin_vel_w - platform_lin_vel_w)
        rel_quat_xyzw = _quat_xyzw_multiply(_quat_xyzw_conjugate(platform_quat_xyzw), robot_quat_xyzw)
        rel_quat_wxyz = _quat_xyzw_to_wxyz(rel_quat_xyzw)
        rel_ang_vel = _quat_xyzw_apply_inverse(platform_quat_xyzw, robot_ang_vel_w - platform_ang_vel_w)

        gravity_world = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=self.dtype)
        projected_gravity = _quat_xyzw_apply_inverse(robot_quat_xyzw, gravity_world)

        return torch.cat(
            (
                rel_pos,
                rel_lin_vel,
                rel_quat_wxyz,
                rel_ang_vel,
                projected_gravity,
                self._raw_action,
            ),
            dim=-1,
        )
