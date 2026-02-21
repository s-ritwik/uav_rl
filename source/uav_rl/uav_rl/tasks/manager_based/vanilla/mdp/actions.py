from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from ..controllers import PX4LikeVelocityController, RotorAllocator, RotorMotorModel, quat_wxyz_to_xyzw


@configclass
class PX4LikeVelocityActionCfg(ActionTermCfg):
    """Action term config for policy actions [vx, vy, vz, yaw_rate]."""

    class_type: type[ActionTerm] = MISSING

    body_name: str = "body"
    rotor_names: tuple[str, str, str, str] = ("rotor0", "rotor1", "rotor2", "rotor3")

    action_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    action_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    velocity_limits: tuple[float, float, float] = (5.0, 5.0, 3.0)
    yaw_rate_limit: float = 2.5

    # Controller parameters (Pegasus example 12 defaults).
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

    # Pegasus-style motor + mapper defaults.
    rotor_constant: tuple[float, float, float, float] = (8.54858e-6, 8.54858e-6, 8.54858e-6, 8.54858e-6)
    rolling_moment_coefficient: tuple[float, float, float, float] = (1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6)
    rot_dir: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)
    min_rotor_velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    max_rotor_velocity: tuple[float, float, float, float] = (1100.0, 1100.0, 1100.0, 1100.0)
    drag_coefficients: tuple[float, float, float] = (0.50, 0.30, 0.0)

    input_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    input_scaling: tuple[float, float, float, float] = (1000.0, 1000.0, 1000.0, 1000.0)
    zero_position_armed: tuple[float, float, float, float] = (100.0, 100.0, 100.0, 100.0)
    control_range: tuple[float, float] = (0.0, 1.0)

    fallback_arm_length: float = 0.17


class PX4LikeVelocityAction(ActionTerm):
    """
    Manager action term that mirrors Pegasus example 12 architecture.

    Tick semantics are preserved:
    - apply cached motor command this physics tick,
    - compute next motor command from latest state for the next tick.
    """

    cfg: PX4LikeVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: PX4LikeVelocityActionCfg, env):
        super().__init__(cfg, env)

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"Expected exactly one body matching '{self.cfg.body_name}', got {body_names}")
        self._body_id = body_ids[0]
        self._body_name = body_names[0]

        rotor_ids, rotor_names = self._asset.find_bodies(list(self.cfg.rotor_names), preserve_order=True)
        if len(rotor_ids) != 4:
            raise ValueError(
                f"Expected exactly 4 rotor bodies matching {self.cfg.rotor_names}, got {rotor_names}"
            )
        self._rotor_ids = rotor_ids
        self._rotor_names = rotor_names
        self._wrench_body_ids = self._rotor_ids + [self._body_id]

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._velocity_sp = torch.zeros((self.num_envs, 3), device=self.device)
        self._yaw_rate_sp = torch.zeros((self.num_envs,), device=self.device)

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device).unsqueeze(0)
        self._action_offset = torch.tensor(self.cfg.action_offset, device=self.device).unsqueeze(0)
        self._velocity_limits = torch.tensor(self.cfg.velocity_limits, device=self.device)

        self._controller = PX4LikeVelocityController(
            num_envs=self.num_envs,
            device=self.device,
            dtype=self._raw_actions.dtype,
            mass=self.cfg.mass,
            gravity=self.cfg.gravity,
            max_tilt_deg=self.cfg.max_tilt_deg,
            thrust_limits=self.cfg.thrust_limits,
            velocity_p_gains=self.cfg.velocity_p_gains,
            velocity_i_gains=self.cfg.velocity_i_gains,
            velocity_d_gains=self.cfg.velocity_d_gains,
            velocity_integrator_limits=self.cfg.velocity_integrator_limits,
            velocity_accel_limits=self.cfg.velocity_accel_limits,
            attitude_p_gains=self.cfg.attitude_p_gains,
            rate_p_gains=self.cfg.rate_p_gains,
            rate_i_gains=self.cfg.rate_i_gains,
            rate_d_gains=self.cfg.rate_d_gains,
            rate_limits=self.cfg.rate_limits,
            rate_integrator_limits=self.cfg.rate_integrator_limits,
            torque_limits=self.cfg.torque_limits,
            input_offset=self.cfg.input_offset,
            input_scaling=self.cfg.input_scaling,
            zero_position_armed=self.cfg.zero_position_armed,
            control_range=self.cfg.control_range,
        )

        self._motor_model = RotorMotorModel(
            rotor_constant=self.cfg.rotor_constant,
            rolling_moment_coefficient=self.cfg.rolling_moment_coefficient,
            rot_dir=self.cfg.rot_dir,
            drag_coefficients=self.cfg.drag_coefficients,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

        self._allocator: RotorAllocator | None = None

        self._cached_motor_omega = torch.zeros((self.num_envs, 4), device=self.device)
        self._last_hil_controls = torch.zeros((self.num_envs, 4), device=self.device)
        self._last_torque_sp = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_thrust_sp = torch.zeros((self.num_envs,), device=self.device)

        self._forces = torch.zeros((self.num_envs, 5, 3), device=self.device)
        self._torques = torch.zeros((self.num_envs, 5, 3), device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def last_hil_controls(self) -> torch.Tensor:
        return self._last_hil_controls

    @property
    def last_motor_omega(self) -> torch.Tensor:
        return self._cached_motor_omega

    @property
    def last_torque_sp(self) -> torch.Tensor:
        return self._last_torque_sp

    @property
    def last_thrust_sp(self) -> torch.Tensor:
        return self._last_thrust_sp

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        processed = self._raw_actions * self._action_scale + self._action_offset
        processed[:, :3] = torch.clamp(processed[:, :3], min=-self._velocity_limits, max=self._velocity_limits)
        processed[:, 3] = torch.clamp(processed[:, 3], min=-self.cfg.yaw_rate_limit, max=self.cfg.yaw_rate_limit)

        self._processed_actions[:] = processed
        self._velocity_sp[:] = processed[:, :3]
        self._yaw_rate_sp[:] = processed[:, 3]

    def _fallback_rotor_positions(self) -> torch.Tensor:
        l = float(self.cfg.fallback_arm_length)
        return torch.tensor(
            [
                [l, l, 0.0],
                [-l, l, 0.0],
                [-l, -l, 0.0],
                [l, -l, 0.0],
            ],
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

    def _maybe_initialize_allocator(self):
        if self._allocator is not None:
            return

        body_pos_w = self._asset.data.body_pos_w[:, self._body_id]
        body_quat_w = self._asset.data.body_quat_w[:, self._body_id]
        rotor_pos_w = self._asset.data.body_pos_w[:, self._rotor_ids]

        rel_w = rotor_pos_w - body_pos_w.unsqueeze(1)
        rel_w_flat = rel_w.reshape(-1, 3)
        quat_flat = body_quat_w.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4)
        rel_b = quat_apply_inverse(quat_flat, rel_w_flat).reshape(self.num_envs, 4, 3)

        rotor_positions_b = rel_b[0].detach().clone()
        if torch.linalg.norm(rotor_positions_b[:, :2], dim=-1).min().item() < 1.0e-6:
            rotor_positions_b = self._fallback_rotor_positions()

        self._allocator = RotorAllocator(
            rotor_positions_b=rotor_positions_b,
            rotor_constant=self.cfg.rotor_constant,
            rolling_moment_coefficient=self.cfg.rolling_moment_coefficient,
            rot_dir=self.cfg.rot_dir,
            max_rotor_velocity=self.cfg.max_rotor_velocity,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

    def _apply_cached_wrench(self):
        rotor_forces, rolling_moment = self._motor_model.omega_to_forces(self._cached_motor_omega)
        drag_force = self._motor_model.body_drag(self._asset.data.root_lin_vel_b)

        self._forces.zero_()
        self._torques.zero_()

        self._forces[:, :4, 2] = rotor_forces
        self._forces[:, 4, :] = drag_force
        self._torques[:, 4, 2] = rolling_moment

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._wrench_body_ids,
            forces=self._forces,
            torques=self._torques,
        )

    def _compute_next_command(self):
        self._maybe_initialize_allocator()

        attitude_xyzw = quat_wxyz_to_xyzw(self._asset.data.root_quat_w)
        body_rates = self._asset.data.root_ang_vel_b
        velocity_w = self._asset.data.root_lin_vel_w

        outputs = self._controller.step_velocity_mode(
            attitude_xyzw=attitude_xyzw,
            body_rates=body_rates,
            velocity_w=velocity_w,
            velocity_sp=self._velocity_sp,
            yaw_rate_sp=self._yaw_rate_sp,
            dt=float(self._env.physics_dt),
            accel_ff=None,
        )

        self._last_torque_sp = outputs["torque_sp"]
        self._last_thrust_sp = outputs["thrust_sp"]

        motor_omega = self._allocator.force_torque_to_omega(outputs["thrust_sp"], outputs["torque_sp"])
        self._last_hil_controls = self._controller.hil_mapper.motor_omega_to_hil_controls(motor_omega)
        self._cached_motor_omega = self._controller.hil_mapper.hil_controls_to_motor_omega(self._last_hil_controls)

    def apply_actions(self):
        # 1) Apply command computed on previous physics tick.
        self._apply_cached_wrench()
        # 2) Compute command for the next physics tick.
        self._compute_next_command()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._velocity_sp.zero_()
            self._yaw_rate_sp.zero_()
            self._cached_motor_omega.zero_()
            self._last_hil_controls.zero_()
            self._last_torque_sp.zero_()
            self._last_thrust_sp.zero_()
            self._controller.reset(None)
            return

        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._velocity_sp[ids] = 0.0
        self._yaw_rate_sp[ids] = 0.0
        self._cached_motor_omega[ids] = 0.0
        self._last_hil_controls[ids] = 0.0
        self._last_torque_sp[ids] = 0.0
        self._last_thrust_sp[ids] = 0.0
        self._controller.reset(ids)


PX4LikeVelocityActionCfg.class_type = PX4LikeVelocityAction
