from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

from ...vanilla.controllers import HilActuatorMapper, RotorMotorModel
from ...vanilla.mdp.actions import PX4LikeVelocityActionCfg
from ...vanilla.mdp.randomization import get_domain_randomization_state
from .runtime import ArduPilotFineTuneRuntimeState, State


@configclass
class ArduPilotGuidedVelocityActionCfg(PX4LikeVelocityActionCfg):
    """Action term that keeps the vanilla policy contract but routes commands through ArduPilot SITL."""

    class_type: type[ActionTerm] = MISSING


class ArduPilotGuidedVelocityAction(ActionTerm):
    cfg: ArduPilotGuidedVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: ArduPilotGuidedVelocityActionCfg, env):
        super().__init__(cfg, env)

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"Expected exactly one body matching '{self.cfg.body_name}', got {body_names}")
        self._body_id = body_ids[0]

        rotor_ids, rotor_names = self._asset.find_bodies(list(self.cfg.rotor_names), preserve_order=True)
        if len(rotor_ids) != 4:
            raise ValueError(f"Expected 4 rotor bodies matching {self.cfg.rotor_names}, got {rotor_names}")
        self._rotor_ids = rotor_ids
        self._wrench_body_ids = self._rotor_ids + [self._body_id]
        self._propeller_visual_enabled = bool(self.cfg.propeller_visual_enabled)
        self._propeller_visual_warned = False
        self._rotor_joint_ids: list[int] = []
        self._rotor_joint_names: list[str] = []

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._velocity_sp = torch.zeros((self.num_envs, 3), device=self.device)
        self._yaw_rate_sp = torch.zeros((self.num_envs,), device=self.device)

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device).unsqueeze(0)
        self._action_offset = torch.tensor(self.cfg.action_offset, device=self.device).unsqueeze(0)
        self._velocity_limits = torch.tensor(self.cfg.velocity_limits, device=self.device)
        self._rotor_direction = torch.tensor(
            self.cfg.rot_dir, device=self.device, dtype=self._raw_actions.dtype
        ).unsqueeze(0)
        self._rotor_visual_joint_vel = torch.zeros((self.num_envs, 4), device=self.device, dtype=self._raw_actions.dtype)
        self._thrust_asymmetry_scale = torch.ones((self.num_envs, 4), device=self.device, dtype=self._raw_actions.dtype)
        self._motor_lag_alpha = torch.ones((self.num_envs, 1), device=self.device, dtype=self._raw_actions.dtype)

        domain_rand_cfg = getattr(env.cfg, "domain_randomization", None)
        max_action_delay = 0
        if domain_rand_cfg is not None:
            max_action_delay = int(domain_rand_cfg.action_delay_steps_range[1])
        self._action_delay_buffer = DelayBuffer(max_action_delay, self.num_envs, self.device)

        self._motor_model = RotorMotorModel(
            rotor_constant=self.cfg.rotor_constant,
            rolling_moment_coefficient=self.cfg.rolling_moment_coefficient,
            rot_dir=self.cfg.rot_dir,
            drag_coefficients=self.cfg.drag_coefficients,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._hil_mapper = HilActuatorMapper(
            input_offset=self.cfg.input_offset,
            input_scaling=self.cfg.input_scaling,
            zero_position_armed=self.cfg.zero_position_armed,
            control_min=self.cfg.control_range[0],
            control_max=self.cfg.control_range[1],
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

        self._cached_motor_omega = torch.zeros((self.num_envs, 4), device=self.device)
        self._last_hil_controls = torch.zeros((self.num_envs, 4), device=self.device)
        self._prev_root_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_root_lin_vel_initialized = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._forces = torch.zeros((self.num_envs, 5, 3), device=self.device)
        self._torques = torch.zeros((self.num_envs, 5, 3), device=self.device)

        self._runtime: ArduPilotFineTuneRuntimeState | None = None
        self._skip_next_runtime_reset = False

    def __del__(self):
        try:
            if self._runtime is not None:
                self._runtime.stop()
        except Exception:
            pass
        super().__del__()

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

    def all_envs_ready_for_policy(self) -> bool:
        self._ensure_runtime()
        return self._runtime.all_envs_ready_for_policy()

    def num_ready_envs(self) -> int:
        self._ensure_runtime()
        return self._runtime.num_ready_envs()

    def current_altitudes(self) -> list[float]:
        self._ensure_runtime()
        return self._runtime.current_altitudes()

    def debug_statuses(self) -> list[dict[str, object]]:
        self._ensure_runtime()
        return self._runtime.debug_statuses()

    def ready_mask(self) -> torch.Tensor:
        self._ensure_runtime()
        ready = [
            bool(handle.guided_client is not None and handle.guided_client.ready_for_velocity_commands)
            for handle in self._runtime.envs
        ]
        return torch.tensor(ready, device=self.device, dtype=torch.bool)

    def skip_next_runtime_reset(self):
        self._skip_next_runtime_reset = True

    def _ensure_runtime(self):
        runtime_cfg = self._env.cfg.runtime_cfg
        runtime_cfg.num_sitl_envs = self.num_envs

        runtime = getattr(self._env, "_ardupilot_finetune_runtime", None)
        if runtime is not None and len(runtime.envs) != self.num_envs:
            runtime.stop()
            runtime = None
            setattr(self._env, "_ardupilot_finetune_runtime", None)

        if runtime is None:
            runtime = ArduPilotFineTuneRuntimeState(runtime_cfg)
            setattr(self._env, "_ardupilot_finetune_runtime", runtime)

        self._runtime = runtime
        if not self._runtime.started:
            self._runtime.start()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        delayed_actions = self._action_delay_buffer.compute(actions)

        processed = delayed_actions * self._action_scale + self._action_offset
        processed[:, :3] = torch.clamp(processed[:, :3], min=-self._velocity_limits, max=self._velocity_limits)
        processed[:, 3] = torch.clamp(processed[:, 3], min=-self.cfg.yaw_rate_limit, max=self.cfg.yaw_rate_limit)

        self._processed_actions[:] = processed
        self._velocity_sp[:] = processed[:, :3]
        self._yaw_rate_sp[:] = processed[:, 3]

    def _build_pegasus_state(self, env_id: int) -> State:
        state = State()

        root_pos_w = self._asset.data.root_pos_w[env_id].detach().cpu().numpy()
        root_quat_wxyz = self._asset.data.root_quat_w[env_id].detach().cpu().numpy()
        root_lin_vel_w_t = self._asset.data.root_lin_vel_w[env_id].detach()
        root_lin_vel_b = self._asset.data.root_lin_vel_b[env_id].detach().cpu().numpy()
        root_ang_vel_b = self._asset.data.root_ang_vel_b[env_id].detach().cpu().numpy()

        if bool(self._prev_root_lin_vel_initialized[env_id]):
            lin_acc_w_t = (root_lin_vel_w_t - self._prev_root_lin_vel_w[env_id]) / float(self._env.physics_dt)
        else:
            lin_acc_w_t = torch.zeros_like(root_lin_vel_w_t)
            self._prev_root_lin_vel_initialized[env_id] = True
        self._prev_root_lin_vel_w[env_id] = root_lin_vel_w_t
        root_lin_vel_w = root_lin_vel_w_t.cpu().numpy()
        lin_acc_w = lin_acc_w_t.cpu().numpy()

        state.position = np.asarray(root_pos_w, dtype=np.float64)
        state.attitude = np.asarray(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
            dtype=np.float64,
        )
        state.linear_velocity = np.asarray(root_lin_vel_w, dtype=np.float64)
        state.linear_body_velocity = np.asarray(root_lin_vel_b, dtype=np.float64)
        state.angular_velocity = np.asarray(root_ang_vel_b, dtype=np.float64)
        state.linear_acceleration = np.asarray(lin_acc_w, dtype=np.float64)
        return state

    def _maybe_initialize_propeller_visual(self):
        if not self._propeller_visual_enabled or len(self._rotor_joint_ids) == 4:
            return

        try:
            joint_ids, joint_names = self._asset.find_joints(list(self.cfg.rotor_joint_names), preserve_order=True)
            if len(joint_ids) != 4:
                raise ValueError(f"Expected 4 rotor joints, got {joint_names}")
            self._rotor_joint_ids = joint_ids
            self._rotor_joint_names = joint_names
        except Exception as exc:
            if not self._propeller_visual_warned:
                print(
                    "[WARN][ArduPilotGuidedVelocityAction] Could not enable propeller visual spin "
                    f"for joints {self.cfg.rotor_joint_names}: {exc}"
                )
                self._propeller_visual_warned = True
            self._propeller_visual_enabled = False

    def _handle_propeller_visual(self, rotor_forces: torch.Tensor):
        if not self._propeller_visual_enabled:
            return

        self._maybe_initialize_propeller_visual()
        if not self._propeller_visual_enabled:
            return

        armed_mask = rotor_forces > 0.0
        active_mask = rotor_forces >= float(self.cfg.propeller_visual_idle_force_threshold)
        idle_mask = armed_mask & (~active_mask)

        self._rotor_visual_joint_vel.zero_()
        self._rotor_visual_joint_vel[idle_mask] = float(self.cfg.propeller_visual_idle_speed)
        self._rotor_visual_joint_vel[active_mask] = float(self.cfg.propeller_visual_active_speed)
        self._rotor_visual_joint_vel *= self._rotor_direction

        self._asset.write_joint_velocity_to_sim(self._rotor_visual_joint_vel, joint_ids=self._rotor_joint_ids)

    def _apply_cached_wrench(self):
        rotor_forces, rolling_moment = self._motor_model.omega_to_forces(self._cached_motor_omega)
        rotor_forces = rotor_forces * self._thrust_asymmetry_scale
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

        self._handle_propeller_visual(rotor_forces)

    def _compute_next_command(self):
        self._ensure_runtime()
        dt = float(self._env.physics_dt)
        next_motor_omega = torch.zeros_like(self._cached_motor_omega)

        for env_id, handle in enumerate(self._runtime.envs):
            state = self._build_pegasus_state(env_id)
            imu_data = handle.imu.update(state, dt)
            if imu_data is not None:
                handle.backend.update_sensor("IMU", imu_data)
            handle.backend.update_state(state)

            handle.guided_client.update_kinematics(
                altitude_m=float(state.position[2]),
                linear_velocity_enu=state.linear_velocity,
            )
            handle.guided_client.set_command(
                velocity_sp_enu=self._velocity_sp[env_id].detach().cpu().numpy(),
                yaw_rate_sp=float(self._yaw_rate_sp[env_id].item()),
            )
            handle.guided_client.update(dt)
            handle.backend.update(dt)

            next_motor_omega[env_id] = torch.tensor(
                handle.backend.input_reference(),
                device=self.device,
                dtype=self._cached_motor_omega.dtype,
            )

        self._cached_motor_omega = self._cached_motor_omega + self._motor_lag_alpha * (
            next_motor_omega - self._cached_motor_omega
        )
        self._last_hil_controls = self._hil_mapper.motor_omega_to_hil_controls(self._cached_motor_omega)

    def _hard_reset_sim_state(self, ids: torch.Tensor):
        root_state = self._asset.data.default_root_state[ids].clone()
        root_state[:, 0:3] += self._env.scene.env_origins[ids]
        self._asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=ids)
        self._asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=ids)

        if hasattr(self._asset.data, "default_joint_pos") and hasattr(self._asset.data, "default_joint_vel"):
            default_joint_pos = self._asset.data.default_joint_pos[ids].clone()
            default_joint_vel = self._asset.data.default_joint_vel[ids].clone()
            self._asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=ids)

    def apply_actions(self):
        self._apply_cached_wrench()
        self._compute_next_command()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._velocity_sp.zero_()
            self._yaw_rate_sp.zero_()
            self._cached_motor_omega.zero_()
            self._last_hil_controls.zero_()
            self._prev_root_lin_vel_w.zero_()
            self._prev_root_lin_vel_initialized.zero_()
            self._action_delay_buffer.set_time_lag(0)
            self._action_delay_buffer.reset()
            self._sync_domain_randomization(None)
            if self._propeller_visual_enabled and len(self._rotor_joint_ids) == 4:
                self._rotor_visual_joint_vel.zero_()
                self._asset.write_joint_velocity_to_sim(self._rotor_visual_joint_vel, joint_ids=self._rotor_joint_ids)
            if self._runtime is not None:
                if self._skip_next_runtime_reset:
                    self._skip_next_runtime_reset = False
                    return
                if not self._runtime.all_envs_ready_for_policy():
                    return
                if getattr(self._runtime.cfg.reset, "mode", "soft") in {"hard", "full_takeoff"}:
                    ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                    self._hard_reset_sim_state(ids)
                self._runtime.reset_envs(None)
            return

        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
            id_list = ids.detach().cpu().tolist()
        else:
            id_list = [int(env_id) for env_id in env_ids]
            ids = torch.tensor(id_list, device=self.device, dtype=torch.long)

        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._velocity_sp[ids] = 0.0
        self._yaw_rate_sp[ids] = 0.0
        self._cached_motor_omega[ids] = 0.0
        self._last_hil_controls[ids] = 0.0
        self._prev_root_lin_vel_w[ids] = 0.0
        self._prev_root_lin_vel_initialized[ids] = False
        self._action_delay_buffer.reset(ids)
        self._sync_domain_randomization(ids)
        if self._propeller_visual_enabled and len(self._rotor_joint_ids) == 4:
            self._rotor_visual_joint_vel[ids] = 0.0
            self._asset.write_joint_velocity_to_sim(
                self._rotor_visual_joint_vel[ids],
                joint_ids=self._rotor_joint_ids,
                env_ids=ids,
            )
        if self._runtime is not None:
            if self._skip_next_runtime_reset:
                self._skip_next_runtime_reset = False
                return
            if not self._runtime.all_envs_ready_for_policy():
                return
            if getattr(self._runtime.cfg.reset, "mode", "soft") in {"hard", "full_takeoff"}:
                self._hard_reset_sim_state(ids)
            self._runtime.reset_envs(id_list)

    def _sync_domain_randomization(self, env_ids: torch.Tensor | None):
        state = get_domain_randomization_state(self._env)
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)

        if state is None or not state.cfg.enabled:
            self._action_delay_buffer.set_time_lag(0, ids)
            self._thrust_asymmetry_scale[ids] = 1.0
            self._motor_lag_alpha[ids] = 1.0
            return

        self._action_delay_buffer.set_time_lag(state.action_delay_steps[ids], ids)
        self._thrust_asymmetry_scale[ids] = state.thrust_asymmetry_scale[ids].to(
            device=self.device, dtype=self._raw_actions.dtype
        )
        tau_s = state.motor_lag_tau_s[ids].to(device=self.device, dtype=self._raw_actions.dtype).unsqueeze(-1)
        alpha = torch.ones_like(tau_s)
        active_mask = state.motor_lag_active[ids].unsqueeze(-1)
        dt = float(self._env.physics_dt)
        alpha[active_mask] = dt / torch.clamp(tau_s[active_mask] + dt, min=1.0e-6)
        self._motor_lag_alpha[ids] = torch.clamp(alpha, min=0.0, max=1.0)


ArduPilotGuidedVelocityActionCfg.class_type = ArduPilotGuidedVelocityAction

__all__ = ["ArduPilotGuidedVelocityAction", "ArduPilotGuidedVelocityActionCfg"]
