from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ...ardupilot_finetune.mdp.actions import ArduPilotGuidedVelocityAction
from ...vanilla.mdp.actions import PX4LikeVelocityActionCfg
from .runtime import PX4FineTuneRuntimeState, PX4State


@configclass
class PX4OffboardVelocityActionCfg(PX4LikeVelocityActionCfg):
    """Action term that keeps the vanilla policy contract but routes commands through PX4 SITL."""

    class_type: type[ActionTerm] = MISSING


class PX4OffboardVelocityAction(ArduPilotGuidedVelocityAction):
    cfg: PX4OffboardVelocityActionCfg

    def __init__(self, cfg: PX4OffboardVelocityActionCfg, env):
        super().__init__(cfg, env)
        # Match the standalone PX4 host lifecycle more closely by launching
        # the SITL/runtime before the simulation loop begins.
        self._ensure_runtime()

    def _ensure_runtime(self):
        runtime_cfg = self._env.cfg.runtime_cfg
        runtime_cfg.num_sitl_envs = self.num_envs

        runtime = getattr(self._env, "_px4_finetune_runtime", None)
        if runtime is not None and len(runtime.envs) != self.num_envs:
            runtime.stop()
            runtime = None
            setattr(self._env, "_px4_finetune_runtime", None)

        if runtime is None:
            runtime = PX4FineTuneRuntimeState(runtime_cfg)
            setattr(self._env, "_px4_finetune_runtime", runtime)

        self._runtime = runtime
        if not self._runtime.started:
            self._runtime.start()

    def _build_backend_state(self, env_id: int) -> PX4State:
        state = PX4State()

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

        state.position = root_pos_w.astype("float64", copy=False)
        state.attitude = np.asarray(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
            dtype=np.float64,
        )
        state.linear_velocity = root_lin_vel_w.astype("float64", copy=False)
        state.linear_body_velocity = root_lin_vel_b.astype("float64", copy=False)
        state.angular_velocity = root_ang_vel_b.astype("float64", copy=False)
        state.linear_acceleration = lin_acc_w.astype("float64", copy=False)
        return state

    def _compute_next_command(self):
        self._ensure_runtime()
        dt = float(self._env.physics_dt)
        next_motor_omega = torch.zeros_like(self._cached_motor_omega)

        for env_id, handle in enumerate(self._runtime.envs):
            state = self._build_backend_state(env_id)

            # Mirror the standalone PX4 host ordering:
            # backend/bridge update on the previous tick's state first,
            # then publish fresh sensor/state data for the next tick.
            handle.guided_client.set_command(
                velocity_sp_enu=self._velocity_sp[env_id].detach().cpu().numpy(),
                yaw_rate_sp=float(self._yaw_rate_sp[env_id].item()),
            )
            handle.backend.update(dt)
            handle.guided_client.update(dt)

            imu_data = handle.imu.update(state, dt)
            if imu_data is not None:
                handle.backend.update_sensor("IMU", imu_data)

            gps_data = handle.gps.update(state, dt)
            if gps_data is not None:
                handle.backend.update_sensor("GPS", gps_data)

            barometer_data = handle.barometer.update(state, dt)
            if barometer_data is not None:
                handle.backend.update_sensor("Barometer", barometer_data)

            magnetometer_data = handle.magnetometer.update(state, dt)
            if magnetometer_data is not None:
                handle.backend.update_sensor("Magnetometer", magnetometer_data)

            handle.backend.update_state(state)
            handle.guided_client.update_kinematics(
                altitude_m=float(state.position[2]),
                linear_velocity_enu=state.linear_velocity,
            )

            next_motor_omega[env_id] = torch.tensor(
                handle.backend.input_reference(),
                device=self.device,
                dtype=self._cached_motor_omega.dtype,
            )

        self._cached_motor_omega = self._cached_motor_omega + self._motor_lag_alpha * (
            next_motor_omega - self._cached_motor_omega
        )
        self._last_hil_controls = self._hil_mapper.motor_omega_to_hil_controls(self._cached_motor_omega)


PX4OffboardVelocityActionCfg.class_type = PX4OffboardVelocityAction

__all__ = ["PX4OffboardVelocityAction", "PX4OffboardVelocityActionCfg"]
