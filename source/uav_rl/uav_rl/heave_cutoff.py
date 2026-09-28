"""Physics-rate shutdown and pre-impact capture for the heave task family."""
import torch
from isaaclab.utils import math as math_utils

from uav_rl.platform_reference import platform_reference_data
from uav_rl.transfer.thrust_cutoff import BatchedRotorCutoff


class HeaveCutoffRuntime:
    def __init__(self, action):
        self.action = action
        self.env = action._env
        self.cutoff = BatchedRotorCutoff(action.num_envs, action.device, action.cfg.cutoff)
        self.touched = torch.zeros_like(self.cutoff.active)
        self.previous_valid = torch.zeros_like(self.touched)
        # relative vz, marker vz, XY error, roll, pitch, yaw
        self.previous = torch.zeros((action.num_envs, 6), device=action.device)
        self.impact = torch.zeros_like(self.previous)
        self.impact_residual_fraction = torch.zeros(action.num_envs, device=action.device)
        self.env._heave_cutoff_runtime = self

    def reset(self, ids=None):
        self.cutoff.reset(ids)
        ids = slice(None) if ids is None else ids
        self.touched[ids] = False
        self.previous_valid[ids] = False
        self.previous[ids] = 0
        self.impact[ids] = 0
        self.impact_residual_fraction[ids] = 0

    def check_gate(self):
        cfg = self.cutoff.cfg
        if not cfg.enabled:
            return
        data = self.action._asset.data
        relative = data.root_pos_w - platform_reference_data(self.env).root_pos_w
        relative = relative.clone()
        relative[:, 2] -= cfg.vehicle_z0_m
        # First three actor channels are relative position, including the gear offset.
        observation = getattr(self.env, "obs_buf", {}).get("policy")
        if isinstance(observation, torch.Tensor):
            relative = observation[:, :3]
        eligible = (relative[:, :2].abs() <= cfg.xy_tolerance_m).all(dim=-1)
        eligible &= (relative[:, 2] >= 0) & (relative[:, 2] <= cfg.clearance_m)
        eligible &= torch.isfinite(relative).all(dim=-1) & ~self.touched
        self.cutoff.trigger(eligible, self.action._cached_motor_omega)

    def observe_contact(self):
        """Called before each advance AND at reward time to cover the last substep."""
        env = self.env
        data = self.action._asset.data
        reference = platform_reference_data(env)
        sensor = env.scene.sensors["contact_forces"]
        force = torch.linalg.vector_norm(sensor.data.net_forces_w[:, self.action._body_id], dim=-1)
        threshold = float(env.cfg.post_init_cfg.touchdown.force_threshold_n)
        first = (force > threshold) & ~self.touched & self.previous_valid
        self.impact[first] = self.previous[first]
        self.touched |= first
        initial = self.cutoff.initial_omega.square().sum(dim=-1).clamp_min(1e-12)
        fraction = self.action._cached_motor_omega.square().sum(dim=-1) / initial
        self.impact_residual_fraction[first] = fraction[first]
        self.previous[:, 0] = data.root_lin_vel_w[:, 2] - reference.root_lin_vel_w[:, 2]
        self.previous[:, 1] = reference.root_lin_vel_w[:, 2]
        self.previous[:, 2] = torch.linalg.vector_norm(data.root_pos_w[:, :2] - reference.root_pos_w[:, :2], dim=-1)
        self.previous[:, 3:] = torch.stack(math_utils.euler_xyz_from_quat(data.root_quat_w), dim=-1)
        self.previous_valid[:] = True


def flight_mask(env):
    runtime = getattr(env, "_heave_cutoff_runtime", None)
    if runtime is None:
        return 1.0
    return (~runtime.cutoff.active).float()
