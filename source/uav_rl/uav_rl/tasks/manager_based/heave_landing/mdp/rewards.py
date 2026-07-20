from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from .landing_state import (
    touchdown_flag,
    touchdown_just_happened,
    touchdown_platform_vz,
    touchdown_pre_rel_vz,
    touchdown_roll_pitch_yaw,
    touchdown_xy_error,
    update_touchdown_state,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def failure_termination_penalty(
    env: "ManagerBasedRLEnv",
    penalty: float = -10.0,
    failure_term_names: tuple[str, ...] = ("time_out", "attitude_tilt", "crash_low", "crash_high", "out_of_bounds"),
) -> torch.Tensor:
    """Apply a fixed penalty only for selected failure terminations.

    This lets touchdown remain a success termination while keeping hard penalties
    for crash/out-of-bounds style failures.
    """

    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not failure_term_names:
        return out

    for name in failure_term_names:
        try:
            term = env.termination_manager.get_term(name).to(dtype=torch.bool)
        except Exception:
            continue
        out[term] = float(penalty)
    return out


def touchdown_termination_reward(
    env: "ManagerBasedRLEnv",
    touchdown_term_name: str = "touchdown",
) -> torch.Tensor:
    """Reward 1.0 on steps where episode terminates due to touchdown, else 0.0."""

    try:
        touchdown = env.termination_manager.get_term(touchdown_term_name).to(dtype=torch.bool)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    return touchdown.float()


def _target_tensor(env: "ManagerBasedRLEnv", values: tuple[float, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=env.device, dtype=dtype).unsqueeze(0)


def _relative_position(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    reference_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    reference_asset: RigidObject = env.scene[reference_asset_cfg.name]
    return asset.data.root_pos_w - reference_asset.data.root_pos_w


def position_error_l2(
    env: "ManagerBasedRLEnv",
    target_pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    target = _target_tensor(env, target_pos, pos_rel.dtype)
    return torch.sum(torch.square(pos_rel - target), dim=1)


def horizontal_position_error_tanh(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    std: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Positive XY tracking reward in [0, 1], larger when closer to platform XY target."""
    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    target = _target_tensor(env, target_xy, pos_rel.dtype)
    distance_xy = torch.linalg.norm(pos_rel[:, :2] - target, dim=1)
    return 1.0 - torch.tanh(distance_xy / max(std, 1.0e-3))


def vertical_position_error_l1(
    env: "ManagerBasedRLEnv",
    target_height: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    return torch.abs(pos_rel[:, 2] - target_height)


def vertical_clearance_excess_l1(
    env: "ManagerBasedRLEnv",
    clearance_threshold_m: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Linear penalty for being too high above platform clearance.

    This uses the same clearance definition as observation `root_pos_rel.z`:
    z_clearance = rel_z_in_world - z0.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    reference_asset: RigidObject = env.scene[reference_asset_cfg.name]
    rel_pos_w = asset.data.root_pos_w - reference_asset.data.root_pos_w
    z0_m = float(getattr(getattr(env.cfg, "post_init_cfg", None), "vehicle_z0_m", 0.053))
    z_clearance = rel_pos_w[:, 2] - z0_m
    return torch.clamp(z_clearance - float(clearance_threshold_m), min=0.0)


def speed_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)


def horizontal_speed_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize horizontal (x,y) linear speed for hover-in-place behavior."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def horizontal_velocity_error_tanh(
    env: "ManagerBasedRLEnv",
    target_rel_xy: tuple[float, float] = (0.0, 0.0),
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Positive XY relative-velocity tracking reward in [0, 1], larger when closer to target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reference_asset: RigidObject = env.scene[reference_asset_cfg.name]
    rel_vel_xy = asset.data.root_lin_vel_w[:, :2] - reference_asset.data.root_lin_vel_w[:, :2]
    target = _target_tensor(env, target_rel_xy, rel_vel_xy.dtype)
    distance_xy = torch.linalg.norm(rel_vel_xy - target, dim=1)
    return 1.0 - torch.tanh(distance_xy / max(std, 1.0e-3))


def vertical_speed_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize vertical (z) linear speed to discourage bobbing/falling."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def uav_linear_acceleration_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["body"]),
) -> torch.Tensor:
    """Penalize UAV body linear acceleration using the COM acceleration reported by PhysX.

    The default configuration targets the main UAV body only, which keeps the signal focused on
    the vehicle acceleration instead of summing over every articulation link.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


def raw_action_rate_component_l2(env: "ManagerBasedRLEnv", action_index: int) -> torch.Tensor:
    """Penalize one raw policy action-rate component by squared step-to-step change."""
    action_index = int(action_index)
    return torch.square(env.action_manager.action[:, action_index] - env.action_manager.prev_action[:, action_index])


def raw_action_component_l2(env: "ManagerBasedRLEnv", action_index: int) -> torch.Tensor:
    """Penalize one raw policy action component by squared magnitude."""
    return torch.square(env.action_manager.action[:, int(action_index)])


def near_target_action_xy_l2(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    target_xy: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Penalize raw XY action magnitude more strongly when the robot is already near the platform center."""

    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    target = _target_tensor(env, target_xy, pos_rel.dtype)
    distance_xy = torch.linalg.norm(pos_rel[:, :2] - target, dim=1)
    proximity_gate = 1.0 - torch.tanh(distance_xy / max(std, 1.0e-3))
    return proximity_gate * torch.sum(torch.square(env.action_manager.action[:, :2]), dim=1)


def touchdown_quality_reward(
    env: "ManagerBasedRLEnv",
    max_touchdown_speed_mps: float = 0.25,
    max_xy_error_m: float = 0.20,
    require_xy_within_box: bool = False,
    require_attitude_within_limits: bool = True,
    max_touchdown_roll_deg: float = 10.0,
    max_touchdown_pitch_deg: float = 10.0,
    max_touchdown_yaw_deg: float = 10.0,
    target_touchdown_yaw_deg: float = 0.0,
    touchdown_force_threshold: float = 2.0,
    good_touchdown_reward: float = 4.0,
    bad_touchdown_reward: float = -5.0,
    center_proximity_bonus: float = 0.0,
    low_touchdown_speed_bonus: float = 0.0,
    low_platform_vertical_speed_bonus: float = 0.0,
    platform_vertical_speed_bonus_scale_mps: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="body"),
) -> torch.Tensor:
    """Sparse touchdown reward based on vertical speed and XY alignment at first contact.

    Positive reward if, at touchdown onset (contact force crosses threshold):
    - descent speed is <= max_touchdown_speed_mps, and
    - drone root is within max_xy_error_m of platform center in XY.
    - if enabled, roll, pitch, and wrapped yaw error are within configured limits.
    Good touchdowns can also receive an extra shaped bonus that increases as the
    XY touchdown point approaches the platform center.
    Otherwise a negative reward is issued.
    """

    update_touchdown_state(
        env,
        threshold=touchdown_force_threshold,
        asset_cfg=asset_cfg,
        reference_asset_cfg=reference_asset_cfg,
        sensor_cfg=sensor_cfg,
    )

    just_touched = touchdown_just_happened(env)
    pre_rel_vz = touchdown_pre_rel_vz(env)
    platform_vz_at_touchdown = touchdown_platform_vz(env)

    # Convert relative vertical velocity to a positive descent speed (m/s).
    # In world ENU, downward is negative z velocity.
    descent_speed = (-pre_rel_vz).clamp_min(0.0)
    platform_vertical_speed = torch.abs(platform_vz_at_touchdown)

    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    xy_error = torch.linalg.norm(pos_rel[:, :2], dim=1)

    asset: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    target_yaw_rad = math.radians(float(target_touchdown_yaw_deg))
    yaw_error = torch.atan2(torch.sin(yaw - target_yaw_rad), torch.cos(yaw - target_yaw_rad))

    reward = torch.zeros_like(pre_rel_vz)
    speed_ok = descent_speed <= float(max_touchdown_speed_mps)
    roll_ok = torch.abs(roll) <= math.radians(float(max_touchdown_roll_deg))
    pitch_ok = torch.abs(pitch) <= math.radians(float(max_touchdown_pitch_deg))
    yaw_ok = torch.abs(yaw_error) <= math.radians(float(max_touchdown_yaw_deg))
    if require_attitude_within_limits:
        attitude_ok = roll_ok & pitch_ok & yaw_ok
    else:
        attitude_ok = torch.ones_like(speed_ok, dtype=torch.bool)
    if require_xy_within_box:
        good = speed_ok & attitude_ok & (xy_error <= float(max_xy_error_m))
    else:
        good = speed_ok & attitude_ok

    good_reward = torch.full_like(pre_rel_vz, float(good_touchdown_reward))
    if float(center_proximity_bonus) != 0.0:
        xy_radius = max(float(max_xy_error_m), 1.0e-6)
        closeness = (1.0 - xy_error / xy_radius).clamp(0.0, 1.0)
        # Add an extra touchdown bonus that peaks at the exact platform center.
        good_reward = good_reward + float(center_proximity_bonus) * closeness
    if float(low_touchdown_speed_bonus) != 0.0:
        speed_radius = max(float(max_touchdown_speed_mps), 1.0e-6)
        speed_closeness = (1.0 - descent_speed / speed_radius).clamp(0.0, 1.0)
        # Add an extra touchdown bonus that peaks at zero descent speed.
        good_reward = good_reward + float(low_touchdown_speed_bonus) * speed_closeness
    if float(low_platform_vertical_speed_bonus) != 0.0:
        platform_speed_radius = max(float(platform_vertical_speed_bonus_scale_mps), 1.0e-6)
        platform_speed_closeness = (1.0 - platform_vertical_speed / platform_speed_radius).clamp(0.0, 1.0)
        # Add an extra touchdown bonus that peaks when the platform vertical speed is zero.
        good_reward = good_reward + float(low_platform_vertical_speed_bonus) * platform_speed_closeness

    reward[just_touched] = torch.where(
        good[just_touched],
        good_reward[just_touched],
        torch.full_like(pre_rel_vz[just_touched], float(bad_touchdown_reward)),
    )
    return reward


def touchdown_quality_metrics(
    env: "ManagerBasedRLEnv",
    env_ids,
    max_touchdown_speed_mps: float = 0.25,
    max_xy_error_m: float = 0.20,
    require_xy_within_box: bool = False,
    require_attitude_within_limits: bool = True,
    max_touchdown_roll_deg: float = 10.0,
    max_touchdown_pitch_deg: float = 10.0,
    max_touchdown_yaw_deg: float = 10.0,
    target_touchdown_yaw_deg: float = 0.0,
) -> dict[str, float]:
    """Return reset-time touchdown quality percentages for the environments being reset."""

    if env_ids is None:
        env_ids_tensor = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids_tensor = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        env_ids_tensor = env_ids.to(device=env.device, dtype=torch.long)
    else:
        env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if env_ids_tensor.numel() == 0:
        return {
            "touchdown_rate": 0.0,
            "good_touchdown_rate": 0.0,
            "bad_touchdown_rate": 0.0,
            "good_touchdown_pct": 0.0,
            "bad_touchdown_pct": 0.0,
        }

    touched = touchdown_flag(env)[env_ids_tensor]
    pre_rel_vz = touchdown_pre_rel_vz(env)[env_ids_tensor]
    xy_error = touchdown_xy_error(env)[env_ids_tensor]
    roll, pitch, yaw = touchdown_roll_pitch_yaw(env)
    roll = roll[env_ids_tensor]
    pitch = pitch[env_ids_tensor]
    yaw = yaw[env_ids_tensor]

    descent_speed = (-pre_rel_vz).clamp_min(0.0)
    target_yaw_rad = math.radians(float(target_touchdown_yaw_deg))
    yaw_error = torch.atan2(torch.sin(yaw - target_yaw_rad), torch.cos(yaw - target_yaw_rad))

    speed_ok = descent_speed <= float(max_touchdown_speed_mps)
    roll_ok = torch.abs(roll) <= math.radians(float(max_touchdown_roll_deg))
    pitch_ok = torch.abs(pitch) <= math.radians(float(max_touchdown_pitch_deg))
    yaw_ok = torch.abs(yaw_error) <= math.radians(float(max_touchdown_yaw_deg))
    if require_attitude_within_limits:
        attitude_ok = roll_ok & pitch_ok & yaw_ok
    else:
        attitude_ok = torch.ones_like(speed_ok, dtype=torch.bool)
    if require_xy_within_box:
        good = touched & speed_ok & attitude_ok & (xy_error <= float(max_xy_error_m))
    else:
        good = touched & speed_ok & attitude_ok
    bad = touched & ~good

    touchdown_count = touched.sum()
    touchdown_den = float(touchdown_count.item())
    reset_den = float(env_ids_tensor.numel())

    good_touchdown_pct = float(good.sum().item() / touchdown_den) if touchdown_den > 0.0 else 0.0
    bad_touchdown_pct = float(bad.sum().item() / touchdown_den) if touchdown_den > 0.0 else 0.0

    return {
        "touchdown_rate": float(touchdown_count.item() / reset_den),
        "good_touchdown_rate": float(good.sum().item() / reset_den),
        "bad_touchdown_rate": float(bad.sum().item() / reset_den),
        "good_touchdown_pct": good_touchdown_pct,
        "bad_touchdown_pct": bad_touchdown_pct,
    }


def angular_rate_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


def angular_velocity_rate_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize step-to-step change in measured body-frame angular velocity wx, wy, wz."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_ang_vel_b = asset.data.root_ang_vel_b

    state_name = "_heave_landing_prev_root_ang_vel_b"
    prev_ang_vel_b = getattr(env, state_name, None)
    if prev_ang_vel_b is None or prev_ang_vel_b.shape != current_ang_vel_b.shape:
        setattr(env, state_name, current_ang_vel_b.clone())
        return torch.zeros(env.num_envs, device=env.device, dtype=current_ang_vel_b.dtype)

    delta = current_ang_vel_b - prev_ang_vel_b

    # Do not penalize the first step after reset where the previous-state buffer is stale by construction.
    fresh_envs = env.episode_length_buf <= 1
    delta = delta.clone()
    delta[fresh_envs] = 0.0

    prev_ang_vel_b.copy_(current_ang_vel_b)
    return torch.sum(torch.square(delta), dim=1)


def angular_rate_xy_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize body-frame roll/pitch rates only."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def yaw_rate_error_l2(
    env: "ManagerBasedRLEnv",
    target_yaw_rate: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared yaw-rate error using body-frame z angular velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_rate = asset.data.root_ang_vel_b[:, 2]
    return torch.square(yaw_rate - float(target_yaw_rate))


def yaw_error_l2(
    env: "ManagerBasedRLEnv",
    target_yaw: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared wrapped yaw error in world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.root_quat_w
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_error = torch.atan2(torch.sin(yaw - target_yaw), torch.cos(yaw - target_yaw))
    return torch.square(yaw_error)
