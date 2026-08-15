from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .randomization import apply_additive_state_noise, apply_quaternion_state_noise


def root_pos_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """World-frame CG-to-CG relative position with z clearance offset applied."""
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    rel_pos_w = asset.data.root_pos_w - reference_asset.data.root_pos_w
    z0_m = float(getattr(getattr(env.cfg, "post_init_cfg", None), "vehicle_z0_m", 0.053))
    rel_pos_w = rel_pos_w.clone()
    rel_pos_w[:, 2] = rel_pos_w[:, 2] - z0_m
    return apply_additive_state_noise(env, rel_pos_w, env.cfg.domain_randomization.position_noise_std_m)


def root_lin_vel_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """World-frame root linear velocity of asset relative to reference."""
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    rel_lin_vel_w = asset.data.root_lin_vel_w - reference_asset.data.root_lin_vel_w
    return apply_additive_state_noise(env, rel_lin_vel_w, env.cfg.domain_randomization.linear_velocity_noise_std_mps)


def root_quat_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """UAV root quaternion in world frame."""
    asset = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w
    return apply_quaternion_state_noise(env, quat_w, env.cfg.domain_randomization.attitude_noise_std_rad)


def root_ang_vel_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """World-frame root angular velocity of asset relative to reference."""
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    rel_ang_vel_w = asset.data.root_ang_vel_w - reference_asset.data.root_ang_vel_w
    return apply_additive_state_noise(env, rel_ang_vel_w, env.cfg.domain_randomization.angular_velocity_noise_std_rps)


def projected_gravity_noisy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Projected gravity with optional estimator noise."""
    asset = env.scene[asset_cfg.name]
    return apply_additive_state_noise(
        env,
        asset.data.projected_gravity_b,
        env.cfg.domain_randomization.projected_gravity_noise_std,
    )


def motor_omega(env, action_name: str = "control") -> torch.Tensor:
    return env.action_manager.get_term(action_name).last_motor_omega


def root_lin_vel_w(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exact world-frame linear velocity for privileged critic observations."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def future_platform_heave(
    env,
    horizon_s: float = 6.0,
    sample_rate_hz: float = 20.0,
) -> torch.Tensor:
    """Future platform heave trajectory for privileged critic observations.

    Returns the next ``horizon_s`` seconds of platform heave motion as a flat sequence sampled at
    ``sample_rate_hz``. The sequence contains the heave motion only, without the constant height bias.
    """
    heave_motion = getattr(env, "_csv_heave_platform_motion", None)
    if heave_motion is None:
        raise RuntimeError("CSVHeavePlatformMotion is not attached to the environment.")
    return heave_motion.future_heave(horizon_s=horizon_s, sample_rate_hz=sample_rate_hz)


def future_platform_pos_z_w(
    env,
    horizon_s: float = 6.0,
    sample_rate_hz: float = 20.0,
) -> torch.Tensor:
    """Future platform world-frame z samples for privileged critic observations."""
    heave_motion = getattr(env, "_csv_heave_platform_motion", None)
    if heave_motion is None:
        raise RuntimeError("CSVHeavePlatformMotion is not attached to the environment.")
    return heave_motion.future_world_z(horizon_s=horizon_s, sample_rate_hz=sample_rate_hz)
