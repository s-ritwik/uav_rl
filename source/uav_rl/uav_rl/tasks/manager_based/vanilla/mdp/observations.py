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
    """Root position of asset relative to reference, expressed in reference frame."""
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    rel_pos_w = asset.data.root_pos_w - reference_asset.data.root_pos_w
    rel_pos_b = math_utils.quat_apply_inverse(reference_asset.data.root_quat_w, rel_pos_w)
    return apply_additive_state_noise(env, rel_pos_b, env.cfg.domain_randomization.position_noise_std_m)


def root_lin_vel_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Root linear velocity of asset relative to reference, expressed in reference frame."""
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    rel_lin_vel_w = asset.data.root_lin_vel_w - reference_asset.data.root_lin_vel_w
    rel_lin_vel_b = math_utils.quat_apply_inverse(reference_asset.data.root_quat_w, rel_lin_vel_w)
    return apply_additive_state_noise(env, rel_lin_vel_b, env.cfg.domain_randomization.linear_velocity_noise_std_mps)


def root_quat_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Root orientation of asset relative to reference frame."""
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    rel_quat = math_utils.quat_mul(math_utils.quat_inv(reference_asset.data.root_quat_w), asset.data.root_quat_w)
    return apply_quaternion_state_noise(env, rel_quat, env.cfg.domain_randomization.attitude_noise_std_rad)


def root_ang_vel_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Root angular velocity of asset relative to reference, expressed in reference frame."""
    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    rel_ang_vel_w = asset.data.root_ang_vel_w - reference_asset.data.root_ang_vel_w
    rel_ang_vel_b = math_utils.quat_apply_inverse(reference_asset.data.root_quat_w, rel_ang_vel_w)
    return apply_additive_state_noise(env, rel_ang_vel_b, env.cfg.domain_randomization.angular_velocity_noise_std_rps)


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
