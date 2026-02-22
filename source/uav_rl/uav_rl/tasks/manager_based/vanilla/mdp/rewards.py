from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def position_error_tanh(
    env: "ManagerBasedRLEnv",
    target_pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
    std: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Positive hover reward in [0, 1], larger when closer to the platform-frame target."""
    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    target = _target_tensor(env, target_pos, pos_rel.dtype)
    distance = torch.linalg.norm(pos_rel - target, dim=1)
    return 1.0 - torch.tanh(distance / max(std, 1.0e-3))


def horizontal_position_error_l2(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    target = _target_tensor(env, target_xy, pos_rel.dtype)
    return torch.sum(torch.square(pos_rel[:, :2] - target), dim=1)


def vertical_position_error_l1(
    env: "ManagerBasedRLEnv",
    target_height: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    pos_rel = _relative_position(env, asset_cfg, reference_asset_cfg)
    return torch.abs(pos_rel[:, 2] - target_height)


def speed_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)


def horizontal_speed_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize horizontal (x,y) linear speed for hover-in-place behavior."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def vertical_speed_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize vertical (z) linear speed to discourage bobbing/falling."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def angular_rate_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
