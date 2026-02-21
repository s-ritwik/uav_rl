from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_above_maximum(
    env: "ManagerBasedRLEnv",
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height


def root_distance_from_origin(
    env: "ManagerBasedRLEnv",
    max_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_rel = asset.data.root_pos_w - env.scene.env_origins
    dist_xy = torch.linalg.norm(pos_rel[:, :2], dim=1)
    return dist_xy > max_distance
