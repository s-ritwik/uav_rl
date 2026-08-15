from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from .landing_state import touchdown_just_happened, update_touchdown_state

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


def root_roll_pitch_above_maximum(
    env: "ManagerBasedRLEnv",
    maximum_angle_deg: float = 35.0,
    maximum_roll_deg: float | None = None,
    maximum_pitch_deg: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if absolute roll or pitch exceeds the configured angle limit."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    maximum_roll_rad = torch.deg2rad(
        torch.tensor(
            float(maximum_angle_deg if maximum_roll_deg is None else maximum_roll_deg),
            device=env.device,
            dtype=roll.dtype,
        )
    )
    maximum_pitch_rad = torch.deg2rad(
        torch.tensor(
            float(maximum_angle_deg if maximum_pitch_deg is None else maximum_pitch_deg),
            device=env.device,
            dtype=pitch.dtype,
        )
    )
    return (torch.abs(roll) > maximum_roll_rad) | (torch.abs(pitch) > maximum_pitch_rad)


def touchdown_terminate(
    env: "ManagerBasedRLEnv",
    threshold: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="body"),
) -> torch.Tensor:
    update_touchdown_state(
        env,
        threshold=threshold,
        asset_cfg=asset_cfg,
        reference_asset_cfg=reference_asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    return touchdown_just_happened(env).clone()
