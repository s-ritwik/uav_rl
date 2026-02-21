from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def root_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def command_velocity(env, action_name: str = "control") -> torch.Tensor:
    return env.action_manager.get_term(action_name).processed_actions[:, :3]


def command_yaw_rate(env, action_name: str = "control") -> torch.Tensor:
    return env.action_manager.get_term(action_name).processed_actions[:, 3:4]


def motor_omega(env, action_name: str = "control") -> torch.Tensor:
    return env.action_manager.get_term(action_name).last_motor_omega
