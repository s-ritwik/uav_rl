from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


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


def illegal_contact_with_debug(
    env: "ManagerBasedRLEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    print_every_steps: int = 1,
) -> torch.Tensor:
    """Terminate on illegal contact and print a debug line when triggered."""
    contact_sensor: "ContactSensor" = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Match IsaacLab illegal_contact logic but keep per-env force magnitudes for debug output.
    contact_force_mag = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    terminated = torch.any(contact_force_mag > threshold, dim=1)

    if torch.any(terminated):
        step = int(getattr(env, "common_step_counter", -1))
        if print_every_steps <= 1 or (step >= 0 and step % int(print_every_steps) == 0):
            hit_env_ids = terminated.nonzero(as_tuple=False).squeeze(-1)
            hit_forces = torch.max(contact_force_mag[hit_env_ids], dim=1)[0]
            num_hits = int(hit_env_ids.numel())
            max_force = float(torch.max(hit_forces).item())
            sample_count = min(8, num_hits)
            sample_ids = hit_env_ids[:sample_count].detach().cpu().tolist()
            sample_forces = hit_forces[:sample_count].detach().cpu().tolist()
            sample_forces = [round(float(v), 3) for v in sample_forces]
            # print(
            #     f"[DEBUG][capsule_contact] step={step} num_envs={num_hits} "
            #     f"max_force={max_force:.3f}N env_ids={sample_ids} forces_N={sample_forces}"
            # )

    return terminated
