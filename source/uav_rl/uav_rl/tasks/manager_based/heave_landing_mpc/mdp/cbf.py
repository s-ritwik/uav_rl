from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_heave_cbf_h0_components(
    env: "ManagerBasedRLEnv",
    d_min_m: float = 0.156,
    landing_velocity_mps: float = -0.2,
    a_rel_mps2: float = 0.7,
    eps: float = 1.0e-4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the deployment-style heave CBF h0 without acceleration terms.

    h0 = (clearance - d_min) - dstop, where d_min is already the minimum
    allowed root-to-platform clearance. No acceleration or hdot term is used.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    reference_asset: RigidObject = env.scene[reference_asset_cfg.name]

    d0 = asset.data.root_pos_w[:, 2] - reference_asset.data.root_pos_w[:, 2]
    vr0 = asset.data.root_lin_vel_w[:, 2] - reference_asset.data.root_lin_vel_w[:, 2]
    vr_neg = 0.5 * (vr0 - torch.sqrt(torch.square(vr0) + float(eps)))

    a_rel = max(float(a_rel_mps2), 1.0e-6)
    v_land = float(landing_velocity_mps)
    dstop = (torch.square(vr_neg) - v_land * v_land) / (2.0 * a_rel)
    h0 = (d0 - float(d_min_m)) - dstop
    return h0, dstop, vr_neg, d0, vr0


def heave_cbf_h0(
    env: "ManagerBasedRLEnv",
    d_min_m: float = 0.156,
    landing_velocity_mps: float = -0.2,
    a_rel_mps2: float = 0.7,
    eps: float = 1.0e-4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Return h0 as a single-column observation/reward signal."""

    h0, _, _, _, _ = compute_heave_cbf_h0_components(
        env,
        d_min_m=d_min_m,
        landing_velocity_mps=landing_velocity_mps,
        a_rel_mps2=a_rel_mps2,
        eps=eps,
        asset_cfg=asset_cfg,
        reference_asset_cfg=reference_asset_cfg,
    )
    return h0.unsqueeze(-1)


def heave_cbf_features(
    env: "ManagerBasedRLEnv",
    d_min_m: float = 0.156,
    landing_velocity_mps: float = -0.2,
    a_rel_mps2: float = 0.7,
    eps: float = 1.0e-4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Critic-friendly CBF features: [h0, dstop, downward_relative_velocity]."""

    h0, dstop, vr_neg, _, _ = compute_heave_cbf_h0_components(
        env,
        d_min_m=d_min_m,
        landing_velocity_mps=landing_velocity_mps,
        a_rel_mps2=a_rel_mps2,
        eps=eps,
        asset_cfg=asset_cfg,
        reference_asset_cfg=reference_asset_cfg,
    )
    return torch.stack((h0, dstop, vr_neg), dim=-1)
