from __future__ import annotations

import torch

from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import SceneEntityCfg

from ...landing_sway.mdp import terminations as landing_sway_terminations


def _action_term(env):
    try:
        return env.action_manager.get_term("control")
    except Exception:
        return None


def _warmup_active(env) -> bool:
    action_term = _action_term(env)
    if action_term is None or not hasattr(action_term, "warmup_active"):
        return False
    return bool(action_term.warmup_active())


def _ready_mask(env) -> torch.Tensor | None:
    action_term = _action_term(env)
    if action_term is None or not hasattr(action_term, "ready_mask"):
        return None
    return action_term.ready_mask()


def _guard_until_ready(env, terminated: torch.Tensor) -> torch.Tensor:
    if _warmup_active(env):
        return torch.zeros_like(terminated, dtype=torch.bool)
    ready_mask = _ready_mask(env)
    if ready_mask is None:
        return terminated
    guarded = terminated.clone()
    guarded[~ready_mask] = False
    return guarded


def time_out_after_takeoff(env) -> torch.Tensor:
    if _warmup_active(env):
        env.episode_length_buf[:] = 0
        return torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    ready_mask = _ready_mask(env)
    if ready_mask is not None:
        env.episode_length_buf[~ready_mask] = 0
    return _guard_until_ready(env, env_mdp.time_out(env))


def root_height_below_minimum_after_takeoff(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _guard_until_ready(
        env,
        env_mdp.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg),
    )


def root_height_above_maximum_after_takeoff(
    env,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _guard_until_ready(
        env,
        landing_sway_terminations.root_height_above_maximum(
            env,
            maximum_height=maximum_height,
            asset_cfg=asset_cfg,
        ),
    )


def root_distance_from_origin_after_takeoff(
    env,
    max_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _guard_until_ready(
        env,
        landing_sway_terminations.root_distance_from_origin(env, max_distance=max_distance, asset_cfg=asset_cfg),
    )


def root_roll_pitch_above_maximum_after_takeoff(
    env,
    maximum_angle_deg: float = 35.0,
    maximum_roll_deg: float | None = None,
    maximum_pitch_deg: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _guard_until_ready(
        env,
        landing_sway_terminations.root_roll_pitch_above_maximum(
            env,
            maximum_angle_deg=maximum_angle_deg,
            maximum_roll_deg=maximum_roll_deg,
            maximum_pitch_deg=maximum_pitch_deg,
            asset_cfg=asset_cfg,
        ),
    )


def touchdown_terminate_after_takeoff(
    env,
    threshold: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="body"),
) -> torch.Tensor:
    return _guard_until_ready(
        env,
        landing_sway_terminations.touchdown_terminate(
            env,
            threshold=threshold,
            asset_cfg=asset_cfg,
            reference_asset_cfg=reference_asset_cfg,
            sensor_cfg=sensor_cfg,
        ),
    )


__all__ = [
    "root_distance_from_origin_after_takeoff",
    "root_height_above_maximum_after_takeoff",
    "root_height_below_minimum_after_takeoff",
    "root_roll_pitch_above_maximum_after_takeoff",
    "time_out_after_takeoff",
    "touchdown_terminate_after_takeoff",
]
