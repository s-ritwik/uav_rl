from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp as env_mdp

from ...vanilla.mdp import terminations as vanilla_terminations


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
    if action_term is None:
        return None

    if not hasattr(action_term, "ready_mask"):
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


def illegal_contact_after_takeoff(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    print_every_steps: int = 1,
) -> torch.Tensor:
    return _guard_until_ready(
        env,
        vanilla_terminations.illegal_contact_with_debug(
            env,
            threshold=threshold,
            sensor_cfg=sensor_cfg,
            print_every_steps=print_every_steps,
        ),
    )


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
        vanilla_terminations.root_height_above_maximum(env, maximum_height=maximum_height, asset_cfg=asset_cfg),
    )


def root_distance_from_origin_after_takeoff(
    env,
    max_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _guard_until_ready(
        env,
        vanilla_terminations.root_distance_from_origin(env, max_distance=max_distance, asset_cfg=asset_cfg),
    )


__all__ = [
    "illegal_contact_after_takeoff",
    "root_distance_from_origin_after_takeoff",
    "root_height_above_maximum_after_takeoff",
    "root_height_below_minimum_after_takeoff",
    "time_out_after_takeoff",
]
