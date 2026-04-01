from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def _resolve_body_ids(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> list[int]:
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if body_ids is not None:
        if isinstance(body_ids, torch.Tensor):
            return [int(v) for v in body_ids.detach().cpu().tolist()]
        return [int(v) for v in body_ids]

    body_names = getattr(sensor_cfg, "body_names", None)
    if body_names is None:
        return []
    if isinstance(body_names, str):
        body_names = [body_names]
    asset = env.scene[asset_cfg.name]
    ids, _ = asset.find_bodies(list(body_names), preserve_order=True)
    return [int(v) for v in ids]


def _ensure_state(env):
    num_envs = env.num_envs
    device = env.device

    if not hasattr(env, "_landing_touchdown_flag"):
        env._landing_touchdown_flag = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._landing_touchdown_just_happened = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._landing_touchdown_pre_rel_vz = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._landing_prev_rel_vz = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._landing_touchdown_force_norm = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._landing_state_step = -1


def clear_touchdown_state(env, env_ids: torch.Tensor | None = None) -> None:
    _ensure_state(env)
    if env_ids is None:
        env._landing_touchdown_flag.zero_()
        env._landing_touchdown_just_happened.zero_()
        env._landing_touchdown_pre_rel_vz.zero_()
        env._landing_prev_rel_vz.zero_()
        env._landing_touchdown_force_norm.zero_()
        env._landing_state_step = -1
        return

    ids = env_ids.to(device=env.device, dtype=torch.long)
    env._landing_touchdown_flag[ids] = False
    env._landing_touchdown_just_happened[ids] = False
    env._landing_touchdown_pre_rel_vz[ids] = 0.0
    env._landing_prev_rel_vz[ids] = 0.0
    env._landing_touchdown_force_norm[ids] = 0.0


def _reset_new_episodes(env) -> None:
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        clear_touchdown_state(env, reset_ids)


def update_touchdown_state(
    env,
    threshold: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="body"),
) -> None:
    _ensure_state(env)
    _reset_new_episodes(env)

    step = int(getattr(env, "common_step_counter", -1))
    if env._landing_state_step == step:
        return

    asset = env.scene[asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _resolve_body_ids(env, sensor_cfg, asset_cfg)
    if not body_ids:
        body_ids = [0]

    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
    contact_force_norm = torch.linalg.norm(net_contact_forces, dim=-1)
    contact_force_norm = torch.amax(contact_force_norm, dim=(1, 2))

    rel_vz = asset.data.root_lin_vel_w[:, 2] - reference_asset.data.root_lin_vel_w[:, 2]
    just_happened = (~env._landing_touchdown_flag) & (contact_force_norm > float(threshold))

    env._landing_touchdown_just_happened.zero_()
    env._landing_touchdown_just_happened[just_happened] = True
    env._landing_touchdown_pre_rel_vz[just_happened] = env._landing_prev_rel_vz[just_happened]
    env._landing_touchdown_force_norm[:] = contact_force_norm
    env._landing_touchdown_flag |= just_happened
    env._landing_prev_rel_vz[:] = rel_vz
    env._landing_state_step = step


def touchdown_flag(env) -> torch.Tensor:
    _ensure_state(env)
    return env._landing_touchdown_flag


def touchdown_just_happened(env) -> torch.Tensor:
    _ensure_state(env)
    return env._landing_touchdown_just_happened


def touchdown_pre_rel_vz(env) -> torch.Tensor:
    _ensure_state(env)
    return env._landing_touchdown_pre_rel_vz
