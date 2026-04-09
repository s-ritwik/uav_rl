from __future__ import annotations

from collections.abc import Sequence

import torch

from ...landing_sway.mdp.events import *  # noqa: F401, F403
from ...landing_sway.mdp.events import MultiSinePlatformMotion as LandingSwayMultiSinePlatformMotion


def _ready_mask(env) -> torch.Tensor | None:
    try:
        action_term = env.action_manager.get_term("control")
    except Exception:
        return None

    if not hasattr(action_term, "ready_mask"):
        return None

    return action_term.ready_mask()


class MultiSinePlatformMotionAfterReady(LandingSwayMultiSinePlatformMotion):
    """Freeze landing_sway platform motion until PX4 SITL reaches the pre-policy hover state."""

    def __init__(self, cfg, env):
        self._activated = None
        self._activation_time_s = None
        super().__init__(cfg, env)
        self._activated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._activation_time_s = torch.zeros(self.num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_tensor = self._resolve_env_ids(env_ids)
        super().reset(env_ids)
        if self._activated is None or self._activation_time_s is None:
            return
        if env_ids_tensor.numel() == 0:
            return
        self._activated[env_ids_tensor] = False
        self._activation_time_s[env_ids_tensor] = 0.0
        self._write_motion_to_sim(env_ids_tensor, 0.0)

    def __call__(self, env, env_ids, asset_cfg=None, stage_cfg=None) -> None:
        del asset_cfg, stage_cfg
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        ready_mask = _ready_mask(env)
        if ready_mask is None:
            super().__call__(env, env_ids_tensor)
            return

        ready_ids = env_ids_tensor[ready_mask[env_ids_tensor]]
        if ready_ids.numel() == 0:
            return

        current_time_s = self._current_time_s()
        newly_ready = ready_ids[~self._activated[ready_ids]]
        if newly_ready.numel() > 0:
            self._activated[newly_ready] = True
            self._activation_time_s[newly_ready] = current_time_s

        local_time_s = current_time_s - self._activation_time_s[ready_ids]
        for env_id, motion_time_s in zip(ready_ids.tolist(), local_time_s.tolist()):
            single_env_id = torch.tensor([env_id], device=self.device, dtype=torch.long)
            self._write_motion_to_sim(single_env_id, float(motion_time_s))


__all__ = [name for name in globals() if not name.startswith("_")]
