from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch


@dataclass
class VanillaPolicySpec:
    """Network contract for the current vanilla task."""

    obs_dim: int = 20
    action_dim: int = 4
    actor_hidden_dims: tuple[int, ...] = (128, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (128, 128, 64)
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    activation: str = "elu"
    init_noise_std: float = 1.0


def resolve_checkpoint_path(
    *,
    load_run: str | None,
    checkpoint_name: str | None,
    checkpoint_path: str | None,
    log_root: str | Path,
) -> Path:
    """Resolve an RSL-RL checkpoint from a direct path or a run folder."""

    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return path

    if not load_run:
        raise ValueError("Provide either `checkpoint_path` or `load_run` for policy mode.")

    run_dir = Path(log_root).expanduser().resolve() / load_run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if checkpoint_name:
        path = run_dir / checkpoint_name
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found in run directory: {path}")
        return path

    model_paths = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(re.search(r"model_(\d+)\.pt$", path.name).group(1)),
    )
    if not model_paths:
        raise FileNotFoundError(f"No `model_*.pt` files found in: {run_dir}")
    return model_paths[-1]


class RslRlPolicy:
    """Load an exported JIT policy or rebuild the vanilla actor from a checkpoint."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        policy_jit: str | None = None,
        checkpoint_path: str | None = None,
        spec: VanillaPolicySpec | None = None,
    ):
        self.device = torch.device(device)
        self.spec = spec if spec is not None else VanillaPolicySpec()

        if policy_jit is not None:
            self._mode = "jit"
            self._module = torch.jit.load(str(Path(policy_jit).expanduser().resolve()), map_location=self.device)
            self._module.eval()
        elif checkpoint_path is not None:
            self._mode = "checkpoint"
            self._module = self._load_actor_critic_from_checkpoint(Path(checkpoint_path).expanduser().resolve())
            self._module.eval()
        else:
            raise ValueError("Either `policy_jit` or `checkpoint_path` must be provided.")

    def _load_actor_critic_from_checkpoint(self, checkpoint_path: Path):
        from rsl_rl.modules import ActorCritic

        obs_template = {"obs": torch.zeros((1, self.spec.obs_dim), device=self.device)}
        obs_groups = {"policy": ["obs"], "critic": ["obs"]}
        actor_critic = ActorCritic(
            obs=obs_template,
            obs_groups=obs_groups,
            num_actions=self.spec.action_dim,
            actor_obs_normalization=self.spec.actor_obs_normalization,
            critic_obs_normalization=self.spec.critic_obs_normalization,
            actor_hidden_dims=list(self.spec.actor_hidden_dims),
            critic_hidden_dims=list(self.spec.critic_hidden_dims),
            activation=self.spec.activation,
            init_noise_std=self.spec.init_noise_std,
        ).to(self.device)

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return actor_critic

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if self._mode == "jit":
            return self._module(obs)
        return self._module.act_inference({"obs": obs})

