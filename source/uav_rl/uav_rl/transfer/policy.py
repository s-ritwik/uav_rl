from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
from torch import nn


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
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        model_state = checkpoint["model_state_dict"]
        actor = self._build_actor_from_state_dict(model_state).to(self.device)
        actor.load_state_dict(self._extract_actor_state_dict(model_state), strict=True)
        return actor

    def _activation_module(self) -> nn.Module:
        activation = self.spec.activation.lower()
        if activation == "elu":
            return nn.ELU()
        if activation == "relu":
            return nn.ReLU()
        if activation == "leaky_relu":
            return nn.LeakyReLU()
        if activation == "selu":
            return nn.SELU()
        if activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation for checkpoint reconstruction: {self.spec.activation}")

    def _extract_actor_state_dict(self, model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        actor_state: dict[str, torch.Tensor] = {}
        for key, value in model_state.items():
            if key.startswith("actor."):
                actor_state[key[len("actor."):]] = value
        if not actor_state:
            raise KeyError("Checkpoint does not contain `actor.*` parameters.")
        return actor_state

    def _build_actor_from_state_dict(self, model_state: dict[str, torch.Tensor]) -> nn.Sequential:
        actor_state = self._extract_actor_state_dict(model_state)
        layer_ids = sorted(
            {
                int(match.group(1))
                for key in actor_state
                if (match := re.match(r"(\d+)\.weight$", key)) is not None
            }
        )
        if not layer_ids:
            raise KeyError("Checkpoint actor state dict does not contain linear layer weights.")

        modules: list[nn.Module] = []
        last_layer_id = layer_ids[-1]
        for layer_id in layer_ids:
            weight = actor_state[f"{layer_id}.weight"]
            out_features, in_features = weight.shape
            modules.append(nn.Linear(in_features, out_features))
            if layer_id != last_layer_id:
                modules.append(self._activation_module())
        return nn.Sequential(*modules)

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if self._mode == "jit":
            return self._module(obs)
        return self._module(obs)
