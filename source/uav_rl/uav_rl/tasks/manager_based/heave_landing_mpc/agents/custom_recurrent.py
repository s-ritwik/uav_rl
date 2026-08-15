from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.networks import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


def _as_hidden_dim_list(hidden_dims: int | Sequence[int] | None, fallback_dim: int, fallback_layers: int) -> list[int]:
    if hidden_dims is None:
        return [int(fallback_dim)] * int(fallback_layers)
    if isinstance(hidden_dims, int):
        return [int(hidden_dims)]
    return [int(dim) for dim in hidden_dims]


class StackedMemory(nn.Module):
    """Stack multiple 1-layer GRU/LSTM blocks with potentially different hidden sizes."""

    def __init__(self, input_size: int, hidden_dims: Sequence[int], rnn_type: str = "gru") -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("StackedMemory requires at least one hidden dimension.")
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Expected 'gru' or 'lstm'.")
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM

        self.hidden_dims = list(hidden_dims)
        self.blocks = nn.ModuleList()
        current_input_size = int(input_size)
        for hidden_dim in self.hidden_dims:
            self.blocks.append(rnn_cls(input_size=current_input_size, hidden_size=int(hidden_dim), num_layers=1))
            current_input_size = int(hidden_dim)
        self.hidden_state: HiddenState = None

    @property
    def output_dim(self) -> int:
        return int(self.hidden_dims[-1])

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        batch_mode = masks is not None
        if batch_mode:
            hidden_states = self._normalize_batch_hidden_state(hidden_state)
            out = input
            for block, block_hidden_state in zip(self.blocks, hidden_states, strict=True):
                out, _ = block(out, block_hidden_state)
            out = unpad_trajectories(out, masks)
            return out

        out = input.unsqueeze(0)
        hidden_states = self._normalize_inference_hidden_state()
        next_hidden_states: list[HiddenState] = []
        for block, block_hidden_state in zip(self.blocks, hidden_states, strict=True):
            out, next_hidden_state = block(out, block_hidden_state)
            next_hidden_states.append(next_hidden_state)
        self.hidden_state = self._pack_hidden_state(next_hidden_states)
        return out

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self.hidden_state = None if hidden_state is None else hidden_state
            return
        if self.hidden_state is None:
            return
        hidden_states = self._normalize_any_hidden_state(self.hidden_state)
        for block_hidden_state in hidden_states:
            self._zero_done_hidden_state(block_hidden_state, dones)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_state is None:
            return
        hidden_states = self._normalize_any_hidden_state(self.hidden_state)
        if dones is None:
            detached_hidden_states = [self._detach_block_hidden_state(block_hidden_state) for block_hidden_state in hidden_states]
            self.hidden_state = self._pack_hidden_state(detached_hidden_states)
            return
        for block_hidden_state in hidden_states:
            self._detach_done_hidden_state(block_hidden_state, dones)

    def _normalize_batch_hidden_state(self, hidden_state: HiddenState) -> list[HiddenState]:
        if hidden_state is None:
            raise ValueError("Hidden states not passed to stacked memory module during policy update")
        return self._normalize_any_hidden_state(hidden_state)

    def _normalize_inference_hidden_state(self) -> list[HiddenState]:
        if self.hidden_state is None:
            return [None] * len(self.blocks)
        return self._normalize_any_hidden_state(self.hidden_state)

    def _normalize_any_hidden_state(self, hidden_state: HiddenState) -> list[HiddenState]:
        if hidden_state is None:
            return [None] * len(self.blocks)
        if self.rnn_type == "gru":
            if isinstance(hidden_state, torch.Tensor):
                if len(self.blocks) != 1:
                    raise ValueError("Expected one hidden-state tensor per GRU block.")
                return [hidden_state]
            if isinstance(hidden_state, tuple):
                return list(hidden_state)
            if isinstance(hidden_state, list):
                return hidden_state
            raise TypeError(f"Unsupported hidden state type for GRU: {type(hidden_state)}")
        # LSTM path expects each block state to be a tuple(h, c).
        if len(self.blocks) == 1 and isinstance(hidden_state, tuple) and len(hidden_state) == 2 and isinstance(hidden_state[0], torch.Tensor):
            return [hidden_state]
        if isinstance(hidden_state, list):
            return hidden_state
        if isinstance(hidden_state, tuple):
            return list(hidden_state)
        raise TypeError(f"Unsupported hidden state type for LSTM: {type(hidden_state)}")

    def _pack_hidden_state(self, hidden_states: list[HiddenState]) -> HiddenState:
        if len(hidden_states) == 1:
            return hidden_states[0]
        return tuple(hidden_states)

    @staticmethod
    def _zero_done_hidden_state(hidden_state: HiddenState, dones: torch.Tensor) -> None:
        if isinstance(hidden_state, tuple):
            for tensor in hidden_state:
                tensor[..., dones == 1, :] = 0.0
        else:
            hidden_state[..., dones == 1, :] = 0.0

    @staticmethod
    def _detach_block_hidden_state(hidden_state: HiddenState) -> HiddenState:
        if isinstance(hidden_state, tuple):
            return tuple(tensor.detach() for tensor in hidden_state)
        return hidden_state.detach()

    @staticmethod
    def _detach_done_hidden_state(hidden_state: HiddenState, dones: torch.Tensor) -> None:
        if isinstance(hidden_state, tuple):
            for tensor in hidden_state:
                tensor[..., dones == 1, :] = tensor[..., dones == 1, :].detach()
        else:
            hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()


class ActorCriticSeparateRecurrent(ActorCriticRecurrent):
    """Custom recurrent actor-critic with asymmetric GRU/LSTM stacks for actor and critic."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        actor_rnn_hidden_dims: int | Sequence[int] | None = None,
        critic_rnn_hidden_dims: int | Sequence[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticSeparateRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys())
            )
        nn.Module.__init__(self)

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticSeparateRecurrent module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticSeparateRecurrent module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        self.actor_rnn_hidden_dims = _as_hidden_dim_list(actor_rnn_hidden_dims, rnn_hidden_dim, rnn_num_layers)
        self.critic_rnn_hidden_dims = _as_hidden_dim_list(critic_rnn_hidden_dims, rnn_hidden_dim, rnn_num_layers)

        self.memory_a = StackedMemory(num_actor_obs, self.actor_rnn_hidden_dims, rnn_type)
        if self.state_dependent_std:
            self.actor = MLP(self.memory_a.output_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(self.memory_a.output_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor RNN stack ({rnn_type}): {self.actor_rnn_hidden_dims}")
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.memory_c = StackedMemory(num_critic_obs, self.critic_rnn_hidden_dims, rnn_type)
        self.critic = MLP(self.memory_c.output_dim, 1, critic_hidden_dims, activation)
        print(f"Critic RNN stack ({rnn_type}): {self.critic_rnn_hidden_dims}")
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs, masks, hidden_state).squeeze(0)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs).squeeze(0)
        if self.state_dependent_std:
            return self.actor(out_mem)[..., 0, :]
        return self.actor(out_mem)

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        out_mem = self.memory_c(obs, masks, hidden_state).squeeze(0)
        return self.critic(out_mem)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
