from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)

import rsl_rl.modules as rsl_rl_modules
import rsl_rl.runners.on_policy_runner as rsl_rl_on_policy_runner

from .custom_recurrent import ActorCriticSeparateRecurrent


rsl_rl_modules.ActorCriticSeparateRecurrent = ActorCriticSeparateRecurrent
rsl_rl_on_policy_runner.ActorCriticSeparateRecurrent = ActorCriticSeparateRecurrent


@configclass
class HeaveLandingCustomGruActorCriticCfg(RslRlPpoActorCriticRecurrentCfg):
    """Custom recurrent actor-critic config with asymmetric GRU stacks.

    Example:
    - `actor_rnn_hidden_dims=[128]`
    - `critic_rnn_hidden_dims=[256, 128]`
    """

    class_name: str = "ActorCriticSeparateRecurrent"
    actor_rnn_hidden_dims: int | list[int] | None = MISSING
    critic_rnn_hidden_dims: int | list[int] | None = MISSING


@configclass
class HeaveLandingFfPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "heave_landing_ff"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        critic_hidden_dims=[1024, 512, 128],
        actor_hidden_dims=[128, 64, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=1e-2,
        max_grad_norm=1.0,
    )


@configclass
class HeaveLandingGruPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 100
    experiment_name = "heave_landing_gru"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    # Custom GRU additions:
    # 1. `actor_rnn_hidden_dims` can be an int or a list like [128] or [256, 128].
    # 2. `critic_rnn_hidden_dims` can be different from the actor.
    # 3. Lists create sequential 1-layer GRU blocks with those hidden sizes.
    policy = HeaveLandingCustomGruActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        critic_hidden_dims=[128],
        actor_hidden_dims=[32],
        activation="elu",
        rnn_type="gru",
        # Fallback values used only if the custom actor/critic lists are omitted.
        # rnn_hidden_dim=128,
        # rnn_num_layers=1,
        # Current default: single GRU layer for actor, larger single GRU layer for critic.
        actor_rnn_hidden_dims=[128, 64],
        critic_rnn_hidden_dims=[512, 256],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5e-3,
        num_learning_epochs=5,
        num_mini_batches=6,
        learning_rate=1.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=1e-2,
        max_grad_norm=1.0,
    )


# Default registry target for the generic task id.
PPORunnerCfg = HeaveLandingFfPPORunnerCfg
