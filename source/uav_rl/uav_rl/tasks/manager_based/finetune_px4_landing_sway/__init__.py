import gymnasium as gym

from ..landing_sway import agents as landing_sway_agents


gym.register(
    id="finetune_px4_landing_sway",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.px4_finetune_landing_sway_env_cfg:PX4FineTuneLandingSwayEnvCfg",
        "rl_games_cfg_entry_point": f"{landing_sway_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.px4_finetune_landing_sway_env_cfg:PX4FineTuneLandingSwayPPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{landing_sway_agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{landing_sway_agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{landing_sway_agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


gym.register(
    id="Uav-FineTunePX4LandingSway-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.px4_finetune_landing_sway_env_cfg:PX4FineTuneLandingSwayEnvCfg",
        "rl_games_cfg_entry_point": f"{landing_sway_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.px4_finetune_landing_sway_env_cfg:PX4FineTuneLandingSwayPPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{landing_sway_agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{landing_sway_agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{landing_sway_agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
