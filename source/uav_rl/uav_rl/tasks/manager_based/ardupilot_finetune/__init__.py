import gymnasium as gym

from ..vanilla import agents as vanilla_agents


gym.register(
    id="ardupilot_finetune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ardupilot_finetune_env_cfg:ArduPilotFineTuneEnvCfg",
        "rl_games_cfg_entry_point": f"{vanilla_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.ardupilot_finetune_env_cfg:ArduPilotFineTunePPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{vanilla_agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{vanilla_agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{vanilla_agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Uav-ArduPilotFineTune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ardupilot_finetune_env_cfg:ArduPilotFineTuneEnvCfg",
        "rl_games_cfg_entry_point": f"{vanilla_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.ardupilot_finetune_env_cfg:ArduPilotFineTunePPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{vanilla_agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{vanilla_agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{vanilla_agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
