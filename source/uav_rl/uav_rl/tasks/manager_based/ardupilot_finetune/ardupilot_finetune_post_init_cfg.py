from isaaclab.utils import configclass

from ..vanilla.mdp.randomization import VanillaDomainRandomizationCfg
from ..vanilla.vanilla_post_init_cfg import (
    PLATFORM_STAGE_TRACK_XY_CFG,
    VanillaPlatformMotionCfg,
    VanillaPlatformPlacementCfg,
    VanillaPostInitCfg,
    VanillaResetPoseRangeCfg,
    VanillaResetSpawnCfg,
    VanillaResetVelocityRangeCfg,
    VanillaSceneLayoutCfg,
)
from . import mdp


ARDUPILOT_FINETUNE_STAGE_CFG = PLATFORM_STAGE_TRACK_XY_CFG.replace(
    name="ardupilot_finetune_track_xy",
    x=PLATFORM_STAGE_TRACK_XY_CFG.x.replace(amplitude_range=(0.2, 1.0), frequency_range_hz=(0.15, 0.35)),
    y=PLATFORM_STAGE_TRACK_XY_CFG.y.replace(amplitude_range=(0.2, 1.0), frequency_range_hz=(0.15, 0.35)),
    max_linear_speed=1.0,
    max_linear_acceleration=4.0,
)


@configclass
class ArduPilotFineTunePostInitCfg(VanillaPostInitCfg):
    """Task tuning for ArduPilot SITL fine-tuning."""

    scene: VanillaSceneLayoutCfg = VanillaSceneLayoutCfg(env_spacing=20.0)
    reset_spawn: VanillaResetSpawnCfg = VanillaResetSpawnCfg(
        pose_range=VanillaResetPoseRangeCfg(
            x=(0.0, 0.0),
            y=(0.0, 0.0),
            z=(0.07, 0.07),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        velocity_range=VanillaResetVelocityRangeCfg(
            x=(0.0, 0.0),
            y=(0.0, 0.0),
            z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    platform_motion: VanillaPlatformMotionCfg = VanillaPlatformMotionCfg(
        placement=VanillaPlatformPlacementCfg(pos=(1.5, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
        stage_cfg=ARDUPILOT_FINETUNE_STAGE_CFG,
    )
    domain_randomization: VanillaDomainRandomizationCfg = VanillaDomainRandomizationCfg(
        enabled=False,
        mass_noise_enabled=False,
        action_delay_enabled=False,
        state_estimation_noise_enabled=False,
        thrust_asymmetry_enabled=False,
        motor_lag_enabled=False,
    )
    runtime_cfg: mdp.ArduPilotFineTuneRuntimeCfg = mdp.ArduPilotFineTuneRuntimeCfg()

    def apply(self, env_cfg) -> None:
        super().apply(env_cfg)
        env_cfg.scene.robot.init_state.pos = (
            self.reset_spawn.pose_range.x[0],
            self.reset_spawn.pose_range.y[0],
            self.reset_spawn.pose_range.z[0],
        )
        env_cfg.events.reset_root = None
        env_cfg.events.move_platform.func = mdp.MultiSinePlatformMotionAfterReady
        self.runtime_cfg.num_sitl_envs = env_cfg.scene.num_envs
        env_cfg.runtime_cfg = self.runtime_cfg
