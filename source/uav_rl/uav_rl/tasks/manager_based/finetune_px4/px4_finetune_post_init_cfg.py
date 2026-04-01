from isaaclab.utils import configclass

from ..vanilla.mdp.randomization import VanillaDomainRandomizationCfg
from ..vanilla.vanilla_post_init_cfg import (
    PLATFORM_STAGE_TRACK_XY_CFG,
    VanillaPlatformMotionCfg,
    VanillaPlatformPlacementCfg,
    VanillaPostInitCfg,
    VanillaRewardWeightsCfg,
    VanillaResetPoseRangeCfg,
    VanillaResetSpawnCfg,
    VanillaResetVelocityRangeCfg,
    VanillaSceneLayoutCfg,
)
from . import mdp


PX4_FINETUNE_STAGE_CFG = PLATFORM_STAGE_TRACK_XY_CFG.replace(
    name="px4_finetune_track_xy",
    x=PLATFORM_STAGE_TRACK_XY_CFG.x.replace(amplitude_range=(0.2, 1.0), frequency_range_hz=(0.15, 0.35)),
    y=PLATFORM_STAGE_TRACK_XY_CFG.y.replace(amplitude_range=(0.2, 1.0), frequency_range_hz=(0.15, 0.35)),
    max_linear_speed=1.0,
    max_linear_acceleration=4.0,
)


@configclass
class PX4FineTunePostInitCfg(VanillaPostInitCfg):
    """Task tuning for PX4 SITL fine-tuning."""

    scene: VanillaSceneLayoutCfg = VanillaSceneLayoutCfg(env_spacing=20.0)
    reward_weights: VanillaRewardWeightsCfg = VanillaRewardWeightsCfg()
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
        stage_cfg=PX4_FINETUNE_STAGE_CFG,
    )
    domain_randomization: VanillaDomainRandomizationCfg = VanillaDomainRandomizationCfg(
        enabled=False,
        mass_noise_enabled=False,
        action_delay_enabled=False,
        state_estimation_noise_enabled=False,
        thrust_asymmetry_enabled=False,
        motor_lag_enabled=False,
    )
    runtime_cfg: mdp.PX4FineTuneRuntimeCfg = mdp.PX4FineTuneRuntimeCfg(
        launch=mdp.PX4LaunchCfg(
            backend_connection_baseport=4560,
            offboard_baseport=14540,
            autolaunch=True,
            startup_delay_s=0.0,
        ),
        reset=mdp.PX4ResetModeCfg(
            mode="hard",
            allow_full_takeoff_option=True,
            auto_takeoff_alt_m=3.0,
            ready_timeout_s=180.0,
            takeoff_altitude_tolerance_m=0.12,
            hover_speed_tolerance_mps=0.12,
            hover_settle_time_s=0.75,
            soft_reset_settle_time_s=0.5,
            full_takeoff_every_n_resets=0,
        ),
        bridge=mdp.PX4BridgeCfg(
            command_rate_hz=25.0,
            cmd_timeout_s=0.5,
            arm_delay_s=3.0,
            require_position_ready=True,
            source_system_base=200,
        ),
        sensors=mdp.PX4SensorCfg(
            imu_update_rate_hz=250.0,
            gps_update_rate_hz=250.0,
            barometer_update_rate_hz=250.0,
            magnetometer_update_rate_hz=250.0,
            home_latitude_deg=38.736832,
            home_longitude_deg=-9.137977,
            home_altitude_m=90.0,
        ),
    )

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
