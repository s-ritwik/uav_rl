from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from . import mdp

if TYPE_CHECKING:
    from .vanilla_env_cfg import VanillaEnvCfg


@configclass
class VanillaRewardWeightsCfg:
    """Reward weights applied in ``VanillaEnvCfg.__post_init__``."""

    alive: float = 0.2
    terminated: float = -10.0
    position_track: float = 2.5
    vertical_position: float = -2.0
    horizontal_speed: float = -0.08
    vertical_speed: float = -0.08
    angular_rate: float = -0.05
    yaw_error: float = -1.0
    upright: float = -3.0


@configclass
class VanillaSceneLayoutCfg:
    """Scene-wide layout knobs applied before manager startup."""

    env_spacing: float = 10.0


@configclass
class VanillaResetPoseRangeCfg:
    """Robot reset pose ranges used by ``reset_root_state_uniform``."""

    x: tuple[float, float] = (-5.0, 5.0)
    y: tuple[float, float] = (-5.0, 5.0)
    z: tuple[float, float] = (0.7, 5.0)
    roll: tuple[float, float] = (-0.2, 0.2)
    pitch: tuple[float, float] = (-0.15, 0.15)
    yaw: tuple[float, float] = (-3.141592653589793, 3.141592653589793)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }


@configclass
class VanillaResetVelocityRangeCfg:
    """Robot reset velocity ranges used by ``reset_root_state_uniform``."""

    x: tuple[float, float] = (-0.2, 0.2)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (-0.2, 0.2)
    roll: tuple[float, float] = (-0.2, 0.2)
    pitch: tuple[float, float] = (-0.2, 0.2)
    yaw: tuple[float, float] = (-0.5, 0.5)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }


@configclass
class VanillaResetSpawnCfg:
    """Grouped reset spawn ranges for pose and velocity."""

    pose_range: VanillaResetPoseRangeCfg = VanillaResetPoseRangeCfg()
    velocity_range: VanillaResetVelocityRangeCfg = VanillaResetVelocityRangeCfg()


@configclass
class VanillaPlatformPlacementCfg:
    """Platform initial placement in the scene."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.1)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


PLATFORM_STAGE_TRACK_XY_CFG = mdp.PlatformMotionStageCfg(
    name="track_xy",
    x=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 8),
        amplitude_range=(0.4, 2.0),
        frequency_range_hz=(0.2, 0.4),
        phase_range_rad=(0.0, 2.0 * 3.141592653589793),
        spectral_decay=1.0,
    ),
    y=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 8),
        amplitude_range=(0.4, 2.0),
        frequency_range_hz=(0.2, 0.4),
        phase_range_rad=(0.0, 2.0 * 3.141592653589793),
        spectral_decay=1.0,
    ),
    max_linear_speed=1.2,
    max_linear_acceleration=5.0,
)

PLATFORM_STAGE_TRACK_XY_ROLL_PITCH_CFG = PLATFORM_STAGE_TRACK_XY_CFG.replace(
    name="track_xy_roll_pitch",
    roll=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * 3.141592653589793),
        spectral_decay=1.0,
    ),
    pitch=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * 3.141592653589793),
        spectral_decay=1.0,
    ),
    max_angular_speed=0.75,
    max_angular_acceleration=2.5,
)

PLATFORM_STAGE_TRACK_XY_ROLL_PITCH_HEAVE_CFG = PLATFORM_STAGE_TRACK_XY_ROLL_PITCH_CFG.replace(
    name="track_xy_roll_pitch_heave",
    z=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * 3.141592653589793),
        spectral_decay=1.0,
    ),
    max_linear_speed=2.25,
    max_linear_acceleration=6.0,
)


@configclass
class VanillaPlatformMotionCfg:
    """Platform placement and motion overrides."""

    placement: VanillaPlatformPlacementCfg = VanillaPlatformPlacementCfg()
    stage_cfg: mdp.PlatformMotionStageCfg = PLATFORM_STAGE_TRACK_XY_CFG


@configclass
class VanillaPostInitCfg:
    """Single place to tune layout, reset, reward, platform, and domain randomization."""

    scene: VanillaSceneLayoutCfg = VanillaSceneLayoutCfg()
    reset_spawn: VanillaResetSpawnCfg = VanillaResetSpawnCfg()
    reward_weights: VanillaRewardWeightsCfg = VanillaRewardWeightsCfg()
    platform_motion: VanillaPlatformMotionCfg = VanillaPlatformMotionCfg()

    domain_randomization: mdp.VanillaDomainRandomizationCfg = mdp.VanillaDomainRandomizationCfg(
        enabled=True,
        mass_noise_enabled=True,
        mass_noise_probability=0.5,
        mass_noise_std_kg=0.1,
        mass_noise_clip_kg=0.3,
        action_delay_enabled=True,
        action_delay_probability=0.3,
        action_delay_steps_range=(1, 3),
        state_estimation_noise_enabled=True,
        state_estimation_noise_probability=0.5,
        position_noise_std_m=0.03,
        linear_velocity_noise_std_mps=0.08,
        angular_velocity_noise_std_rps=0.04,
        attitude_noise_std_rad=0.015,
        projected_gravity_noise_std=0.04,
        thrust_asymmetry_enabled=True,
        thrust_asymmetry_probability=0.5,
        thrust_asymmetry_scale_range=(0.9, 1.1),
        motor_lag_enabled=True,
        motor_lag_probability=0.4,
        motor_lag_time_constant_s_range=(0.02, 0.08),
    )

    def apply(self, env_cfg: VanillaEnvCfg) -> None:
        """Apply overrides to the environment config before manager startup."""

        env_cfg.scene.env_spacing = self.scene.env_spacing

        env_cfg.scene.platform.init_state.pos = self.platform_motion.placement.pos
        env_cfg.scene.platform.init_state.rot = self.platform_motion.placement.rot
        env_cfg.events.move_platform.params["stage_cfg"] = self.platform_motion.stage_cfg

        env_cfg.events.reset_root.params["pose_range"] = self.reset_spawn.pose_range.as_dict()
        env_cfg.events.reset_root.params["velocity_range"] = self.reset_spawn.velocity_range.as_dict()

        env_cfg.rewards.alive.weight = self.reward_weights.alive
        env_cfg.rewards.terminated.weight = self.reward_weights.terminated
        env_cfg.rewards.position_track.weight = self.reward_weights.position_track
        env_cfg.rewards.vertical_position.weight = self.reward_weights.vertical_position
        env_cfg.rewards.horizontal_speed.weight = self.reward_weights.horizontal_speed
        env_cfg.rewards.vertical_speed.weight = self.reward_weights.vertical_speed
        env_cfg.rewards.angular_rate.weight = self.reward_weights.angular_rate
        env_cfg.rewards.yaw_error.weight = self.reward_weights.yaw_error
        env_cfg.rewards.upright.weight = self.reward_weights.upright

        env_cfg.domain_randomization = self.domain_randomization
        env_cfg.events.domain_randomization.params["rand_cfg"] = env_cfg.domain_randomization
