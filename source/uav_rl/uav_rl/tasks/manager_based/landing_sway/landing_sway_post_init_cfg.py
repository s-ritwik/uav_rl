from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from . import mdp

if TYPE_CHECKING:
    from .landing_sway_env_cfg import LandingSwayEnvCfg


@configclass
class LandingSwayRewardWeightsCfg:
    """Reward weights applied in ``LandingSwayEnvCfg.__post_init__``."""

    # mdp.is_alive: +ve reward each non-terminal step.
    alive: float = 0.2
    # mdp.failure_termination_penalty: scales failure penalty value (usually keep at 1.0).
    terminated: float = 15.0
    # mdp.touchdown_termination_reward: optional extra reward on touchdown termination (usually 0.0).
    touchdown_terminated: float = 0.0
    # mdp.horizontal_position_error_tanh wrt platform-frame target XY [0, 0].
    position_track: float = 100
    # mdp.vertical_position_error_l1: |rel_z - target_height| term.
    vertical_position: float = 0.0
    # mdp.vertical_clearance_excess_l1: linear penalty for clearance above threshold. 
    # pushes agent to land
    vertical_clearance_excess: float = 0#-1.0
    # mdp.horizontal_speed_l2: penalize XY linear speed.
    horizontal_speed: float = -0.08
    # mdp.vertical_speed_l2: penalize Z linear speed.
    vertical_speed: float = -0.08
    # mdp.angular_rate_l2: penalize body angular rates.
    angular_rate: float = -0.05
    # mdp.yaw_error_l2: penalize yaw error to target yaw.
    yaw_error: float = -1.0
    # mdp.flat_orientation_l2: penalize tilt from upright.
    upright: float = -2.0
    # mdp.touchdown_quality_reward multiplier. Keep 1.0 when using explicit good/bad touchdown values below.
    touchdown_quality: float = 1.0


@configclass
class LandingSwayTouchdownCfg:
    """Touchdown detection + quality thresholds."""

    # Contact-force threshold that marks touchdown onset.
    force_threshold_n: float = 2.0
    # Good touchdown if descent_speed <= this value.
    max_touchdown_speed_mps: float = 0.4
    # XY-center tolerance used only when require_xy_within_box=True.
    max_xy_error_m: float = 0.40
    # Stage switch: False -> train only for low touchdown speed; True -> also require near-box touchdown.
    require_xy_within_box: bool = False
    # Reward value applied on good touchdown event.
    good_touchdown_reward: float = 3000.0
    # Reward value applied on bad touchdown event.
    bad_touchdown_reward: float = -50.0


@configclass
class LandingSwayVerticalClearanceCfg:
    """Linear vertical-clearance penalty threshold."""

    # Penalty activates when z_clearance > threshold_m.
    threshold_m: float = 0.0


@configclass
class LandingSwayTerminationPenaltyCfg:
    """Penalty applied for selected failure termination terms."""

    # Penalty value used by mdp.failure_termination_penalty for matched failure terms.
    failure_penalty: float = -10.0
    # Termination term names considered as failures (touchdown intentionally excluded).
    failure_term_names: tuple[str, ...] = ("time_out", "capsule_contact", "crash_low", "crash_high", "out_of_bounds")


@configclass
class LandingSwayTerminationThresholdsCfg:
    """Thresholds for termination terms (not weights)."""

    # contact_forces threshold for capsule_contact termination.
    capsule_contact_threshold_n: float = 1.0
    # root height below this -> crash_low.
    crash_low_min_height_m: float = -2.0
    # root height above this -> crash_high.
    crash_high_max_height_m: float = 7.0
    # XY distance from env origin above this -> out_of_bounds.
    out_of_bounds_max_distance_m: float = 6.0


@configclass
class LandingSwaySceneLayoutCfg:
    """Scene-wide layout knobs applied before manager startup."""

    env_spacing: float = 10.0


@configclass
class LandingSwayResetPoseRangeCfg:
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
class LandingSwayResetVelocityRangeCfg:
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
class LandingSwayResetSpawnCfg:
    """Grouped reset spawn ranges for pose and velocity."""

    pose_range: LandingSwayResetPoseRangeCfg = LandingSwayResetPoseRangeCfg()
    velocity_range: LandingSwayResetVelocityRangeCfg = LandingSwayResetVelocityRangeCfg()


@configclass
class LandingSwayPlatformPlacementCfg:
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
class LandingSwayPlatformMotionCfg:
    """Platform placement and motion overrides."""

    placement: LandingSwayPlatformPlacementCfg = LandingSwayPlatformPlacementCfg()
    stage_cfg: mdp.PlatformMotionStageCfg = PLATFORM_STAGE_TRACK_XY_CFG


@configclass
class LandingSwayPostInitCfg:
    """Single place to tune layout, reset, reward, platform, and domain randomization."""

    scene: LandingSwaySceneLayoutCfg = LandingSwaySceneLayoutCfg()
    reset_spawn: LandingSwayResetSpawnCfg = LandingSwayResetSpawnCfg()
    reward_weights: LandingSwayRewardWeightsCfg = LandingSwayRewardWeightsCfg()
    vertical_clearance: LandingSwayVerticalClearanceCfg = LandingSwayVerticalClearanceCfg()
    touchdown: LandingSwayTouchdownCfg = LandingSwayTouchdownCfg()
    termination_penalty: LandingSwayTerminationPenaltyCfg = LandingSwayTerminationPenaltyCfg()
    termination_thresholds: LandingSwayTerminationThresholdsCfg = LandingSwayTerminationThresholdsCfg()
    platform_motion: LandingSwayPlatformMotionCfg = LandingSwayPlatformMotionCfg()
    vehicle_z0_m: float = 0.053

    domain_randomization: mdp.LandingSwayDomainRandomizationCfg = mdp.LandingSwayDomainRandomizationCfg(
        # Flag for overall DR enable/disable
        enabled=True,
        # Additive noise on mass
        mass_noise_enabled=True,
        mass_noise_probability=0.5,
        mass_noise_std_kg=0.1,
        mass_noise_clip_kg=0.3,
        # Action delay DR
        action_delay_enabled=True,
        action_delay_probability=0.3,
        action_delay_steps_range=(1, 3),
        # State estimation noise DR
        state_estimation_noise_enabled=True,
        state_estimation_noise_probability=0.5,
        position_noise_std_m=0.03,
        # unmodeled dynamics and disturbances.
        linear_velocity_noise_std_mps=0.08,
        angular_velocity_noise_std_rps=0.04,
        attitude_noise_std_rad=0.015,
        projected_gravity_noise_std=0.04,
        # Thrust asymmetry DR
        thrust_asymmetry_enabled=True,
        thrust_asymmetry_probability=0.5,
        thrust_asymmetry_scale_range=(0.9, 1.1),
        # Motor lag DR
        motor_lag_enabled=True,
        motor_lag_probability=0.4,
        motor_lag_time_constant_s_range=(0.02, 0.08),
        # Domain randomisation on  Vel PID Gains
        velocity_gain_noise_enabled=True,
        velocity_gain_noise_probability=0.3,
        velocity_p_gain_noise_std=(0.15, 0.15, 0.35),
        velocity_i_gain_noise_std=(0.05, 0.05, 0.20),
        velocity_d_gain_noise_std=(0.03, 0.03, 0.05),
    )

    def apply(self, env_cfg: LandingSwayEnvCfg) -> None:
        """Apply overrides to the environment config before manager startup."""

        env_cfg.scene.env_spacing = self.scene.env_spacing

        env_cfg.scene.platform.init_state.pos = self.platform_motion.placement.pos
        env_cfg.scene.platform.init_state.rot = self.platform_motion.placement.rot
        env_cfg.events.move_platform.params["stage_cfg"] = self.platform_motion.stage_cfg

        env_cfg.events.reset_root.params["pose_range"] = self.reset_spawn.pose_range.as_dict()
        env_cfg.events.reset_root.params["velocity_range"] = self.reset_spawn.velocity_range.as_dict()

        env_cfg.rewards.alive.weight = self.reward_weights.alive
        env_cfg.rewards.terminated.weight = self.reward_weights.terminated
        env_cfg.rewards.touchdown_terminated.weight = self.reward_weights.touchdown_terminated
        env_cfg.rewards.position_track.weight = self.reward_weights.position_track
        env_cfg.rewards.vertical_position.weight = self.reward_weights.vertical_position
        env_cfg.rewards.vertical_clearance_excess.weight = self.reward_weights.vertical_clearance_excess
        env_cfg.rewards.horizontal_speed.weight = self.reward_weights.horizontal_speed
        env_cfg.rewards.vertical_speed.weight = self.reward_weights.vertical_speed
        env_cfg.rewards.angular_rate.weight = self.reward_weights.angular_rate
        env_cfg.rewards.yaw_error.weight = self.reward_weights.yaw_error
        env_cfg.rewards.upright.weight = self.reward_weights.upright
        env_cfg.rewards.touchdown_quality.weight = self.reward_weights.touchdown_quality

        env_cfg.rewards.vertical_clearance_excess.params["clearance_threshold_m"] = float(self.vertical_clearance.threshold_m)
        env_cfg.rewards.terminated.params["penalty"] = float(self.termination_penalty.failure_penalty)
        env_cfg.rewards.terminated.params["failure_term_names"] = tuple(self.termination_penalty.failure_term_names)

        # Touchdown reward/termination thresholds should match.
        env_cfg.rewards.touchdown_quality.params["touchdown_force_threshold"] = float(self.touchdown.force_threshold_n)
        env_cfg.rewards.touchdown_quality.params["max_touchdown_speed_mps"] = float(self.touchdown.max_touchdown_speed_mps)
        env_cfg.rewards.touchdown_quality.params["max_xy_error_m"] = float(self.touchdown.max_xy_error_m)
        env_cfg.rewards.touchdown_quality.params["require_xy_within_box"] = bool(self.touchdown.require_xy_within_box)
        env_cfg.rewards.touchdown_quality.params["good_touchdown_reward"] = float(self.touchdown.good_touchdown_reward)
        env_cfg.rewards.touchdown_quality.params["bad_touchdown_reward"] = float(self.touchdown.bad_touchdown_reward)
        env_cfg.terminations.touchdown.params["threshold"] = float(self.touchdown.force_threshold_n)

        env_cfg.terminations.capsule_contact.params["threshold"] = float(self.termination_thresholds.capsule_contact_threshold_n)
        env_cfg.terminations.crash_low.params["minimum_height"] = float(self.termination_thresholds.crash_low_min_height_m)
        env_cfg.terminations.crash_high.params["maximum_height"] = float(self.termination_thresholds.crash_high_max_height_m)
        env_cfg.terminations.out_of_bounds.params["max_distance"] = float(self.termination_thresholds.out_of_bounds_max_distance_m)

        env_cfg.domain_randomization = self.domain_randomization
        env_cfg.events.domain_randomization.params["rand_cfg"] = env_cfg.domain_randomization
