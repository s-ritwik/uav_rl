from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from . import mdp

if TYPE_CHECKING:
    from .heave_landing_env_cfg import HeaveLandingEnvCfg


@configclass
class HeaveLandingRewardWeightsCfg:
    """Reward weights applied in ``HeaveLandingEnvCfg.__post_init__``."""

    # mdp.is_alive: +ve reward each non-terminal step.
    alive: float = 0.2
    # mdp.failure_termination_penalty: scales failure penalty value 
    terminated: float = 50.0
    # mdp.touchdown_termination_reward: optional extra reward on touchdown termination 
    touchdown_terminated: float = 50.0
    # mdp.horizontal_position_error_tanh wrt platform-frame target XY [0, 0].
    position_track: float = 1.0
    # mdp.vertical_position_error_l1: |rel_z - target_height| term.
    vertical_position: float = 0.0
    # mdp.vertical_clearance_excess_l1: linear penalty for clearance above threshold. 
    # pushes agent to land
    vertical_clearance_excess: float = -0.5
    # mdp.horizontal_speed_l2: penalize XY linear speed.
    horizontal_speed: float = -0.08
    # mdp.vertical_speed_l2: penalize Z linear speed.
    vertical_speed: float = -0.01
    # mdp.raw_action_component_l2: penalize large raw policy vx action.
    action_magnitude_x: float = -2.5
    # mdp.raw_action_component_l2: penalize large raw policy vy action.
    action_magnitude_y: float = -2.5
    # mdp.raw_action_component_l2: penalize large raw policy vz action.
    action_magnitude_z: float = -2.5
    # mdp.raw_action_component_l2: penalize large raw policy yaw-rate action.
    action_magnitude_yaw_rate: float = -1.2
    # mdp.raw_action_rate_component_l2: continuity penalty on step-to-step change in raw policy vx action.
    velocity_action_rate_x: float = -40.0
    # mdp.raw_action_rate_component_l2: continuity penalty on step-to-step change in raw policy vy action.
    velocity_action_rate_y: float = -40.0
    # mdp.raw_action_rate_component_l2: continuity penalty on step-to-step change in raw policy vz action.
    velocity_action_rate_z: float = -20.0
    # mdp.uav_linear_acceleration_l2: penalize UAV body COM linear acceleration.
    uav_acceleration: float = -0.5
    # mdp.angular_rate_l2: penalize body angular rates.
    angular_rate: float = -0.5
    # mdp.angular_velocity_rate_l2: Continuity error :penalize step-to-step change in measured body-frame wx, wy, wz.
    angular_velocity_rate: float = -50.0
    # mdp.angular_rate_xy_l2: penalize body-frame roll/pitch rates only.
    angular_rate_xy: float = -1.0#0.0
    # mdp.yaw_rate_error_l2: penalize body-frame yaw-rate error around target yaw rate.
    yaw_rate_error: float = -10.0
    # mdp.yaw_error_l2: penalize yaw error to target yaw.
    yaw_error: float = -2.0
    # mdp.flat_orientation_l2: penalize tilt from upright.
    upright: float = -20.0
    # mdp.touchdown_quality_reward multiplier. Keep 1.0 when using explicit good/bad touchdown values below.
    touchdown_quality: float = 1.0
    # mdp.horizontal_velocity_error_tanh: positive bounded reward for matching platform XY velocity.
    horizontal_velocity_match: float = 0.0
    # mdp.near_target_action_xy_l2: penalize large raw xy action when already near the platform center.
    near_target_action_xy: float = 0.0


@configclass
class HeaveLandingPositionTrackCfg:
    """Shape parameters for the dense XY position reward."""

    std_m: float = 0.6


@configclass
class HeaveLandingNearTargetActionCfg:
    """Shape parameters for the near-target XY action penalty."""

    std_m: float = 0.6

@configclass
class HeaveLandingTouchdownCfg:
    """Touchdown detection + quality thresholds."""

    # Contact-force threshold that marks touchdown onset.
    force_threshold_n: float = 2.0
    # Good touchdown if descent_speed <= this value.
    max_touchdown_speed_mps: float = 0.25
    # XY-center tolerance used only when require_xy_within_box=True.
    max_xy_error_m: float = 0.2
    # Stage switch: False -> train only for low touchdown speed; True -> also require near-box touchdown.
    require_xy_within_box: bool = True
    # If True, good touchdown also requires roll/pitch/yaw to satisfy the attitude limits below.
    require_attitude_within_limits: bool = True
    # Good touchdown only if absolute roll at touchdown is within this limit.
    max_touchdown_roll_deg: float = 12.0
    # Good touchdown only if absolute pitch at touchdown is within this limit.
    max_touchdown_pitch_deg: float = 12.0
    # Good touchdown only if wrapped yaw error to target_touchdown_yaw_deg is within this limit.
    max_touchdown_yaw_deg: float = 80.0
    # World-frame yaw target used by the touchdown yaw gate.
    target_touchdown_yaw_deg: float = 0.0
    # Reward value applied on good touchdown event.
    good_touchdown_reward: float = 5000.0
    # Reward value applied on bad touchdown event.
    bad_touchdown_reward: float = -300.0
    # Extra shaped bonus on good touchdowns that increases as XY touchdown error approaches zero.
    center_proximity_bonus: float = 1000.0


@configclass
class HeaveLandingVerticalClearanceCfg:
    """Linear vertical-clearance penalty threshold."""

    # Penalty activates when z_clearance > threshold_m.
    threshold_m: float = 1.0


@configclass
class HeaveLandingTerminationPenaltyCfg:
    """Penalty applied for selected failure termination terms."""

    # Penalty value used by mdp.failure_termination_penalty for matched failure terms.
    failure_penalty: float = -10.0
    # Termination term names considered as failures (touchdown intentionally excluded).
    failure_term_names: tuple[str, ...] = ( "time_out","attitude_tilt", "crash_low", "crash_high", "out_of_bounds")


@configclass
class HeaveLandingTerminationThresholdsCfg:
    """Thresholds for termination terms (not weights)."""

    # absolute roll angle above this -> attitude_tilt.
    attitude_tilt_max_roll_deg: float = 30.0
    # absolute pitch angle above this -> attitude_tilt.
    attitude_tilt_max_pitch_deg: float = 30.0
    # root height below this -> crash_low.
    crash_low_min_height_m: float = -1.0
    # root height above this -> crash_high.
    crash_high_max_height_m: float = 7.0
    # XY distance from env origin above this -> out_of_bounds.
    out_of_bounds_max_distance_m: float = 6.0


@configclass
class HeaveLandingSceneLayoutCfg:
    """Scene-wide layout knobs applied before manager startup."""

    env_spacing: float = 6.0


@configclass
class HeaveLandingEpisodeCfg:
    """Episode-level timing knobs."""

    # Episode timeout used by the `time_out` termination term.
    timeout_s: float = 12.0


@configclass
class HeaveLandingActionCommandLimitsCfg:
    """Policy command limits applied before controller execution."""

    # Explicit lower clipping limits for commanded vx, vy, vz in m/s.
    velocity_lower_limits: tuple[float, float, float] = (-0.8, -0.8, -0.8)
    # Explicit upper clipping limits for commanded vx, vy, vz in m/s.
    velocity_upper_limits: tuple[float, float, float] = (0.8, 0.8, 1.0)
    # Legacy symmetric clipping limit for commanded yaw rate in rad/s.
    yaw_rate_limit: float = 3.0
    # Explicit lower clipping limit for commanded yaw rate in rad/s.
    yaw_rate_lower_limit: float = -35.0*math.pi/180.0
    # Explicit upper clipping limit for commanded yaw rate in rad/s.
    yaw_rate_upper_limit: float = 35.0*math.pi/180.0


@configclass
class HeaveLandingResetPoseRangeCfg:
    """Robot reset pose ranges used by ``reset_root_state_uniform``."""

    x: tuple[float, float] = (-0.3, 0.3)
    y: tuple[float, float] = (-0.3, 0.3)
    z: tuple[float, float] = (2.5, 4.0)
    roll: tuple[float, float] = (-0.2, 0.2)
    pitch: tuple[float, float] = (-0.15, 0.15)
    yaw: tuple[float, float] = (-3.141592653589793/10.0, 3.141592653589793/10.0)

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
class HeaveLandingResetVelocityRangeCfg:
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
class HeaveLandingResetSpawnCfg:
    """Grouped reset spawn ranges for pose and velocity."""

    pose_range: HeaveLandingResetPoseRangeCfg = HeaveLandingResetPoseRangeCfg()
    velocity_range: HeaveLandingResetVelocityRangeCfg = HeaveLandingResetVelocityRangeCfg()


@configclass
class HeaveLandingPlatformPlacementCfg:
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
    max_linear_speed=1.0,
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
        amplitude_range=(0.2, 1.0),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * 3.141592653589793),
        spectral_decay=1.0,
    ),
    max_linear_speed=2.25,
    max_linear_acceleration=6.0,
)

PLATFORM_STAGE_HEAVE_CFG = mdp.PlatformMotionStageCfg(
    name="heave",
    z=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.2, 1.0),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * 3.141592653589793),
        spectral_decay=1.0,
    ),
    max_linear_speed=2.25,
    max_linear_acceleration=6.0,
)


@configclass
class HeaveLandingPlatformMotionCfg:
    """Platform placement and episode-level motion sampling overrides."""

    placement: HeaveLandingPlatformPlacementCfg = HeaveLandingPlatformPlacementCfg()
    stationary_env_probability: float = 0.0


@configclass
class HeaveLandingCsvHeaveMotionCfg:
    """Recorded heave-trace playback settings."""

    dataset_dir: str = str(Path(__file__).resolve().parent / "train_data_normalised")
    sample_rate_hz: float = 20.0
    min_remaining_s: float = 60.0
    scale: float = .32
    bias_m: float = 1.5
    randomize_bias: bool = False
    bias_range_m: tuple[float, float] = (0.5, 2.5)


@configclass
class HeaveLandingPostInitCfg:
    """Single place to tune layout, reset, reward, platform, and domain randomization."""

    scene: HeaveLandingSceneLayoutCfg = HeaveLandingSceneLayoutCfg()
    episode: HeaveLandingEpisodeCfg = HeaveLandingEpisodeCfg()
    action_command_limits: HeaveLandingActionCommandLimitsCfg = HeaveLandingActionCommandLimitsCfg()
    reset_spawn: HeaveLandingResetSpawnCfg = HeaveLandingResetSpawnCfg()
    reward_weights: HeaveLandingRewardWeightsCfg = HeaveLandingRewardWeightsCfg()
    position_track: HeaveLandingPositionTrackCfg = HeaveLandingPositionTrackCfg()
    near_target_action: HeaveLandingNearTargetActionCfg = HeaveLandingNearTargetActionCfg()
    vertical_clearance: HeaveLandingVerticalClearanceCfg = HeaveLandingVerticalClearanceCfg()
    touchdown: HeaveLandingTouchdownCfg = HeaveLandingTouchdownCfg()
    termination_penalty: HeaveLandingTerminationPenaltyCfg = HeaveLandingTerminationPenaltyCfg()
    termination_thresholds: HeaveLandingTerminationThresholdsCfg = HeaveLandingTerminationThresholdsCfg()
    platform_motion: HeaveLandingPlatformMotionCfg = HeaveLandingPlatformMotionCfg()
    csv_heave_motion: HeaveLandingCsvHeaveMotionCfg = HeaveLandingCsvHeaveMotionCfg()
    vehicle_z0_m: float = 0.15 

    domain_randomization: mdp.HeaveLandingDomainRandomizationCfg = mdp.HeaveLandingDomainRandomizationCfg(
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

    def apply(self, env_cfg: HeaveLandingEnvCfg) -> None:
        """Apply overrides to the environment config before manager startup."""

        env_cfg.scene.env_spacing = self.scene.env_spacing
        env_cfg.episode_length_s = float(self.episode.timeout_s)

        env_cfg.scene.platform.init_state.pos = self.platform_motion.placement.pos
        env_cfg.scene.platform.init_state.rot = self.platform_motion.placement.rot
        env_cfg.events.move_platform.params["stationary_env_probability"] = float(
            self.platform_motion.stationary_env_probability
        )
        env_cfg.events.move_platform.params["dataset_dir"] = str(self.csv_heave_motion.dataset_dir)
        env_cfg.events.move_platform.params["sample_rate_hz"] = float(self.csv_heave_motion.sample_rate_hz)
        env_cfg.events.move_platform.params["min_remaining_s"] = float(self.csv_heave_motion.min_remaining_s)
        env_cfg.events.move_platform.params["scale"] = float(self.csv_heave_motion.scale)
        env_cfg.events.move_platform.params["bias_m"] = float(self.csv_heave_motion.bias_m)
        env_cfg.events.move_platform.params["randomize_bias"] = bool(self.csv_heave_motion.randomize_bias)
        env_cfg.events.move_platform.params["bias_range_m"] = tuple(float(x) for x in self.csv_heave_motion.bias_range_m)

        env_cfg.events.reset_root.params["pose_range"] = self.reset_spawn.pose_range.as_dict()
        env_cfg.events.reset_root.params["velocity_range"] = self.reset_spawn.velocity_range.as_dict()

        env_cfg.rewards.alive.weight = self.reward_weights.alive
        env_cfg.rewards.terminated.weight = self.reward_weights.terminated
        env_cfg.rewards.touchdown_terminated.weight = self.reward_weights.touchdown_terminated
        env_cfg.rewards.position_track.weight = self.reward_weights.position_track
        env_cfg.rewards.position_track.params["std"] = float(self.position_track.std_m)
        env_cfg.rewards.vertical_position.weight = self.reward_weights.vertical_position
        env_cfg.rewards.vertical_clearance_excess.weight = self.reward_weights.vertical_clearance_excess
        env_cfg.rewards.horizontal_speed.weight = self.reward_weights.horizontal_speed
        env_cfg.rewards.vertical_speed.weight = self.reward_weights.vertical_speed
        env_cfg.rewards.action_magnitude_x.weight = self.reward_weights.action_magnitude_x
        env_cfg.rewards.action_magnitude_y.weight = self.reward_weights.action_magnitude_y
        env_cfg.rewards.action_magnitude_z.weight = self.reward_weights.action_magnitude_z
        env_cfg.rewards.action_magnitude_yaw_rate.weight = self.reward_weights.action_magnitude_yaw_rate
        env_cfg.rewards.velocity_action_rate_x.weight = self.reward_weights.velocity_action_rate_x
        env_cfg.rewards.velocity_action_rate_y.weight = self.reward_weights.velocity_action_rate_y
        env_cfg.rewards.velocity_action_rate_z.weight = self.reward_weights.velocity_action_rate_z
        env_cfg.rewards.uav_acceleration.weight = self.reward_weights.uav_acceleration
        env_cfg.rewards.angular_rate.weight = self.reward_weights.angular_rate
        env_cfg.rewards.angular_velocity_rate.weight = self.reward_weights.angular_velocity_rate
        env_cfg.rewards.angular_rate_xy.weight = self.reward_weights.angular_rate_xy
        env_cfg.rewards.yaw_rate_error.weight = self.reward_weights.yaw_rate_error
        env_cfg.rewards.yaw_error.weight = self.reward_weights.yaw_error
        env_cfg.rewards.upright.weight = self.reward_weights.upright
        env_cfg.rewards.touchdown_quality.weight = self.reward_weights.touchdown_quality
        env_cfg.rewards.horizontal_velocity_match.weight = self.reward_weights.horizontal_velocity_match
        env_cfg.rewards.near_target_action_xy.weight = self.reward_weights.near_target_action_xy
        env_cfg.rewards.near_target_action_xy.params["std"] = float(self.near_target_action.std_m)

        env_cfg.rewards.vertical_clearance_excess.params["clearance_threshold_m"] = float(self.vertical_clearance.threshold_m)
        env_cfg.rewards.terminated.params["penalty"] = float(self.termination_penalty.failure_penalty)
        env_cfg.rewards.terminated.params["failure_term_names"] = tuple(self.termination_penalty.failure_term_names)

        # Touchdown reward/termination thresholds should match.
        env_cfg.rewards.touchdown_quality.params["touchdown_force_threshold"] = float(self.touchdown.force_threshold_n)
        env_cfg.rewards.touchdown_quality.params["max_touchdown_speed_mps"] = float(self.touchdown.max_touchdown_speed_mps)
        env_cfg.rewards.touchdown_quality.params["max_xy_error_m"] = float(self.touchdown.max_xy_error_m)
        env_cfg.rewards.touchdown_quality.params["require_xy_within_box"] = bool(self.touchdown.require_xy_within_box)
        env_cfg.rewards.touchdown_quality.params["require_attitude_within_limits"] = bool(
            self.touchdown.require_attitude_within_limits
        )
        env_cfg.rewards.touchdown_quality.params["max_touchdown_roll_deg"] = float(self.touchdown.max_touchdown_roll_deg)
        env_cfg.rewards.touchdown_quality.params["max_touchdown_pitch_deg"] = float(self.touchdown.max_touchdown_pitch_deg)
        env_cfg.rewards.touchdown_quality.params["max_touchdown_yaw_deg"] = float(self.touchdown.max_touchdown_yaw_deg)
        env_cfg.rewards.touchdown_quality.params["target_touchdown_yaw_deg"] = float(
            self.touchdown.target_touchdown_yaw_deg
        )
        env_cfg.rewards.touchdown_quality.params["good_touchdown_reward"] = float(self.touchdown.good_touchdown_reward)
        env_cfg.rewards.touchdown_quality.params["bad_touchdown_reward"] = float(self.touchdown.bad_touchdown_reward)
        env_cfg.rewards.touchdown_quality.params["center_proximity_bonus"] = float(self.touchdown.center_proximity_bonus)
        env_cfg.terminations.touchdown.params["threshold"] = float(self.touchdown.force_threshold_n)
        env_cfg.curriculum.touchdown_quality_metrics.params["max_touchdown_speed_mps"] = float(
            self.touchdown.max_touchdown_speed_mps
        )
        env_cfg.curriculum.touchdown_quality_metrics.params["max_xy_error_m"] = float(self.touchdown.max_xy_error_m)
        env_cfg.curriculum.touchdown_quality_metrics.params["require_xy_within_box"] = bool(
            self.touchdown.require_xy_within_box
        )
        env_cfg.curriculum.touchdown_quality_metrics.params["require_attitude_within_limits"] = bool(
            self.touchdown.require_attitude_within_limits
        )
        env_cfg.curriculum.touchdown_quality_metrics.params["max_touchdown_roll_deg"] = float(
            self.touchdown.max_touchdown_roll_deg
        )
        env_cfg.curriculum.touchdown_quality_metrics.params["max_touchdown_pitch_deg"] = float(
            self.touchdown.max_touchdown_pitch_deg
        )
        env_cfg.curriculum.touchdown_quality_metrics.params["max_touchdown_yaw_deg"] = float(
            self.touchdown.max_touchdown_yaw_deg
        )
        env_cfg.curriculum.touchdown_quality_metrics.params["target_touchdown_yaw_deg"] = float(
            self.touchdown.target_touchdown_yaw_deg
        )

        env_cfg.actions.control.velocity_lower_limits = tuple(
            float(v) for v in self.action_command_limits.velocity_lower_limits
        )
        env_cfg.actions.control.velocity_upper_limits = tuple(
            float(v) for v in self.action_command_limits.velocity_upper_limits
        )
        env_cfg.actions.control.yaw_rate_limit = float(self.action_command_limits.yaw_rate_limit)
        env_cfg.actions.control.yaw_rate_lower_limit = float(self.action_command_limits.yaw_rate_lower_limit)
        env_cfg.actions.control.yaw_rate_upper_limit = float(self.action_command_limits.yaw_rate_upper_limit)

        env_cfg.terminations.attitude_tilt.params["maximum_roll_deg"] = float(
            self.termination_thresholds.attitude_tilt_max_roll_deg
        )
        env_cfg.terminations.attitude_tilt.params["maximum_pitch_deg"] = float(
            self.termination_thresholds.attitude_tilt_max_pitch_deg
        )
        env_cfg.terminations.crash_low.params["minimum_height"] = float(self.termination_thresholds.crash_low_min_height_m)
        env_cfg.terminations.crash_high.params["maximum_height"] = float(self.termination_thresholds.crash_high_max_height_m)
        env_cfg.terminations.out_of_bounds.params["max_distance"] = float(self.termination_thresholds.out_of_bounds_max_distance_m)

        env_cfg.domain_randomization = self.domain_randomization
        env_cfg.events.domain_randomization.params["rand_cfg"] = env_cfg.domain_randomization
