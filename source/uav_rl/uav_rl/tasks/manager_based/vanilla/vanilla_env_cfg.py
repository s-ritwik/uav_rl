import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from uav_rl.assets import IRIS_CFG

from . import mdp

PLATFORM_ARUCO_TEXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "Aruco" / "aruco_mark_fractal.png"
)

DOMAIN_RANDOMIZATION_CFG = mdp.VanillaDomainRandomizationCfg(
    enabled=False,
    mass_noise_enabled=True,
    mass_noise_probability=0.5,
    mass_noise_std_kg=0.1,
    mass_noise_clip_kg=0.3,
    action_delay_enabled=True,
    action_delay_probability=0.5,
    action_delay_steps_range=(1, 3),
    state_estimation_noise_enabled=True,
    state_estimation_noise_probability=0.5,
    position_noise_std_m=0.02,
    linear_velocity_noise_std_mps=0.03,
    angular_velocity_noise_std_rps=0.03,
    attitude_noise_std_rad=0.015,
    projected_gravity_noise_std=0.02,
    thrust_asymmetry_enabled=True,
    thrust_asymmetry_probability=0.5,
    thrust_asymmetry_scale_range=(0.9, 1.1),
    motor_lag_enabled=True,
    motor_lag_probability=0.5,
    motor_lag_time_constant_s_range=(0.02, 0.08),
)

PLATFORM_STAGE_TRACK_XY = mdp.PlatformMotionStageCfg(
    name="track_xy",
    x=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 8),
        amplitude_range=(0.4, 2),
        frequency_range_hz=(0.2, 0.4),
        phase_range_rad=(0.0, 2.0 * math.pi),
        spectral_decay=1.0,
    ),
    y=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 8),
        amplitude_range=(0.4, 2),
        frequency_range_hz=(0.2, 0.4),
        phase_range_rad=(0.0, 2.0 * math.pi),
        spectral_decay=1.0,
    ),
    max_linear_speed=1.2,
    max_linear_acceleration=5.0,
)

PLATFORM_STAGE_TRACK_XY_ROLL_PITCH = PLATFORM_STAGE_TRACK_XY.replace(
    name="track_xy_roll_pitch",
    roll=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * math.pi),
        spectral_decay=1.0,
    ),
    pitch=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * math.pi),
        spectral_decay=1.0,
    ),
    max_angular_speed=0.75,
    max_angular_acceleration=2.5,
)

PLATFORM_STAGE_TRACK_XY_ROLL_PITCH_HEAVE = PLATFORM_STAGE_TRACK_XY_ROLL_PITCH.replace(
    name="track_xy_roll_pitch_heave",
    z=mdp.HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        phase_range_rad=(0.0, 2.0 * math.pi),
        spectral_decay=1.0,
    ),
    max_linear_speed=2.25,
    max_linear_acceleration=6.0,
)


@configclass
class VanillaSceneCfg(InteractiveSceneCfg):
    """Scene config: local Iris on a flat plane."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=IRIS_CFG.spawn.replace(usd_path="/home/rycker/src/uav_rl/source/uav_rl/uav_rl/assets/robots/iris/iris_capsule.usd"),
    )

    # Track contact forces on robot bodies for contact-based termination.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/platform",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.28, 0.28),
                roughness=0.4,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )


@configclass
class ActionsCfg:
    """Policy action is [vx, vy, vz, yaw_rate]."""

    control = mdp.PX4LikeVelocityActionCfg(
        class_type=mdp.PX4LikeVelocityAction,
        asset_name="robot",
        action_scale=(1.0, 1.0, 1.0, 1.0),
        action_offset=(0.0, 0.0, 0.0, 0.0),
        velocity_limits=(1.2, 1.2, 1),
        yaw_rate_limit=3.0,
    )


@configclass
class ObservationsCfg:
    """Observations for learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        root_pos_rel = ObsTerm(func=mdp.root_pos_rel)
        root_lin_vel_rel = ObsTerm(func=mdp.root_lin_vel_rel)
        root_quat_rel = ObsTerm(func=mdp.root_quat_rel)
        root_ang_vel_rel = ObsTerm(func=mdp.root_ang_vel_rel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity_noisy)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Environment reset terms."""

    domain_randomization = EventTerm(
        func=mdp.SampleVanillaDomainRandomization,
        mode="reset",
        params={
            "rand_cfg": DOMAIN_RANDOMIZATION_CFG,
            "mass_asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
        },
    )

    add_platform_top_decal = EventTerm(
        func=mdp.add_platform_top_decal,
        mode="startup",
        params={
            "platform_name": "platform",
            "platform_size": (1.0, 1.0, 0.2),
            "texture_path": str(PLATFORM_ARUCO_TEXTURE_PATH),
        },
    )

    move_platform = EventTerm(
        func=mdp.MultiSinePlatformMotion,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            # Swap this preset as training progresses: XY -> deck attitude -> heave.
            "stage_cfg": PLATFORM_STAGE_TRACK_XY,
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
                "z": (0.7, 5),
                "roll": (-0.2, 0.2),
                "pitch": (-0.15, 0.15),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for hovering 1.0 m above the platform center."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # Track target hover setpoint in platform frame: (x, y, z) = (0, 0, 1.0).
    position_track = RewTerm(
        func=mdp.position_error_tanh,
        weight=2.5,
        params={
            "target_pos": (0.0, 0.0, 1.0),
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
        },
    )
    # horizontal_position = RewTerm(
    #     func=mdp.horizontal_position_error_l2,
    #     weight=-1.5,
    #     params={
    #         "target_xy": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "reference_asset_cfg": SceneEntityCfg("platform"),
    #     },
    # )
    vertical_position = RewTerm(
        func=mdp.vertical_position_error_l1,
        weight=-2.0,
        params={
            "target_height": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
        },
    )

    # Stabilize around the hover setpoint.
    horizontal_speed = RewTerm(func=mdp.horizontal_speed_l2, weight=-0.08)
    vertical_speed = RewTerm(func=mdp.vertical_speed_l2, weight=-0.08)
    angular_rate = RewTerm(func=mdp.angular_rate_l2, weight=-0.05)
    yaw_error = RewTerm(
        func=mdp.yaw_error_l2,
        weight=-1,
        params={"target_yaw": 0.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")})
    # horizontal_velocity_match = RewTerm(
    #     func=mdp.horizontal_velocity_error_l2,
    #     weight=-0.5,
    #     params={
    #         "target_rel_xy": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "reference_asset_cfg": SceneEntityCfg("platform"),
    #     },
    # )
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    # action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.003)


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    capsule_contact = DoneTerm(
        func=mdp.illegal_contact_with_debug,
        params={
            # ContactSensor resolves rigid-body names, not mesh child prim names, so I've added a negligible mass Capsule 
            # collider to the main body of the Iris USD and track that body's contacts here.
            # In this USD, available bodies are: body, rotor0, rotor1, rotor2, rotor3.
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
            "threshold": 1.0,
            "print_every_steps": 1,
        },
    )
    crash_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.1})
    crash_high = DoneTerm(func=mdp.root_height_above_maximum, params={"maximum_height": 7.0})
    out_of_bounds = DoneTerm(func=mdp.root_distance_from_origin, params={"max_distance": 9.0})


@configclass
class VanillaEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based vanilla UAV environment using Iris + PX4-like controller."""

    scene: VanillaSceneCfg = VanillaSceneCfg(num_envs=1024, env_spacing=10.0)

    domain_randomization: mdp.VanillaDomainRandomizationCfg = DOMAIN_RANDOMIZATION_CFG
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 10.0
        self.events.domain_randomization.params["rand_cfg"] = self.domain_randomization

        # Required so contact sensors receive contact reports from the USD articulation.
        self.scene.robot.spawn.activate_contact_sensors = True

        self.viewer.eye = (8.0, 8.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        self.sim.dt = 1.0 / 250.0
        self.sim.render_interval = 4
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
