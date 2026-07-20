import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from .landing_sway_vision_post_init_cfg import PLATFORM_STAGE_TRACK_XY_CFG, LandingSwayVisionPostInitCfg

PLATFORM_ARUCO_TEXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "Aruco" / "aruco_mark_fractal.png"
)

@configclass
class LandingSwaySceneCfg(InteractiveSceneCfg):
    """Scene config: local Iris on a flat plane."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=IRIS_CFG.spawn.replace(usd_path="/home/rycker/src/uav_rl/source/uav_rl/uav_rl/assets/robots/iris/iris_legs.usd"),
    )

    # Track contact forces on robot bodies for contact-based termination.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    # platform = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/platform",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 1.0, 0.2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.28, 0.28, 0.28),
    #             roughness=0.4,
    #             metallic=0.0,
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/platform",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=4.0,
                dynamic_friction=4.0,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
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

    control = mdp.LandingSwayVelocityActionCfg(
        class_type=mdp.LandingSwayVelocityAction,
        asset_name="robot",
        action_scale=(1.0, 1.0, 1.0, 1.0),
        action_offset=(0.0, 0.0, 0.0, 0.0),
        velocity_lower_limits=(-1.2, -1.2, -1.0),
        velocity_upper_limits=(1.2, 1.2, 1.0),
        yaw_rate_limit=3.0,
        yaw_rate_lower_limit=-3.0,
        yaw_rate_upper_limit=3.0,
    )


@configclass
class ObservationsCfg:
    """Observations for learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        vision_rel_pos = ObsTerm(func=mdp.vision_rel_pos)
        vision_rel_lin_vel = ObsTerm(func=mdp.vision_rel_lin_vel)
        vision_rel_quat = ObsTerm(func=mdp.vision_rel_quat)
        vision_rel_ang_vel = ObsTerm(func=mdp.vision_rel_ang_vel)
        vision_line_of_sight = ObsTerm(func=mdp.vision_line_of_sight)
        vision_status = ObsTerm(func=mdp.vision_status)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Environment reset terms."""

    domain_randomization = EventTerm(
        func=mdp.SampleLandingSwayDomainRandomization,
        mode="reset",
        params={
            "rand_cfg": mdp.LandingSwayDomainRandomizationCfg(),
            "mass_asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
        },
    )

    add_platform_top_decal = EventTerm(
        func=mdp.add_platform_top_decal,
        mode="startup",
        params={
            "platform_name": "platform",
            "platform_size": (1.0, 1.0, 0.2),
            "decal_size_xy": (0.70, 0.70),
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
            "stage_cfg": PLATFORM_STAGE_TRACK_XY_CFG,
            "stationary_env_probability": 0.0,
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
    """Reward terms for landing on the platform with a smooth touchdown."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    terminated = RewTerm(
        func=mdp.failure_termination_penalty,
        weight=1.0,
        params={
            "penalty": -10.0,
            "failure_term_names": ("time_out", "attitude_tilt", "crash_low", "crash_high", "out_of_bounds"),
        },
    )
    touchdown_terminated = RewTerm(
        func=mdp.touchdown_termination_reward,
        weight=0.0,
        params={"touchdown_term_name": "touchdown"},
    )

    # Track target hover setpoint in platform frame XY: (x, y) = (0, 0).
    position_track = RewTerm(
        func=mdp.horizontal_position_error_tanh,
        weight=2.5,
        params={
            "target_xy": (0.0, 0.0),
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
        },
    )
    vertical_position = RewTerm(
        func=mdp.vertical_position_error_l1,
        weight=-2.0,
        params={
            "target_height": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
        },
    )
    vertical_clearance_excess = RewTerm(
        func=mdp.vertical_clearance_excess_l1,
        weight=-1.0,
        params={
            "clearance_threshold_m": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
        },
    )

    # Stabilize around the hover setpoint.
    horizontal_speed = RewTerm(func=mdp.horizontal_speed_l2, weight=-0.08)
    vertical_speed = RewTerm(func=mdp.vertical_speed_l2, weight=-0.08)
    velocity_action_rate_x = RewTerm(func=mdp.raw_action_rate_component_l2, weight=0.0, params={"action_index": 0})
    velocity_action_rate_y = RewTerm(func=mdp.raw_action_rate_component_l2, weight=0.0, params={"action_index": 1})
    velocity_action_rate_z = RewTerm(func=mdp.raw_action_rate_component_l2, weight=0.0, params={"action_index": 2})
    uav_acceleration = RewTerm(
        func=mdp.uav_linear_acceleration_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["body"])},
    )
    angular_rate = RewTerm(func=mdp.angular_rate_l2, weight=-0.05)
    angular_velocity_rate = RewTerm(func=mdp.angular_velocity_rate_l2, weight=0.0)
    angular_rate_xy = RewTerm(func=mdp.angular_rate_xy_l2, weight=0.0)
    yaw_rate_error = RewTerm(
        func=mdp.yaw_rate_error_l2,
        weight=-0.05,
        params={"target_yaw_rate": 0.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    yaw_error = RewTerm(
        func=mdp.yaw_error_l2,
        weight=-1,
        params={"target_yaw": 0.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # penaliszing xy of projected gracity
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")})
    touchdown_quality = RewTerm(
        func=mdp.touchdown_quality_reward,
        weight=1.0,
        params={
            "max_touchdown_speed_mps": 0.25,
            "max_xy_error_m": 0.20,
            "require_xy_within_box": False,
            "require_attitude_within_limits": True,
            "max_touchdown_roll_deg": 10.0,
            "max_touchdown_pitch_deg": 10.0,
            "max_touchdown_yaw_deg": 10.0,
            "target_touchdown_yaw_deg": 0.0,
            "touchdown_force_threshold": 2.0,
            "good_touchdown_reward": 5.0,
            "bad_touchdown_reward": -2.0,
            "center_proximity_bonus": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
        },
    )
    horizontal_velocity_match = RewTerm(
        func=mdp.horizontal_velocity_error_tanh,
        weight=0.5,
        params={
            "target_rel_xy": (0.0, 0.0),
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
        },
    )
    near_target_action_xy = RewTerm(
        func=mdp.near_target_action_xy_l2,
        weight=0.0,
        params={
            "target_xy": (0.0, 0.0),
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
        },
    )
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_magnitude_x = RewTerm(func=mdp.raw_action_component_l2, weight=0.0, params={"action_index": 0})
    action_magnitude_y = RewTerm(func=mdp.raw_action_component_l2, weight=0.0, params={"action_index": 1})
    action_magnitude_z = RewTerm(func=mdp.raw_action_component_l2, weight=0.0, params={"action_index": 2})
    action_magnitude_yaw_rate = RewTerm(func=mdp.raw_action_component_l2, weight=0.0, params={"action_index": 3})


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    attitude_tilt = DoneTerm(
        func=mdp.root_roll_pitch_above_maximum,
        params={
            "maximum_roll_deg": 35.0,
            "maximum_pitch_deg": 35.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    touchdown = DoneTerm(
        func=mdp.touchdown_terminate,
        params={
            "threshold": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
        },
    )
    crash_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.1})
    crash_high = DoneTerm(func=mdp.root_height_above_maximum, params={"maximum_height": 7.0})
    out_of_bounds = DoneTerm(func=mdp.root_distance_from_origin, params={"max_distance": 9.0})


@configclass
class CurriculumCfg:
    """Logging-only curriculum terms used to surface episode metrics."""

    touchdown_quality_metrics = CurrTerm(
        func=mdp.touchdown_quality_metrics,
        params={
            "max_touchdown_speed_mps": 0.25,
            "max_xy_error_m": 0.20,
            "require_xy_within_box": False,
            "require_attitude_within_limits": True,
            "max_touchdown_roll_deg": 10.0,
            "max_touchdown_pitch_deg": 10.0,
            "max_touchdown_yaw_deg": 10.0,
            "target_touchdown_yaw_deg": 0.0,
        },
    )


@configclass
class LandingSwayVisionEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based landing-sway UAV environment using Iris + PX4-like controller."""

    scene: LandingSwaySceneCfg = LandingSwaySceneCfg(num_envs=1024, env_spacing=10.0)

    post_init_cfg: LandingSwayVisionPostInitCfg = LandingSwayVisionPostInitCfg()
    domain_randomization: mdp.LandingSwayDomainRandomizationCfg = mdp.LandingSwayDomainRandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 10.0
        self.post_init_cfg.apply(self)

        # Required so contact sensors receive contact reports from the USD articulation.
        self.scene.robot.spawn.activate_contact_sensors = True

        self.viewer.eye = (8.0, 8.0, 6.0)
        self.viewer.lookat = (-5.0, -5.0, 2.0)
        self.viewer.resolution = (1920, 1080)

        self.sim.dt = 1.0 / 250.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
