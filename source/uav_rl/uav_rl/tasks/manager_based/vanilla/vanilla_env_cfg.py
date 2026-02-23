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
        velocity_limits=(6.0, 6.0, 4.0),
        yaw_rate_limit=3.0,
    )


@configclass
class ObservationsCfg:
    """Observations for learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        root_pos_rel = ObsTerm(func=mdp.root_pos_rel)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        last_action = ObsTerm(func=mdp.last_action)
        command_velocity = ObsTerm(func=mdp.command_velocity)
        command_yaw_rate = ObsTerm(func=mdp.command_yaw_rate)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Environment reset terms."""

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
        func=mdp.move_platform_sinusoidal,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "amplitude_m": 1.0,
            "frequency_hz": 0.3,
            "axis": "x",
            "phase_rad": 0.0,
            "phase_per_env": True,
            "phase_span_rad": 2.0 * math.pi,
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.8, 1.2),
                "roll": (-0.15, 0.15),
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
    horizontal_position = RewTerm(
        func=mdp.horizontal_position_error_l2,
        weight=-1.5,
        params={
            "target_xy": (0.0, 0.0),
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

    # Stabilize around the hover setpoint.
    horizontal_speed = RewTerm(func=mdp.horizontal_speed_l2, weight=-0.08)
    vertical_speed = RewTerm(func=mdp.vertical_speed_l2, weight=-0.08)
    angular_rate = RewTerm(func=mdp.angular_rate_l2, weight=-0.05)
    yaw_error = RewTerm(
        func=mdp.yaw_error_l2,
        weight=-0.25,
        params={"target_yaw": 0.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")})

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    # action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.003)


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    capsule_contact = DoneTerm(
        func=mdp.illegal_contact_with_debug,
        params={
            # ContactSensor resolves rigid-body names, not mesh child prim names.
            # In this USD, available bodies are: body, rotor0, rotor1, rotor2, rotor3.
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
            "threshold": 1.0,
            "print_every_steps": 1,
        },
    )
    crash_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.1})
    crash_high = DoneTerm(func=mdp.root_height_above_maximum, params={"maximum_height": 4.0})
    out_of_bounds = DoneTerm(func=mdp.root_distance_from_origin, params={"max_distance": 5.0})


@configclass
class VanillaEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based vanilla UAV environment using Iris + PX4-like controller."""

    scene: VanillaSceneCfg = VanillaSceneCfg(num_envs=1024, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 10.0

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
