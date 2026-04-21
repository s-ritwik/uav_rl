from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..landing_sway.agents.rsl_rl_ppo_cfg import PPORunnerCfg as LandingSwayPPORunnerCfg
from ..landing_sway.landing_sway_env_cfg import (
    ActionsCfg as LandingSwayActionsCfg,
    LandingSwayEnvCfg,
    ObservationsCfg as LandingSwayObservationsCfg,
    RewardsCfg as LandingSwayRewardsCfg,
    LandingSwaySceneCfg,
    TerminationsCfg as LandingSwayTerminationsCfg,
)
from . import mdp
from .px4_finetune_landing_sway_post_init_cfg import PX4FineTuneLandingSwayPostInitCfg


@configclass
class ActionsCfg(LandingSwayActionsCfg):
    """Use PX4 SITL OFFBOARD velocity control while keeping the landing_sway action contract."""

    control = mdp.PX4OffboardVelocityActionCfg(
        class_type=mdp.PX4OffboardVelocityAction,
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
class TerminationsCfg(LandingSwayTerminationsCfg):
    """Defer landing_sway terminations until PX4 reaches the pre-policy hover state."""

    time_out = DoneTerm(func=mdp.time_out_after_takeoff, time_out=True)
    attitude_tilt = DoneTerm(
        func=mdp.root_roll_pitch_above_maximum_after_takeoff,
        params={
            "maximum_roll_deg": 35.0,
            "maximum_pitch_deg": 35.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    touchdown = DoneTerm(
        func=mdp.touchdown_terminate_after_takeoff,
        params={
            "threshold": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_asset_cfg": SceneEntityCfg("platform"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
        },
    )
    crash_low = DoneTerm(func=mdp.root_height_below_minimum_after_takeoff, params={"minimum_height": 0.1})
    crash_high = DoneTerm(func=mdp.root_height_above_maximum_after_takeoff, params={"maximum_height": 7.0})
    out_of_bounds = DoneTerm(func=mdp.root_distance_from_origin_after_takeoff, params={"max_distance": 9.0})


@configclass
class PX4FineTuneLandingSwayEnvCfg(LandingSwayEnvCfg):
    """Landing-sway task executed through PX4 SITL OFFBOARD velocity control."""

    scene: LandingSwaySceneCfg = LandingSwaySceneCfg(num_envs=8, env_spacing=20.0)
    observations: LandingSwayObservationsCfg = LandingSwayObservationsCfg()
    rewards: LandingSwayRewardsCfg = LandingSwayRewardsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    post_init_cfg: PX4FineTuneLandingSwayPostInitCfg = PX4FineTuneLandingSwayPostInitCfg()
    runtime_cfg: mdp.PX4FineTuneRuntimeCfg = mdp.PX4FineTuneRuntimeCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 10
        self.sim.dt = 1.0 / 250.0
        self.sim.render_interval = self.decimation
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class PX4FineTuneLandingSwayPPORunnerCfg(LandingSwayPPORunnerCfg):
    """RSL-RL defaults for low-parallel PX4 SITL landing_sway fine-tuning."""

    experiment_name = "landing_sway"
    num_steps_per_env = 48
    max_iterations = 1500
    save_interval = 25
