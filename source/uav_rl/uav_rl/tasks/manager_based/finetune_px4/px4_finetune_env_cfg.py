from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..vanilla.agents.rsl_rl_ppo_cfg import PPORunnerCfg as VanillaPPORunnerCfg
from ..vanilla.vanilla_env_cfg import ActionsCfg as VanillaActionsCfg
from ..vanilla.vanilla_env_cfg import TerminationsCfg as VanillaTerminationsCfg
from ..vanilla.vanilla_env_cfg import VanillaEnvCfg, VanillaSceneCfg
from . import mdp
from .px4_finetune_post_init_cfg import PX4FineTunePostInitCfg


@configclass
class ActionsCfg(VanillaActionsCfg):
    """Use PX4 SITL OFFBOARD velocity control instead of the vanilla PX4-like controller."""

    control = mdp.PX4OffboardVelocityActionCfg(
        class_type=mdp.PX4OffboardVelocityAction,
        asset_name="robot",
        action_scale=(1.0, 1.0, 1.0, 1.0),
        action_offset=(0.0, 0.0, 0.0, 0.0),
        velocity_limits=(1.2, 1.2, 1.0),
        yaw_rate_limit=3.0,
    )


@configclass
class TerminationsCfg(VanillaTerminationsCfg):
    """Ignore task terminations until PX4 reaches the pre-policy hover state."""

    time_out = DoneTerm(func=mdp.time_out_after_takeoff, time_out=True)
    capsule_contact = DoneTerm(
        func=mdp.illegal_contact_after_takeoff,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
            "threshold": 1.0,
            "print_every_steps": 1,
        },
    )
    crash_low = DoneTerm(func=mdp.root_height_below_minimum_after_takeoff, params={"minimum_height": 0.1})
    crash_high = DoneTerm(func=mdp.root_height_above_maximum_after_takeoff, params={"maximum_height": 7.0})
    out_of_bounds = DoneTerm(func=mdp.root_distance_from_origin_after_takeoff, params={"max_distance": 9.0})


@configclass
class PX4FineTuneEnvCfg(VanillaEnvCfg):
    """Low-throughput PX4 SITL fine-tune task that keeps the vanilla policy contract."""

    scene: VanillaSceneCfg = VanillaSceneCfg(num_envs=8, env_spacing=20.0)
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    post_init_cfg: PX4FineTunePostInitCfg = PX4FineTunePostInitCfg()
    runtime_cfg: mdp.PX4FineTuneRuntimeCfg = mdp.PX4FineTuneRuntimeCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the working standalone PX4 transfer host timing.
        self.decimation = 10
        self.sim.dt = 1.0 / 250.0
        self.sim.render_interval = self.decimation
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class PX4FineTunePPORunnerCfg(VanillaPPORunnerCfg):
    """RSL-RL defaults for low-parallel PX4 SITL fine-tuning from a pretrained vanilla policy."""

    experiment_name = "vanilla"
    num_steps_per_env = 48
    max_iterations = 1500
    save_interval = 25
