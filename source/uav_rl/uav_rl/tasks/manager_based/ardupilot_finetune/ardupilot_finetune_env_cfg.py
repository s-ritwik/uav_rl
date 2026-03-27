from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..vanilla.agents.rsl_rl_ppo_cfg import PPORunnerCfg as VanillaPPORunnerCfg
from ..vanilla.vanilla_env_cfg import ActionsCfg as VanillaActionsCfg
from ..vanilla.vanilla_env_cfg import TerminationsCfg as VanillaTerminationsCfg
from ..vanilla.vanilla_env_cfg import VanillaEnvCfg, VanillaSceneCfg
from . import mdp
from .ardupilot_finetune_post_init_cfg import ArduPilotFineTunePostInitCfg


@configclass
class ActionsCfg(VanillaActionsCfg):
    """Use ArduPilot-in-the-loop instead of the vanilla PX4-like controller."""

    control = mdp.ArduPilotGuidedVelocityActionCfg(
        class_type=mdp.ArduPilotGuidedVelocityAction,
        asset_name="robot",
        action_scale=(1.0, 1.0, 1.0, 1.0),
        action_offset=(0.0, 0.0, 0.0, 0.0),
        velocity_limits=(1.2, 1.2, 1.0),
        yaw_rate_limit=3.0,
    )


@configclass
class TerminationsCfg(VanillaTerminationsCfg):
    """Ignore task terminations until ArduPilot reaches the pre-policy hover state."""

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
class ArduPilotFineTuneEnvCfg(VanillaEnvCfg):
    """Low-throughput fine-tune task that keeps the vanilla policy contract."""

    scene: VanillaSceneCfg = VanillaSceneCfg(num_envs=8, env_spacing=20.0)
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    post_init_cfg: ArduPilotFineTunePostInitCfg = ArduPilotFineTunePostInitCfg()
    runtime_cfg: mdp.ArduPilotFineTuneRuntimeCfg = mdp.ArduPilotFineTuneRuntimeCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the Pegasus standalone ArduPilot app timing: run physics at 800 Hz and
        # keep the outer policy/update rate at 25 Hz.
        self.decimation = 32
        self.sim.dt = 1.0 / 800.0
        self.sim.render_interval = self.decimation
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class ArduPilotFineTunePPORunnerCfg(VanillaPPORunnerCfg):
    """RSL-RL defaults for low-parallel fine-tuning from a pretrained vanilla policy."""

    # Keep the same experiment root as vanilla so train.py --resume can load a vanilla checkpoint
    # without requiring changes to the launcher script.
    experiment_name = "vanilla"
    num_steps_per_env = 48
    max_iterations = 1500
    save_interval = 25
