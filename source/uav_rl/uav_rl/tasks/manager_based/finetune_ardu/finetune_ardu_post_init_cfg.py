from isaaclab.utils import configclass

from ..ardupilot_finetune.ardupilot_finetune_post_init_cfg import ArduPilotFineTunePostInitCfg


@configclass
class FineTuneArduPostInitCfg(ArduPilotFineTunePostInitCfg):
    """Canonical ArduPilot fine-tune post-init config."""

    pass
