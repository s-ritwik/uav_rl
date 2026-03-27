from isaaclab.utils import configclass

from ..ardupilot_finetune.ardupilot_finetune_env_cfg import (
    ActionsCfg,
    ArduPilotFineTuneEnvCfg,
    ArduPilotFineTunePPORunnerCfg,
    TerminationsCfg,
)
from .finetune_ardu_post_init_cfg import FineTuneArduPostInitCfg


@configclass
class FineTuneArduEnvCfg(ArduPilotFineTuneEnvCfg):
    """Canonical ArduPilot fine-tune task name."""

    post_init_cfg: FineTuneArduPostInitCfg = FineTuneArduPostInitCfg()


@configclass
class FineTuneArduPPORunnerCfg(ArduPilotFineTunePPORunnerCfg):
    """Alias runner cfg under the finetune_ardu task namespace."""

    pass
