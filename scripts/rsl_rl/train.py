# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video and camera-driven vision tasks
if args_cli.video or ("vision" in (args_cli.task or "").lower()):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import uav_rl.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _resolve_resume_path(log_root_path: str, load_run: str | None, load_checkpoint: str | None) -> str:
    """Resolve a checkpoint path with support for absolute paths and cross-experiment loading.

    IsaacLab's default helper resolves checkpoints under:
        <log_root_path>/<load_run>/<load_checkpoint>

    For transfer learning, we also support:
    - Passing a direct file path via --checkpoint
    - Passing a run directory path via --load_run, optionally with --checkpoint
    """

    # 1) Direct checkpoint file path.
    if load_checkpoint:
        ckpt_path = Path(load_checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (Path.cwd() / ckpt_path)
        if ckpt_path.is_file():
            return str(ckpt_path.resolve())

    # 2) Direct run directory path.
    if load_run:
        run_path = Path(load_run).expanduser()
        if not run_path.is_absolute():
            run_path = (Path.cwd() / run_path)
        if run_path.is_dir():
            # If a checkpoint filename was provided, prefer it.
            if load_checkpoint:
                candidate = run_path / load_checkpoint
                if candidate.is_file():
                    return str(candidate.resolve())
            # Otherwise, emulate get_checkpoint_path() behavior within the run directory.
            pattern = re.compile(load_checkpoint or r".*")
            candidates = [p.name for p in run_path.iterdir() if p.is_file() and pattern.match(p.name)]
            if not candidates:
                raise ValueError(f"No checkpoints in the directory: '{str(run_path)}' match '{load_checkpoint}'.")
            candidates.sort(key=lambda m: f"{m:0>15}")
            return str((run_path / candidates[-1]).resolve())

    # 3) Default: resolve under this experiment's log root.
    return get_checkpoint_path(log_root_path, load_run or ".*", load_checkpoint or ".*")


def _maybe_convert_scalar_std_checkpoint_to_log_std(checkpoint_path: str) -> str:
    """Convert older scalar-std checkpoints to log-std checkpoints for RSL-RL ActorCritic.

    Some checkpoints store an unconstrained `std` parameter (which can go negative).
    Newer configs can use `noise_std_type="log"` which stores `log_std` and ensures `std=exp(log_std) > 0`.

    This shim allows loading scalar-std checkpoints into log-std policies by rewriting the checkpoint
    to a temporary file.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        return checkpoint_path

    try:
        ckpt = torch.load(str(path), map_location="cpu")
    except Exception:
        return checkpoint_path

    if not isinstance(ckpt, dict):
        return checkpoint_path

    msd = ckpt.get("model_state_dict")
    if not isinstance(msd, dict):
        return checkpoint_path

    # Only convert the canonical key used by rsl_rl.modules.ActorCritic.
    if "std" not in msd or "log_std" in msd:
        return checkpoint_path

    std = msd.get("std")
    if not torch.is_tensor(std):
        try:
            std = torch.as_tensor(std)
        except Exception:
            return checkpoint_path

    # Prevent log(0) and keep numerical stability.
    log_std = torch.log(std.to(dtype=torch.float32).clamp_min(1.0e-6))
    msd = dict(msd)
    msd.pop("std", None)
    msd["log_std"] = log_std
    ckpt = dict(ckpt)
    ckpt["model_state_dict"] = msd

    tmp_path = Path("/tmp") / f"{path.stem}_logstd_{time.time_ns()}{path.suffix}"
    torch.save(ckpt, str(tmp_path))
    print(f"[INFO]: Converted checkpoint scalar std -> log_std: {path} -> {tmp_path}")
    return str(tmp_path)


def _maybe_warmup_ardupilot_takeoff(env, env_cfg):
    """Step zero actions until the autopilot SITL finishes the pre-policy takeoff sequence."""

    runtime_cfg = getattr(env_cfg, "runtime_cfg", None)
    if runtime_cfg is None or not getattr(runtime_cfg, "enabled", False):
        return

    # Gymnasium wrappers require an initial reset before any step.
    # This occurs before the ArduPilot runtime is started, so it does not
    # interfere with the later takeoff-to-policy handoff.
    env.reset()

    try:
        action_term = env.unwrapped.action_manager.get_term("control")
    except Exception:
        return

    if not hasattr(action_term, "all_envs_ready_for_policy"):
        return

    zero_actions = torch.zeros(
        (env.unwrapped.num_envs,) + env.unwrapped.single_action_space.shape,
        device=env.unwrapped.device,
        dtype=torch.float32,
    )
    warmup_timeout_s = float(getattr(runtime_cfg.reset, "ready_timeout_s", 90.0))
    target_altitude_m = float(getattr(runtime_cfg.reset, "auto_takeoff_alt_m", 0.0))

    print(
        f"[INFO]: Autopilot warmup: waiting for {env.unwrapped.num_envs} envs to reach "
        f"{target_altitude_m:.1f} m before PPO starts."
    )

    if hasattr(action_term, "set_warmup_active"):
        action_term.set_warmup_active(True)

    try:
        start_time = time.monotonic()
        last_status_time = 0.0
        while not action_term.all_envs_ready_for_policy():
            _, _, terminated, time_outs, _ = env.step(zero_actions)
            now = time.monotonic()
            if torch.any(terminated) or torch.any(time_outs):
                term_manager = getattr(env.unwrapped, "termination_manager", None)
                term_debug = {}
                if term_manager is not None:
                    for term_name in term_manager.active_terms:
                        term_debug[term_name] = bool(torch.any(term_manager.get_term(term_name)).item())
                raise RuntimeError(
                    "Autopilot warmup reset triggered before policy handoff. "
                    f"terminated={bool(torch.any(terminated).item())} "
                    f"time_outs={bool(torch.any(time_outs).item())} "
                    f"terms={term_debug}"
                )
            if now - start_time > warmup_timeout_s:
                raise RuntimeError(
                    "Autopilot warmup timed out after "
                    f"{warmup_timeout_s:.1f}s with {action_term.num_ready_envs()}/{env.unwrapped.num_envs} envs ready."
                )
            if now - last_status_time >= 5.0:
                altitudes = action_term.current_altitudes()
                mean_altitude = statistics.mean(altitudes) if altitudes else 0.0
                extra_status = ""
                if hasattr(action_term, "debug_statuses"):
                    statuses = action_term.debug_statuses()
                    if statuses:
                        status0 = statuses[0]
                        extra_status = (
                            ", env0 state="
                            f"{status0.get('takeoff_state')} connected={status0.get('connected')} "
                            f"gps={status0.get('gps_fix_ready')} pos={status0.get('position_estimate_ready')} "
                            f"speed={float(status0.get('speed_mps', 0.0)):.2f}"
                        )
                print(
                    "[INFO]: Autopilot warmup status: "
                    f"{action_term.num_ready_envs()}/{env.unwrapped.num_envs} ready, "
                    f"mean altitude {mean_altitude:.2f} m{extra_status}"
                )
                last_status_time = now

        # Let the first PPO-side reset keep the live hover state instead of disturbing SITL right after takeoff.
        if hasattr(action_term, "skip_next_runtime_reset"):
            action_term.skip_next_runtime_reset()
        print("[INFO]: Autopilot warmup complete. Starting PPO rollout collection.")
    finally:
        if hasattr(action_term, "set_warmup_active"):
            action_term.set_warmup_active(False)


def _stop_ardupilot_runtime(env):
    """Best-effort shutdown for SITL subprocesses owned by the action term."""

    if env is None:
        return

    try:
        action_term = env.unwrapped.action_manager.get_term("control")
    except Exception:
        return

    runtime = getattr(action_term, "_runtime", None)
    if runtime is None:
        return

    try:
        runtime.stop()
    except Exception as exc:
        print(f"[WARN]: Failed to stop autopilot runtime cleanly: {exc}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    env = None
    try:
        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # save resume path before creating a new log_dir
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = _resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            resume_path = _maybe_convert_scalar_std_checkpoint_to_log_std(resume_path)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        raw_env = env

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner from rsl-rl
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        # load the checkpoint
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # For ArduPilot-in-the-loop tasks, let SITL arm, take off, and settle immediately
        # before PPO starts. Doing this here avoids long post-warmup pauses while the runner
        # is still being built or checkpoints are loaded.
        _maybe_warmup_ardupilot_takeoff(raw_env, env_cfg)

        # run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    finally:
        _stop_ardupilot_runtime(env)
        if env is not None:
            env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
