# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys
from typing import Sequence

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug_actions",
    action="store_true",
    default=False,
    help="Print policy actions and velocity setpoints during play.",
)
parser.add_argument(
    "--debug_action_interval",
    type=int,
    default=50,
    help="Number of simulation steps between debug action prints.",
)
parser.add_argument(
    "--termination_stats",
    action="store_true",
    default=False,
    help="Print termination-cause percentages during deterministic play.",
)
parser.add_argument(
    "--termination_stats_interval",
    type=int,
    default=250,
    help="Number of play steps between termination-stat prints.",
)
parser.add_argument(
    "--log_touchdown_velocities",
    action="store_true",
    default=False,
    help="Log touchdown descent velocities to CSV during play only.",
)
parser.add_argument(
    "--touchdown_velocity_log_file",
    type=str,
    default=None,
    help="Optional CSV output path for touchdown-velocity samples.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import uav_rl.tasks  # noqa: F401
from uav_rl.tasks.manager_based.heave_landing.agents.custom_exporter import (
    export_custom_policy_as_jit,
    export_custom_policy_as_onnx,
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    touchdown_log_file = None
    touchdown_log_handle = None
    touchdown_log_writer = None
    touchdown_log_count = 0
    touchdown_log_warned = False

    if args_cli.log_touchdown_velocities:
        touchdown_log_file = args_cli.touchdown_velocity_log_file
        if touchdown_log_file is None:
            touchdown_log_file = os.path.join(
                log_dir,
                "analysis",
                f"touchdown_velocities_play_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            )
        os.makedirs(os.path.dirname(touchdown_log_file), exist_ok=True)
        touchdown_log_handle = open(touchdown_log_file, "w", newline="", encoding="utf-8")
        touchdown_log_writer = csv.writer(touchdown_log_handle)
        touchdown_log_writer.writerow(
            [
                "play_step",
                "env_id",
                "touchdown_speed_for_reward_mps",
                "touchdown_force_n",
                "touchdown_xy_error_m",
                "touchdown_roll_rad",
                "touchdown_pitch_rad",
                "touchdown_yaw_rad",
            ]
        )
        print(f"[INFO] Logging touchdown velocities to: {touchdown_log_file}")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.log_touchdown_velocities:
        orig_reset_idx = env.unwrapped._reset_idx

        def _reset_idx_with_touchdown_logging(env_ids: Sequence[int]):
            nonlocal touchdown_log_count, touchdown_log_warned
            try:
                env_ids_tensor = torch.as_tensor(env_ids, device=env.unwrapped.device, dtype=torch.long)
                term_manager = getattr(env.unwrapped, "termination_manager", None)
                touchdown_mask = None
                if term_manager is not None:
                    touchdown_mask = term_manager.get_term("touchdown").to(dtype=torch.bool)

                pre_rel_vz = getattr(env.unwrapped, "_landing_touchdown_pre_rel_vz", None)
                if touchdown_mask is not None and pre_rel_vz is not None and env_ids_tensor.numel() > 0:
                    touchdown_env_ids = env_ids_tensor[touchdown_mask[env_ids_tensor]]
                    if touchdown_env_ids.numel() > 0:
                        touchdown_force = getattr(env.unwrapped, "_landing_touchdown_force_norm", None)
                        touchdown_xy_error = getattr(env.unwrapped, "_landing_touchdown_xy_error", None)
                        touchdown_roll = getattr(env.unwrapped, "_landing_touchdown_roll", None)
                        touchdown_pitch = getattr(env.unwrapped, "_landing_touchdown_pitch", None)
                        touchdown_yaw = getattr(env.unwrapped, "_landing_touchdown_yaw", None)
                        descent_speed = (-pre_rel_vz[touchdown_env_ids]).clamp_min(0.0)

                        for row_idx, env_id in enumerate(touchdown_env_ids.detach().cpu().tolist()):
                            idx = int(env_id)
                            touchdown_log_writer.writerow(
                                [
                                    int(env.unwrapped.common_step_counter),
                                    idx,
                                    float(descent_speed[row_idx].item()),
                                    float(touchdown_force[idx].item()) if touchdown_force is not None else float("nan"),
                                    float(touchdown_xy_error[idx].item()) if touchdown_xy_error is not None else float("nan"),
                                    float(touchdown_roll[idx].item()) if touchdown_roll is not None else float("nan"),
                                    float(touchdown_pitch[idx].item()) if touchdown_pitch is not None else float("nan"),
                                    float(touchdown_yaw[idx].item()) if touchdown_yaw is not None else float("nan"),
                                ]
                            )
                            touchdown_log_count += 1
                        touchdown_log_handle.flush()
            except Exception as exc:
                if not touchdown_log_warned:
                    print(f"[WARN] Touchdown logging hook failed: {exc}")
                    touchdown_log_warned = True
            return orig_reset_idx(env_ids)

        env.unwrapped._reset_idx = _reset_idx_with_touchdown_logging

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if policy_nn.__class__.__name__ == "ActorCriticSeparateRecurrent":
        export_custom_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_custom_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    debug_term_missing_warned = False
    termination_counts: dict[str, int] = {}
    termination_total = 0

    def update_termination_stats() -> None:
        nonlocal termination_total
        term_manager = getattr(env.unwrapped, "termination_manager", None)
        if term_manager is None:
            return
        for term_name in term_manager.active_terms:
            term = term_manager.get_term(term_name)
            count = int(term.to(dtype=torch.bool).sum().item())
            if count > 0:
                termination_counts[term_name] = termination_counts.get(term_name, 0) + count
                termination_total += count

    def print_termination_stats(prefix: str) -> None:
        if termination_total <= 0:
            print(f"{prefix} termination_stats: no terminations yet")
            return
        parts = []
        for term_name, count in sorted(termination_counts.items()):
            percent = 100.0 * count / termination_total
            parts.append(f"{term_name}={count} ({percent:.2f}%)")
        print(f"{prefix} termination_stats total={termination_total}: " + ", ".join(parts))

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if args_cli.termination_stats:
                update_termination_stats()
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        timestep += 1
        if (
            args_cli.termination_stats
            and timestep % max(1, args_cli.termination_stats_interval) == 0
        ):
            print_termination_stats(f"[TERMINATION_STATS step={timestep}]")

        if args_cli.debug_actions and timestep % max(1, args_cli.debug_action_interval) == 0:
            try:
                control_term = env.unwrapped.action_manager.get_term("control")
                raw = control_term.raw_actions[0].detach().cpu().tolist()
                setpoint = control_term.processed_actions[0].detach().cpu().tolist()
                print(
                    f"[DEBUG_ACTIONS step={timestep}] "
                    f"raw={raw} vel_sp={setpoint[:3]} yaw_rate_sp={setpoint[3]:.4f}"
                )
                if hasattr(control_term, "last_motor_omega"):
                    motor_omega = control_term.last_motor_omega[0].detach().cpu().tolist()
                    print(f"[DEBUG_ACTIONS step={timestep}] motor_omega={motor_omega}")
            except Exception as exc:
                if not debug_term_missing_warned:
                    print(f"[WARN] --debug_actions enabled but could not read action term 'control': {exc}")
                    debug_term_missing_warned = True

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.termination_stats:
        print_termination_stats("[TERMINATION_STATS final]")
    if touchdown_log_handle is not None:
        touchdown_log_handle.close()
        print(f"[INFO] Logged {touchdown_log_count} touchdown samples to: {touchdown_log_file}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
