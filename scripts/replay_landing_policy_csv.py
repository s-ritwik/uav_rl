#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_POLICY_PATH = REPO_ROOT / "source" / "uav_rl" / "uav_rl" / "transfer" / "policy.py"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


OBS_COLUMNS = (
    "rel_pos_x",
    "rel_pos_y",
    "rel_pos_z",
    "obs_rel_lin_vel_x",
    "obs_rel_lin_vel_y",
    "obs_rel_lin_vel_z",
    "obs_rel_quat_w",
    "obs_rel_quat_x",
    "obs_rel_quat_y",
    "obs_rel_quat_z",
    "obs_rel_ang_vel_x",
    "obs_rel_ang_vel_y",
    "obs_rel_ang_vel_z",
    "obs_projected_gravity_x",
    "obs_projected_gravity_y",
    "obs_projected_gravity_z",
    "obs_last_action_vx",
    "obs_last_action_vy",
    "obs_last_action_vz",
    "obs_last_action_yaw_rate",
)

REL_POS_COLUMNS = ("rel_pos_x", "rel_pos_y", "rel_pos_z")
LOGGED_ACTION_COLUMNS = ("action_vx", "action_vy", "action_vz", "action_yaw_rate")


@lru_cache(maxsize=1)
def _load_policy_module() -> Any:
    if not TRANSFER_POLICY_PATH.is_file():
        raise FileNotFoundError(f"Policy module not found: {TRANSFER_POLICY_PATH}")
    module_name = "uav_rl_transfer_policy_standalone"
    spec = importlib.util.spec_from_file_location(module_name, TRANSFER_POLICY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {TRANSFER_POLICY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compute_limits(values: np.ndarray, pad_fraction: float = 0.1) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * pad_fraction, 1.0e-3)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a landing_sway observation CSV and live-plot rel-pos against policy velocity output."
    )
    parser.add_argument("csv_path", type=Path, help="CSV file with logged observations/actions.")
    parser.add_argument(
        "--policy-jit",
        type=str,
        default=None,
        help="Path to an exported TorchScript policy. If omitted, a checkpoint source or logged actions must be used.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Direct path to an RSL-RL checkpoint file (`model_*.pt`).",
    )
    parser.add_argument(
        "--load-run",
        type=str,
        default=None,
        help="Run directory name under --log-root to resolve a checkpoint from.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Specific checkpoint filename inside --load-run. Defaults to the latest `model_*.pt`.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/rsl_rl/landing_sway"),
        help="Root directory used with --load-run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for policy inference.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Time scaling for replay. 1.0 = recorded speed, 2.0 = 2x faster.",
    )
    parser.add_argument(
        "--overlay-logged-actions",
        action="store_true",
        help="Plot the logged action_vx/vy/vz as dashed lines on the right column.",
    )
    return parser.parse_args()


def load_csv(csv_path: Path) -> dict[str, np.ndarray]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    missing = [name for name in ("t", "policy_step", *OBS_COLUMNS, *LOGGED_ACTION_COLUMNS) if name not in rows[0]]
    if missing:
        raise KeyError(f"CSV is missing required columns: {missing}")

    times = np.asarray([float(row["t"]) for row in rows], dtype=np.float32)
    policy_steps = np.asarray([int(float(row["policy_step"])) for row in rows], dtype=np.int64)
    observations = np.asarray([[float(row[name]) for name in OBS_COLUMNS] for row in rows], dtype=np.float32)
    rel_pos = np.asarray([[float(row[name]) for name in REL_POS_COLUMNS] for row in rows], dtype=np.float32)
    logged_actions = np.asarray([[float(row[name]) for name in LOGGED_ACTION_COLUMNS] for row in rows], dtype=np.float32)

    return {
        "times": times,
        "policy_steps": policy_steps,
        "observations": observations,
        "rel_pos": rel_pos,
        "logged_actions": logged_actions,
    }


def build_policy(args: argparse.Namespace) -> Any | None:
    if not (args.policy_jit or args.checkpoint_path or args.load_run):
        return None

    policy_module = _load_policy_module()
    RslRlPolicy = policy_module.RslRlPolicy
    resolve_checkpoint_path = policy_module.resolve_checkpoint_path

    if args.policy_jit:
        return RslRlPolicy(device=args.device, policy_jit=args.policy_jit)

    if args.checkpoint_path or args.load_run:
        checkpoint_path = resolve_checkpoint_path(
            load_run=args.load_run,
            checkpoint_name=args.checkpoint_name,
            checkpoint_path=args.checkpoint_path,
            log_root=args.log_root,
        )
        return RslRlPolicy(device=args.device, checkpoint_path=str(checkpoint_path))

    return None


def infer_policy_actions(policy: Any, observations: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []
    for start in range(0, observations.shape[0], chunk_size):
        obs_batch = torch.from_numpy(observations[start : start + chunk_size])
        act_batch = policy.act(obs_batch).detach().cpu().numpy().astype(np.float32)
        outputs.append(act_batch)
    return np.concatenate(outputs, axis=0)


def replay_plot(
    *,
    times: np.ndarray,
    policy_steps: np.ndarray,
    rel_pos: np.ndarray,
    action_xyz: np.ndarray,
    logged_action_xyz: np.ndarray | None,
    playback_speed: float,
    action_source_label: str,
) -> None:
    import matplotlib.pyplot as plt

    if playback_speed <= 0.0:
        raise ValueError(f"playback-speed must be > 0, got {playback_speed}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig.canvas.manager.set_window_title("Landing Sway CSV Replay")

    pos_labels = ("rel_pos_x [m]", "rel_pos_y [m]", "rel_pos_z [m]")
    act_labels = ("vx [m/s]", "vy [m/s]", "vz [m/s]")
    pos_titles = ("Rel Pos X", "Rel Pos Y", "Rel Pos Z")
    act_titles = ("Policy Vx", "Policy Vy", "Policy Vz")

    pos_lines = []
    act_lines = []
    logged_lines = []

    for row in range(3):
        left_ax = axes[row, 0]
        right_ax = axes[row, 1]

        (pos_line,) = left_ax.plot([], [], color="tab:blue", linewidth=2.0)
        (act_line,) = right_ax.plot([], [], color="tab:red", linewidth=2.0, label=action_source_label)
        pos_lines.append(pos_line)
        act_lines.append(act_line)

        left_ax.set_ylabel(pos_labels[row])
        right_ax.set_ylabel(act_labels[row])
        left_ax.set_title(pos_titles[row])
        right_ax.set_title(act_titles[row])

        left_ax.set_ylim(*_compute_limits(rel_pos[:, row]))
        right_ax.set_ylim(*_compute_limits(action_xyz[:, row] if logged_action_xyz is None else np.concatenate((action_xyz[:, row], logged_action_xyz[:, row]))))

        if logged_action_xyz is not None:
            (logged_line,) = right_ax.plot(
                [],
                [],
                color="0.4",
                linewidth=1.5,
                linestyle="--",
                label="logged_action",
            )
            logged_lines.append(logged_line)
            right_ax.legend(loc="upper right")
        else:
            logged_lines.append(None)

    t_min = float(times[0])
    t_max = float(times[-1])
    for column in range(2):
        axes[-1, column].set_xlabel("t [s]")
        for row in range(3):
            axes[row, column].set_xlim(t_min, t_max)
            axes[row, column].grid(True, alpha=0.3)

    header = fig.suptitle("", fontsize=13)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show(block=False)

    for index in range(times.shape[0]):
        if not plt.fignum_exists(fig.number):
            break

        upto = index + 1
        current_t = float(times[index])
        current_step = int(policy_steps[index])

        for row in range(3):
            pos_lines[row].set_data(times[:upto], rel_pos[:upto, row])
            act_lines[row].set_data(times[:upto], action_xyz[:upto, row])
            if logged_action_xyz is not None and logged_lines[row] is not None:
                logged_lines[row].set_data(times[:upto], logged_action_xyz[:upto, row])

        header.set_text(f"Landing Sway Replay | t={current_t:.2f}s | policy_step={current_step}")
        fig.canvas.draw_idle()

        if index == 0:
            plt.pause(0.001)
            continue

        dt = max(float(times[index] - times[index - 1]), 0.0)
        plt.pause(max(dt / playback_speed, 0.001))

    if plt.fignum_exists(fig.number):
        plt.show()


def main() -> None:
    args = parse_args()
    data = load_csv(args.csv_path.expanduser().resolve())
    policy = build_policy(args)

    if policy is None:
        action_matrix = data["logged_actions"]
        action_source = "logged_action"
        print("No policy source provided; replaying logged action_vx/vy/vz from the CSV.")
    else:
        action_matrix = infer_policy_actions(policy, data["observations"])
        action_source = "policy_output"
        print(
            f"Loaded policy on {args.device}; replaying {data['observations'].shape[0]} observations "
            f"from {args.csv_path.expanduser().resolve()}"
        )

    logged_xyz = data["logged_actions"][:, :3] if args.overlay_logged_actions else None
    replay_plot(
        times=data["times"],
        policy_steps=data["policy_steps"],
        rel_pos=data["rel_pos"],
        action_xyz=action_matrix[:, :3],
        logged_action_xyz=logged_xyz,
        playback_speed=args.playback_speed,
        action_source_label=action_source,
    )


if __name__ == "__main__":
    main()
