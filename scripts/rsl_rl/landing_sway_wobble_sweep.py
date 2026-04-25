#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class Candidate:
    name: str
    overrides: dict[str, object]


CONTROL_OVERRIDES = {
    "env.events.move_platform.params.stationary_env_probability": 0.2,
    "env.rewards.position_track.weight": 10.0,
    "env.rewards.touchdown_terminated.weight": 50.0,
    "env.rewards.vertical_clearance_excess.weight": -0.5,
    "env.rewards.near_target_action_xy.weight": 0.0,
}


def build_candidates() -> list[Candidate]:
    return [
        Candidate(
            "w01_mag2p0_rate40",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -2.0,
                "env.rewards.action_magnitude_y.weight": -2.0,
                "env.rewards.velocity_action_rate_x.weight": -40.0,
                "env.rewards.velocity_action_rate_y.weight": -40.0,
            },
        ),
        Candidate(
            "w02_mag2p5_rate40",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -2.5,
                "env.rewards.action_magnitude_y.weight": -2.5,
                "env.rewards.velocity_action_rate_x.weight": -40.0,
                "env.rewards.velocity_action_rate_y.weight": -40.0,
            },
        ),
        Candidate(
            "w03_mag2p5_rate50",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -2.5,
                "env.rewards.action_magnitude_y.weight": -2.5,
                "env.rewards.velocity_action_rate_x.weight": -50.0,
                "env.rewards.velocity_action_rate_y.weight": -50.0,
            },
        ),
        Candidate(
            "w04_mag3p0_rate50",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -3.0,
                "env.rewards.action_magnitude_y.weight": -3.0,
                "env.rewards.velocity_action_rate_x.weight": -50.0,
                "env.rewards.velocity_action_rate_y.weight": -50.0,
            },
        ),
        Candidate(
            "w05_mag3p0_rate60",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -3.0,
                "env.rewards.action_magnitude_y.weight": -3.0,
                "env.rewards.velocity_action_rate_x.weight": -60.0,
                "env.rewards.velocity_action_rate_y.weight": -60.0,
            },
        ),
        Candidate(
            "w06_mag2p5_rate50_near1",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -2.5,
                "env.rewards.action_magnitude_y.weight": -2.5,
                "env.rewards.velocity_action_rate_x.weight": -50.0,
                "env.rewards.velocity_action_rate_y.weight": -50.0,
                "env.rewards.near_target_action_xy.weight": -1.0,
            },
        ),
        Candidate(
            "w07_mag2p5_rate50_near2",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -2.5,
                "env.rewards.action_magnitude_y.weight": -2.5,
                "env.rewards.velocity_action_rate_x.weight": -50.0,
                "env.rewards.velocity_action_rate_y.weight": -50.0,
                "env.rewards.near_target_action_xy.weight": -2.0,
            },
        ),
        Candidate(
            "w08_mag3p0_rate60_near1_upright",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -3.0,
                "env.rewards.action_magnitude_y.weight": -3.0,
                "env.rewards.velocity_action_rate_x.weight": -60.0,
                "env.rewards.velocity_action_rate_y.weight": -60.0,
                "env.rewards.near_target_action_xy.weight": -1.0,
                "env.rewards.angular_rate_xy.weight": -1.5,
                "env.rewards.upright.weight": -25.0,
            },
        ),
        Candidate(
            "w09_mag2p0_rate50_near1",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -2.0,
                "env.rewards.action_magnitude_y.weight": -2.0,
                "env.rewards.velocity_action_rate_x.weight": -50.0,
                "env.rewards.velocity_action_rate_y.weight": -50.0,
                "env.rewards.near_target_action_xy.weight": -1.0,
            },
        ),
        Candidate(
            "w10_mag2p5_rate60_near1",
            {
                **CONTROL_OVERRIDES,
                "env.rewards.action_magnitude_x.weight": -2.5,
                "env.rewards.action_magnitude_y.weight": -2.5,
                "env.rewards.velocity_action_rate_x.weight": -60.0,
                "env.rewards.velocity_action_rate_y.weight": -60.0,
                "env.rewards.near_target_action_xy.weight": -1.0,
            },
        ),
    ]


METRIC_KEYS = [
    "Train/mean_reward",
    "Episode_Reward/touchdown_quality",
    "Episode_Termination/touchdown",
    "Episode_Termination/time_out",
    "Episode_Reward/position_track",
    "Episode_Reward/horizontal_velocity_match",
    "Episode_Reward/action_magnitude_x",
    "Episode_Reward/action_magnitude_y",
    "Episode_Reward/velocity_action_rate_x",
    "Episode_Reward/velocity_action_rate_y",
    "Episode_Reward/upright",
    "Episode_Reward/angular_rate_xy",
]


def scalar_tail_mean(event_path: Path, key: str, n: int = 50) -> float | None:
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    if key not in ea.Tags().get("scalars", []):
        return None
    vals = ea.Scalars(key)
    if not vals:
        return None
    tail = vals[-min(n, len(vals)) :]
    return mean(v.value for v in tail)


def summarize_run(run_dir: Path) -> dict[str, float | str]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event file in {run_dir}")
    event_path = event_files[-1]
    out: dict[str, float | str] = {"run_dir": str(run_dir), "event_file": str(event_path)}
    for key in METRIC_KEYS:
        out[key] = scalar_tail_mean(event_path, key)
    touchdown = out["Episode_Termination/touchdown"] or 0.0
    quality = out["Episode_Reward/touchdown_quality"] or 0.0
    mean_reward = out["Train/mean_reward"] or 0.0
    pos = out["Episode_Reward/position_track"] or 0.0
    vel = out["Episode_Reward/horizontal_velocity_match"] or 0.0
    amagx = out["Episode_Reward/action_magnitude_x"] or 0.0
    amagy = out["Episode_Reward/action_magnitude_y"] or 0.0
    ratex = out["Episode_Reward/velocity_action_rate_x"] or 0.0
    ratey = out["Episode_Reward/velocity_action_rate_y"] or 0.0
    upright = out["Episode_Reward/upright"] or 0.0
    angrxy = out["Episode_Reward/angular_rate_xy"] or 0.0
    # Higher is better. Negative action magnitude/rate rewards become larger as they approach zero.
    out["composite_score"] = (
        2000.0 * touchdown
        + 30.0 * quality
        + 1.0 * mean_reward
        + 80.0 * pos
        + 120.0 * vel
        + 120.0 * ratex
        + 120.0 * ratey
        + 60.0 * amagx
        + 60.0 * amagy
        + 80.0 * upright
        + 80.0 * angrxy
    )
    return out


def find_latest_run(prefix: str) -> Path:
    root = Path("logs/rsl_rl/landing_sway")
    matches = sorted(root.glob(f"*_{prefix}"))
    if not matches:
        raise FileNotFoundError(f"No run dir for prefix {prefix}")
    return matches[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="landing_sway")
    ap.add_argument("--num-envs", type=int, default=4098)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--count", type=int, default=len(build_candidates()))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-csv", default="logs/rsl_rl/landing_sway/wobble_sweep_results.csv")
    args = ap.parse_args()

    candidates = build_candidates()[args.start_index : args.start_index + args.count]
    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, cand in enumerate(candidates, start=args.start_index):
        run_name = f"landing_sway_2.8.4_{cand.name}"
        cmd = [
            sys.executable,
            "scripts/rsl_rl/train.py",
            "--task",
            args.task,
            "--headless",
            "--device",
            args.device,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.max_iterations),
            "--seed",
            str(args.seed),
            "--run_name",
            run_name,
        ]
        for key, value in cand.overrides.items():
            cmd.append(f"{key}={value}")
        print(f"=== Candidate {idx}: {cand.name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)
        run_dir = find_latest_run(run_name)
        summary = summarize_run(run_dir)
        summary["candidate"] = cand.name
        summary["index"] = idx
        summary["seed"] = args.seed
        summary["num_envs"] = args.num_envs
        summary["max_iterations"] = args.max_iterations
        summary["overrides"] = json.dumps(cand.overrides, sort_keys=True)
        write_header = not results_path.exists()
        with results_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
