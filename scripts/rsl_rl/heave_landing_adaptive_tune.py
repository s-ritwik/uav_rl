#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROTECTED_OVERRIDES: dict[str, object] = {
    "env.post_init_cfg.episode.timeout_s": 40.0,
    "env.post_init_cfg.touchdown.force_threshold_n": 2.0,
    "env.post_init_cfg.touchdown.max_touchdown_speed_mps": 0.3,
    "env.post_init_cfg.touchdown.max_xy_error_m": 0.4,
    "env.post_init_cfg.touchdown.require_xy_within_box": True,
    "env.post_init_cfg.touchdown.require_attitude_within_limits": True,
    "env.post_init_cfg.touchdown.max_touchdown_roll_deg": 12.0,
    "env.post_init_cfg.touchdown.max_touchdown_pitch_deg": 12.0,
}


BASE_REWARD_OVERRIDES: dict[str, object] = {
    "env.post_init_cfg.reward_weights.alive": 0.2,
    "env.post_init_cfg.reward_weights.terminated": 50.0,
    "env.post_init_cfg.reward_weights.touchdown_terminated": 50.0,
    "env.post_init_cfg.reward_weights.position_track": 10.0,
    "env.post_init_cfg.reward_weights.vertical_position": 0.0,
    "env.post_init_cfg.reward_weights.vertical_clearance_excess": -0.5,
    "env.post_init_cfg.reward_weights.horizontal_speed": -0.08,
    "env.post_init_cfg.reward_weights.vertical_speed": -0.01,
    "env.post_init_cfg.reward_weights.action_magnitude_x": -2.5,
    "env.post_init_cfg.reward_weights.action_magnitude_y": -2.5,
    "env.post_init_cfg.reward_weights.action_magnitude_z": -2.5,
    "env.post_init_cfg.reward_weights.action_magnitude_yaw_rate": -1.2,
    "env.post_init_cfg.reward_weights.velocity_action_rate_x": -50.0,
    "env.post_init_cfg.reward_weights.velocity_action_rate_y": -50.0,
    "env.post_init_cfg.reward_weights.velocity_action_rate_z": -50.0,
    "env.post_init_cfg.reward_weights.uav_acceleration": -0.5,
    "env.post_init_cfg.reward_weights.angular_rate": -0.5,
    "env.post_init_cfg.reward_weights.angular_velocity_rate": -50.0,
    "env.post_init_cfg.reward_weights.angular_rate_xy": -1.0,
    "env.post_init_cfg.reward_weights.yaw_rate_error": -10.0,
    "env.post_init_cfg.reward_weights.yaw_error": -2.0,
    "env.post_init_cfg.reward_weights.upright": -20.0,
    "env.post_init_cfg.reward_weights.touchdown_quality": 1.0,
    "env.post_init_cfg.reward_weights.horizontal_velocity_match": 2.0,
    "env.post_init_cfg.reward_weights.near_target_action_xy": 0.0,
    "env.post_init_cfg.position_track.std_m": 2.0,
    "env.post_init_cfg.near_target_action.std_m": 0.6,
    "env.post_init_cfg.vertical_clearance.threshold_m": 1.0,
}


METRIC_KEYS = [
    "Train/mean_reward",
    "Episode_Reward/touchdown_quality",
    "Episode_Termination/touchdown",
    "Episode_Termination/time_out",
    "Episode_Termination/attitude_tilt",
    "Episode_Termination/out_of_bounds",
    "Episode_Reward/position_track",
    "Episode_Reward/horizontal_velocity_match",
    "Episode_Reward/vertical_speed",
    "Episode_Reward/action_magnitude_z",
    "Episode_Reward/velocity_action_rate_z",
    "Episode_Reward/uav_acceleration",
    "Episode_Reward/angular_rate_xy",
    "Episode_Reward/upright",
]


@dataclass
class Candidate:
    name: str
    overrides: dict[str, object] = field(default_factory=dict)
    rationale: str = ""


def _tail_mean(event_path: Path, key: str, n: int = 50) -> float | None:
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    if key not in ea.Tags().get("scalars", []):
        return None
    scalars = ea.Scalars(key)
    if not scalars:
        return None
    tail = scalars[-min(n, len(scalars)) :]
    return mean(item.value for item in tail)


def summarize_run(run_dir: Path) -> dict[str, float | str | int]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    event_path = event_files[-1]
    summary: dict[str, float | str | int] = {
        "run_dir": str(run_dir),
        "event_file": str(event_path),
    }
    for key in METRIC_KEYS:
        summary[key] = _tail_mean(event_path, key)

    touchdown = float(summary["Episode_Termination/touchdown"] or 0.0)
    timeout = float(summary["Episode_Termination/time_out"] or 0.0)
    quality = float(summary["Episode_Reward/touchdown_quality"] or 0.0)
    reward = float(summary["Train/mean_reward"] or 0.0)
    pos = float(summary["Episode_Reward/position_track"] or 0.0)
    vel = float(summary["Episode_Reward/horizontal_velocity_match"] or 0.0)
    z_rate = float(summary["Episode_Reward/velocity_action_rate_z"] or 0.0)
    z_mag = float(summary["Episode_Reward/action_magnitude_z"] or 0.0)
    upright = float(summary["Episode_Reward/upright"] or 0.0)
    att_tilt = float(summary["Episode_Termination/attitude_tilt"] or 0.0)

    summary["composite_score"] = (
        2500.0 * touchdown
        - 1200.0 * timeout
        - 600.0 * att_tilt
        + 35.0 * quality
        + 1.0 * reward
        + 70.0 * pos
        + 120.0 * vel
        + 80.0 * z_rate
        + 50.0 * z_mag
        + 60.0 * upright
    )
    return summary


def _sanitize_tag(name: str) -> str:
    return (
        name.replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def find_latest_run(task: str, run_name: str) -> Path:
    root = Path("logs/rsl_rl") / task
    matches = sorted(root.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No run directory found for {run_name} under {root}")
    return matches[-1]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mutate(base: dict[str, object], **updates: float | int | bool | str) -> dict[str, object]:
    out = dict(base)
    out.update(updates)
    return out


def candidate_from_summary(index: int, best: dict[str, float | str | int], previous: dict[str, float | str | int]) -> Candidate:
    best_touchdown = float(best.get("Episode_Termination/touchdown") or 0.0)
    best_quality = float(best.get("Episode_Reward/touchdown_quality") or 0.0)
    best_timeout = float(best.get("Episode_Termination/time_out") or 0.0)

    base = dict(BASE_REWARD_OVERRIDES)
    # Seed exploration from best-so-far if present.
    for key in BASE_REWARD_OVERRIDES:
        if key in best:
            continue

    # Hand-tuned sequence for the first few runs to test the heave-specific hypotheses.
    if index == 0:
        return Candidate(
            name="r01_baseline_40s",
            overrides=base,
            rationale="Baseline current observations with 40s timeout and protected touchdown thresholds fixed.",
        )
    if index == 1:
        return Candidate(
            name="r02_relax_z_follow_heave",
            overrides=mutate(
                base,
                **{
                    "env.post_init_cfg.reward_weights.vertical_speed": 0.0,
                    "env.post_init_cfg.reward_weights.action_magnitude_z": -1.0,
                    "env.post_init_cfg.reward_weights.velocity_action_rate_z": -10.0,
                    "env.post_init_cfg.reward_weights.uav_acceleration": -0.2,
                    "env.post_init_cfg.reward_weights.angular_velocity_rate": -20.0,
                },
            ),
            rationale="Remove absolute-z-speed bias and relax Z command continuity so the agent can match heave motion.",
        )
    if index == 2:
        return Candidate(
            name="r03_relax_z_plus_finish",
            overrides=mutate(
                base,
                **{
                    "env.post_init_cfg.reward_weights.vertical_speed": 0.0,
                    "env.post_init_cfg.reward_weights.action_magnitude_z": -1.0,
                    "env.post_init_cfg.reward_weights.velocity_action_rate_z": -10.0,
                    "env.post_init_cfg.reward_weights.uav_acceleration": -0.2,
                    "env.post_init_cfg.reward_weights.angular_velocity_rate": -20.0,
                    "env.post_init_cfg.reward_weights.touchdown_terminated": 75.0,
                    "env.post_init_cfg.reward_weights.vertical_clearance_excess": -1.0,
                },
            ),
            rationale="Same heave-tracking relaxation, plus stronger finish pressure to reduce timeout behavior.",
        )
    if index == 3:
        return Candidate(
            name="r04_relax_z_finish_soft",
            overrides=mutate(
                base,
                **{
                    "env.post_init_cfg.reward_weights.vertical_speed": 0.0,
                    "env.post_init_cfg.reward_weights.action_magnitude_z": -1.0,
                    "env.post_init_cfg.reward_weights.velocity_action_rate_z": -10.0,
                    "env.post_init_cfg.reward_weights.uav_acceleration": -0.2,
                    "env.post_init_cfg.reward_weights.angular_velocity_rate": -20.0,
                    "env.post_init_cfg.reward_weights.touchdown_terminated": 75.0,
                    "env.post_init_cfg.reward_weights.vertical_clearance_excess": -1.0,
                    "env.post_init_cfg.reward_weights.touchdown_quality": 2.0,
                    "env.post_init_cfg.reward_weights.angular_rate_xy": -2.0,
                    "env.post_init_cfg.reward_weights.upright": -30.0,
                },
            ),
            rationale="Push touchdown quality and attitude once heave matching is less constrained.",
        )

    # Adaptive phase: branch from best observed behavior.
    overrides = dict(base)
    rationale = []

    if best_touchdown < 0.5 or best_timeout > 0.4:
        # Too conservative or not finishing. Permit more Z motion and push landing harder.
        overrides.update(
            {
                "env.post_init_cfg.reward_weights.vertical_speed": 0.0,
                "env.post_init_cfg.reward_weights.action_magnitude_z": -0.5,
                "env.post_init_cfg.reward_weights.velocity_action_rate_z": -5.0,
                "env.post_init_cfg.reward_weights.uav_acceleration": 0.0,
                "env.post_init_cfg.reward_weights.angular_velocity_rate": -10.0,
                "env.post_init_cfg.reward_weights.touchdown_terminated": clamp(75.0 + 10.0 * (index - 3), 75.0, 120.0),
                "env.post_init_cfg.reward_weights.vertical_clearance_excess": clamp(-1.0 - 0.25 * (index - 3), -2.5, -1.0),
            }
        )
        rationale.append("Best run is not finishing enough; relax Z penalties further and increase landing pressure.")
    elif best_touchdown < 0.9:
        overrides.update(
            {
                "env.post_init_cfg.reward_weights.vertical_speed": 0.0,
                "env.post_init_cfg.reward_weights.action_magnitude_z": -1.0,
                "env.post_init_cfg.reward_weights.velocity_action_rate_z": -10.0,
                "env.post_init_cfg.reward_weights.uav_acceleration": -0.1,
                "env.post_init_cfg.reward_weights.touchdown_terminated": 90.0,
                "env.post_init_cfg.reward_weights.vertical_clearance_excess": -1.5,
                "env.post_init_cfg.reward_weights.touchdown_quality": 1.5,
            }
        )
        rationale.append("Touchdown improving but still below 90%; keep Z relaxed and raise landing incentives.")
    elif best_quality < 10.0:
        overrides.update(
            {
                "env.post_init_cfg.reward_weights.vertical_speed": -0.005,
                "env.post_init_cfg.reward_weights.action_magnitude_z": -1.5,
                "env.post_init_cfg.reward_weights.velocity_action_rate_z": -15.0,
                "env.post_init_cfg.reward_weights.uav_acceleration": -0.2,
                "env.post_init_cfg.reward_weights.angular_velocity_rate": -30.0,
                "env.post_init_cfg.reward_weights.angular_rate_xy": -2.0,
                "env.post_init_cfg.reward_weights.upright": -35.0,
                "env.post_init_cfg.reward_weights.touchdown_quality": 2.5,
            }
        )
        rationale.append("Touchdown is landing but quality is weak; strengthen soft-touch and attitude shaping.")
    else:
        # Local refinement around a good landing regime.
        z_mag = -1.0 - 0.25 * ((index - 4) % 3)
        z_rate = -10.0 - 5.0 * ((index - 4) % 3)
        td_quality = 2.0 + 0.5 * ((index - 4) % 4)
        clearance = -1.0 - 0.25 * ((index - 4) % 4)
        overrides.update(
            {
                "env.post_init_cfg.reward_weights.vertical_speed": -0.0025,
                "env.post_init_cfg.reward_weights.action_magnitude_z": z_mag,
                "env.post_init_cfg.reward_weights.velocity_action_rate_z": z_rate,
                "env.post_init_cfg.reward_weights.uav_acceleration": -0.15,
                "env.post_init_cfg.reward_weights.angular_velocity_rate": -25.0,
                "env.post_init_cfg.reward_weights.angular_rate_xy": -1.5,
                "env.post_init_cfg.reward_weights.upright": -30.0,
                "env.post_init_cfg.reward_weights.touchdown_quality": td_quality,
                "env.post_init_cfg.reward_weights.vertical_clearance_excess": clearance,
                "env.post_init_cfg.reward_weights.touchdown_terminated": 80.0,
            }
        )
        rationale.append("Refining around a stable touchdown regime with small Z smoothness/quality perturbations.")

    # Alternate one secondary knob per round so the search does not collapse to a single axis.
    phase = (index - 4) % 4
    if phase == 0:
        overrides["env.post_init_cfg.reward_weights.position_track"] = 12.0
        rationale.append("Slightly stronger XY centering.")
    elif phase == 1:
        overrides["env.post_init_cfg.reward_weights.horizontal_velocity_match"] = 3.0
        rationale.append("Slightly stronger XY zero-velocity shaping.")
    elif phase == 2:
        overrides["env.post_init_cfg.reward_weights.yaw_rate_error"] = -6.0
        overrides["env.post_init_cfg.reward_weights.yaw_error"] = -1.0
        rationale.append("Relax yaw shaping so Z synchronization dominates.")
    else:
        overrides["env.post_init_cfg.reward_weights.horizontal_speed"] = -0.12
        rationale.append("More XY damping while preserving heave-following focus.")

    return Candidate(
        name=f"r{index + 1:02d}_{_sanitize_tag('_'.join(rationale[:2])[:48])}",
        overrides=overrides,
        rationale=" ".join(rationale),
    )


def command_for_candidate(args: argparse.Namespace, cand: Candidate) -> list[str]:
    run_name = f"{args.run_prefix}_{cand.name}"
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

    merged = dict(PROTECTED_OVERRIDES)
    merged.update(cand.overrides)
    for key, value in merged.items():
        cmd.append(f"{key}={value}")
    return cmd


def write_results(csv_path: Path, json_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="heave_landing")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--run-prefix", default="heave_landing_adapt")
    ap.add_argument("--results-csv", default="logs/rsl_rl/heave_landing/adaptive_tune_results.csv")
    ap.add_argument("--results-json", default="logs/rsl_rl/heave_landing/adaptive_tune_results.json")
    ap.add_argument("--resume-run-dir", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results_csv = Path(args.results_csv)
    results_json = Path(args.results_json)
    json_rows: list[dict[str, object]] = []

    best_summary: dict[str, float | str | int] | None = None
    previous_summary: dict[str, float | str | int] = {}

    base_fieldnames = [
        "index",
        "candidate",
        "run_name",
        "rationale",
        "run_dir",
        "event_file",
        "composite_score",
    ] + METRIC_KEYS + sorted(PROTECTED_OVERRIDES.keys()) + sorted(BASE_REWARD_OVERRIDES.keys())

    for resume_index, run_dir_str in enumerate(args.resume_run_dir):
        run_dir = Path(run_dir_str)
        summary = summarize_run(run_dir)
        summary["index"] = resume_index
        run_name = run_dir.name
        run_name_pos = run_name.find(args.run_prefix)
        if run_name_pos >= 0:
            run_name = run_name[run_name_pos:]
        summary["run_name"] = run_name
        if run_name.startswith(f"{args.run_prefix}_"):
            summary["candidate"] = run_name[len(args.run_prefix) + 1 :]
        else:
            summary["candidate"] = run_name
        summary["rationale"] = "resumed existing run"
        json_rows.append(summary)
        if best_summary is None or float(summary["composite_score"]) > float(best_summary["composite_score"]):
            best_summary = summary
        previous_summary = summary

    if json_rows:
        write_results(results_csv, results_json, json_rows, base_fieldnames)

    for index in range(len(json_rows), args.runs):
        candidate = candidate_from_summary(index, best_summary or {}, previous_summary)
        run_name = f"{args.run_prefix}_{candidate.name}"
        cmd = command_for_candidate(args, candidate)

        print(f"=== Heave Candidate {index + 1}/{args.runs}: {candidate.name} ===", flush=True)
        print(candidate.rationale, flush=True)
        print(" ".join(cmd), flush=True)

        if args.dry_run:
            continue

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("MPLCONFIGDIR", str(Path("logs/.mplconfig").resolve()))
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        subprocess.run(cmd, check=True, env=env)

        run_dir = find_latest_run(args.task, run_name)
        summary = summarize_run(run_dir)
        summary["index"] = index
        summary["candidate"] = candidate.name
        summary["run_name"] = run_name
        summary["rationale"] = candidate.rationale

        merged = dict(PROTECTED_OVERRIDES)
        merged.update(candidate.overrides)
        for key, value in merged.items():
            summary[key] = value

        json_rows.append(summary)
        write_results(results_csv, results_json, json_rows, base_fieldnames)

        if best_summary is None or float(summary["composite_score"]) > float(best_summary["composite_score"]):
            best_summary = summary
        previous_summary = summary

        print(
            f"Completed {candidate.name}: touchdown={summary['Episode_Termination/touchdown']}, "
            f"quality={summary['Episode_Reward/touchdown_quality']}, reward={summary['Train/mean_reward']}, "
            f"score={summary['composite_score']}",
            flush=True,
        )

    if best_summary is not None:
        print("\n=== Best Heave Candidate ===", flush=True)
        print(json.dumps(best_summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
