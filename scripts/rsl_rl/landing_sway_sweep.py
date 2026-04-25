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


METRIC_KEYS = [
    'Episode_Termination/touchdown',
    'Episode_Reward/touchdown_quality',
    'Episode_Reward/position_track',
    'Episode_Reward/horizontal_velocity_match',
    'Episode_Reward/velocity_action_rate_x',
    'Episode_Reward/velocity_action_rate_y',
    'Episode_Reward/velocity_action_rate_z',
    'Episode_Reward/action_magnitude_x',
    'Episode_Reward/action_magnitude_y',
    'Episode_Reward/action_magnitude_z',
    'Episode_Reward/angular_rate',
    'Episode_Reward/angular_rate_xy',
    'Episode_Reward/upright',
    'Train/mean_reward',
]


CANDIDATES = [
    Candidate('c00_rebuild', {
        'agent.algorithm.entropy_coef': 0.005,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
    }),
    Candidate('c01_low_entropy', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
    }),
    Candidate('c02_vel3_pos12', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 12.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 3.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.4,
        'env.rewards.horizontal_speed.weight': -0.05,
        'env.rewards.action_magnitude_x.weight': -2.0,
        'env.rewards.action_magnitude_y.weight': -2.0,
        'env.rewards.action_magnitude_z.weight': -2.0,
        'env.rewards.velocity_action_rate_x.weight': -40.0,
        'env.rewards.velocity_action_rate_y.weight': -40.0,
        'env.rewards.velocity_action_rate_z.weight': -20.0,
        'env.rewards.angular_rate_xy.weight': -1.5,
        'env.rewards.upright.weight': -25.0,
    }),
    Candidate('c03_vel4_pos15', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.4,
        'env.rewards.horizontal_speed.weight': -0.03,
        'env.rewards.action_magnitude_x.weight': -2.5,
        'env.rewards.action_magnitude_y.weight': -2.5,
        'env.rewards.action_magnitude_z.weight': -2.0,
        'env.rewards.velocity_action_rate_x.weight': -45.0,
        'env.rewards.velocity_action_rate_y.weight': -45.0,
        'env.rewards.velocity_action_rate_z.weight': -25.0,
        'env.rewards.angular_rate_xy.weight': -2.0,
        'env.rewards.upright.weight': -30.0,
    }),
    Candidate('c04_vel4_strict', {
        'agent.algorithm.entropy_coef': 0.0005,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.position_track.params.std': 1.2,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.35,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.action_magnitude_x.weight': -3.0,
        'env.rewards.action_magnitude_y.weight': -3.0,
        'env.rewards.action_magnitude_z.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -50.0,
        'env.rewards.velocity_action_rate_y.weight': -50.0,
        'env.rewards.velocity_action_rate_z.weight': -25.0,
        'env.rewards.angular_rate_xy.weight': -2.0,
        'env.rewards.upright.weight': -35.0,
    }),
    Candidate('c05_vel5_touchdown_push', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 20.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 5.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.4,
        'env.rewards.horizontal_speed.weight': -0.03,
        'env.rewards.vertical_clearance_excess.weight': -1.0,
        'env.rewards.action_magnitude_x.weight': -2.5,
        'env.rewards.action_magnitude_y.weight': -2.5,
        'env.rewards.action_magnitude_z.weight': -2.0,
        'env.rewards.velocity_action_rate_x.weight': -60.0,
        'env.rewards.velocity_action_rate_y.weight': -60.0,
        'env.rewards.velocity_action_rate_z.weight': -30.0,
        'env.rewards.angular_rate_xy.weight': -3.0,
        'env.rewards.upright.weight': -35.0,
    }),
    Candidate('c06_vel6_strong_smooth', {
        'agent.algorithm.entropy_coef': 0.0005,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 6.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.3,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.vertical_clearance_excess.weight': -1.0,
        'env.rewards.action_magnitude_x.weight': -3.5,
        'env.rewards.action_magnitude_y.weight': -3.5,
        'env.rewards.action_magnitude_z.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -60.0,
        'env.rewards.velocity_action_rate_y.weight': -60.0,
        'env.rewards.velocity_action_rate_z.weight': -30.0,
        'env.rewards.angular_rate_xy.weight': -4.0,
        'env.rewards.angular_rate.weight': -1.0,
        'env.rewards.upright.weight': -40.0,
    }),
    Candidate('c07_hspeed_keep', {
        'agent.algorithm.entropy_coef': 0.0005,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 12.0,
        'env.rewards.position_track.params.std': 2.0,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.3,
        'env.rewards.horizontal_speed.weight': -0.08,
        'env.rewards.action_magnitude_x.weight': -3.0,
        'env.rewards.action_magnitude_y.weight': -3.0,
        'env.rewards.velocity_action_rate_x.weight': -70.0,
        'env.rewards.velocity_action_rate_y.weight': -70.0,
        'env.rewards.velocity_action_rate_z.weight': -30.0,
        'env.rewards.angular_rate_xy.weight': -2.0,
        'env.rewards.upright.weight': -30.0,
    }),
    Candidate('c08_max_smooth', {
        'agent.algorithm.entropy_coef': 0.0005,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 10.0,
        'env.rewards.horizontal_velocity_match.weight': 6.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.3,
        'env.rewards.horizontal_speed.weight': -0.03,
        'env.rewards.action_magnitude_x.weight': -4.0,
        'env.rewards.action_magnitude_y.weight': -4.0,
        'env.rewards.action_magnitude_z.weight': -3.0,
        'env.rewards.velocity_action_rate_x.weight': -80.0,
        'env.rewards.velocity_action_rate_y.weight': -80.0,
        'env.rewards.velocity_action_rate_z.weight': -35.0,
        'env.rewards.angular_rate_xy.weight': -4.0,
        'env.rewards.angular_rate.weight': -1.0,
        'env.rewards.upright.weight': -40.0,
    }),
    Candidate('c09_pos18_vel3', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 18.0,
        'env.rewards.position_track.params.std': 1.2,
        'env.rewards.horizontal_velocity_match.weight': 3.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.35,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.action_magnitude_x.weight': -3.0,
        'env.rewards.action_magnitude_y.weight': -3.0,
        'env.rewards.velocity_action_rate_x.weight': -50.0,
        'env.rewards.velocity_action_rate_y.weight': -50.0,
        'env.rewards.velocity_action_rate_z.weight': -20.0,
        'env.rewards.angular_rate_xy.weight': -2.0,
        'env.rewards.upright.weight': -30.0,
    }),
    Candidate('c10_lr_low', {
        'agent.algorithm.entropy_coef': 0.001,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.4,
        'env.rewards.horizontal_speed.weight': -0.03,
        'env.rewards.action_magnitude_x.weight': -2.5,
        'env.rewards.action_magnitude_y.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -45.0,
        'env.rewards.velocity_action_rate_y.weight': -45.0,
        'env.rewards.velocity_action_rate_z.weight': -25.0,
        'env.rewards.angular_rate_xy.weight': -2.0,
        'env.rewards.upright.weight': -30.0,
    }),
    Candidate('c11_lr_low_strict', {
        'agent.algorithm.entropy_coef': 0.0005,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.position_track.params.std': 1.2,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.35,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.action_magnitude_x.weight': -3.0,
        'env.rewards.action_magnitude_y.weight': -3.0,
        'env.rewards.action_magnitude_z.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -50.0,
        'env.rewards.velocity_action_rate_y.weight': -50.0,
        'env.rewards.velocity_action_rate_z.weight': -25.0,
        'env.rewards.angular_rate_xy.weight': -2.0,
        'env.rewards.upright.weight': -35.0,
    }),
    Candidate('c12_push_finish', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 20.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 5.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.4,
        'env.rewards.vertical_clearance_excess.weight': -1.0,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.action_magnitude_x.weight': -2.5,
        'env.rewards.action_magnitude_y.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -60.0,
        'env.rewards.velocity_action_rate_y.weight': -60.0,
        'env.rewards.velocity_action_rate_z.weight': -30.0,
        'env.rewards.angular_rate_xy.weight': -3.0,
        'env.rewards.upright.weight': -35.0,
    }),
    Candidate('c13_accel_ang', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.35,
        'env.rewards.horizontal_speed.weight': -0.03,
        'env.rewards.action_magnitude_x.weight': -2.5,
        'env.rewards.action_magnitude_y.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -50.0,
        'env.rewards.velocity_action_rate_y.weight': -50.0,
        'env.rewards.velocity_action_rate_z.weight': -25.0,
        'env.rewards.uav_acceleration.weight': -1.0,
        'env.rewards.angular_rate.weight': -1.0,
        'env.rewards.angular_velocity_rate.weight': -80.0,
        'env.rewards.angular_rate_xy.weight': -3.0,
        'env.rewards.upright.weight': -40.0,
    }),
    Candidate('c14_min_action', {
        'agent.algorithm.entropy_coef': 0.0005,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 12.0,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.35,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.action_magnitude_x.weight': -5.0,
        'env.rewards.action_magnitude_y.weight': -5.0,
        'env.rewards.action_magnitude_z.weight': -3.0,
        'env.rewards.velocity_action_rate_x.weight': -90.0,
        'env.rewards.velocity_action_rate_y.weight': -90.0,
        'env.rewards.velocity_action_rate_z.weight': -40.0,
        'env.rewards.angular_rate_xy.weight': -4.0,
        'env.rewards.upright.weight': -40.0,
    }),
    Candidate('c15_vel8_pos18', {
        'agent.algorithm.entropy_coef': 0.0005,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 18.0,
        'env.rewards.position_track.params.std': 1.2,
        'env.rewards.horizontal_velocity_match.weight': 8.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.3,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.vertical_clearance_excess.weight': -1.0,
        'env.rewards.action_magnitude_x.weight': -4.0,
        'env.rewards.action_magnitude_y.weight': -4.0,
        'env.rewards.action_magnitude_z.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -80.0,
        'env.rewards.velocity_action_rate_y.weight': -80.0,
        'env.rewards.velocity_action_rate_z.weight': -35.0,
        'env.rewards.angular_rate_xy.weight': -4.0,
        'env.rewards.upright.weight': -40.0,
    }),
    Candidate('c16_touchdown75', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.touchdown_terminated.weight': 75.0,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.4,
        'env.rewards.horizontal_speed.weight': -0.03,
        'env.rewards.vertical_clearance_excess.weight': -1.0,
        'env.rewards.action_magnitude_x.weight': -3.0,
        'env.rewards.action_magnitude_y.weight': -3.0,
        'env.rewards.velocity_action_rate_x.weight': -60.0,
        'env.rewards.velocity_action_rate_y.weight': -60.0,
        'env.rewards.velocity_action_rate_z.weight': -30.0,
        'env.rewards.angular_rate_xy.weight': -2.0,
        'env.rewards.upright.weight': -35.0,
    }),
    Candidate('c17_yaw_relaxed', {
        'agent.algorithm.entropy_coef': 0.001,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.horizontal_velocity_match.weight': 4.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.4,
        'env.rewards.horizontal_speed.weight': -0.03,
        'env.rewards.action_magnitude_x.weight': -2.5,
        'env.rewards.action_magnitude_y.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -50.0,
        'env.rewards.velocity_action_rate_y.weight': -50.0,
        'env.rewards.velocity_action_rate_z.weight': -25.0,
        'env.rewards.angular_rate_xy.weight': -2.5,
        'env.rewards.upright.weight': -35.0,
        'env.rewards.yaw_rate_error.weight': -5.0,
    }),
    Candidate('c18_zero_hspeed', {
        'agent.algorithm.entropy_coef': 0.001,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 12.0,
        'env.rewards.position_track.params.std': 1.5,
        'env.rewards.horizontal_velocity_match.weight': 6.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.35,
        'env.rewards.horizontal_speed.weight': 0.0,
        'env.rewards.action_magnitude_x.weight': -3.0,
        'env.rewards.action_magnitude_y.weight': -3.0,
        'env.rewards.action_magnitude_z.weight': -2.0,
        'env.rewards.velocity_action_rate_x.weight': -70.0,
        'env.rewards.velocity_action_rate_y.weight': -70.0,
        'env.rewards.velocity_action_rate_z.weight': -30.0,
        'env.rewards.angular_rate_xy.weight': -3.0,
        'env.rewards.upright.weight': -35.0,
    }),
    Candidate('c19_balanced_final', {
        'agent.algorithm.entropy_coef': 0.0005,
        'agent.algorithm.learning_rate': 5e-5,
        'env.events.move_platform.params.stationary_env_probability': 0.3,
        'env.rewards.position_track.weight': 15.0,
        'env.rewards.position_track.params.std': 1.3,
        'env.rewards.horizontal_velocity_match.weight': 5.0,
        'env.rewards.horizontal_velocity_match.params.std': 0.35,
        'env.rewards.horizontal_speed.weight': -0.02,
        'env.rewards.vertical_clearance_excess.weight': -1.0,
        'env.rewards.action_magnitude_x.weight': -3.5,
        'env.rewards.action_magnitude_y.weight': -3.5,
        'env.rewards.action_magnitude_z.weight': -2.5,
        'env.rewards.velocity_action_rate_x.weight': -70.0,
        'env.rewards.velocity_action_rate_y.weight': -70.0,
        'env.rewards.velocity_action_rate_z.weight': -30.0,
        'env.rewards.angular_rate.weight': -1.0,
        'env.rewards.angular_rate_xy.weight': -3.0,
        'env.rewards.upright.weight': -40.0,
    }),
]


def scalar_tail_mean(event_path: Path, key: str, n: int = 50) -> float | None:
    ea = EventAccumulator(str(event_path), size_guidance={'scalars': 0})
    ea.Reload()
    if key not in ea.Tags().get('scalars', []):
        return None
    vals = ea.Scalars(key)
    if not vals:
        return None
    tail = vals[-min(n, len(vals)):]
    return mean(v.value for v in tail)


def summarize_run(run_dir: Path) -> dict[str, float | str]:
    event_files = sorted(run_dir.glob('events.out.tfevents.*'))
    if not event_files:
        raise FileNotFoundError(f'No event file in {run_dir}')
    event_path = event_files[-1]
    out: dict[str, float | str] = {'run_dir': str(run_dir), 'event_file': str(event_path)}
    for key in METRIC_KEYS:
        out[key] = scalar_tail_mean(event_path, key)
    touchdown = out['Episode_Termination/touchdown'] or 0.0
    quality = out['Episode_Reward/touchdown_quality'] or 0.0
    pos = out['Episode_Reward/position_track'] or 0.0
    vel = out['Episode_Reward/horizontal_velocity_match'] or 0.0
    ratex = out['Episode_Reward/velocity_action_rate_x'] or 0.0
    ratey = out['Episode_Reward/velocity_action_rate_y'] or 0.0
    ratez = out['Episode_Reward/velocity_action_rate_z'] or 0.0
    amagx = out['Episode_Reward/action_magnitude_x'] or 0.0
    amagy = out['Episode_Reward/action_magnitude_y'] or 0.0
    amagz = out['Episode_Reward/action_magnitude_z'] or 0.0
    angr = out['Episode_Reward/angular_rate'] or 0.0
    angrxy = out['Episode_Reward/angular_rate_xy'] or 0.0
    upright = out['Episode_Reward/upright'] or 0.0
    mean_reward = out['Train/mean_reward'] or 0.0
    out['composite_score'] = (
        2000.0 * touchdown
        + 20.0 * quality
        + 50.0 * pos
        + 100.0 * vel
        + 50.0 * ratex
        + 50.0 * ratey
        + 25.0 * ratez
        + 20.0 * amagx
        + 20.0 * amagy
        + 10.0 * amagz
        + 20.0 * angr
        + 50.0 * angrxy
        + 100.0 * upright
        + 0.5 * mean_reward
    )
    return out


def find_latest_run(prefix: str) -> Path:
    root = Path('logs/rsl_rl/landing_sway')
    matches = sorted(root.glob(f'*_{prefix}'))
    if not matches:
        raise FileNotFoundError(f'No run dir for prefix {prefix}')
    return matches[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='landing_sway')
    ap.add_argument('--num-envs', type=int, default=2048)
    ap.add_argument('--max-iterations', type=int, default=1000)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--count', type=int, default=len(CANDIDATES))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--results-csv', default='logs/rsl_rl/landing_sway/sweep_results.csv')
    args = ap.parse_args()

    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    selected = CANDIDATES[args.start_index: args.start_index + args.count]

    for idx, cand in enumerate(selected, start=args.start_index):
        run_name = f'landing_sway_2.8.0_{cand.name}'
        cmd = [
            sys.executable,
            'scripts/rsl_rl/train.py',
            '--task', args.task,
            '--headless',
            '--device', args.device,
            '--num_envs', str(args.num_envs),
            '--max_iterations', str(args.max_iterations),
            '--seed', str(args.seed),
            '--run_name', run_name,
        ]
        for key, value in cand.overrides.items():
            cmd.append(f'{key}={value}')
        print(f'=== Candidate {idx}: {cand.name} ===')
        print(' '.join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)
        run_dir = find_latest_run(run_name)
        summary = summarize_run(run_dir)
        summary['candidate'] = cand.name
        summary['index'] = idx
        summary['num_envs'] = args.num_envs
        summary['max_iterations'] = args.max_iterations
        summary['seed'] = args.seed
        summary['overrides'] = json.dumps(cand.overrides, sort_keys=True)
        write_header = not results_path.exists()
        with results_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
