# Heave Landing Vision GRU Tuning Handoff

## Objective

Fine-tune the pretrained `heave_landing_gru` policy on
`heave_landing_vision_gru` with exactly 16 environments.

The success gate is evaluated at iteration 500:

- `Episode_Termination/touchdown` must remain at or above 70%, not merely hit
  70% in one noisy iteration.
- Use rolling 20, 50, and 100 iteration means. Treat the run as successful only
  when all three are at least `0.70` at iteration 500.
- Verify the checkpoint contains only finite tensors.
- Report `Curriculum/touchdown_quality_metrics/good_touchdown_rate` separately.
  It is not currently part of the 70% gate, but it is essential landing-quality
  evidence.

For this tuning phase, modify only:

`source/uav_rl/uav_rl/tasks/manager_based/heave_landing_vision/agents/rsl_rl_ppo_cfg.py`

Do not change environment observations, rewards, termination thresholds, camera
code, or the user-modified post-init file while running this PPO-only sweep.

## Laptop State At Handoff

The laptop training process was stopped intentionally after capturing its state.

- Last trial: `T5`
- Run directory:
  `logs/rsl_rl/heave_landing_vision_gru/2026-08-18_07-00-57_vision_heave_0.05_mb2_lr2e5_ep5_s48`
- Last event iteration: `415`
- Latest saved checkpoint: `model_400.pt`
- Metrics at iteration 414:
  - touchdown latest: `35.42%`
  - touchdown rolling 20: `43.91%`
  - touchdown rolling 50: `45.76%`
  - touchdown rolling 100: `49.05%`
  - out-of-bounds rolling 100: `51.81%`
  - good-touchdown rolling 100: `0%`
  - mean reward rolling 100: `-104.91`
- T5 did not reach iteration 500 because the tuning work is being transferred to
  the desktop.

The stopped laptop tmux session was `vision_heave_tune5`. Do not assume a tmux
session copied to another machine; only files/checkpoints are transferable.

## Git And Worktree State

The source checkpoint is tracked by Git:

`logs/rsl_rl/heave_landing_gru/2026-07-20_09-26-48_heave_gru_5.0.1/model_14500.pt`

The checkpoint mapper is tracked by Git in commit `7899781`:

`source/uav_rl/uav_rl/tasks/manager_based/heave_landing_vision/adapt_heave_gru_checkpoint.py`

Mapped checkpoints and vision run checkpoints under `logs/` are generally not
tracked. Copy them explicitly to the desktop if a staged checkpoint is required.

### Desktop Transfer Checklist

Transfer both of these artifacts:

- `HEAVE_LANDING_VISION_TUNING_HANDOFF.md`
- `logs/rsl_rl/heave_landing_vision_gru/2026-08-17_22-45-26_vision_heave_0.03b_fixed_lr2e5_mb4_ep10_s48/model_375_stage2_lr5e6.pt`

The staged checkpoint is ready to train: it contains the exact `model_375.pt`
model weights, a fresh optimizer at `5e-6`, and iteration zero. Copying the raw
`model_375.pt` is optional unless the desktop session needs to reproduce or alter
the staging procedure.

For example, replace the destination placeholders and run from the repository
root:

```bash
rsync -av --relative \
  HEAVE_LANDING_VISION_TUNING_HANDOFF.md \
  logs/rsl_rl/heave_landing_vision_gru/2026-08-17_22-45-26_vision_heave_0.03b_fixed_lr2e5_mb4_ep10_s48/model_375_stage2_lr5e6.pt \
  '<desktop-host>:<desktop-repository-path>/'
```

Verify the copied checkpoint on the desktop:

```bash
sha256sum logs/rsl_rl/heave_landing_vision_gru/2026-08-17_22-45-26_vision_heave_0.03b_fixed_lr2e5_mb4_ep10_s48/model_375_stage2_lr5e6.pt
```

Expected checkpoint metadata:

```text
size: 6994265 bytes
sha256: bfb342eda7f81cff51babd5d7289212e776aa5eb23c5da2960cae51c8328d1c4
iteration: 0
actor first GRU input matrix: (384, 18)
critic first GRU input matrix: (1536, 154)
all model tensors finite: true
optimizer states: 0
saved optimizer LR: 5e-6
weights identical to model_375.pt: true
```

The handoff Markdown is currently a new worktree file. If transferring through
Git rather than `rsync`, add and commit it first. Do not force-add the 21 MB
raw checkpoint unless the repository's checkpoint-storage policy permits it;
direct transfer of the prepared 7 MB staged checkpoint is safer.

At handoff, these files have uncommitted modifications:

- `source/uav_rl/uav_rl/tasks/manager_based/heave_landing_vision/agents/rsl_rl_ppo_cfg.py`
  - tuning changes made by Codex
- `source/uav_rl/uav_rl/tasks/manager_based/heave_landing_vision/heave_landing_vision_post_init_cfg.py`
  - user-owned changes; do not overwrite or revert

Check `git status --short` before editing. If unexpected changes appear while a
desktop Codex session is working, stop and resolve ownership before proceeding.

## Policy Compatibility And Mapping

The source and vision policies have the same recurrent and MLP architecture:

- actor GRU stack: `[128, 64]`
- actor MLP: `[32] -> 4 actions`
- critic GRU stack: `[512, 256]`
- critic MLP: `[128] -> 1 value`

Their observation widths differ:

- source actor/critic: `20 / 146`
- vision actor/critic: `18 / 154`

The mapper remaps the first actor and critic GRU input matrices by observation
meaning, resets iteration to zero, clears Adam state, and writes the requested
fixed optimizer learning rate.

Important actor mapping:

- preserve relative position, velocity, and quaternion
- drop source relative angular velocity
- shift projected gravity and last action into vision indices
- initialize the new `vision_available` input column to zero

Important critic mapping:

- route source relative pose/velocity/quaternion into the vision critic's
  privileged true-state columns
- preserve projected gravity and last action
- shift source future-platform/world-velocity inputs into vision critic columns
- initialize vision-only critic inputs to zero

Never load `model_14500.pt` directly into the vision task. Its input matrices do
not match.

## Regenerate A Mapped Source Checkpoint

Run from the repository root in the Isaac environment:

```bash
python3 source/uav_rl/uav_rl/tasks/manager_based/heave_landing_vision/adapt_heave_gru_checkpoint.py \
  --source logs/rsl_rl/heave_landing_gru/2026-07-20_09-26-48_heave_gru_5.0.1/model_14500.pt \
  --output logs/rsl_rl/heave_landing_gru/2026-07-20_09-26-48_heave_gru_5.0.1/model_14500_vision_transfer_lr2e5.pt \
  --learning-rate 2e-5 \
  --force
```

Expected validation output:

```text
memory_a.blocks.0.weight_ih_l0: (384, 18)
memory_c.blocks.0.weight_ih_l0: (1536, 154)
optimizer: fresh state at learning_rate=2e-05, iteration=0
```

`train.py` supports a direct absolute checkpoint path. Use `--resume` together
with `--checkpoint`; do not use `--load_run` on the unadapted source run.

## Completed Trial Results

All rates below are rolling 100-iteration means at the named checkpoint.

| Trial | PPO changes | Best checkpoint | Best touchdown | At iteration 500 | Good touchdown | Outcome |
|---|---|---:|---:|---:|---:|---|
| T1 | steps 48, epochs 5, minibatches 6, LR `5e-7` adaptive, entropy `5e-3` | 500 | 59.21% | 59.21% | 0% | Improved, then collapsed to 30.36% by checkpoint 900 |
| T2 | steps 64, epochs 5, minibatches 4, LR `1e-5` fixed, entropy `5e-3` | 100 | 50.49% | not run | 0% | Stopped at 100; no useful improvement |
| T3a | steps 48, epochs 10, minibatches 4, LR `2e-5` fixed, entropy `1e-3` | interrupted at 64 | n/a | not run | 0% | Power interruption |
| T3b | same as T3a | **375** | **65.97%** | 54.27% | 0.05% at 500 | Closest to target, but unstable after checkpoint 375 |
| T4 | steps 48, epochs 10, minibatches 1, LR `5e-5` fixed, entropy `1e-3` | 100 | 47.85% | 23.53% | 0.23% at 500 | Severe collapse |
| T5 | steps 48, epochs 5, minibatches 2, LR `2e-5` fixed, entropy `5e-3` | 150 | 58.56% | stopped at 415 | 0.13% at 150 | Stable early, then regressed to 49.05% by iteration 414 |

Additional T3b details:

- run directory:
  `logs/rsl_rl/heave_landing_vision_gru/2026-08-17_22-45-26_vision_heave_0.03b_fixed_lr2e5_mb4_ep10_s48`
- recommended staged checkpoint: `model_375.pt`
- checkpoint-375 rolling 100:
  - touchdown: `65.9701%`
  - out-of-bounds: `34.4635%`
  - good touchdown: `0.0625%`
  - mean reward: `-99.7922`
- checkpoint-500 rolling 100:
  - touchdown: `54.2682%`
  - out-of-bounds: `46.9570%`
  - good touchdown: `0.0521%`

T3b and T4 overlapped in wall-clock execution because a sandboxed stop command
did not reach the host PID. Their environments were separate, but GPU contention
affected throughput. Do not repeat this: run exactly one training process.

## Current PPO File At Handoff

The GRU runner currently contains the T5 parameters:

```python
num_steps_per_env = 48
save_interval = 25

algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=5e-3,
    num_learning_epochs=5,
    num_mini_batches=2,
    learning_rate=2.0e-5,
    schedule="fixed",
    gamma=0.99,
    lam=0.95,
    desired_kl=5e-3,
    max_grad_norm=1.0,
)
```

The recurrent architecture must remain unchanged so transferred checkpoints stay
compatible.

## Recommended Next Trial: Staged Low-LR Fine-Tuning

T3b `model_375.pt` is the strongest measured checkpoint and is only four
percentage points below the gate. Do not continue it with the same optimizer/LR;
that exact continuation degraded by checkpoint 500.

Recommended T6:

1. Copy the prepared `model_375_stage2_lr5e6.pt` to the desktop.
2. Verify that it has:
   - model weights unchanged
   - Adam state cleared
   - checkpoint iteration reset to zero
   - optimizer LR set to `5e-6`
3. Change only the vision GRU PPO block to:

```python
num_steps_per_env = 48
save_interval = 25

algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=1e-3,
    num_learning_epochs=5,
    num_mini_batches=2,
    learning_rate=5e-6,
    schedule="fixed",
    gamma=0.99,
    lam=0.95,
    desired_kl=5e-3,
    max_grad_norm=1.0,
)
```

Rationale:

- T3b proves that the four-minibatch configuration can reach 65.97%.
- It later moves away from that policy, so the next step should reduce update
  pressure rather than restart aggressive adaptation.
- Two minibatches use all 16 environments and provide eight complete recurrent
  trajectories per update.
- Lower entropy reduces pressure to preserve stochastic action noise near a
  nearly successful landing policy.

The staged checkpoint already exists on the laptop. The following command is
provided only to reproduce it from raw `model_375.pt` if necessary, adjusting
paths if the repository location differs:

```bash
python3 - <<'PY'
from pathlib import Path
import torch

source = Path("logs/rsl_rl/heave_landing_vision_gru/2026-08-17_22-45-26_vision_heave_0.03b_fixed_lr2e5_mb4_ep10_s48/model_375.pt")
output = source.with_name("model_375_stage2_lr5e6.pt")

checkpoint = torch.load(source, map_location="cpu", weights_only=False)
checkpoint["optimizer_state_dict"]["state"] = {}
for group in checkpoint["optimizer_state_dict"].get("param_groups", []):
    group["lr"] = 5e-6
checkpoint["iter"] = 0
infos = dict(checkpoint.get("infos") or {})
infos.update({
    "stage2_source": str(source.resolve()),
    "stage2_learning_rate": 5e-6,
})
checkpoint["infos"] = infos
torch.save(checkpoint, output)
print(output)
PY
```

Fresh optimizer state is important. With a fixed schedule, loading an existing
checkpoint restores the optimizer param-group LR and can silently override the
new config LR.

## Launch T6 On The Desktop

First verify that no other Isaac/Python training process is using the GPU:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Launch one tmux session:

```bash
tmux new-session -s vision_heave_tune6
```

Inside tmux, activate the Isaac environment and run:

```bash
python3 scripts/rsl_rl/train.py \
  --task=heave_landing_vision_gru \
  --num_envs=16 \
  --headless \
  --max_iterations=5000 \
  --run_name=vision_heave_0.06_stage2_from_t3_model375_lr5e6_mb2 \
  --resume \
  --checkpoint="$(pwd)/logs/rsl_rl/heave_landing_vision_gru/2026-08-17_22-45-26_vision_heave_0.03b_fixed_lr2e5_mb4_ep10_s48/model_375_stage2_lr5e6.pt"
```

Expected first-iteration sample count is `16 * 48 = 768`.

## Monitoring

Attach to the run with:

```bash
tmux attach -t vision_heave_tune6
```

Checkpoints should appear every 25 iterations. Do not wait only for console
output; read TensorBoard scalars directly:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path("REPLACE_WITH_RUN_DIRECTORY")
event = next(run.glob("events.out.tfevents.*"))
ea = EventAccumulator(str(event), size_guidance={"scalars": 0})
ea.Reload()

tags = [
    "Episode_Termination/touchdown",
    "Episode_Termination/out_of_bounds",
    "Curriculum/touchdown_quality_metrics/touchdown_rate",
    "Curriculum/touchdown_quality_metrics/good_touchdown_rate",
    "Curriculum/touchdown_quality_metrics/good_touchdown_pct",
    "Train/mean_reward",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/learning_rate",
    "Policy/mean_noise_std",
]

for tag in tags:
    values = ea.Scalars(tag)
    data = np.asarray([entry.value for entry in values], dtype=float)
    windows = [w for w in (20, 50, 100) if len(data) >= w]
    means = " ".join(f"m{w}={data[-w:].mean():.6f}" for w in windows)
    print(f"{tag}: step={values[-1].step} latest={data[-1]:.6f} {means}")
PY
```

At each checkpoint also verify model tensors are finite. At iteration 500, the
success audit must use metrics filtered to `step <= 500`; do not accidentally
include later degradation.

## Process-Control Rules

- Run exactly one Isaac training process.
- `num_mini_batches` must divide 16 exactly. RSL-RL's recurrent generator uses
  integer division and can silently discard environments otherwise. T1 used six
  minibatches and therefore used only 12 of 16 environments per epoch.
- Use host-level `nvidia-smi` to find the real PID. A sandboxed `ps` or `kill`
  may operate in a different PID namespace.
- Stop failed runs using the host PID and `SIGINT`, then verify GPU memory is
  released before launching the next trial.
- Do not infer success from shaped reward. Several failed trials improved mean
  reward while touchdown rate collapsed. Touchdown and out-of-bounds are the
  authoritative behavior metrics.

## Decision Tree After T6

1. At checkpoints 100, 200, 300, 400, and 500, record rolling 20/50/100 metrics.
2. Continue through iteration 500 unless there is numerical failure or a clear
   sustained collapse.
3. At iteration 500:
   - if touchdown rolling 20/50/100 are all at least 70%, stop the run and report
     the final PPO config and checkpoint;
   - otherwise stop the run and record the exact metrics.
4. If T6 fails, the PPO-only evidence is strong that the reward/observation
   problem is limiting performance:
   - all tested PPO configurations produce very few good touchdowns;
   - shaped reward often improves while touchdown worsens;
   - the best sustained touchdown observed is 65.97%.
5. Do not silently modify rewards or vision observations. Report the PPO-only
   result and ask the user for permission to inspect/tune reward alignment,
   touchdown-quality weighting, vision validity/dropout behavior, and the
   distribution mismatch between mapped simulator observations and camera
   observations.

## Completion Status

The tuning objective is **not complete** at handoff.

- Required sustained touchdown: `>= 70%` at iteration 500
- Best measured rolling-100 touchdown: `65.97%` at T3b checkpoint 375
- Best completed iteration-500 result: `59.21%` in T1
- Good-touchdown quality remains near zero in all vision trials

The desktop Codex session should continue from the recommended staged T6 trial,
not declare success based on a single iteration above 70%.
