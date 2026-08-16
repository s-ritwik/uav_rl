"""Adapt a simulator-observation heave GRU checkpoint for the vision task.

The two tasks share their recurrent/MLP architecture, but their first GRU
inputs differ:

* heave_landing_gru actor/critic: 20 / 146
* heave_landing_vision_gru actor/critic: 18 / 154

This utility preserves all compatible learned parameters and remaps the first
actor and critic GRU input matrices by observation meaning. The output starts
with a fresh optimizer and iteration counter so it behaves as transfer learning
rather than continuing the source run's adaptive optimizer state.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


ACTOR_INPUT_KEY = "memory_a.blocks.0.weight_ih_l0"
CRITIC_INPUT_KEY = "memory_c.blocks.0.weight_ih_l0"
SOURCE_ACTOR_INPUTS = 20
VISION_ACTOR_INPUTS = 18
SOURCE_CRITIC_INPUTS = 146
VISION_CRITIC_INPUTS = 154


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source heave_landing_gru checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint for heave_landing_vision_gru.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5.0e-7,
        help="Learning rate stored in the fresh optimizer parameter group.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output checkpoint if it exists.")
    return parser.parse_args()


def _validate_matrix(matrix: torch.Tensor, key: str, expected_inputs: int) -> None:
    if not torch.is_tensor(matrix) or matrix.ndim != 2 or matrix.shape[1] != expected_inputs:
        shape = tuple(matrix.shape) if torch.is_tensor(matrix) else type(matrix).__name__
        raise ValueError(f"Expected '{key}' to have {expected_inputs} inputs, got {shape}.")


def _remap_columns(
    source: torch.Tensor,
    output_width: int,
    ranges: list[tuple[slice, slice]],
) -> torch.Tensor:
    output = source.new_zeros((source.shape[0], output_width))
    for source_slice, output_slice in ranges:
        output[:, output_slice] = source[:, source_slice]
    return output


def adapt_checkpoint(source_path: Path, output_path: Path, learning_rate: float, force: bool = False) -> None:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_path}")
    if source_path == output_path:
        raise ValueError("Source and output checkpoint paths must differ.")
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to overwrite it.")
    if learning_rate <= 0.0:
        raise ValueError(f"Learning rate must be positive, got {learning_rate}.")

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain model_state_dict and optimizer_state_dict.")

    model = copy.deepcopy(checkpoint["model_state_dict"])
    if ACTOR_INPUT_KEY not in model or CRITIC_INPUT_KEY not in model:
        raise KeyError("Checkpoint does not use the expected stacked-GRU actor/critic parameter names.")
    _validate_matrix(model[ACTOR_INPUT_KEY], ACTOR_INPUT_KEY, SOURCE_ACTOR_INPUTS)
    _validate_matrix(model[CRITIC_INPUT_KEY], CRITIC_INPUT_KEY, SOURCE_CRITIC_INPUTS)

    # Base actor: pos(3), vel(3), quat(4), ang_vel(3), gravity(3), action(4).
    # Vision actor: vision pos(3), vel(3), quat(4), gravity(3), action(4), valid(1).
    # The new vision-valid column starts at zero; source angular velocity is dropped.
    model[ACTOR_INPUT_KEY] = _remap_columns(
        model[ACTOR_INPUT_KEY],
        VISION_ACTOR_INPUTS,
        [
            (slice(0, 10), slice(0, 10)),
            (slice(13, 16), slice(10, 13)),
            (slice(16, 20), slice(13, 17)),
        ],
    )

    # Preserve the source value function by routing source pose/velocity/quaternion
    # through the vision critic's privileged true-state inputs. Vision-only critic
    # inputs begin at zero and can be learned during fine-tuning.
    model[CRITIC_INPUT_KEY] = _remap_columns(
        model[CRITIC_INPUT_KEY],
        VISION_CRITIC_INPUTS,
        [
            (slice(0, 3), slice(18, 21)),
            (slice(3, 6), slice(21, 24)),
            (slice(6, 10), slice(24, 28)),
            (slice(13, 16), slice(10, 13)),
            (slice(16, 20), slice(13, 17)),
            (slice(20, 146), slice(28, 154)),
        ],
    )

    optimizer = copy.deepcopy(checkpoint["optimizer_state_dict"])
    optimizer["state"] = {}
    for group in optimizer.get("param_groups", []):
        group["lr"] = learning_rate

    checkpoint["model_state_dict"] = model
    checkpoint["optimizer_state_dict"] = optimizer
    checkpoint["iter"] = 0
    infos = dict(checkpoint.get("infos") or {})
    infos.update(
        {
            "transfer_source": str(source_path),
            "transfer_target": "heave_landing_vision_gru",
            "transfer_learning_rate": learning_rate,
        }
    )
    checkpoint["infos"] = infos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Wrote transferred checkpoint: {output_path}")
    print(f"  {ACTOR_INPUT_KEY}: {tuple(model[ACTOR_INPUT_KEY].shape)}")
    print(f"  {CRITIC_INPUT_KEY}: {tuple(model[CRITIC_INPUT_KEY].shape)}")
    print(f"  optimizer: fresh state at learning_rate={learning_rate:g}, iteration=0")


def main() -> None:
    args = _parse_args()
    adapt_checkpoint(args.source, args.output, args.learning_rate, args.force)


if __name__ == "__main__":
    main()
