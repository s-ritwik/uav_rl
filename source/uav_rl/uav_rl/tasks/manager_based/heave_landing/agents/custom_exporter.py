from __future__ import annotations

import copy
import os

import torch


class _GruBlockWithState(torch.nn.Module):
    """Single GRU block paired with a persistent hidden-state buffer."""

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        self.register_buffer("hidden_state", torch.zeros(1, 1, rnn.hidden_size))


class _StackedGruTorchPolicyExporter(torch.nn.Module):
    """JIT exporter for custom stacked/asymmetric GRU actor policies."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        self.blocks = torch.nn.ModuleList([_GruBlockWithState(copy.deepcopy(block)) for block in policy.memory_a.blocks])
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        x = self.normalizer(x)
        out = x.unsqueeze(0)
        for block in self.blocks:
            out, next_hidden_state = block.rnn(out, block.hidden_state)
            block.hidden_state[:] = next_hidden_state
        out = out.squeeze(0)
        return self.actor(out)

    @torch.jit.export
    def reset(self):
        for block in self.blocks:
            block.hidden_state[:] = 0.0

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(export_path)


class _StackedGruOnnxPolicyExporter(torch.nn.Module):
    """ONNX exporter for custom stacked/asymmetric GRU actor policies."""

    def __init__(self, policy, normalizer=None, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(policy.actor)
        self.rnn_blocks = copy.deepcopy(policy.memory_a.blocks)
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x, *hidden_states):
        x = self.normalizer(x)
        out = x.unsqueeze(0)
        next_hidden_states = []
        for block, hidden_state in zip(self.rnn_blocks, hidden_states, strict=True):
            out, next_hidden_state = block(out, hidden_state)
            next_hidden_states.append(next_hidden_state)
        out = out.squeeze(0)
        actions = self.actor(out)
        return (actions, *next_hidden_states)

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.rnn_blocks[0].input_size)
        hidden_states = [torch.zeros(1, 1, block.hidden_size) for block in self.rnn_blocks]
        input_names = ["obs"] + [f"h_in_{i}" for i in range(len(self.rnn_blocks))]
        output_names = ["actions"] + [f"h_out_{i}" for i in range(len(self.rnn_blocks))]
        torch.onnx.export(
            self,
            (obs, *hidden_states),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={},
        )


def export_custom_policy_as_jit(policy, normalizer, path: str, filename: str = "policy.pt") -> None:
    exporter = _StackedGruTorchPolicyExporter(policy, normalizer)
    exporter.export(path, filename)


def export_custom_policy_as_onnx(
    policy, normalizer, path: str, filename: str = "policy.onnx", verbose: bool = False
) -> None:
    exporter = _StackedGruOnnxPolicyExporter(policy, normalizer, verbose=verbose)
    exporter.export(path, filename)
