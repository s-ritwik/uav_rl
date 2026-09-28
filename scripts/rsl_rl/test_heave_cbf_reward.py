"""CPU checks of the production CBF functions without booting Isaac Sim.

Run with the Isaac Python environment. AST extraction avoids simulator imports;
the executed function bodies are taken directly from the production source.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[2] / "source/uav_rl/uav_rl/tasks/manager_based/heave_landing/mdp/rewards.py"
NAMES = {"cbf_braking_envelope_loss", "cbf_braking_envelope_penalty", "cbf_braking_metrics"}
tree = ast.parse(SOURCE.read_text())
scope = {
    "torch": torch,
    "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
    # Fixtures below provide marker-referenced state, not physical body-center state.
    "platform_reference_data": lambda env, name="platform": env.scene[name].data,
    "flight_mask": lambda env: 1.0,
}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES], type_ignores=[]), str(SOURCE), "exec"), scope)


class CbfRewardTests(unittest.TestCase):
    def test_envelope_and_scale(self):
        loss, h = scope["cbf_braking_envelope_loss"](
            torch.tensor([1., .2, .2, .02, .02, 1.]),
            torch.tensor([-1., -.4, -.8, -.2, -.5, 0.]), .7, .25, .1,
        )
        torch.testing.assert_close(h, torch.tensor([.33035714, .13035714, -.2125, .02, -.11392857, 1.]))
        torch.testing.assert_close(loss * -.25, torch.tensor([0., 0., -.8125, 0., -.31964286, 0.]))

    def test_large_violation_is_capped(self):
        loss, _ = scope["cbf_braking_envelope_loss"](
            torch.tensor([0.]), torch.tensor([-4.]), .7, .25, .25, 4.0
        )
        torch.testing.assert_close(loss, torch.tensor([4.0]))

    def test_safe_hover_ascent_and_touchdown_have_no_bonus(self):
        loss, _ = scope["cbf_braking_envelope_loss"](
            torch.tensor([0., 0., 1., 10.]), torch.tensor([-.25, 1., 0., 0.]), .7, .25, .1
        )
        torch.testing.assert_close(loss, torch.zeros(4))

    def test_contact_uses_preimpact_loss_and_reset_does_not_leak(self):
        robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., .285]]), root_lin_vel_w=torch.tensor([[0., 0., -.5]])))
        platform = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(1, 3), root_lin_vel_w=torch.zeros(1, 3)))
        contact = torch.tensor([False])
        env = SimpleNamespace(scene={"robot": robot, "platform": platform}, episode_length_buf=torch.tensor([2]), reset_buf=torch.tensor([False]), termination_manager=SimpleNamespace(get_term=lambda _: contact))
        fn = scope["cbf_braking_envelope_penalty"]
        first = fn(env).clone()
        self.assertGreater(first.item(), 0)
        contact[:] = True
        env.reset_buf[:] = True
        env.episode_length_buf += 1
        robot.data.root_lin_vel_w[:] = 0
        torch.testing.assert_close(fn(env), first)
        contact[:] = False
        env.reset_buf[:] = False
        env.episode_length_buf[:] = 1
        robot.data.root_pos_w[:, 2] = 2
        torch.testing.assert_close(fn(env), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
