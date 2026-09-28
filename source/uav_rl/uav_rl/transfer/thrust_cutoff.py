"""Latched, simulation-time rotor shutdown models shared by RL and Pegasus."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ThrustCutoffCfg:
    enabled: bool = True
    clearance_m: float = 0.07
    xy_tolerance_m: float = 0.10
    vehicle_z0_m: float = 0.165
    delay_s: float = 0.05
    thrust_tau_s: float = 0.25
    # Exponentials never reach zero; snap off below this fraction of trigger thrust.
    off_thrust_fraction: float = 0.001

    def validate(self):
        values = (self.clearance_m, self.xy_tolerance_m, self.vehicle_z0_m, self.delay_s,
                  self.thrust_tau_s, self.off_thrust_fraction)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Cutoff parameters must be finite")
        if min(values[:4]) < 0 or self.thrust_tau_s <= 0 or not 0 < self.off_thrust_fraction < 1:
            raise ValueError("Invalid cutoff clearance, delay, decay, or off threshold")


class RotorCutoff:
    """Hold actual trigger rotor speeds through the delay, then decay their thrust."""
    def __init__(self, cfg=None):
        self.cfg = cfg or ThrustCutoffCfg()
        self.cfg.validate()
        self.reset()

    def reset(self):
        self.active = False
        self.elapsed_s = 0.0
        self.initial_omega = None
        self.off = False

    def trigger(self, omega):
        if self.cfg.enabled and not self.active:
            self.active = True
            self.initial_omega = np.asarray(omega, dtype=float).copy()

    def output(self, live_omega):
        if not self.active:
            return live_omega
        age = max(self.elapsed_s - self.cfg.delay_s, 0.0)
        fraction = math.exp(-age / self.cfg.thrust_tau_s)
        self.off = fraction <= self.cfg.off_thrust_fraction
        return self.initial_omega * (0.0 if self.off else math.sqrt(fraction))

    def advance(self, dt):
        if self.active:
            self.elapsed_s += float(dt)


class BatchedRotorCutoff:
    def __init__(self, num_envs, device, cfg):
        import torch
        self.cfg = cfg
        cfg.validate()
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.elapsed_s = torch.zeros(num_envs, device=device)
        self.initial_omega = torch.zeros((num_envs, 4), device=device)
        self.off = torch.zeros_like(self.active)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.active[ids] = False
        self.elapsed_s[ids] = 0
        self.initial_omega[ids] = 0
        self.off[ids] = False

    def trigger(self, eligible, omega):
        if self.cfg.enabled:
            new = eligible & ~self.active
            self.initial_omega[new] = omega[new]
            self.active |= new

    def output(self, live_omega):
        import torch
        age = (self.elapsed_s - self.cfg.delay_s).clamp_min(0)
        fraction = torch.exp(-age / self.cfg.thrust_tau_s)
        self.off[:] = self.active & (fraction <= self.cfg.off_thrust_fraction)
        scale = torch.where(self.off, 0.0, torch.sqrt(fraction))
        return torch.where(self.active[:, None], self.initial_omega * scale[:, None], live_omega)

    def advance(self, dt):
        self.elapsed_s += self.active * float(dt)
