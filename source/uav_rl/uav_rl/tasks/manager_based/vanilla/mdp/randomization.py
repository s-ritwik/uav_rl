from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


@configclass
class VanillaDomainRandomizationCfg:
    """Domain-randomization switches and sampling ranges for the vanilla UAV task."""

    enabled: bool = False

    mass_noise_enabled: bool = True
    mass_noise_probability: float = 0.5
    mass_noise_std_kg: float = 0.1
    mass_noise_clip_kg: float = 0.3

    action_delay_enabled: bool = True
    action_delay_probability: float = 0.2
    action_delay_steps_range: tuple[int, int] = (1, 3)

    state_estimation_noise_enabled: bool = True
    state_estimation_noise_probability: float = 0.3
    position_noise_std_m: float = 0.02
    linear_velocity_noise_std_mps: float = 0.03
    angular_velocity_noise_std_rps: float = 0.03
    attitude_noise_std_rad: float = 0.015
    projected_gravity_noise_std: float = 0.02

    thrust_asymmetry_enabled: bool = True
    thrust_asymmetry_probability: float = 0.2
    thrust_asymmetry_scale_range: tuple[float, float] = (0.9, 1.1)

    motor_lag_enabled: bool = True
    motor_lag_probability: float = 0.5
    motor_lag_time_constant_s_range: tuple[float, float] = (0.02, 0.08)


class VanillaDomainRandomizationState:
    """Per-environment runtime state sampled from :class:`VanillaDomainRandomizationCfg`."""

    def __init__(self, cfg: VanillaDomainRandomizationCfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.mass_noise_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.action_delay_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.state_estimation_noise_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.thrust_asymmetry_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motor_lag_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.mass_delta_kg = torch.zeros(num_envs, device=device)
        self.action_delay_steps = torch.zeros(num_envs, dtype=torch.int, device=device)
        self.thrust_asymmetry_scale = torch.ones((num_envs, 4), device=device)
        self.motor_lag_tau_s = torch.zeros(num_envs, device=device)

    def reset_envs(self, env_ids: torch.Tensor):
        self.mass_noise_active[env_ids] = False
        self.action_delay_active[env_ids] = False
        self.state_estimation_noise_active[env_ids] = False
        self.thrust_asymmetry_active[env_ids] = False
        self.motor_lag_active[env_ids] = False

        self.mass_delta_kg[env_ids] = 0.0
        self.action_delay_steps[env_ids] = 0
        self.thrust_asymmetry_scale[env_ids] = 1.0
        self.motor_lag_tau_s[env_ids] = 0.0

    def sample_envs(self, env_ids: torch.Tensor):
        self.reset_envs(env_ids)
        if not self.cfg.enabled or env_ids.numel() == 0:
            return

        num_envs = int(env_ids.numel())

        mass_mask = self._sample_mask(num_envs, self.cfg.mass_noise_enabled, self.cfg.mass_noise_probability)
        self.mass_noise_active[env_ids] = mass_mask
        mass_delta = torch.randn(num_envs, device=self.device) * float(self.cfg.mass_noise_std_kg)
        mass_delta = torch.clamp(
            mass_delta,
            min=-float(self.cfg.mass_noise_clip_kg),
            max=float(self.cfg.mass_noise_clip_kg),
        )
        self.mass_delta_kg[env_ids] = mass_delta * mass_mask.to(dtype=mass_delta.dtype)

        delay_mask = self._sample_mask(num_envs, self.cfg.action_delay_enabled, self.cfg.action_delay_probability)
        self.action_delay_active[env_ids] = delay_mask
        min_delay, max_delay = self.cfg.action_delay_steps_range
        delay_steps = torch.randint(min_delay, max_delay + 1, (num_envs,), device=self.device, dtype=torch.int)
        self.action_delay_steps[env_ids] = delay_steps * delay_mask.to(dtype=delay_steps.dtype)

        state_mask = self._sample_mask(
            num_envs, self.cfg.state_estimation_noise_enabled, self.cfg.state_estimation_noise_probability
        )
        self.state_estimation_noise_active[env_ids] = state_mask

        thrust_mask = self._sample_mask(
            num_envs, self.cfg.thrust_asymmetry_enabled, self.cfg.thrust_asymmetry_probability
        )
        self.thrust_asymmetry_active[env_ids] = thrust_mask
        thrust_scales = self._sample_uniform(self.cfg.thrust_asymmetry_scale_range, (num_envs, 4))
        thrust_scales = torch.where(thrust_mask.unsqueeze(-1), thrust_scales, torch.ones_like(thrust_scales))
        self.thrust_asymmetry_scale[env_ids] = thrust_scales

        lag_mask = self._sample_mask(num_envs, self.cfg.motor_lag_enabled, self.cfg.motor_lag_probability)
        self.motor_lag_active[env_ids] = lag_mask
        lag_tau = self._sample_uniform(self.cfg.motor_lag_time_constant_s_range, (num_envs,))
        self.motor_lag_tau_s[env_ids] = lag_tau * lag_mask.to(dtype=lag_tau.dtype)

    def _sample_mask(self, num_envs: int, enabled: bool, probability: float) -> torch.Tensor:
        if not enabled or probability <= 0.0:
            return torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if probability >= 1.0:
            return torch.ones(num_envs, dtype=torch.bool, device=self.device)
        return torch.rand(num_envs, device=self.device) < float(probability)

    def _sample_uniform(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = value_range
        return torch.rand(shape, device=self.device) * (high - low) + low


def get_domain_randomization_state(env) -> VanillaDomainRandomizationState | None:
    return getattr(env, "_vanilla_domain_randomization", None)


def apply_additive_state_noise(env, values: torch.Tensor, std: float) -> torch.Tensor:
    state = get_domain_randomization_state(env)
    if state is None or not state.cfg.enabled or std <= 0.0:
        return values
    active_mask = state.state_estimation_noise_active
    if not bool(torch.any(active_mask)):
        return values

    noise = torch.randn_like(values) * float(std)
    mask = active_mask.view(values.shape[0], *([1] * (values.ndim - 1))).to(dtype=values.dtype)
    return values + noise * mask


def apply_quaternion_state_noise(env, quat_wxyz: torch.Tensor, std_rad: float) -> torch.Tensor:
    state = get_domain_randomization_state(env)
    if state is None or not state.cfg.enabled or std_rad <= 0.0:
        return quat_wxyz
    active_mask = state.state_estimation_noise_active
    if not bool(torch.any(active_mask)):
        return quat_wxyz

    axis = torch.randn((quat_wxyz.shape[0], 3), device=quat_wxyz.device, dtype=quat_wxyz.dtype)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1.0e-6)
    angle = torch.randn((quat_wxyz.shape[0],), device=quat_wxyz.device, dtype=quat_wxyz.dtype) * float(std_rad)
    delta_quat = math_utils.quat_from_angle_axis(angle, axis)
    noisy_quat = math_utils.quat_mul(delta_quat, quat_wxyz)
    return torch.where(active_mask.unsqueeze(-1), noisy_quat, quat_wxyz)


def apply_controller_state_estimation_noise(
    env,
    quat_wxyz: torch.Tensor,
    body_rates_b: torch.Tensor,
    velocity_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = get_domain_randomization_state(env)
    if state is None or not state.cfg.enabled or not bool(torch.any(state.state_estimation_noise_active)):
        return quat_wxyz, body_rates_b, velocity_w

    quat_wxyz = apply_quaternion_state_noise(env, quat_wxyz, state.cfg.attitude_noise_std_rad)
    body_rates_b = apply_additive_state_noise(env, body_rates_b, state.cfg.angular_velocity_noise_std_rps)
    velocity_w = apply_additive_state_noise(env, velocity_w, state.cfg.linear_velocity_noise_std_mps)
    return quat_wxyz, body_rates_b, velocity_w
