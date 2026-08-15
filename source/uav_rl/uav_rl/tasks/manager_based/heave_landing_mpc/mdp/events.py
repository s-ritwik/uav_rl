from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .randomization import HeaveLandingDomainRandomizationCfg, HeaveLandingDomainRandomizationState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class HarmonicAxisMotionCfg:
    """Sampling ranges for one harmonic motion channel."""

    enabled: bool = False
    num_terms_range: tuple[int, int] = (2, 10)
    amplitude_range: tuple[float, float] = (0.0, 0.0)
    frequency_range_hz: tuple[float, float] = (0.1, 0.5)
    phase_range_rad: tuple[float, float] = (0.0, 2.0 * math.pi)
    bias_range: tuple[float, float] = (0.0, 0.0)
    spectral_decay: float = 1.0


@configclass
class PlatformMotionStageCfg:
    """Structured motion stage for platform translation/rotation channels."""

    name: str = "track_xy"
    x: HarmonicAxisMotionCfg = HarmonicAxisMotionCfg()
    y: HarmonicAxisMotionCfg = HarmonicAxisMotionCfg()
    z: HarmonicAxisMotionCfg = HarmonicAxisMotionCfg()
    roll: HarmonicAxisMotionCfg = HarmonicAxisMotionCfg()
    pitch: HarmonicAxisMotionCfg = HarmonicAxisMotionCfg()
    yaw: HarmonicAxisMotionCfg = HarmonicAxisMotionCfg()
    max_linear_speed: float = 2.5
    max_linear_acceleration: float = 8.0
    max_angular_speed: float = 1.0
    max_angular_acceleration: float = 4.0


def add_platform_top_decal(
    env: "ManagerBasedEnv",
    env_ids: Sequence[int] | None,
    texture_path: str,
    platform_name: str = "platform",
    platform_size: tuple[float, float, float] = (1.0, 1.0, 0.2),
    decal_z_offset: float = 5.0e-4,
) -> None:
    """Create a thin textured quad on top of each per-env platform."""
    del env_ids  # startup event applies globally

    texture_file = Path(texture_path).expanduser()
    if not texture_file.is_file():
        print(
            "[WARN][heave_landing_mpc] Platform texture PNG not found at "
            f"'{texture_file}'. Platform will load without top decal."
        )
        return

    from pxr import Sdf, UsdGeom, UsdShade

    stage = env.scene.stage
    material_path = Sdf.Path("/World/Looks/platform_top_material")
    material = UsdShade.Material.Define(stage, material_path)

    pbr_shader = UsdShade.Shader.Define(stage, material_path.AppendPath("PreviewSurface"))
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    uv_reader = UsdShade.Shader.Define(stage, material_path.AppendPath("PrimvarReader_st"))
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex_shader = UsdShade.Shader.Define(stage, material_path.AppendPath("TopTexture"))
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_file)))
    tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
    tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex_shader.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")

    half_x = 0.5 * float(platform_size[0])
    half_y = 0.5 * float(platform_size[1])
    top_z = 0.5 * float(platform_size[2]) + float(decal_z_offset)

    for env_prim_path in env.scene.env_prim_paths:
        platform_path = Sdf.Path(f"{env_prim_path}/{platform_name}")
        if not stage.GetPrimAtPath(platform_path).IsValid():
            continue

        decal_mesh_path = platform_path.AppendPath("top_decal")
        decal_mesh = UsdGeom.Mesh.Define(stage, decal_mesh_path)
        decal_mesh.CreatePointsAttr(
            [
                (-half_x, -half_y, top_z),
                (half_x, -half_y, top_z),
                (half_x, half_y, top_z),
                (-half_x, half_y, top_z),
            ]
        )
        decal_mesh.CreateFaceVertexCountsAttr([4])
        decal_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        decal_mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 4)
        decal_mesh.SetNormalsInterpolation("vertex")
        decal_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        primvars_api = UsdGeom.PrimvarsAPI(decal_mesh)
        st_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st_primvar.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        UsdShade.MaterialBindingAPI(decal_mesh.GetPrim()).Bind(material)


class SampleHeaveLandingDomainRandomization(ManagerTermBase):
    """Sample per-environment randomization flags/parameters and apply physics-side mass noise."""

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.rand_cfg: HeaveLandingDomainRandomizationCfg = cfg.params["rand_cfg"]
        self.mass_asset_cfg: SceneEntityCfg = cfg.params["mass_asset_cfg"]
        self.asset: Articulation = env.scene[self.mass_asset_cfg.name]
        self.runtime_state = HeaveLandingDomainRandomizationState(self.rand_cfg, self.num_envs, self.device)
        # heave_landing_mpc-specific runtime slot.
        setattr(env, "_heave_landing_domain_randomization", self.runtime_state)
        # Legacy key retained for compatibility with older code paths.
        setattr(env, "_vanilla_domain_randomization", self.runtime_state)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        rand_cfg: HeaveLandingDomainRandomizationCfg | None = None,
        mass_asset_cfg: SceneEntityCfg | None = None,
    ) -> None:
        del env, rand_cfg, mass_asset_cfg
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        self.runtime_state.sample_envs(env_ids_tensor)
        self._apply_mass_noise(env_ids_tensor)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def _apply_mass_noise(self, env_ids: torch.Tensor) -> None:
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        if self.mass_asset_cfg.body_ids == slice(None):
            body_ids_cpu = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids_cpu = torch.tensor(self.mass_asset_cfg.body_ids, dtype=torch.int, device="cpu")
        if body_ids_cpu.numel() == 0:
            return

        masses = self.asset.root_physx_view.get_masses()
        masses[env_ids_cpu[:, None], body_ids_cpu] = self.asset.data.default_mass[env_ids_cpu[:, None], body_ids_cpu].clone()

        mass_delta = self.runtime_state.mass_delta_kg[env_ids].to(device="cpu").unsqueeze(-1)
        masses[env_ids_cpu[:, None], body_ids_cpu] += mass_delta / float(body_ids_cpu.numel())
        masses = torch.clamp(masses, min=1.0e-6)
        self.asset.root_physx_view.set_masses(masses, env_ids_cpu)

        ratios = masses[env_ids_cpu[:, None], body_ids_cpu] / self.asset.data.default_mass[env_ids_cpu[:, None], body_ids_cpu]
        inertias = self.asset.root_physx_view.get_inertias()
        inertias[env_ids_cpu[:, None], body_ids_cpu] = (
            self.asset.data.default_inertia[env_ids_cpu[:, None], body_ids_cpu] * ratios[..., None]
        )
        self.asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


# Backward-compatible alias for older task configs.
SampleVanillaDomainRandomization = SampleHeaveLandingDomainRandomization


class CSVHeavePlatformMotion(ManagerTermBase):
    """Drive platform heave from recorded CSV traces, sampled per environment at reset."""

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        env._csv_heave_platform_motion = self
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.dataset_dir = Path(cfg.params["dataset_dir"]).expanduser()
        self.sample_rate_hz = float(cfg.params.get("sample_rate_hz", 20.0))
        self.min_remaining_s = float(cfg.params.get("min_remaining_s", 60.0))
        self.scale = float(cfg.params.get("scale", 1.0))
        self.bias_m = float(cfg.params.get("bias_m", 1.5))
        self.randomize_bias = bool(cfg.params.get("randomize_bias", False))
        self.bias_range_m = tuple(float(x) for x in cfg.params.get("bias_range_m", (0.5, 2.5)))
        self.stationary_env_probability = float(cfg.params.get("stationary_env_probability", 0.0))
        self.platform: RigidObject = env.scene[self.asset_cfg.name]

        if self.sample_rate_hz <= 0.0:
            raise ValueError("CSV heave sample_rate_hz must be positive.")
        if self.min_remaining_s < 0.0:
            raise ValueError("CSV heave min_remaining_s must be non-negative.")
        if len(self.bias_range_m) != 2:
            raise ValueError("CSV heave bias_range_m must contain exactly two values.")
        if self.bias_range_m[0] > self.bias_range_m[1]:
            raise ValueError("CSV heave bias_range_m min must be <= max.")

        csv_paths = sorted(self.dataset_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV datasets found in '{self.dataset_dir}'.")

        self._datasets: list[torch.Tensor] = []
        self._max_start_index = torch.empty((len(csv_paths),), device=self.device, dtype=torch.long)
        min_remaining_samples = int(math.ceil(self.min_remaining_s * self.sample_rate_hz))

        for dataset_id, csv_path in enumerate(csv_paths):
            with csv_path.open("r", encoding="utf-8") as file:
                values = [float(line.strip()) for line in file if line.strip()]
            if not values:
                raise ValueError(f"CSV dataset '{csv_path}' is empty.")
            if len(values) <= min_remaining_samples:
                raise ValueError(
                    f"CSV dataset '{csv_path}' has only {len(values)} samples; need more than {min_remaining_samples} "
                    "to leave at least 1 minute remaining after the chosen start."
                )
            dataset = torch.tensor(values, device=self.device, dtype=torch.float32)
            self._datasets.append(dataset)
            self._max_start_index[dataset_id] = len(values) - min_remaining_samples - 1

        self._dataset_ids = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._start_indices = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._episode_start_time_s = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._bias_m = torch.full((self.num_envs,), self.bias_m, device=self.device, dtype=torch.float32)
        self._stationary_env_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        current_time_s = self._current_time_s()
        dataset_ids = torch.randint(len(self._datasets), (env_ids_tensor.numel(),), device=self.device)
        sampled_start_indices = torch.empty_like(dataset_ids)
        for dataset_id in range(len(self._datasets)):
            mask = dataset_ids == dataset_id
            if torch.any(mask):
                max_start_index = int(self._max_start_index[dataset_id].item())
                sampled_start_indices[mask] = torch.randint(
                    max_start_index + 1, (int(mask.sum().item()),), device=self.device
                )

        self._dataset_ids[env_ids_tensor] = dataset_ids
        self._start_indices[env_ids_tensor] = sampled_start_indices
        self._episode_start_time_s[env_ids_tensor] = current_time_s
        self._stationary_env_mask[env_ids_tensor] = False
        if self.randomize_bias:
            bias_min_m, bias_max_m = self.bias_range_m
            self._bias_m[env_ids_tensor] = torch.empty(
                env_ids_tensor.numel(), device=self.device, dtype=torch.float32
            ).uniform_(bias_min_m, bias_max_m)
        else:
            self._bias_m[env_ids_tensor] = self.bias_m

        if self.stationary_env_probability > 0.0:
            stationary_draw = torch.rand(env_ids_tensor.numel(), device=self.device) < self.stationary_env_probability
            self._stationary_env_mask[env_ids_tensor] = stationary_draw

        self._write_motion_to_sim(env_ids_tensor, current_time_s)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
        dataset_dir: str | None = None,
        sample_rate_hz: float | None = None,
        min_remaining_s: float | None = None,
        scale: float | None = None,
        bias_m: float | None = None,
        randomize_bias: bool | None = None,
        bias_range_m: tuple[float, float] | None = None,
        stationary_env_probability: float | None = None,
    ) -> None:
        del (
            env,
            asset_cfg,
            dataset_dir,
            sample_rate_hz,
            min_remaining_s,
            scale,
            bias_m,
            randomize_bias,
            bias_range_m,
            stationary_env_probability,
        )
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return
        self._write_motion_to_sim(env_ids_tensor, self._current_time_s())

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def _write_motion_to_sim(self, env_ids: torch.Tensor, time_s: float) -> None:
        default_state = self.platform.data.default_root_state[env_ids]
        base_pos_local = default_state[:, 0:3]
        base_quat = default_state[:, 3:7]

        dataset_ids = self._dataset_ids[env_ids]
        start_indices = self._start_indices[env_ids]
        elapsed_samples = torch.clamp(
            (time_s - self._episode_start_time_s[env_ids]) * self.sample_rate_hz,
            min=0.0,
        )

        z_values = torch.zeros((env_ids.numel(),), device=self.device, dtype=base_pos_local.dtype)
        z_velocities = torch.zeros_like(z_values)

        for dataset_id, dataset in enumerate(self._datasets):
            mask = dataset_ids == dataset_id
            if not torch.any(mask):
                continue

            dataset_indices = torch.nonzero(mask, as_tuple=False).flatten()
            sample_positions = start_indices[mask].to(torch.float32) + elapsed_samples[mask]
            max_valid_position = float(dataset.numel() - 1)
            sample_positions = torch.clamp(sample_positions, max=max_valid_position)

            lower_idx = torch.floor(sample_positions).to(torch.long)
            upper_idx = torch.clamp(lower_idx + 1, max=dataset.numel() - 1)
            alpha = sample_positions - lower_idx.to(sample_positions.dtype)

            lower_values = dataset[lower_idx]
            upper_values = dataset[upper_idx]

            z_values[dataset_indices] = torch.lerp(lower_values, upper_values, alpha)
            z_velocities[dataset_indices] = (upper_values - lower_values) * self.sample_rate_hz

        sampled_bias = self._bias_m[env_ids].to(dtype=z_values.dtype)
        z_offset = sampled_bias + self.scale * z_values
        z_velocity = self.scale * z_velocities

        stationary_mask = self._stationary_env_mask[env_ids]
        z_offset[stationary_mask] = sampled_bias[stationary_mask]
        z_velocity[stationary_mask] = 0.0

        pos_local = base_pos_local.clone()
        pos_local[:, 2] += z_offset
        pos_world = pos_local + self._env.scene.env_origins[env_ids]
        pose_world = torch.cat((pos_world, base_quat), dim=-1)

        velocity_world = torch.zeros((env_ids.numel(), 6), device=self.device, dtype=base_pos_local.dtype)
        velocity_world[:, 2] = z_velocity

        self.platform.write_root_pose_to_sim(pose_world, env_ids=env_ids)
        self.platform.write_root_velocity_to_sim(velocity_world, env_ids=env_ids)

    def future_heave(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
        horizon_s: float = 6.0,
        sample_rate_hz: float | None = None,
    ) -> torch.Tensor:
        """Return future heave offsets for the next horizon in seconds.

        The returned sequence is the future heave motion only, i.e. the scaled CSV values without the
        constant platform bias term. This keeps the signal focused on upcoming platform motion.
        """
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return torch.zeros((0, 0), device=self.device)

        output_rate_hz = self.sample_rate_hz if sample_rate_hz is None else float(sample_rate_hz)
        if output_rate_hz <= 0.0:
            raise ValueError("future_heave sample_rate_hz must be positive.")
        if horizon_s <= 0.0:
            raise ValueError("future_heave horizon_s must be positive.")

        num_future_steps = int(round(horizon_s * output_rate_hz))
        if num_future_steps <= 0:
            raise ValueError("future_heave horizon results in zero future samples.")

        future_heave, _ = self._sample_future_trace(env_ids_tensor, horizon_s=horizon_s, sample_rate_hz=output_rate_hz)
        return future_heave

    def future_world_z(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
        horizon_s: float = 6.0,
        sample_rate_hz: float | None = None,
    ) -> torch.Tensor:
        """Return future platform world-frame z samples for the next horizon in seconds."""
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return torch.zeros((0, 0), device=self.device)

        output_rate_hz = self.sample_rate_hz if sample_rate_hz is None else float(sample_rate_hz)
        if output_rate_hz <= 0.0:
            raise ValueError("future_world_z sample_rate_hz must be positive.")
        if horizon_s <= 0.0:
            raise ValueError("future_world_z horizon_s must be positive.")

        future_heave, num_future_steps = self._sample_future_trace(
            env_ids_tensor, horizon_s=horizon_s, sample_rate_hz=output_rate_hz
        )
        default_root_state = self.platform.data.default_root_state[env_ids_tensor]
        base_world_z = default_root_state[:, 2] + self._env.scene.env_origins[env_ids_tensor, 2]
        sampled_bias = self._bias_m[env_ids_tensor].to(dtype=future_heave.dtype)
        return base_world_z[:, None] + sampled_bias[:, None] + future_heave

    def _sample_future_trace(
        self,
        env_ids_tensor: torch.Tensor,
        horizon_s: float,
        sample_rate_hz: float,
    ) -> tuple[torch.Tensor, int]:
        num_future_steps = int(round(horizon_s * sample_rate_hz))
        if num_future_steps <= 0:
            raise ValueError("future trace horizon results in zero future samples.")

        current_time_s = self._current_time_s()
        future_times_s = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.float32) / sample_rate_hz
        )
        current_sample_position = (
            self._start_indices[env_ids_tensor].to(torch.float32)
            + (current_time_s - self._episode_start_time_s[env_ids_tensor]) * self.sample_rate_hz
        )
        sample_positions = current_sample_position[:, None] + future_times_s[None, :] * self.sample_rate_hz

        future_heave = torch.zeros(
            (env_ids_tensor.numel(), num_future_steps),
            device=self.device,
            dtype=torch.float32,
        )
        dataset_ids = self._dataset_ids[env_ids_tensor]

        for dataset_id, dataset in enumerate(self._datasets):
            mask = dataset_ids == dataset_id
            if not torch.any(mask):
                continue

            env_mask_indices = torch.nonzero(mask, as_tuple=False).flatten()
            sample_positions_subset = torch.clamp(sample_positions[mask], max=float(dataset.numel() - 1))

            lower_idx = torch.floor(sample_positions_subset).to(torch.long)
            upper_idx = torch.clamp(lower_idx + 1, max=dataset.numel() - 1)
            alpha = sample_positions_subset - lower_idx.to(sample_positions_subset.dtype)

            lower_values = dataset[lower_idx]
            upper_values = dataset[upper_idx]
            future_heave[env_mask_indices] = torch.lerp(lower_values, upper_values, alpha)

        future_heave = self.scale * future_heave
        stationary_mask = self._stationary_env_mask[env_ids_tensor]
        future_heave[stationary_mask] = 0.0
        return future_heave, num_future_steps

    def _current_time_s(self) -> float:
        return float(getattr(self._env, "common_step_counter", 0)) * float(self._env.step_dt)


class MultiSinePlatformMotion(ManagerTermBase):
    """Sample a per-episode multi-sine motion profile and apply it at interval updates."""

    _CHANNEL_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
    _TRANSLATION_IDS = (0, 1, 2)
    _ROTATION_IDS = (3, 4, 5)

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.stage_cfg: PlatformMotionStageCfg = cfg.params["stage_cfg"]
        self.stationary_env_probability = float(cfg.params.get("stationary_env_probability", 0.0))
        self.platform: RigidObject = env.scene[self.asset_cfg.name]

        self._max_terms = max(
            getattr(self.stage_cfg, channel_name).num_terms_range[1]
            for channel_name in self._CHANNEL_NAMES
            if getattr(self.stage_cfg, channel_name).enabled
        )
        if self._max_terms <= 0:
            raise ValueError("Platform motion stage must enable at least one channel with a positive term count.")

        shape = (self.num_envs, len(self._CHANNEL_NAMES), self._max_terms)
        self._amplitudes = torch.zeros(shape, device=self.device)
        self._omegas = torch.zeros(shape, device=self.device)
        self._phases = torch.zeros(shape, device=self.device)
        self._bias = torch.zeros((self.num_envs, len(self._CHANNEL_NAMES)), device=self.device)
        self._stationary_env_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        self._amplitudes[env_ids_tensor] = 0.0
        self._omegas[env_ids_tensor] = 0.0
        self._phases[env_ids_tensor] = 0.0
        self._bias[env_ids_tensor] = 0.0
        self._stationary_env_mask[env_ids_tensor] = False

        if self.stationary_env_probability > 0.0:
            stationary_draw = torch.rand(env_ids_tensor.numel(), device=self.device) < self.stationary_env_probability
            self._stationary_env_mask[env_ids_tensor] = stationary_draw

        for channel_id, channel_name in enumerate(self._CHANNEL_NAMES):
            self._sample_channel(env_ids_tensor, channel_id, getattr(self.stage_cfg, channel_name))

        self._apply_stage_limits(env_ids_tensor)
        stationary_env_ids = env_ids_tensor[self._stationary_env_mask[env_ids_tensor]]
        if stationary_env_ids.numel() > 0:
            self._amplitudes[stationary_env_ids] = 0.0
            self._omegas[stationary_env_ids] = 0.0
            self._phases[stationary_env_ids] = 0.0
            self._bias[stationary_env_ids] = 0.0
        self._write_motion_to_sim(env_ids_tensor, self._current_time_s())

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
        stage_cfg: PlatformMotionStageCfg | None = None,
        stationary_env_probability: float | None = None,
    ) -> None:
        del env, asset_cfg, stage_cfg, stationary_env_probability
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return
        self._write_motion_to_sim(env_ids_tensor, self._current_time_s())

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def _sample_channel(self, env_ids: torch.Tensor, channel_id: int, channel_cfg: HarmonicAxisMotionCfg) -> None:
        if not channel_cfg.enabled:
            return

        min_terms, max_terms = channel_cfg.num_terms_range
        if min_terms < 1 or max_terms < min_terms:
            raise ValueError(
                f"Invalid num_terms_range={channel_cfg.num_terms_range} for channel '{self._CHANNEL_NAMES[channel_id]}'."
            )

        num_envs = env_ids.numel()
        term_ids = torch.arange(self._max_terms, device=self.device).unsqueeze(0)
        num_terms = torch.randint(min_terms, max_terms + 1, (num_envs, 1), device=self.device)
        active_mask = term_ids < num_terms

        amplitudes = self._sample_uniform(channel_cfg.amplitude_range, (num_envs, self._max_terms))
        frequencies_hz = self._sample_uniform(channel_cfg.frequency_range_hz, (num_envs, self._max_terms))
        phases = self._sample_uniform(channel_cfg.phase_range_rad, (num_envs, self._max_terms))
        bias = self._sample_uniform(channel_cfg.bias_range, (num_envs,))

        frequencies_hz, order = torch.sort(frequencies_hz, dim=1)
        amplitudes = torch.gather(amplitudes, 1, order)
        phases = torch.gather(phases, 1, order)

        if channel_cfg.spectral_decay > 0.0:
            decay = torch.pow(
                torch.arange(1, self._max_terms + 1, device=self.device, dtype=amplitudes.dtype),
                channel_cfg.spectral_decay,
            )
            amplitudes = amplitudes / decay.unsqueeze(0)

        amplitudes = amplitudes * active_mask
        omegas = 2.0 * math.pi * frequencies_hz * active_mask
        phases = phases * active_mask

        self._amplitudes[env_ids, channel_id] = amplitudes
        self._omegas[env_ids, channel_id] = omegas
        self._phases[env_ids, channel_id] = phases
        self._bias[env_ids, channel_id] = bias

    def _sample_uniform(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        lower, upper = value_range
        return torch.rand(shape, device=self.device) * (upper - lower) + lower

    def _apply_stage_limits(self, env_ids: torch.Tensor) -> None:
        amplitudes = self._amplitudes[env_ids]
        omegas = self._omegas[env_ids]

        translation_ids = list(self._TRANSLATION_IDS)
        angular_ids = list(self._ROTATION_IDS)

        lin_speed_bound = torch.linalg.norm(torch.sum(torch.abs(amplitudes[:, translation_ids] * omegas[:, translation_ids]), dim=-1), dim=1)
        lin_acc_bound = torch.linalg.norm(
            torch.sum(torch.abs(amplitudes[:, translation_ids] * torch.square(omegas[:, translation_ids])), dim=-1),
            dim=1,
        )
        ang_speed_bound = torch.linalg.norm(torch.sum(torch.abs(amplitudes[:, angular_ids] * omegas[:, angular_ids]), dim=-1), dim=1)
        ang_acc_bound = torch.linalg.norm(
            torch.sum(torch.abs(amplitudes[:, angular_ids] * torch.square(omegas[:, angular_ids])), dim=-1),
            dim=1,
        )

        lin_scale = torch.ones_like(lin_speed_bound)
        if self.stage_cfg.max_linear_speed > 0.0:
            lin_scale = torch.minimum(
                lin_scale,
                self.stage_cfg.max_linear_speed / torch.clamp(lin_speed_bound, min=1.0e-6),
            )
        if self.stage_cfg.max_linear_acceleration > 0.0:
            lin_scale = torch.minimum(
                lin_scale,
                self.stage_cfg.max_linear_acceleration / torch.clamp(lin_acc_bound, min=1.0e-6),
            )

        ang_scale = torch.ones_like(ang_speed_bound)
        if self.stage_cfg.max_angular_speed > 0.0:
            ang_scale = torch.minimum(
                ang_scale,
                self.stage_cfg.max_angular_speed / torch.clamp(ang_speed_bound, min=1.0e-6),
            )
        if self.stage_cfg.max_angular_acceleration > 0.0:
            ang_scale = torch.minimum(
                ang_scale,
                self.stage_cfg.max_angular_acceleration / torch.clamp(ang_acc_bound, min=1.0e-6),
            )

        lin_scale = torch.clamp(lin_scale, max=1.0)[:, None, None]
        ang_scale = torch.clamp(ang_scale, max=1.0)[:, None, None]
        self._amplitudes[env_ids[:, None], translation_ids, :] = (
            self._amplitudes[env_ids[:, None], translation_ids, :] * lin_scale
        )
        self._amplitudes[env_ids[:, None], angular_ids, :] = (
            self._amplitudes[env_ids[:, None], angular_ids, :] * ang_scale
        )

    def _evaluate_channels(self, env_ids: torch.Tensor, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        amplitudes = self._amplitudes[env_ids]
        omegas = self._omegas[env_ids]
        phases = self._phases[env_ids]

        phase_t = omegas * time_s + phases
        channel_values = self._bias[env_ids] + torch.sum(amplitudes * torch.sin(phase_t), dim=-1)
        channel_rates = torch.sum(amplitudes * omegas * torch.cos(phase_t), dim=-1)
        return channel_values, channel_rates

    def _evaluate_orientation(
        self,
        env_ids: torch.Tensor,
        time_s: float,
        base_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel_values, _ = self._evaluate_channels(env_ids, time_s)
        roll = channel_values[:, 3]
        pitch = channel_values[:, 4]
        yaw = channel_values[:, 5]
        quat = math_utils.quat_mul(base_quat, math_utils.quat_from_euler_xyz(roll, pitch, yaw))

        dt = max(float(self._env.step_dt), 1.0e-5)
        prev_values, _ = self._evaluate_channels(env_ids, time_s - 0.5 * dt)
        next_values, _ = self._evaluate_channels(env_ids, time_s + 0.5 * dt)
        quat_prev = math_utils.quat_mul(
            base_quat,
            math_utils.quat_from_euler_xyz(prev_values[:, 3], prev_values[:, 4], prev_values[:, 5]),
        )
        quat_next = math_utils.quat_mul(
            base_quat,
            math_utils.quat_from_euler_xyz(next_values[:, 3], next_values[:, 4], next_values[:, 5]),
        )
        quat_delta = math_utils.quat_mul(quat_next, math_utils.quat_inv(quat_prev))
        angular_velocity = math_utils.axis_angle_from_quat(quat_delta) / dt
        return quat, angular_velocity

    def _write_motion_to_sim(self, env_ids: torch.Tensor, time_s: float) -> None:
        default_state = self.platform.data.default_root_state[env_ids]
        base_pos_local = default_state[:, 0:3]
        base_quat = default_state[:, 3:7]

        channel_values, channel_rates = self._evaluate_channels(env_ids, time_s)
        pos_local = base_pos_local + channel_values[:, :3]
        quat_local, angular_velocity = self._evaluate_orientation(env_ids, time_s, base_quat)

        pos_world = pos_local + self._env.scene.env_origins[env_ids]
        pose_world = torch.cat((pos_world, quat_local), dim=-1)
        velocity_world = torch.cat((channel_rates[:, :3], angular_velocity), dim=-1)

        self.platform.write_root_pose_to_sim(pose_world, env_ids=env_ids)
        self.platform.write_root_velocity_to_sim(velocity_world, env_ids=env_ids)

    def _current_time_s(self) -> float:
        return float(getattr(self._env, "common_step_counter", 0)) * float(self._env.step_dt)
