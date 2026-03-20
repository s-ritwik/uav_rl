from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path

import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation


@dataclass
class HarmonicAxisMotionCfg:
    """Sampling ranges for one harmonic motion channel."""

    enabled: bool = False
    num_terms_range: tuple[int, int] = (2, 10)
    amplitude_range: tuple[float, float] = (0.0, 0.0)
    frequency_range_hz: tuple[float, float] = (0.1, 0.5)
    phase_range_rad: tuple[float, float] = (0.0, 2.0 * math.pi)
    bias_range: tuple[float, float] = (0.0, 0.0)
    spectral_decay: float = 1.0


@dataclass
class PlatformMotionStageCfg:
    """Structured platform motion family for transfer experiments."""

    name: str = "track_xy"
    x: HarmonicAxisMotionCfg = field(default_factory=HarmonicAxisMotionCfg)
    y: HarmonicAxisMotionCfg = field(default_factory=HarmonicAxisMotionCfg)
    z: HarmonicAxisMotionCfg = field(default_factory=HarmonicAxisMotionCfg)
    roll: HarmonicAxisMotionCfg = field(default_factory=HarmonicAxisMotionCfg)
    pitch: HarmonicAxisMotionCfg = field(default_factory=HarmonicAxisMotionCfg)
    yaw: HarmonicAxisMotionCfg = field(default_factory=HarmonicAxisMotionCfg)
    max_linear_speed: float = 2.5
    max_linear_acceleration: float = 8.0
    max_angular_speed: float = 1.0
    max_angular_acceleration: float = 4.0


TRACK_XY_STAGE = PlatformMotionStageCfg(
    name="track_xy",
    x=HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 10),
        amplitude_range=(0.05, 0.35),
        frequency_range_hz=(0.05, 0.35),
        spectral_decay=1.0,
    ),
    y=HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 10),
        amplitude_range=(0.05, 0.35),
        frequency_range_hz=(0.05, 0.35),
        spectral_decay=1.0,
    ),
    max_linear_speed=2.0,
    max_linear_acceleration=5.0,
)

TRACK_XY_ROLL_PITCH_STAGE = replace(
    TRACK_XY_STAGE,
    name="track_xy_roll_pitch",
    roll=HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        spectral_decay=1.0,
    ),
    pitch=HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        spectral_decay=1.0,
    ),
    max_angular_speed=0.75,
    max_angular_acceleration=2.5,
)

TRACK_XY_ROLL_PITCH_HEAVE_STAGE = replace(
    TRACK_XY_ROLL_PITCH_STAGE,
    name="track_xy_roll_pitch_heave",
    z=HarmonicAxisMotionCfg(
        enabled=True,
        num_terms_range=(2, 6),
        amplitude_range=(0.02, 0.10),
        frequency_range_hz=(0.05, 0.25),
        spectral_decay=1.0,
    ),
    max_linear_speed=2.25,
    max_linear_acceleration=6.0,
)


def get_stage_preset(name: str) -> PlatformMotionStageCfg:
    presets = {
        TRACK_XY_STAGE.name: TRACK_XY_STAGE,
        TRACK_XY_ROLL_PITCH_STAGE.name: TRACK_XY_ROLL_PITCH_STAGE,
        TRACK_XY_ROLL_PITCH_HEAVE_STAGE.name: TRACK_XY_ROLL_PITCH_HEAVE_STAGE,
    }
    if name not in presets:
        raise KeyError(f"Unknown platform motion stage '{name}'. Available: {sorted(presets.keys())}")
    return presets[name]


@dataclass
class PlatformState:
    position: np.ndarray
    quat_wxyz: np.ndarray
    quat_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


class MultiSineMotionProfile:
    """Single-platform version of the structured multi-sine generator used in vanilla."""

    CHANNEL_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
    TRANSLATION_IDS = (0, 1, 2)
    ROTATION_IDS = (3, 4, 5)

    def __init__(
        self,
        stage_cfg: PlatformMotionStageCfg,
        *,
        base_position: tuple[float, float, float] = (0.0, 0.0, 0.1),
        rng_seed: int | None = None,
    ):
        self.stage_cfg = stage_cfg
        self.base_position = np.asarray(base_position, dtype=np.float64)
        self.rng = np.random.default_rng(rng_seed)
        self._max_terms = max(
            getattr(self.stage_cfg, channel_name).num_terms_range[1]
            for channel_name in self.CHANNEL_NAMES
            if getattr(self.stage_cfg, channel_name).enabled
        )
        if self._max_terms <= 0:
            raise ValueError("At least one motion channel must be enabled.")

        self.amplitudes = np.zeros((len(self.CHANNEL_NAMES), self._max_terms), dtype=np.float64)
        self.omegas = np.zeros_like(self.amplitudes)
        self.phases = np.zeros_like(self.amplitudes)
        self.bias = np.zeros((len(self.CHANNEL_NAMES),), dtype=np.float64)

    def sample(self) -> None:
        self.amplitudes.fill(0.0)
        self.omegas.fill(0.0)
        self.phases.fill(0.0)
        self.bias.fill(0.0)

        for channel_id, channel_name in enumerate(self.CHANNEL_NAMES):
            self._sample_channel(channel_id, getattr(self.stage_cfg, channel_name))

        self._apply_stage_limits()

    def _sample_channel(self, channel_id: int, channel_cfg: HarmonicAxisMotionCfg) -> None:
        if not channel_cfg.enabled:
            return

        min_terms, max_terms = channel_cfg.num_terms_range
        if min_terms < 1 or max_terms < min_terms:
            raise ValueError(f"Invalid num_terms_range {channel_cfg.num_terms_range} for channel {self.CHANNEL_NAMES[channel_id]}")

        num_terms = int(self.rng.integers(min_terms, max_terms + 1))
        amplitudes = self.rng.uniform(*channel_cfg.amplitude_range, size=self._max_terms)
        frequencies_hz = self.rng.uniform(*channel_cfg.frequency_range_hz, size=self._max_terms)
        phases = self.rng.uniform(*channel_cfg.phase_range_rad, size=self._max_terms)
        bias = float(self.rng.uniform(*channel_cfg.bias_range))

        order = np.argsort(frequencies_hz)
        amplitudes = amplitudes[order]
        frequencies_hz = frequencies_hz[order]
        phases = phases[order]

        if channel_cfg.spectral_decay > 0.0:
            amplitudes = amplitudes / np.power(np.arange(1, self._max_terms + 1, dtype=np.float64), channel_cfg.spectral_decay)

        amplitudes[num_terms:] = 0.0
        frequencies_hz[num_terms:] = 0.0
        phases[num_terms:] = 0.0

        self.amplitudes[channel_id] = amplitudes
        self.omegas[channel_id] = 2.0 * math.pi * frequencies_hz
        self.phases[channel_id] = phases
        self.bias[channel_id] = bias

    def _apply_stage_limits(self) -> None:
        translation_amp = self.amplitudes[list(self.TRANSLATION_IDS)]
        translation_omega = self.omegas[list(self.TRANSLATION_IDS)]
        rotation_amp = self.amplitudes[list(self.ROTATION_IDS)]
        rotation_omega = self.omegas[list(self.ROTATION_IDS)]

        lin_speed_bound = np.linalg.norm(np.sum(np.abs(translation_amp * translation_omega), axis=-1))
        lin_acc_bound = np.linalg.norm(np.sum(np.abs(translation_amp * np.square(translation_omega)), axis=-1))
        ang_speed_bound = np.linalg.norm(np.sum(np.abs(rotation_amp * rotation_omega), axis=-1))
        ang_acc_bound = np.linalg.norm(np.sum(np.abs(rotation_amp * np.square(rotation_omega)), axis=-1))

        lin_scale = 1.0
        if self.stage_cfg.max_linear_speed > 0.0:
            lin_scale = min(lin_scale, self.stage_cfg.max_linear_speed / max(lin_speed_bound, 1.0e-6))
        if self.stage_cfg.max_linear_acceleration > 0.0:
            lin_scale = min(lin_scale, self.stage_cfg.max_linear_acceleration / max(lin_acc_bound, 1.0e-6))

        ang_scale = 1.0
        if self.stage_cfg.max_angular_speed > 0.0:
            ang_scale = min(ang_scale, self.stage_cfg.max_angular_speed / max(ang_speed_bound, 1.0e-6))
        if self.stage_cfg.max_angular_acceleration > 0.0:
            ang_scale = min(ang_scale, self.stage_cfg.max_angular_acceleration / max(ang_acc_bound, 1.0e-6))

        self.amplitudes[list(self.TRANSLATION_IDS)] *= min(lin_scale, 1.0)
        self.amplitudes[list(self.ROTATION_IDS)] *= min(ang_scale, 1.0)

    def _evaluate_channels(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        phase_t = (self.omegas * time_s) + self.phases
        values = self.bias + np.sum(self.amplitudes * np.sin(phase_t), axis=-1)
        rates = np.sum(self.amplitudes * self.omegas * np.cos(phase_t), axis=-1)
        return values, rates

    def evaluate(self, time_s: float, dt: float) -> PlatformState:
        values, rates = self._evaluate_channels(time_s)
        position = self.base_position + values[:3]
        linear_velocity = rates[:3]

        rot = Rotation.from_euler("XYZ", values[3:6], degrees=False)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.asarray((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]), dtype=np.float64)

        prev_values, _ = self._evaluate_channels(time_s - 0.5 * dt)
        next_values, _ = self._evaluate_channels(time_s + 0.5 * dt)
        rot_prev = Rotation.from_euler("XYZ", prev_values[3:6], degrees=False)
        rot_next = Rotation.from_euler("XYZ", next_values[3:6], degrees=False)
        delta_rot = rot_next * rot_prev.inv()
        angular_velocity = delta_rot.as_rotvec() / max(dt, 1.0e-6)

        return PlatformState(
            position=position.astype(np.float32),
            quat_wxyz=quat_wxyz.astype(np.float32),
            quat_xyzw=quat_xyzw.astype(np.float32),
            linear_velocity=linear_velocity.astype(np.float32),
            angular_velocity=angular_velocity.astype(np.float32),
        )


class MovingPlatform:
    """Spawn and drive a kinematic platform with a top decal in standalone Isaac Sim."""

    def __init__(
        self,
        world,
        *,
        prim_path: str = "/World/platform",
        texture_path: str,
        physics_dt: float,
        stage_cfg: PlatformMotionStageCfg,
        rng_seed: int | None = None,
        size: tuple[float, float, float] = (1.0, 1.0, 0.2),
        base_position: tuple[float, float, float] = (0.0, 0.0, 0.1),
        add_top_decal: bool = True,
    ):
        self.world = world
        self.prim_path = prim_path
        self.texture_path = str(Path(texture_path).expanduser().resolve())
        self.physics_dt = float(physics_dt)
        self.size = tuple(float(x) for x in size)
        self.base_position = tuple(float(x) for x in base_position)
        self.profile = MultiSineMotionProfile(stage_cfg, base_position=self.base_position, rng_seed=rng_seed)
        self.current_state: PlatformState | None = None
        self._sim_time = 0.0

        self.prim = self.world.scene.add(
            DynamicCuboid(
                prim_path=self.prim_path,
                name="platform",
                position=np.asarray(self.base_position, dtype=np.float32),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                scale=np.asarray(self.size, dtype=np.float32),
                size=1.0,
                color=np.asarray((0.28, 0.28, 0.28), dtype=np.float32),
                mass=1.0,
            )
        )

        stage_prim = self.world.stage.GetPrimAtPath(self.prim_path)
        self._stage_prim = stage_prim
        rigid_body_api = UsdPhysics.RigidBodyAPI(stage_prim)
        if not rigid_body_api.GetKinematicEnabledAttr().IsValid():
            rigid_body_api.CreateKinematicEnabledAttr(True)
        rigid_body_api.GetKinematicEnabledAttr().Set(True)

        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage_prim)
        if not physx_rigid_body_api.GetDisableGravityAttr().IsValid():
            physx_rigid_body_api.CreateDisableGravityAttr(True)
        physx_rigid_body_api.GetDisableGravityAttr().Set(True)

        xformable = UsdGeom.Xformable(stage_prim)
        ordered_ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
        self._translate_op = ordered_ops.get("xformOp:translate")
        if self._translate_op is None:
            self._translate_op = xformable.AddTranslateOp()
        self._orient_op = ordered_ops.get("xformOp:orient")
        if self._orient_op is None:
            self._orient_op = xformable.AddOrientOp()
        self._decal_frame_path = f"{self.prim_path}/decal_frame"
        decal_frame = UsdGeom.Xform.Define(self.world.stage, self._decal_frame_path)
        decal_frame_xform = UsdGeom.Xformable(decal_frame.GetPrim())
        decal_scale_op = None
        for op in decal_frame_xform.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:scale":
                decal_scale_op = op
                break
        if decal_scale_op is None:
            decal_scale_op = decal_frame_xform.AddScaleOp()
        decal_scale_op.Set(
            Gf.Vec3f(
                float(1.0 / max(self.size[0], 1.0e-6)),
                float(1.0 / max(self.size[1], 1.0e-6)),
                float(1.0 / max(self.size[2], 1.0e-6)),
            )
        )

        if add_top_decal:
            self._add_top_decal()
        self.reset_profile()

    def reset_profile(self) -> None:
        self.profile.sample()
        self._sim_time = 0.0
        self.current_state = self.profile.evaluate(0.0, self.physics_dt)
        self._apply_state(self.current_state)

    def update(self, dt: float) -> None:
        self._sim_time += float(dt)
        self.current_state = self.profile.evaluate(self._sim_time, self.physics_dt)
        self._apply_state(self.current_state)

    def _apply_state(self, state: PlatformState) -> None:
        # Match the vanilla task's kinematic-platform pattern: PhysX owns the collision body,
        # while the motion profile remains the source of truth for platform velocity.
        self.prim.set_world_pose(position=state.position, orientation=state.quat_wxyz)
        self._translate_op.Set(Gf.Vec3d(float(state.position[0]), float(state.position[1]), float(state.position[2])))
        self._orient_op.Set(
            Gf.Quatd(
                float(state.quat_wxyz[0]),
                Gf.Vec3d(float(state.quat_wxyz[1]), float(state.quat_wxyz[2]), float(state.quat_wxyz[3])),
            )
        )

    def _add_top_decal(self) -> None:
        texture_file = Path(self.texture_path)
        if not texture_file.is_file():
            print(
                "[WARN][transfer] Platform texture PNG not found at "
                f"'{texture_file}'. Platform will load without top decal."
            )
            return

        stage = self.world.stage
        # The cube geometry is unit-sized in local coordinates and the parent prim carries the scale.
        half_x = 0.5
        half_y = 0.5
        # Keep the world-space offset above the top face large enough to avoid z-fighting.
        local_decal_z_offset = 5.0e-4 / max(self.size[2], 1.0e-6)
        top_z = 0.5 + local_decal_z_offset

        decal_mesh_path = Sdf.Path(f"{self.prim_path}/top_decal")
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
        decal_mesh.CreateDoubleSidedAttr(True)

        primvars_api = UsdGeom.PrimvarsAPI(decal_mesh)
        st_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st_primvar.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

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
        UsdShade.MaterialBindingAPI(decal_mesh.GetPrim()).Bind(material)
