from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
            "[WARN][vanilla] Platform texture PNG not found at "
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


def move_platform_sinusoidal(
    env: "ManagerBasedEnv",
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    amplitude_m: float = 1.0,
    frequency_hz: float = 0.3,
    axis: str = "x",
    phase_rad: float = 0.0,
    phase_per_env: bool = True,
    phase_span_rad: float = 2.0 * math.pi,
) -> None:
    """Move a rigid platform sinusoidally in environment-local frame."""
    platform: RigidObject = env.scene[asset_cfg.name]

    if env_ids is None or isinstance(env_ids, slice):
        env_ids_tensor = torch.arange(env.scene.num_envs, device=platform.device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        env_ids_tensor = env_ids.to(device=platform.device, dtype=torch.long)
    else:
        env_ids_tensor = torch.tensor(env_ids, device=platform.device, dtype=torch.long)

    if env_ids_tensor.numel() == 0:
        return

    axis_id = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if axis_id is None:
        raise ValueError(f"Unsupported platform motion axis '{axis}'. Use one of: x, y, z.")

    time_s = float(env.common_step_counter) * float(env.step_dt)
    omega_t = (2.0 * math.pi * float(frequency_hz) * time_s) + float(phase_rad)

    if phase_per_env:
        # Spread phase offsets across env IDs in [0, phase_span_rad).
        env_phase = float(phase_span_rad) * env_ids_tensor.to(dtype=torch.float32) / float(max(env.scene.num_envs, 1))
        offsets = float(amplitude_m) * torch.sin(omega_t + env_phase)
    else:
        shared_offset = float(amplitude_m) * math.sin(omega_t)
        offsets = torch.full((env_ids_tensor.numel(),), shared_offset, device=platform.device, dtype=torch.float32)

    default_state = platform.data.default_root_state[env_ids_tensor]
    pos_local = default_state[:, 0:3].clone()
    quat_local = default_state[:, 3:7]
    pos_local[:, axis_id] += offsets.to(dtype=pos_local.dtype)

    pos_world = pos_local + env.scene.env_origins[env_ids_tensor]
    pose_world = torch.cat((pos_world, quat_local), dim=-1)
    platform.write_root_pose_to_sim(pose_world, env_ids=env_ids_tensor)

    zero_vel = torch.zeros((env_ids_tensor.numel(), 6), device=platform.device)
    platform.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_tensor)
