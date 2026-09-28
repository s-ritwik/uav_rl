"""Exact PNG frame from the authored decal geometry and texture coordinates."""
from dataclasses import replace
import numpy as np
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation


def marker_pose(stage, platform_path):
    """Reference the published state to the PNG center using its USD mesh and UVs."""
    marker_path = f"{platform_path}/top_decal"
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(marker_path))
    if not mesh:
        raise RuntimeError(f"Cannot publish ArUco pose: missing decal mesh {marker_path}")

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    uv = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").ComputeFlattened(), dtype=np.float64)
    if points.shape != (4, 3) or indices.shape != (4,) or uv.shape != (4, 2):
        raise RuntimeError(f"Expected a textured quad at {marker_path}")

    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    corners_w = np.asarray([transform.Transform(Gf.Vec3d(*point)) for point in points[indices]])
    # The texture center is UV (0.5, 0.5), not the decal prim's transform origin.
    uv_to_world, _, rank, _ = np.linalg.lstsq(
        np.column_stack((uv - 0.5, np.ones(4))), corners_w, rcond=None
    )
    if rank != 3:
        raise RuntimeError(f"Degenerate texture coordinates at {marker_path}")
    x_axis, y_axis, marker_position = uv_to_world
    normal = np.cross(x_axis, y_axis)
    if min(np.linalg.norm(x_axis), np.linalg.norm(normal)) < 1.0e-12:
        raise RuntimeError(f"Degenerate decal geometry at {marker_path}")
    x_axis = x_axis / np.linalg.norm(x_axis)
    normal = normal / np.linalg.norm(normal)
    y_axis = np.cross(normal, x_axis)
    marker_quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, normal))).as_quat()
    return marker_position, marker_quat


def marker_local_pose(stage, platform_path):
    position, quat = marker_pose(stage, platform_path)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(platform_path))
    rotation = transform.RemoveScaleShear().ExtractRotationQuat()
    body_rotation = Rotation.from_quat([*rotation.GetImaginary(), rotation.GetReal()])
    offset = body_rotation.inv().apply(position - np.asarray(transform.ExtractTranslation()))
    return offset, (body_rotation.inv() * Rotation.from_quat(quat)).as_quat()


def platform_marker_state(stage, platform):
    state = platform.current_state
    if state is None:
        return None
    position, quat = marker_pose(stage, platform.prim_path)
    offset = position - state.position
    return replace(state, position=position, quat_xyzw=quat, quat_wxyz=quat[[3, 0, 1, 2]],
                   linear_velocity=state.linear_velocity + np.cross(state.angular_velocity, offset))
