"""Marker-frame views of platform state; rigid-body origins remain unchanged."""
from types import SimpleNamespace

import torch
from isaaclab.utils import math as math_utils


def platform_reference_data(env, name="platform"):
    """World pose/twist of the authored PNG, for task observations and objectives.

    Position is the texture center; orientation follows its UV axes. Linear
    velocity includes omega cross body-to-marker offset. This view does not
    subtract vehicle gear height or mutate the asset's physical root state.
    Reset/motion writers must continue using the rigid-body root, not this view.
    """
    asset = env.scene[name]
    data = asset.data
    cache = getattr(env, "_platform_marker_calibration", None)
    if cache is None:
        cache = env._platform_marker_calibration = {}
    if name not in cache:
        from uav_rl.transfer.marker_reference import marker_local_pose

        path = f"{env.scene.env_prim_paths[0]}/{name}"
        stage = env.scene.stage
        if stage.GetPrimAtPath(path + "/top_decal").IsValid():
            position, quat_xyzw = marker_local_pose(stage, path)
            cache[name] = (
                torch.as_tensor(position, device=data.root_pos_w.device, dtype=data.root_pos_w.dtype),
                torch.as_tensor(quat_xyzw[[3, 0, 1, 2]], device=data.root_pos_w.device, dtype=data.root_pos_w.dtype),
            )
        else:
            # Observation shapes are probed before the startup decal event runs.
            params = env.cfg.events.add_platform_top_decal.params
            height = 0.5 * params["platform_size"][2] + params.get("decal_z_offset", 5.0e-4)
            position = data.root_pos_w.new_tensor([0.0, 0.0, height])
            quat = data.root_pos_w.new_tensor([1.0, 0.0, 0.0, 0.0])
            return _reference(data, position, quat)
    return _reference(data, *cache[name])


def _reference(data, position, quat):
    offset = math_utils.quat_apply(data.root_quat_w, position.expand_as(data.root_pos_w))
    return SimpleNamespace(
        root_pos_w=data.root_pos_w + offset,
        root_quat_w=math_utils.quat_mul(data.root_quat_w, quat.expand_as(data.root_quat_w)),
        root_lin_vel_w=data.root_lin_vel_w + torch.cross(data.root_ang_vel_w, offset, dim=-1),
        root_ang_vel_w=data.root_ang_vel_w,
    )
