from __future__ import annotations

import math

import torch

from isaaclab.utils import math as math_utils


class _VisionObservationState:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.last_step = -1
        self.filtered_pos = torch.zeros((num_envs, 3), device=self.device)
        self.filtered_vel = torch.zeros((num_envs, 3), device=self.device)
        self.filtered_quat = _identity_quat(num_envs, self.device)
        self.filtered_ang_vel = torch.zeros((num_envs, 3), device=self.device)
        self.last_seen_s = torch.full((num_envs,), -1.0e9, device=self.device)
        self.raw_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.filtered_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.visible_fraction = torch.zeros(num_envs, device=self.device)
        self.cached_rel_pos = torch.zeros((num_envs, 3), device=self.device)
        self.cached_rel_lin_vel = torch.zeros((num_envs, 3), device=self.device)
        self.cached_rel_quat = _identity_quat(num_envs, self.device)
        self.cached_rel_ang_vel = torch.zeros((num_envs, 3), device=self.device)
        self.cached_line_of_sight = torch.zeros((num_envs, 3), device=self.device)
        self.cached_status = torch.zeros((num_envs, 4), device=self.device)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.filtered_pos[env_ids] = 0.0
        self.filtered_vel[env_ids] = 0.0
        self.filtered_quat[env_ids] = _identity_quat(int(env_ids.numel()), self.device, dtype=self.filtered_quat.dtype)
        self.filtered_ang_vel[env_ids] = 0.0
        self.last_seen_s[env_ids] = -1.0e9
        self.raw_valid[env_ids] = False
        self.filtered_valid[env_ids] = False
        self.visible_fraction[env_ids] = 0.0
        self.cached_rel_pos[env_ids] = 0.0
        self.cached_rel_lin_vel[env_ids] = 0.0
        self.cached_rel_quat[env_ids] = _identity_quat(
            int(env_ids.numel()), self.device, dtype=self.cached_rel_quat.dtype
        )
        self.cached_rel_ang_vel[env_ids] = 0.0
        self.cached_line_of_sight[env_ids] = 0.0
        self.cached_status[env_ids] = 0.0


def _identity_quat(num_envs: int, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    quat = torch.zeros((num_envs, 4), device=device, dtype=dtype)
    quat[:, 0] = 1.0
    return quat


def _expand_vector(values: tuple[float, ...], num_envs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1)


def _integrate_quaternion(quat_wxyz: torch.Tensor, ang_vel: torch.Tensor, dt: float) -> torch.Tensor:
    delta_axis_angle = ang_vel * float(dt)
    delta_norm = torch.linalg.norm(delta_axis_angle, dim=-1)
    delta_axis = delta_axis_angle / delta_norm.unsqueeze(-1).clamp(min=1.0e-6)
    delta_quat = math_utils.quat_from_angle_axis(delta_norm, delta_axis)
    identity = _identity_quat(quat_wxyz.shape[0], quat_wxyz.device, dtype=quat_wxyz.dtype)
    delta_quat = torch.where((delta_norm > 1.0e-6).unsqueeze(-1), delta_quat, identity)
    return math_utils.quat_unique(math_utils.quat_mul(delta_quat, quat_wxyz))


def _apply_orientation_noise(quat_wxyz: torch.Tensor, std_rad: float, valid_mask: torch.Tensor) -> torch.Tensor:
    if std_rad <= 0.0 or not bool(torch.any(valid_mask)):
        return quat_wxyz
    axis = torch.randn((quat_wxyz.shape[0], 3), device=quat_wxyz.device, dtype=quat_wxyz.dtype)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1.0e-6)
    angle = torch.randn((quat_wxyz.shape[0],), device=quat_wxyz.device, dtype=quat_wxyz.dtype) * float(std_rad)
    delta_quat = math_utils.quat_from_angle_axis(angle, axis)
    noisy_quat = math_utils.quat_unique(math_utils.quat_mul(delta_quat, quat_wxyz))
    return torch.where(valid_mask.unsqueeze(-1), noisy_quat, quat_wxyz)


def _get_vision_state(env) -> _VisionObservationState:
    state = getattr(env, "_landing_sway_vision_observation_state", None)
    if state is None or state.num_envs != env.num_envs:
        state = _VisionObservationState(env.num_envs, env.device)
        setattr(env, "_landing_sway_vision_observation_state", state)
    return state


def _update_vision_cache(env) -> _VisionObservationState:
    state = _get_vision_state(env)
    step = int(getattr(env, "common_step_counter", -1))
    if state.last_step == step:
        return state

    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    state.reset_envs(reset_ids)

    vision_cfg = env.cfg.post_init_cfg.vision_observation
    robot = env.scene["robot"]
    platform = env.scene["platform"]
    dtype = robot.data.root_pos_w.dtype
    device = robot.data.root_pos_w.device
    num_envs = env.num_envs
    step_dt = float(env.step_dt)
    now_s = float(getattr(env, "common_step_counter", 0)) * step_dt

    camera_offset_b = _expand_vector(vision_cfg.camera_offset_body_m, num_envs, device, dtype)
    camera_quat_b = math_utils.normalize(
        _expand_vector(vision_cfg.camera_quat_body_wxyz, num_envs, device, dtype)
    )
    marker_offset_p = _expand_vector(vision_cfg.marker_offset_platform_m, num_envs, device, dtype)
    marker_quat_p = math_utils.normalize(
        _expand_vector(vision_cfg.marker_quat_platform_wxyz, num_envs, device, dtype)
    )

    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = math_utils.quat_unique(robot.data.root_quat_w)
    platform_pos_w = platform.data.root_pos_w
    platform_quat_w = math_utils.quat_unique(platform.data.root_quat_w)

    camera_pos_w, camera_quat_w = math_utils.combine_frame_transforms(
        robot_pos_w, robot_quat_w, camera_offset_b, camera_quat_b
    )
    marker_pos_w, marker_quat_w = math_utils.combine_frame_transforms(
        platform_pos_w, platform_quat_w, marker_offset_p, marker_quat_p
    )

    t_marker_cam, q_marker_cam = math_utils.subtract_frame_transforms(
        camera_pos_w, camera_quat_w, marker_pos_w, marker_quat_w
    )
    t_robot_marker, q_robot_marker = math_utils.subtract_frame_transforms(
        marker_pos_w, marker_quat_w, robot_pos_w, robot_quat_w
    )
    q_robot_marker = math_utils.quat_unique(q_robot_marker)

    half_size = float(vision_cfg.marker_size_m) * 0.5
    marker_corners = torch.tensor(
        [
            [-half_size, -half_size, 0.0],
            [half_size, -half_size, 0.0],
            [half_size, half_size, 0.0],
            [-half_size, half_size, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    rotated_corners = math_utils.quat_apply(
        q_marker_cam[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
        marker_corners.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3),
    ).reshape(num_envs, 4, 3)
    corners_cam = rotated_corners + t_marker_cam.unsqueeze(1)

    width = float(vision_cfg.image_width_px)
    height = float(vision_cfg.image_height_px)
    fx = 0.5 * width / math.tan(0.5 * math.radians(float(vision_cfg.horizontal_fov_deg)))
    fy = 0.5 * height / math.tan(0.5 * math.radians(float(vision_cfg.vertical_fov_deg)))
    cx = 0.5 * width
    cy = 0.5 * height

    depth = corners_cam[..., 2]
    u = fx * corners_cam[..., 0] / depth.clamp(min=1.0e-6) + cx
    v = fy * corners_cam[..., 1] / depth.clamp(min=1.0e-6) + cy
    corner_visible = (
        (depth > float(vision_cfg.min_depth_m))
        & (depth < float(vision_cfg.max_depth_m))
        & (u >= 0.0)
        & (u < width)
        & (v >= 0.0)
        & (v < height)
    )
    visible_fraction = corner_visible.to(dtype=dtype).mean(dim=-1)
    raw_valid = corner_visible.sum(dim=-1) >= int(vision_cfg.min_visible_corners)
    raw_valid &= (t_marker_cam[:, 2] > float(vision_cfg.min_depth_m)) & (
        t_marker_cam[:, 2] < float(vision_cfg.max_depth_m)
    )
    dropout_prob = float(vision_cfg.detection_dropout_prob)
    if dropout_prob > 0.0:
        raw_valid &= torch.rand(num_envs, device=device) >= dropout_prob

    measured_pos = t_robot_marker.clone()
    pos_noise_std = float(vision_cfg.position_noise_std_m)
    if pos_noise_std > 0.0 and bool(torch.any(raw_valid)):
        measured_pos = measured_pos + torch.randn_like(measured_pos) * pos_noise_std * raw_valid.unsqueeze(-1)
    measured_quat = _apply_orientation_noise(
        q_robot_marker,
        std_rad=float(vision_cfg.orientation_noise_std_rad),
        valid_mask=raw_valid,
    )

    prev_pos = state.filtered_pos.clone().to(dtype=dtype)
    prev_vel = state.filtered_vel.clone().to(dtype=dtype)
    prev_quat = state.filtered_quat.clone().to(dtype=dtype)
    prev_ang_vel = state.filtered_ang_vel.clone().to(dtype=dtype)
    prev_last_seen_s = state.last_seen_s.clone().to(dtype=dtype)
    prev_filtered_valid = state.filtered_valid.clone()

    timeout_s = float(vision_cfg.measurement_timeout_s)
    within_timeout = prev_filtered_valid & ((now_s - prev_last_seen_s) <= timeout_s)
    predicted_pos = prev_pos + prev_vel * step_dt
    predicted_quat = _integrate_quaternion(prev_quat, prev_ang_vel, step_dt)

    state.filtered_pos = torch.where(within_timeout.unsqueeze(-1), predicted_pos, prev_pos)
    state.filtered_quat = torch.where(within_timeout.unsqueeze(-1), predicted_quat, prev_quat)
    state.filtered_vel = torch.where(within_timeout.unsqueeze(-1), prev_vel, torch.zeros_like(prev_vel))
    state.filtered_ang_vel = torch.where(within_timeout.unsqueeze(-1), prev_ang_vel, torch.zeros_like(prev_ang_vel))

    if bool(torch.any(raw_valid)):
        pos_blend = float(vision_cfg.position_blend)
        vel_blend = float(vision_cfg.velocity_blend)
        ang_blend = float(vision_cfg.angular_velocity_blend)

        prior_valid_for_update = within_timeout & raw_valid
        raw_vel = torch.zeros_like(measured_pos)
        raw_vel[prior_valid_for_update] = (
            measured_pos[prior_valid_for_update] - predicted_pos[prior_valid_for_update]
        ) / step_dt

        quat_error = math_utils.quat_mul(measured_quat, math_utils.quat_inv(predicted_quat))
        raw_ang_vel = torch.zeros_like(measured_pos)
        raw_ang_vel[prior_valid_for_update] = (
            math_utils.axis_angle_from_quat(quat_error[prior_valid_for_update]) / step_dt
        )

        updated_pos = torch.where(
            prior_valid_for_update.unsqueeze(-1),
            torch.lerp(predicted_pos, measured_pos, pos_blend),
            measured_pos,
        )
        updated_vel = torch.where(
            prior_valid_for_update.unsqueeze(-1),
            torch.lerp(prev_vel, raw_vel, vel_blend),
            raw_vel,
        )
        updated_quat = measured_quat
        updated_ang_vel = torch.where(
            prior_valid_for_update.unsqueeze(-1),
            torch.lerp(prev_ang_vel, raw_ang_vel, ang_blend),
            raw_ang_vel,
        )

        state.filtered_pos = torch.where(raw_valid.unsqueeze(-1), updated_pos, state.filtered_pos)
        state.filtered_vel = torch.where(raw_valid.unsqueeze(-1), updated_vel, state.filtered_vel)
        state.filtered_quat = torch.where(raw_valid.unsqueeze(-1), updated_quat, state.filtered_quat)
        state.filtered_ang_vel = torch.where(raw_valid.unsqueeze(-1), updated_ang_vel, state.filtered_ang_vel)
        state.last_seen_s = torch.where(raw_valid, torch.full_like(prev_last_seen_s, now_s), state.last_seen_s)

    state.raw_valid = raw_valid
    state.filtered_valid = raw_valid | within_timeout
    state.visible_fraction = visible_fraction

    stale_mask = ~state.filtered_valid
    if bool(torch.any(stale_mask)):
        state.filtered_pos[stale_mask] = 0.0
        state.filtered_vel[stale_mask] = 0.0
        state.filtered_quat[stale_mask] = _identity_quat(int(stale_mask.sum().item()), device, dtype=dtype)
        state.filtered_ang_vel[stale_mask] = 0.0

    line_of_sight_norm = torch.linalg.norm(state.filtered_pos, dim=-1, keepdim=True).clamp(min=1.0e-6)
    line_of_sight = state.filtered_pos / line_of_sight_norm
    line_of_sight = torch.where(state.filtered_valid.unsqueeze(-1), line_of_sight, torch.zeros_like(line_of_sight))

    age_fraction = torch.clamp(
        torch.full_like(state.last_seen_s, now_s) - state.last_seen_s,
        min=0.0,
        max=timeout_s,
    ) / max(timeout_s, 1.0e-6)

    valid_quat = torch.where(
        state.filtered_valid.unsqueeze(-1),
        state.filtered_quat,
        _identity_quat(num_envs, device, dtype=dtype),
    )

    state.cached_rel_pos = torch.where(
        state.filtered_valid.unsqueeze(-1), state.filtered_pos, torch.zeros_like(state.filtered_pos)
    )
    state.cached_rel_lin_vel = torch.where(
        state.filtered_valid.unsqueeze(-1), state.filtered_vel, torch.zeros_like(state.filtered_vel)
    )
    state.cached_rel_quat = valid_quat
    state.cached_rel_ang_vel = torch.where(
        state.filtered_valid.unsqueeze(-1), state.filtered_ang_vel, torch.zeros_like(state.filtered_ang_vel)
    )
    state.cached_line_of_sight = line_of_sight
    state.cached_status = torch.stack(
        [
            raw_valid.to(dtype=dtype),
            state.filtered_valid.to(dtype=dtype),
            visible_fraction,
            age_fraction,
        ],
        dim=-1,
    )
    state.last_step = step
    return state


def vision_rel_pos(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_rel_pos


def vision_rel_lin_vel(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_rel_lin_vel


def vision_rel_quat(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_rel_quat


def vision_rel_ang_vel(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_rel_ang_vel


def vision_line_of_sight(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_line_of_sight


def vision_status(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_status
