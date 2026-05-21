from __future__ import annotations

import math

import torch

from isaaclab.utils import math as math_utils


class _VisionObservationState:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.last_step = -1

        self.filter_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.filtered_pos = torch.zeros((num_envs, 3), device=self.device)
        self.filtered_vel = torch.zeros((num_envs, 3), device=self.device)
        self.filtered_acc = torch.zeros((num_envs, 3), device=self.device)
        self.filtered_quat = _identity_quat(num_envs, self.device)
        self.filtered_ang_vel = torch.zeros((num_envs, 3), device=self.device)
        self.position_cov = _batched_eye(9, num_envs, self.device, torch.float32) * 0.01

        self.last_seen_s = torch.full((num_envs,), -1.0e9, device=self.device)
        self.reject_counts = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.z_smoother_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.z_smoother_value = torch.zeros(num_envs, device=self.device)
        self.z_smoother_time_s = torch.zeros(num_envs, device=self.device)

        self.raw_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.filtered_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.visible_fraction = torch.zeros(num_envs, device=self.device)
        self.raw_pos = torch.zeros((num_envs, 3), device=self.device)
        self.raw_quat = _identity_quat(num_envs, self.device)
        self.raw_rpy_deg = torch.zeros((num_envs, 3), device=self.device)
        self.filtered_rpy_deg = torch.zeros((num_envs, 3), device=self.device)

        self.cached_rel_pos = torch.zeros((num_envs, 3), device=self.device)
        self.cached_rel_lin_vel = torch.zeros((num_envs, 3), device=self.device)
        self.cached_rel_quat = _identity_quat(num_envs, self.device)
        self.cached_rel_ang_vel = torch.zeros((num_envs, 3), device=self.device)
        self.cached_line_of_sight = torch.zeros((num_envs, 3), device=self.device)
        self.cached_status = torch.zeros((num_envs, 4), device=self.device)
        self.cached_raw_rel_pos = torch.zeros((num_envs, 3), device=self.device)
        self.cached_raw_rel_quat = _identity_quat(num_envs, self.device)
        self.cached_raw_rpy_deg = torch.zeros((num_envs, 3), device=self.device)
        self.cached_filtered_rpy_deg = torch.zeros((num_envs, 3), device=self.device)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        count = int(env_ids.numel())
        self.filter_initialized[env_ids] = False
        self.filtered_pos[env_ids] = 0.0
        self.filtered_vel[env_ids] = 0.0
        self.filtered_acc[env_ids] = 0.0
        self.filtered_quat[env_ids] = _identity_quat(count, self.device, dtype=self.filtered_quat.dtype)
        self.filtered_ang_vel[env_ids] = 0.0
        self.position_cov[env_ids] = _batched_eye(9, count, self.device, self.position_cov.dtype) * 0.01
        self.last_seen_s[env_ids] = -1.0e9
        self.reject_counts[env_ids] = 0
        self.z_smoother_initialized[env_ids] = False
        self.z_smoother_value[env_ids] = 0.0
        self.z_smoother_time_s[env_ids] = 0.0
        self.raw_valid[env_ids] = False
        self.filtered_valid[env_ids] = False
        self.visible_fraction[env_ids] = 0.0
        self.raw_pos[env_ids] = 0.0
        self.raw_quat[env_ids] = _identity_quat(count, self.device, dtype=self.raw_quat.dtype)
        self.raw_rpy_deg[env_ids] = 0.0
        self.filtered_rpy_deg[env_ids] = 0.0
        self.cached_rel_pos[env_ids] = 0.0
        self.cached_rel_lin_vel[env_ids] = 0.0
        self.cached_rel_quat[env_ids] = _identity_quat(count, self.device, dtype=self.cached_rel_quat.dtype)
        self.cached_rel_ang_vel[env_ids] = 0.0
        self.cached_line_of_sight[env_ids] = 0.0
        self.cached_status[env_ids] = 0.0
        self.cached_raw_rel_pos[env_ids] = 0.0
        self.cached_raw_rel_quat[env_ids] = _identity_quat(count, self.device, dtype=self.cached_raw_rel_quat.dtype)
        self.cached_raw_rpy_deg[env_ids] = 0.0
        self.cached_filtered_rpy_deg[env_ids] = 0.0


def _identity_quat(num_envs: int, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    quat = torch.zeros((num_envs, 4), device=device, dtype=dtype)
    quat[:, 0] = 1.0
    return quat


def _batched_eye(size: int, batch: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(size, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)


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


def _euler_xyz_deg_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_wxyz)
    return torch.stack((roll, pitch, yaw), dim=-1) * (180.0 / math.pi)


def _get_vision_state(env) -> _VisionObservationState:
    state = getattr(env, "_landing_sway_vision_observation_state", None)
    if state is None or state.num_envs != env.num_envs:
        state = _VisionObservationState(env.num_envs, env.device)
        setattr(env, "_landing_sway_vision_observation_state", state)
    return state


def _predict_position_filter(state: _VisionObservationState, env_ids: torch.Tensor, dt: float, q_diagonal: float) -> None:
    if env_ids.numel() == 0:
        return
    pos = state.filtered_pos[env_ids]
    vel = state.filtered_vel[env_ids]
    acc = state.filtered_acc[env_ids]
    quat = state.filtered_quat[env_ids]
    ang_vel = state.filtered_ang_vel[env_ids]
    cov = state.position_cov[env_ids]

    state.filtered_pos[env_ids] = pos + vel * dt + 0.5 * acc * (dt * dt)
    state.filtered_vel[env_ids] = vel + acc * dt
    state.filtered_quat[env_ids] = _integrate_quaternion(quat, ang_vel, dt)

    batch = int(env_ids.numel())
    dtype = cov.dtype
    device = cov.device
    f = _batched_eye(9, batch, device, dtype)
    f[:, 0:3, 3:6] = torch.eye(3, device=device, dtype=dtype) * dt
    f[:, 0:3, 6:9] = torch.eye(3, device=device, dtype=dtype) * (0.5 * dt * dt)
    f[:, 3:6, 6:9] = torch.eye(3, device=device, dtype=dtype) * dt
    q = _batched_eye(9, batch, device, dtype) * float(q_diagonal)
    state.position_cov[env_ids] = f @ cov @ torch.transpose(f, 1, 2) + q


def _initialize_filter(
    state: _VisionObservationState,
    env_ids: torch.Tensor,
    measured_pos: torch.Tensor,
    measured_quat: torch.Tensor,
) -> None:
    if env_ids.numel() == 0:
        return
    batch = int(env_ids.numel())
    dtype = state.filtered_pos.dtype
    device = state.filtered_pos.device
    state.filter_initialized[env_ids] = True
    state.filtered_pos[env_ids] = measured_pos[env_ids]
    state.filtered_vel[env_ids] = 0.0
    state.filtered_acc[env_ids] = 0.0
    state.filtered_quat[env_ids] = math_utils.quat_unique(measured_quat[env_ids])
    state.filtered_ang_vel[env_ids] = 0.0
    state.position_cov[env_ids] = _batched_eye(9, batch, device, dtype) * 0.01
    state.reject_counts[env_ids] = 0
    state.z_smoother_initialized[env_ids] = False
    state.z_smoother_value[env_ids] = 0.0
    state.z_smoother_time_s[env_ids] = 0.0


def _update_position_filter(
    state: _VisionObservationState,
    env_ids: torch.Tensor,
    measured_pos: torch.Tensor,
    measured_quat: torch.Tensor,
    dt: float,
    position_r_diagonal: float,
    ang_blend: float,
) -> None:
    if env_ids.numel() == 0:
        return
    cov = state.position_cov[env_ids]
    batch = int(env_ids.numel())
    dtype = cov.dtype
    device = cov.device
    r_pos = _batched_eye(3, batch, device, dtype) * float(position_r_diagonal)
    hph_t = cov[:, 0:3, 0:3]
    s = hph_t + r_pos
    k = cov[:, :, 0:3] @ torch.linalg.inv(s)
    innovation = measured_pos[env_ids] - state.filtered_pos[env_ids]
    dx = (k @ innovation.unsqueeze(-1)).squeeze(-1)

    state.filtered_pos[env_ids] = state.filtered_pos[env_ids] + dx[:, 0:3]
    state.filtered_vel[env_ids] = state.filtered_vel[env_ids] + dx[:, 3:6]
    state.filtered_acc[env_ids] = state.filtered_acc[env_ids] + dx[:, 6:9]

    kh = torch.zeros_like(cov)
    kh[:, :, 0:3] = k
    i9 = _batched_eye(9, batch, device, dtype)
    ikh = i9 - kh
    state.position_cov[env_ids] = ikh @ cov @ torch.transpose(ikh, 1, 2) + k @ r_pos @ torch.transpose(k, 1, 2)

    quat_error = math_utils.quat_mul(measured_quat[env_ids], math_utils.quat_inv(state.filtered_quat[env_ids]))
    raw_ang_vel = math_utils.axis_angle_from_quat(quat_error) / max(float(dt), 1.0e-6)
    state.filtered_ang_vel[env_ids] = torch.lerp(state.filtered_ang_vel[env_ids], raw_ang_vel, float(ang_blend))
    state.filtered_quat[env_ids] = math_utils.quat_unique(measured_quat[env_ids])
    state.reject_counts[env_ids] = 0


def _apply_z_output_smoother(
    state: _VisionObservationState,
    z_values: torch.Tensor,
    valid_mask: torch.Tensor,
    now_s: float,
    tau_s: float,
) -> torch.Tensor:
    output = z_values.clone()
    invalid_mask = ~valid_mask
    if bool(torch.any(invalid_mask)):
        state.z_smoother_initialized[invalid_mask] = False
        state.z_smoother_value[invalid_mask] = 0.0
        state.z_smoother_time_s[invalid_mask] = 0.0

    init_mask = valid_mask & ~state.z_smoother_initialized
    if bool(torch.any(init_mask)):
        state.z_smoother_initialized[init_mask] = True
        state.z_smoother_value[init_mask] = z_values[init_mask]
        state.z_smoother_time_s[init_mask] = float(now_s)
        output[init_mask] = z_values[init_mask]

    update_mask = valid_mask & state.z_smoother_initialized & ~init_mask
    if bool(torch.any(update_mask)):
        dt = torch.full_like(state.z_smoother_time_s[update_mask], float(now_s)) - state.z_smoother_time_s[update_mask]
        alpha = 1.0 - torch.exp(-dt / max(float(tau_s), 1.0e-6))
        smoothed = state.z_smoother_value[update_mask] + alpha * (z_values[update_mask] - state.z_smoother_value[update_mask])
        state.z_smoother_value[update_mask] = smoothed
        state.z_smoother_time_s[update_mask] = float(now_s)
        output[update_mask] = smoothed

    return output


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

    state.raw_valid = raw_valid
    state.visible_fraction = visible_fraction
    state.raw_pos = torch.where(raw_valid.unsqueeze(-1), measured_pos, torch.zeros_like(measured_pos))
    state.raw_quat = torch.where(
        raw_valid.unsqueeze(-1),
        measured_quat,
        _identity_quat(num_envs, device, dtype=dtype),
    )
    state.raw_rpy_deg = torch.where(
        raw_valid.unsqueeze(-1),
        _euler_xyz_deg_from_quat(state.raw_quat),
        torch.zeros_like(state.raw_rpy_deg),
    )

    q_diagonal = float(getattr(vision_cfg, "q_diagonal", 0.01))
    predict_ids = state.filter_initialized.nonzero(as_tuple=False).squeeze(-1)
    _predict_position_filter(state, predict_ids, step_dt, q_diagonal)

    timeout_s = float(getattr(vision_cfg, "marker_timeout_s", vision_cfg.measurement_timeout_s))
    timed_out_mask = (~raw_valid) & state.filter_initialized & ((now_s - state.last_seen_s) > timeout_s)
    if bool(torch.any(timed_out_mask)):
        state.filter_initialized[timed_out_mask] = False
        state.reject_counts[timed_out_mask] = 0
        state.z_smoother_initialized[timed_out_mask] = False
        state.z_smoother_value[timed_out_mask] = 0.0
        state.z_smoother_time_s[timed_out_mask] = 0.0
        state.filtered_vel[timed_out_mask] = 0.0
        state.filtered_acc[timed_out_mask] = 0.0
        state.filtered_ang_vel[timed_out_mask] = 0.0

    measured_pos_for_filter = measured_pos.clone()
    measured_pos_for_filter[:, 2] = (
        float(getattr(vision_cfg, "z_measurement_scale", 1.0)) * measured_pos_for_filter[:, 2]
        + float(getattr(vision_cfg, "z_measurement_bias", 0.0))
    )

    init_mask = raw_valid & ~state.filter_initialized
    init_ids = init_mask.nonzero(as_tuple=False).squeeze(-1)
    _initialize_filter(state, init_ids, measured_pos_for_filter, measured_quat)

    active_update_mask = raw_valid & state.filter_initialized & ~init_mask
    if bool(torch.any(active_update_mask)):
        pos_innov = torch.linalg.norm(measured_pos_for_filter - state.filtered_pos, dim=-1)
        z_innov = torch.abs(measured_pos_for_filter[:, 2] - state.filtered_pos[:, 2])
        quat_error = math_utils.quat_mul(measured_quat, math_utils.quat_inv(state.filtered_quat))
        ang_innov_deg = torch.linalg.norm(math_utils.axis_angle_from_quat(quat_error), dim=-1) * (180.0 / math.pi)

        accepted = active_update_mask
        accepted &= pos_innov < float(getattr(vision_cfg, "pos_innov_threshold_m", 0.35))
        accepted &= z_innov < float(getattr(vision_cfg, "z_innov_threshold_m", 0.12))
        if not bool(getattr(vision_cfg, "position_only_filter", True)):
            accepted &= ang_innov_deg < float(getattr(vision_cfg, "angle_innov_threshold_deg", 20.0))

        accepted_ids = accepted.nonzero(as_tuple=False).squeeze(-1)
        _update_position_filter(
            state,
            accepted_ids,
            measured_pos_for_filter,
            measured_quat,
            step_dt,
            float(getattr(vision_cfg, "position_r_diagonal", 0.03)),
            float(getattr(vision_cfg, "angular_velocity_blend", 0.35)),
        )

        rejected_mask = active_update_mask & ~accepted
        if bool(torch.any(rejected_mask)):
            state.reject_counts[rejected_mask] = state.reject_counts[rejected_mask] + 1
            reinit_mask = rejected_mask & (
                state.reject_counts >= int(max(getattr(vision_cfg, "reinit_after_rejects", 3), 1))
            )
            reinit_ids = reinit_mask.nonzero(as_tuple=False).squeeze(-1)
            _initialize_filter(state, reinit_ids, measured_pos_for_filter, measured_quat)

    if bool(torch.any(raw_valid)):
        state.last_seen_s = torch.where(raw_valid, torch.full_like(state.last_seen_s, now_s), state.last_seen_s)

    state.filtered_valid = state.filter_initialized & ((now_s - state.last_seen_s) <= timeout_s)

    filtered_pos_output = state.filtered_pos.clone()
    if bool(getattr(vision_cfg, "enable_z_output_smoother", True)):
        filtered_pos_output[:, 2] = _apply_z_output_smoother(
            state,
            filtered_pos_output[:, 2],
            state.filtered_valid,
            now_s,
            float(getattr(vision_cfg, "z_output_smoother_tau_s", 0.20)),
        )
    else:
        _ = _apply_z_output_smoother(
            state,
            filtered_pos_output[:, 2],
            torch.zeros_like(state.filtered_valid),
            now_s,
            1.0,
        )

    state.filtered_rpy_deg = torch.where(
        state.filtered_valid.unsqueeze(-1),
        _euler_xyz_deg_from_quat(state.filtered_quat),
        torch.zeros_like(state.filtered_rpy_deg),
    )

    line_of_sight_norm = torch.linalg.norm(filtered_pos_output, dim=-1, keepdim=True).clamp(min=1.0e-6)
    line_of_sight = filtered_pos_output / line_of_sight_norm
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
        state.filtered_valid.unsqueeze(-1), filtered_pos_output, torch.zeros_like(filtered_pos_output)
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
    state.cached_raw_rel_pos = torch.where(raw_valid.unsqueeze(-1), state.raw_pos, torch.zeros_like(state.raw_pos))
    state.cached_raw_rel_quat = torch.where(
        raw_valid.unsqueeze(-1),
        state.raw_quat,
        _identity_quat(num_envs, device, dtype=dtype),
    )
    state.cached_raw_rpy_deg = torch.where(raw_valid.unsqueeze(-1), state.raw_rpy_deg, torch.zeros_like(state.raw_rpy_deg))
    state.cached_filtered_rpy_deg = torch.where(
        state.filtered_valid.unsqueeze(-1),
        state.filtered_rpy_deg,
        torch.zeros_like(state.filtered_rpy_deg),
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


def vision_raw_rel_pos(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_raw_rel_pos


def vision_raw_rel_quat(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_raw_rel_quat


def vision_raw_rpy_deg(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_raw_rpy_deg


def vision_filtered_rpy_deg(env) -> torch.Tensor:
    return _update_vision_cache(env).cached_filtered_rpy_deg
