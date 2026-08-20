from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils import math as math_utils

from ...heave_landing.mdp.observations import (
    future_platform_pos_z_w,
    projected_gravity_noisy,
    root_lin_vel_rel,
    root_lin_vel_w,
    root_pos_rel,
    root_quat_rel,
)


@dataclass
class _MarkerDefinition:
    marker_id: int
    bits_size: int
    bits_matrix: np.ndarray
    corners_normalized: np.ndarray


class _PositionOnlyMekf:
    def __init__(self, dt: float, q_diagonal: float, r_diagonal: float, position_r_diagonal: float):
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.a = np.zeros(3, dtype=np.float64)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.dt = float(dt)
        self.P = np.eye(15, dtype=np.float64) * 0.01
        self.Q = np.eye(15, dtype=np.float64) * float(q_diagonal)
        self.R = np.eye(6, dtype=np.float64) * float(r_diagonal)
        self.R[0:3, 0:3] = np.eye(3, dtype=np.float64) * float(position_r_diagonal)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
        )

    @staticmethod
    def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
        return Rotation.from_rotvec(rotvec).as_quat().astype(np.float64)

    def predict(self) -> None:
        dt = float(self.dt)
        self.p = self.p + self.v * dt + 0.5 * self.a * dt * dt
        self.v = self.v + self.a * dt
        self.q = (Rotation.from_quat(self.q) * Rotation.from_rotvec(self.omega * dt)).as_quat()

        f = np.zeros((15, 15), dtype=np.float64)
        i3 = np.eye(3, dtype=np.float64)
        f[0:3, 0:3] = i3
        f[0:3, 3:6] = i3 * dt
        f[0:3, 6:9] = i3 * (0.5 * dt * dt)
        f[3:6, 3:6] = i3
        f[3:6, 6:9] = i3 * dt
        f[6:9, 6:9] = i3
        f[9:12, 9:12] = i3 - dt * self._skew(self.omega)
        f[9:12, 12:15] = i3 * dt
        f[12:15, 12:15] = i3
        self.P = f @ self.P @ f.T + self.Q

    def update_position(self, tvec_meas: np.ndarray) -> None:
        innovation_pos = tvec_meas - self.p
        h = np.zeros((3, 15), dtype=np.float64)
        h[0:3, 0:3] = np.eye(3, dtype=np.float64)
        r_pos = self.R[0:3, 0:3]
        s = h @ self.P @ h.T + r_pos
        k = self.P @ h.T @ np.linalg.inv(s)
        dx = k @ innovation_pos
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.a += dx[6:9]
        i15 = np.eye(15, dtype=np.float64)
        ikh = i15 - k @ h
        self.P = ikh @ self.P @ ikh.T + k @ r_pos @ k.T

    def update_full(self, tvec_meas: np.ndarray, rvec_meas: np.ndarray) -> None:
        q_meas = self._quat_from_rotvec(rvec_meas)
        innovation_pos = tvec_meas - self.p
        q_err = Rotation.from_quat(q_meas) * Rotation.from_quat(self.q).inv()
        innovation_rot = q_err.as_rotvec()
        innovation = np.concatenate([innovation_pos, innovation_rot], axis=0)

        h = np.zeros((6, 15), dtype=np.float64)
        h[0:3, 0:3] = np.eye(3, dtype=np.float64)
        h[3:6, 9:12] = np.eye(3, dtype=np.float64)

        s = h @ self.P @ h.T + self.R
        k = self.P @ h.T @ np.linalg.inv(s)
        dx = k @ innovation

        self.p += dx[0:3]
        self.v += dx[3:6]
        self.a += dx[6:9]
        self.q = (Rotation.from_quat(self.q) * Rotation.from_rotvec(dx[9:12])).as_quat()
        self.omega += dx[12:15]

        i15 = np.eye(15, dtype=np.float64)
        ikh = i15 - k @ h
        self.P = ikh @ self.P @ ikh.T + k @ self.R @ k.T


@dataclass
class _PerEnvVisionState:
    filter: _PositionOnlyMekf
    filter_initialized: bool = False
    filter_needs_reinit: bool = True
    last_marker_seen_s: float = -1.0e9
    consecutive_rejects: int = 0
    z_smoother_initialized: bool = False
    z_smoother_value: float = 0.0
    z_smoother_time_s: float = 0.0


class _BatchedVisionFilter:
    def __init__(self, num_envs: int, device: torch.device, dt: float, q_diagonal: float, r_diagonal: float, position_r_diagonal: float):
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = torch.float64
        self.dt = float(dt)

        self.i3 = torch.eye(3, device=device, dtype=self.dtype)
        self.i15 = torch.eye(15, device=device, dtype=self.dtype)
        self.p_init = self.i15 * 0.01
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=self.dtype)

        self.r_full = torch.eye(6, device=device, dtype=self.dtype) * float(r_diagonal)
        self.r_full[:3, :3] = self.i3 * float(position_r_diagonal)
        self.r_pos = self.r_full[:3, :3]
        self.h_full = torch.zeros((6, 15), device=device, dtype=self.dtype)
        self.h_full[:3, :3] = self.i3
        self.h_full[3:6, 9:12] = self.i3

        self.p = torch.zeros((self.num_envs, 3), device=device, dtype=self.dtype)
        self.v = torch.zeros((self.num_envs, 3), device=device, dtype=self.dtype)
        self.a = torch.zeros((self.num_envs, 3), device=device, dtype=self.dtype)
        self.q = self.identity_quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.omega = torch.zeros((self.num_envs, 3), device=device, dtype=self.dtype)
        self.p_cov = self.p_init.unsqueeze(0).repeat(self.num_envs, 1, 1)

        self.initialized = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)
        self.needs_reinit = torch.ones((self.num_envs,), device=device, dtype=torch.bool)
        self.last_marker_seen_s = torch.full((self.num_envs,), -1.0e9, device=device, dtype=self.dtype)
        self.consecutive_rejects = torch.zeros((self.num_envs,), device=device, dtype=torch.int64)

        self.z_smoother_initialized = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)
        self.z_smoother_value = torch.zeros((self.num_envs,), device=device, dtype=self.dtype)
        self.z_smoother_time_s = torch.zeros((self.num_envs,), device=device, dtype=self.dtype)

    @staticmethod
    def _safe_axis(rotvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angle = torch.linalg.norm(rotvec, dim=-1)
        axis = torch.zeros_like(rotvec)
        valid = angle > 1.0e-9
        if torch.any(valid):
            axis[valid] = rotvec[valid] / angle[valid].unsqueeze(-1)
        if torch.any(~valid):
            axis[~valid, 0] = 1.0
        return angle, axis

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        idx = env_ids.to(device=self.device, dtype=torch.long)
        self.p[idx] = 0.0
        self.v[idx] = 0.0
        self.a[idx] = 0.0
        self.q[idx] = self.identity_quat
        self.omega[idx] = 0.0
        self.p_cov[idx] = self.p_init
        self.initialized[idx] = False
        self.needs_reinit[idx] = True
        self.last_marker_seen_s[idx] = -1.0e9
        self.consecutive_rejects[idx] = 0
        self.z_smoother_initialized[idx] = False
        self.z_smoother_value[idx] = 0.0
        self.z_smoother_time_s[idx] = 0.0

    def _set_state(self, idx: torch.Tensor, pos_meas: torch.Tensor, quat_meas_wxyz: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        self.p[idx] = pos_meas
        self.v[idx] = 0.0
        self.a[idx] = 0.0
        self.q[idx] = math_utils.quat_unique(quat_meas_wxyz)
        self.omega[idx] = 0.0
        self.p_cov[idx] = self.p_init
        self.initialized[idx] = True
        self.needs_reinit[idx] = False
        self.consecutive_rejects[idx] = 0
        self.z_smoother_initialized[idx] = False
        self.z_smoother_value[idx] = 0.0
        self.z_smoother_time_s[idx] = 0.0

    def _update_position(self, idx: torch.Tensor, pos_meas: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        cov = self.p_cov[idx]
        innovation = pos_meas - self.p[idx]
        s_mat = cov[:, 0:3, 0:3] + self.r_pos.unsqueeze(0)
        k_gain = cov[:, :, 0:3] @ torch.linalg.inv(s_mat)
        dx = (k_gain @ innovation.unsqueeze(-1)).squeeze(-1)
        self.p[idx] = self.p[idx] + dx[:, 0:3]
        self.v[idx] = self.v[idx] + dx[:, 3:6]
        self.a[idx] = self.a[idx] + dx[:, 6:9]

        kh = torch.zeros((idx.numel(), 15, 15), device=self.device, dtype=self.dtype)
        kh[:, :, 0:3] = k_gain
        ikh = self.i15.unsqueeze(0) - kh
        self.p_cov[idx] = ikh @ cov @ ikh.transpose(1, 2) + k_gain @ self.r_pos.unsqueeze(0) @ k_gain.transpose(1, 2)

    def _update_full(self, idx: torch.Tensor, pos_meas: torch.Tensor, quat_meas_wxyz: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        cov = self.p_cov[idx]
        innovation_pos = pos_meas - self.p[idx]
        quat_err = math_utils.quat_mul(quat_meas_wxyz, math_utils.quat_inv(self.q[idx]))
        innovation_rot = math_utils.axis_angle_from_quat(quat_err)
        innovation = torch.cat([innovation_pos, innovation_rot], dim=-1)

        h_full = self.h_full.unsqueeze(0).expand(idx.numel(), -1, -1)
        ph_t = cov @ self.h_full.transpose(0, 1).unsqueeze(0)
        s_mat = h_full @ ph_t + self.r_full.unsqueeze(0)
        k_gain = ph_t @ torch.linalg.inv(s_mat)
        dx = (k_gain @ innovation.unsqueeze(-1)).squeeze(-1)

        self.p[idx] = self.p[idx] + dx[:, 0:3]
        self.v[idx] = self.v[idx] + dx[:, 3:6]
        self.a[idx] = self.a[idx] + dx[:, 6:9]
        delta_angle, delta_axis = self._safe_axis(dx[:, 9:12])
        delta_quat = math_utils.quat_from_angle_axis(delta_angle, delta_axis)
        self.q[idx] = math_utils.quat_unique(math_utils.quat_mul(self.q[idx], delta_quat))
        self.omega[idx] = self.omega[idx] + dx[:, 12:15]

        kh = k_gain @ h_full
        ikh = self.i15.unsqueeze(0) - kh
        self.p_cov[idx] = ikh @ cov @ ikh.transpose(1, 2) + k_gain @ self.r_full.unsqueeze(0) @ k_gain.transpose(1, 2)

    def apply_measurements(
        self, detected_mask: torch.Tensor, raw_pos_m: torch.Tensor, raw_quat_xyzw: torch.Tensor, now_s: float, cfg
    ) -> None:
        if not torch.any(detected_mask):
            return

        detected_idx = detected_mask.nonzero(as_tuple=False).squeeze(-1)
        self.last_marker_seen_s[detected_idx] = float(now_s)

        pos_meas = raw_pos_m[detected_idx].clone().to(self.dtype)
        pos_meas[:, 2] = float(cfg.z_measurement_scale) * pos_meas[:, 2] + float(cfg.z_measurement_bias)
        quat_meas_wxyz = raw_quat_xyzw[detected_idx].roll(1, dims=-1).to(self.dtype)
        quat_meas_wxyz = math_utils.quat_unique(quat_meas_wxyz)

        init_local = self.needs_reinit[detected_idx] | (~self.initialized[detected_idx])
        self._set_state(detected_idx[init_local], pos_meas[init_local], quat_meas_wxyz[init_local])

        remain_idx = detected_idx[~init_local]
        if remain_idx.numel() == 0:
            return

        pos_meas_remain = pos_meas[~init_local]
        quat_meas_remain = quat_meas_wxyz[~init_local]
        pos_innov = torch.linalg.norm(pos_meas_remain - self.p[remain_idx], dim=-1)
        z_innov = torch.abs(pos_meas_remain[:, 2] - self.p[remain_idx, 2])

        if bool(cfg.position_only_filter):
            accept_local = (pos_innov < float(cfg.pos_innov_threshold_m)) & (z_innov < float(cfg.z_innov_threshold_m))
        else:
            quat_err = math_utils.quat_mul(quat_meas_remain, math_utils.quat_inv(self.q[remain_idx]))
            ang_innov_deg = torch.linalg.norm(math_utils.axis_angle_from_quat(quat_err), dim=-1) * (180.0 / math.pi)
            accept_local = (
                (pos_innov < float(cfg.pos_innov_threshold_m))
                & (z_innov < float(cfg.z_innov_threshold_m))
                & (ang_innov_deg < float(cfg.angle_innov_threshold_deg))
            )

        accept_idx = remain_idx[accept_local]
        if accept_idx.numel() > 0:
            if bool(cfg.position_only_filter):
                self._update_position(accept_idx, pos_meas_remain[accept_local])
                self.q[accept_idx] = quat_meas_remain[accept_local]
                self.omega[accept_idx] = 0.0
            else:
                self._update_full(accept_idx, pos_meas_remain[accept_local], quat_meas_remain[accept_local])
            self.consecutive_rejects[accept_idx] = 0

        reject_idx = remain_idx[~accept_local]
        if reject_idx.numel() == 0:
            return

        self.consecutive_rejects[reject_idx] = self.consecutive_rejects[reject_idx] + 1
        reinit_local = self.consecutive_rejects[reject_idx] >= max(int(cfg.reinit_after_rejects), 1)
        if torch.any(reinit_local):
            reject_pos = pos_meas_remain[~accept_local]
            reject_quat = quat_meas_remain[~accept_local]
            self._set_state(reject_idx[reinit_local], reject_pos[reinit_local], reject_quat[reinit_local])

    def handle_timeouts(self, detected_mask: torch.Tensor, now_s: float, marker_timeout_s: float) -> None:
        timeout_mask = (~detected_mask) & ((float(now_s) - self.last_marker_seen_s) > float(marker_timeout_s))
        if not torch.any(timeout_mask):
            return
        self.initialized[timeout_mask] = False
        self.needs_reinit[timeout_mask] = True
        self.consecutive_rejects[timeout_mask] = 0
        self.z_smoother_initialized[timeout_mask] = False
        self.z_smoother_value[timeout_mask] = 0.0
        self.z_smoother_time_s[timeout_mask] = 0.0

    def _apply_z_output_smoother(self, z_value: torch.Tensor, valid_mask: torch.Tensor, now_s: float, cfg) -> torch.Tensor:
        if not bool(cfg.enable_z_output_smoother):
            return z_value

        invalid_mask = ~valid_mask
        if torch.any(invalid_mask):
            self.z_smoother_initialized[invalid_mask] = False
            self.z_smoother_value[invalid_mask] = 0.0
            self.z_smoother_time_s[invalid_mask] = 0.0

        prev_initialized = self.z_smoother_initialized.clone()
        out = z_value.clone()

        init_mask = valid_mask & (~prev_initialized)
        if torch.any(init_mask):
            self.z_smoother_initialized[init_mask] = True
            self.z_smoother_value[init_mask] = out[init_mask]
            self.z_smoother_time_s[init_mask] = float(now_s)

        update_mask = valid_mask & prev_initialized
        if torch.any(update_mask):
            dt = float(now_s) - self.z_smoother_time_s[update_mask]
            default_dt = 1.0 / max(float(cfg.detector_fps), 1.0)
            dt = torch.where((dt <= 0.0) | (dt > 0.5), torch.full_like(dt, default_dt), dt)
            alpha = 1.0 - torch.exp(-dt / float(cfg.z_output_smoother_tau_s))
            updated = alpha * out[update_mask] + (1.0 - alpha) * self.z_smoother_value[update_mask]
            self.z_smoother_value[update_mask] = updated
            self.z_smoother_time_s[update_mask] = float(now_s)
            out[update_mask] = updated

        return out

    def outputs(self, now_s: float, cfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filtered_valid = self.initialized & ((float(now_s) - self.last_marker_seen_s) < float(cfg.marker_timeout_s))
        filtered_pos = self.p.clone()
        filtered_pos[:, 2] = self._apply_z_output_smoother(filtered_pos[:, 2], filtered_valid, now_s, cfg)
        filtered_vel = self.v.clone()
        filtered_quat_wxyz = math_utils.quat_unique(self.q.clone())
        filtered_quat_xyzw = filtered_quat_wxyz.roll(-1, dims=-1)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(filtered_quat_wxyz)
        filtered_rpy_deg = torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / math.pi)
        age = torch.clamp(float(now_s) - self.last_marker_seen_s, min=0.0, max=float(cfg.marker_timeout_s))
        age_fraction = age / max(float(cfg.marker_timeout_s), 1.0e-6)
        return filtered_valid, filtered_pos, filtered_vel, filtered_quat_xyzw, filtered_rpy_deg, age_fraction


class _HeaveLandingRgbVisionState:
    def __init__(self, env):
        robot = env.scene["robot"]
        self.device = robot.data.root_pos_w.device
        self.num_envs = int(env.num_envs)
        self.last_step = -1

        cfg = env.cfg.post_init_cfg.vision_observation
        cv2.setNumThreads(max(int(cfg.opencv_threads), 1))

        marker_defs = self._load_fractal_markers(Path(cfg.fractal_config_path))
        self._marker_lookup = {marker.marker_id: marker for marker in marker_defs}
        self._aruco_detectors = self._build_detectors(marker_defs)

        detector_dt = 1.0 / max(float(cfg.detector_fps), 1.0)
        self._batched_filter = _BatchedVisionFilter(
            num_envs=self.num_envs,
            device=self.device,
            dt=detector_dt,
            q_diagonal=float(cfg.q_diagonal),
            r_diagonal=float(cfg.r_diagonal),
            position_r_diagonal=float(cfg.position_r_diagonal),
        )

        self.camera_to_uav_rotation_matrix: torch.Tensor | None = None
        self.camera_to_uav_offset: torch.Tensor | None = None
        self.detector_period_steps = max(1, int(round(detector_dt / max(float(env.step_dt), 1.0e-6))))

        self.raw_valid = np.zeros((self.num_envs,), dtype=bool)
        self.filtered_valid = np.zeros((self.num_envs,), dtype=bool)
        self.visible_fraction = np.zeros((self.num_envs,), dtype=np.float64)
        self.raw_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.raw_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (self.num_envs, 1))
        self.raw_rpy_deg = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.filtered_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.filtered_vel = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.filtered_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (self.num_envs, 1))
        self.filtered_rpy_deg = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.markers_found = np.zeros((self.num_envs,), dtype=np.int32)
        self.age_fraction = np.ones((self.num_envs,), dtype=np.float64)
        self._diagnostic_frame_saved = False

        dtype = robot.data.root_pos_w.dtype
        self.cached_rel_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
        self.cached_rel_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
        self.cached_rel_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=dtype)
        self.cached_rel_quat[:, 0] = 1.0
        self.cached_status = torch.zeros((self.num_envs, 4), device=self.device, dtype=dtype)
        self.cached_raw_rel_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
        self.cached_raw_rel_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=dtype)
        self.cached_raw_rel_quat[:, 0] = 1.0
        self.cached_raw_rpy_deg = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
        self.cached_filtered_rpy_deg = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
        self.episode_sample_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.episode_available_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.episode_xy_error_direct_sum = torch.zeros((self.num_envs,), device=self.device, dtype=dtype)
        self.episode_xy_error_flip_x_sum = torch.zeros((self.num_envs,), device=self.device, dtype=dtype)
        self.episode_xy_error_flip_y_sum = torch.zeros((self.num_envs,), device=self.device, dtype=dtype)
        self.episode_xy_error_flip_xy_sum = torch.zeros((self.num_envs,), device=self.device, dtype=dtype)
        self.episode_xy_error_swap_xy_sum = torch.zeros((self.num_envs,), device=self.device, dtype=dtype)

    @staticmethod
    def _load_fractal_markers(path: Path) -> list[_MarkerDefinition]:
        if not path.is_file():
            raise FileNotFoundError(f"Fractal config '{path}' does not exist.")
        text = path.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("%YAML") and line.strip() != "---")
        text = re.sub(r"(\bid:)([^\s])", r"\1 \2", text)
        parsed = yaml.safe_load(text)
        markers: list[_MarkerDefinition] = []
        for entry in parsed.get("markers", []):
            marker_id = int(entry["id"])
            bits = np.asarray(entry["bits"], dtype=np.uint8)
            bits_size = int(round(math.sqrt(bits.size)))
            bits_matrix = bits.reshape((bits_size, bits_size))
            corners = np.asarray(entry["corners"], dtype=np.float64)
            markers.append(
                _MarkerDefinition(
                    marker_id=marker_id,
                    bits_size=bits_size,
                    bits_matrix=bits_matrix,
                    corners_normalized=corners,
                )
            )
        return markers

    @staticmethod
    def _make_detector_parameters() -> Any:
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(params, "cornerRefinementWinSize"):
            params.cornerRefinementWinSize = 5
        if hasattr(params, "adaptiveThreshWinSizeMin"):
            params.adaptiveThreshWinSizeMin = 3
        if hasattr(params, "adaptiveThreshWinSizeMax"):
            params.adaptiveThreshWinSizeMax = 61
        if hasattr(params, "adaptiveThreshWinSizeStep"):
            params.adaptiveThreshWinSizeStep = 4
        if hasattr(params, "adaptiveThreshConstant"):
            params.adaptiveThreshConstant = 7.0
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.01
        if hasattr(params, "maxMarkerPerimeterRate"):
            params.maxMarkerPerimeterRate = 4.0
        if hasattr(params, "polygonalApproxAccuracyRate"):
            params.polygonalApproxAccuracyRate = 0.05
        if hasattr(params, "minCornerDistanceRate"):
            params.minCornerDistanceRate = 0.01
        if hasattr(params, "minDistanceToBorder"):
            params.minDistanceToBorder = 3
        if hasattr(params, "errorCorrectionRate"):
            params.errorCorrectionRate = 0.8
        return params

    @classmethod
    def _build_detectors(cls, markers: list[_MarkerDefinition]) -> dict[int, dict[str, Any]]:
        grouped: dict[int, list[_MarkerDefinition]] = {}
        for marker in markers:
            grouped.setdefault(marker.bits_size, []).append(marker)

        detectors: dict[int, dict[str, Any]] = {}
        for bits_size, defs in grouped.items():
            defs = sorted(defs, key=lambda item: item.marker_id)
            bytes_list = np.concatenate(
                [cv2.aruco.Dictionary.getByteListFromBits(marker.bits_matrix) for marker in defs], axis=0
            )
            dictionary = cv2.aruco.Dictionary(bytes_list, bits_size, 0)
            params = cls._make_detector_parameters()
            detector = cv2.aruco.ArucoDetector(dictionary, params) if hasattr(cv2.aruco, "ArucoDetector") else None
            detectors[bits_size] = {
                "definitions": defs,
                "dictionary": dictionary,
                "detector": detector,
                "params": params,
                "local_to_actual": {index: marker.marker_id for index, marker in enumerate(defs)},
            }
        return detectors

    @staticmethod
    def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    @staticmethod
    def _safe_normalize(vec: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
        return vec / torch.linalg.norm(vec).clamp(min=eps)

    def _solve_planar_pnp_torch(
        self, object_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray
    ) -> tuple[bool, torch.Tensor | None, torch.Tensor | None]:
        if len(object_points) < 4 or len(image_points) < 4:
            return False, None, None

        obj_xy = torch.as_tensor(object_points[:, :2], device=self.device, dtype=torch.float64)
        img_uv = torch.as_tensor(image_points, device=self.device, dtype=torch.float64)
        k_mat = torch.as_tensor(camera_matrix, device=self.device, dtype=torch.float64)

        x = obj_xy[:, 0]
        y = obj_xy[:, 1]
        u = img_uv[:, 0]
        v = img_uv[:, 1]
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        row_0 = torch.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=-1)
        row_1 = torch.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=-1)
        a_mat = torch.empty((2 * obj_xy.shape[0], 9), device=self.device, dtype=torch.float64)
        a_mat[0::2] = row_0
        a_mat[1::2] = row_1

        try:
            _, _, vh = torch.linalg.svd(a_mat, full_matrices=False)
            homography = vh[-1].reshape(3, 3)
            b_mat = torch.linalg.inv(k_mat) @ homography
        except RuntimeError:
            return False, None, None

        b1 = b_mat[:, 0]
        b2 = b_mat[:, 1]
        b3 = b_mat[:, 2]
        scale = 0.5 * (torch.linalg.norm(b1) + torch.linalg.norm(b2))
        if not torch.isfinite(scale) or float(scale) < 1.0e-9:
            return False, None, None

        if float(b3[2]) < 0.0:
            b1 = -b1
            b2 = -b2
            b3 = -b3

        r1 = self._safe_normalize(b1 / scale)
        r2 = b2 / scale
        r2 = r2 - torch.dot(r1, r2) * r1
        if float(torch.linalg.norm(r2)) < 1.0e-9:
            return False, None, None
        r2 = self._safe_normalize(r2)
        r3 = self._safe_normalize(torch.cross(r1, r2, dim=0))
        r_approx = torch.stack([r1, r2, r3], dim=-1)

        try:
            u_mat, _, vh_rot = torch.linalg.svd(r_approx)
            r_marker_cam = u_mat @ vh_rot
        except RuntimeError:
            return False, None, None

        if float(torch.det(r_marker_cam)) < 0.0:
            u_mat[:, -1] *= -1.0
            r_marker_cam = u_mat @ vh_rot

        t_marker_cam = b3 / scale
        return True, r_marker_cam, t_marker_cam

    def _transform_measurements_torch(
        self, r_marker_cam: torch.Tensor, t_marker_cam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_cam_marker = r_marker_cam.transpose(0, 1)
        t_cam_marker = -r_cam_marker @ t_marker_cam.reshape(3, 1)
        offset_cam = self.camera_to_uav_offset.reshape(3, 1)
        out_t = (t_cam_marker + r_cam_marker @ offset_cam).reshape(3)

        if self.camera_to_uav_rotation_matrix is not None:
            r_uav_marker = r_cam_marker @ self.camera_to_uav_rotation_matrix
        else:
            r_flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=self.device, dtype=torch.float64))
            r_uav_marker = r_flip @ r_cam_marker

        quat_wxyz = math_utils.quat_unique(math_utils.quat_from_matrix(r_uav_marker.unsqueeze(0)))[0].to(torch.float64)
        rvec = math_utils.axis_angle_from_quat(quat_wxyz.unsqueeze(0))[0].to(torch.float64)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_wxyz.unsqueeze(0))
        euler_deg = torch.stack([pitch[0], roll[0], yaw[0]], dim=0).to(torch.float64) * (180.0 / math.pi)
        quat_xyzw = quat_wxyz.roll(-1, dims=-1)
        return out_t, rvec, quat_xyzw, euler_deg

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        idx = env_ids.detach().cpu().tolist()
        self._batched_filter.reset(env_ids)
        self.raw_valid[idx] = False
        self.filtered_valid[idx] = False
        self.visible_fraction[idx] = 0.0
        self.raw_pos[idx] = 0.0
        self.raw_quat_xyzw[idx] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.raw_rpy_deg[idx] = 0.0
        self.filtered_pos[idx] = 0.0
        self.filtered_vel[idx] = 0.0
        self.filtered_quat_xyzw[idx] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.filtered_rpy_deg[idx] = 0.0
        self.markers_found[idx] = 0
        self.age_fraction[idx] = 1.0
        self.episode_sample_count[env_ids] = 0
        self.episode_available_count[env_ids] = 0
        self.episode_xy_error_direct_sum[env_ids] = 0.0
        self.episode_xy_error_flip_x_sum[env_ids] = 0.0
        self.episode_xy_error_flip_y_sum[env_ids] = 0.0
        self.episode_xy_error_flip_xy_sum[env_ids] = 0.0
        self.episode_xy_error_swap_xy_sum[env_ids] = 0.0

    def accumulate_episode_alignment(self, env, filtered_valid: torch.Tensor, filtered_pos: torch.Tensor) -> None:
        """Accumulate full-episode vision availability and XY frame-alignment diagnostics."""

        self.episode_sample_count += 1
        self.episode_available_count += filtered_valid.to(dtype=torch.long)
        if not torch.any(filtered_valid):
            return

        robot = env.scene["robot"]
        platform = env.scene["platform"]
        true_xy = robot.data.root_pos_w[:, :2] - platform.data.root_pos_w[:, :2]
        observed_xy = filtered_pos[:, :2].to(dtype=true_xy.dtype)
        sign_flip_x = observed_xy.new_tensor([-1.0, 1.0])
        sign_flip_y = observed_xy.new_tensor([1.0, -1.0])

        def _accumulate(target: torch.Tensor, candidate: torch.Tensor) -> None:
            target += torch.linalg.norm(candidate - true_xy, dim=1) * filtered_valid

        _accumulate(self.episode_xy_error_direct_sum, observed_xy)
        _accumulate(self.episode_xy_error_flip_x_sum, observed_xy * sign_flip_x)
        _accumulate(self.episode_xy_error_flip_y_sum, observed_xy * sign_flip_y)
        _accumulate(self.episode_xy_error_flip_xy_sum, -observed_xy)
        _accumulate(self.episode_xy_error_swap_xy_sum, observed_xy[:, [1, 0]])

    def ensure_camera_transform(self, camera: TiledCamera) -> None:
        if self.camera_to_uav_rotation_matrix is not None and self.camera_to_uav_offset is not None:
            return
        if not getattr(camera, "_sensor_prims", None):
            raise RuntimeError("Camera prims are not initialized yet.")
        camera_prim = camera._sensor_prims[0]
        translation_vehicle_camera_usd = np.zeros((3,), dtype=np.float64)
        rotation_vehicle_camera_usd = np.eye(3, dtype=np.float64)

        translate_attr = camera_prim.GetPrim().GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            try:
                translation_vehicle_camera_usd = np.array(translate_attr.Get(), dtype=np.float64)
            except Exception:
                pass

        orient_attr = camera_prim.GetPrim().GetAttribute("xformOp:orient")
        if orient_attr and orient_attr.IsValid():
            try:
                quat = orient_attr.Get()
                imag = quat.GetImaginary()
                rotation_vehicle_camera_usd = Rotation.from_quat(
                    [float(imag[0]), float(imag[1]), float(imag[2]), float(quat.GetReal())]
                ).as_matrix()
            except Exception:
                pass

        rotation_camera_usd_from_camera_cv = np.diag([1.0, -1.0, -1.0])
        rotation_vehicle_camera_cv = rotation_vehicle_camera_usd @ rotation_camera_usd_from_camera_cv
        rotation_camera_cv_vehicle = rotation_vehicle_camera_cv.T
        translation_camera_cv_vehicle = -rotation_camera_cv_vehicle @ translation_vehicle_camera_usd
        self.camera_to_uav_rotation_matrix = torch.as_tensor(
            rotation_camera_cv_vehicle, device=self.device, dtype=torch.float64
        )
        self.camera_to_uav_offset = torch.as_tensor(
            translation_camera_cv_vehicle, device=self.device, dtype=torch.float64
        )

    @staticmethod
    def _reset_z_output_smoother(env_state: _PerEnvVisionState) -> None:
        env_state.z_smoother_initialized = False
        env_state.z_smoother_value = 0.0
        env_state.z_smoother_time_s = 0.0

    def _apply_z_output_smoother(self, env_state: _PerEnvVisionState, z_value: float, valid: bool, stamp_s: float, cfg) -> float:
        if not bool(cfg.enable_z_output_smoother):
            return float(z_value)
        if not valid:
            self._reset_z_output_smoother(env_state)
            return float(z_value)
        if not env_state.z_smoother_initialized:
            env_state.z_smoother_initialized = True
            env_state.z_smoother_value = float(z_value)
            env_state.z_smoother_time_s = float(stamp_s)
            return env_state.z_smoother_value
        dt = float(stamp_s - env_state.z_smoother_time_s)
        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / max(float(cfg.detector_fps), 1.0)
        alpha = float(np.clip(1.0 - math.exp(-dt / float(cfg.z_output_smoother_tau_s)), 0.0, 1.0))
        env_state.z_smoother_value = alpha * float(z_value) + (1.0 - alpha) * env_state.z_smoother_value
        env_state.z_smoother_time_s = float(stamp_s)
        return env_state.z_smoother_value

    def _measurement_update(self, env_state: _PerEnvVisionState, raw_position: np.ndarray, raw_rvec: np.ndarray, now_s: float, cfg) -> None:
        q_meas = Rotation.from_rotvec(raw_rvec).as_quat()
        tvec_filter_meas = raw_position.astype(np.float64).copy()
        tvec_filter_meas[2] = float(cfg.z_measurement_scale) * tvec_filter_meas[2] + float(cfg.z_measurement_bias)

        if env_state.filter_needs_reinit or not env_state.filter_initialized:
            env_state.filter.p = tvec_filter_meas
            env_state.filter.v[:] = 0.0
            env_state.filter.a[:] = 0.0
            env_state.filter.q = q_meas
            env_state.filter.omega[:] = 0.0
            env_state.filter.P[:] = np.eye(15, dtype=np.float64) * 0.01
            env_state.filter_needs_reinit = False
            env_state.filter_initialized = True
            env_state.consecutive_rejects = 0
            self._reset_z_output_smoother(env_state)
            return

        pos_innov = float(np.linalg.norm(tvec_filter_meas - env_state.filter.p))
        z_innov = float(abs(tvec_filter_meas[2] - env_state.filter.p[2]))
        accepted = False

        if bool(cfg.position_only_filter):
            if pos_innov < float(cfg.pos_innov_threshold_m) and z_innov < float(cfg.z_innov_threshold_m):
                env_state.filter.update_position(tvec_filter_meas)
                env_state.filter.q = q_meas
                env_state.filter.omega[:] = 0.0
                accepted = True
        else:
            q_err = Rotation.from_quat(q_meas) * Rotation.from_quat(env_state.filter.q).inv()
            ang_innov_deg = float(np.linalg.norm(q_err.as_rotvec()) * 180.0 / math.pi)
            if (
                pos_innov < float(cfg.pos_innov_threshold_m)
                and z_innov < float(cfg.z_innov_threshold_m)
                and ang_innov_deg < float(cfg.angle_innov_threshold_deg)
            ):
                env_state.filter.update_full(tvec_filter_meas, raw_rvec)
                accepted = True

        if accepted:
            env_state.consecutive_rejects = 0
            return

        env_state.consecutive_rejects += 1
        if env_state.consecutive_rejects >= max(int(cfg.reinit_after_rejects), 1):
            env_state.filter.p = tvec_filter_meas
            env_state.filter.v[:] = 0.0
            env_state.filter.a[:] = 0.0
            env_state.filter.q = q_meas
            env_state.filter.omega[:] = 0.0
            env_state.filter.P[:] = np.eye(15, dtype=np.float64) * 0.01
            env_state.consecutive_rejects = 0
            env_state.filter_needs_reinit = False
            env_state.filter_initialized = True
            self._reset_z_output_smoother(env_state)

    def _process_env_frame(
        self,
        env_index: int,
        frame_rgb: np.ndarray,
        camera_matrix: np.ndarray,
        now_s: float,
        cfg,
    ) -> None:
        frame = np.asarray(frame_rgb)
        raw_position = np.zeros(3, dtype=np.float64)
        raw_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        raw_rpy_deg = np.zeros(3, dtype=np.float64)
        detected = False
        markers_found = 0
        visible_fraction = 0.0

        if frame.size == 0:
            self.raw_valid[env_index] = False
            self.raw_pos[env_index] = raw_position
            self.raw_quat_xyzw[env_index] = raw_quat_xyzw
            self.raw_rpy_deg[env_index] = raw_rpy_deg
            self.markers_found[env_index] = 0
            self.visible_fraction[env_index] = 0.0
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame_bgr = cv2.cvtColor(frame[:, :, :4], cv2.COLOR_RGBA2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        detected_markers: list[tuple[np.ndarray, int]] = []
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []

        for bundle in self._aruco_detectors.values():
            detector = bundle["detector"]
            dictionary = bundle["dictionary"]
            params = bundle["params"]
            if detector is not None:
                corners_list, ids, _ = detector.detectMarkers(gray)
                if ids is None or len(ids) == 0:
                    corners_list, ids, _ = detector.detectMarkers(gray_clahe)
            else:
                corners_list, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
                if ids is None or len(ids) == 0:
                    corners_list, ids, _ = cv2.aruco.detectMarkers(gray_clahe, dictionary, parameters=params)
            if ids is None or len(ids) == 0:
                continue

            for corners, local_id in zip(corners_list, ids.flatten()):
                corner_array = corners.reshape((4, 2)).astype(np.float32)
                refined = cv2.cornerSubPix(
                    gray,
                    corner_array.reshape((-1, 1, 2)),
                    winSize=(3, 3),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                )
                refined = refined.reshape((4, 2)).astype(np.float32)
                actual_id = bundle["local_to_actual"][int(local_id)]
                marker_def = self._marker_lookup[actual_id]
                object_corners = marker_def.corners_normalized[:, :3] * (float(cfg.marker_size_m) * 0.5)
                object_points.extend(object_corners.astype(np.float32))
                image_points.extend(refined.astype(np.float32))
                detected_markers.append((refined, actual_id))

        detected = len(detected_markers) > 0

        if detected:
            object_points_np = np.asarray(object_points, dtype=np.float64)
            image_points_np = np.asarray(image_points, dtype=np.float64)
            solve_ok, r_marker_cam, t_marker_cam = self._solve_planar_pnp_torch(
                object_points_np, image_points_np, np.asarray(camera_matrix, dtype=np.float64)
            )
            detected = bool(solve_ok)
            if detected:
                raw_position_t, _, raw_quat_xyzw_t, raw_rpy_deg_t = self._transform_measurements_torch(
                    r_marker_cam, t_marker_cam
                )
                raw_position = raw_position_t.detach().cpu().numpy()
                raw_quat_xyzw = raw_quat_xyzw_t.detach().cpu().numpy()
                raw_rpy_deg = raw_rpy_deg_t.detach().cpu().numpy()
                markers_found = len(detected_markers)
                visible_fraction = 1.0

        self.raw_valid[env_index] = detected
        self.raw_pos[env_index] = raw_position
        self.raw_quat_xyzw[env_index] = raw_quat_xyzw
        self.raw_rpy_deg[env_index] = raw_rpy_deg
        self.markers_found[env_index] = markers_found
        self.visible_fraction[env_index] = visible_fraction

    def _sync_cached_tensors(
        self,
        dtype: torch.dtype,
        filtered_valid_t: torch.Tensor,
        filtered_pos_t: torch.Tensor,
        filtered_vel_t: torch.Tensor,
        filtered_quat_xyzw_t: torch.Tensor,
        filtered_rpy_deg_t: torch.Tensor,
        age_fraction_t: torch.Tensor,
    ) -> None:
        raw_quat_xyzw_t = torch.as_tensor(self.raw_quat_xyzw, device=self.device, dtype=dtype)
        raw_quat_wxyz_t = raw_quat_xyzw_t.roll(1, dims=-1)

        self.cached_rel_pos = filtered_pos_t.to(dtype=dtype)
        self.cached_rel_lin_vel = filtered_vel_t.to(dtype=dtype)
        self.cached_rel_quat = filtered_quat_xyzw_t.roll(1, dims=-1).to(dtype=dtype)
        self.cached_status = torch.stack(
            [
                torch.as_tensor(self.raw_valid.astype(np.float64), device=self.device, dtype=dtype),
                filtered_valid_t.to(dtype=dtype),
                torch.as_tensor(self.visible_fraction.astype(np.float64), device=self.device, dtype=dtype),
                age_fraction_t.to(dtype=dtype),
            ],
            dim=-1,
        )
        self.cached_raw_rel_pos = torch.as_tensor(self.raw_pos, device=self.device, dtype=dtype)
        self.cached_raw_rel_quat = raw_quat_wxyz_t
        self.cached_raw_rpy_deg = torch.as_tensor(self.raw_rpy_deg, device=self.device, dtype=dtype)
        self.cached_filtered_rpy_deg = filtered_rpy_deg_t.to(dtype=dtype)

    def update(self, env):
        step = int(getattr(env, "common_step_counter", -1))
        if self.last_step == step:
            return self

        reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        self.reset_envs(reset_ids)

        cfg = env.cfg.post_init_cfg.vision_observation
        camera: TiledCamera = env.scene["onboard_camera"]
        if not camera.is_initialized:
            raise RuntimeError("Onboard camera sensor is not initialized. Make sure cameras are enabled for the app.")
        self.ensure_camera_transform(camera)

        robot = env.scene["robot"]
        dtype = robot.data.root_pos_w.dtype
        now_s = float(max(step, 0)) * float(env.step_dt)
        should_process = (step % self.detector_period_steps) == 0

        if should_process:
            rgb_batch = camera.data.output["rgb"].detach().cpu().numpy()
            intrinsic_batch = camera.data.intrinsic_matrices.detach().cpu().numpy()
            if not self._diagnostic_frame_saved and self.num_envs > 0:
                diagnostic_frame = np.asarray(rgb_batch[0])
                if diagnostic_frame.dtype != np.uint8:
                    diagnostic_frame = np.clip(diagnostic_frame, 0, 255).astype(np.uint8)
                if diagnostic_frame.shape[-1] == 4:
                    diagnostic_bgr = cv2.cvtColor(diagnostic_frame[:, :, :4], cv2.COLOR_RGBA2BGR)
                else:
                    diagnostic_bgr = cv2.cvtColor(diagnostic_frame[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.imwrite("/tmp/heave_landing_vision_camera_env0.png", diagnostic_bgr)
                self._diagnostic_frame_saved = True
            for env_index in range(self.num_envs):
                self._process_env_frame(env_index, rgb_batch[env_index], intrinsic_batch[env_index], now_s, cfg)

            detected_mask_t = torch.as_tensor(self.raw_valid, device=self.device, dtype=torch.bool)
            raw_pos_t = torch.as_tensor(self.raw_pos, device=self.device, dtype=torch.float64)
            raw_quat_xyzw_t = torch.as_tensor(self.raw_quat_xyzw, device=self.device, dtype=torch.float64)
            self._batched_filter.apply_measurements(detected_mask_t, raw_pos_t, raw_quat_xyzw_t, now_s, cfg)
        else:
            detected_mask_t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self._batched_filter.handle_timeouts(detected_mask_t, now_s, float(cfg.marker_timeout_s))
        (
            filtered_valid_t,
            filtered_pos_t,
            filtered_vel_t,
            filtered_quat_xyzw_t,
            filtered_rpy_deg_t,
            age_fraction_t,
        ) = self._batched_filter.outputs(now_s, cfg)
        self.accumulate_episode_alignment(env, filtered_valid_t, filtered_pos_t)

        self.filtered_valid = filtered_valid_t.detach().cpu().numpy()
        self.filtered_pos = filtered_pos_t.detach().cpu().numpy()
        self.filtered_vel = filtered_vel_t.detach().cpu().numpy()
        self.filtered_quat_xyzw = filtered_quat_xyzw_t.detach().cpu().numpy()
        self.filtered_rpy_deg = filtered_rpy_deg_t.detach().cpu().numpy()
        self.age_fraction = age_fraction_t.detach().cpu().numpy()

        self._sync_cached_tensors(
            dtype,
            filtered_valid_t,
            filtered_pos_t,
            filtered_vel_t,
            filtered_quat_xyzw_t,
            filtered_rpy_deg_t,
            age_fraction_t,
        )
        self.last_step = step
        return self


def _get_vision_state(env) -> _HeaveLandingRgbVisionState:
    state = getattr(env, "_heave_landing_rgb_vision_state", None)
    if state is None or state.num_envs != env.num_envs:
        state = _HeaveLandingRgbVisionState(env)
        setattr(env, "_heave_landing_rgb_vision_state", state)
    return state


def _update_vision_state(env) -> _HeaveLandingRgbVisionState:
    return _get_vision_state(env).update(env)


def vision_rel_pos(env) -> torch.Tensor:
    return _update_vision_state(env).cached_rel_pos


def vision_rel_lin_vel(env) -> torch.Tensor:
    return _update_vision_state(env).cached_rel_lin_vel


def vision_rel_quat(env) -> torch.Tensor:
    return _update_vision_state(env).cached_rel_quat


def vision_available(env) -> torch.Tensor:
    state = _update_vision_state(env)
    return state.cached_status[:, 0:1]


def vision_status(env) -> torch.Tensor:
    return _update_vision_state(env).cached_status


def vision_raw_rel_pos(env) -> torch.Tensor:
    return _update_vision_state(env).cached_raw_rel_pos


def vision_raw_rel_quat(env) -> torch.Tensor:
    return _update_vision_state(env).cached_raw_rel_quat


def vision_raw_rpy_deg(env) -> torch.Tensor:
    return _update_vision_state(env).cached_raw_rpy_deg


def vision_filtered_rpy_deg(env) -> torch.Tensor:
    return _update_vision_state(env).cached_filtered_rpy_deg
