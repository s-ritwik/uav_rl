from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image

try:
    from .vision_omni_viewer import OmniVisionOverlayWindow
except ImportError:
    from vision_omni_viewer import OmniVisionOverlayWindow


@dataclass
class InProcessVisionConfig:
    camera_prim_path: str
    image_topic: str
    marker_size_m: float
    fractal_config_path: str
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    camera_to_uav_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detector_fps: float = 10.0
    resolution: tuple[int, int] = (640, 360)
    display_scale: float = 0.5
    enable_overlay: bool = True
    marker_timeout_s: float = 0.1
    pos_innov_threshold_m: float = 0.35
    angle_innov_threshold_deg: float = 20.0
    position_r_diagonal: float = 0.03
    q_diagonal: float = 0.01
    r_diagonal: float = 0.15
    z_measurement_scale: float = 1.0
    z_measurement_bias: float = 0.0
    z_innov_threshold_m: float = 0.12
    enable_z_output_smoother: bool = True
    z_output_smoother_tau_s: float = 0.20
    reinit_after_rejects: int = 3
    position_only_filter: bool = True
    window_name: str = "Onboard Vision"


@dataclass
class VisionPoseEstimate:
    stamp_s: float
    header_stamp_sec: int
    header_stamp_nanosec: int
    raw_valid: bool
    filtered_valid: bool
    markers_found: int
    raw_position_m: np.ndarray
    raw_rvec: np.ndarray
    raw_quat_xyzw: np.ndarray
    raw_rpy_deg: np.ndarray
    filtered_position_m: np.ndarray
    filtered_velocity_mps: np.ndarray
    filtered_quat_xyzw: np.ndarray
    filtered_rpy_deg: np.ndarray


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

        F = np.zeros((15, 15), dtype=np.float64)
        I3 = np.eye(3, dtype=np.float64)
        F[0:3, 0:3] = I3
        F[0:3, 3:6] = I3 * dt
        F[0:3, 6:9] = I3 * (0.5 * dt * dt)
        F[3:6, 3:6] = I3
        F[3:6, 6:9] = I3 * dt
        F[6:9, 6:9] = I3
        F[9:12, 9:12] = I3 - dt * self._skew(self.omega)
        F[9:12, 12:15] = I3 * dt
        F[12:15, 12:15] = I3
        self.P = F @ self.P @ F.T + self.Q

    def update_position(self, tvec_meas: np.ndarray) -> None:
        innovation_pos = tvec_meas - self.p
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3, dtype=np.float64)
        Rpos = self.R[0:3, 0:3]
        S = H @ self.P @ H.T + Rpos
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ innovation_pos
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.a += dx[6:9]
        I15 = np.eye(15, dtype=np.float64)
        IKH = I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rpos @ K.T

    def update_full(self, tvec_meas: np.ndarray, rvec_meas: np.ndarray) -> None:
        q_meas = self._quat_from_rotvec(rvec_meas)
        innovation_pos = tvec_meas - self.p
        q_err = Rotation.from_quat(q_meas) * Rotation.from_quat(self.q).inv()
        innovation_rot = q_err.as_rotvec()
        innov = np.concatenate([innovation_pos, innovation_rot], axis=0)

        H = np.zeros((6, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3, dtype=np.float64)
        H[3:6, 9:12] = np.eye(3, dtype=np.float64)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ innov

        self.p += dx[0:3]
        self.v += dx[3:6]
        self.a += dx[6:9]
        self.q = (Rotation.from_quat(self.q) * Rotation.from_rotvec(dx[9:12])).as_quat()
        self.omega += dx[12:15]

        I15 = np.eye(15, dtype=np.float64)
        IKH = I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T


class InProcessFractalVisionSystem:
    def __init__(self, config: InProcessVisionConfig):
        self.config = config
        self._marker_definitions = self._load_fractal_markers(Path(config.fractal_config_path))
        self._marker_lookup = {marker.marker_id: marker for marker in self._marker_definitions}
        self._aruco_detectors = self._build_detectors(self._marker_definitions)
        self._node = None
        self._image_sub = None
        self._stream_started = False
        self._latest_frame_bgr: np.ndarray | None = None
        self._latest_frame_seq = 0
        self._processed_frame_seq = -1
        self._last_image_wall_time_s = 0.0
        self._latest_header_stamp_sec = 0
        self._latest_header_stamp_nanosec = 0
        self._filter = _PositionOnlyMekf(
            dt=1.0 / max(float(config.detector_fps), 1.0),
            q_diagonal=float(config.q_diagonal),
            r_diagonal=float(config.r_diagonal),
            position_r_diagonal=float(config.position_r_diagonal),
        )
        self._filter_initialized = False
        self._filter_needs_reinit = False
        self._last_marker_seen_s = -1.0e9
        self._last_msg_s = 0.0
        self._last_update_valid = False
        self._consecutive_rejects = 0
        self._next_update_s = 0.0
        self._z_smoother_initialized = False
        self._z_smoother_value = 0.0
        self._z_smoother_time_s = 0.0
        self._last_no_image_log_s = 0.0
        self._overlay_available = bool(config.enable_overlay)
        self._overlay_window: OmniVisionOverlayWindow | None = None
        self.last_estimate = VisionPoseEstimate(
            stamp_s=0.0,
            header_stamp_sec=0,
            header_stamp_nanosec=0,
            raw_valid=False,
            filtered_valid=False,
            markers_found=0,
            raw_position_m=np.zeros(3, dtype=np.float64),
            raw_rvec=np.zeros(3, dtype=np.float64),
            raw_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            raw_rpy_deg=np.zeros(3, dtype=np.float64),
            filtered_position_m=np.zeros(3, dtype=np.float64),
            filtered_velocity_mps=np.zeros(3, dtype=np.float64),
            filtered_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            filtered_rpy_deg=np.zeros(3, dtype=np.float64),
        )

    @staticmethod
    def _load_fractal_markers(path: Path) -> list[_MarkerDefinition]:
        text = path.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("%YAML") and line.strip() != "---")
        text = re.sub(r"(\bid:)([^\s])", r"\1 \2", text)
        parsed = yaml.safe_load(text)
        markers = []
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

    def start(self) -> None:
        if self._stream_started:
            return
        if self.config.enable_overlay and self._overlay_window is None:
            try:
                scaled_width = int(round(self.config.resolution[0] * float(self.config.display_scale)))
                scaled_height = int(round(self.config.resolution[1] * float(self.config.display_scale)))
                self._overlay_window = OmniVisionOverlayWindow(
                    title=self.config.window_name,
                    width=scaled_width,
                    height=scaled_height,
                )
            except Exception as exc:
                self._overlay_available = False
                print(f"[vision_inprocess] disabling overlay viewer because omni.ui window creation failed: {exc}")
        try:
            rclpy.init()
        except Exception:
            pass
        self._node = rclpy.create_node(f"uav_rl_vision_detector_{int(time.time() * 1000) % 100000}")
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=max(1, int(self.config.detector_fps)),
        )
        self._image_sub = self._node.create_subscription(Image, self.config.image_topic, self._image_cb, qos_best_effort)
        self._stream_started = True
        self._next_update_s = time.monotonic()

    def needs_render(self, now_s: float | None = None) -> bool:
        if not self._stream_started:
            return False
        if now_s is None:
            now_s = time.monotonic()
        return float(now_s) >= float(self._next_update_s)

    def stop(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._node = None
        self._image_sub = None
        self._stream_started = False
        self._latest_frame_bgr = None
        if self._overlay_window is not None:
            try:
                self._overlay_window.destroy()
            except Exception:
                pass
            self._overlay_window = None

    def update(self) -> VisionPoseEstimate | None:
        if not self._stream_started or self._node is None:
            return None

        try:
            rclpy.spin_once(self._node, timeout_sec=0.0)
        except ExternalShutdownException:
            return None
        except Exception:
            if not rclpy.ok():
                return None
            raise

        now_s = time.monotonic()
        period = 1.0 / max(float(self.config.detector_fps), 1.0)
        if now_s < self._next_update_s:
            return None
        self._next_update_s = now_s + period

        image_stale = self._last_image_wall_time_s > 0.0 and (now_s - self._last_image_wall_time_s) > max(1.0, 3.0 * period)
        if self._latest_frame_bgr is None or image_stale:
            if (now_s - self._last_no_image_log_s) > 2.0:
                self._last_no_image_log_s = now_s
                print(
                    "[vision_inprocess] waiting for image topic "
                    f"'{self.config.image_topic}' last_image_age={now_s - self._last_image_wall_time_s:.2f}s"
                )
            return None
        if self._latest_frame_seq == self._processed_frame_seq:
            return None
        frame_bgr = self._latest_frame_bgr.copy()
        if frame_bgr.size == 0:
            return None
        self._processed_frame_seq = self._latest_frame_seq
        estimate, overlay = self._process_frame(frame_bgr, now_s)
        self.last_estimate = estimate
        if self.config.enable_overlay and self._overlay_available:
            self._show_overlay(overlay, estimate)
        return estimate

    def _image_cb(self, msg: Image) -> None:
        frame_bgr = self._ros_image_to_bgr(msg)
        if frame_bgr is None:
            return
        self._latest_frame_bgr = frame_bgr
        self._latest_frame_seq += 1
        self._last_image_wall_time_s = time.monotonic()
        self._latest_header_stamp_sec = int(getattr(msg.header.stamp, "sec", 0))
        self._latest_header_stamp_nanosec = int(getattr(msg.header.stamp, "nanosec", 0))

    @staticmethod
    def _ros_image_to_bgr(msg: Image) -> np.ndarray | None:
        height = int(msg.height)
        width = int(msg.width)
        if height <= 0 or width <= 0:
            return None

        encoding = str(msg.encoding).lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == "mono8":
            rows = data.reshape(height, int(msg.step))
            image = rows[:, :width]
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
        }.get(encoding)
        if channels is None:
            return None

        rows = data.reshape(height, int(msg.step))
        compact = rows[:, : width * channels]
        image = compact.reshape(height, width, channels)

        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _process_frame(self, frame_bgr: np.ndarray, now_s: float) -> tuple[VisionPoseEstimate, np.ndarray]:
        self._last_update_valid = False
        dt = now_s - self._last_msg_s
        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / max(float(self.config.detector_fps), 1.0)
        self._last_msg_s = now_s
        self._filter.dt = dt
        if self._filter_initialized:
            self._filter.predict()

        overlay = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detected_markers: list[tuple[np.ndarray, int]] = []
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []

        for bundle in self._aruco_detectors.values():
            detector = bundle["detector"]
            dictionary = bundle["dictionary"]
            params = bundle["params"]
            if detector is not None:
                corners_list, ids, _ = detector.detectMarkers(gray)
            else:
                corners_list, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            if ids is None or len(ids) == 0:
                continue

            cv2.aruco.drawDetectedMarkers(overlay, corners_list, ids, borderColor=(0, 0, 255))
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
                object_corners = marker_def.corners_normalized[:, :3] * (float(self.config.marker_size_m) * 0.5)
                object_points.extend(object_corners.astype(np.float32))
                image_points.extend(refined.astype(np.float32))
                detected_markers.append((refined, actual_id))

        detected = len(detected_markers) > 0
        raw_position = np.zeros(3, dtype=np.float64)
        raw_rvec = np.zeros(3, dtype=np.float64)
        raw_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        raw_rpy_deg = np.zeros(3, dtype=np.float64)

        if detected:
            solve_ok, rvec_marker_cam, tvec_marker_cam = cv2.solvePnP(
                np.asarray(object_points, dtype=np.float32),
                np.asarray(image_points, dtype=np.float32),
                self.config.camera_matrix.astype(np.float64),
                self.config.distortion_coefficients.astype(np.float64),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            detected = bool(solve_ok)
            if detected:
                raw_position, raw_rvec, raw_quat, raw_rpy_deg = self._transform_measurements(
                    rvec_marker_cam.reshape(3), tvec_marker_cam.reshape(3)
                )
                self._last_marker_seen_s = now_s
                self._measurement_update(raw_position, raw_rvec, now_s)
                self._draw_pose_axes(overlay, rvec_marker_cam, tvec_marker_cam)
        if not detected and (now_s - self._last_marker_seen_s) > float(self.config.marker_timeout_s):
            self._filter_needs_reinit = True
            self._filter_initialized = False
            self._consecutive_rejects = 0
            self._reset_z_output_smoother()

        filtered_valid = self._filter_initialized and (now_s - self._last_marker_seen_s) < float(self.config.marker_timeout_s)
        filtered_position = self._filter.p.copy()
        filtered_position[2] = self._apply_z_output_smoother(filtered_position[2], filtered_valid, now_s)
        filtered_velocity = self._filter.v.copy()
        filtered_quat = self._filter.q.copy()
        filtered_rpy_deg = Rotation.from_quat(filtered_quat).as_euler("xyz", degrees=True)

        estimate = VisionPoseEstimate(
            stamp_s=now_s,
            header_stamp_sec=int(self._latest_header_stamp_sec),
            header_stamp_nanosec=int(self._latest_header_stamp_nanosec),
            raw_valid=detected,
            filtered_valid=filtered_valid,
            markers_found=len(detected_markers),
            raw_position_m=raw_position,
            raw_rvec=raw_rvec,
            raw_quat_xyzw=raw_quat,
            raw_rpy_deg=raw_rpy_deg,
            filtered_position_m=filtered_position,
            filtered_velocity_mps=filtered_velocity,
            filtered_quat_xyzw=filtered_quat,
            filtered_rpy_deg=filtered_rpy_deg,
        )
        return estimate, overlay

    def _measurement_update(self, raw_position: np.ndarray, raw_rvec: np.ndarray, now_s: float) -> None:
        q_meas = Rotation.from_rotvec(raw_rvec).as_quat()
        tvec_filter_meas = raw_position.astype(np.float64).copy()
        tvec_filter_meas[2] = (
            float(self.config.z_measurement_scale) * tvec_filter_meas[2] + float(self.config.z_measurement_bias)
        )

        if self._filter_needs_reinit or not self._filter_initialized:
            self._filter.p = tvec_filter_meas
            self._filter.v[:] = 0.0
            self._filter.a[:] = 0.0
            self._filter.q = q_meas
            self._filter.omega[:] = 0.0
            self._filter.P[:] = np.eye(15, dtype=np.float64) * 0.01
            self._filter_needs_reinit = False
            self._filter_initialized = True
            self._consecutive_rejects = 0
            self._last_update_valid = True
            self._reset_z_output_smoother()
            return

        pos_innov = float(np.linalg.norm(tvec_filter_meas - self._filter.p))
        z_innov = float(abs(tvec_filter_meas[2] - self._filter.p[2]))
        accepted = False

        if self.config.position_only_filter:
            if pos_innov < float(self.config.pos_innov_threshold_m) and z_innov < float(self.config.z_innov_threshold_m):
                self._filter.update_position(tvec_filter_meas)
                self._filter.q = q_meas
                self._filter.omega[:] = 0.0
                accepted = True
        else:
            q_err = Rotation.from_quat(q_meas) * Rotation.from_quat(self._filter.q).inv()
            ang_innov_deg = float(np.linalg.norm(q_err.as_rotvec()) * 180.0 / math.pi)
            if (
                pos_innov < float(self.config.pos_innov_threshold_m)
                and z_innov < float(self.config.z_innov_threshold_m)
                and ang_innov_deg < float(self.config.angle_innov_threshold_deg)
            ):
                self._filter.update_full(tvec_filter_meas, raw_rvec)
                accepted = True

        if accepted:
            self._consecutive_rejects = 0
            self._last_update_valid = True
            return

        self._consecutive_rejects += 1
        if self._consecutive_rejects >= max(int(self.config.reinit_after_rejects), 1):
            self._filter.p = tvec_filter_meas
            self._filter.v[:] = 0.0
            self._filter.a[:] = 0.0
            self._filter.q = q_meas
            self._filter.omega[:] = 0.0
            self._filter.P[:] = np.eye(15, dtype=np.float64) * 0.01
            self._consecutive_rejects = 0
            self._filter_needs_reinit = False
            self._filter_initialized = True
            self._last_update_valid = True
            self._reset_z_output_smoother()

    def _transform_measurements(
        self, rvec_marker_cam: np.ndarray, tvec_marker_cam: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r_marker_cam, _ = cv2.Rodrigues(np.asarray(rvec_marker_cam, dtype=np.float64).reshape(3, 1))
        r_cam_marker = r_marker_cam.T
        t_cam_marker = -r_cam_marker @ np.asarray(tvec_marker_cam, dtype=np.float64).reshape(3, 1)
        offset_cam = np.asarray(self.config.camera_to_uav_offset, dtype=np.float64).reshape(3, 1)
        offset_marker = r_cam_marker @ offset_cam
        out_t = (t_cam_marker + offset_marker).reshape(3)
        r_flip = np.diag([1.0, -1.0, -1.0])
        r_uav_marker = r_flip @ r_cam_marker
        out_rvec, _ = cv2.Rodrigues(r_uav_marker)
        out_quat = Rotation.from_matrix(r_uav_marker).as_quat().astype(np.float64)
        roll, pitch, yaw = Rotation.from_matrix(r_uav_marker).as_euler("xyz", degrees=False)
        # Match the old node's raw-euler convention: x=pitch, y=roll, z=yaw.
        euler_deg = np.array([pitch, roll, yaw], dtype=np.float64) * (180.0 / math.pi)
        return out_t, out_rvec.reshape(3), out_quat, euler_deg

    def _draw_pose_axes(self, overlay: np.ndarray, rvec_marker_cam: np.ndarray, tvec_marker_cam: np.ndarray) -> None:
        try:
            axis_len = max(float(self.config.marker_size_m) * 0.25, 0.05)
            cv2.drawFrameAxes(
                overlay,
                self.config.camera_matrix.astype(np.float64),
                self.config.distortion_coefficients.astype(np.float64),
                rvec_marker_cam,
                tvec_marker_cam,
                axis_len,
                2,
            )
        except Exception:
            pass

    def _reset_z_output_smoother(self) -> None:
        self._z_smoother_initialized = False
        self._z_smoother_value = 0.0
        self._z_smoother_time_s = 0.0

    def _apply_z_output_smoother(self, z_value: float, valid: bool, stamp_s: float) -> float:
        if not self.config.enable_z_output_smoother:
            return float(z_value)
        if not valid:
            self._reset_z_output_smoother()
            return float(z_value)
        if not self._z_smoother_initialized:
            self._z_smoother_initialized = True
            self._z_smoother_value = float(z_value)
            self._z_smoother_time_s = float(stamp_s)
            return self._z_smoother_value
        dt = float(stamp_s - self._z_smoother_time_s)
        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / max(float(self.config.detector_fps), 1.0)
        alpha = float(np.clip(1.0 - math.exp(-dt / float(self.config.z_output_smoother_tau_s)), 0.0, 1.0))
        self._z_smoother_value = alpha * float(z_value) + (1.0 - alpha) * self._z_smoother_value
        self._z_smoother_time_s = float(stamp_s)
        return self._z_smoother_value

    def _show_overlay(self, overlay: np.ndarray, estimate: VisionPoseEstimate) -> None:
        try:
            lines = [
                f"markers={estimate.markers_found} raw={'yes' if estimate.raw_valid else 'no'} filtered={'yes' if estimate.filtered_valid else 'no'}",
                "raw xyz [m]: %.2f %.2f %.2f" % tuple(estimate.raw_position_m.tolist()),
                "raw p/r/y [deg]: %.1f %.1f %.1f" % tuple(estimate.raw_rpy_deg.tolist()),
                "filt xyz [m]: %.2f %.2f %.2f" % tuple(estimate.filtered_position_m.tolist()),
                "filt vel [m/s]: %.2f %.2f %.2f" % tuple(estimate.filtered_velocity_mps.tolist()),
                "filt r/p/y [deg]: %.1f %.1f %.1f" % tuple(estimate.filtered_rpy_deg.tolist()),
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    overlay,
                    line,
                    (12, 28 + 26 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            display = overlay
            if not math.isclose(float(self.config.display_scale), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                display = cv2.resize(
                    overlay,
                    None,
                    fx=float(self.config.display_scale),
                    fy=float(self.config.display_scale),
                    interpolation=cv2.INTER_AREA,
                )
            if self._overlay_window is None:
                return
            rgba = cv2.cvtColor(display, cv2.COLOR_BGR2RGBA)
            self._overlay_window.update_rgba(rgba.tobytes(), rgba.shape[1], rgba.shape[0])
        except Exception as exc:
            self._overlay_available = False
            print(f"[vision_inprocess] disabling overlay viewer because overlay update failed: {exc}")
            try:
                if self._overlay_window is not None:
                    self._overlay_window.destroy()
            except Exception:
                pass
            self._overlay_window = None
