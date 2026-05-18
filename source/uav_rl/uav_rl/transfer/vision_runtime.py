from __future__ import annotations

import json
import os
import selectors
import shlex
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


@dataclass
class VisionRuntimeConfig:
    enabled: bool = True
    image_topic: str = "/rgb"
    raw_pose_topic: str = "/ar_pose/raw"
    filtered_pose_topic: str = "/ar_pose/mekf_filtered"
    marker_size_m: float = 1.0
    fractal_config_file: str = "configuration_fractal_m7.yml"
    camparam_config_file: str = "CamParameters_gazebo_720p.yml"
    config_dir: str = ""
    workspace_setup: str = "/home/rycker/projects/ros2_ws/install/setup.bash"
    position_only_filter: bool = True
    visualize_detector_window: bool = False
    display_window_name: str = "Isaac Vision Feed"
    display_scale: float = 0.5
    z_measurement_scale: float = 1.0
    z_measurement_bias: float = 0.0
    z_innov_threshold: float = 0.12
    enable_z_output_smoother: bool = True
    z_output_smoother_tau: float = 0.20
    reinit_after_rejects: int = 3
    camera_to_uav_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    headless: bool = False
    detector_camera_fps: float = 10.0
    detector_video_fps: float = 2.0
    detector_video_queue_max: int = 4
    enable_video_recording: bool = False
    enable_overlay_viewer: bool = True
    detector_nice: int = 10
    detector_opencv_threads: int = 1
    detector_cpu_affinity: str = ""


@dataclass
class PoseSnapshot:
    valid: bool = False
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angle_norm_deg: float = 0.0
    stamp_sec: float = 0.0


class VisionDetectorProcess:
    """Launch the existing fractal/MEKF detector as a child process."""

    def __init__(self, config: VisionRuntimeConfig):
        self.config = config
        self.process: subprocess.Popen[Any] | None = None
        self.params_file: Path | None = None
        self._exit_logged = False

    def _write_params_file(self) -> Path:
        params = f"""\
/**:
  ros__parameters:
    camera_topic: "{self.config.image_topic}"
    camera_fps: {float(self.config.detector_camera_fps)}
    video_fps: {float(self.config.detector_video_fps)}
    video_queue_max: {int(self.config.detector_video_queue_max)}
    enable_video_recording: {str(bool(self.config.enable_video_recording)).lower()}
    fractal_config_file: "{self.config.fractal_config_file}"
    camparam_config_file: "{self.config.camparam_config_file}"
    marker_size: {float(self.config.marker_size_m)}
    visualize_marker: {str(bool(self.config.visualize_detector_window)).lower()}
    position_only_filter: {str(bool(self.config.position_only_filter)).lower()}
    z_measurement_scale: {float(self.config.z_measurement_scale)}
    z_measurement_bias: {float(self.config.z_measurement_bias)}
    z_innov_threshold: {float(self.config.z_innov_threshold)}
    enable_z_output_smoother: {str(bool(self.config.enable_z_output_smoother)).lower()}
    z_output_smoother_tau: {float(self.config.z_output_smoother_tau)}
    reinit_after_rejects: {int(self.config.reinit_after_rejects)}
    camera_to_uav_offset: [{self.config.camera_to_uav_offset[0]}, {self.config.camera_to_uav_offset[1]}, {self.config.camera_to_uav_offset[2]}]
"""
        if self.config.config_dir:
            params += f'    config_dir: "{self.config.config_dir}"\n'

        fd, path = tempfile.mkstemp(prefix="uav_rl_vision_", suffix=".yaml")
        os.close(fd)
        params_path = Path(path)
        params_path.write_text(params, encoding="utf-8")
        return params_path

    def _ros_shell_prefix(self) -> str:
        workspace_setup = Path(self.config.workspace_setup)
        source_cmd = (
            "unset PYTHONPATH OLD_PYTHONPATH LD_LIBRARY_PATH "
            "AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH; "
            "source /opt/ros/humble/setup.bash"
        )
        if workspace_setup.is_file():
            source_cmd += f" && source {workspace_setup}"
        return source_cmd

    def _detector_executable(self) -> str:
        workspace_setup = Path(self.config.workspace_setup)
        install_prefix = workspace_setup.parent if workspace_setup.name == "setup.bash" else workspace_setup
        detector_path = (
            install_prefix
            / "precision_landing_using_vision"
            / "lib"
            / "precision_landing_using_vision"
            / "fractal_pose_mekf_video_save_node"
        )
        return str(detector_path)

    def _detector_cpu_affinity(self) -> str:
        if self.config.detector_cpu_affinity:
            return str(self.config.detector_cpu_affinity)
        cpu_count = os.cpu_count() or 1
        if cpu_count < 8:
            return ""
        start = max(cpu_count - 4, 0)
        return f"{start}-{cpu_count - 1}"

    def start(self):
        if not self.config.enabled:
            return

        self.params_file = self._write_params_file()
        detector_executable = self._detector_executable()
        detector_affinity = self._detector_cpu_affinity()
        detector_threads = max(int(self.config.detector_opencv_threads), 1)
        launch_prefix = (
            "export "
            f"OMP_NUM_THREADS={detector_threads} "
            f"OPENBLAS_NUM_THREADS={detector_threads} "
            f"MKL_NUM_THREADS={detector_threads} "
            f"NUMEXPR_NUM_THREADS={detector_threads} "
            f"VECLIB_MAXIMUM_THREADS={detector_threads} "
            f"OPENCV_FOR_THREADS_NUM={detector_threads} "
            "OPENCV_OPENCL_RUNTIME=disabled; "
        )
        priority_cmd = ""
        if int(self.config.detector_nice) != 0:
            priority_cmd += f"nice -n {int(self.config.detector_nice)} "
        if detector_affinity:
            priority_cmd += f"taskset -c {shlex.quote(detector_affinity)} "
        cmd = (
            f"{self._ros_shell_prefix()} && "
            f"{launch_prefix}"
            f"exec {priority_cmd}{detector_executable} "
            f"--ros-args --params-file {self.params_file}"
        )
        print(
            "[vision_runtime] starting detector "
            f"nice={int(self.config.detector_nice)} "
            f"threads={detector_threads} "
            f"affinity='{detector_affinity or 'default'}'"
        )
        self.process = subprocess.Popen(
            ["/bin/bash", "-lc", cmd],
            stdout=None,
            stderr=None,
            start_new_session=True,
        )

    def update(self):
        if self.process is None:
            return
        code = self.process.poll()
        if code is not None and not self._exit_logged:
            print(f"[vision_runtime] detector process exited with code {code}")
            self._exit_logged = True

    def stop(self):
        if self.process is not None:
            _terminate_process_group(self.process)
        self.process = None
        if self.params_file and self.params_file.exists():
            try:
                self.params_file.unlink()
            except Exception:
                pass
        self.params_file = None


class VisionPoseRelayProcess:
    """Subscribe to custom vision messages in the ROS 2 workspace Python and relay JSON to this app."""

    def __init__(self, config: VisionRuntimeConfig):
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.selector = selectors.DefaultSelector()
        self.raw_pose = PoseSnapshot()
        self.filtered_pose = PoseSnapshot()
        self._exit_logged = False

    def start(self):
        if not self.config.enabled:
            return

        bridge_script = Path(__file__).with_name("vision_pose_bridge.py").resolve()
        cmd = (
            f"{VisionDetectorProcess(self.config)._ros_shell_prefix()} && "
            f"exec /usr/bin/python3 {bridge_script} "
            f"--raw-topic {self.config.raw_pose_topic} "
            f"--filtered-topic {self.config.filtered_pose_topic}"
        )
        self.process = subprocess.Popen(
            ["/bin/bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if self.process.stdout is not None:
            self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def _snapshot_from_payload(self, payload: dict[str, Any]) -> PoseSnapshot:
        return PoseSnapshot(
            valid=bool(payload.get("valid", False)),
            pos=tuple(float(v) for v in payload.get("pos", (0.0, 0.0, 0.0))),
            vel=tuple(float(v) for v in payload.get("vel", (0.0, 0.0, 0.0))),
            euler_deg=tuple(float(v) for v in payload.get("euler_deg", (0.0, 0.0, 0.0))),
            angle_norm_deg=float(payload.get("angle_norm_deg", 0.0)),
            stamp_sec=float(payload.get("stamp_sec", 0.0)),
        )

    def update(self):
        if self.process is None:
            return

        for key, _mask in self.selector.select(timeout=0.0):
            line = key.fileobj.readline()
            while line:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    line = key.fileobj.readline()
                    continue
                snapshot = self._snapshot_from_payload(payload)
                if payload.get("stream") == "raw":
                    self.raw_pose = snapshot
                elif payload.get("stream") == "filtered":
                    self.filtered_pose = snapshot
                line = key.fileobj.readline()

        code = self.process.poll()
        if code is not None and not self._exit_logged:
            print(f"[vision_runtime] pose relay process exited with code {code}")
            self._exit_logged = True

    def stop(self):
        if self.process is not None:
            if self.process.stdout is not None:
                try:
                    self.selector.unregister(self.process.stdout)
                except Exception:
                    pass
            _terminate_process_group(self.process)
        self.process = None


class VisionViewerProcess:
    """Open the annotated OpenCV window in a separate ROS 2 Python process."""

    def __init__(self, config: VisionRuntimeConfig):
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self._exit_logged = False

    def start(self):
        if not self.config.enabled or self.config.headless or not self.config.enable_overlay_viewer:
            return

        viewer_script = Path(__file__).with_name("vision_overlay_viewer.py").resolve()
        cmd = (
            f"{VisionDetectorProcess(self.config)._ros_shell_prefix()} && "
            f"exec /usr/bin/python3 {viewer_script} "
            f"--image-topic {self.config.image_topic} "
            f"--raw-topic {self.config.raw_pose_topic} "
            f"--filtered-topic {self.config.filtered_pose_topic} "
            f"--display-scale {float(self.config.display_scale)} "
            f"--window-name \"{self.config.display_window_name}\""
        )
        self.process = subprocess.Popen(
            ["/bin/bash", "-lc", cmd],
            stdout=None,
            stderr=None,
            start_new_session=True,
        )

    def update(self):
        if self.process is None:
            return
        code = self.process.poll()
        if code is not None and not self._exit_logged:
            print(f"[vision_runtime] viewer process exited with code {code}")
            self._exit_logged = True

    def stop(self):
        if self.process is not None:
            _terminate_process_group(self.process)
        self.process = None


def _terminate_process_group(process: subprocess.Popen[Any], timeout_s: float = 3.0):
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


class VisionOverlay:
    """Subscribe to the image topic and show an annotated OpenCV window."""

    def __init__(self, config: VisionRuntimeConfig):
        self.config = config
        self.node = None
        self.latest_frame_bgr: np.ndarray | None = None
        self.raw_pose = PoseSnapshot()
        self.filtered_pose = PoseSnapshot()
        self._window_initialized = False

        if not self.config.enabled or self.config.headless:
            return

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node("transfer_vision_overlay")
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.node.create_subscription(Image, self.config.image_topic, self._image_cb, qos_best_effort)

    def _image_cb(self, msg: Image):
        self.latest_frame_bgr = self._ros_image_to_bgr(msg)

    def _ros_image_to_bgr(self, msg: Image) -> np.ndarray | None:
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

    def update(self):
        if self.node is None:
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._render()

    def _format_pose_line(self, prefix: str, pose: PoseSnapshot) -> str:
        valid_text = "OK" if pose.valid else "MISS"
        return (
            f"{prefix} [{valid_text}] "
            f"x={pose.pos[0]:+.2f} y={pose.pos[1]:+.2f} z={pose.pos[2]:+.2f} "
            f"roll={pose.euler_deg[0]:+.1f} pitch={pose.euler_deg[1]:+.1f} yaw={pose.euler_deg[2]:+.1f}"
        )

    def _render(self):
        if self.latest_frame_bgr is None:
            return

        frame = self.latest_frame_bgr.copy()
        overlay_lines = [
            self._format_pose_line("RAW ", self.raw_pose),
            self._format_pose_line("MEKF", self.filtered_pose),
        ]
        color_raw = (0, 220, 0) if self.raw_pose.valid else (0, 0, 255)
        color_filt = (255, 200, 0) if self.filtered_pose.valid else (0, 0, 255)
        colors = [color_raw, color_filt]

        y = 28
        for line, color in zip(overlay_lines, colors):
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 28

        if self.config.display_scale != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=float(self.config.display_scale),
                fy=float(self.config.display_scale),
                interpolation=cv2.INTER_AREA,
            )

        if not self._window_initialized:
            cv2.namedWindow(self.config.display_window_name, cv2.WINDOW_NORMAL)
            self._window_initialized = True

        cv2.imshow(self.config.display_window_name, frame)
        cv2.waitKey(1)

    def stop(self):
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None
        if self._window_initialized:
            try:
                cv2.destroyWindow(self.config.display_window_name)
            except Exception:
                pass
            self._window_initialized = False


class VisionRuntime:
    def __init__(self, config: VisionRuntimeConfig):
        self.config = config
        self.detector = VisionDetectorProcess(config)
        self.pose_relay = VisionPoseRelayProcess(config)
        self.viewer = VisionViewerProcess(config)

    def start(self):
        self.detector.start()
        self.pose_relay.start()
        self.viewer.start()

    def update(self):
        self.detector.update()
        self.pose_relay.update()
        self.viewer.update()

    def stop(self):
        self.viewer.stop()
        self.pose_relay.stop()
        self.detector.stop()
