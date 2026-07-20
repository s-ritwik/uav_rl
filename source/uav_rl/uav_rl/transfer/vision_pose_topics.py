from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from .vision_inprocess import VisionPoseEstimate
    except ImportError:
        from vision_inprocess import VisionPoseEstimate


@dataclass
class VisionPoseTopicsConfig:
    enabled: bool
    raw_pose_topic: str
    filtered_pose_topic: str
    true_pose_topic: str
    workspace_setup: str


def _ros_shell_prefix(workspace_setup: str) -> str:
    workspace_setup_path = Path(workspace_setup)
    source_cmd = (
        "unset PYTHONPATH OLD_PYTHONPATH LD_LIBRARY_PATH "
        "AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH; "
        "source /opt/ros/humble/setup.bash"
    )
    if workspace_setup_path.is_file():
        source_cmd += f" && source {workspace_setup_path}"
    return source_cmd


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


class VisionPoseTopicPublisherProcess:
    def __init__(self, config: VisionPoseTopicsConfig):
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self._exit_logged = False

    def start(self):
        if not self.config.enabled or self.process is not None:
            return
        script_path = Path(__file__).with_name("vision_pose_publisher.py").resolve()
        cmd = (
            f"{_ros_shell_prefix(self.config.workspace_setup)} && "
            f"exec /usr/bin/python3 {script_path} "
            f"--raw-topic {shlex.quote(self.config.raw_pose_topic)} "
            f"--filtered-topic {shlex.quote(self.config.filtered_pose_topic)} "
            f"--true-topic {shlex.quote(self.config.true_pose_topic)}"
        )
        self.process = subprocess.Popen(
            ["/bin/bash", "-lc", cmd],
            stdin=subprocess.PIPE,
            stdout=None,
            stderr=None,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def publish_estimate(self, estimate: "VisionPoseEstimate", true_payload: dict | None = None) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            return
        payload = {
            "header_stamp_sec": int(estimate.header_stamp_sec),
            "header_stamp_nanosec": int(estimate.header_stamp_nanosec),
            "raw": {
                "valid": bool(estimate.raw_valid),
                "position_m": estimate.raw_position_m.tolist(),
                "rvec": estimate.raw_rvec.tolist(),
                "quat_xyzw": estimate.raw_quat_xyzw.tolist(),
                "euler_deg": estimate.raw_rpy_deg.tolist(),
            },
            "filtered": {
                "valid": bool(estimate.filtered_valid),
                "position_m": estimate.filtered_position_m.tolist(),
                "velocity_mps": estimate.filtered_velocity_mps.tolist(),
                "quat_xyzw": estimate.filtered_quat_xyzw.tolist(),
                "euler_deg": estimate.filtered_rpy_deg.tolist(),
            },
        }
        if true_payload is not None:
            payload["true"] = true_payload
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            pass
        except Exception:
            pass

    def publish_true_pose(self, true_payload: dict, *, header_stamp_sec: int = 0, header_stamp_nanosec: int = 0) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            return
        payload = {
            "header_stamp_sec": int(header_stamp_sec),
            "header_stamp_nanosec": int(header_stamp_nanosec),
            "true": true_payload,
        }
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            pass
        except Exception:
            pass

    def update(self):
        if self.process is None:
            return
        code = self.process.poll()
        if code is not None and not self._exit_logged:
            print(f"[vision_pose_topics] publisher process exited with code {code}")
            self._exit_logged = True

    def stop(self):
        if self.process is not None:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
            _terminate_process_group(self.process)
        self.process = None
