#!/usr/bin/env python
"""
| File: app_px4.py
| Description: PX4-backed transfer app with ROS 2 velocity-command bridge.
"""

from __future__ import annotations

import argparse
import atexit
import traceback
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import carb
from isaacsim import SimulationApp

HEAVE_TRAIN_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "tasks" / "manager_based" / "heave_landing" / "train_data_normalised"
).resolve()

parser = argparse.ArgumentParser(
    description="Run multiple PX4-backed vehicles and command them through ROS 2 velocity topics."
)
parser.add_argument("--num_drones", type=int, default=1, help="Number of drones to spawn.")
parser.add_argument("--gap_x_axis", type=float, default=1.0, help="Spacing between vehicles along x [m].")
parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
parser.add_argument("--namespace", type=str, default="transfer", help="ROS 2 topic namespace prefix.")
parser.add_argument(
    "--px4_offboard_baseport",
    type=int,
    default=14540,
    help="Base UDP port used by the PX4 offboard MAVLink endpoint. Each vehicle uses baseport + vehicle_id.",
)
parser.add_argument(
    "--auto_takeoff_alt",
    type=float,
    default=2.0,
    help="Automatic takeoff altitude [m]. Set to 0 to disable automatic takeoff.",
)
parser.add_argument(
    "--cmd_timeout",
    type=float,
    default=0.5,
    help="If no ROS 2 cmd_vel arrives for this many seconds, send zero velocity.",
)
parser.add_argument(
    "--arm_delay",
    type=float,
    default=5.0,
    help="Minimum delay after the first PX4 heartbeat before attempting OFFBOARD arm/takeoff.",
)
parser.add_argument(
    "--send_rate_hz",
    type=float,
    default=25.0,
    help="Rate at which offboard velocity setpoints are sent to PX4.",
)
parser.add_argument(
    "--px4_vehicle_model",
    type=str,
    default=None,
    help="Override the PX4 airframe model used by Pegasus. Defaults to PegasusInterface().px4_default_airframe.",
)
parser.add_argument("--platform_x", type=float, default=1.5, help="Platform center x-position [m].")
parser.add_argument("--platform_y", type=float, default=0.0, help="Platform center y-position [m].")
parser.add_argument("--platform_z", type=float, default=0.1, help="Platform center z-position [m].")
parser.add_argument(
    "--platform_texture",
    type=str,
    default=None,
    help="PNG texture applied to the platform top decal.",
)
parser.add_argument("--platform_static_friction", type=float, default=40.0, help="Platform static friction.")
parser.add_argument("--platform_dynamic_friction", type=float, default=40.0, help="Platform dynamic friction.")
parser.add_argument("--platform_restitution", type=float, default=0.0, help="Platform restitution.")
parser.add_argument(
    "--motion_stage",
    type=str,
    default="stationary",
    choices=("stationary", "track_xy", "track_xy_roll_pitch", "track_xy_roll_pitch_heave", "heave"),
    help="Platform motion preset matching the vanilla task stages.",
)
parser.add_argument("--platform_seed", type=int, default=0, help="Random seed for the platform motion sampler.")
parser.add_argument(
    "--heave_csv_dir",
    type=str,
    default=str(HEAVE_TRAIN_DATA_DIR),
    help="Directory containing heave CSV traces used when --motion_stage=heave.",
)
parser.add_argument(
    "--heave_sample_rate_hz",
    type=float,
    default=20.0,
    help="Sampling rate of the heave CSV traces [Hz].",
)
parser.add_argument(
    "--heave_min_remaining_s",
    type=float,
    default=60.0,
    help="Minimum remaining CSV duration after the sampled start point [s].",
)
parser.add_argument(
    "--heave_scale",
    type=float,
    default=1.0,
    help="Scale factor applied to the normalized heave CSV values.",
)
parser.add_argument(
    "--heave_bias_m",
    type=float,
    default=1.5,
    help="Constant vertical bias added to the heave trace so the platform stays above ground.",
)
parser.add_argument(
    "--disable_vision",
    action="store_true",
    help="Disable the onboard vision detector subprocess and OpenCV overlay.",
)
parser.add_argument(
    "--vision_image_topic",
    type=str,
    default="/rgb",
    help="ROS 2 image topic published by the onboard Isaac camera graph.",
)
parser.add_argument(
    "--vision_raw_pose_topic",
    type=str,
    default="/ar_pose/raw",
    help="Raw vision pose topic from the fractal detector.",
)
parser.add_argument(
    "--vision_filtered_pose_topic",
    type=str,
    default="/ar_pose/mekf_filtered",
    help="Filtered vision pose topic from the fractal detector.",
)
parser.add_argument(
    "--vision_true_pose_topic",
    type=str,
    default="/ar_pose/true",
    help="Ground-truth relative pose topic from the UAV root to the platform top center.",
)
parser.add_argument(
    "--vision_marker_size_m",
    type=float,
    default=0.70,
    help="Physical fractal marker size in meters passed to the detector.",
)
parser.add_argument(
    "--vision_config_dir",
    type=str,
    default="/home/rycker/projects/ros2_ws/src/precision_landing_using_vision/precision_landing_using_vision/config",
    help="Directory containing the fractal and camera intrinsics YAML files.",
)
parser.add_argument(
    "--vision_fractal_config_file",
    type=str,
    default="configuration_fractal_m7.yml",
    help="Fractal marker geometry YAML file name.",
)
parser.add_argument(
    "--vision_camparam_config_file",
    type=str,
    default="CamParameters_gazebo_720p.yml",
    help="Camera intrinsics YAML file name for the onboard Isaac camera.",
)
parser.add_argument(
    "--vision_workspace_setup",
    type=str,
    default="/home/rycker/projects/ros2_ws/install/setup.bash",
    help="ROS 2 workspace setup script that provides the precision_landing_using_vision package.",
)
parser.add_argument(
    "--vision_display_scale",
    type=float,
    default=0.5,
    help="Scale factor for the annotated Fractal feed window.",
)
def _str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


parser.add_argument(
    "--fractal_on",
    "--fractal-on",
    type=_str2bool,
    nargs="?",
    const=True,
    default=False,
    help="If true, open a small annotated fractal-detection feed window inside Isaac when vision starts.",
)
parser.add_argument(
    "--vision_disable_overlay_viewer",
    action="store_true",
    help="Disable the annotated fractal feed window while keeping the detector runtime active.",
)
parser.add_argument(
    "--vision_start_mode",
    type=str,
    default="after_takeoff",
    choices=("immediate", "after_takeoff"),
    help="When to start the external vision detector/viewer runtime.",
)
parser.add_argument(
    "--vision_camera_frame_skip",
    type=int,
    default=1,
    help="frameSkipCount applied to embedded ROS camera publishers; 1 means publish every 2nd rendered frame.",
)
parser.add_argument(
    "--vision_camera_queue_size",
    type=int,
    default=4,
    help="queueSize applied to embedded ROS camera publishers.",
)
parser.add_argument(
    "--vision_render_width",
    type=int,
    default=1920,
    help="Target width for the onboard camera render product used by the vision detector.",
)
parser.add_argument(
    "--vision_render_height",
    type=int,
    default=1080,
    help="Target height for the onboard camera render product used by the vision detector.",
)
parser.add_argument(
    "--vision_main_viewport_width",
    type=int,
    default=1920,
    help="Main Isaac viewport render width used when running the vision app non-headless.",
)
parser.add_argument(
    "--vision_main_viewport_height",
    type=int,
    default=1080,
    help="Main Isaac viewport render height used when running the vision app non-headless.",
)
parser.add_argument(
    "--vision_detector_camera_fps",
    type=float,
    default=20.0,
    help="Camera FPS hint passed to the external fractal detector.",
)
parser.add_argument(
    "--vision_detector_video_fps",
    type=float,
    default=2.0,
    help="Recorded video FPS passed to the external fractal detector.",
)
parser.add_argument(
    "--vision_detector_video_queue_max",
    type=int,
    default=4,
    help="Maximum detector video writer queue length.",
)
parser.add_argument(
    "--vision_camera_offset_x",
    type=float,
    default=0.0,
    help="Camera-to-UAV x offset passed to the detector transform [m].",
)
parser.add_argument(
    "--vision_camera_offset_y",
    type=float,
    default=0.0,
    help="Camera-to-UAV y offset passed to the detector transform [m].",
)
parser.add_argument(
    "--vision_camera_offset_z",
    type=float,
    default=0.0,
    help="Camera-to-UAV z offset passed to the detector transform [m].",
)

args_cli, _ = parser.parse_known_args()
if args_cli.num_drones < 1:
    parser.error("--num_drones must be greater than or equal to 1.")


def _bootstrap_isaac_ros2_python():
    """Prefer Isaac Sim's bundled ROS 2 Python packages over the system ROS install.

    Isaac Sim runs on Python 3.11 while Ubuntu 22.04 ROS Humble packages are built for Python 3.10.
    If the system ROS paths appear first in sys.path, importing rclpy fails with a missing
    `_rclpy_pybind11` extension. The ROS bridge ships a matching Python 3.11 bundle under the
    extension tree, so make that path win before any ROS package import.
    """

    ros_distro = os.environ.setdefault("ROS_DISTRO", "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    isaac_root = Path("/home/rycker/isaacsim").resolve()
    ros_bridge_root = isaac_root / "exts" / "isaacsim.ros2.bridge" / ros_distro
    ros_python_root = ros_bridge_root / "rclpy"
    ros_lib_root = ros_bridge_root / "lib"

    if ros_python_root.is_dir():
        sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
        if str(ros_python_root) not in sys.path:
            sys.path.insert(0, str(ros_python_root))

    if ros_lib_root.is_dir():
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        paths = [p for p in ld_library_path.split(":") if p]
        if str(ros_lib_root) not in paths:
            paths.append(str(ros_lib_root))
            os.environ["LD_LIBRARY_PATH"] = ":".join(paths)


_bootstrap_isaac_ros2_python()

IRIS_USD_PATH = str((Path(__file__).resolve().parents[1] / "assets" / "robots" / "iris" / "iris_cam_1080.usd").resolve())
PLATFORM_SIZE = (1.0, 1.0, 0.2)
PLATFORM_ARUCO_TEXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "Aruco" / "aruco_mark_fractal.png"
).resolve()

if args_cli.platform_texture is None:
    args_cli.platform_texture = str(PLATFORM_ARUCO_TEXTURE_PATH)


def _kill_stale_px4_instance(vehicle_id: int, px4_dir: str):
    """Remove orphan PX4 SITL processes for the requested instance before relaunch."""

    px4_binary = str((Path(px4_dir).expanduser() / "build" / "px4_sitl_default" / "bin" / "px4").resolve())
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=", "args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return

    matching_pids: list[int] = []
    instance_token = f" -i {int(vehicle_id)} "
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or px4_binary not in line or instance_token not in f" {line} ":
            continue
        try:
            pid_text, _ = line.split(" ", 1)
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            matching_pids.append(pid)

    for pid in matching_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception:
            continue

    if matching_pids:
        time.sleep(0.5)

simulation_app_config = {"headless": args_cli.headless}
if not args_cli.headless:
    simulation_app_config["width"] = max(int(args_cli.vision_main_viewport_width), 320)
    simulation_app_config["height"] = max(int(args_cli.vision_main_viewport_height), 180)
simulation_app = SimulationApp(simulation_app_config)

import omni.timeline
from omni.isaac.core.world import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.sensors.camera")

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from pymavlink import mavutil
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation
import rclpy
from std_msgs.msg import Bool

from pegasus.simulator.params import SIMULATION_ENVIRONMENTS, WORLD_SETTINGS
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rotations import rot_ENU_to_NED

try:
    from .ardupilot_ros import PlatformRos2Publisher
    from .moving_platform import CsvHeaveMotionProfile, HarmonicAxisMotionCfg, MovingPlatform, PlatformMotionStageCfg
    from .topics import cmd_vel_topic, disarm_topic, platform_pose_topic, platform_twist_topic, pose_topic, twist_inertial_topic, twist_topic
    from .vision_inprocess import InProcessFractalVisionSystem, InProcessVisionConfig
    from .vision_pose_topics import VisionPoseTopicPublisherProcess, VisionPoseTopicsConfig
except ImportError:
    from ardupilot_ros import PlatformRos2Publisher
    from moving_platform import CsvHeaveMotionProfile, HarmonicAxisMotionCfg, MovingPlatform, PlatformMotionStageCfg
    from topics import cmd_vel_topic, disarm_topic, platform_pose_topic, platform_twist_topic, pose_topic, twist_inertial_topic, twist_topic
    from vision_inprocess import InProcessFractalVisionSystem, InProcessVisionConfig
    from vision_pose_topics import VisionPoseTopicPublisherProcess, VisionPoseTopicsConfig


class PX4Ros2VelocityBridge(Backend):
    """ROS 2 backend that drives PX4 OFFBOARD velocity control from geometry_msgs/Twist."""

    def __init__(
        self,
        vehicle_id: int,
        namespace: str = "transfer",
        offboard_baseport: int = 14540,
        auto_takeoff_alt: float = 4.5,
        cmd_timeout: float = 0.5,
        arm_delay: float = 3.0,
        send_rate_hz: float = 25.0,
        num_rotors: int = 4,
    ):
        super().__init__(config=None)

        self._vehicle_id = vehicle_id
        self._namespace = namespace
        self._offboard_port = offboard_baseport + vehicle_id
        self._auto_takeoff_alt = max(float(auto_takeoff_alt), 0.0)
        self._cmd_timeout = max(float(cmd_timeout), 0.0)
        self._arm_delay = max(float(arm_delay), 0.0)
        self._send_period = 1.0 / max(float(send_rate_hz), 1.0)
        self._input_ref = [0.0 for _ in range(num_rotors)]

        self._connection = None
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._flightmode = None
        self._offboard_enabled = False
        self._armed = False
        self._pending_arm_state = None
        self._disarm_requested = False
        self._last_disarm_request_time = 0.0
        self._position_estimate_ready = False
        self._first_heartbeat_time = None
        self._last_wait_log_time = 0.0
        self._last_action_time = 0.0
        self._last_mode_request_time = 0.0
        self._last_arm_request_time = 0.0
        self._last_send_time = 0.0
        self._prestream_count = 0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"
        self._ready_since = None

        self._current_altitude = 0.0
        self._current_vertical_speed = 0.0
        self._takeoff_target_altitude = None
        self._last_takeoff_log_time = 0.0
        self._px4_ready_for_takeoff = False

        self._latest_cmd = Twist()
        self._last_cmd_time = 0.0

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(f"px4_ros2_bridge_{vehicle_id}")
        topic = cmd_vel_topic(namespace, vehicle_id)
        self._cmd_sub = self.node.create_subscription(Twist, topic, self._cmd_vel_callback, 10)
        disarm_cmd_topic = disarm_topic(namespace, vehicle_id)
        self._disarm_sub = self.node.create_subscription(Bool, disarm_cmd_topic, self._disarm_callback, 10)

        carb.log_warn(
            f"[PX4Ros2VelocityBridge] vehicle_id={vehicle_id} listening on ROS 2 topic '{topic}' "
            f"and disarm topic '{disarm_cmd_topic}' and PX4 offboard port {self._offboard_port}"
        )

    def _cmd_vel_callback(self, msg: Twist):
        self._latest_cmd = msg
        self._last_cmd_time = time.monotonic()

    def _disarm_callback(self, msg: Bool):
        if bool(msg.data):
            self._disarm_requested = True
            carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: received disarm request")

    def _drain_mavlink(self):
        if self._connection is None:
            return

        while True:
            msg = self._connection.recv_match(blocking=False)
            if msg is None:
                break

            msg_type = msg.get_type()
            if msg_type == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                self._connected = True
                self._target_system = msg.get_srcSystem()
                self._target_component = msg.get_srcComponent()
                self._connection.target_system = self._target_system
                self._connection.target_component = self._target_component
                self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    self._flightmode = mavutil.mode_string_v10(msg)
                except Exception:
                    self._flightmode = None
                self._offboard_enabled = self._flightmode == "OFFBOARD"
                if self._first_heartbeat_time is None:
                    self._first_heartbeat_time = time.monotonic()
            elif msg_type in ("LOCAL_POSITION_NED", "ODOMETRY"):
                self._position_estimate_ready = True
            elif msg_type == "COMMAND_ACK":
                command = getattr(msg, "command", None)
                result = getattr(msg, "result", None)
                if command in (
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                ):
                    carb.log_warn(
                        f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: COMMAND_ACK command={command} result={result} flightmode={self._flightmode}"
                    )
                if command == mavutil.mavlink.MAV_CMD_DO_SET_MODE and result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self._offboard_enabled = True
                if command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        if self._pending_arm_state is not None:
                            self._armed = bool(self._pending_arm_state)
                            self._pending_arm_state = None
                        if not self._armed:
                            self._disarm_requested = False
                    else:
                        self._pending_arm_state = None
            elif msg_type == "STATUSTEXT":
                text = getattr(msg, "text", "")
                if isinstance(text, bytes):
                    text = text.decode(errors="ignore")
                text = str(text).strip()
                if not text:
                    continue
                carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: STATUSTEXT: {text}")
                lowered_text = text.lower()
                if "ready for takeoff" in lowered_text:
                    self._px4_ready_for_takeoff = True
                    self._last_arm_request_time = 0.0
                elif "arming denied" in lowered_text:
                    self._px4_ready_for_takeoff = False

    def _wait_ready(self, now: float) -> bool:
        wait_reasons = []
        if not self._connected:
            wait_reasons.append("heartbeat")
        elif self._first_heartbeat_time is not None and (now - self._first_heartbeat_time) < self._arm_delay:
            wait_reasons.append(f"arm_delay {self._arm_delay:.1f}s")

        if wait_reasons and (now - self._last_wait_log_time) >= 2.0:
            carb.log_warn(
                f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: waiting for " + ", ".join(wait_reasons)
            )
            self._last_wait_log_time = now

        return not wait_reasons

    def _current_takeoff_target_altitude(self) -> float:
        if self._takeoff_target_altitude is None:
            return self._current_altitude + self._auto_takeoff_alt
        return float(self._takeoff_target_altitude)

    def _log_takeoff_status(self, now: float):
        if self._takeoff_state == "ready":
            return
        if now - self._last_takeoff_log_time < 1.0:
            return

        carb.log_warn(
            "[PX4Ros2VelocityBridge] drone%d: takeoff_state=%s alt=%.2f/%.2f vz=%.2f armed=%s offboard=%s prestream=%d"
            % (
                self._vehicle_id,
                self._takeoff_state,
                self._current_altitude,
                self._current_takeoff_target_altitude(),
                self._current_vertical_speed,
                self._armed,
                self._offboard_enabled,
                self._prestream_count,
            )
        )
        self._last_takeoff_log_time = now

    def _desired_velocity_enu(self, now: float) -> tuple[np.ndarray, float]:
        if self._takeoff_state == "ready":
            cmd_age = now - self._last_cmd_time
            if self._last_cmd_time == 0.0 or cmd_age > self._cmd_timeout:
                return np.zeros(3, dtype=np.float32), 0.0
            vel_enu = np.array(
                [
                    float(self._latest_cmd.linear.x),
                    float(self._latest_cmd.linear.y),
                    float(self._latest_cmd.linear.z),
                ],
                dtype=np.float32,
            )
            return vel_enu, -float(self._latest_cmd.angular.z)

        if self._auto_takeoff_alt <= 0.0:
            return np.zeros(3, dtype=np.float32), 0.0

        alt_error = self._current_takeoff_target_altitude() - self._current_altitude
        climb_speed = 0.0
        if alt_error > 0.15:
            climb_speed = min(1.0, max(0.25, 0.8 * alt_error))
        elif alt_error < -0.15:
            climb_speed = max(-0.6, min(-0.15, 0.8 * alt_error))

        return np.array([0.0, 0.0, climb_speed], dtype=np.float32), 0.0

    def _send_velocity_command(self, now: float):
        if not self._connected or self._target_system is None or self._target_component is None:
            return
        if now - self._last_send_time < self._send_period:
            return

        vel_enu, yaw_rate = self._desired_velocity_enu(now)
        vel_local_ned = rot_ENU_to_NED.apply(vel_enu)

        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        self._connection.mav.set_position_target_local_ned_send(
            int(now * 1000.0) & 0xFFFFFFFF,
            self._target_system,
            self._target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0.0,
            0.0,
            0.0,
            float(vel_local_ned[0]),
            float(vel_local_ned[1]),
            float(vel_local_ned[2]),
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_rate),
        )
        self._last_send_time = now
        if self._takeoff_state != "ready":
            self._prestream_count += 1

    def _request_offboard_and_arm(self, now: float):
        if not self._wait_ready(now):
            return
        if self._takeoff_state == "ready":
            return
        if self._prestream_count < max(10, int(round(1.0 / self._send_period))):
            return

        requested_action = False
        if not self._offboard_enabled and now - self._last_mode_request_time >= 1.0:
            self._connection.set_mode("OFFBOARD")
            self._last_action_time = now
            self._last_mode_request_time = now
            carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: requested OFFBOARD mode")
            requested_action = True

        arm_retry_period = 0.2 if self._px4_ready_for_takeoff else 1.0
        if not self._armed and now - self._last_arm_request_time >= arm_retry_period:
            self._connection.mav.command_long_send(
                self._target_system,
                self._target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            self._pending_arm_state = True
            self._last_action_time = now
            self._last_arm_request_time = now
            carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: retrying arm")
            requested_action = True

        if requested_action:
            return

        if self._offboard_enabled and self._armed and self._takeoff_state != "taking_off":
            self._takeoff_state = "taking_off"
            self._ready_since = None
            carb.log_warn(
                f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: armed in OFFBOARD, climbing to {self._current_takeoff_target_altitude():.2f} m"
            )

    def _request_force_disarm(self, now: float):
        if not self._disarm_requested:
            return
        if not self._connected or self._target_system is None or self._target_component is None:
            return
        if not self._armed:
            self._disarm_requested = False
            return
        if now - self._last_disarm_request_time < 1.0:
            return

        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0.0,
            21196.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._pending_arm_state = False
        self._last_disarm_request_time = now
        carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: requested force disarm")

    def _update_takeoff_state(self, now: float):
        if self._takeoff_state != "taking_off":
            return

        altitude_error = abs(self._current_altitude - self._current_takeoff_target_altitude())
        if altitude_error <= 0.15 and abs(self._current_vertical_speed) <= 0.15:
            if self._ready_since is None:
                self._ready_since = now
            elif now - self._ready_since >= 0.5:
                self._takeoff_state = "ready"
                carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: ready for velocity commands")
        else:
            self._ready_since = None

    def update_sensor(self, sensor_type: str, data):
        return

    def update_graphical_sensor(self, sensor_type: str, data):
        return

    def update_state(self, state):
        self._current_altitude = float(state.position[2])
        self._current_vertical_speed = float(state.linear_velocity[2])
        if self._auto_takeoff_alt > 0.0 and self._takeoff_target_altitude is None:
            self._takeoff_target_altitude = self._current_altitude + self._auto_takeoff_alt

    def input_reference(self):
        return self._input_ref

    def update(self, dt: float):
        if not rclpy.ok():
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception:
            return
        self._drain_mavlink()
        now = time.monotonic()
        self._send_velocity_command(now)
        self._request_force_disarm(now)
        self._request_offboard_and_arm(now)
        self._update_takeoff_state(now)
        self._log_takeoff_status(now)

    def start(self):
        self._connection = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{self._offboard_port}",
            source_system=200 + self._vehicle_id,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,
        )
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._flightmode = None
        self._offboard_enabled = False
        self._armed = False
        self._pending_arm_state = None
        self._disarm_requested = False
        self._last_disarm_request_time = 0.0
        self._position_estimate_ready = False
        self._first_heartbeat_time = None
        self._last_wait_log_time = 0.0
        self._last_action_time = 0.0
        self._last_mode_request_time = 0.0
        self._last_arm_request_time = 0.0
        self._last_send_time = 0.0
        self._prestream_count = 0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"
        self._ready_since = None
        self._takeoff_target_altitude = None
        self._last_takeoff_log_time = 0.0
        self._px4_ready_for_takeoff = False
        self._latest_cmd = Twist()
        self._last_cmd_time = 0.0

    def stop(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        try:
            self.node.destroy_node()
        except Exception:
            pass

    def reset(self):
        self._latest_cmd = Twist()
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._flightmode = None
        self._offboard_enabled = False
        self._armed = False
        self._pending_arm_state = None
        self._disarm_requested = False
        self._last_disarm_request_time = 0.0
        self._position_estimate_ready = False
        self._first_heartbeat_time = None
        self._last_wait_log_time = 0.0
        self._last_action_time = 0.0
        self._last_mode_request_time = 0.0
        self._last_arm_request_time = 0.0
        self._last_send_time = 0.0
        self._prestream_count = 0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"
        self._ready_since = None
        self._takeoff_target_altitude = None
        self._last_takeoff_log_time = 0.0
        self._px4_ready_for_takeoff = False
        self._last_cmd_time = 0.0

    @property
    def ready_for_velocity_commands(self) -> bool:
        return self._takeoff_state == "ready"


class VehicleStateRos2Publisher(Backend):
    """Publish the minimal state topics expected by the transfer policy node."""

    def __init__(self, vehicle_id: int, namespace: str = "transfer"):
        super().__init__(config=None)

        self._vehicle_id = vehicle_id
        self._namespace = namespace
        self._input_ref = [0.0, 0.0, 0.0, 0.0]

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(f"transfer_state_publisher_{vehicle_id}")
        self.pose_pub = self.node.create_publisher(PoseStamped, pose_topic(namespace, vehicle_id), 10)
        self.twist_pub = self.node.create_publisher(TwistStamped, twist_topic(namespace, vehicle_id), 10)
        self.twist_inertial_pub = self.node.create_publisher(
            TwistStamped, twist_inertial_topic(namespace, vehicle_id), 10
        )
        self.latest_state = None

        carb.log_warn(
            "[transfer.app_px4] state publisher for drone%d on %s, %s, %s"
            % (
                vehicle_id,
                pose_topic(namespace, vehicle_id),
                twist_topic(namespace, vehicle_id),
                twist_inertial_topic(namespace, vehicle_id),
            )
        )

    def update_sensor(self, sensor_type: str, data):
        return

    def update_graphical_sensor(self, sensor_type: str, data):
        return

    def update_state(self, state):
        self.latest_state = state
        stamp = self.node.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.pose.position.x = float(state.position[0])
        pose.pose.position.y = float(state.position[1])
        pose.pose.position.z = float(state.position[2])
        pose.pose.orientation.x = float(state.attitude[0])
        pose.pose.orientation.y = float(state.attitude[1])
        pose.pose.orientation.z = float(state.attitude[2])
        pose.pose.orientation.w = float(state.attitude[3])

        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = f"{self._namespace}_base_link"
        twist.twist.linear.x = float(state.linear_body_velocity[0])
        twist.twist.linear.y = float(state.linear_body_velocity[1])
        twist.twist.linear.z = float(state.linear_body_velocity[2])
        twist.twist.angular.x = float(state.angular_velocity[0])
        twist.twist.angular.y = float(state.angular_velocity[1])
        twist.twist.angular.z = float(state.angular_velocity[2])

        twist_inertial = TwistStamped()
        twist_inertial.header.stamp = stamp
        twist_inertial.header.frame_id = "map"
        twist_inertial.twist.linear.x = float(state.linear_velocity[0])
        twist_inertial.twist.linear.y = float(state.linear_velocity[1])
        twist_inertial.twist.linear.z = float(state.linear_velocity[2])

        if not rclpy.ok():
            return
        try:
            self.pose_pub.publish(pose)
            self.twist_pub.publish(twist)
            self.twist_inertial_pub.publish(twist_inertial)
        except Exception:
            return

    def input_reference(self):
        return self._input_ref

    def update(self, dt: float):
        if not rclpy.ok():
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception:
            return

    def start(self):
        return

    def stop(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass

    def reset(self):
        return


def _build_platform_stage_cfg(name: str) -> PlatformMotionStageCfg:
    if name == "stationary":
        return PlatformMotionStageCfg(
            name="stationary",
            x=HarmonicAxisMotionCfg(
                enabled=True,
                num_terms_range=(1, 1),
                amplitude_range=(0.0, 0.0),
                frequency_range_hz=(0.0, 0.0),
                phase_range_rad=(0.0, 0.0),
                spectral_decay=0.0,
            ),
            max_linear_speed=0.0,
            max_linear_acceleration=0.0,
            max_angular_speed=0.0,
            max_angular_acceleration=0.0,
        )
    track_xy = PlatformMotionStageCfg(
        name="track_xy",
        x=HarmonicAxisMotionCfg(
            enabled=True,
            num_terms_range=(2, 8),
            amplitude_range=(0.4, 0.65),
            frequency_range_hz=(0.1, 0.4),
            phase_range_rad=(0.0, 2.0 * math.pi),
            spectral_decay=1.0,
        ),
        y=HarmonicAxisMotionCfg(
            enabled=True,
            num_terms_range=(2, 8),
            amplitude_range=(0.4, 0.65),
            frequency_range_hz=(0.1, 0.4),
            phase_range_rad=(0.0, 2.0 * math.pi),
            spectral_decay=1.0,
        ),
        max_linear_speed=1.0,
        max_linear_acceleration=5.0,
    )
    if name == "track_xy":
        return track_xy
    if name == "track_xy_roll_pitch":
        return PlatformMotionStageCfg(
            name="track_xy_roll_pitch",
            x=track_xy.x,
            y=track_xy.y,
            roll=HarmonicAxisMotionCfg(
                enabled=True,
                num_terms_range=(2, 6),
                amplitude_range=(0.02, 0.10),
                frequency_range_hz=(0.05, 0.25),
                phase_range_rad=(0.0, 2.0 * math.pi),
                spectral_decay=1.0,
            ),
            pitch=HarmonicAxisMotionCfg(
                enabled=True,
                num_terms_range=(2, 6),
                amplitude_range=(0.02, 0.10),
                frequency_range_hz=(0.05, 0.25),
                phase_range_rad=(0.0, 2.0 * math.pi),
                spectral_decay=1.0,
            ),
            max_linear_speed=track_xy.max_linear_speed,
            max_linear_acceleration=track_xy.max_linear_acceleration,
            max_angular_speed=0.75,
            max_angular_acceleration=2.5,
        )
    if name == "track_xy_roll_pitch_heave":
        return PlatformMotionStageCfg(
            name="track_xy_roll_pitch_heave",
            x=track_xy.x,
            y=track_xy.y,
            z=HarmonicAxisMotionCfg(
                enabled=True,
                num_terms_range=(2, 6),
                amplitude_range=(0.02, 0.10),
                frequency_range_hz=(0.05, 0.25),
                phase_range_rad=(0.0, 2.0 * math.pi),
                spectral_decay=1.0,
            ),
            roll=HarmonicAxisMotionCfg(
                enabled=True,
                num_terms_range=(2, 6),
                amplitude_range=(0.02, 0.10),
                frequency_range_hz=(0.05, 0.25),
                phase_range_rad=(0.0, 2.0 * math.pi),
                spectral_decay=1.0,
            ),
            pitch=HarmonicAxisMotionCfg(
                enabled=True,
                num_terms_range=(2, 6),
                amplitude_range=(0.02, 0.10),
                frequency_range_hz=(0.05, 0.25),
                phase_range_rad=(0.0, 2.0 * math.pi),
                spectral_decay=1.0,
            ),
            max_linear_speed=2.25,
            max_linear_acceleration=6.0,
            max_angular_speed=0.75,
            max_angular_acceleration=2.5,
        )
    raise ValueError(f"Unknown motion stage '{name}'")


class PegasusApp:
    """Standalone app for running multiple PX4 vehicles with ROS 2 velocity command bridges."""

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world_settings = dict(WORLD_SETTINGS["px4"])
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        self.stop_sim = False
        self._shutdown_complete = False
        self._signal_count = 0
        self.velocity_bridges = []
        self.state_publishers = []
        self.px4_backends = []
        self.vision_system: InProcessFractalVisionSystem | None = None
        self._vision_system_started = False
        self.platform_motion_enabled = args_cli.motion_stage != "stationary"
        self.platform_motion_started = not self.platform_motion_enabled
        self.vision_pose_publisher: VisionPoseTopicPublisherProcess | None = None
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        atexit.register(self._shutdown)

        platform_stage_cfg = None
        platform_motion_profile = None
        if args_cli.motion_stage == "heave":
            platform_motion_profile = CsvHeaveMotionProfile(
                dataset_dir=args_cli.heave_csv_dir,
                base_position=(args_cli.platform_x, args_cli.platform_y, args_cli.platform_z),
                sample_rate_hz=args_cli.heave_sample_rate_hz,
                min_remaining_s=args_cli.heave_min_remaining_s,
                scale=args_cli.heave_scale,
                bias_m=args_cli.heave_bias_m,
                rng_seed=args_cli.platform_seed,
            )
        else:
            platform_stage_cfg = _build_platform_stage_cfg(args_cli.motion_stage)

        self.platform = MovingPlatform(
            self.world,
            texture_path=args_cli.platform_texture,
            physics_dt=float(WORLD_SETTINGS["px4"]["physics_dt"]),
            stage_cfg=platform_stage_cfg,
            motion_profile=platform_motion_profile,
            rng_seed=args_cli.platform_seed,
            size=PLATFORM_SIZE,
            base_position=(args_cli.platform_x, args_cli.platform_y, args_cli.platform_z),
            add_top_decal=True,
            top_decal_size_xy=(float(args_cli.vision_marker_size_m), float(args_cli.vision_marker_size_m)),
        )
        self._apply_platform_physics_material()
        self.world.add_physics_callback("platform_motion", self._on_platform_physics_step)
        self.platform_publishers = [
            PlatformRos2Publisher(args_cli.namespace, vehicle_id) for vehicle_id in range(args_cli.num_drones)
        ]

        for vehicle_id in range(args_cli.num_drones):
            self.vehicle_factory(vehicle_id, gap_x_axis=args_cli.gap_x_axis)

        self._configure_embedded_ros_camera_publishers(enable=not args_cli.disable_vision)
        self.world.reset()
        self._configure_embedded_ros_camera_publishers(enable=not args_cli.disable_vision)
        self._maybe_start_vision_system(force=args_cli.vision_start_mode == "immediate")
        self.platform.reset_profile()
        if args_cli.motion_stage == "heave":
            carb.log_warn(
                "[transfer.app_px4] Heave motion enabled "
                f"dataset='{self.platform.profile.current_dataset_path.name}' "
                f"start_index={self.platform.profile.current_start_index} "
                f"scale={args_cli.heave_scale:.3f} bias_m={args_cli.heave_bias_m:.3f}"
            )
        self._publish_platform_state()

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float):
        config_multirotor = MultirotorConfig()
        _kill_stale_px4_instance(vehicle_id, self.pg.px4_path)

        px4_backend = PX4MavlinkBackend(
            PX4MavlinkBackendConfig(
                {
                    "vehicle_id": vehicle_id,
                    "px4_autolaunch": True,
                    "px4_dir": self.pg.px4_path,
                    "px4_vehicle_model": args_cli.px4_vehicle_model or self.pg.px4_default_airframe,
                }
            )
        )
        self.px4_backends.append(px4_backend)

        velocity_bridge = PX4Ros2VelocityBridge(
            vehicle_id=vehicle_id,
            namespace=args_cli.namespace,
            offboard_baseport=args_cli.px4_offboard_baseport,
            auto_takeoff_alt=args_cli.auto_takeoff_alt,
            cmd_timeout=args_cli.cmd_timeout,
            arm_delay=args_cli.arm_delay,
            send_rate_hz=args_cli.send_rate_hz,
            num_rotors=config_multirotor.thrust_curve._num_rotors,
        )
        self.velocity_bridges.append(velocity_bridge)

        state_publisher = VehicleStateRos2Publisher(vehicle_id=vehicle_id, namespace=args_cli.namespace)
        self.state_publishers.append(state_publisher)

        config_multirotor.backends = [
            px4_backend,
            state_publisher,
            velocity_bridge,
        ]

        Multirotor(
            f"/World/drone{vehicle_id}",
            IRIS_USD_PATH,
            vehicle_id,
            [gap_x_axis * vehicle_id, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        carb.log_warn(
            "[transfer.app_px4] ROS topics for drone%d: %s, %s, %s, %s, %s"
            % (
                vehicle_id,
                pose_topic(args_cli.namespace, vehicle_id),
                twist_topic(args_cli.namespace, vehicle_id),
                twist_inertial_topic(args_cli.namespace, vehicle_id),
                platform_pose_topic(args_cli.namespace, vehicle_id),
                platform_twist_topic(args_cli.namespace, vehicle_id),
            )
        )

    def _apply_platform_physics_material(self):
        material_path = Sdf.Path("/World/Physics_Materials/platform_physics_material")
        stage = self.world.stage
        material = UsdShade.Material.Define(stage, material_path)

        usd_physics_material_api = UsdPhysics.MaterialAPI(material.GetPrim())
        if not usd_physics_material_api:
            usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        usd_physics_material_api.CreateStaticFrictionAttr(float(args_cli.platform_static_friction))
        usd_physics_material_api.CreateDynamicFrictionAttr(float(args_cli.platform_dynamic_friction))
        usd_physics_material_api.CreateRestitutionAttr(float(args_cli.platform_restitution))

        physx_material_api = PhysxSchema.PhysxMaterialAPI(material.GetPrim())
        if not physx_material_api:
            physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material_api.CreateFrictionCombineModeAttr("max")
        physx_material_api.CreateRestitutionCombineModeAttr("min")

        platform_prim = stage.GetPrimAtPath(self.platform.prim_path)
        material_binding_api = UsdShade.MaterialBindingAPI(platform_prim)
        material_binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

    def _find_drone_camera_prims(self) -> list[str]:
        stage = self.world.stage
        if stage is None:
            return []

        camera_paths: list[str] = []
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Camera":
                continue
            prim_path = str(prim.GetPath())
            if not prim_path.startswith("/World/drone"):
                continue
            camera_paths.append(prim_path)
        return sorted(set(camera_paths))

    def _configure_embedded_ros_camera_publishers(self, enable: bool) -> None:
        stage = self.world.stage
        if stage is None:
            return

        changed_attrs = 0
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if "ROS_Camera" not in prim_path:
                continue
            try:
                enabled_attr = prim.GetAttribute("inputs:enabled")
                if enabled_attr and enabled_attr.IsValid() and enabled_attr.Get() is not bool(enable):
                    enabled_attr.Set(bool(enable))
                    changed_attrs += 1
                if not enable:
                    continue

                frame_skip_attr = prim.GetAttribute("inputs:frameSkipCount")
                if frame_skip_attr and frame_skip_attr.IsValid():
                    desired_frame_skip = int(args_cli.vision_camera_frame_skip)
                    if frame_skip_attr.Get() != desired_frame_skip:
                        frame_skip_attr.Set(desired_frame_skip)
                        changed_attrs += 1

                queue_size_attr = prim.GetAttribute("inputs:queueSize")
                if queue_size_attr and queue_size_attr.IsValid():
                    desired_queue_size = int(args_cli.vision_camera_queue_size)
                    if queue_size_attr.Get() != desired_queue_size:
                        queue_size_attr.Set(desired_queue_size)
                        changed_attrs += 1

                node_type_attr = prim.GetAttribute("inputs:type")
                topic_name_attr = prim.GetAttribute("inputs:topicName")
                if (
                    node_type_attr
                    and node_type_attr.IsValid()
                    and topic_name_attr
                    and topic_name_attr.IsValid()
                    and str(node_type_attr.Get()) == "rgb"
                    and topic_name_attr.Get() != args_cli.vision_image_topic
                ):
                    topic_name_attr.Set(args_cli.vision_image_topic)
                    changed_attrs += 1
            except Exception:
                continue

        if changed_attrs > 0:
            carb.log_warn(
                "[transfer.app_px4] "
                f"{'Configured' if enable else 'Disabled'} embedded ROS camera node(s) "
                f"frame_skip={int(args_cli.vision_camera_frame_skip)} "
                f"queue_size={int(args_cli.vision_camera_queue_size)} "
                f"image_topic='{args_cli.vision_image_topic}'"
            )

    def _get_camera_calibration(self, camera_path: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        camera_prim = self.world.stage.GetPrimAtPath(camera_path)
        lens_distortion_model = camera_prim.GetAttribute("omni:lensdistortion:model").Get()

        if lens_distortion_model == "opencvPinhole":
            fx = float(camera_prim.GetAttribute("omni:lensdistortion:opencvPinhole:fx").Get())
            fy = float(camera_prim.GetAttribute("omni:lensdistortion:opencvPinhole:fy").Get())
            cx = float(camera_prim.GetAttribute("omni:lensdistortion:opencvPinhole:cx").Get())
            cy = float(camera_prim.GetAttribute("omni:lensdistortion:opencvPinhole:cy").Get())
            distortion = [
                float(camera_prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{name}").Get())
                for name in ("k1", "k2", "p1", "p2", "k3")
            ]
        elif lens_distortion_model == "opencvFisheye":
            fx = float(camera_prim.GetAttribute("omni:lensdistortion:opencvFisheye:fx").Get())
            fy = float(camera_prim.GetAttribute("omni:lensdistortion:opencvFisheye:fy").Get())
            cx = float(camera_prim.GetAttribute("omni:lensdistortion:opencvFisheye:cx").Get())
            cy = float(camera_prim.GetAttribute("omni:lensdistortion:opencvFisheye:cy").Get())
            distortion = [
                float(camera_prim.GetAttribute(f"omni:lensdistortion:opencvFisheye:{name}").Get())
                for name in ("k1", "k2", "k3", "k4")
            ]
            distortion.append(0.0)
        else:
            focal_length = float(camera_prim.GetAttribute("focalLength").Get())
            horizontal_aperture = float(camera_prim.GetAttribute("horizontalAperture").Get())
            vertical_aperture = float(camera_prim.GetAttribute("verticalAperture").Get())
            fx = float(width) * focal_length / horizontal_aperture
            fy = float(height) * focal_length / vertical_aperture
            if not math.isclose(fx, fy, rel_tol=1e-9, abs_tol=1e-9):
                carb.log_warn(
                    f"[transfer.app_px4] Forcing fy to fx for generated intrinsics ({fy:.3f} != {fx:.3f})"
                )
                fy = fx
            cx = float(width) * 0.5
            cy = float(height) * 0.5
            distortion = [0.0] * 5

        distortion = (distortion + [0.0] * 5)[:5]
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion_coeffs = np.array(distortion, dtype=np.float64)
        return camera_matrix, distortion_coeffs

    def _create_vision_system(self):
        if args_cli.disable_vision:
            self.vision_system = None
            carb.log_warn("[transfer.app_px4] Vision system disabled.")
            return

        camera_paths = self._find_drone_camera_prims()
        if not camera_paths:
            carb.log_warn("[transfer.app_px4] No onboard camera prim found under /World/drone*; disabling vision.")
            self.vision_system = None
            return

        if args_cli.num_drones > 1:
            carb.log_warn(
                "[transfer.app_px4] Multiple drones requested; the in-process detector is currently attached to the first camera only."
            )

        camera_path = camera_paths[0]
        fractal_config_path = (
            Path(args_cli.vision_config_dir).expanduser() / args_cli.vision_fractal_config_file
        ).resolve()
        if not fractal_config_path.is_file():
            carb.log_warn(
                f"[transfer.app_px4] Fractal config '{fractal_config_path}' does not exist; disabling vision."
            )
            self.vision_system = None
            return

        width = max(int(args_cli.vision_render_width), 1)
        height = max(int(args_cli.vision_render_height), 1)
        camera_matrix, distortion_coeffs = self._get_camera_calibration(camera_path, width, height)
        if args_cli.fractal_on and args_cli.headless:
            carb.log_warn("[transfer.app_px4] fractal_on requested, but the Fractal feed window is disabled in headless mode.")
        enable_overlay = (
            bool(args_cli.fractal_on)
            and (not args_cli.headless)
            and (not args_cli.vision_disable_overlay_viewer)
        )

        vision_cfg = InProcessVisionConfig(
            camera_prim_path=camera_path,
            image_topic=args_cli.vision_image_topic,
            marker_size_m=args_cli.vision_marker_size_m,
            fractal_config_path=str(fractal_config_path),
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coeffs,
            camera_to_uav_offset=(
                args_cli.vision_camera_offset_x,
                args_cli.vision_camera_offset_y,
                args_cli.vision_camera_offset_z,
            ),
            detector_fps=args_cli.vision_detector_camera_fps,
            resolution=(width, height),
            display_scale=args_cli.vision_display_scale,
            enable_overlay=enable_overlay,
            window_name="Fractal",
        )
        self.vision_system = InProcessFractalVisionSystem(vision_cfg)

    def _maybe_start_vision_system(self, force: bool = False):
        if self.vision_system is None:
            self._create_vision_system()
        if self.vision_system is None:
            return
        if self._vision_system_started:
            return
        if not force and args_cli.vision_start_mode == "after_takeoff":
            if not self.velocity_bridges or not self.velocity_bridges[0].ready_for_velocity_commands:
                return
        self.vision_system.start()
        self.vision_pose_publisher = VisionPoseTopicPublisherProcess(
            VisionPoseTopicsConfig(
                enabled=True,
                raw_pose_topic=args_cli.vision_raw_pose_topic,
                filtered_pose_topic=args_cli.vision_filtered_pose_topic,
                true_pose_topic=args_cli.vision_true_pose_topic,
                workspace_setup=args_cli.vision_workspace_setup,
            )
        )
        self.vision_pose_publisher.start()
        self._vision_system_started = True
        carb.log_warn(
            "[transfer.app_px4] In-process vision started "
            f"camera_prim='{self.vision_system.config.camera_prim_path}' "
            f"resolution={self.vision_system.config.resolution} "
            f"detector_fps={self.vision_system.config.detector_fps:.1f} "
            f"start_mode='{args_cli.vision_start_mode}'"
        )

    def _compute_true_ar_pose_payload(self) -> dict | None:
        if not self.state_publishers:
            return None
        vehicle_state = getattr(self.state_publishers[0], "latest_state", None)
        platform_state = self.platform.current_state
        if vehicle_state is None or platform_state is None:
            return None

        robot_pos_w = np.asarray(vehicle_state.position, dtype=np.float64)
        robot_quat_xyzw = np.asarray(vehicle_state.attitude, dtype=np.float64)
        robot_vel_w = np.asarray(vehicle_state.linear_velocity, dtype=np.float64)

        platform_quat_xyzw = np.asarray(platform_state.quat_xyzw, dtype=np.float64)
        platform_vel_w = np.asarray(platform_state.linear_velocity, dtype=np.float64)

        rot_platform = Rotation.from_quat(platform_quat_xyzw)
        platform_top_offset_w = rot_platform.apply(np.array([0.0, 0.0, 0.5 * float(PLATFORM_SIZE[2])], dtype=np.float64))
        platform_top_pos_w = np.asarray(platform_state.position, dtype=np.float64) + platform_top_offset_w
        rel_pos_w = robot_pos_w - platform_top_pos_w
        rel_pos_platform = rot_platform.inv().apply(rel_pos_w)

        rel_vel_w = robot_vel_w - platform_vel_w
        rel_vel_platform = rot_platform.inv().apply(rel_vel_w)

        rel_rot = rot_platform.inv() * Rotation.from_quat(robot_quat_xyzw)
        rel_quat_xyzw = rel_rot.as_quat().astype(np.float64)
        roll, pitch, yaw = rel_rot.as_euler("xyz", degrees=False)
        rel_rpy_deg = np.array([roll, pitch, yaw], dtype=np.float64) * (180.0 / math.pi)

        return {
            "valid": True,
            "position_m": rel_pos_platform.tolist(),
            "velocity_mps": rel_vel_platform.tolist(),
            "quat_xyzw": rel_quat_xyzw.tolist(),
            "euler_deg": rel_rpy_deg.tolist(),
        }

    def _get_true_pose_stamp(self) -> tuple[int, int]:
        if self.state_publishers:
            node = getattr(self.state_publishers[0], "node", None)
            if node is not None:
                try:
                    stamp = node.get_clock().now().to_msg()
                    return int(stamp.sec), int(stamp.nanosec)
                except Exception:
                    pass
        wall_ns = time.time_ns()
        return int(wall_ns // 1_000_000_000), int(wall_ns % 1_000_000_000)

    def _handle_signal(self, signum, _frame):
        self._signal_count += 1
        if self._signal_count == 1:
            carb.log_warn(f"[app_px4] received signal {signum}, stopping simulation")
            self.stop_sim = True
            return

        carb.log_warn(f"[app_px4] received second signal {signum}, forcing cleanup")
        self._shutdown()
        raise SystemExit(128 + signum)

    def _shutdown(self):
        if self._shutdown_complete:
            return

        self._shutdown_complete = True
        self.stop_sim = True

        for velocity_bridge in getattr(self, "velocity_bridges", []):
            try:
                velocity_bridge.stop()
            except Exception as exc:
                carb.log_warn(f"[transfer.app_px4] Failed while stopping PX4 ROS bridge: {exc}")

        for state_publisher in getattr(self, "state_publishers", []):
            try:
                state_publisher.stop()
            except Exception:
                pass

        for platform_publisher in getattr(self, "platform_publishers", []):
            try:
                platform_publisher.destroy()
            except Exception:
                pass

        for px4_backend in getattr(self, "px4_backends", []):
            try:
                px4_backend.stop()
            except Exception as exc:
                carb.log_warn(f"[transfer.app_px4] Failed while stopping PX4 backend: {exc}")

        if self.vision_system is not None:
            try:
                self.vision_system.stop()
            except Exception as exc:
                carb.log_warn(f"[transfer.app_px4] Failed while stopping vision system: {exc}")
        if self.vision_pose_publisher is not None:
            try:
                self.vision_pose_publisher.stop()
            except Exception as exc:
                carb.log_warn(f"[transfer.app_px4] Failed while stopping vision pose publisher: {exc}")

        for vehicle_id in range(args_cli.num_drones):
            try:
                _kill_stale_px4_instance(vehicle_id, self.pg.px4_path)
            except Exception:
                pass

        try:
            self.timeline.stop()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass

    def _publish_platform_state(self):
        for platform_publisher in self.platform_publishers:
            try:
                platform_publisher.publish(self.platform.current_state)
            except Exception:
                continue

    def _on_platform_physics_step(self, dt: float):
        if self.platform_motion_started:
            self.platform.update(dt)
        self._publish_platform_state()

    def run(self):
        self.timeline.play()

        try:
            while simulation_app.is_running() and not self.stop_sim:
                now_s = time.monotonic()
                render_frame = not args_cli.headless
                if self.vision_system is not None and self._vision_system_started:
                    render_frame = render_frame or self.vision_system.needs_render(now_s)
                self.world.step(render=render_frame)
                self._maybe_start_vision_system()
                estimate = None
                true_payload = None
                true_stamp_sec = 0
                true_stamp_nanosec = 0
                if self.vision_pose_publisher is not None:
                    true_payload = self._compute_true_ar_pose_payload()
                    if true_payload is not None:
                        true_stamp_sec, true_stamp_nanosec = self._get_true_pose_stamp()
                if self.vision_system is not None:
                    estimate = self.vision_system.update()
                    if estimate is not None and self.vision_pose_publisher is not None:
                        self.vision_pose_publisher.publish_estimate(
                            estimate, true_payload=true_payload
                        )
                if self.vision_pose_publisher is not None and true_payload is not None and estimate is None:
                    self.vision_pose_publisher.publish_true_pose(
                        true_payload,
                        header_stamp_sec=true_stamp_sec,
                        header_stamp_nanosec=true_stamp_nanosec,
                    )
                if self.vision_pose_publisher is not None:
                    self.vision_pose_publisher.update()
                if self.platform_motion_enabled and not self.platform_motion_started and self.velocity_bridges:
                    if self.velocity_bridges[0].ready_for_velocity_commands:
                        self.platform_motion_started = True
                        carb.log_warn("[transfer.app_px4] Platform motion enabled")
        except KeyboardInterrupt:
            self.stop_sim = True
        except BaseException as exc:
            carb.log_error(f"[transfer.app_px4] Unhandled exception in run loop: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self.stop_sim = True
        finally:
            carb.log_warn(
                "[transfer.app_px4] exiting run loop "
                f"simulation_app.is_running={simulation_app.is_running()} stop_sim={self.stop_sim}"
            )
            carb.log_warn("PegasusApp PX4 Simulation App is closing.")
            self._shutdown()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
