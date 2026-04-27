#!/usr/bin/env python
"""
| File: 13_ardu_ros.py
| Description: Multi-vehicle ArduPilot example with a ROS 2 velocity-command bridge.
"""

import argparse
import atexit
import os
import signal
import subprocess
import time
from pathlib import Path
import numpy as np
import math
# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Run multiple ArduPilot-backed vehicles and command them through ROS 2 velocity topics."
)
parser.add_argument("--num_drones", type=int, default=1, help="Number of drones to spawn.")
parser.add_argument("--gap_x_axis", type=float, default=1.0, help="Spacing between vehicles along x [m].")
parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
parser.add_argument("--namespace", type=str, default="transfer", help="ROS 2 topic namespace prefix.")
parser.add_argument(
    "--bridge_baseport",
    type=int,
    default=14650,
    help="Base UDP port used by the ROS 2 MAVLink bridge. Each vehicle uses baseport + 10 * vehicle_id.",
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
    default=3.0,
    help="Minimum delay after the first ArduPilot heartbeat before attempting GUIDED arm/takeoff.",
)
parser.add_argument(
    "--strict_prearm",
    action="store_true",
    help="Keep ArduPilot's default EKF/pre-arm startup instead of using the fast SITL AHRS mode.",
)
parser.add_argument(
    "--cleanup",
    action="store_true",
    help="Kill ArduPilot SITL, sim_vehicle.py, and MAVProxy processes and exit.",
)
parser.add_argument(
    "--show_ardupilot_ui",
    action="store_true",
    help="Launch each ArduPilot instance in its own terminal with MAVProxy console and map.",
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
    choices=("stationary", "track_xy", "track_xy_roll_pitch", "track_xy_roll_pitch_heave"),
    help="Platform motion preset matching the vanilla task stages.",
)
parser.add_argument("--platform_seed", type=int, default=0, help="Random seed for the platform motion sampler.")

args_cli, _ = parser.parse_known_args()
if args_cli.num_drones < 1:
    parser.error("--num_drones must be greater than or equal to 1.")

FAST_START_PARAM_TEXT = "\n".join(
    [
        "AHRS_EKF_TYPE 10",
        "EK2_ENABLE 0",
        "EK3_ENABLE 0",
        "",
    ]
)

IRIS_USD_PATH = str((Path(__file__).resolve().parents[1] / "assets" / "robots" / "iris" / "iris_capsule.usd").resolve())
PLATFORM_SIZE = (1.0, 1.0, 0.2)
PLATFORM_ARUCO_TEXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "Aruco" / "aruco_mark_fractal.png"
).resolve()

if args_cli.platform_texture is None:
    args_cli.platform_texture = str(PLATFORM_ARUCO_TEXTURE_PATH)


def _matching_ardupilot_process(cmdline: str) -> bool:
    cmdline_lower = cmdline.lower()
    return (
        "sim_vehicle.py" in cmdline_lower
        or "mavproxy.py" in cmdline_lower
        or "arducopter" in cmdline_lower
    )


def _get_process_table():
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.SubprocessError:
        return {}

    table = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        table[pid] = {"ppid": ppid, "cmdline": parts[2]}
    return table


def _descendants(process_table, root_pid: int):
    children = {}
    for pid, info in process_table.items():
        children.setdefault(info["ppid"], []).append(pid)

    ordered = []
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        stack.extend(children.get(pid, []))
    return list(reversed(ordered))


def _terminate_process_tree(root_pid: int, timeout: float = 5.0, process_table=None) -> int:
    if process_table is None:
        process_table = _get_process_table()

    if root_pid not in process_table:
        return 0

    targets = _descendants(process_table, root_pid)
    if not targets:
        return 0

    try:
        os.killpg(os.getpgid(root_pid), signal.SIGINT)
    except OSError:
        for pid in targets:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [pid for pid in targets if os.path.exists(f"/proc/{pid}")]
        if not alive:
            return len(targets)
        time.sleep(0.1)

    try:
        os.killpg(os.getpgid(root_pid), signal.SIGKILL)
    except OSError:
        for pid in targets:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    time.sleep(0.2)
    return len(targets)


def cleanup_ardupilot_processes() -> int:
    process_table = _get_process_table()
    root_pids = [
        pid for pid, info in process_table.items() if _matching_ardupilot_process(info["cmdline"])
    ]

    cleaned = 0
    for pid in sorted(set(root_pids)):
        cleaned += _terminate_process_tree(pid, timeout=2.0, process_table=process_table)

    return cleaned


if args_cli.cleanup:
    cleaned = cleanup_ardupilot_processes()
    print(f"[13_ardu_ros] cleanup complete, terminated {cleaned} process entries.")
    raise SystemExit(0)

simulation_app = SimulationApp({"headless": args_cli.headless})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from pymavlink import mavutil
from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, WORLD_SETTINGS
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend,
    ArduPilotMavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.tools.ardupilot_launch_tool import ArduPilotLaunchTool
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rotations import rot_ENU_to_NED

from scipy.spatial.transform import Rotation
import rclpy
from std_msgs.msg import Bool

try:
    from .ardupilot_ros import PlatformRos2Publisher
    from .moving_platform import HarmonicAxisMotionCfg, MovingPlatform, PlatformMotionStageCfg
    from .topics import (
        cmd_vel_topic,
        disarm_topic,
        platform_pose_topic,
        platform_twist_topic,
        pose_topic,
        twist_inertial_topic,
        twist_topic,
    )
except ImportError:
    from ardupilot_ros import PlatformRos2Publisher
    from moving_platform import HarmonicAxisMotionCfg, MovingPlatform, PlatformMotionStageCfg
    from topics import (
        cmd_vel_topic,
        disarm_topic,
        platform_pose_topic,
        platform_twist_topic,
        pose_topic,
        twist_inertial_topic,
        twist_topic,
    )


class MultiOutArduPilotLaunchTool(ArduPilotLaunchTool):
    """Launch ArduPilot SITL with two MAVProxy outputs: one for Pegasus, one for the ROS bridge."""

    def __init__(
        self,
        ardupilot_dir,
        vehicle_id: int,
        ardupilot_model: str,
        out_ports,
        fast_start: bool = True,
        show_ui: bool = False,
    ):
        super().__init__(ardupilot_dir, vehicle_id, ardupilot_model)
        self.out_ports = list(out_ports)
        self.fast_start = fast_start
        self.show_ui = show_ui
        self._fast_start_param_file = None

    def _ensure_fast_start_param_file(self):
        if self._fast_start_param_file is not None:
            return self._fast_start_param_file

        param_path = os.path.join(self.root_fs.name, f"pegasus_fast_start_{self.vehicle_id}.parm")
        with open(param_path, "w", encoding="ascii") as param_file:
            param_file.write(FAST_START_PARAM_TEXT)
        self._fast_start_param_file = param_path
        return self._fast_start_param_file

    def launch_ardupilot(self):
        command = [
            "python3",
            f"{self.ardupilot_dir}/Tools/autotest/sim_vehicle.py",
            "-v",
            "ArduCopter",
            "-f",
            self._get_vehicle_frame(),
            "--model",
            self.model,
            "-I",
            str(self.vehicle_id),
            "--sysid",
            str(self.vehicle_id + 1),
        ]

        if self._sitl_already_exists():
            command.append("--no-rebuild")

        if self.fast_start:
            command.extend(["--add-param-file", self._ensure_fast_start_param_file()])

        if self.show_ui:
            command.extend(["--console", "--map"])

        for port in self.out_ports:
            command.extend(["--out", f"udp:127.0.0.1:{port}"])

        if self.show_ui:
            command_str = " ".join(command)
            self.ardupilot_process = subprocess.Popen(
                ["gnome-terminal", "--", "bash", "-lc", command_str],
                cwd=self.root_fs.name,
                shell=False,
                env=self.environment,
                preexec_fn=os.setsid,
            )
            return

        self.ardupilot_process = subprocess.Popen(
            command,
            cwd=self.root_fs.name,
            shell=False,
            env=self.environment,
            preexec_fn=os.setsid,
        )

    def kill_ardupilot(self):
        if self.ardupilot_process is None:
            return

        _terminate_process_tree(self.ardupilot_process.pid, timeout=5.0)
        self.ardupilot_process = None


class ArduPilotRos2VelocityBridge(Backend):
    """ROS 2 backend that forwards geometry_msgs/Twist commands to ArduPilot GUIDED velocity control."""

    def __init__(
        self,
        vehicle_id: int,
        namespace: str = "transfer",
        bridge_baseport: int = 14650,
        auto_takeoff_alt: float = 2.0,
        cmd_timeout: float = 0.5,
        send_rate_hz: float = 25.0,
        num_rotors: int = 4,
    ):
        super().__init__(config=None)

        self._vehicle_id = vehicle_id
        self._namespace = namespace
        self._bridge_port = bridge_baseport + vehicle_id * 10
        self._auto_takeoff_alt = max(float(auto_takeoff_alt), 0.0)
        self._cmd_timeout = max(float(cmd_timeout), 0.0)
        self._send_period = 1.0 / max(float(send_rate_hz), 1.0)
        self._input_ref = [0.0 for _ in range(num_rotors)]
        self._arm_delay = max(float(args_cli.arm_delay), 0.0)
        self._require_position_ready = bool(args_cli.strict_prearm)

        self._connection = None
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._armed = False
        self._current_altitude = 0.0
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_wait_log_time = 0.0
        self._disarm_requested = False
        self._last_disarm_request_time = 0.0

        self._latest_cmd = Twist()
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(f"ardupilot_ros2_bridge_{vehicle_id}")
        topic = cmd_vel_topic(namespace, vehicle_id)
        self._cmd_sub = self.node.create_subscription(Twist, topic, self._cmd_vel_callback, 10)
        disarm_cmd_topic = disarm_topic(namespace, vehicle_id)
        self._disarm_sub = self.node.create_subscription(Bool, disarm_cmd_topic, self._disarm_callback, 10)

        carb.log_warn(
            f"[ArduPilotRos2VelocityBridge] vehicle_id={vehicle_id} listening on ROS 2 topic "
            f"'{topic}' and disarm topic '{disarm_cmd_topic}' and MAVLink bridge port {self._bridge_port}"
        )

    def _cmd_vel_callback(self, msg: Twist):
        self._latest_cmd = msg
        self._last_cmd_time = time.monotonic()

    def _disarm_callback(self, msg: Bool):
        if bool(msg.data):
            self._disarm_requested = True
            self._latest_cmd = Twist()
            now = time.monotonic()
            if self._connected and self._target_system is not None and self._target_component is not None:
                self._send_force_disarm(now)
            carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: received disarm request")

    def _drain_mavlink(self):
        if self._connection is None:
            return

        while True:
            msg = self._connection.recv_match(blocking=False)
            if msg is None:
                break

            if msg.get_type() == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                self._connected = True
                self._target_system = msg.get_srcSystem()
                self._target_component = msg.get_srcComponent()
                self._armed = bool(
                    getattr(msg, "base_mode", 0) & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                if self._first_heartbeat_time is None:
                    self._first_heartbeat_time = time.monotonic()
                if self._takeoff_state == "disarming" and not self._armed:
                    self._disarm_requested = False
                    self._takeoff_state = "disarmed"
                    carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: disarmed")
            elif msg.get_type() == "GPS_RAW_INT":
                self._gps_fix_ready = getattr(msg, "fix_type", 0) >= 3
            elif msg.get_type() in ("GLOBAL_POSITION_INT", "LOCAL_POSITION_NED"):
                self._position_estimate_ready = True

    def _send_command_long(self, command, params):
        if self._connection is None or self._target_system is None or self._target_component is None:
            return

        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            command,
            0,
            *params,
        )

    def _update_takeoff_state(self, now: float):
        if not self._connected:
            return

        if self._takeoff_state == "ready":
            return

        if self._first_heartbeat_time is None:
            return

        wait_reasons = []
        if (now - self._first_heartbeat_time) < self._arm_delay:
            wait_reasons.append(f"arm_delay {self._arm_delay:.1f}s")
        if not self._gps_fix_ready:
            wait_reasons.append("gps_fix")
        if self._require_position_ready and not self._position_estimate_ready:
            wait_reasons.append("position_estimate")

        if wait_reasons:
            if now - self._last_wait_log_time >= 2.0:
                carb.log_warn(
                    f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: waiting for "
                    + ", ".join(wait_reasons)
                )
                self._last_wait_log_time = now
            return

        # Step through GUIDED -> ARM -> TAKEOFF with light retries.
        if self._takeoff_state == "pending":
            self._connection.set_mode_apm("GUIDED")
            self._takeoff_state = "arming"
            self._last_action_time = now
            carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: requested GUIDED mode")
            return

        if self._takeoff_state == "arming":
            if self._connection.motors_armed():
                self._send_command_long(
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._auto_takeoff_alt],
                )
                self._takeoff_state = "taking_off"
                self._last_action_time = now
                carb.log_warn(
                    f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: armed, taking off to {self._auto_takeoff_alt:.2f} m"
                )
                return

            if now - self._last_action_time >= 1.0:
                self._connection.set_mode_apm("GUIDED")
                self._connection.arducopter_arm()
                self._last_action_time = now
                carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: retrying arm")
            return

        if self._takeoff_state == "taking_off":
            if self._current_altitude >= 0.8 * self._auto_takeoff_alt:
                self._takeoff_state = "ready"
                carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: ready for velocity commands")
                return

            if now - self._last_action_time >= 2.0:
                self._send_command_long(
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._auto_takeoff_alt],
                )
                self._last_action_time = now

    def _send_force_disarm(self, now: float):
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
        self._takeoff_state = "disarming"
        self._last_disarm_request_time = now
        carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: requested force disarm")

    def _request_disarm(self, now: float):
        if not self._disarm_requested:
            return
        if not self._connected or self._target_system is None or self._target_component is None:
            return
        if not self._armed:
            self._disarm_requested = False
            self._takeoff_state = "disarmed"
            return
        if now - self._last_disarm_request_time < 0.1:
            return
        self._send_force_disarm(now)

    def _send_velocity_command(self, now: float):
        if not self._connected or self._takeoff_state != "ready":
            return

        if now - self._last_send_time < self._send_period:
            return

        cmd_age = now - self._last_cmd_time
        if self._last_cmd_time == 0.0 or cmd_age > self._cmd_timeout:
            vx = 0.0
            vy = 0.0
            vz = 0.0
            yaw_rate = 0.0
        else:
            # ROS Twist is interpreted as world-frame ENU velocity.
            # Convert it once into ArduPilot's LOCAL_NED frame.
            vel_enu = np.array(
                [
                    float(self._latest_cmd.linear.x),
                    float(self._latest_cmd.linear.y),
                    float(self._latest_cmd.linear.z),
                ]
            )
            vel_local_ned = rot_ENU_to_NED.apply(vel_enu)
            vx = float(vel_local_ned[0])
            vy = float(vel_local_ned[1])
            vz = float(vel_local_ned[2])
            yaw_rate = -float(self._latest_cmd.angular.z)

        # Match ArduPilot's known-good guided velocity path: position ignored,
        # velocity provided, acceleration explicitly zeroed, yaw ignored,
        # yaw-rate optionally commanded.
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
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
            vx,
            vy,
            vz,
            0.0,
            0.0,
            0.0,
            0.0,
            yaw_rate,
        )
        self._last_send_time = now

    def update_sensor(self, sensor_type: str, data):
        return

    def update_graphical_sensor(self, sensor_type: str, data):
        return

    def update_state(self, state):
        self._current_altitude = float(state.position[2])

    def input_reference(self):
        return self._input_ref

    def update(self, dt: float):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._drain_mavlink()

        now = time.monotonic()
        self._request_disarm(now)
        self._update_takeoff_state(now)
        self._send_velocity_command(now)

    def start(self):
        self._connection = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{self._bridge_port}",
            source_system=200 + self._vehicle_id,
        )
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._armed = False
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._last_wait_log_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"
        self._disarm_requested = False
        self._last_disarm_request_time = 0.0

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
        self._armed = False
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._last_wait_log_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"
        self._disarm_requested = False
        self._last_disarm_request_time = 0.0

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

        carb.log_warn(
            "[transfer.app] state publisher for drone%d on %s, %s, %s"
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

        self.pose_pub.publish(pose)
        self.twist_pub.publish(twist)
        self.twist_inertial_pub.publish(twist_inertial)

    def input_reference(self):
        return self._input_ref

    def update(self, dt: float):
        rclpy.spin_once(self.node, timeout_sec=0.0)

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
        return PlatformMotionStageCfg(name="stationary")

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
    """Standalone app for running multiple ArduPilot vehicles with ROS 2 velocity command bridges."""

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world_settings = dict(WORLD_SETTINGS["ardupilot"])
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        self.stop_sim = False
        self._shutdown_complete = False
        self._signal_count = 0
        self.ardupilot_tools = []
        self.ardupilot_started = False
        self.velocity_bridges = []
        self.state_publishers = []
        self.platform_motion_enabled = args_cli.motion_stage != "stationary"
        self.platform_motion_started = not self.platform_motion_enabled

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        atexit.register(self._shutdown)

        self.platform = MovingPlatform(
            self.world,
            texture_path=args_cli.platform_texture,
            physics_dt=float(WORLD_SETTINGS["ardupilot"]["physics_dt"]),
            stage_cfg=_build_platform_stage_cfg(args_cli.motion_stage),
            rng_seed=args_cli.platform_seed,
            size=PLATFORM_SIZE,
            base_position=(args_cli.platform_x, args_cli.platform_y, args_cli.platform_z),
            add_top_decal=True,
        )
        self._apply_platform_physics_material()
        self.world.add_physics_callback("platform_motion", self._on_platform_physics_step)
        self.platform_publishers = [
            PlatformRos2Publisher(args_cli.namespace, vehicle_id) for vehicle_id in range(args_cli.num_drones)
        ]

        for vehicle_id in range(args_cli.num_drones):
            self.vehicle_factory(vehicle_id, gap_x_axis=args_cli.gap_x_axis)

        self.world.reset()
        self.platform.reset_profile()
        self._publish_platform_state()

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float):
        config_multirotor = MultirotorConfig()

        backend_config = ArduPilotMavlinkBackendConfig(
            {
                "vehicle_id": vehicle_id,
                "ardupilot_autolaunch": False,
                "ardupilot_dir": self.pg.ardupilot_path,
                "ardupilot_vehicle_model": "gazebo-iris",
            }
        )

        velocity_bridge = ArduPilotRos2VelocityBridge(
            vehicle_id=vehicle_id,
            namespace=args_cli.namespace,
            bridge_baseport=args_cli.bridge_baseport,
            auto_takeoff_alt=args_cli.auto_takeoff_alt,
            cmd_timeout=args_cli.cmd_timeout,
            num_rotors=config_multirotor.thrust_curve._num_rotors,
        )
        self.velocity_bridges.append(velocity_bridge)
        state_publisher = VehicleStateRos2Publisher(vehicle_id=vehicle_id, namespace=args_cli.namespace)
        self.state_publishers.append(state_publisher)

        config_multirotor.backends = [
            ArduPilotMavlinkBackend(config=backend_config),
            state_publisher,
            velocity_bridge,
        ]

        self.ardupilot_tools.append(
            MultiOutArduPilotLaunchTool(
                ardupilot_dir=self.pg.ardupilot_path,
                vehicle_id=vehicle_id,
                ardupilot_model="gazebo-iris",
                out_ports=[
                    14550 + vehicle_id * 10,
                    args_cli.bridge_baseport + vehicle_id * 10,
                ],
                fast_start=not args_cli.strict_prearm,
                show_ui=args_cli.show_ardupilot_ui,
            )
        )

        Multirotor(
            f"/World/drone{vehicle_id}",
            IRIS_USD_PATH,
            vehicle_id,
            [gap_x_axis * vehicle_id, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        carb.log_warn(
            "[transfer.app_ardu] ROS topics for drone%d: %s, %s, %s, %s, %s"
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

    def _launch_ardupilot(self):
        if self.ardupilot_started:
            return

        for launch_tool in self.ardupilot_tools:
            launch_tool.launch_ardupilot()

        self.ardupilot_started = True
        carb.log_warn("[13_ardu_ros] ArduPilot SITL instances launched")

    def _kill_ardupilot(self):
        if not self.ardupilot_started:
            return

        for launch_tool in self.ardupilot_tools:
            try:
                launch_tool.kill_ardupilot()
            except Exception as exc:
                carb.log_warn(f"[13_ardu_ros] Failed to kill ArduPilot instance cleanly: {exc}")

        self.ardupilot_started = False

    def _handle_signal(self, signum, _frame):
        self._signal_count += 1
        if self._signal_count == 1:
            carb.log_warn(f"[13_ardu_ros] received signal {signum}, stopping simulation")
            self.stop_sim = True
            return

        carb.log_warn(f"[13_ardu_ros] received second signal {signum}, forcing cleanup")
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
                carb.log_warn(f"[transfer.app_ardu] Failed while stopping ArduPilot ROS bridge: {exc}")

        for state_publisher in getattr(self, "state_publishers", []):
            try:
                state_publisher.stop()
            except Exception:
                pass

        try:
            self._kill_ardupilot()
        except Exception as exc:
            carb.log_warn(f"[13_ardu_ros] Failed while stopping ArduPilot: {exc}")

        try:
            self.timeline.stop()
        except Exception:
            pass

        for platform_publisher in getattr(self, "platform_publishers", []):
            try:
                platform_publisher.destroy()
            except Exception:
                pass

        try:
            simulation_app.close()
        except Exception:
            pass

    def _publish_platform_state(self):
        for platform_publisher in self.platform_publishers:
            platform_publisher.publish(self.platform.current_state)

    def _on_platform_physics_step(self, dt: float):
        if self.platform_motion_started:
            self.platform.update(dt)
        self._publish_platform_state()

    def run(self):
        self.timeline.play()
        self._launch_ardupilot()

        try:
            while simulation_app.is_running() and not self.stop_sim:
                self.world.step(render=not args_cli.headless)
                if self.platform_motion_enabled and not self.platform_motion_started and self.velocity_bridges:
                    if self.velocity_bridges[0].ready_for_velocity_commands:
                        self.platform_motion_started = True
                        carb.log_warn("[transfer.app_ardu] Platform motion enabled")
        except KeyboardInterrupt:
            self.stop_sim = True
        finally:
            carb.log_warn("PegasusApp Simulation App is closing.")
            self._shutdown()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
