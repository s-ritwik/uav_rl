from __future__ import annotations

import os
import subprocess
import time

import carb
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.backends.tools.ardupilot_launch_tool import ArduPilotLaunchTool

try:
    from .process_utils import terminate_process_tree
    from .topics import cmd_vel_topic, platform_pose_topic, platform_twist_topic
except ImportError:
    from process_utils import terminate_process_tree
    from topics import cmd_vel_topic, platform_pose_topic, platform_twist_topic


FAST_START_PARAM_TEXT = "\n".join(
    [
        "AHRS_EKF_TYPE 10",
        "EK2_ENABLE 0",
        "EK3_ENABLE 0",
        "",
    ]
)


class MultiOutArduPilotLaunchTool(ArduPilotLaunchTool):
    """Launch ArduPilot SITL with one output for Pegasus and one for the ROS command bridge."""

    def __init__(
        self,
        ardupilot_dir: str,
        vehicle_id: int,
        ardupilot_model: str,
        out_ports: list[int],
        fast_start: bool = True,
        show_ui: bool = False,
    ):
        super().__init__(ardupilot_dir, vehicle_id, ardupilot_model)
        self.out_ports = list(out_ports)
        self.fast_start = fast_start
        self.show_ui = show_ui
        self._fast_start_param_file: str | None = None

    def _ensure_fast_start_param_file(self) -> str:
        if self._fast_start_param_file is not None:
            return self._fast_start_param_file

        param_path = os.path.join(self.root_fs.name, f"pegasus_fast_start_{self.vehicle_id}.parm")
        with open(param_path, "w", encoding="ascii") as param_file:
            param_file.write(FAST_START_PARAM_TEXT)
        self._fast_start_param_file = param_path
        return param_path

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
        terminate_process_tree(self.ardupilot_process.pid, timeout=5.0)
        self.ardupilot_process = None


class ArduPilotRos2VelocityBridge(Backend):
    """ROS 2 bridge that auto-takes off and forwards Twist commands to ArduPilot guided velocity control."""

    def __init__(
        self,
        vehicle_id: int,
        *,
        namespace: str,
        bridge_baseport: int,
        auto_takeoff_alt: float,
        cmd_timeout: float,
        arm_delay: float,
        require_position_ready: bool,
        send_rate_hz: float = 20.0,
        num_rotors: int = 4,
    ):
        super().__init__(config=None)

        self._vehicle_id = vehicle_id
        self._bridge_port = bridge_baseport + vehicle_id * 10
        self._auto_takeoff_alt = max(float(auto_takeoff_alt), 0.0)
        self._cmd_timeout = max(float(cmd_timeout), 0.0)
        self._send_period = 1.0 / max(float(send_rate_hz), 1.0)
        self._arm_delay = max(float(arm_delay), 0.0)
        self._require_position_ready = bool(require_position_ready)
        self._input_ref = [0.0 for _ in range(num_rotors)]

        self._connection = None
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._current_altitude = 0.0
        self._body_to_local_ned = Rotation.identity()
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_wait_log_time = 0.0

        self._latest_cmd = Twist()
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(f"transfer_ardupilot_bridge_{vehicle_id}")
        self._cmd_sub = self.node.create_subscription(Twist, cmd_vel_topic(namespace, vehicle_id), self._cmd_vel_callback, 10)

        carb.log_warn(
            f"[ArduPilotRos2VelocityBridge] vehicle_id={vehicle_id} listening on '{cmd_vel_topic(namespace, vehicle_id)}' "
            f"and MAVLink bridge port {self._bridge_port}"
        )

    def _cmd_vel_callback(self, msg: Twist):
        self._latest_cmd = msg
        self._last_cmd_time = time.monotonic()

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
                if self._first_heartbeat_time is None:
                    self._first_heartbeat_time = time.monotonic()
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
        if not self._connected or self._takeoff_state == "ready" or self._first_heartbeat_time is None:
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
                    f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: waiting for " + ", ".join(wait_reasons)
                )
                self._last_wait_log_time = now
            return

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
            vel_body_frd = np.array(
                [
                    float(self._latest_cmd.linear.x),
                    -float(self._latest_cmd.linear.y),
                    -float(self._latest_cmd.linear.z),
                ]
            )
            vel_local_ned = self._body_to_local_ned.apply(vel_body_frd)
            vx = float(vel_local_ned[0])
            vy = float(vel_local_ned[1])
            vz = float(vel_local_ned[2])
            yaw_rate = -float(self._latest_cmd.angular.z)

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
        del sensor_type, data

    def update_graphical_sensor(self, sensor_type: str, data):
        del sensor_type, data

    def update_state(self, state):
        self._current_altitude = float(state.position[2])
        self._body_to_local_ned = Rotation.from_quat(state.get_attitude_ned_frd())

    def input_reference(self):
        return self._input_ref

    def update(self, dt: float):
        del dt
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._drain_mavlink()

        now = time.monotonic()
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
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._last_wait_log_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"

    def stop(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def reset(self):
        self._latest_cmd = Twist()
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._last_wait_log_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"


class PlatformRos2Publisher:
    """ROS 2 publisher for the moving platform state."""

    def __init__(self, namespace: str, vehicle_id: int):
        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(f"transfer_platform_publisher_{vehicle_id}")
        self.pose_pub = self.node.create_publisher(PoseStamped, platform_pose_topic(namespace, vehicle_id), 10)
        self.twist_pub = self.node.create_publisher(TwistStamped, platform_twist_topic(namespace, vehicle_id), 10)

    def publish(self, state):
        if state is None or not rclpy.ok():
            return

        pose_msg = PoseStamped()
        twist_msg = TwistStamped()
        now = self.node.get_clock().now().to_msg()

        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(state.position[0])
        pose_msg.pose.position.y = float(state.position[1])
        pose_msg.pose.position.z = float(state.position[2])
        pose_msg.pose.orientation.x = float(state.quat_xyzw[0])
        pose_msg.pose.orientation.y = float(state.quat_xyzw[1])
        pose_msg.pose.orientation.z = float(state.quat_xyzw[2])
        pose_msg.pose.orientation.w = float(state.quat_xyzw[3])

        twist_msg.header.stamp = now
        twist_msg.header.frame_id = "map"
        twist_msg.twist.linear.x = float(state.linear_velocity[0])
        twist_msg.twist.linear.y = float(state.linear_velocity[1])
        twist_msg.twist.linear.z = float(state.linear_velocity[2])
        twist_msg.twist.angular.x = float(state.angular_velocity[0])
        twist_msg.twist.angular.y = float(state.angular_velocity[1])
        twist_msg.twist.angular.z = float(state.angular_velocity[2])

        try:
            self.pose_pub.publish(pose_msg)
            self.twist_pub.publish(twist_msg)
        except Exception:
            return

    def destroy(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
