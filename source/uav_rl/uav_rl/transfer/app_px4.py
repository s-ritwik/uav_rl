#!/usr/bin/env python
"""
| File: app_px4.py
| Description: PX4-backed transfer app with ROS 2 velocity-command bridge.
"""

from __future__ import annotations

import argparse
import atexit
import ast
import math
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import carb
from isaacsim import SimulationApp

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
    default=3.0,
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
    default=str((Path(__file__).resolve().parents[1] / "assets" / "Aruco" / "aruco_mark_fractal.png").resolve()),
    help="PNG texture applied to the platform top decal.",
)
parser.add_argument(
    "--motion_stage",
    type=str,
    default="track_xy",
    choices=("track_xy", "track_xy_roll_pitch", "track_xy_roll_pitch_heave"),
    help="Platform motion preset matching the vanilla task stages.",
)
parser.add_argument("--platform_seed", type=int, default=0, help="Random seed for the platform motion sampler.")

args_cli, _ = parser.parse_known_args()
if args_cli.num_drones < 1:
    parser.error("--num_drones must be greater than or equal to 1.")

IRIS_USD_PATH = str((Path(__file__).resolve().parents[1] / "assets" / "robots" / "iris" / "iris_capsule.usd").resolve())
PLATFORM_SIZE = (1.0, 1.0, 0.2)

simulation_app = SimulationApp({"headless": args_cli.headless})

import omni.timeline
from omni.isaac.core.world import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from pymavlink import mavutil
from scipy.spatial.transform import Rotation
import rclpy

from pegasus.simulator.params import SIMULATION_ENVIRONMENTS, WORLD_SETTINGS
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rotations import rot_ENU_to_NED

try:
    from .ardupilot_ros import PlatformRos2Publisher
    from .moving_platform import HarmonicAxisMotionCfg, MovingPlatform, PlatformMotionStageCfg
    from .topics import cmd_vel_topic, platform_pose_topic, platform_twist_topic, pose_topic, twist_inertial_topic, twist_topic
except ImportError:
    from ardupilot_ros import PlatformRos2Publisher
    from moving_platform import HarmonicAxisMotionCfg, MovingPlatform, PlatformMotionStageCfg
    from topics import cmd_vel_topic, platform_pose_topic, platform_twist_topic, pose_topic, twist_inertial_topic, twist_topic


class PX4Ros2VelocityBridge(Backend):
    """ROS 2 backend that drives PX4 OFFBOARD velocity control from geometry_msgs/Twist."""

    def __init__(
        self,
        vehicle_id: int,
        namespace: str = "transfer",
        offboard_baseport: int = 14540,
        auto_takeoff_alt: float = 2.0,
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

        self._latest_cmd = Twist()
        self._last_cmd_time = 0.0

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(f"px4_ros2_bridge_{vehicle_id}")
        topic = cmd_vel_topic(namespace, vehicle_id)
        self._cmd_sub = self.node.create_subscription(Twist, topic, self._cmd_vel_callback, 10)

        carb.log_warn(
            f"[PX4Ros2VelocityBridge] vehicle_id={vehicle_id} listening on ROS 2 topic '{topic}' "
            f"and PX4 offboard port {self._offboard_port}"
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
                if command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self._armed = True

    def _wait_ready(self, now: float) -> bool:
        wait_reasons = []
        if not self._connected:
            wait_reasons.append("heartbeat")
        elif self._first_heartbeat_time is not None and (now - self._first_heartbeat_time) < self._arm_delay:
            wait_reasons.append(f"arm_delay {self._arm_delay:.1f}s")
        if not self._position_estimate_ready:
            wait_reasons.append("position_estimate")

        if wait_reasons and (now - self._last_wait_log_time) >= 2.0:
            carb.log_warn(
                f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: waiting for " + ", ".join(wait_reasons)
            )
            self._last_wait_log_time = now

        return not wait_reasons

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

        alt_error = self._auto_takeoff_alt - self._current_altitude
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

        if not self._armed and now - self._last_arm_request_time >= 1.0:
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
            self._last_action_time = now
            self._last_arm_request_time = now
            carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: retrying arm")
            return

        if not self._offboard_enabled and now - self._last_mode_request_time >= 1.0:
            self._connection.set_mode("OFFBOARD")
            self._last_action_time = now
            self._last_mode_request_time = now
            carb.log_warn(f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: requested OFFBOARD mode")
            return

        if self._offboard_enabled and self._armed and self._takeoff_state == "pending":
            self._takeoff_state = "taking_off"
            self._ready_since = None
            carb.log_warn(
                f"[PX4Ros2VelocityBridge] drone{self._vehicle_id}: armed in OFFBOARD, climbing to {self._auto_takeoff_alt:.2f} m"
            )

    def _update_takeoff_state(self, now: float):
        if self._takeoff_state != "taking_off":
            return

        altitude_error = abs(self._current_altitude - self._auto_takeoff_alt)
        if altitude_error <= 0.12 and abs(self._current_vertical_speed) <= 0.12:
            if self._ready_since is None:
                self._ready_since = now
            elif now - self._ready_since >= 0.75:
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

    def input_reference(self):
        return self._input_ref

    def update(self, dt: float):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._drain_mavlink()
        now = time.monotonic()
        self._send_velocity_command(now)
        self._request_offboard_and_arm(now)
        self._update_takeoff_state(now)

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


def _load_vanilla_add_platform_top_decal():
    module_path = Path(__file__).resolve().parents[1] / "tasks" / "manager_based" / "vanilla" / "mdp" / "events.py"
    source = module_path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(module_path))
    target_fn = None
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "add_platform_top_decal":
            target_fn = node
            break
    if target_fn is None:
        raise ImportError(f"Unable to find 'add_platform_top_decal' in '{module_path}'")

    fn_module = ast.Module(body=[target_fn], type_ignores=[])
    ast.fix_missing_locations(fn_module)
    namespace: dict[str, object] = {
        "Path": Path,
        "Sequence": __import__("typing").Sequence,
    }
    exec(compile(fn_module, str(module_path), "exec"), namespace)
    return namespace["add_platform_top_decal"]


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
        self.platform_motion_started = False
        self._vanilla_add_platform_top_decal = _load_vanilla_add_platform_top_decal()

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        atexit.register(self._shutdown)

        self.platform = MovingPlatform(
            self.world,
            texture_path=args_cli.platform_texture,
            physics_dt=float(WORLD_SETTINGS["px4"]["physics_dt"]),
            stage_cfg=_build_platform_stage_cfg(args_cli.motion_stage),
            rng_seed=args_cli.platform_seed,
            size=PLATFORM_SIZE,
            base_position=(args_cli.platform_x, args_cli.platform_y, args_cli.platform_z),
            add_top_decal=False,
        )
        self.world.add_physics_callback("platform_motion", self._on_platform_physics_step)
        self.platform_publishers = [
            PlatformRos2Publisher(args_cli.namespace, vehicle_id) for vehicle_id in range(args_cli.num_drones)
        ]

        for vehicle_id in range(args_cli.num_drones):
            self.vehicle_factory(vehicle_id, gap_x_axis=args_cli.gap_x_axis)

        self.world.reset()
        self.platform.reset_profile()
        self._apply_platform_top_decal()
        self._publish_platform_state()

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float):
        config_multirotor = MultirotorConfig()

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

        try:
            self.timeline.stop()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass

    def _apply_platform_top_decal(self):
        env_proxy = SimpleNamespace(
            scene=SimpleNamespace(
                stage=self.world.stage,
                env_prim_paths=["/World"],
            )
        )
        self._vanilla_add_platform_top_decal(
            env_proxy,
            None,
            texture_path=args_cli.platform_texture,
            platform_name="platform/decal_frame",
            platform_size=PLATFORM_SIZE,
        )

    def _publish_platform_state(self):
        for platform_publisher in self.platform_publishers:
            platform_publisher.publish(self.platform.current_state)

    def _on_platform_physics_step(self, dt: float):
        if self.platform_motion_started:
            self.platform.update(dt)
        self._publish_platform_state()

    def run(self):
        self.timeline.play()

        try:
            while simulation_app.is_running() and not self.stop_sim:
                self.world.step(render=not args_cli.headless)
                if not self.platform_motion_started and self.velocity_bridges:
                    if self.velocity_bridges[0].ready_for_velocity_commands:
                        self.platform_motion_started = True
                        carb.log_warn("[transfer.app_px4] Platform motion enabled")
        except KeyboardInterrupt:
            self.stop_sim = True
        finally:
            carb.log_warn("PegasusApp PX4 Simulation App is closing.")
            self._shutdown()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
