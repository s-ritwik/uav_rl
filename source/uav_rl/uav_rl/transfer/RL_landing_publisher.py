from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# Centralized ROS topic defaults for the Jetson + mocap integration.
# Leave the CLI args unset to use these defaults, or override any topic from the command line.
DEFAULT_TOPIC_SUFFIXES = {
    "robot_pose": "/state/pose",
    "robot_twist_body": "/state/twist",
    "robot_twist_inertial": "/state/twist_inertial",
    "platform_pose": "/platform/state/pose",
    "platform_twist": "/platform/state/twist",
    "cmd_vel": "/cmd_vel",
    "velocity_cmd": "/policy_cmd/velocity",
    "yaw_rate_cmd": "/policy_cmd/yaw_rate",
    "disarm_cmd": "/policy_cmd/disarm",
}


def _default_log_root() -> str:
    return str((Path(__file__).resolve().parents[4] / "logs" / "rsl_rl" / "landing_sway").resolve())


def _vehicle_ns(namespace: str, vehicle_id: int) -> str:
    return f"{namespace}{vehicle_id}"


def _default_topic(namespace: str, vehicle_id: int, key: str) -> str:
    return f"{_vehicle_ns(namespace, vehicle_id)}{DEFAULT_TOPIC_SUFFIXES[key]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 node that reconstructs the landing_sway policy observation from mocap/state topics "
            "and publishes velocity commands."
        )
    )
    parser.add_argument("--namespace", type=str, default="transfer", help="ROS namespace prefix, e.g. 'transfer'.")
    parser.add_argument("--vehicle-id", type=int, default=0, help="Vehicle id suffix used in ROS topics.")
    parser.add_argument("--policy-rate-hz", type=float, default=25.0, help="Policy inference rate.")
    parser.add_argument("--cmd-publish-rate-hz", type=float, default=25.0, help="Rate for publishing cmd_vel.")
    parser.add_argument("--policy-jit", type=str, default="/home/rycker/src/uav_rl/logs/rsl_rl/landing_sway/2026-04-27_09-51-07_landing_sway_2.8.7/exported/policy.pt", help="Path to exported policy.pt.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to an RSL-RL checkpoint file.")
    parser.add_argument(
        "--load-run",
        type=str,
        default=None,
        help="Run directory under logs/rsl_rl/landing_sway when checkpoint-path is not provided.",
    )
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint filename inside the run directory.")
    parser.add_argument("--log-root", type=str, default=_default_log_root(), help="Root folder for landing_sway logs.")
    parser.add_argument("--policy-device", type=str, default=None, help="Torch device for policy inference.")
    ## ------------ ADD CG to Landing gear distance -----------------#
    parser.add_argument(
        "--vehicle-z0-m",
        type=float,
        default=0.15,
        help="Constant landing-gear/root offset subtracted from relative z to match landing_sway observations.",
    )
    parser.add_argument("--velocity-limit-x", type=float, default=None, help="Deprecated symmetric |vx| limit.")
    parser.add_argument("--velocity-limit-y", type=float, default=None, help="Deprecated symmetric |vy| limit.")
    parser.add_argument("--velocity-limit-z", type=float, default=None, help="Deprecated symmetric |vz| limit.")
    parser.add_argument("--yaw-rate-limit", type=float, default=None, help="Deprecated symmetric |yaw_rate| limit.")
    parser.add_argument("--velocity-lower-limit-x", type=float, default=-0.8)
    parser.add_argument("--velocity-lower-limit-y", type=float, default=-0.8)
    parser.add_argument("--velocity-lower-limit-z", type=float, default=-0.8)
    parser.add_argument("--velocity-upper-limit-x", type=float, default=0.8)
    parser.add_argument("--velocity-upper-limit-y", type=float, default=0.8)
    parser.add_argument("--velocity-upper-limit-z", type=float, default=1.0)
    parser.add_argument("--yaw-rate-lower-limit", type=float, default=-35.0 * np.pi / 180.0)
    parser.add_argument("--yaw-rate-upper-limit", type=float, default=35.0 * np.pi / 180.0)
    parser.add_argument(
        "--enable-proximity-disarm",
        type=int,
        default=1,
        help="If 1, trigger disarm when relative |x|,|y|,|z| are below thresholds.",
    )
    parser.add_argument("--disarm-rel-x-threshold", type=float, default=0.8)
    parser.add_argument("--disarm-rel-y-threshold", type=float, default=0.8)
    parser.add_argument("--disarm-rel-z-threshold", type=float, default=0.15)
    parser.add_argument("--disarm-via-service", type=int, default=1)
    parser.add_argument("--mavros-ns", type=str, default="/mavros")
    parser.add_argument("--disarm-service-timeout", type=float, default=0.25)
    parser.add_argument("--disarm-service-attempts", type=int, default=1)
    parser.add_argument("--disarm-response-timeout", type=float, default=0.3)
    parser.add_argument("--disarm-fire-and-forget", type=int, default=0)
    parser.add_argument("--post-disarm-hold-seconds", type=float, default=5.0)
    parser.add_argument("--enable-csv", type=int, default=1, help="Enable CSV logging for policy steps.")
    parser.add_argument("--enable-async-logging", type=int, default=1)
    parser.add_argument("--log-queue-size", type=int, default=4096)
    parser.add_argument("--log-flush-period-s", type=float, default=0.5)
    parser.add_argument(
        "--csv-log-dir",
        type=str,
        default=None,
        help="Directory for CSV logs. Defaults to --log-root if unset.",
    )
    parser.add_argument("--debug-every", type=int, default=1, help="Print action debug every N policy steps.")
    parser.add_argument("--robot-pose-topic", type=str, default=None, help="ROS topic for robot pose input.")
    parser.add_argument(
        "--robot-twist-body-topic",
        type=str,
        default=None,
        help="ROS topic for robot body-frame angular velocity input.",
    )
    parser.add_argument(
        "--robot-twist-body-msg-type",
        type=str,
        choices=("auto", "twist", "imu"),
        default="auto",
        help="Message type used by --robot-twist-body-topic. 'auto' selects IMU for topics containing '/imu/'.",
    )
    parser.add_argument(
        "--robot-twist-inertial-topic",
        type=str,
        default=None,
        help="ROS topic for robot inertial/world-frame linear velocity input.",
    )
    parser.add_argument("--platform-pose-topic", type=str, default=None, help="ROS topic for platform pose input.")
    parser.add_argument("--platform-twist-topic", type=str, default=None, help="ROS topic for platform twist input.")
    parser.add_argument(
        "--cmd-vel-topic",
        type=str,
        default=None,
        help="Combined Twist output topic. Leave unset to use the default transfer cmd_vel topic.",
    )
    parser.add_argument(
        "--velocity-cmd-topic",
        type=str,
        default=None,
        help="Separate Vector3Stamped velocity output topic.",
    )
    parser.add_argument(
        "--yaw-rate-cmd-topic",
        type=str,
        default=None,
        help="Separate Float32 yaw-rate output topic.",
    )
    parser.add_argument(
        "--disarm-cmd-topic",
        type=str,
        default=None,
        help="Local Bool disarm output topic used by the transfer PX4 stack.",
    )
    return parser


def _resolve_policy_device(requested: str | None) -> str:
    if requested:
        return requested

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _resolve_robot_twist_body_msg_type(topic: str, requested: str) -> str:
    if requested != "auto":
        return requested
    topic_lower = topic.lower()
    if "/imu/" in topic_lower or topic_lower.endswith("/imu") or topic_lower.endswith("/imu/data"):
        return "imu"
    return "twist"


def _ensure_ros_runtime_env() -> None:
    sentinel = "_UAV_RL_TRANSFER_ROS_ENV"
    desired_env = {
        "PYTHONNOUSERSITE": "1",
        "LD_LIBRARY_PATH": "/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu",
        "PYTHONPATH": "/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages",
    }

    if os.environ.get(sentinel) == "1":
        return

    if all(os.environ.get(key) == value for key, value in desired_env.items()):
        return

    env = os.environ.copy()
    env.update(desired_env)
    env[sentinel] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _run(args)


def _run(args) -> None:
    _ensure_ros_runtime_env()
    simulation_app = None

    try:
        import rclpy
    except (ModuleNotFoundError, ImportError):
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True})
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

        import rclpy

    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    import torch
    from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Vector3Stamped
    try:
        from mavros_msgs.srv import CommandLong
    except ImportError:
        CommandLong = None
    from scipy.spatial.transform import Rotation
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Bool, Float32

    try:
        from .policy import RslRlPolicy, resolve_checkpoint_path
    except ImportError:
        from policy import RslRlPolicy, resolve_checkpoint_path

    class RLLandingPublisher:
        def __init__(self):
            try:
                rclpy.init()
            except Exception:
                pass

            self.start_time = time.monotonic()
            self.node = rclpy.create_node(f"rl_landing_publisher_{args.vehicle_id}")
            self.robot_pose_topic = args.robot_pose_topic or _default_topic(args.namespace, args.vehicle_id, "robot_pose")
            self.robot_twist_body_topic = args.robot_twist_body_topic or _default_topic(
                args.namespace, args.vehicle_id, "robot_twist_body"
            )
            self.robot_twist_inertial_topic = args.robot_twist_inertial_topic or _default_topic(
                args.namespace, args.vehicle_id, "robot_twist_inertial"
            )
            self.platform_pose_topic = args.platform_pose_topic or _default_topic(
                args.namespace, args.vehicle_id, "platform_pose"
            )
            self.platform_twist_topic = args.platform_twist_topic or _default_topic(
                args.namespace, args.vehicle_id, "platform_twist"
            )
            self.cmd_vel_topic = args.cmd_vel_topic or _default_topic(args.namespace, args.vehicle_id, "cmd_vel")
            self.velocity_cmd_topic = args.velocity_cmd_topic or _default_topic(
                args.namespace, args.vehicle_id, "velocity_cmd"
            )
            self.yaw_rate_cmd_topic = args.yaw_rate_cmd_topic or _default_topic(
                args.namespace, args.vehicle_id, "yaw_rate_cmd"
            )
            self.disarm_cmd_topic = args.disarm_cmd_topic or _default_topic(
                args.namespace, args.vehicle_id, "disarm_cmd"
            )
            self.robot_twist_body_msg_type = _resolve_robot_twist_body_msg_type(
                self.robot_twist_body_topic, args.robot_twist_body_msg_type
            )

            qos_best_effort = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
            )

            self.cmd_pub = self.node.create_publisher(Twist, self.cmd_vel_topic, 10)
            self.velocity_cmd_pub = self.node.create_publisher(Vector3Stamped, self.velocity_cmd_topic, 10)
            self.yaw_rate_cmd_pub = self.node.create_publisher(Float32, self.yaw_rate_cmd_topic, 10)
            self.disarm_cmd_pub = self.node.create_publisher(Bool, self.disarm_cmd_topic, 10)

            self.pose_sub = self.node.create_subscription(
                PoseStamped, self.robot_pose_topic, self._pose_callback, qos_best_effort
            )
            if self.robot_twist_body_msg_type == "imu":
                self.twist_sub = self.node.create_subscription(
                    Imu, self.robot_twist_body_topic, self._imu_callback, qos_best_effort
                )
            else:
                self.twist_sub = self.node.create_subscription(
                    TwistStamped, self.robot_twist_body_topic, self._twist_callback, qos_best_effort
                )
            self.twist_inertial_sub = self.node.create_subscription(
                TwistStamped,
                self.robot_twist_inertial_topic,
                self._twist_inertial_callback,
                qos_best_effort,
            )
            self.platform_pose_sub = self.node.create_subscription(
                PoseStamped, self.platform_pose_topic, self._platform_pose_callback, qos_best_effort
            )
            self.platform_twist_sub = self.node.create_subscription(
                TwistStamped, self.platform_twist_topic, self._platform_twist_callback, qos_best_effort
            )

            checkpoint_path = (
                resolve_checkpoint_path(
                    load_run=args.load_run,
                    checkpoint_name=args.checkpoint_name,
                    checkpoint_path=args.checkpoint_path,
                    log_root=args.log_root,
                )
                if args.policy_jit is None
                else None
            )

            self.policy = RslRlPolicy(
                device=_resolve_policy_device(args.policy_device),
                policy_jit=args.policy_jit,
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            )
            self.device = torch.device(_resolve_policy_device(args.policy_device))
            self.velocity_lower_limits = np.array(
                [
                    -args.velocity_limit_x if args.velocity_limit_x is not None else args.velocity_lower_limit_x,
                    -args.velocity_limit_y if args.velocity_limit_y is not None else args.velocity_lower_limit_y,
                    -args.velocity_limit_z if args.velocity_limit_z is not None else args.velocity_lower_limit_z,
                ],
                dtype=np.float32,
            )
            self.velocity_upper_limits = np.array(
                [
                    args.velocity_limit_x if args.velocity_limit_x is not None else args.velocity_upper_limit_x,
                    args.velocity_limit_y if args.velocity_limit_y is not None else args.velocity_upper_limit_y,
                    args.velocity_limit_z if args.velocity_limit_z is not None else args.velocity_upper_limit_z,
                ],
                dtype=np.float32,
            )
            self.yaw_rate_lower_limit = float(
                -args.yaw_rate_limit if args.yaw_rate_limit is not None else args.yaw_rate_lower_limit
            )
            self.yaw_rate_upper_limit = float(
                args.yaw_rate_limit if args.yaw_rate_limit is not None else args.yaw_rate_upper_limit
            )
            self.vehicle_z0_m = float(args.vehicle_z0_m)
            self.enable_proximity_disarm = bool(args.enable_proximity_disarm)
            self.disarm_rel_x_threshold = float(args.disarm_rel_x_threshold)
            self.disarm_rel_y_threshold = float(args.disarm_rel_y_threshold)
            self.disarm_rel_z_threshold = float(args.disarm_rel_z_threshold)
            self.disarm_via_service = bool(args.disarm_via_service)
            self.mavros_ns = str(args.mavros_ns)
            self.disarm_service_timeout = float(args.disarm_service_timeout)
            self.disarm_service_attempts = int(args.disarm_service_attempts)
            self.disarm_response_timeout = float(args.disarm_response_timeout)
            self.disarm_fire_and_forget = bool(args.disarm_fire_and_forget)
            self.post_disarm_hold_seconds = float(args.post_disarm_hold_seconds)
            self.enable_csv = bool(args.enable_csv)
            self.enable_async_logging = bool(args.enable_async_logging)
            self.log_queue_size = max(8, int(args.log_queue_size))
            self.log_flush_period_s = max(0.05, float(args.log_flush_period_s))
            self.csv_log_dir = str(args.csv_log_dir) if args.csv_log_dir else str(args.log_root)
            self.last_obs_action = np.zeros((4,), dtype=np.float32)
            self.last_cmd_action = np.zeros((4,), dtype=np.float32)

            self.robot_pos = None
            self.robot_quat_xyzw = None
            self.robot_lin_vel_w = None
            self.robot_ang_vel_b = None
            self.platform_pos = None
            self.platform_quat_xyzw = None
            self.platform_lin_vel_w = None
            self.platform_ang_vel_w = None

            self.timer = self.node.create_timer(1.0 / max(args.cmd_publish_rate_hz, 1.0), self._on_timer)
            self.step_count = 0
            self.last_policy_time = 0.0
            self.policy_period = 1.0 / max(args.policy_rate_hz, 1.0)
            self._warned_unready = False
            self._proximity_disarm_done = False
            self.done_enter_time = None
            self.disarm_time = None
            self._stopped = False
            self._start_monotonic = time.monotonic()
            self._csv_file = None
            self._csv_writer = None
            self._log_path = None
            self._csv_fieldnames = []
            self._log_queue = None
            self._log_thread = None
            self._workers_started = False
            self._workers_stopped = False
            self._log_dropped = 0
            self._log_sentinel = object()

            self._init_csv_logger()
            self._start_background_workers()

            self.cmd_long_cli = None
            if self.enable_proximity_disarm and self.disarm_via_service:
                if CommandLong is None:
                    self._log_warn(
                        "mavros_msgs.srv.CommandLong is unavailable; disabling service-based disarm."
                    )
                    self.disarm_via_service = False
                else:
                    self.cmd_long_cli = self.node.create_client(CommandLong, f"{self.mavros_ns}/cmd/command")

            self._log_info(
                "RL landing publisher topics: "
                f"robot_pose='{self.robot_pose_topic}', "
                f"robot_twist_body='{self.robot_twist_body_topic}' ({self.robot_twist_body_msg_type}), "
                f"robot_twist_inertial='{self.robot_twist_inertial_topic}', "
                f"platform_pose='{self.platform_pose_topic}', "
                f"platform_twist='{self.platform_twist_topic}', "
                f"cmd_vel_out='{self.cmd_vel_topic}', "
                f"velocity_out='{self.velocity_cmd_topic}', "
                f"yaw_rate_out='{self.yaw_rate_cmd_topic}', "
                f"disarm_out='{self.disarm_cmd_topic}', "
                f"vel_limits=({self.velocity_lower_limits.tolist()} .. {self.velocity_upper_limits.tolist()}), "
                f"yaw_limits=({self.yaw_rate_lower_limit:.3f} .. {self.yaw_rate_upper_limit:.3f})\n"
                "Proximity disarm: "
                f"enabled={self.enable_proximity_disarm}, "
                f"thresholds=({self.disarm_rel_x_threshold:.3f}, "
                f"{self.disarm_rel_y_threshold:.3f}, {self.disarm_rel_z_threshold:.3f}), "
                f"service={self.disarm_via_service}, "
                f"mavros_ns='{self.mavros_ns}'"
            )

        def _elapsed_seconds(self) -> float:
            return time.monotonic() - self.start_time

        def _log(self, level: str, message: str) -> None:
            stream = sys.stderr if level == "WARN" else sys.stdout
            print(
                f"[{self._elapsed_seconds():8.3f}s] [{level}] [{self.node.get_name()}]: {message}",
                file=stream,
                flush=True,
            )

        def _log_info(self, message: str) -> None:
            self._log("INFO", message)

        def _log_warn(self, message: str) -> None:
            self._log("WARN", message)

        def _init_csv_logger(self) -> None:
            if not self.enable_csv:
                self._log_info("CSV logging disabled by parameter enable_csv=False")
                self._csv_file = None
                self._csv_writer = None
                self._log_path = None
                self._csv_fieldnames = []
                return
            try:
                os.makedirs(self.csv_log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._log_path = os.path.join(self.csv_log_dir, f"logs_{timestamp}_rl_policy.csv")
                self._csv_fieldnames = [
                    "t",
                    "policy_step",
                    "robot_pos_x",
                    "robot_pos_y",
                    "robot_pos_z",
                    "platform_pos_x",
                    "platform_pos_y",
                    "platform_pos_z",
                    "rel_pos_x",
                    "rel_pos_y",
                    "rel_pos_z",
                    "obs_rel_lin_vel_x",
                    "obs_rel_lin_vel_y",
                    "obs_rel_lin_vel_z",
                    "obs_rel_quat_w",
                    "obs_rel_quat_x",
                    "obs_rel_quat_y",
                    "obs_rel_quat_z",
                    "obs_rel_ang_vel_x",
                    "obs_rel_ang_vel_y",
                    "obs_rel_ang_vel_z",
                    "obs_projected_gravity_x",
                    "obs_projected_gravity_y",
                    "obs_projected_gravity_z",
                    "obs_last_action_vx",
                    "obs_last_action_vy",
                    "obs_last_action_vz",
                    "obs_last_action_yaw_rate",
                    "action_vx",
                    "action_vy",
                    "action_vz",
                    "action_yaw_rate",
                ]
                self._csv_file = open(self._log_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fieldnames)
                self._csv_writer.writeheader()
                self._csv_file.flush()
                self._log_info(f"CSV logging to {self._log_path}")
            except Exception as exc:
                self._log_warn(f"CSV logging disabled: {exc}")
                self._csv_file = None
                self._csv_writer = None
                self._log_path = None
                self._csv_fieldnames = []

        def _start_background_workers(self) -> None:
            if self._workers_started:
                return
            if self.enable_async_logging and self._csv_writer is not None:
                self._log_queue = queue.Queue(maxsize=self.log_queue_size)
                self._log_thread = threading.Thread(target=self._logging_worker, name="csv-logger", daemon=True)
                self._log_thread.start()
            self._workers_started = True

        def _stop_background_workers(self) -> None:
            if self._workers_stopped:
                return
            self._workers_stopped = True
            if self._log_thread is not None and self._log_queue is not None:
                self._enqueue_log_item(self._log_sentinel, allow_drop=True)
                self._log_thread.join(timeout=3.0)
                if self._log_thread.is_alive():
                    self._log_warn("CSV logger thread did not stop cleanly.")
            self._log_thread = None

        def _enqueue_log_item(self, item, allow_drop: bool) -> bool:
            if self._log_queue is None:
                return False
            try:
                self._log_queue.put_nowait(item)
                return True
            except queue.Full:
                if not allow_drop:
                    return False
                try:
                    self._log_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._log_queue.put_nowait(item)
                    return True
                except queue.Full:
                    return False

        def _flush_csv(self) -> None:
            if self._csv_file is None:
                return
            try:
                self._csv_file.flush()
            except Exception as exc:
                self._log_warn(f"CSV flush failed: {exc}")

        def _write_csv_row(self, row: dict) -> bool:
            if self._csv_writer is None:
                return False
            try:
                self._csv_writer.writerow(row)
                return True
            except Exception as exc:
                self._log_warn(f"CSV log write failed: {exc}")
                self._csv_writer = None
                return False

        def _logging_worker(self) -> None:
            if self._log_queue is None:
                return
            pending = 0
            last_flush = time.monotonic()
            while True:
                item = None
                try:
                    item = self._log_queue.get(timeout=0.2)
                except queue.Empty:
                    pass
                if item is self._log_sentinel:
                    break
                if item is not None and self._write_csv_row(item):
                    pending += 1
                now = time.monotonic()
                if pending > 0 and (now - last_flush) >= self.log_flush_period_s:
                    self._flush_csv()
                    pending = 0
                    last_flush = now
            if pending > 0:
                self._flush_csv()

        def _build_csv_row(self, t_rel: float, obs: np.ndarray, action: np.ndarray):
            if self._csv_writer is None:
                return None
            if obs is None or action is None:
                return None
            if self.robot_pos is None or self.platform_pos is None:
                return None

            rel_pos = obs[0:3]
            return {
                "t": float(t_rel),
                "policy_step": int(self.step_count),
                "robot_pos_x": float(self.robot_pos[0]),
                "robot_pos_y": float(self.robot_pos[1]),
                "robot_pos_z": float(self.robot_pos[2]),
                "platform_pos_x": float(self.platform_pos[0]),
                "platform_pos_y": float(self.platform_pos[1]),
                "platform_pos_z": float(self.platform_pos[2]),
                "rel_pos_x": float(rel_pos[0]),
                "rel_pos_y": float(rel_pos[1]),
                "rel_pos_z": float(rel_pos[2]),
                "obs_rel_lin_vel_x": float(obs[3]),
                "obs_rel_lin_vel_y": float(obs[4]),
                "obs_rel_lin_vel_z": float(obs[5]),
                "obs_rel_quat_w": float(obs[6]),
                "obs_rel_quat_x": float(obs[7]),
                "obs_rel_quat_y": float(obs[8]),
                "obs_rel_quat_z": float(obs[9]),
                "obs_rel_ang_vel_x": float(obs[10]),
                "obs_rel_ang_vel_y": float(obs[11]),
                "obs_rel_ang_vel_z": float(obs[12]),
                "obs_projected_gravity_x": float(obs[13]),
                "obs_projected_gravity_y": float(obs[14]),
                "obs_projected_gravity_z": float(obs[15]),
                "obs_last_action_vx": float(obs[16]),
                "obs_last_action_vy": float(obs[17]),
                "obs_last_action_vz": float(obs[18]),
                "obs_last_action_yaw_rate": float(obs[19]),
                "action_vx": float(action[0]),
                "action_vy": float(action[1]),
                "action_vz": float(action[2]),
                "action_yaw_rate": float(action[3]),
            }

        def _log_csv_row(self, t_rel: float, obs: np.ndarray, action: np.ndarray) -> None:
            row = self._build_csv_row(t_rel, obs, action)
            if row is None:
                return
            if self.enable_async_logging and self._log_queue is not None:
                if not self._enqueue_log_item(row, allow_drop=False):
                    self._log_dropped += 1
                    self._log_warn(f"CSV queue full; dropped rows={self._log_dropped}")
                return
            if self._write_csv_row(row):
                self._flush_csv()

        def _close_log(self) -> None:
            self._stop_background_workers()
            if self._csv_file:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                except Exception as exc:
                    self._log_warn(f"CSV log close failed: {exc}")
                finally:
                    self._csv_file = None
                    self._csv_writer = None

        def _pose_callback(self, msg: PoseStamped):
            self.robot_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32
            )
            self.robot_quat_xyzw = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=np.float32,
            )

        def _twist_callback(self, msg: TwistStamped):
            self.robot_ang_vel_b = np.array(
                [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=np.float32
            )

        def _imu_callback(self, msg: Imu):
            self.robot_ang_vel_b = np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float32
            )

        def _twist_inertial_callback(self, msg: TwistStamped):
            self.robot_lin_vel_w = np.array(
                [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32
            )

        def _platform_pose_callback(self, msg: PoseStamped):
            self.platform_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32
            )
            self.platform_quat_xyzw = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=np.float32,
            )

        def _platform_twist_callback(self, msg: TwistStamped):
            self.platform_lin_vel_w = np.array(
                [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32
            )
            self.platform_ang_vel_w = np.array(
                [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=np.float32
            )

        def _ready(self) -> bool:
            return all(
                value is not None
                for value in (
                    self.robot_pos,
                    self.robot_quat_xyzw,
                    self.robot_lin_vel_w,
                    self.robot_ang_vel_b,
                    self.platform_pos,
                    self.platform_quat_xyzw,
                    self.platform_lin_vel_w,
                    self.platform_ang_vel_w,
                )
            )

        def _build_observation(self) -> np.ndarray:
            robot_rot = Rotation.from_quat(self.robot_quat_xyzw)

            # Observation order must match landing_sway/PolicyCfg exactly:
            # rel_pos, rel_lin_vel, rel_quat, rel_ang_vel, projected_gravity, last_action.
            rel_pos = (self.robot_pos - self.platform_pos).astype(np.float32)
            rel_pos[2] -= self.vehicle_z0_m

            rel_lin_vel = (self.robot_lin_vel_w - self.platform_lin_vel_w).astype(np.float32)
            rel_quat_wxyz = _quat_xyzw_to_wxyz(self.robot_quat_xyzw.astype(np.float32))

            # Robot angular velocity is received in robot body frame. Convert to world,
            # then subtract platform world angular velocity without additional frame rotation.
            robot_ang_vel_w = robot_rot.apply(self.robot_ang_vel_b)
            rel_ang_vel = (robot_ang_vel_w - self.platform_ang_vel_w).astype(np.float32)

            # landing_sway uses robot projected gravity in body frame.
            projected_gravity = robot_rot.inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float32))

            return np.concatenate(
                (
                    rel_pos.astype(np.float32),
                    rel_lin_vel.astype(np.float32),
                    rel_quat_wxyz.astype(np.float32),
                    rel_ang_vel.astype(np.float32),
                    projected_gravity.astype(np.float32),
                    self.last_cmd_action,
                )
            )

        def _compute_rel_pos_world(self) -> np.ndarray | None:
            if self.robot_pos is None or self.platform_pos is None:
                return None
            rel_pos = (self.robot_pos - self.platform_pos).astype(np.float32)
            rel_pos[2] -= self.vehicle_z0_m
            return rel_pos

        def _check_and_trigger_proximity_disarm(self, now_sec: float) -> None:
            if not self.enable_proximity_disarm or self._proximity_disarm_done:
                return
            rel_pos = self._compute_rel_pos_world()
            if rel_pos is None:
                return

            ax, ay, az = float(abs(rel_pos[0])), float(abs(rel_pos[1])), float(abs(rel_pos[2]))
            if ax <= self.disarm_rel_x_threshold and ay <= self.disarm_rel_y_threshold and az <= self.disarm_rel_z_threshold:
                self._log_warn(
                    "Proximity disarm trigger met: "
                    f"|rel_x|={ax:.3f}, |rel_y|={ay:.3f}, |rel_z|={az:.3f}"
                )
                if self.done_enter_time is None:
                    self.done_enter_time = now_sec
                if self._try_force_disarm():
                    if self.disarm_time is None:
                        self.disarm_time = now_sec
                    self._proximity_disarm_done = True

        def _try_force_disarm(self) -> bool:
            if not self.disarm_via_service or self.cmd_long_cli is None:
                self._publish_local_disarm()
                return True

            for attempt in range(self.disarm_service_attempts):
                if self.cmd_long_cli.wait_for_service(timeout_sec=self.disarm_service_timeout):
                    break
                self._log_warn(
                    f"Waiting for CommandLong service (attempt {attempt + 1}/{self.disarm_service_attempts})"
                )
            else:
                self._log_warn("CommandLong service not available; falling back to local disarm topic")
                self._publish_local_disarm()
                return True

            req = CommandLong.Request()
            req.broadcast = False
            req.command = 400
            req.confirmation = 0
            req.param1 = 0.0
            req.param2 = 21196.0
            req.param3 = 0.0
            req.param4 = 0.0
            req.param5 = 0.0
            req.param6 = 0.0
            req.param7 = 0.0

            future = self.cmd_long_cli.call_async(req)
            if self.disarm_fire_and_forget:
                future.add_done_callback(self._on_force_disarm_response)
                self._log_warn("Sent MAV_CMD_COMPONENT_ARM_DISARM (force disarm, fire-and-forget)")
                return True

            rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.disarm_response_timeout)
            if future.done():
                return self._on_force_disarm_response(future)
            else:
                self._log_warn("Force disarm timed out with no response")
                return False

        def _on_force_disarm_response(self, future) -> bool:
            try:
                resp = future.result()
                if resp and bool(resp.success):
                    self._log_warn(f"Force disarm accepted (result={resp.result})")
                    return True
                else:
                    self._log_warn(f"Force disarm failed (resp={resp})")
                    return False
            except Exception as exc:
                self._log_warn(f"Force disarm call error: {exc}")
                return False

        def _publish_local_disarm(self) -> None:
            msg = Bool()
            msg.data = True
            self.disarm_cmd_pub.publish(msg)
            self._log_warn(
                f"Published local disarm request on '{self.disarm_cmd_topic}'"
            )

        def _stop_node(self, reason: str) -> None:
            if self._stopped:
                return
            self._stopped = True
            self._log_info(reason)
            try:
                self.timer.cancel()
            except Exception:
                pass
            raise ExternalShutdownException()

        def _publish_cmd(self, world_velocity_sp: np.ndarray, yaw_rate_sp: float):
            msg = Twist()
            msg.linear.x = float(world_velocity_sp[0])
            msg.linear.y = float(world_velocity_sp[1])
            msg.linear.z = float(world_velocity_sp[2])
            msg.angular.z = float(yaw_rate_sp)
            self.cmd_pub.publish(msg)

            velocity_msg = Vector3Stamped()
            velocity_msg.header.stamp = self.node.get_clock().now().to_msg()
            velocity_msg.vector.x = float(world_velocity_sp[0])
            velocity_msg.vector.y = float(world_velocity_sp[1])
            velocity_msg.vector.z = float(world_velocity_sp[2])
            self.velocity_cmd_pub.publish(velocity_msg)

            yaw_rate_msg = Float32()
            yaw_rate_msg.data = float(yaw_rate_sp)
            self.yaw_rate_cmd_pub.publish(yaw_rate_msg)

        def _on_timer(self):
            now = time.monotonic()
            if not self._ready():
                if not self._warned_unready:
                    self._log_info(
                        "Waiting for robot/platform pose and twist topics before activating the landing policy."
                    )
                    self._warned_unready = True
                return

            self._warned_unready = False
            self._check_and_trigger_proximity_disarm(now)
            if self._proximity_disarm_done:
                self.last_obs_action[:] = 0.0
                self.last_cmd_action[:] = 0.0
                self._publish_cmd(self.last_cmd_action[:3], float(self.last_cmd_action[3]))
                ref_time = self.disarm_time if self.disarm_time is not None else self.done_enter_time
                if ref_time is not None and (now - ref_time) >= self.post_disarm_hold_seconds:
                    self._stop_node("Landing completed; shutting down node.")
                return

            if now - self.last_policy_time >= self.policy_period:
                obs = self._build_observation()
                obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                with torch.inference_mode():
                    raw_action = self.policy.act(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)

                cmd_action = raw_action.copy()
                cmd_action[:3] = np.clip(cmd_action[:3], self.velocity_lower_limits, self.velocity_upper_limits)
                cmd_action[3] = float(np.clip(cmd_action[3], self.yaw_rate_lower_limit, self.yaw_rate_upper_limit))
                self.last_obs_action = raw_action
                self.last_cmd_action = cmd_action
                self.last_policy_time = now
                t_rel = now - self._start_monotonic
                self._log_csv_row(t_rel, obs, cmd_action)
                self.step_count += 1

                if args.debug_every > 0 and self.step_count % args.debug_every == 0:
                    self._log_info(
                        f"policy_step={self.step_count} obs_z={float(obs[2]):.3f} "
                        f"raw_action={self.last_obs_action.tolist()} "
                        f"vel_sp={self.last_cmd_action[:3].tolist()} yaw_rate={float(self.last_cmd_action[3]):.3f}"
                        f" rel_pos={self._compute_rel_pos_world().tolist() if self._compute_rel_pos_world() is not None else None}"
                    )

            self._publish_cmd(self.last_cmd_action[:3], float(self.last_cmd_action[3]))

        def run(self):
            try:
                rclpy.spin(self.node)
            except ExternalShutdownException:
                pass
            except KeyboardInterrupt:
                pass
            finally:
                self._publish_cmd(np.zeros((3,), dtype=np.float32), 0.0)
                self._close_log()
                self.node.destroy_node()

    try:
        RLLandingPublisher().run()
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
