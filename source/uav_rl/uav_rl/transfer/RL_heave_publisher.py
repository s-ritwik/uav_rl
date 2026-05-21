from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


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

DEFAULT_POLICY_JIT = (
    "/home/rycker/src/uav_rl/logs/rsl_rl/heave_landing_gru/"
    "2026-05-19_00-14-23_heave_gru_3.0.1/exported/policy.pt"
)


def _default_log_root() -> str:
    return str((Path(__file__).resolve().parents[4] / "logs" / "rsl_rl" / "heave_landing_gru").resolve())


def _vehicle_ns(namespace: str, vehicle_id: int) -> str:
    return f"{namespace}{vehicle_id}"


def _default_topic(namespace: str, vehicle_id: int, key: str) -> str:
    return f"{_vehicle_ns(namespace, vehicle_id)}{DEFAULT_TOPIC_SUFFIXES[key]}"


def _resolve_default_policy_jit() -> str:
    env_override = os.environ.get("UAV_RL_HEAVE_POLICY_JIT", "").strip()
    if env_override:
        return env_override
    return DEFAULT_POLICY_JIT


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 node that reconstructs the heave_landing policy observation from transfer state topics, "
            "publishes velocity commands, and optionally safety-filters vz with an acados CBF solve."
        )
    )
    parser.add_argument("--namespace", type=str, default="transfer", help="ROS namespace prefix, e.g. 'transfer'.")
    parser.add_argument("--vehicle-id", type=int, default=0, help="Vehicle id suffix used in ROS topics.")
    parser.add_argument("--policy-rate-hz", type=float, default=20.0, help="Policy inference rate. 20 Hz matches heave_landing training.")
    parser.add_argument("--cmd-publish-rate-hz", type=float, default=20.0, help="Velocity command publish rate.")
    parser.add_argument(
        "--policy-jit",
        type=str,
        default=_resolve_default_policy_jit(),
        help="Path to exported heave_landing policy.pt. Pass an empty string or 'none' to disable JIT loading.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to an RSL-RL checkpoint file.")
    parser.add_argument(
        "--load-run",
        type=str,
        default=None,
        help="Run directory under logs/rsl_rl/heave_landing_gru when checkpoint-path is not provided.",
    )
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint filename inside the run directory.")
    parser.add_argument("--log-root", type=str, default=_default_log_root(), help="Root folder for heave_landing_gru logs.")
    parser.add_argument("--policy-device", type=str, default=None, help="Torch device for policy inference.")
    parser.add_argument(
        "--vehicle-z0-m",
        type=float,
        default=0.15,
        help="CG-to-landing-gear/root offset subtracted from relative z, matching heave_landing observations.",
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
        "--opt-on",
        type=int,
        default=0,
        help="If 1, optimize only vz with acados around the RL vz target.",
    )
    parser.add_argument("--desired-hover-alt", type=float, default=5.2)
    parser.add_argument(
        "--back-to-hover-duration",
        type=float,
        default=5.0,
        help="Seconds to wait at hover altitude before retrying.",
    )
    parser.add_argument(
        "--resume-after-recovery",
        type=int,
        default=1,
        help="If 1, restart inference after the hover wait.",
    )
    parser.add_argument(
        "--reset-policy-state-on-recovery",
        type=int,
        default=0,
        help="If 1, call the policy reset() before restarting inference.",
    )
    parser.add_argument("--recovery-vz-up-max", type=float, default=1.2)
    parser.add_argument("--recovery-vz-down-max", type=float, default=0.5)
    parser.add_argument("--v-min", type=float, default=-0.6)
    parser.add_argument("--v-max", type=float, default=1.5)
    parser.add_argument("--u-min", type=float, default=-2.0)
    parser.add_argument("--u-max", type=float, default=2.0)
    parser.add_argument(
        "--cbf-d-min",
        type=float,
        default=0.15,
        help="Direct override for CBF clearance.",
    )
    parser.add_argument(
        "--cbf-gamma",
        type=float,
        default=2.24,
        help="Direct override for CBF gamma.",
    )
    parser.add_argument(
        "--landing-velocity",
        type=float,
        default=-0.2,
        help="Direct override for landing velocity.",
    )
    parser.add_argument("--cbf-a-rel", type=float, default=0.7)
    parser.add_argument("--cbf-eps", type=float, default=1e-4)
    parser.add_argument("--opt-match-weight", type=float, default=1.0)
    parser.add_argument("--opt-u-reg", type=float, default=1.0e-3)
    parser.add_argument(
        "--enable-proximity-disarm",
        type=int,
        default=1,
        help="If 1, trigger disarm when relative |x|,|y|,|z| are below thresholds.",
    )
    parser.add_argument("--disarm-rel-x-threshold", type=float, default=0.15)
    parser.add_argument("--disarm-rel-y-threshold", type=float, default=0.15)
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
        "--platform-acc-topic",
        type=str,
        default=None,
        help="Optional AccelStamped topic for platform linear acceleration. If unset, acceleration is estimated from platform twist.",
    )
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
    from geometry_msgs.msg import AccelStamped, PoseStamped, Twist, TwistStamped, Vector3Stamped
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

    class _TransferPolicy:
        def __init__(
            self,
            *,
            device: str,
            policy_jit: str | None,
            checkpoint_path: str | None,
        ):
            self._device = torch.device(device)
            self._mode = None
            self._policy = None
            self._supports_reset = False

            if policy_jit is not None:
                self._mode = "jit"
                self._policy = torch.jit.load(policy_jit, map_location=self._device)
                self._policy.eval()
                self._supports_reset = hasattr(self._policy, "reset")
                return

            if checkpoint_path is not None:
                self._mode = "checkpoint"
                self._policy = RslRlPolicy(device=self._device, checkpoint_path=checkpoint_path)
                self._supports_reset = False
                return

            raise ValueError("Either `policy_jit` or `checkpoint_path` must be provided.")

        @property
        def supports_reset(self) -> bool:
            return self._supports_reset

        def act(self, obs_tensor: torch.Tensor) -> torch.Tensor:
            if self._mode == "jit":
                out = self._policy(obs_tensor)
            else:
                out = self._policy.act(obs_tensor)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

        def reset(self) -> None:
            if self._supports_reset:
                self._policy.reset()

    class RLHeavePublisher:
        STATE_TRACK_POLICY = 0
        STATE_RECOVER_TO_HOVER = 1
        STATE_RECOVER_HOLD = 2

        def __init__(self):
            try:
                rclpy.init()
            except Exception:
                pass

            self.start_time = time.monotonic()
            self.node = rclpy.create_node(f"rl_heave_publisher_{args.vehicle_id}")

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
            self.platform_acc_topic = _normalize_optional_string(args.platform_acc_topic)
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
            self.platform_acc_sub = None
            if self.platform_acc_topic is not None:
                self.platform_acc_sub = self.node.create_subscription(
                    AccelStamped, self.platform_acc_topic, self._platform_acc_callback, qos_best_effort
                )

            policy_jit = _normalize_optional_string(args.policy_jit)
            checkpoint_path = None
            if policy_jit is None:
                checkpoint_path = str(
                    resolve_checkpoint_path(
                        load_run=args.load_run,
                        checkpoint_name=args.checkpoint_name,
                        checkpoint_path=args.checkpoint_path,
                        log_root=args.log_root,
                    )
                )

            policy_device = _resolve_policy_device(args.policy_device)
            self.policy = _TransferPolicy(
                device=policy_device,
                policy_jit=policy_jit,
                checkpoint_path=checkpoint_path,
            )
            self.device = torch.device(policy_device)
            self.vehicle_z0_m = float(args.vehicle_z0_m)
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
            self.opt_on = bool(args.opt_on)
            self.desired_hover_alt = float(args.desired_hover_alt)
            self.back_to_hover_duration = float(args.back_to_hover_duration)
            self.resume_after_recovery = bool(args.resume_after_recovery)
            self.reset_policy_state_on_recovery = bool(args.reset_policy_state_on_recovery)
            self.recovery_vz_up_max = float(args.recovery_vz_up_max)
            self.recovery_vz_down_max = float(args.recovery_vz_down_max)
            self.v_min = float(args.v_min)
            self.v_max = float(args.v_max)
            self.u_min = float(args.u_min)
            self.u_max = float(args.u_max)
            self.cbf_d_min = float(args.cbf_d_min)
            self.cbf_gamma = float(args.cbf_gamma)
            self.landing_velocity = float(args.landing_velocity)
            self.cbf_a_rel = float(args.cbf_a_rel)
            self.cbf_eps = float(args.cbf_eps)
            self.opt_match_weight = float(args.opt_match_weight)
            self.opt_u_reg = float(args.opt_u_reg)
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

            self.robot_pos = None
            self.robot_quat_xyzw = None
            self.robot_lin_vel_w = None
            self.robot_ang_vel_b = None
            self.robot_lin_acc_w = np.zeros((3,), dtype=np.float32)
            self.platform_pos = None
            self.platform_quat_xyzw = None
            self.platform_lin_vel_w = None
            self.platform_ang_vel_w = None
            self.platform_lin_acc_w = np.zeros((3,), dtype=np.float32)
            self._input_ready = {
                "robot_pose": False,
                "robot_twist_body": False,
                "robot_twist_inertial": False,
                "platform_pose": False,
                "platform_twist": False,
            }
            self._prev_robot_lin_vel_w = None
            self._prev_robot_lin_vel_time = None
            self._prev_platform_lin_vel_w = None
            self._prev_platform_lin_vel_time = None
            self._platform_acc_from_topic = False
            self._use_imu_linear_acc = self.robot_twist_body_msg_type == "imu"

            self.timer_period = 1.0 / max(float(args.cmd_publish_rate_hz), 1.0)
            self.timer = self.node.create_timer(self.timer_period, self._on_timer)
            self.last_policy_time = 0.0
            self.policy_period = 1.0 / max(float(args.policy_rate_hz), 1.0)
            self._infer_every_tick = abs(float(args.policy_rate_hz) - float(args.cmd_publish_rate_hz)) < 1.0e-9
            self.step_count = 0
            self.last_policy_action = np.zeros((4,), dtype=np.float32)
            self.last_action = np.zeros((4,), dtype=np.float32)
            self._printed_inference_start = False
            self._warned_unready = False
            self._last_unready_log_time = 0.0
            self._proximity_disarm_done = False
            self.done_enter_time = None
            self.disarm_time = None
            self._stopped = False
            self._start_monotonic = time.monotonic()
            self.state = self.STATE_TRACK_POLICY
            self.recovery_count = 0
            self.recovery_enter_time = None
            self.recovery_hold_start = None
            self.recovery_reason = None
            self._recovery_hold_complete_logged = False
            self._pending_policy_reset = False
            self._last_cbf_metrics = None
            self._last_opt_status = "idle"
            self._acados_solver = None
            self._acados_cache_key = None
            self.cmd_long_cli = None
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

            if self.enable_proximity_disarm and self.disarm_via_service:
                if CommandLong is None:
                    self._log_warn(
                        "mavros_msgs.srv.CommandLong is unavailable; falling back to local disarm topic."
                    )
                    self.disarm_via_service = False
                else:
                    self.cmd_long_cli = self.node.create_client(CommandLong, f"{self.mavros_ns}/cmd/command")

            platform_acc_mode = self.platform_acc_topic if self.platform_acc_topic is not None else "finite-difference(platform_twist)"
            self._log_info(
                "RL heave publisher topics: "
                f"robot_pose='{self.robot_pose_topic}', "
                f"robot_twist_body='{self.robot_twist_body_topic}' ({self.robot_twist_body_msg_type}), "
                f"robot_twist_inertial='{self.robot_twist_inertial_topic}', "
                f"platform_pose='{self.platform_pose_topic}', "
                f"platform_twist='{self.platform_twist_topic}', "
                f"platform_acc='{platform_acc_mode}', "
                f"cmd_vel_out='{self.cmd_vel_topic}', "
                f"velocity_out='{self.velocity_cmd_topic}', "
                f"yaw_rate_out='{self.yaw_rate_cmd_topic}', "
                f"disarm_out='{self.disarm_cmd_topic}', "
                f"vel_limits=({self.velocity_lower_limits.tolist()} .. {self.velocity_upper_limits.tolist()}), "
                f"yaw_limits=({self.yaw_rate_lower_limit:.3f} .. {self.yaw_rate_upper_limit:.3f})\n"
                f"policy_jit='{policy_jit}' checkpoint='{checkpoint_path}' reset_supported={self.policy.supports_reset} device='{self.device}'\n"
                "heave CBF: "
                f"opt_on={self.opt_on}, d_min={self.cbf_d_min:.3f}, gamma={self.cbf_gamma:.3f}, "
                f"a_rel={self.cbf_a_rel:.3f}, landing_velocity={self.landing_velocity:.3f}\n"
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

        def _state_name(self) -> str:
            if self.state == self.STATE_TRACK_POLICY:
                return "TRACK_POLICY"
            if self.state == self.STATE_RECOVER_TO_HOVER:
                return "RECOVER_TO_HOVER"
            if self.state == self.STATE_RECOVER_HOLD:
                return "RECOVER_HOLD"
            return f"UNKNOWN_{self.state}"

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
                self._log_path = os.path.join(self.csv_log_dir, f"logs_{timestamp}_rl_heave_policy.csv")
                self._csv_fieldnames = [
                    "t",
                    "policy_step",
                    "state",
                    "recovery_count",
                    "opt_on",
                    "opt_status",
                    "recovery_reason",
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
                    "policy_action_vx",
                    "policy_action_vy",
                    "policy_action_vz",
                    "policy_action_yaw_rate",
                    "action_vx",
                    "action_vy",
                    "action_vz",
                    "action_yaw_rate",
                    "current_altitude",
                    "v_curr",
                    "a_curr",
                    "marker_pos",
                    "v_p",
                    "a_p",
                    "h_0",
                    "hdot",
                    "hdot_plus_gamma_h",
                    "dstop",
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

        def _build_csv_row(
            self,
            elapsed_time: float,
            obs: np.ndarray,
            policy_action: np.ndarray,
            final_action: np.ndarray,
            metrics: dict | None,
        ):
            if self._csv_writer is None:
                return None
            if obs is None or policy_action is None or final_action is None:
                return None
            if self.robot_pos is None or self.platform_pos is None:
                return None
            rel_pos = self._compute_rel_pos_world()
            return {
                "t": float(elapsed_time),
                "policy_step": int(self.step_count),
                "state": self._state_name(),
                "recovery_count": int(self.recovery_count),
                "opt_on": int(self.opt_on),
                "opt_status": self._last_opt_status,
                "recovery_reason": self.recovery_reason,
                "robot_pos_x": float(self.robot_pos[0]),
                "robot_pos_y": float(self.robot_pos[1]),
                "robot_pos_z": float(self.robot_pos[2]),
                "platform_pos_x": float(self.platform_pos[0]),
                "platform_pos_y": float(self.platform_pos[1]),
                "platform_pos_z": float(self.platform_pos[2]),
                "rel_pos_x": None if rel_pos is None else float(rel_pos[0]),
                "rel_pos_y": None if rel_pos is None else float(rel_pos[1]),
                "rel_pos_z": None if rel_pos is None else float(rel_pos[2]),
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
                "policy_action_vx": float(policy_action[0]),
                "policy_action_vy": float(policy_action[1]),
                "policy_action_vz": float(policy_action[2]),
                "policy_action_yaw_rate": float(policy_action[3]),
                "action_vx": float(final_action[0]),
                "action_vy": float(final_action[1]),
                "action_vz": float(final_action[2]),
                "action_yaw_rate": float(final_action[3]),
                "current_altitude": float(self.robot_pos[2]),
                "v_curr": None if self.robot_lin_vel_w is None else float(self.robot_lin_vel_w[2]),
                "a_curr": float(self.robot_lin_acc_w[2]),
                "marker_pos": float(self.platform_pos[2]),
                "v_p": None if self.platform_lin_vel_w is None else float(self.platform_lin_vel_w[2]),
                "a_p": float(self.platform_lin_acc_w[2]),
                "h_0": None if metrics is None else metrics.get("h_0"),
                "hdot": None if metrics is None else metrics.get("hdot"),
                "hdot_plus_gamma_h": None if metrics is None else metrics.get("hdot_plus_gamma_h"),
                "dstop": None if metrics is None else metrics.get("dstop"),
            }

        def _log_csv_row(
            self,
            elapsed_time: float,
            obs: np.ndarray,
            policy_action: np.ndarray,
            final_action: np.ndarray,
            metrics: dict | None,
        ) -> None:
            row = self._build_csv_row(elapsed_time, obs, policy_action, final_action, metrics)
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
            if self._csv_file is not None:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                except Exception as exc:
                    self._log_warn(f"CSV log close failed: {exc}")
                finally:
                    self._csv_file = None
                    self._csv_writer = None

        def _message_time_sec(self, header) -> float:
            stamp = getattr(header, "stamp", None)
            if stamp is None:
                return time.monotonic()
            sec = float(getattr(stamp, "sec", 0.0))
            nanosec = float(getattr(stamp, "nanosec", 0.0))
            stamp_sec = sec + 1.0e-9 * nanosec
            if stamp_sec > 0.0:
                return stamp_sec
            return time.monotonic()

        def _estimate_linear_acceleration(
            self,
            velocity: np.ndarray,
            timestamp_sec: float,
            prev_velocity: np.ndarray | None,
            prev_timestamp_sec: float | None,
        ) -> np.ndarray | None:
            if prev_velocity is None or prev_timestamp_sec is None:
                return None
            dt = float(timestamp_sec - prev_timestamp_sec)
            if dt <= 1.0e-5:
                return None
            return ((velocity - prev_velocity) / dt).astype(np.float32)

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
            self._input_ready["robot_pose"] = True

        def _twist_callback(self, msg: TwistStamped):
            self.robot_ang_vel_b = np.array(
                [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=np.float32
            )
            self._input_ready["robot_twist_body"] = True

        def _imu_callback(self, msg: Imu):
            self.robot_ang_vel_b = np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float32
            )

            quat_xyzw = None
            q = msg.orientation
            norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
            if norm >= 1.0e-6:
                quat_xyzw = np.array([q.x / norm, q.y / norm, q.z / norm, q.w / norm], dtype=np.float32)
            elif self.robot_quat_xyzw is not None:
                quat_xyzw = self.robot_quat_xyzw

            ax = float(msg.linear_acceleration.x)
            ay = float(msg.linear_acceleration.y)
            az = float(msg.linear_acceleration.z)
            if quat_xyzw is not None:
                try:
                    rot = Rotation.from_quat(quat_xyzw)
                    a_world = rot.apply(np.array([ax, ay, az], dtype=np.float32))
                    a_world[2] -= 9.80665
                    self.robot_lin_acc_w = a_world.astype(np.float32)
                except Exception:
                    self.robot_lin_acc_w = np.array([0.0, 0.0, az - 9.80665], dtype=np.float32)
            else:
                self.robot_lin_acc_w = np.array([0.0, 0.0, az - 9.80665], dtype=np.float32)

            self._input_ready["robot_twist_body"] = True

        def _twist_inertial_callback(self, msg: TwistStamped):
            velocity = np.array(
                [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32
            )
            timestamp_sec = self._message_time_sec(msg.header)
            if not self._use_imu_linear_acc:
                estimate = self._estimate_linear_acceleration(
                    velocity,
                    timestamp_sec,
                    self._prev_robot_lin_vel_w,
                    self._prev_robot_lin_vel_time,
                )
                if estimate is not None:
                    self.robot_lin_acc_w = estimate
            self.robot_lin_vel_w = velocity
            self._prev_robot_lin_vel_w = velocity
            self._prev_robot_lin_vel_time = timestamp_sec
            self._input_ready["robot_twist_inertial"] = True

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
            self._input_ready["platform_pose"] = True

        def _platform_twist_callback(self, msg: TwistStamped):
            velocity = np.array(
                [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32
            )
            timestamp_sec = self._message_time_sec(msg.header)
            if not self._platform_acc_from_topic:
                estimate = self._estimate_linear_acceleration(
                    velocity,
                    timestamp_sec,
                    self._prev_platform_lin_vel_w,
                    self._prev_platform_lin_vel_time,
                )
                if estimate is not None:
                    self.platform_lin_acc_w = estimate
            self.platform_lin_vel_w = velocity
            self.platform_ang_vel_w = np.array(
                [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=np.float32
            )
            self._prev_platform_lin_vel_w = velocity
            self._prev_platform_lin_vel_time = timestamp_sec
            self._input_ready["platform_twist"] = True

        def _platform_acc_callback(self, msg: AccelStamped):
            self.platform_lin_acc_w = np.array(
                [msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z], dtype=np.float32
            )
            self._platform_acc_from_topic = True

        def _ready(self) -> bool:
            return all(self._input_ready.values())

        def _build_observation(self) -> np.ndarray:
            robot_rot = Rotation.from_quat(self.robot_quat_xyzw)
            rel_pos = (self.robot_pos - self.platform_pos).astype(np.float32)
            rel_pos[2] -= self.vehicle_z0_m
            rel_lin_vel = (self.robot_lin_vel_w - self.platform_lin_vel_w).astype(np.float32)
            robot_ang_vel_w = robot_rot.apply(self.robot_ang_vel_b)
            rel_ang_vel = (robot_ang_vel_w - self.platform_ang_vel_w).astype(np.float32)
            projected_gravity = robot_rot.inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float32))
            return np.concatenate(
                (
                    rel_pos,
                    rel_lin_vel,
                    _quat_xyzw_to_wxyz(self.robot_quat_xyzw.astype(np.float32)),
                    rel_ang_vel,
                    projected_gravity.astype(np.float32),
                    self.last_action.astype(np.float32),
                )
            )

        def _compute_rel_pos_world(self) -> np.ndarray | None:
            if self.robot_pos is None or self.platform_pos is None:
                return None
            rel_pos = (self.robot_pos - self.platform_pos).astype(np.float32)
            rel_pos[2] -= self.vehicle_z0_m
            return rel_pos

        def _compute_rel_lin_vel_world(self) -> np.ndarray | None:
            if self.robot_lin_vel_w is None or self.platform_lin_vel_w is None:
                return None
            return (self.robot_lin_vel_w - self.platform_lin_vel_w).astype(np.float32)

        def _check_and_trigger_proximity_disarm(self, now_sec: float) -> None:
            if not self.enable_proximity_disarm or self._proximity_disarm_done:
                return
            rel_pos = self._compute_rel_pos_world()
            if rel_pos is None:
                return

            ax = float(abs(rel_pos[0]))
            ay = float(abs(rel_pos[1]))
            az = float(abs(rel_pos[2]))
            if (
                ax <= self.disarm_rel_x_threshold
                and ay <= self.disarm_rel_y_threshold
                and az <= self.disarm_rel_z_threshold
            ):
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

        def _publish_cmd(self, world_velocity_sp: np.ndarray, yaw_rate_sp: float) -> None:
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

            self._log_warn("Force disarm timed out with no response")
            return False

        def _on_force_disarm_response(self, future) -> bool:
            try:
                resp = future.result()
                if resp and bool(resp.success):
                    self._log_warn(f"Force disarm accepted (result={resp.result})")
                    return True
                self._log_warn(f"Force disarm failed (resp={resp})")
                return False
            except Exception as exc:
                self._log_warn(f"Force disarm call error: {exc}")
                return False

        def _publish_local_disarm(self) -> None:
            msg = Bool()
            msg.data = True
            self.disarm_cmd_pub.publish(msg)
            self._log_warn(f"Published local disarm request on '{self.disarm_cmd_topic}'")

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

        def _compute_cbf_metrics_current(self):
            if (
                self.robot_pos is None
                or self.platform_pos is None
                or self.robot_lin_vel_w is None
                or self.platform_lin_vel_w is None
            ):
                return None
            try:
                d0 = float(self.robot_pos[2] - self.platform_pos[2])
                vr0 = float(self.robot_lin_vel_w[2] - self.platform_lin_vel_w[2])
                vr_neg = 0.5 * (vr0 - math.sqrt(vr0 * vr0 + self.cbf_eps))
                dstop = (vr_neg * vr_neg - self.landing_velocity * self.landing_velocity) / (2.0 * self.cbf_a_rel)
                h0 = (d0 - self.cbf_d_min) - dstop
                u0_est = float(self.robot_lin_acc_w[2])
                a_p_use = float(self.platform_lin_acc_w[2])
                hdot = vr_neg - (vr_neg / self.cbf_a_rel) * (u0_est - a_p_use)
                metrics = {
                    "d0": d0,
                    "d0_minus_dmin": d0 - self.cbf_d_min,
                    "vr0": vr0,
                    "vr_neg": vr_neg,
                    "v_land": self.landing_velocity,
                    "h_0": h0,
                    "hdot": hdot,
                    "hdot_plus_gamma_h": hdot + self.cbf_gamma * h0,
                    "dstop": dstop,
                    "u0": u0_est,
                }
                self._last_cbf_metrics = metrics
                return metrics
            except Exception:
                return None

        def _ensure_acados_solver(self, dt: float):
            cache_key = (round(dt, 12),)
            if self._acados_solver is not None and self._acados_cache_key == cache_key:
                return self._acados_solver

            try:
                import casadi as ca
                from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
            except Exception as exc:
                self._log_warn(f"acados import failed ({type(exc).__name__}: {exc})")
                return None

            try:
                x = ca.SX.sym("x", 2)
                u = ca.SX.sym("u", 1)
                p = ca.SX.sym("p", 4)

                v_rl_k = p[0]
                marker_z0_k = p[1]
                v_p0_k = p[2]
                a_p0_k = p[3]

                model = AcadosModel()
                model.name = f"rl_heave_cbf_{os.getpid()}_{int(round(dt * 1e6))}"
                model.x = x
                model.u = u
                model.p = p
                model.disc_dyn_expr = ca.vertcat(
                    x[0] + x[1] * dt,
                    x[1] + u[0] * dt,
                )
                model.cost_expr_ext_cost = self.opt_u_reg * ca.power(u[0], 2)
                model.cost_expr_ext_cost_e = self.opt_match_weight * ca.power(x[1] - v_rl_k, 2)

                vr0_expr = x[1] - v_p0_k
                vr_neg_expr = 0.5 * (vr0_expr - ca.sqrt(vr0_expr * vr0_expr + self.cbf_eps))
                h0_expr = (
                    (x[0] - marker_z0_k - self.cbf_d_min)
                    - (vr_neg_expr * vr_neg_expr - self.landing_velocity * self.landing_velocity)
                    / (2.0 * self.cbf_a_rel)
                )
                hdot_expr = vr_neg_expr - (vr_neg_expr / self.cbf_a_rel) * (u[0] - a_p0_k)
                cbf_expr = hdot_expr + self.cbf_gamma * h0_expr
                model.con_h_expr = ca.vertcat(cbf_expr, h0_expr)
                model.con_h_expr_0 = ca.vertcat(cbf_expr, h0_expr)

                ocp = AcadosOcp()
                ocp.model = model
                ocp.solver_options.N_horizon = 1
                ocp.parameter_values = np.zeros(4, dtype=float)
                ocp.cost.cost_type = "EXTERNAL"
                ocp.cost.cost_type_e = "EXTERNAL"
                ocp.constraints.x0 = np.zeros(2, dtype=float)
                ocp.constraints.idxbx = np.array([1], dtype=np.int32)
                ocp.constraints.lbx = np.array([self.v_min], dtype=float)
                ocp.constraints.ubx = np.array([self.v_max], dtype=float)
                ocp.constraints.idxbx_e = np.array([1], dtype=np.int32)
                ocp.constraints.lbx_e = np.array([self.v_min], dtype=float)
                ocp.constraints.ubx_e = np.array([self.v_max], dtype=float)
                ocp.constraints.idxbu = np.array([0], dtype=np.int32)
                ocp.constraints.lbu = np.array([self.u_min], dtype=float)
                ocp.constraints.ubu = np.array([self.u_max], dtype=float)
                ocp.constraints.lh_0 = np.array([0.0, 0.0], dtype=float)
                ocp.constraints.uh_0 = np.array([1.0e9, 1.0e9], dtype=float)
                ocp.constraints.lh = np.array([0.0, 0.0], dtype=float)
                ocp.constraints.uh = np.array([1.0e9, 1.0e9], dtype=float)
                ocp.solver_options.tf = dt
                ocp.solver_options.integrator_type = "DISCRETE"
                ocp.solver_options.nlp_solver_type = "SQP"
                ocp.solver_options.nlp_solver_max_iter = 100
                ocp.solver_options.hessian_approx = "EXACT"
                ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
                ocp.solver_options.print_level = 0
                ocp.code_gen_opts.code_export_directory = os.path.join(
                    "/tmp", f"acados_rl_heave_{os.getpid()}_{int(round(dt * 1e6))}"
                )
                json_file = os.path.join(
                    "/tmp", f"acados_rl_heave_{os.getpid()}_{int(round(dt * 1e6))}.json"
                )
                solver = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
                self._acados_solver = solver
                self._acados_cache_key = cache_key
                return solver
            except Exception as exc:
                self._log_warn(f"acados solver setup failed ({type(exc).__name__}: {exc})")
                return None

        def _optimize_vz(self, vz_rl: float):
            if (
                self.robot_pos is None
                or self.platform_pos is None
                or self.robot_lin_vel_w is None
                or self.platform_lin_vel_w is None
            ):
                self._last_opt_status = "unready"
                return None

            dt = float(self.timer_period)
            solver = self._ensure_acados_solver(dt)
            if solver is None:
                self._last_opt_status = "solver_unavailable"
                return None

            z_curr = float(self.robot_pos[2])
            v_curr = float(self.robot_lin_vel_w[2])
            marker_pos = float(self.platform_pos[2])
            v_p0 = float(self.platform_lin_vel_w[2])
            a_p0 = float(self.platform_lin_acc_w[2])
            vz_rl = float(np.clip(vz_rl, self.velocity_lower_limits[2], self.velocity_upper_limits[2]))
            x0 = np.array([z_curr, v_curr], dtype=float)

            try:
                solver.set(0, "lbx", x0)
                solver.set(0, "ubx", x0)
                solver.set(0, "p", np.array([vz_rl, marker_pos, v_p0, a_p0], dtype=float))
                solver.set(1, "p", np.array([vz_rl, marker_pos, v_p0, a_p0], dtype=float))
                solver.set(0, "x", x0)
                solver.set(0, "u", np.array([0.0], dtype=float))
                solver.set(1, "x", np.array([z_curr + v_curr * dt, vz_rl], dtype=float))
            except Exception as exc:
                self._log_warn(f"acados parameter setup failed ({type(exc).__name__}: {exc})")
                self._last_opt_status = "parameter_setup_failed"
                return None

            try:
                status = solver.solve()
            except Exception as exc:
                self._log_warn(f"acados solve failed ({type(exc).__name__}: {exc})")
                self._last_opt_status = "solve_exception"
                return None

            if status != 0:
                self._log_warn(f"acados returned non-zero status: {status}")
                self._last_opt_status = f"solve_status_{status}"
                return None

            try:
                v_cmd = float(solver.get(1, "x")[1])
                u0 = float(solver.get(0, "u")[0])
                d0 = z_curr - marker_pos
                vr0 = v_curr - v_p0
                vr_neg = 0.5 * (vr0 - math.sqrt(vr0 * vr0 + self.cbf_eps))
                dstop = (vr_neg * vr_neg - self.landing_velocity * self.landing_velocity) / (2.0 * self.cbf_a_rel)
                h0 = (d0 - self.cbf_d_min) - dstop
                hdot = vr_neg - (vr_neg / self.cbf_a_rel) * (u0 - a_p0)
                self._last_cbf_metrics = {
                    "d0": d0,
                    "d0_minus_dmin": d0 - self.cbf_d_min,
                    "vr0": vr0,
                    "vr_neg": vr_neg,
                    "v_land": self.landing_velocity,
                    "h_0": h0,
                    "hdot": hdot,
                    "hdot_plus_gamma_h": hdot + self.cbf_gamma * h0,
                    "dstop": dstop,
                    "u0": u0,
                }
                self._last_opt_status = "ok"
                return float(np.clip(v_cmd, self.velocity_lower_limits[2], self.velocity_upper_limits[2]))
            except Exception as exc:
                self._log_warn(f"acados solution extraction failed ({type(exc).__name__}: {exc})")
                self._last_opt_status = "solution_extract_failed"
                return None

        def _enter_recovery(self, reason: str, now: float) -> None:
            if self.state != self.STATE_TRACK_POLICY:
                return
            self.recovery_count += 1
            self.recovery_reason = reason
            self.recovery_enter_time = now
            self.recovery_hold_start = None
            self._recovery_hold_complete_logged = False
            self.state = self.STATE_RECOVER_TO_HOVER
            self._pending_policy_reset = self.reset_policy_state_on_recovery
            self._log_warn(f"Entering recovery loop: {reason}")

        def _reset_policy_context(self) -> None:
            if self._pending_policy_reset:
                if self.policy.supports_reset:
                    try:
                        self.policy.reset()
                        self._log_info("Policy state reset before retry.")
                    except Exception as exc:
                        self._log_warn(f"Policy reset() failed: {exc}")
                else:
                    self._log_warn("Policy reset requested, but the loaded policy does not expose reset().")
            self.last_policy_action[:] = 0.0
            self.last_action[:] = 0.0
            self.last_policy_time = 0.0
            self._pending_policy_reset = False

        def _recovery_cmd(self) -> np.ndarray:
            vz = 0.0
            if self.robot_pos is not None:
                err = float(self.desired_hover_alt - self.robot_pos[2])
                vz = float(np.clip(err, -self.recovery_vz_down_max, self.recovery_vz_up_max))
            cmd = np.zeros((4,), dtype=np.float32)
            cmd[2] = vz
            return cmd

        def _on_timer(self) -> None:
            now = time.monotonic()
            if not self._ready():
                if (not self._warned_unready) or ((now - self._last_unready_log_time) >= 2.0):
                    missing = [key for key, ready in self._input_ready.items() if not ready]
                    self._log_info(
                        "Waiting for inputs before activating heave policy. Missing: " + ", ".join(missing)
                    )
                    self._warned_unready = True
                    self._last_unready_log_time = now
                return

            self._warned_unready = False
            if not self._printed_inference_start:
                self._log_info("All required inputs received. Starting heave policy inference.")
                self._printed_inference_start = True

            self._check_and_trigger_proximity_disarm(now)
            if self._proximity_disarm_done:
                self.last_policy_action[:] = 0.0
                self.last_action[:] = 0.0
                self._publish_cmd(self.last_action[:3], float(self.last_action[3]))
                ref_time = self.disarm_time if self.disarm_time is not None else self.done_enter_time
                if ref_time is not None and (now - ref_time) >= self.post_disarm_hold_seconds:
                    self._stop_node("Landing completed; shutting down node.")
                return

            if self.state == self.STATE_RECOVER_TO_HOVER:
                recovery_action = self._recovery_cmd()
                self.last_action = recovery_action
                self._publish_cmd(recovery_action[:3], float(recovery_action[3]))
                if self.robot_pos is not None and abs(float(self.robot_pos[2]) - self.desired_hover_alt) <= 0.05:
                    self.state = self.STATE_RECOVER_HOLD
                    self.recovery_hold_start = now
                    self._log_info("Entered RECOVER_HOLD")
                return

            if self.state == self.STATE_RECOVER_HOLD:
                recovery_action = self._recovery_cmd()
                self.last_action = recovery_action
                self._publish_cmd(recovery_action[:3], float(recovery_action[3]))
                if self.recovery_hold_start is not None and (now - self.recovery_hold_start) >= self.back_to_hover_duration:
                    if self.resume_after_recovery:
                        self._reset_policy_context()
                        self.state = self.STATE_TRACK_POLICY
                        self.recovery_reason = None
                        self._log_info("Recovery hold complete; restarting policy inference.")
                    elif not self._recovery_hold_complete_logged:
                        self._recovery_hold_complete_logged = True
                        self._log_info(
                            "Recovery hold complete; staying at hover because resume_after_recovery=false."
                        )
                return

            metrics = self._compute_cbf_metrics_current()
            if metrics is not None and float(metrics["h_0"]) < 0.0:
                self._last_opt_status = "h_negative"
                self._enter_recovery("current h < 0", now)
                recovery_action = self._recovery_cmd()
                self.last_action = recovery_action
                self._publish_cmd(recovery_action[:3], float(recovery_action[3]))
                return

            should_infer = self._infer_every_tick or ((now - self.last_policy_time) >= self.policy_period)
            obs = None
            just_inferred = False
            if should_infer:
                obs = self._build_observation()
                obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                try:
                    with torch.inference_mode():
                        action = self.policy.act(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)
                except Exception as exc:
                    self._log_warn(f"Policy inference failed ({type(exc).__name__}: {exc})")
                    self._last_opt_status = "policy_inference_failed"
                    self._enter_recovery("policy inference failed", now)
                    recovery_action = self._recovery_cmd()
                    self.last_action = recovery_action
                    self._publish_cmd(recovery_action[:3], float(recovery_action[3]))
                    return

                action[:3] = np.clip(action[:3], self.velocity_lower_limits, self.velocity_upper_limits)
                action[3] = float(np.clip(action[3], self.yaw_rate_lower_limit, self.yaw_rate_upper_limit))
                self.last_policy_action = action
                self.last_policy_time = now
                self.step_count += 1
                just_inferred = True

            final_action = self.last_policy_action.copy()
            if self.opt_on:
                vz_safe = self._optimize_vz(float(self.last_policy_action[2]))
                if vz_safe is None:
                    self._enter_recovery("optimization failed", now)
                    recovery_action = self._recovery_cmd()
                    self.last_action = recovery_action
                    self._publish_cmd(recovery_action[:3], float(recovery_action[3]))
                    return
                final_action[2] = vz_safe
            else:
                self._last_opt_status = "bypass"

            final_action[:3] = np.clip(final_action[:3], self.velocity_lower_limits, self.velocity_upper_limits)
            final_action[3] = float(np.clip(final_action[3], self.yaw_rate_lower_limit, self.yaw_rate_upper_limit))
            self.last_action = final_action
            self._publish_cmd(final_action[:3], float(final_action[3]))

            if just_inferred and obs is not None:
                t_rel = now - self._start_monotonic
                self._log_csv_row(t_rel, obs, self.last_policy_action, final_action, self._last_cbf_metrics)
                if args.debug_every > 0 and self.step_count % args.debug_every == 0:
                    h0 = None if self._last_cbf_metrics is None else self._last_cbf_metrics.get("h_0")
                    rel_pos = self._compute_rel_pos_world()
                    rel_vel = self._compute_rel_lin_vel_world()
                    self._log_info(
                        f"policy_step={self.step_count} state={self._state_name()} h={h0} "
                        f"policy_vz={float(self.last_policy_action[2]):.3f} final_vz={float(final_action[2]):.3f} "
                        f"yaw_rate={float(final_action[3]):.3f} opt_status={self._last_opt_status} "
                        f"rel_pos={None if rel_pos is None else rel_pos.tolist()} "
                        f"rel_vel_true={None if rel_vel is None else rel_vel.tolist()}"
                    )

        def run(self) -> None:
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
        RLHeavePublisher().run()
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
