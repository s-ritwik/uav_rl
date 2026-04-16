from __future__ import annotations

import argparse
import os
import sys
import time
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
    parser.add_argument("--policy-jit", type=str, default=None, help="Path to exported policy.pt.")
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
        default=0.053,
        help="Constant landing-gear/root offset subtracted from relative z to match landing_sway observations.",
    )
    parser.add_argument("--velocity-limit-x", type=float, default=1.2)
    parser.add_argument("--velocity-limit-y", type=float, default=1.2)
    parser.add_argument("--velocity-limit-z", type=float, default=1.0)
    parser.add_argument("--yaw-rate-limit", type=float, default=3.0)
    parser.add_argument("--debug-every", type=int, default=0, help="Print action debug every N policy steps.")
    parser.add_argument("--robot-pose-topic", type=str, default=None, help="ROS topic for robot pose input.")
    parser.add_argument(
        "--robot-twist-body-topic",
        type=str,
        default=None,
        help="ROS topic for robot body-frame angular velocity input.",
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
    return parser


def _resolve_policy_device(requested: str | None) -> str:
    if requested:
        return requested

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


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

    import torch
    from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Vector3Stamped
    from scipy.spatial.transform import Rotation
    from std_msgs.msg import Float32

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

            self.cmd_pub = self.node.create_publisher(Twist, self.cmd_vel_topic, 10)
            self.velocity_cmd_pub = self.node.create_publisher(Vector3Stamped, self.velocity_cmd_topic, 10)
            self.yaw_rate_cmd_pub = self.node.create_publisher(Float32, self.yaw_rate_cmd_topic, 10)

            self.pose_sub = self.node.create_subscription(
                PoseStamped, self.robot_pose_topic, self._pose_callback, 10
            )
            self.twist_sub = self.node.create_subscription(
                TwistStamped, self.robot_twist_body_topic, self._twist_callback, 10
            )
            self.twist_inertial_sub = self.node.create_subscription(
                TwistStamped,
                self.robot_twist_inertial_topic,
                self._twist_inertial_callback,
                10,
            )
            self.platform_pose_sub = self.node.create_subscription(
                PoseStamped, self.platform_pose_topic, self._platform_pose_callback, 10
            )
            self.platform_twist_sub = self.node.create_subscription(
                TwistStamped, self.platform_twist_topic, self._platform_twist_callback, 10
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
            self.velocity_limits = np.array(
                [args.velocity_limit_x, args.velocity_limit_y, args.velocity_limit_z], dtype=np.float32
            )
            self.yaw_rate_limit = float(args.yaw_rate_limit)
            self.vehicle_z0_m = float(args.vehicle_z0_m)
            self.last_action = np.zeros((4,), dtype=np.float32)

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

            self.node.get_logger().info(
                "RL landing publisher topics: "
                f"robot_pose='{self.robot_pose_topic}', "
                f"robot_twist_body='{self.robot_twist_body_topic}', "
                f"robot_twist_inertial='{self.robot_twist_inertial_topic}', "
                f"platform_pose='{self.platform_pose_topic}', "
                f"platform_twist='{self.platform_twist_topic}', "
                f"cmd_vel_out='{self.cmd_vel_topic}', "
                f"velocity_out='{self.velocity_cmd_topic}', "
                f"yaw_rate_out='{self.yaw_rate_cmd_topic}'"
            )

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
            platform_rot = Rotation.from_quat(self.platform_quat_xyzw)

            # Observation order must match landing_sway/PolicyCfg exactly:
            # rel_pos, rel_lin_vel, rel_quat, rel_ang_vel, projected_gravity, last_action.
            rel_pos = platform_rot.inv().apply(self.robot_pos - self.platform_pos)
            rel_pos[2] -= self.vehicle_z0_m

            rel_lin_vel = platform_rot.inv().apply(self.robot_lin_vel_w - self.platform_lin_vel_w)
            rel_quat_xyzw = (platform_rot.inv() * robot_rot).as_quat()

            # Robot angular velocity is received in robot body frame. Convert to world,
            # subtract platform world angular velocity, then express the result in platform frame.
            robot_ang_vel_w = robot_rot.apply(self.robot_ang_vel_b)
            rel_ang_vel = platform_rot.inv().apply(robot_ang_vel_w - self.platform_ang_vel_w)

            # landing_sway uses robot projected gravity in body frame.
            projected_gravity = robot_rot.inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float32))

            return np.concatenate(
                (
                    rel_pos.astype(np.float32),
                    rel_lin_vel.astype(np.float32),
                    _quat_xyzw_to_wxyz(rel_quat_xyzw.astype(np.float32)),
                    rel_ang_vel.astype(np.float32),
                    projected_gravity.astype(np.float32),
                    self.last_action,
                )
            )

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
                    self.node.get_logger().info(
                        "Waiting for robot/platform pose and twist topics before activating the landing policy."
                    )
                    self._warned_unready = True
                return

            self._warned_unready = False

            if now - self.last_policy_time >= self.policy_period:
                obs = self._build_observation()
                obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                with torch.inference_mode():
                    action = self.policy.act(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)

                action[:3] = np.clip(action[:3], -self.velocity_limits, self.velocity_limits)
                action[3] = float(np.clip(action[3], -self.yaw_rate_limit, self.yaw_rate_limit))
                self.last_action = action
                self.last_policy_time = now
                self.step_count += 1

                if args.debug_every > 0 and self.step_count % args.debug_every == 0:
                    self.node.get_logger().info(
                        f"policy_step={self.step_count} obs_z={float(obs[2]):.3f} "
                        f"vel_sp={self.last_action[:3].tolist()} yaw_rate={float(self.last_action[3]):.3f}"
                    )

            self._publish_cmd(self.last_action[:3], float(self.last_action[3]))

        def run(self):
            try:
                rclpy.spin(self.node)
            except KeyboardInterrupt:
                pass
            finally:
                self._publish_cmd(np.zeros((3,), dtype=np.float32), 0.0)
                self.node.destroy_node()

    try:
        RLLandingPublisher().run()
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
