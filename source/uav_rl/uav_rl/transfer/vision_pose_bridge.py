#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from precision_landing_using_vision_msgs.msg import LandingTargetVision


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _serialize_pose(stream_name: str, msg: LandingTargetVision) -> dict[str, object]:
    return {
        "stream": stream_name,
        "stamp_sec": _stamp_to_sec(msg.header.stamp),
        "valid": bool(msg.rel_pos_valid),
        "pos": [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ],
        "vel": [
            float(msg.rel_vel.x),
            float(msg.rel_vel.y),
            float(msg.rel_vel.z),
        ],
        "euler_deg": [
            float(msg.euler_angle.x),
            float(msg.euler_angle.y),
            float(msg.euler_angle.z),
        ],
        "angle_norm_deg": float(msg.angle_norm),
    }


class PoseRelayNode:
    def __init__(self, raw_topic: str, filtered_topic: str):
        self.node = rclpy.create_node("uav_rl_vision_pose_bridge")
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.node.create_subscription(
            LandingTargetVision, raw_topic, lambda msg: self._emit("raw", msg), qos_best_effort
        )
        self.node.create_subscription(
            LandingTargetVision, filtered_topic, lambda msg: self._emit("filtered", msg), qos_best_effort
        )

    def _emit(self, stream_name: str, msg: LandingTargetVision):
        payload = _serialize_pose(stream_name, msg)
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def spin(self):
        rclpy.spin(self.node)

    def stop(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Relay LandingTargetVision messages to JSON lines on stdout.")
    parser.add_argument("--raw-topic", type=str, required=True)
    parser.add_argument("--filtered-topic", type=str, required=True)
    args = parser.parse_args()

    rclpy.init()
    relay = PoseRelayNode(args.raw_topic, args.filtered_topic)
    try:
        relay.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        relay.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
