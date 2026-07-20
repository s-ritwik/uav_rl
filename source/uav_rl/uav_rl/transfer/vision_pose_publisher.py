#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import select
import sys

import rclpy
from geometry_msgs.msg import Quaternion
from precision_landing_using_vision_msgs.msg import LandingTargetVision
from rclpy.executors import ExternalShutdownException


def _quat_from_payload(values) -> Quaternion:
    q = Quaternion()
    q.x = float(values[0])
    q.y = float(values[1])
    q.z = float(values[2])
    q.w = float(values[3])
    return q


def _raw_msg_from_payload(payload: dict) -> LandingTargetVision:
    raw = payload["raw"]
    msg = LandingTargetVision()
    msg.header.stamp.sec = int(payload.get("header_stamp_sec", 0))
    msg.header.stamp.nanosec = int(payload.get("header_stamp_nanosec", 0))
    msg.header.frame_id = "ARpose_raw"
    msg.target_type = LandingTargetVision.STATIONARY
    msg.rel_pos_valid = 1 if bool(raw.get("valid", False)) else 0
    msg.rel_vel_valid = 0

    if bool(raw.get("valid", False)):
        msg.pose.position.x = float(raw["position_m"][0])
        msg.pose.position.y = float(raw["position_m"][1])
        msg.pose.position.z = float(raw["position_m"][2])
        msg.pose.orientation = _quat_from_payload(raw["quat_xyzw"])
        msg.rel_vel.x = float(raw["rvec"][0])
        msg.rel_vel.y = float(raw["rvec"][1])
        msg.rel_vel.z = float(raw["rvec"][2])
        msg.euler_angle.x = float(raw["euler_deg"][0])
        msg.euler_angle.y = float(raw["euler_deg"][1])
        msg.euler_angle.z = float(raw["euler_deg"][2])
        msg.angle_norm = math.sqrt(sum(float(v) * float(v) for v in raw["euler_deg"]))
    else:
        msg.pose.orientation.w = 1.0
        msg.angle_norm = 0.0

    return msg


def _filtered_msg_from_payload(payload: dict) -> LandingTargetVision:
    filtered = payload["filtered"]
    msg = LandingTargetVision()
    msg.header.stamp.sec = int(payload.get("header_stamp_sec", 0))
    msg.header.stamp.nanosec = int(payload.get("header_stamp_nanosec", 0))
    msg.header.frame_id = "ARpose_mekf_filtered"
    msg.target_type = LandingTargetVision.STATIONARY
    valid = bool(filtered.get("valid", False))
    msg.rel_pos_valid = 1 if valid else 0
    msg.rel_vel_valid = 1 if valid else 0

    msg.pose.position.x = float(filtered["position_m"][0])
    msg.pose.position.y = float(filtered["position_m"][1])
    msg.pose.position.z = float(filtered["position_m"][2])
    msg.pose.orientation = _quat_from_payload(filtered["quat_xyzw"])
    msg.rel_vel.x = float(filtered["velocity_mps"][0])
    msg.rel_vel.y = float(filtered["velocity_mps"][1])
    msg.rel_vel.z = float(filtered["velocity_mps"][2])
    msg.euler_angle.x = float(filtered["euler_deg"][0])
    msg.euler_angle.y = float(filtered["euler_deg"][1])
    msg.euler_angle.z = float(filtered["euler_deg"][2])
    msg.angle_norm = 2.0 * math.sqrt(
        float(filtered["quat_xyzw"][0]) ** 2
        + float(filtered["quat_xyzw"][1]) ** 2
        + float(filtered["quat_xyzw"][2]) ** 2
    ) * 180.0 / math.pi
    return msg


def _true_msg_from_payload(payload: dict) -> LandingTargetVision:
    truth = payload["true"]
    msg = LandingTargetVision()
    msg.header.stamp.sec = int(payload.get("header_stamp_sec", 0))
    msg.header.stamp.nanosec = int(payload.get("header_stamp_nanosec", 0))
    msg.header.frame_id = "ARpose_true"
    msg.target_type = LandingTargetVision.STATIONARY
    valid = bool(truth.get("valid", False))
    msg.rel_pos_valid = 1 if valid else 0
    msg.rel_vel_valid = 1 if valid else 0

    msg.pose.position.x = float(truth["position_m"][0])
    msg.pose.position.y = float(truth["position_m"][1])
    msg.pose.position.z = float(truth["position_m"][2])
    msg.pose.orientation = _quat_from_payload(truth["quat_xyzw"])
    msg.rel_vel.x = float(truth["velocity_mps"][0])
    msg.rel_vel.y = float(truth["velocity_mps"][1])
    msg.rel_vel.z = float(truth["velocity_mps"][2])
    msg.euler_angle.x = float(truth["euler_deg"][0])
    msg.euler_angle.y = float(truth["euler_deg"][1])
    msg.euler_angle.z = float(truth["euler_deg"][2])
    msg.angle_norm = 2.0 * math.sqrt(
        float(truth["quat_xyzw"][0]) ** 2
        + float(truth["quat_xyzw"][1]) ** 2
        + float(truth["quat_xyzw"][2]) ** 2
    ) * 180.0 / math.pi
    return msg


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish LandingTargetVision topics from newline-delimited JSON.")
    parser.add_argument("--raw-topic", required=True)
    parser.add_argument("--filtered-topic", required=True)
    parser.add_argument("--true-topic", default="")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("uav_rl_vision_pose_publisher")
    raw_pub = node.create_publisher(LandingTargetVision, args.raw_topic, 10)
    filtered_pub = node.create_publisher(LandingTargetVision, args.filtered_topic, 10)
    true_pub = node.create_publisher(LandingTargetVision, args.true_topic, 10) if args.true_topic else None

    try:
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=0.0)
            except ExternalShutdownException:
                break
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == "":
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "raw" in payload:
                raw_pub.publish(_raw_msg_from_payload(payload))
            if "filtered" in payload:
                filtered_pub.publish(_filtered_msg_from_payload(payload))
            if true_pub is not None and "true" in payload:
                true_pub.publish(_true_msg_from_payload(payload))
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
