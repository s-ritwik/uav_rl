#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from precision_landing_using_vision_msgs.msg import LandingTargetVision
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


@dataclass
class PoseSnapshot:
    valid: bool = False
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


class VisionOverlayViewer:
    def __init__(self, image_topic: str, raw_pose_topic: str, filtered_pose_topic: str, display_scale: float, window_name: str):
        self.display_scale = float(display_scale)
        self.window_name = window_name
        self.latest_frame_bgr: np.ndarray | None = None
        self.raw_pose = PoseSnapshot()
        self.filtered_pose = PoseSnapshot()

        rclpy.init()
        self.node = rclpy.create_node("uav_rl_vision_overlay_viewer")
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.node.create_subscription(Image, image_topic, self._image_cb, qos_best_effort)
        self.node.create_subscription(LandingTargetVision, raw_pose_topic, self._raw_pose_cb, qos_best_effort)
        self.node.create_subscription(LandingTargetVision, filtered_pose_topic, self._filtered_pose_cb, qos_best_effort)

    def _pose_from_msg(self, msg: LandingTargetVision) -> PoseSnapshot:
        return PoseSnapshot(
            valid=bool(msg.rel_pos_valid),
            pos=(
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ),
            euler_deg=(
                float(msg.euler_angle.x),
                float(msg.euler_angle.y),
                float(msg.euler_angle.z),
            ),
        )

    def _raw_pose_cb(self, msg: LandingTargetVision):
        self.raw_pose = self._pose_from_msg(msg)

    def _filtered_pose_cb(self, msg: LandingTargetVision):
        self.filtered_pose = self._pose_from_msg(msg)

    def _image_cb(self, msg: Image):
        self.latest_frame_bgr = self._ros_image_to_bgr(msg)

    def _ros_image_to_bgr(self, msg: Image) -> np.ndarray | None:
        height = int(msg.height)
        width = int(msg.width)
        if height <= 0 or width <= 0:
            return None

        encoding = str(msg.encoding).lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == "mono8":
            rows = data.reshape(height, int(msg.step))
            image = rows[:, :width]
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
        }.get(encoding)
        if channels is None:
            return None

        rows = data.reshape(height, int(msg.step))
        compact = rows[:, : width * channels]
        image = compact.reshape(height, width, channels)

        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _format_pose_line(self, prefix: str, pose: PoseSnapshot) -> str:
        valid_text = "OK" if pose.valid else "MISS"
        return (
            f"{prefix} [{valid_text}] "
            f"x={pose.pos[0]:+.2f} y={pose.pos[1]:+.2f} z={pose.pos[2]:+.2f} "
            f"roll={pose.euler_deg[0]:+.1f} pitch={pose.euler_deg[1]:+.1f} yaw={pose.euler_deg[2]:+.1f}"
        )

    def _render(self):
        if self.latest_frame_bgr is None:
            return

        frame = self.latest_frame_bgr.copy()
        overlay_lines = [
            self._format_pose_line("RAW ", self.raw_pose),
            self._format_pose_line("MEKF", self.filtered_pose),
        ]
        colors = [
            (0, 220, 0) if self.raw_pose.valid else (0, 0, 255),
            (255, 200, 0) if self.filtered_pose.valid else (0, 0, 255),
        ]

        y = 28
        for line, color in zip(overlay_lines, colors):
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
            y += 28

        if self.display_scale != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            while rclpy.ok():
                try:
                    rclpy.spin_once(self.node, timeout_sec=0.01)
                except ExternalShutdownException:
                    break
                except Exception:
                    if not rclpy.ok():
                        break
                    raise
                self._render()
        finally:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="OpenCV overlay viewer for onboard vision topics.")
    parser.add_argument("--image-topic", type=str, required=True)
    parser.add_argument("--raw-topic", type=str, required=True)
    parser.add_argument("--filtered-topic", type=str, required=True)
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--window-name", type=str, default="Isaac Vision Feed")
    args = parser.parse_args()

    viewer = VisionOverlayViewer(
        image_topic=args.image_topic,
        raw_pose_topic=args.raw_topic,
        filtered_pose_topic=args.filtered_topic,
        display_scale=args.display_scale,
        window_name=args.window_name,
    )
    viewer.run()


if __name__ == "__main__":
    main()
