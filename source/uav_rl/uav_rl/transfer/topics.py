from __future__ import annotations


def vehicle_namespace(namespace: str, vehicle_id: int) -> str:
    return f"{namespace}{vehicle_id}"


def cmd_vel_topic(namespace: str, vehicle_id: int) -> str:
    return f"{vehicle_namespace(namespace, vehicle_id)}/cmd_vel"


def pose_topic(namespace: str, vehicle_id: int) -> str:
    return f"{vehicle_namespace(namespace, vehicle_id)}/state/pose"


def twist_topic(namespace: str, vehicle_id: int) -> str:
    return f"{vehicle_namespace(namespace, vehicle_id)}/state/twist"


def twist_inertial_topic(namespace: str, vehicle_id: int) -> str:
    return f"{vehicle_namespace(namespace, vehicle_id)}/state/twist_inertial"


def platform_pose_topic(namespace: str, vehicle_id: int) -> str:
    return f"{vehicle_namespace(namespace, vehicle_id)}/platform/state/pose"


def platform_twist_topic(namespace: str, vehicle_id: int) -> str:
    return f"{vehicle_namespace(namespace, vehicle_id)}/platform/state/twist"
