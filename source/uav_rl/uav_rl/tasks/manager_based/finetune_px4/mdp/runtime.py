from __future__ import annotations

import atexit
import importlib.util
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
import types

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation
import yaml

from isaaclab.utils import configclass

q_ENU_to_NED = np.array([0.70711, 0.70711, 0.0, 0.0])
rot_ENU_to_NED = Rotation.from_quat(q_ENU_to_NED)
q_FLU_to_FRD = np.array([1.0, 0.0, 0.0, 0.0])
rot_FLU_to_FRD = Rotation.from_quat(q_FLU_to_FRD)
PEGASUS_EXTENSION_ROOT = Path("/home/rycker/src/PegasusSimulator/extensions/pegasus.simulator")
PEGASUS_LOGIC_ROOT = PEGASUS_EXTENSION_ROOT / "pegasus" / "simulator" / "logic"
PEGASUS_CONFIG_PATH = PEGASUS_EXTENSION_ROOT / "config" / "configs.yaml"


def _read_pegasus_config() -> dict:
    if not PEGASUS_CONFIG_PATH.exists():
        return {}
    with open(PEGASUS_CONFIG_PATH, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _ensure_package(module_name: str, package_path: Path) -> ModuleType:
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    module = types.ModuleType(module_name)
    module.__path__ = [str(package_path)]
    sys.modules[module_name] = module
    return module


def _load_module(module_name: str, module_path: Path):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{module_name}' from '{module_path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_px4_backend_modules():
    _ensure_package("pegasus", PEGASUS_EXTENSION_ROOT / "pegasus")
    _ensure_package("pegasus.simulator", PEGASUS_EXTENSION_ROOT / "pegasus" / "simulator")
    _ensure_package("pegasus.simulator.logic", PEGASUS_LOGIC_ROOT)
    _ensure_package("pegasus.simulator.logic.backends", PEGASUS_LOGIC_ROOT / "backends")
    _ensure_package("pegasus.simulator.logic.backends.tools", PEGASUS_LOGIC_ROOT / "backends" / "tools")
    _ensure_package("pegasus.simulator.logic.interface", PEGASUS_LOGIC_ROOT / "interface")
    sensors_pkg = _ensure_package("pegasus.simulator.logic.sensors", PEGASUS_LOGIC_ROOT / "sensors")

    interface_module_name = "pegasus.simulator.logic.interface.pegasus_interface"
    if interface_module_name not in sys.modules:
        config = _read_pegasus_config()
        interface_module = types.ModuleType(interface_module_name)

        class PegasusInterface:  # type: ignore[no-redef]
            def __init__(self):
                self._px4_path = str(Path(config.get("px4_dir", "~/src/PX4-Autopilot")).expanduser())
                self._px4_default_airframe = str(config.get("px4_default_airframe", "gazebo-classic_iris"))
                global_coordinates = config.get("global_coordinates", {})
                self._latitude = float(global_coordinates.get("latitude", 38.736832))
                self._longitude = float(global_coordinates.get("longitude", -9.137977))
                self._altitude = float(global_coordinates.get("altitude", 90.0))

            @property
            def px4_path(self):
                return self._px4_path

            @property
            def px4_default_airframe(self):
                return self._px4_default_airframe

            @property
            def latitude(self):
                return self._latitude

            @property
            def longitude(self):
                return self._longitude

            @property
            def altitude(self):
                return self._altitude

        interface_module.PegasusInterface = PegasusInterface
        sys.modules[interface_module_name] = interface_module

    _load_module("pegasus.simulator.logic.rotations", PEGASUS_LOGIC_ROOT / "rotations.py")
    _load_module("pegasus.simulator.logic.state", PEGASUS_LOGIC_ROOT / "state.py")
    _load_module("pegasus.simulator.logic.sensors.geo_mag_utils", PEGASUS_LOGIC_ROOT / "sensors" / "geo_mag_utils.py")
    sensor_module = _load_module("pegasus.simulator.logic.sensors.sensor", PEGASUS_LOGIC_ROOT / "sensors" / "sensor.py")
    sensors_pkg.Sensor = sensor_module.Sensor
    barometer_module = _load_module("pegasus.simulator.logic.sensors.barometer", PEGASUS_LOGIC_ROOT / "sensors" / "barometer.py")
    sensors_pkg.Barometer = barometer_module.Barometer
    gps_module = _load_module("pegasus.simulator.logic.sensors.gps", PEGASUS_LOGIC_ROOT / "sensors" / "gps.py")
    sensors_pkg.GPS = gps_module.GPS
    imu_module = _load_module("pegasus.simulator.logic.sensors.imu", PEGASUS_LOGIC_ROOT / "sensors" / "imu.py")
    sensors_pkg.IMU = imu_module.IMU
    magnetometer_module = _load_module("pegasus.simulator.logic.sensors.magnetometer", PEGASUS_LOGIC_ROOT / "sensors" / "magnetometer.py")
    sensors_pkg.Magnetometer = magnetometer_module.Magnetometer
    _load_module("pegasus.simulator.logic.backends.backend", PEGASUS_LOGIC_ROOT / "backends" / "backend.py")
    _load_module(
        "pegasus.simulator.logic.backends.tools.px4_launch_tool",
        PEGASUS_LOGIC_ROOT / "backends" / "tools" / "px4_launch_tool.py",
    )


def _load_px4_backend_classes():
    _bootstrap_px4_backend_modules()
    module = _load_module(
        "pegasus.simulator.logic.backends.px4_mavlink_backend",
        PEGASUS_LOGIC_ROOT / "backends" / "px4_mavlink_backend.py",
    )
    PX4MavlinkBackend = module.PX4MavlinkBackend
    PX4MavlinkBackendConfig = module.PX4MavlinkBackendConfig

    return PX4MavlinkBackend, PX4MavlinkBackendConfig


def _load_pegasus_sensor_classes():
    _bootstrap_px4_backend_modules()
    module = sys.modules["pegasus.simulator.logic.sensors"]
    return module.IMU, module.GPS, module.Barometer, module.Magnetometer


def _load_pegasus_state_class():
    _bootstrap_px4_backend_modules()
    module = sys.modules["pegasus.simulator.logic.state"]
    return module.State


def _kill_stale_px4_instance(vehicle_id: int, px4_dir: str):
    """Remove orphan PX4 SITL processes for the requested instance before relaunch."""

    px4_binary = str((Path(px4_dir).expanduser() / "build" / "px4_sitl_default" / "bin" / "px4").resolve())
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=", "args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return

    matching_pids: list[int] = []
    instance_token = f" -i {int(vehicle_id)} "
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or px4_binary not in line or instance_token not in f" {line} ":
            continue
        try:
            pid_text, _ = line.split(" ", 1)
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            matching_pids.append(pid)

    for pid in matching_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception:
            continue

    if matching_pids:
        time.sleep(0.5)


class PX4GpsSensor:
    def __init__(self, origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float, update_rate_hz: float):
        self.origin_lat_deg = float(origin_lat_deg)
        self.origin_lon_deg = float(origin_lon_deg)
        self.origin_alt_m = float(origin_alt_m)
        self.update_period = 1.0 / max(float(update_rate_hz), 1.0)
        self._accumulated_time = 0.0

    def update(self, state: PX4State, dt: float) -> dict[str, float] | None:
        self._accumulated_time += float(dt)
        if self._accumulated_time < self.update_period:
            return None
        self._accumulated_time = 0.0

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(self.origin_lat_deg))
        latitude = self.origin_lat_deg + float(state.position[1]) / max(meters_per_deg_lat, 1.0)
        longitude = self.origin_lon_deg + float(state.position[0]) / max(meters_per_deg_lon, 1.0)
        speed_xy = float(np.linalg.norm(state.linear_velocity[:2]))

        return {
            "fix_type": 3,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": self.origin_alt_m + float(state.position[2]),
            "eph": 1.0,
            "epv": 1.0,
            "speed": speed_xy,
            "velocity_north": float(state.linear_velocity[1]),
            "velocity_east": float(state.linear_velocity[0]),
            "velocity_down": float(-state.linear_velocity[2]),
            "cog": 0.0,
            "sattelites_visible": 10,
            "latitude_gt": latitude,
            "longitude_gt": longitude,
            "altitude_gt": self.origin_alt_m + float(state.position[2]),
        }


class PX4BarometerSensor:
    def __init__(self, origin_alt_m: float, update_rate_hz: float):
        self.origin_alt_m = float(origin_alt_m)
        self.update_period = 1.0 / max(float(update_rate_hz), 1.0)
        self._accumulated_time = 0.0

    def update(self, state: PX4State, dt: float) -> dict[str, float] | None:
        self._accumulated_time += float(dt)
        if self._accumulated_time < self.update_period:
            return None
        self._accumulated_time = 0.0

        pressure_altitude = self.origin_alt_m + float(state.position[2])
        return {
            "absolute_pressure": 1013.25,
            "pressure_altitude": pressure_altitude,
            "temperature": 15.0,
        }


class PX4MagnetometerSensor:
    def __init__(self, update_rate_hz: float):
        self.update_period = 1.0 / max(float(update_rate_hz), 1.0)
        self._accumulated_time = 0.0
        self._field_world = np.array([0.21523, 0.0, 0.42741], dtype=np.float64)

    def update(self, state: PX4State, dt: float) -> dict[str, list[float]] | None:
        self._accumulated_time += float(dt)
        if self._accumulated_time < self.update_period:
            return None
        self._accumulated_time = 0.0

        attitude_flu_enu = Rotation.from_quat(state.attitude)
        rot_body_to_world = rot_ENU_to_NED * attitude_flu_enu * rot_FLU_to_FRD.inv()
        magnetic_field_body = rot_body_to_world.inv().apply(self._field_world)
        return {"magnetic_field": magnetic_field_body.tolist()}


class PX4State:
    """Minimal Pegasus-compatible state in ENU/FLU convention."""

    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.attitude = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # xyzw
        self.linear_body_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.linear_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.linear_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def get_position_ned(self):
        return rot_ENU_to_NED.apply(self.position)

    def get_attitude_ned_frd(self):
        return (rot_ENU_to_NED * Rotation.from_quat(self.attitude) * rot_FLU_to_FRD).as_quat()

    def get_linear_body_velocity_ned_frd(self):
        linear_acc_body_flu = Rotation.from_quat(self.attitude).inv().apply(self.linear_acceleration)
        return rot_FLU_to_FRD.apply(linear_acc_body_flu)

    def get_linear_velocity_ned(self):
        return rot_ENU_to_NED.apply(self.linear_velocity)

    def get_angular_velocity_frd(self):
        return rot_FLU_to_FRD.apply(self.angular_velocity)

    def get_linear_acceleration_ned(self):
        return rot_ENU_to_NED.apply(self.linear_acceleration)


@configclass
class PX4LaunchCfg:
    px4_dir: str = ""
    vehicle_model: str = ""
    backend_connection_baseport: int = 4560
    offboard_baseport: int = 14540
    autolaunch: bool = True
    startup_delay_s: float = 0.0


@configclass
class PX4ResetModeCfg:
    mode: str = "hard"
    allow_full_takeoff_option: bool = True
    auto_takeoff_alt_m: float = 5.0
    ready_timeout_s: float = 180.0
    takeoff_altitude_tolerance_m: float = 0.12
    hover_speed_tolerance_mps: float = 0.12
    hover_settle_time_s: float = 0.75
    soft_reset_settle_time_s: float = 0.5
    full_takeoff_every_n_resets: int = 0


@configclass
class PX4BridgeCfg:
    command_rate_hz: float = 25.0
    cmd_timeout_s: float = 0.5
    arm_delay_s: float = 3.0
    require_position_ready: bool = True
    source_system_base: int = 200


@configclass
class PX4SensorCfg:
    imu_update_rate_hz: float = 250.0
    gps_update_rate_hz: float = 250.0
    barometer_update_rate_hz: float = 250.0
    magnetometer_update_rate_hz: float = 250.0
    home_latitude_deg: float = 38.736832
    home_longitude_deg: float = -9.137977
    home_altitude_m: float = 90.0


@configclass
class PX4FineTuneRuntimeCfg:
    enabled: bool = True
    num_sitl_envs: int = 8
    launch: PX4LaunchCfg = PX4LaunchCfg()
    reset: PX4ResetModeCfg = PX4ResetModeCfg()
    bridge: PX4BridgeCfg = PX4BridgeCfg()
    sensors: PX4SensorCfg = PX4SensorCfg()


class PX4OffboardVelocityClient:
    """Direct MAVLink client for PX4 OFFBOARD local-NED velocity control."""

    def __init__(
        self,
        vehicle_id: int,
        offboard_baseport: int,
        auto_takeoff_alt: float,
        cmd_timeout: float,
        send_rate_hz: float,
        arm_delay_s: float,
        require_position_ready: bool,
        source_system_base: int,
        takeoff_altitude_tolerance_m: float,
        hover_speed_tolerance_mps: float,
        hover_settle_time_s: float,
    ):
        self.vehicle_id = int(vehicle_id)
        self.offboard_port = int(offboard_baseport) + self.vehicle_id
        self.auto_takeoff_alt = max(float(auto_takeoff_alt), 0.0)
        self.cmd_timeout = max(float(cmd_timeout), 0.0)
        self.send_period = 1.0 / max(float(send_rate_hz), 1.0)
        self.arm_delay_s = max(float(arm_delay_s), 0.0)
        self.require_position_ready = bool(require_position_ready)
        self.source_system_base = int(source_system_base)
        self.takeoff_altitude_tolerance_m = max(float(takeoff_altitude_tolerance_m), 0.0)
        self.hover_speed_tolerance_mps = max(float(hover_speed_tolerance_mps), 0.0)
        self.hover_settle_time_s = max(float(hover_settle_time_s), 0.0)

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
        self._takeoff_state = "pending" if self.auto_takeoff_alt > 0.0 else "ready"
        self._ready_since = None
        self._reset_hold_until = 0.0

        self._current_altitude = 0.0
        self._current_vertical_speed = 0.0
        self._latest_cmd = np.zeros(4, dtype=np.float64)
        self._last_cmd_time = 0.0

    def start(self):
        self._connection = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{self.offboard_port}",
            source_system=self.source_system_base + self.vehicle_id,
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
        self._takeoff_state = "pending" if self.auto_takeoff_alt > 0.0 else "ready"
        self._ready_since = None
        self._reset_hold_until = 0.0
        self._latest_cmd[:] = 0.0
        self._last_cmd_time = 0.0

    def stop(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def set_command(self, velocity_sp_enu: np.ndarray, yaw_rate_sp: float):
        self._latest_cmd[:3] = np.asarray(velocity_sp_enu, dtype=np.float64)
        self._latest_cmd[3] = float(yaw_rate_sp)
        self._last_cmd_time = time.monotonic()

    def update_kinematics(self, altitude_m: float, linear_velocity_enu: np.ndarray):
        self._current_altitude = float(altitude_m)
        self._current_vertical_speed = float(linear_velocity_enu[2])

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
                if command == mavutil.mavlink.MAV_CMD_DO_SET_MODE and result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self._offboard_enabled = True
                if command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self._armed = True

    def _wait_ready(self, now: float) -> bool:
        wait_reasons = []
        if not self._connected:
            wait_reasons.append("heartbeat")
        elif self._first_heartbeat_time is not None and (now - self._first_heartbeat_time) < self.arm_delay_s:
            wait_reasons.append(f"arm_delay {self.arm_delay_s:.1f}s")
        if self.require_position_ready and not self._position_estimate_ready:
            wait_reasons.append("position_estimate")

        if wait_reasons and (now - self._last_wait_log_time) >= 2.0:
            print(f"[INFO]: PX4 env {self.vehicle_id} waiting for {', '.join(wait_reasons)}")
            self._last_wait_log_time = now

        return not wait_reasons

    def _desired_velocity_enu(self, now: float) -> tuple[np.ndarray, float]:
        if self._takeoff_state == "ready":
            cmd_age = now - self._last_cmd_time
            if self._last_cmd_time == 0.0 or cmd_age > self.cmd_timeout:
                return np.zeros(3, dtype=np.float64), 0.0
            return self._latest_cmd[:3].copy(), -float(self._latest_cmd[3])

        if self.auto_takeoff_alt <= 0.0:
            return np.zeros(3, dtype=np.float64), 0.0

        alt_error = self.auto_takeoff_alt - self._current_altitude
        climb_speed = 0.0
        if alt_error > 0.15:
            climb_speed = min(1.0, max(0.25, 0.8 * alt_error))
        elif alt_error < -0.15:
            climb_speed = max(-0.6, min(-0.15, 0.8 * alt_error))
        return np.array([0.0, 0.0, climb_speed], dtype=np.float64), 0.0

    def _send_velocity_command(self, now: float):
        if not self._connected or self._target_system is None or self._target_component is None:
            return
        if now < self._reset_hold_until:
            return
        if now - self._last_send_time < self.send_period:
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
        if self._prestream_count < max(10, int(round(1.0 / self.send_period))):
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
            return

        if not self._offboard_enabled and now - self._last_mode_request_time >= 1.0:
            self._connection.set_mode("OFFBOARD")
            self._last_action_time = now
            self._last_mode_request_time = now
            return

        if self._offboard_enabled and self._armed and self._takeoff_state == "pending":
            self._takeoff_state = "taking_off"
            self._ready_since = None

    def _update_takeoff_state(self, now: float):
        if self._takeoff_state != "taking_off":
            return

        altitude_error = abs(self._current_altitude - self.auto_takeoff_alt)
        if altitude_error <= self.takeoff_altitude_tolerance_m and abs(self._current_vertical_speed) <= self.hover_speed_tolerance_mps:
            if self._ready_since is None:
                self._ready_since = now
            elif now - self._ready_since >= self.hover_settle_time_s:
                self._takeoff_state = "ready"
        else:
            self._ready_since = None

    def update(self, dt: float):
        del dt
        self._drain_mavlink()
        now = time.monotonic()
        self._send_velocity_command(now)
        self._request_offboard_and_arm(now)
        self._update_takeoff_state(now)

    def reset(self, mode: str, settle_time_s: float):
        self._latest_cmd[:] = 0.0
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._last_mode_request_time = 0.0
        self._last_arm_request_time = 0.0
        self._prestream_count = 0
        self._ready_since = None
        self._reset_hold_until = time.monotonic() + max(float(settle_time_s), 0.0)

        if mode == "soft":
            if self._connected and self._armed:
                self._takeoff_state = "ready"
            else:
                self._takeoff_state = "pending" if self.auto_takeoff_alt > 0.0 else "ready"
            return
        if mode == "full_takeoff":
            self._takeoff_state = "pending" if self.auto_takeoff_alt > 0.0 else "ready"
            return
        raise ValueError(f"Unsupported reset mode: {mode}")

    @property
    def ready_for_velocity_commands(self) -> bool:
        return self._takeoff_state == "ready"

    @property
    def current_altitude(self) -> float:
        return self._current_altitude

    @property
    def status(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "gps_fix_ready": self._position_estimate_ready,
            "position_estimate_ready": self._position_estimate_ready,
            "takeoff_state": self._takeoff_state,
            "altitude_m": self._current_altitude,
            "speed_mps": abs(self._current_vertical_speed),
        }


@dataclass
class PX4EnvRuntime:
    env_id: int
    backend: object | None = None
    guided_client: PX4OffboardVelocityClient | None = None
    imu: SimpleImuSensor | None = None
    gps: object | None = None
    barometer: object | None = None
    magnetometer: object | None = None
    reset_count: int = 0


class PX4FineTuneRuntimeState:
    """Persistent PX4 SITL + backend state reused across environment resets."""

    def __init__(self, cfg: PX4FineTuneRuntimeCfg):
        self.cfg = cfg
        self.envs = [PX4EnvRuntime(env_id=i) for i in range(int(cfg.num_sitl_envs))]
        self.started = False
        self._atexit_registered = False

    def start(self):
        if self.started:
            return

        for handle in self.envs:
            self._initialize_env_handle(handle)

        startup_delay_s = max(float(self.cfg.launch.startup_delay_s), 0.0)
        if startup_delay_s > 0.0:
            time.sleep(startup_delay_s)

        self.started = True
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

    def stop(self):
        for handle in self.envs:
            self._stop_env_handle(handle)
        self.started = False

    def reset_envs(self, env_ids: list[int] | np.ndarray | tuple[int, ...] | None):
        if env_ids is None:
            ids = range(len(self.envs))
        else:
            ids = [int(env_id) for env_id in env_ids]

        for env_id in ids:
            handle = self.envs[env_id]
            handle.reset_count += 1
            mode = self.cfg.reset.mode
            every_n = int(self.cfg.reset.full_takeoff_every_n_resets)
            if mode == "soft" and every_n > 0 and handle.reset_count % every_n == 0:
                mode = "full_takeoff"
            if mode == "hard":
                self._restart_env_handle(handle)
                continue
            handle.imu.reset()
            handle.backend.reset()
            handle.guided_client.reset(mode=mode, settle_time_s=self.cfg.reset.soft_reset_settle_time_s)

    def num_ready_envs(self) -> int:
        count = 0
        for handle in self.envs:
            if handle.guided_client is not None and handle.guided_client.ready_for_velocity_commands:
                count += 1
        return count

    def all_envs_ready_for_policy(self) -> bool:
        return len(self.envs) > 0 and self.num_ready_envs() == len(self.envs)

    def current_altitudes(self) -> list[float]:
        return [
            0.0 if handle.guided_client is None else float(handle.guided_client.current_altitude)
            for handle in self.envs
        ]

    def debug_statuses(self) -> list[dict[str, object]]:
        return [
            {"connected": False, "gps_fix_ready": False, "position_estimate_ready": False, "takeoff_state": "missing"}
            if handle.guided_client is None
            else dict(handle.guided_client.status)
            for handle in self.envs
        ]

    def _resolve_px4_launch_defaults(self) -> tuple[str, str]:
        config = _read_pegasus_config()
        px4_dir = self.cfg.launch.px4_dir or str(Path(config.get("px4_dir", "~/src/PX4-Autopilot")).expanduser())
        vehicle_model = self.cfg.launch.vehicle_model or str(config.get("px4_default_airframe", "gazebo-classic_iris"))
        return px4_dir, vehicle_model

    def _initialize_env_handle(self, handle: PX4EnvRuntime):
        PX4MavlinkBackend, PX4MavlinkBackendConfig = _load_px4_backend_classes()
        px4_dir, vehicle_model = self._resolve_px4_launch_defaults()
        _kill_stale_px4_instance(handle.env_id, px4_dir)

        handle.backend = PX4MavlinkBackend(
            PX4MavlinkBackendConfig(
                {
                    "vehicle_id": handle.env_id,
                    "connection_baseport": self.cfg.launch.backend_connection_baseport,
                    "px4_autolaunch": self.cfg.launch.autolaunch,
                    "px4_dir": px4_dir,
                    "px4_vehicle_model": vehicle_model,
                    "enable_lockstep": True,
                    "update_rate": self.cfg.sensors.imu_update_rate_hz,
                }
            )
        )
        handle.backend.start()
        handle.guided_client = PX4OffboardVelocityClient(
            vehicle_id=handle.env_id,
            offboard_baseport=self.cfg.launch.offboard_baseport,
            auto_takeoff_alt=self.cfg.reset.auto_takeoff_alt_m,
            cmd_timeout=self.cfg.bridge.cmd_timeout_s,
            send_rate_hz=self.cfg.bridge.command_rate_hz,
            arm_delay_s=self.cfg.bridge.arm_delay_s,
            require_position_ready=self.cfg.bridge.require_position_ready,
            source_system_base=self.cfg.bridge.source_system_base,
            takeoff_altitude_tolerance_m=self.cfg.reset.takeoff_altitude_tolerance_m,
            hover_speed_tolerance_mps=self.cfg.reset.hover_speed_tolerance_mps,
            hover_settle_time_s=self.cfg.reset.hover_settle_time_s,
        )
        handle.guided_client.start()
        IMU, GPS, Barometer, Magnetometer = _load_pegasus_sensor_classes()
        handle.imu = IMU({"update_rate": self.cfg.sensors.imu_update_rate_hz})
        handle.gps = GPS({"update_rate": self.cfg.sensors.gps_update_rate_hz})
        handle.barometer = Barometer({"update_rate": self.cfg.sensors.barometer_update_rate_hz})
        handle.magnetometer = Magnetometer({"update_rate": self.cfg.sensors.magnetometer_update_rate_hz})

        for sensor in (handle.imu, handle.gps, handle.barometer, handle.magnetometer):
            sensor.initialize(
                vehicle=None,
                origin_lat=self.cfg.sensors.home_latitude_deg,
                origin_lon=self.cfg.sensors.home_longitude_deg,
                origin_alt=self.cfg.sensors.home_altitude_m,
            )

    def _stop_env_handle(self, handle: PX4EnvRuntime):
        px4_dir, _ = self._resolve_px4_launch_defaults()
        if handle.guided_client is not None:
            handle.guided_client.stop()
        if handle.backend is not None:
            handle.backend.stop()
        _kill_stale_px4_instance(handle.env_id, px4_dir)

    def _restart_env_handle(self, handle: PX4EnvRuntime):
        print(f"[INFO]: PX4 hard reset for env {handle.env_id}. Relaunching SITL from ground state.")
        self._stop_env_handle(handle)
        time.sleep(0.5)
        self._initialize_env_handle(handle)
        startup_delay_s = max(float(self.cfg.launch.startup_delay_s), 0.0)
        if startup_delay_s > 0.0:
            time.sleep(startup_delay_s)


__all__ = [
    "PX4LaunchCfg",
    "PX4ResetModeCfg",
    "PX4BridgeCfg",
    "PX4SensorCfg",
    "PX4FineTuneRuntimeCfg",
    "PX4FineTuneRuntimeState",
    "PX4EnvRuntime",
    "PX4OffboardVelocityClient",
    "PX4State",
]


PX4State = _load_pegasus_state_class()
