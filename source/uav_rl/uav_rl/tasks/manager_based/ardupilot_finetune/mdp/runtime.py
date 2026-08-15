from __future__ import annotations

import atexit
import importlib.util
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from isaaclab.utils import configclass

q_ENU_to_NED = np.array([0.70711, 0.70711, 0.0, 0.0])
rot_ENU_to_NED = Rotation.from_quat(q_ENU_to_NED)
q_FLU_to_FRD = np.array([1.0, 0.0, 0.0, 0.0])
rot_FLU_to_FRD = Rotation.from_quat(q_FLU_to_FRD)
GRAVITY_VECTOR = np.array([0.0, 0.0, -9.80665])
PEGASUS_EXTENSION_ROOT = Path(
    os.environ.get("PEGASUS_EXT_PATH", Path.home() / "src/PegasusSimulator/extensions/pegasus.simulator")
).expanduser()

FAST_START_PARAM_TEXT = "\n".join(
    [
        "AHRS_EKF_TYPE 10",
        "EK2_ENABLE 0",
        "EK3_ENABLE 0",
        "",
    ]
)


def _load_ardupilot_plugin_class():
    plugin_path = PEGASUS_EXTENSION_ROOT / "pegasus/simulator/logic/backends/tools/ArduPilotPlugin.py"
    spec = importlib.util.spec_from_file_location("uav_rl_ardupilot_plugin", plugin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ArduPilotPlugin from {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArduPilotPlugin


ArduPilotPlugin = _load_ardupilot_plugin_class()


def _load_ardupilot_launch_tool_class():
    launch_tool_path = PEGASUS_EXTENSION_ROOT / "pegasus/simulator/logic/backends/tools/ardupilot_launch_tool.py"
    spec = importlib.util.spec_from_file_location("uav_rl_ardupilot_launch_tool", launch_tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ArduPilotLaunchTool from {launch_tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArduPilotLaunchTool


ArduPilotLaunchTool = _load_ardupilot_launch_tool_class()


class State:
    """Minimal vehicle state in ENU/FLU convention matching Pegasus expectations."""

    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.attitude = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # xyzw
        self.linear_body_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.linear_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.linear_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)


class SimpleImuSensor:
    """Local IMU model copied to avoid importing Pegasus packages in train.py."""

    def __init__(self, update_rate_hz: float = 250.0):
        self.update_rate_hz = float(update_rate_hz)
        self._prev_linear_velocity = np.zeros(3, dtype=np.float64)
        self._initialized = False

    def reset(self):
        self._prev_linear_velocity[:] = 0.0
        self._initialized = False

    def update(self, state: State, dt: float) -> dict[str, np.ndarray]:
        if self._initialized:
            linear_acceleration_inertial = (state.linear_velocity - self._prev_linear_velocity) / max(float(dt), 1.0e-6)
        else:
            linear_acceleration_inertial = np.zeros(3, dtype=np.float64)
            self._initialized = True
        linear_acceleration_inertial = linear_acceleration_inertial - GRAVITY_VECTOR
        self._prev_linear_velocity = state.linear_velocity.copy()

        linear_acceleration_body_flu = Rotation.from_quat(state.attitude).inv().apply(linear_acceleration_inertial)
        angular_velocity_frd = rot_FLU_to_FRD.apply(state.angular_velocity)
        linear_acceleration_frd = rot_FLU_to_FRD.apply(linear_acceleration_body_flu)
        attitude_frd_ned = rot_ENU_to_NED * Rotation.from_quat(state.attitude) * rot_FLU_to_FRD

        return {
            "orientation": attitude_frd_ned.as_quat(),
            "angular_velocity": angular_velocity_frd,
            "linear_acceleration": linear_acceleration_frd,
        }


class SensorMsg:
    def __init__(self):
        self.xacc = 0.0
        self.yacc = 0.0
        self.zacc = 0.0
        self.xgyro = 0.0
        self.ygyro = 0.0
        self.zgyro = 0.0
        self.sim_position = [0.0, 0.0, 0.0]
        self.sim_attitude = [1.0, 0.0, 0.0, 0.0]
        self.sim_velocity_inertial = [0.0, 0.0, 0.0]


class ThrusterControl:
    def __init__(
        self,
        num_rotors: int = 4,
        input_offset=(0.0, 0.0, 0.0, 0.0),
        input_scaling=(1000.0, 1000.0, 1000.0, 1000.0),
        input_min: int = 1000,
        input_max: int = 2000,
        zero_position_armed=(100.0, 100.0, 100.0, 100.0),
    ):
        self.num_rotors = int(num_rotors)
        self.input_offset = list(input_offset)
        self.input_scaling = list(input_scaling)
        self.input_min = int(input_min)
        self.input_max = int(input_max)
        self.input_range_inv = 1.0 / max(self.input_max - self.input_min, 1)
        self.zero_position_armed = list(zero_position_armed)
        self._input_reference = [0.0 for _ in range(self.num_rotors)]

    @property
    def input_reference(self):
        return self._input_reference

    def update_input_reference(self, pwms):
        servos = pwms[: self.num_rotors]
        if len(servos) < self.num_rotors:
            return
        for i in range(self.num_rotors):
            pwm = servos[i]
            raw_cmd = (pwm - self.input_min) * self.input_range_inv
            raw_cmd = float(np.clip(raw_cmd, 0.0, 1.0))
            self._input_reference[i] = ((raw_cmd + self.input_offset[i]) * self.input_scaling[i]) + self.zero_position_armed[i]

    def zero_input_reference(self):
        self._input_reference = [0.0 for _ in range(self.num_rotors)]


class LocalArduPilotBackend:
    """Minimal ArduPilot HIL backend used by the manager-based fine-tune task."""

    def __init__(
        self,
        vehicle_id: int,
        connection_baseport: int = 14550,
        num_rotors: int = 4,
        input_offset=(0.0, 0.0, 0.0, 0.0),
        input_scaling=(1000.0, 1000.0, 1000.0, 1000.0),
        input_min: int = 1000,
        input_max: int = 2000,
        zero_position_armed=(100.0, 100.0, 100.0, 100.0),
        enable_lockstep: bool = True,
        connection_timeout_s: float = 5.0,
    ):
        self.vehicle_id = int(vehicle_id)
        self.connection_baseport = int(connection_baseport)
        self.connection_ip = "127.0.0.1"
        self.connection_type = "udpin"
        self._connection_port = f"{self.connection_type}:{self.connection_ip}:{self.connection_baseport + self.vehicle_id * 10}"
        self._sensor_data = SensorMsg()
        self._rotor_data = ThrusterControl(
            num_rotors=num_rotors,
            input_offset=input_offset,
            input_scaling=input_scaling,
            input_min=input_min,
            input_max=input_max,
            zero_position_armed=zero_position_armed,
        )
        self._armed = False
        self._connection = None
        self.ap = None
        self._current_utime = 0.0
        self._enable_lockstep = bool(enable_lockstep)
        self._connection_timeout_s = max(float(connection_timeout_s), 0.1)

    def start(self):
        self._sensor_data = SensorMsg()
        self.ap = ArduPilotPlugin(fdm_port_in=9002 + self.vehicle_id * 10)
        self.ap.isLockStep = self._enable_lockstep
        self.ap.connectionTimeoutMaxCount = max(int(round(self._connection_timeout_s / 0.01)), 1)
        self.ap.drain_unread_packets()
        self._connection = mavutil.mavlink_connection(self._connection_port)
        self._current_utime = 0.0
        self._armed = False
        self._rotor_data.zero_input_reference()

    def stop(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self.ap = None

    def reset(self):
        self._current_utime = 0.0
        self._armed = False
        self._rotor_data.zero_input_reference()

    def update_sensor(self, sensor_type: str, data):
        if sensor_type != "IMU":
            return
        self._sensor_data.xacc = float(data["linear_acceleration"][0])
        self._sensor_data.yacc = float(data["linear_acceleration"][1])
        self._sensor_data.zacc = float(data["linear_acceleration"][2])
        self._sensor_data.xgyro = float(data["angular_velocity"][0])
        self._sensor_data.ygyro = float(data["angular_velocity"][1])
        self._sensor_data.zgyro = float(data["angular_velocity"][2])

    def update_state(self, state: State):
        position_ned = rot_ENU_to_NED.apply(state.position)
        attitude_frd_ned = rot_ENU_to_NED * Rotation.from_quat(state.attitude) * rot_FLU_to_FRD
        linear_velocity_ned = rot_ENU_to_NED.apply(state.linear_velocity)

        self._sensor_data.sim_position[0] = float(position_ned[0])
        self._sensor_data.sim_position[1] = float(position_ned[1])
        self._sensor_data.sim_position[2] = float(position_ned[2])
        quat = attitude_frd_ned.as_quat()
        self._sensor_data.sim_attitude[0] = float(quat[3])
        self._sensor_data.sim_attitude[1] = float(quat[0])
        self._sensor_data.sim_attitude[2] = float(quat[1])
        self._sensor_data.sim_attitude[3] = float(quat[2])
        self._sensor_data.sim_velocity_inertial[0] = float(linear_velocity_ned[0])
        self._sensor_data.sim_velocity_inertial[1] = float(linear_velocity_ned[1])
        self._sensor_data.sim_velocity_inertial[2] = float(linear_velocity_ned[2])

    def _update_is_armed(self):
        if self._connection is None:
            return
        msg = self._connection.recv_match(blocking=False)
        if msg is not None and msg.get_type() == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            self._armed = bool(self._connection.motors_armed())

    def update(self, dt: float):
        if self._connection is None or self.ap is None:
            self.start()

        self._current_utime += float(dt)
        _, servos = self.ap.pre_update(sim_time=self._current_utime)
        self._update_is_armed()
        if self._armed and servos != ():
            self._rotor_data.update_input_reference(servos)
        else:
            self._rotor_data.zero_input_reference()
        self.ap.post_update(sim_time=self._current_utime, sensor_data=self._sensor_data)

    def input_reference(self):
        return self._rotor_data.input_reference


def _terminate_process_tree(pid: int, timeout: float = 5.0):
    if pid <= 0:
        return

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


class MultiOutArduPilotLaunchTool(ArduPilotLaunchTool):
    """Launch ArduPilot SITL with one MAVLink out for HIL and one for guided control."""

    def __init__(
        self,
        ardupilot_dir: str,
        vehicle_id: int,
        ardupilot_model: str,
        out_ports: list[int],
        fast_start: bool = True,
        show_ui: bool = False,
    ):
        super().__init__(ardupilot_dir, int(vehicle_id), ardupilot_model)
        self.out_ports = list(out_ports)
        self.fast_start = fast_start
        self.show_ui = show_ui
        self._fast_start_param_file: str | None = None
        self._mavproxy_process: subprocess.Popen | None = None

    def _default_param_files(self) -> list[str]:
        default_dir = os.path.join(self.ardupilot_dir, "Tools", "autotest", "default_params")
        param_files = [
            os.path.join(default_dir, "copter.parm"),
            os.path.join(default_dir, f"{self.ardupilot_model}.parm"),
        ]
        if self.fast_start:
            param_files.append(self._ensure_fast_start_param_file())
        return param_files

    def _ensure_fast_start_param_file(self) -> str:
        if self._fast_start_param_file is not None:
            return self._fast_start_param_file

        param_path = os.path.join(self.root_fs.name, f"pegasus_fast_start_{self.vehicle_id}.parm")
        with open(param_path, "w", encoding="ascii") as param_file:
            param_file.write(FAST_START_PARAM_TEXT)
        self._fast_start_param_file = param_path
        return self._fast_start_param_file

    def _master_port(self) -> int:
        return 5760 + self.vehicle_id * 10

    def _sitl_port(self) -> int:
        return 5501 + self.vehicle_id * 10

    def _mavproxy_out_ports(self) -> list[str]:
        ports = [f"127.0.0.1:{14550 + self.vehicle_id * 10}"]
        ports.extend([f"udp:127.0.0.1:{port}" for port in self.out_ports])
        return ports

    def _launch_headless_detached(self):
        defaults = ",".join(self._default_param_files())
        arducopter_cmd = [
            os.path.join(self.ardupilot_dir, "build", "sitl", "bin", "arducopter"),
            "--model",
            self.model,
            "--speedup",
            "1",
            "--sysid",
            str(self.vehicle_id + 1),
            "--slave",
            "0",
            "--defaults",
            defaults,
            "--sim-address=127.0.0.1",
            f"-I{self.vehicle_id}",
        ]

        self.ardupilot_process = subprocess.Popen(
            arducopter_cmd,
            cwd=self.root_fs.name,
            shell=False,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # Give ArduPilot a brief moment to open its TCP master port before starting MAVProxy.
        time.sleep(2.0)

        mavproxy_cmd = [
            "mavproxy.py",
            "--retries",
            "5",
            "--master",
            f"tcp:127.0.0.1:{self._master_port()}",
            "--sitl",
            f"127.0.0.1:{self._sitl_port()}",
            "--non-interactive",
        ]
        for out_port in self._mavproxy_out_ports():
            mavproxy_cmd.extend(["--out", out_port])

        self._mavproxy_process = subprocess.Popen(
            mavproxy_cmd,
            cwd=self.root_fs.name,
            shell=False,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    def launch_ardupilot(self):
        if not self.show_ui:
            self._launch_headless_detached()
            return

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
        if self._mavproxy_process is not None:
            _terminate_process_tree(self._mavproxy_process.pid, timeout=5.0)
            self._mavproxy_process = None

        if self.ardupilot_process is None:
            return

        _terminate_process_tree(self.ardupilot_process.pid, timeout=5.0)
        self.ardupilot_process = None


@configclass
class ArduPilotLaunchCfg:
    ardupilot_dir: str = str(Path.home() / "projects/ardupilot")
    vehicle_model: str = "gazebo-iris"
    bridge_baseport: int = 14650
    fast_start: bool = True
    show_ui: bool = False
    launch_once_at_startup: bool = True
    startup_delay_s: float = 3.0


@configclass
class ArduPilotResetModeCfg:
    mode: str = "hard"
    allow_full_takeoff_option: bool = True
    auto_takeoff_alt_m: float = 5.0
    ready_timeout_s: float = 180.0
    takeoff_altitude_tolerance_m: float = 1.0
    hover_speed_tolerance_mps: float = 0.75
    hover_settle_time_s: float = 0.5
    soft_reset_settle_time_s: float = 0.5
    full_takeoff_every_n_resets: int = 0


@configclass
class ArduPilotBridgeCfg:
    command_rate_hz: float = 25.0
    cmd_timeout_s: float = 0.5
    arm_delay_s: float = 3.0
    require_position_ready: bool = False
    source_system_base: int = 200
    enable_hil_lockstep: bool = True
    hil_connection_timeout_s: float = 5.0


@configclass
class ArduPilotSensorCfg:
    imu_update_rate_hz: float = 250.0
    home_latitude_deg: float = 0.0
    home_longitude_deg: float = 0.0
    home_altitude_m: float = 0.0


@configclass
class ArduPilotFineTuneRuntimeCfg:
    enabled: bool = True
    num_sitl_envs: int = 8
    launch: ArduPilotLaunchCfg = ArduPilotLaunchCfg()
    reset: ArduPilotResetModeCfg = ArduPilotResetModeCfg()
    bridge: ArduPilotBridgeCfg = ArduPilotBridgeCfg()
    sensors: ArduPilotSensorCfg = ArduPilotSensorCfg()


class ArduPilotGuidedVelocityClient:
    """Minimal direct MAVLink client for ArduPilot GUIDED velocity control."""

    def __init__(
        self,
        vehicle_id: int,
        bridge_baseport: int,
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
        self.vehicle_id = vehicle_id
        self.bridge_port = bridge_baseport + vehicle_id * 10
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
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_wait_log_time = 0.0

        self._latest_cmd = np.zeros(4, dtype=np.float64)
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._current_altitude = 0.0
        self._current_linear_speed = 0.0
        self._reset_hold_until = 0.0
        self._takeoff_state = "pending" if self.auto_takeoff_alt > 0.0 else "ready"
        self._hover_ready_since: float | None = None

    def start(self):
        self._connection = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{self.bridge_port}",
            source_system=self.source_system_base + self.vehicle_id,
        )
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_wait_log_time = 0.0
        self._takeoff_state = "pending" if self.auto_takeoff_alt > 0.0 else "ready"
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._reset_hold_until = 0.0
        self._latest_cmd[:] = 0.0
        self._current_linear_speed = 0.0
        self._hover_ready_since = None

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
        self._current_linear_speed = float(np.linalg.norm(np.asarray(linear_velocity_enu, dtype=np.float64)))

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

    def _send_command_long(self, command: int, params: list[float]):
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
        if not self._connected:
            return
        if self._takeoff_state == "ready":
            return
        if self._first_heartbeat_time is None:
            return
        if now < self._reset_hold_until:
            return

        wait_reasons = []
        if (now - self._first_heartbeat_time) < self.arm_delay_s:
            wait_reasons.append(f"arm_delay {self.arm_delay_s:.1f}s")
        if not self._gps_fix_ready:
            wait_reasons.append("gps_fix")
        if self.require_position_ready and not self._position_estimate_ready:
            wait_reasons.append("position_estimate")

        if wait_reasons:
            return

        if self._takeoff_state == "pending":
            self._connection.set_mode_apm("GUIDED")
            self._takeoff_state = "arming"
            self._last_action_time = now
            return

        if self._takeoff_state == "arming":
            if self._connection.motors_armed():
                if self.auto_takeoff_alt > 0.0:
                    self._send_command_long(
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.auto_takeoff_alt],
                    )
                    self._takeoff_state = "taking_off"
                else:
                    self._takeoff_state = "ready"
                self._last_action_time = now
                return

            if now - self._last_action_time >= 0.5:
                self._connection.set_mode_apm("GUIDED")
                self._connection.arducopter_arm()
                self._last_action_time = now
            return

        if self._takeoff_state == "taking_off":
            altitude_ready = self._current_altitude >= max(self.auto_takeoff_alt - self.takeoff_altitude_tolerance_m, 0.0)
            speed_ready = self._current_linear_speed <= self.hover_speed_tolerance_mps
            if altitude_ready and speed_ready:
                if self._hover_ready_since is None:
                    self._hover_ready_since = now
                if now - self._hover_ready_since >= self.hover_settle_time_s:
                    self._takeoff_state = "ready"
                    self._hover_ready_since = None
                    return
            else:
                self._hover_ready_since = None
            if now - self._last_action_time >= 1.0:
                self._send_command_long(
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.auto_takeoff_alt],
                )
                self._last_action_time = now

    def _send_velocity_command(self, now: float):
        if not self._connected or self._takeoff_state != "ready":
            return
        if now < self._reset_hold_until:
            return
        if now - self._last_send_time < self.send_period:
            return

        cmd_age = now - self._last_cmd_time
        if self._last_cmd_time == 0.0 or cmd_age > self.cmd_timeout:
            vel_enu = np.zeros(3, dtype=np.float64)
            yaw_rate = 0.0
        else:
            vel_enu = self._latest_cmd[:3]
            yaw_rate = float(self._latest_cmd[3])

        vel_local_ned = rot_ENU_to_NED.apply(vel_enu)
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
            float(vel_local_ned[0]),
            float(vel_local_ned[1]),
            float(vel_local_ned[2]),
            0.0,
            0.0,
            0.0,
            0.0,
            -yaw_rate,
        )
        self._last_send_time = now

    def update(self, dt: float):
        del dt
        self._drain_mavlink()
        now = time.monotonic()
        self._update_takeoff_state(now)
        self._send_velocity_command(now)

    def reset(self, mode: str, settle_time_s: float):
        self._latest_cmd[:] = 0.0
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._reset_hold_until = time.monotonic() + max(float(settle_time_s), 0.0)
        self._hover_ready_since = None

        if mode == "soft":
            if self._connected and self._connection is not None and self._connection.motors_armed():
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
            "gps_fix_ready": self._gps_fix_ready,
            "position_estimate_ready": self._position_estimate_ready,
            "takeoff_state": self._takeoff_state,
            "altitude_m": self._current_altitude,
            "speed_mps": self._current_linear_speed,
        }


@dataclass
class ArduPilotEnvRuntime:
    env_id: int
    launch_tool: MultiOutArduPilotLaunchTool | None = None
    backend: LocalArduPilotBackend | None = None
    guided_client: ArduPilotGuidedVelocityClient | None = None
    imu: SimpleImuSensor | None = None
    reset_count: int = 0


class ArduPilotFineTuneRuntimeState:
    """Persistent SITL + backend state reused across environment resets."""

    def __init__(self, cfg: ArduPilotFineTuneRuntimeCfg):
        self.cfg = cfg
        self.envs = [ArduPilotEnvRuntime(env_id=i) for i in range(int(cfg.num_sitl_envs))]
        self.started = False
        self._atexit_registered = False

    def start(self):
        if self.started:
            return

        for handle in self.envs:
            self._initialize_env_handle(handle)

        for handle in self.envs:
            handle.launch_tool.launch_ardupilot()

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

    def should_train(self) -> bool:
        return self.cfg.enabled

    def num_ready_envs(self) -> int:
        count = 0
        for handle in self.envs:
            if handle.guided_client is not None and handle.guided_client.ready_for_velocity_commands:
                count += 1
        return count

    def all_envs_ready_for_policy(self) -> bool:
        return len(self.envs) > 0 and self.num_ready_envs() == len(self.envs)

    def current_altitudes(self) -> list[float]:
        altitudes: list[float] = []
        for handle in self.envs:
            if handle.guided_client is None:
                altitudes.append(0.0)
            else:
                altitudes.append(float(handle.guided_client.current_altitude))
        return altitudes

    def debug_statuses(self) -> list[dict[str, object]]:
        statuses: list[dict[str, object]] = []
        for handle in self.envs:
            if handle.guided_client is None:
                statuses.append({"connected": False, "gps_fix_ready": False, "position_estimate_ready": False, "takeoff_state": "missing"})
            else:
                statuses.append(dict(handle.guided_client.status))
        return statuses

    def _initialize_env_handle(self, handle: ArduPilotEnvRuntime):
        handle.backend = LocalArduPilotBackend(
            vehicle_id=handle.env_id,
            enable_lockstep=self.cfg.bridge.enable_hil_lockstep,
            connection_timeout_s=self.cfg.bridge.hil_connection_timeout_s,
        )
        handle.guided_client = ArduPilotGuidedVelocityClient(
            vehicle_id=handle.env_id,
            bridge_baseport=self.cfg.launch.bridge_baseport,
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
        handle.imu = SimpleImuSensor(update_rate_hz=self.cfg.sensors.imu_update_rate_hz)
        handle.launch_tool = MultiOutArduPilotLaunchTool(
            ardupilot_dir=self.cfg.launch.ardupilot_dir,
            vehicle_id=handle.env_id,
            ardupilot_model=self.cfg.launch.vehicle_model,
            out_ports=[self.cfg.launch.bridge_baseport + handle.env_id * 10],
            fast_start=self.cfg.launch.fast_start,
            show_ui=self.cfg.launch.show_ui,
        )
        handle.backend.start()
        handle.guided_client.start()

    def _stop_env_handle(self, handle: ArduPilotEnvRuntime):
        if handle.guided_client is not None:
            handle.guided_client.stop()
        if handle.backend is not None:
            handle.backend.stop()
        if handle.launch_tool is not None:
            handle.launch_tool.kill_ardupilot()

    def _restart_env_handle(self, handle: ArduPilotEnvRuntime):
        print(f"[INFO]: ArduPilot hard reset for env {handle.env_id}. Relaunching SITL from ground state.")
        self._stop_env_handle(handle)
        time.sleep(0.5)
        handle.imu.reset()
        handle.backend.start()
        handle.guided_client.start()
        handle.launch_tool.launch_ardupilot()
        startup_delay_s = max(float(self.cfg.launch.startup_delay_s), 0.0)
        if startup_delay_s > 0.0:
            time.sleep(startup_delay_s)


__all__ = [
    "ArduPilotFineTuneRuntimeCfg",
    "ArduPilotFineTuneRuntimeState",
    "ArduPilotEnvRuntime",
    "MultiOutArduPilotLaunchTool",
    "ArduPilotGuidedVelocityClient",
    "State",
]
