"""Pegasus actuator adapter: disarming PX4 does not erase rotor inertia."""
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .landing_diagnostics import LandingDiagnostics
    from .marker_reference import platform_marker_state
    from .thrust_cutoff import RotorCutoff
except ImportError:
    from landing_diagnostics import LandingDiagnostics
    from marker_reference import platform_marker_state
    from thrust_cutoff import RotorCutoff


class CutoffThrustCurve:
    def __init__(self, curve, platform, bridge, cfg):
        self.curve = curve
        self.platform = platform
        self.bridge = bridge
        self.cutoff = RotorCutoff(cfg)
        self.live_reference = np.zeros(curve._num_rotors)
        self._gate_elapsed_s = 0.0
        bridge._rotor_cutoff = self.cutoff
        self.diagnostics = LandingDiagnostics(bridge._vehicle_id, cfg.vehicle_z0_m)
        bridge._landing_diagnostics = self.diagnostics

    def __getattr__(self, name):
        return getattr(self.curve, name)

    def set_input_reference(self, reference):
        self.live_reference = reference

    def update(self, state, dt):
        cfg = self.cutoff.cfg
        marker = platform_marker_state(self.platform.world.stage, self.platform)
        self.diagnostics.record(
            state, marker, dt,
            flight_ready=self.bridge._armed and self.bridge._takeoff_state == "ready",
        )
        if cfg.enabled and not self.cutoff.active:
            self._gate_elapsed_s += float(dt)
            eligible = False
            if self._gate_elapsed_s >= 0.05:
                self._gate_elapsed_s %= 0.05
                relative = Rotation.from_quat(marker.quat_xyzw).inv().apply(state.position - marker.position)
                clearance = relative[2] - cfg.vehicle_z0_m
                eligible = (self.bridge._armed and self.bridge._takeoff_state == "ready"
                            and 0 <= clearance <= cfg.clearance_m
                            and np.all(np.abs(relative[:2]) <= cfg.xy_tolerance_m))
            if eligible or self.bridge._cutoff_request:
                self.cutoff.trigger(self.curve._velocity)
                self.bridge._cutoff_latched = True
                self.bridge._takeoff_state = "cutoff"
                self.diagnostics.cutoff(cfg.delay_s, cfg.thrust_tau_s)

        self.curve.set_input_reference(self.cutoff.output(self.live_reference))
        if self.cutoff.off:
            self.diagnostics.thrust_off()
        output = self.curve.update(state, dt)
        if self.cutoff.active and self.cutoff.elapsed_s >= cfg.delay_s:
            # Physical residual thrust is simulated independently of the PX4 armed bit.
            self.bridge._disarm_requested = True
        self.cutoff.advance(dt)
        return output
