"""One-shot flight event logs from copied, pre-physics velocity samples."""
import numpy as np
from scipy.spatial.transform import Rotation


class LandingDiagnostics:
    def __init__(self, vehicle_id, gear_offset_m):
        self.vehicle_id = vehicle_id
        self.gear_offset_m = gear_offset_m
        self.reset()

    def reset(self):
        self.sample = None
        self.previous_sample = None
        self.impact_sample = None
        self.flight_ready = False
        self.cutoff_time = None
        self.elapsed_s = 0.0
        self.touchdown_logged = False
        self.other_contact_logged = False
        self.off_logged = False

    def record(self, state, marker, dt, flight_ready):
        relative = Rotation.from_quat(marker.quat_xyzw).inv().apply(state.position - marker.position)
        # Copy these values: Pegasus mutates its state after each physics step.
        uav_velocity = np.asarray(state.linear_velocity, dtype=float).copy()
        marker_velocity = np.asarray(marker.linear_velocity, dtype=float).copy()
        self.previous_sample = self.sample
        self.sample = {
            "time": self.elapsed_s, "dt": float(dt),
            "clearance": float(relative[2] - self.gear_offset_m),
            "uav_velocity": uav_velocity, "marker_velocity": marker_velocity,
            "relative_velocity": uav_velocity - marker_velocity,
        }
        self.flight_ready |= bool(flight_ready and self.sample["clearance"] > 0.0)
        self.elapsed_s += float(dt)

    def _print(self, event, details="", pre_contact=False):
        sample = self.impact_sample if pre_contact else self.sample
        if sample is None:
            return
        velocity = sample["relative_velocity"]
        since_cutoff = "none" if self.cutoff_time is None else f"{self.sample['time'] - self.cutoff_time:.3f}s"
        timing = f" pre_contact_sample_dt={sample['dt']:.4f}s" if pre_contact else ""
        print(
            f"[{event}] drone{self.vehicle_id}: "
            f"uav_vz={sample['uav_velocity'][2]:+.3f} m/s "
            f"platform_vz={sample['marker_velocity'][2]:+.3f} m/s "
            f"relative_vz={velocity[2]:+.3f} m/s "
            f"closing_speed={max(-velocity[2], 0.0):.3f} m/s "
            f"relative_xy_speed={np.linalg.norm(velocity[:2]):.3f} m/s "
            f"gear_clearance={sample['clearance']:.3f} m "
            f"since_cutoff={since_cutoff}{timing}{details}",
            flush=True,
        )

    def cutoff(self, delay_s, thrust_tau_s):
        self.cutoff_time = self.sample["time"]
        self.flight_ready = True
        self._print("cutoff", f" delay={delay_s:.3f}s thrust_tau={thrust_tau_s:.3f}s")

    def thrust_off(self):
        if not self.off_logged:
            self._print("thrust-off")
            self.off_logged = True

    def contact(self, surface, on_platform):
        if not self.flight_ready or self.sample is None:
            return
        if on_platform:
            if self.touchdown_logged:
                return
            self.touchdown_logged = True
        else:
            if self.other_contact_logged:
                return
            self.other_contact_logged = True
        # PhysX reports arrive after the state callback can expose collision-braked
        # velocity. Keep the preceding physics sample, not that refreshed state.
        self.impact_sample = self.previous_sample or self.sample
        self._print("touchdown" if on_platform else "contact", f" surface={surface}", pre_contact=True)
