"""DexHand2 controller. Simulation-package-free. Same object on robot and in sim."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from hand.backends.base import HandBackend, HandState
from hand.coupling import Coupling
from hand.grasp_primitives import GraspLibrary
from interface.schema import HandSpec


@dataclass(frozen=True)
class SafetyLimits:
    max_delta_q_rad: float
    velocity_limit_rad_s: float
    soft_limit_margin_rad: float = 0.0

    @classmethod
    def from_spec(cls, spec: HandSpec) -> SafetyLimits:
        raw = spec.raw.get("controller_safety", {})
        max_delta = raw.get("max_delta_q_rad")
        vel = raw.get("velocity_limit_rad_s")
        if max_delta == "REQUIRED_INPUT" or vel == "REQUIRED_INPUT" or max_delta is None or vel is None:
            raise ValueError(
                "controller_safety.max_delta_q_rad and velocity_limit_rad_s are REQUIRED_INPUT. "
                "Pass SafetyLimits explicitly for tests, or fill the spec."
            )
        return cls(max_delta_q_rad=float(max_delta), velocity_limit_rad_s=float(vel))


def mit_torque(
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    q_des_rad: np.ndarray,
    dq_des_rad_s: np.ndarray,
    tau_ff: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """MIT force-position hybrid: τ = kp (qd − q) + kd (dqd − dq) + τ_ff."""
    return kp * (q_des_rad - q_rad) + kd * (dq_des_rad_s - dq_rad_s) + tau_ff


class DexHand2Controller:
    def __init__(
        self,
        spec: HandSpec,
        backend: HandBackend,
        *,
        rate_hz: int | None = None,
        safety: SafetyLimits | None = None,
        coupling: Coupling | None = None,
        latency_ms: float | None = None,
        enable_latency: bool = False,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        self.spec = spec
        self.backend = backend
        self.rate_hz = int(rate_hz if rate_hz is not None else spec.raw["control_rate_hz"])
        self.safety = safety
        self.coupling = coupling or Coupling.from_spec(spec)
        self.library = GraspLibrary(spec)
        n = spec.n_active_dof
        self.kp = kp if kp is not None else _vector_from_map(spec.raw["sim_kp"], spec.joint_order)
        self.kd = kd if kd is not None else _vector_from_map(spec.raw["sim_kv"], spec.joint_order)
        self._q_cmd = np.zeros(n)
        self._dq_cmd = np.zeros(n)
        self._tau_ff = np.zeros(n)
        self._rejected = False
        lat = latency_ms
        if lat is None:
            raw_lat = spec.raw.get("command_latency_ms")
            lat = 0.0 if raw_lat == "REQUIRED_INPUT" or raw_lat is None else float(raw_lat)
        depth = max(1, int(round((lat / 1000.0) * self.rate_hz))) if enable_latency else 1
        self._delay: deque[np.ndarray] = deque(maxlen=depth)

    def set_joint_targets(self, q_active: np.ndarray, dq_active: np.ndarray | None = None) -> bool:
        q = np.asarray(q_active, dtype=np.float64).reshape(self.spec.n_active_dof)
        dq = (
            np.zeros_like(q)
            if dq_active is None
            else np.asarray(dq_active, dtype=np.float64).reshape(self.spec.n_active_dof)
        )
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            self._rejected = True
            return False
        q = self._soft_clamp_limits(q)
        if self.safety is not None:
            if np.max(np.abs(q - self._q_cmd)) > self.safety.max_delta_q_rad:
                self._rejected = True
                return False
            dq = np.clip(dq, -self.safety.velocity_limit_rad_s, self.safety.velocity_limit_rad_s)
        self._rejected = False
        self._q_cmd = q
        self._dq_cmd = dq
        self._emit()
        return True

    def set_grasp_primitive(self, name: str, closure: float) -> bool:
        q = self.library.q_active(name, float(np.clip(closure, 0.0, 1.0)))
        return self.set_joint_targets(q)

    def set_force_mode(self, target_force: float) -> bool:
        # τ_ff only; position target holds. Hardware current mapping is REQUIRED_INPUT.
        self._tau_ff[:] = float(target_force)
        self._emit()
        return True

    def get_state(self) -> HandState:
        return self.backend.read_state()

    @property
    def last_rejected(self) -> bool:
        return self._rejected

    def _soft_clamp_limits(self, q: np.ndarray) -> np.ndarray:
        limits = self.spec.limits_vector()
        margin = 0.0 if self.safety is None else self.safety.soft_limit_margin_rad
        lo = limits[:, 0] + margin
        hi = limits[:, 1] - margin
        return np.clip(q, lo, hi)

    def _emit(self) -> None:
        q = self.coupling.active_to_full(self._q_cmd)[: self.spec.n_active_dof]
        self._delay.append(q.copy())
        q_out = self._delay[0]
        self.backend.write_mit(q_out, self._dq_cmd, self._tau_ff, self.kp, self.kd)


def _vector_from_map(mapping: dict[str, float], order: list[str]) -> np.ndarray:
    return np.array([float(mapping[name]) for name in order], dtype=np.float64)
