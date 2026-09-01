"""Shared safety filter. No simulator imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyFilterResult:
    accepted: bool
    q_out: np.ndarray
    reason: str


class SafetyFilter:
    def __init__(
        self,
        q_min: np.ndarray,
        q_max: np.ndarray,
        max_delta_q: float,
        max_abs_qdot: float,
    ) -> None:
        self.q_min = np.asarray(q_min, dtype=np.float64)
        self.q_max = np.asarray(q_max, dtype=np.float64)
        self.max_delta_q = float(max_delta_q)
        self.max_abs_qdot = float(max_abs_qdot)
        self._last: np.ndarray | None = None

    def filter(self, q: np.ndarray, dt_s: float) -> SafetyFilterResult:
        q = np.asarray(q, dtype=np.float64)
        if not np.isfinite(q).all():
            held = self._last if self._last is not None else np.zeros_like(q)
            return SafetyFilterResult(False, held.copy(), "nan")
        clipped = np.clip(q, self.q_min, self.q_max)
        if self._last is not None:
            delta = clipped - self._last
            if np.max(np.abs(delta)) > self.max_delta_q:
                return SafetyFilterResult(False, self._last.copy(), "jump")
            if dt_s > 0 and np.max(np.abs(delta / dt_s)) > self.max_abs_qdot:
                return SafetyFilterResult(False, self._last.copy(), "velocity")
        self._last = clipped.copy()
        return SafetyFilterResult(True, clipped, "ok")

    def reset(self) -> None:
        self._last = None


@dataclass
class CartesianJumpFilter:
    """Reject SE(3) translation commands that jump more than ``max_delta_m`` per step.

    Used on VLA/WBC wrist position slices before they reach the PD plant.
    Rotation is not filtered here — callers run SO(3) ensemble separately.
    """

    max_delta_m: float = 0.05

    def __post_init__(self) -> None:
        self._last: np.ndarray | None = None

    def filter(self, pos_m: np.ndarray) -> SafetyFilterResult:
        p = np.asarray(pos_m, dtype=np.float64).reshape(3)
        if not np.isfinite(p).all():
            held = self._last if self._last is not None else np.zeros(3)
            return SafetyFilterResult(False, held.copy(), "nan")
        if self._last is not None and float(np.linalg.norm(p - self._last)) > self.max_delta_m:
            return SafetyFilterResult(False, self._last.copy(), "wrist_jump")
        self._last = p.copy()
        return SafetyFilterResult(True, p, "ok")

    def reset(self) -> None:
        self._last = None
