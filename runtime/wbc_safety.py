"""WBC output filter sitting between SONIC and PD. Same binary in sim and real.

Rejects wrist command jumps > 5 cm/step (SONIC paper-scale tracking is ~6 cm mean;
a 5 cm *step* is a glitch, not tracking lag). Holds last accepted command.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla.adapters.rotation import rot6d_to_matrix


@dataclass(frozen=True)
class WbcSafetyLimits:
    max_delta_pos_m: float = 0.05
    max_delta_rot_rad: float = 0.25
    max_nav_accel_m_s2: float = 2.0
    pelvis_height_m: tuple[float, float] = (0.30, 0.80)


@dataclass
class WbcFilterResult:
    accepted: bool
    reason: str
    head_pos: np.ndarray
    left_wrist_pos: np.ndarray
    right_wrist_pos: np.ndarray
    pelvis_height: float
    nav_cmd: np.ndarray


def _rot_geodesic_rad(a6: np.ndarray, b6: np.ndarray) -> float:
    ra = rot6d_to_matrix(a6)
    rb = rot6d_to_matrix(b6)
    r = ra.T @ rb
    c = np.clip((float(np.trace(r)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))


class WbcSafetyFilter:
    def __init__(self, limits: WbcSafetyLimits | None = None) -> None:
        self.limits = limits or WbcSafetyLimits()
        self._last: dict[str, np.ndarray | float] | None = None

    def filter(
        self,
        *,
        head_pos: np.ndarray,
        head_rot6d: np.ndarray,
        left_wrist_pos: np.ndarray,
        left_wrist_rot6d: np.ndarray,
        right_wrist_pos: np.ndarray,
        right_wrist_rot6d: np.ndarray,
        pelvis_height: float,
        nav_cmd: np.ndarray,
        dt_s: float,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> WbcFilterResult:
        head_pos = np.asarray(head_pos, dtype=np.float64).reshape(3)
        left_wrist_pos = np.asarray(left_wrist_pos, dtype=np.float64).reshape(3)
        right_wrist_pos = np.asarray(right_wrist_pos, dtype=np.float64).reshape(3)
        nav_cmd = np.asarray(nav_cmd, dtype=np.float64).reshape(3)
        lo_h, hi_h = self.limits.pelvis_height_m
        pelvis_height = float(np.clip(pelvis_height, lo_h, hi_h))

        packed = {
            "head_pos": head_pos,
            "left_wrist_pos": left_wrist_pos,
            "right_wrist_pos": right_wrist_pos,
            "pelvis_height": pelvis_height,
            "nav_cmd": nav_cmd,
        }
        if left_elbow_pos is not None:
            packed["left_elbow_pos"] = np.asarray(left_elbow_pos, dtype=np.float64).reshape(3)
        if right_elbow_pos is not None:
            packed["right_elbow_pos"] = np.asarray(right_elbow_pos, dtype=np.float64).reshape(3)
        reason = "ok"
        if not np.isfinite(head_pos).all() or not np.isfinite(left_wrist_pos).all() or not np.isfinite(
            right_wrist_pos
        ).all():
            reason = "nan"
        elif left_elbow_pos is not None and not np.isfinite(packed["left_elbow_pos"]).all():
            reason = "nan"
        elif right_elbow_pos is not None and not np.isfinite(packed["right_elbow_pos"]).all():
            reason = "nan"
        elif self._last is not None:
            for key in ("head_pos", "left_wrist_pos", "right_wrist_pos", "left_elbow_pos", "right_elbow_pos"):
                if key not in packed or key not in self._last:
                    continue
                delta = float(np.linalg.norm(packed[key] - self._last[key]))  # type: ignore[operator]
                if delta > self.limits.max_delta_pos_m:
                    reason = f"jump_{key}"
                    break
            if reason == "ok":
                if _rot_geodesic_rad(np.asarray(left_wrist_rot6d), self._last["left_wrist_rot6d"]) > self.limits.max_delta_rot_rad:  # type: ignore[arg-type]
                    reason = "jump_left_wrist_rot"
                elif _rot_geodesic_rad(np.asarray(right_wrist_rot6d), self._last["right_wrist_rot6d"]) > self.limits.max_delta_rot_rad:  # type: ignore[arg-type]
                    reason = "jump_right_wrist_rot"
            if reason == "ok" and dt_s > 0:
                acc = np.linalg.norm((nav_cmd - self._last["nav_cmd"]) / dt_s)  # type: ignore[operator]
                if acc > self.limits.max_nav_accel_m_s2:
                    reason = "nav_accel"
        if reason != "ok" and self._last is not None:
            held = self._last
            return WbcFilterResult(
                False,
                reason,
                np.asarray(held["head_pos"]),
                np.asarray(held["left_wrist_pos"]),
                np.asarray(held["right_wrist_pos"]),
                float(held["pelvis_height"]),
                np.asarray(held["nav_cmd"]),
            )
        self._last = {
            **packed,
            "left_wrist_rot6d": np.asarray(left_wrist_rot6d, dtype=np.float64).reshape(6).copy(),
            "right_wrist_rot6d": np.asarray(right_wrist_rot6d, dtype=np.float64).reshape(6).copy(),
        }
        return WbcFilterResult(
            True, reason, head_pos, left_wrist_pos, right_wrist_pos, pelvis_height, nav_cmd
        )

    def reset(self) -> None:
        self._last = None
