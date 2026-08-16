"""Stacking stability and release-quality metrics. No grasp-success claims."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla.adapters.rotation import quaternion_wxyz_to_matrix

EPS_P_M = 0.02
EPS_THETA_RAD = float(np.deg2rad(3.0))
SETTLE_T_S = 5.0
RELEASE_V_MAX_M_S = 0.05
RELEASE_GAP_MAX_M = 0.005


@dataclass(frozen=True)
class StackGate:
    support_ok: bool
    settle_pos_ok: bool
    settle_rot_ok: bool
    max_drift_m: float
    max_tilt_rad: float

    @property
    def stable(self) -> bool:
        return self.support_ok and self.settle_pos_ok and self.settle_rot_ok


def _angle_from_identity(quat_wxyz: np.ndarray) -> float:
    r = quaternion_wxyz_to_matrix(quat_wxyz)
    # Angle of rotation vs identity: θ = arccos((tr(R)−1)/2)
    tr = float(np.trace(r))
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))


def point_in_support(xy_m: np.ndarray, support_xy: np.ndarray) -> bool:
    """Axis-aligned support polygon given as [[xmin, ymin], [xmax, ymax]]."""
    lo, hi = np.asarray(support_xy, dtype=np.float64).reshape(2, 2)
    x, y = np.asarray(xy_m, dtype=np.float64).reshape(2)
    return bool(lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1])


def stack_stable(
    com_xy_traj_m: np.ndarray,
    quat_wxyz_traj: np.ndarray,
    support_xy: np.ndarray,
    *,
    dt_s: float,
    eps_p_m: float = EPS_P_M,
    eps_theta_rad: float = EPS_THETA_RAD,
    settle_t_s: float = SETTLE_T_S,
) -> StackGate:
    """stable ⇔ c_xy in support ∧ max|Δp| < ε_p ∧ max|Δθ| < ε_θ over settle window."""
    xy = np.asarray(com_xy_traj_m, dtype=np.float64).reshape(-1, 2)
    quats = np.asarray(quat_wxyz_traj, dtype=np.float64).reshape(-1, 4)
    n = min(len(xy), len(quats))
    n_win = max(1, int(round(settle_t_s / dt_s)))
    sl = slice(0, min(n, n_win))
    origin = xy[0]
    drift = np.linalg.norm(xy[sl] - origin, axis=1)
    max_drift = float(drift.max()) if len(drift) else 0.0
    tilts = [_angle_from_identity(q) for q in quats[sl]]
    max_tilt = float(max(tilts)) if tilts else 0.0
    support_ok = all(point_in_support(p, support_xy) for p in xy[sl])
    return StackGate(
        support_ok=support_ok,
        settle_pos_ok=max_drift < eps_p_m,
        settle_rot_ok=max_tilt < eps_theta_rad,
        max_drift_m=max_drift,
        max_tilt_rad=max_tilt,
    )


@dataclass(frozen=True)
class ReleaseQuality:
    velocity_m_s: float
    height_gap_m: float
    velocity_ok: bool
    height_ok: bool


def release_quality(velocity_m_s: float, height_gap_m: float) -> ReleaseQuality:
    return ReleaseQuality(
        velocity_m_s=float(velocity_m_s),
        height_gap_m=float(height_gap_m),
        velocity_ok=float(velocity_m_s) < RELEASE_V_MAX_M_S,
        height_ok=float(height_gap_m) < RELEASE_GAP_MAX_M,
    )
