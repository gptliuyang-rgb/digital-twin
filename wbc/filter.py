"""Filter GMR-retargeted clips before SONIC PPO. Numpy only — no simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Defaults are *gates*, not hardware ratings.
FOOT_PENETRATION_M = -0.005  # m; foot height below this is a penetration
MIN_DURATION_S = 0.4
MAX_ABS_DQ_RAD_S = 40.0  # numerical blow-up, not a motor spec


@dataclass
class ClipFilterResult:
    keep: bool
    reasons: list[str] = field(default_factory=list)
    n_limit_violations: int = 0
    n_foot_penetrations: int = 0
    max_abs_dq_rad_s: float = 0.0


def filter_motion_clip(
    q_rad: np.ndarray,
    *,
    joint_limits_rad: np.ndarray,
    foot_height_m: np.ndarray | None = None,
    dt_s: float = 0.05,
    limit_margin_rad: float = 0.0,
    foot_penetration_m: float = FOOT_PENETRATION_M,
    max_abs_dq_rad_s: float = MAX_ABS_DQ_RAD_S,
    min_duration_s: float = MIN_DURATION_S,
) -> ClipFilterResult:
    """Drop clips with NaN, joint-limit hits, foot penetration, or velocity blow-up.

    `joint_limits_rad` is (n_dof, 2) [lower, upper] from the T800 MJCF `range=`
    (official), never invented. `foot_height_m` is (T, n_feet) in metres.
    """
    q = np.asarray(q_rad, dtype=np.float64)
    limits = np.asarray(joint_limits_rad, dtype=np.float64).reshape(-1, 2)
    reasons: list[str] = []
    if q.ndim != 2:
        return ClipFilterResult(keep=False, reasons=["q_rad must be (T, n_dof)"])
    t, n = q.shape
    if limits.shape[0] != n:
        return ClipFilterResult(keep=False, reasons=[f"limits rows {limits.shape[0]} != n_dof {n}"])
    if not np.isfinite(q).all():
        reasons.append("nan_or_inf")
    duration = t * float(dt_s)
    if duration < min_duration_s:
        reasons.append(f"too_short_{duration:.3f}s")
    lo = limits[:, 0] - limit_margin_rad
    hi = limits[:, 1] + limit_margin_rad
    over = (q < lo) | (q > hi)
    n_lim = int(over.any(axis=1).sum())
    if n_lim:
        reasons.append(f"joint_limit_frames={n_lim}")
    n_pen = 0
    if foot_height_m is not None:
        h = np.asarray(foot_height_m, dtype=np.float64)
        if h.shape[0] != t:
            reasons.append("foot_height_time_mismatch")
        else:
            n_pen = int((h < foot_penetration_m).any(axis=1).sum())
            if n_pen:
                reasons.append(f"foot_penetration_frames={n_pen}")
    dq = np.diff(q, axis=0) / float(dt_s) if t > 1 else np.zeros((0, n))
    max_dq = float(np.max(np.abs(dq))) if dq.size else 0.0
    if max_dq > max_abs_dq_rad_s:
        reasons.append(f"dq_blowup_{max_dq:.1f}")
    return ClipFilterResult(
        keep=len(reasons) == 0,
        reasons=reasons,
        n_limit_violations=n_lim,
        n_foot_penetrations=n_pen,
        max_abs_dq_rad_s=max_dq,
    )


def parse_mjcf_joint_limits(mjcf_text: str, joint_order: list[str]) -> np.ndarray:
    """Read `range="lo hi"` for each joint in `joint_order` from official T800 MJCF."""
    import re

    limits = np.full((len(joint_order), 2), np.nan)
    for i, name in enumerate(joint_order):
        # `\brange=` — not `actuatorfrcrange=`. Greedy `[^>]*range=` latches onto
        # the force range (hundreds of N·m) and silently disables the limit gate.
        match = re.search(
            rf'<joint name="{re.escape(name)}"[^>]*\brange="([^"]+)"',
            mjcf_text,
        )
        if not match:
            raise KeyError(f"joint {name} has no range= in MJCF")
        lo, hi = (float(x) for x in match.group(1).split())
        limits[i] = [lo, hi]
    if not np.isfinite(limits).all():
        raise ValueError("incomplete MJCF joint limits")
    return limits
