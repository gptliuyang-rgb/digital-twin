"""L1a kinematic planner: upsample VLA waypoints to a 10 Hz SONIC reference.

SONIC's kinematic planner runs at 10 Hz and produces a 0.8–2.4 s reference
that the 50 Hz tracker follows. Positions use cubic Hermite (C1); rotations
are SLERP on SO(3) after rot6d → matrix (Zhou 6D is not a vector space);
finger joints are linear. Same code in sim and on the robot — no simulator
imports.

Hands still bypass the WBC token; they are upsampled here only so the
DexHand2 ring and the body reference share a clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interface.schema import CommandVector, command_dim, command_layout
from vla.adapters.rotation import matrix_to_rot6d, rot6d_to_matrix, slerp_matrices
from wbc.dims import load_t800_sonic
from wbc.teleop import FivePointCommand

HORIZON_S_RANGE = (0.8, 2.4)
DEFAULT_RATE_HZ = 10
DEFAULT_HORIZON_S = 1.6


def clamp_horizon_s(horizon_s: float) -> float:
    lo, hi = HORIZON_S_RANGE
    h = float(horizon_s)
    if not np.isfinite(h):
        raise ValueError("horizon_s must be finite")
    return float(np.clip(h, lo, hi))


def cubic_hermite(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """C1 cubic Hermite with finite-difference tangents. ``y_src`` is (N, D)."""
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    if y_src.ndim == 1:
        y_src = y_src.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    n = t_src.shape[0]
    if n < 2:
        raise ValueError("need at least two waypoints")
    if n != y_src.shape[0]:
        raise ValueError("t_src / y_src length mismatch")
    m = np.zeros_like(y_src)
    m[0] = (y_src[1] - y_src[0]) / max(t_src[1] - t_src[0], 1e-12)
    m[-1] = (y_src[-1] - y_src[-2]) / max(t_src[-1] - t_src[-2], 1e-12)
    for i in range(1, n - 1):
        m[i] = (y_src[i + 1] - y_src[i - 1]) / max(t_src[i + 1] - t_src[i - 1], 1e-12)
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    t0 = t_src[idx]
    t1 = t_src[idx + 1]
    dt = np.maximum(t1 - t0, 1e-12)
    u = np.clip((t_dst - t0) / dt, 0.0, 1.0)
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    y0 = y_src[idx]
    y1 = y_src[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    dt_col = dt.reshape(-1, 1)
    out = (
        h00.reshape(-1, 1) * y0
        + h10.reshape(-1, 1) * dt_col * m0
        + h01.reshape(-1, 1) * y1
        + h11.reshape(-1, 1) * dt_col * m1
    )
    return out[:, 0] if squeeze else out


def slerp_rot6d_series(t_src: np.ndarray, rot6d: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Piecewise SLERP of Zhou 6D rotations. ``rot6d`` is (N, 6)."""
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    r = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    t_dst = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    n = t_src.shape[0]
    if n != r.shape[0]:
        raise ValueError("t_src / rot6d length mismatch")
    mats = [rot6d_to_matrix(row) for row in r]
    out = np.empty((t_dst.shape[0], 6), dtype=np.float64)
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    for i, (k, t) in enumerate(zip(idx, t_dst, strict=True)):
        dt = max(t_src[k + 1] - t_src[k], 1e-12)
        alpha = float(np.clip((t - t_src[k]) / dt, 0.0, 1.0))
        out[i] = matrix_to_rot6d(slerp_matrices(mats[k], mats[k + 1], alpha))
    return out


ENUM_FIELDS = ("loco_mode", "left_hand_mode", "right_hand_mode", "tool_trigger")


def interpolate_command_matrix(
    flats: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
    spec,
    *,
    pos_interp: str = "cubic_hermite",
) -> np.ndarray:
    """Interpolate (N, D) command_schema_v1 rows from ``t_src`` onto ``t_dst``.

    Positions (and other non-rotation scalars) use cubic Hermite or linear.
    Zhou 6D is SLERP on SO(3). Enums are nearest-neighbour holds.
    """
    flats = np.asarray(flats, dtype=np.float64)
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    t_dst = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    if flats.ndim != 2:
        raise ValueError(f"flats must be (N, D), got {flats.shape}")
    dim = command_dim(spec)
    if flats.shape[1] != dim:
        raise ValueError(f"command last-dim {flats.shape[1]} != command_schema_v1 {dim}")
    if flats.shape[0] != t_src.shape[0]:
        raise ValueError("t_src / flats length mismatch")
    if flats.shape[0] < 2:
        raise ValueError("need at least two waypoints")
    layout = command_layout(spec)
    n_steps = t_dst.shape[0]
    out = np.empty((n_steps, dim), dtype=np.float64)
    rot_slices = [layout["head_rot6d"], layout["left_wrist_rot6d"], layout["right_wrist_rot6d"]]
    cursor = 0
    while cursor < dim:
        rot_here = next((s for s in rot_slices if s[0] == cursor), None)
        if rot_here is not None:
            a, b = rot_here
            out[:, a:b] = slerp_rot6d_series(t_src, flats[:, a:b], t_dst)
            cursor = b
            continue
        nxt = min((s[0] for s in rot_slices if s[0] > cursor), default=dim)
        block = flats[:, cursor:nxt]
        if pos_interp == "linear":
            for d in range(block.shape[1]):
                out[:, cursor + d] = np.interp(t_dst, t_src, block[:, d])
        else:
            out[:, cursor:nxt] = cubic_hermite(t_src, block, t_dst)
        cursor = nxt
    n = flats.shape[0]
    for name in ENUM_FIELDS:
        a, b = layout[name]
        nearest = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, n - 1)
        out[:, a:b] = flats[nearest, a:b]
    return out


@dataclass(frozen=True)
class PlannedRef:
    t_s: np.ndarray
    commands: np.ndarray  # (T, command_dim)
    elbows: np.ndarray | None  # (T, 6) or None
    rate_hz: int
    horizon_s: float

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


class KinematicPlanner:
    """10 Hz reference window. Simulator-free."""

    def __init__(
        self,
        *,
        rate_hz: int | None = None,
        horizon_s: float | None = None,
        pos_interp: str = "cubic_hermite",
    ) -> None:
        cfg = load_t800_sonic().get("planner", {})
        self.rate_hz = int(rate_hz if rate_hz is not None else cfg.get("rate_hz", DEFAULT_RATE_HZ))
        self.horizon_s = clamp_horizon_s(
            horizon_s if horizon_s is not None else cfg.get("default_horizon_s", DEFAULT_HORIZON_S)
        )
        self.pos_interp = pos_interp
        if self.rate_hz != 10:
            # SONIC paper L1a is 10 Hz. Other rates are allowed for tests but labelled.
            pass

    def plan(
        self,
        waypoints: list[CommandVector],
        times_s: np.ndarray | None = None,
        elbows: list[FivePointCommand] | None = None,
    ) -> PlannedRef:
        if len(waypoints) < 2:
            raise ValueError("planner needs ≥2 waypoints")
        spec = waypoints[0].spec
        n = len(waypoints)
        if times_s is None:
            t_src = np.linspace(0.0, self.horizon_s, n)
        else:
            t_src = np.asarray(times_s, dtype=np.float64).reshape(-1)
            if t_src.shape[0] != n:
                raise ValueError("times_s length must match waypoints")
        n_steps = int(round(self.horizon_s * self.rate_hz)) + 1
        t_dst = np.linspace(float(t_src[0]), float(t_src[0]) + self.horizon_s, n_steps)
        flats = np.stack([wp.to_flat_vector() for wp in waypoints], axis=0)
        out = interpolate_command_matrix(
            flats, t_src, t_dst, spec, pos_interp=self.pos_interp
        )
        elbow_out = None
        if elbows is not None:
            if len(elbows) != n:
                raise ValueError("elbows length must match waypoints")
            e = np.stack([fp.extra_vector() for fp in elbows], axis=0)
            elbow_out = cubic_hermite(t_src, e, t_dst) if self.pos_interp != "linear" else np.stack(
                [np.interp(t_dst, t_src, e[:, d]) for d in range(6)], axis=1
            )
        return PlannedRef(
            t_s=t_dst,
            commands=out,
            elbows=elbow_out,
            rate_hz=self.rate_hz,
            horizon_s=self.horizon_s,
        )
