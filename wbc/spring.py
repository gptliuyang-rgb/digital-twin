"""SONIC §3.3 critically damped root spring. Numpy only — no simulator.

He et al., arXiv:2511.07820v3 Equation 8, applied to pelvis *x*, pelvis *y*,
and projected heading. Filters ``nav_cmd`` (vx, vy, wz) before L1a.

Hands still bypass WBC. This module does not write DexHand2 contact
parameters and does not weld T800+Hand (ADR-009 / ADR-038).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SPRING_YAML = Path(__file__).with_name("spring.yaml")
LN2 = log(2.0)
# Paper: c_pos = 5 ln 2, c_heading = 20 ln 2.
POSITION_DAMPING_C = 5.0 * LN2
HEADING_DAMPING_C = 20.0 * LN2
TARGET_HORIZON_S = 1.0
MAX_NAV_SPEED_MPS = 6.0


class NavCommandError(ValueError):
    """Raised when nav_cmd is the wrong shape, non-finite, or not SI."""


def load_spring_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or SPRING_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("spring.yaml must be a mapping")
    if raw.get("not_dexhand2_contact") is not True:
        raise ValueError("spring.yaml must keep not_dexhand2_contact: true")
    if raw.get("not_table_s4") is not True:
        raise ValueError("spring.yaml must keep not_table_s4: true")
    if int(raw["position_damping_ln2_multiplier"]) != 5:
        raise ValueError("position damping must stay 5 ln(2) (SONIC §3.3)")
    if int(raw["heading_damping_ln2_multiplier"]) != 20:
        raise ValueError("heading damping must stay 20 ln(2) (SONIC §3.3)")
    if float(raw["target_horizon_s"]) != TARGET_HORIZON_S:
        raise ValueError("target_horizon_s must stay 1.0 s (SONIC §3.3)")
    if float(raw["max_nav_speed_mps"]) != MAX_NAV_SPEED_MPS:
        raise ValueError("max_nav_speed_mps must stay 6.0 (SONIC §3.3)")
    return raw


def paper_eq8_as_typeset(
    x0: float,
    v0: float,
    x_t: float,
    t_s: float,
    damping_c: float,
) -> float:
    """Equation 8 *as printed* — displacement form, x(0)=x_T-x_0, not x_0.

    Kept so tests can lock the paper typesetting. Runtime uses
    :func:`critically_damped` (ADR-038).
    """
    t = float(t_s)
    c = float(damping_c)
    delta = float(x_t) - float(x0)
    return (delta + (float(v0) + 0.5 * c * delta) * t) * np.exp(-0.5 * c * t)


def critically_damped(
    x0: float,
    v0: float,
    x_t: float,
    t_s: float,
    damping_c: float,
) -> float:
    """Critically damped step with x(0)=x0, ẋ(0)=v0, equilibrium x_T.

    Paper Eq. 8 typesets the homogeneous piece with (x_T-x_0) and omits the
    x_T offset, so x(0) would be x_T-x_0. The second-order critically damped
    solution is:

        ω = c / 2
        A = x0 - x_T
        B = v0 + ω A
        x(t) = x_T + (A + B t) exp(-ω t)

    That is Eq. 8 with (x0-x_T) in place of (x_T-x0) plus the x_T offset.
    ``damping_c`` is the paper's c (5 ln 2 or 20 ln 2), not ω.
    """
    t = float(t_s)
    if t < 0.0:
        raise ValueError("t_s must be ≥ 0")
    omega = 0.5 * float(damping_c)
    a = float(x0) - float(x_t)
    b = float(v0) + omega * a
    return float(x_t) + (a + b * t) * np.exp(-omega * t)


def wrap_pi(angle_rad: float) -> float:
    """Wrap to (−π, π]."""
    a = (float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi
    if a <= -np.pi:
        a += 2.0 * np.pi
    return float(a)


def critically_damped_heading(
    theta0_rad: float,
    omega0_rad_s: float,
    theta_t_rad: float,
    t_s: float,
    damping_c: float,
) -> float:
    """Heading spring on the shortest-arc error, then wrap."""
    theta0 = float(theta0_rad)
    err = wrap_pi(float(theta_t_rad) - theta0)
    theta_t = theta0 + err
    return wrap_pi(critically_damped(theta0, float(omega0_rad_s), theta_t, t_s, damping_c))


def clamp_nav_speed_mps(vx_mps: float, vy_mps: float, *, max_speed_mps: float = MAX_NAV_SPEED_MPS) -> tuple[float, float]:
    """Clamp planar speed to the paper 6.0 m/s bound. wz is not clamped."""
    vx, vy = float(vx_mps), float(vy_mps)
    speed = float(np.hypot(vx, vy))
    cap = float(max_speed_mps)
    if speed <= cap or speed == 0.0:
        return vx, vy
    scale = cap / speed
    return vx * scale, vy * scale


def parse_nav_cmd(nav_cmd: np.ndarray) -> tuple[float, float, float]:
    cmd = np.asarray(nav_cmd, dtype=np.float64).reshape(-1)
    if cmd.shape != (3,):
        raise NavCommandError(f"nav_cmd must be (3,) [vx, vy, wz], got {cmd.shape}")
    if not np.isfinite(cmd).all():
        raise NavCommandError("nav_cmd must be finite")
    vx, vy = clamp_nav_speed_mps(float(cmd[0]), float(cmd[1]))
    return vx, vy, float(cmd[2])


@dataclass
class RootSpringState:
    """Pelvis xy + heading and their rates. Heading-frame or world; caller picks.

    ``pos_xy_m`` is metres. ``vel_xy_mps`` is m/s. Heading is radians.
    Not a DexHand2 wrist pose.
    """

    pos_xy_m: np.ndarray
    heading_rad: float
    vel_xy_mps: np.ndarray
    yaw_rate_rad_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos_xy_m", np.asarray(self.pos_xy_m, dtype=np.float64).reshape(2))
        object.__setattr__(self, "vel_xy_mps", np.asarray(self.vel_xy_mps, dtype=np.float64).reshape(2))
        if not np.isfinite(self.pos_xy_m).all() or not np.isfinite(self.vel_xy_mps).all():
            raise NavCommandError("root spring state must be finite")
        if not np.isfinite(self.heading_rad) or not np.isfinite(self.yaw_rate_rad_s):
            raise NavCommandError("heading / yaw rate must be finite")


@dataclass(frozen=True)
class RootKeyframe:
    pos_xy_m: np.ndarray
    heading_rad: float
    t_s: float
    source: str = "sonic_eq8"


@dataclass(frozen=True)
class RootSpringRef:
    t_s: np.ndarray
    pos_xy_m: np.ndarray  # (T, 2)
    heading_rad: np.ndarray  # (T,)
    rate_hz: int
    horizon_s: float
    keyframe: RootKeyframe

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


def velocity_target(
    state: RootSpringState,
    nav_cmd: np.ndarray,
    *,
    horizon_s: float = TARGET_HORIZON_S,
) -> tuple[np.ndarray, float]:
    """Paper: x_T = x_0 + v_des * 1.0 s (after planar-speed clamp)."""
    vx, vy, wz = parse_nav_cmd(nav_cmd)
    h = float(horizon_s)
    pos_t = np.asarray(state.pos_xy_m, dtype=np.float64) + np.array([vx, vy], dtype=np.float64) * h
    heading_t = float(state.heading_rad) + wz * h
    return pos_t, heading_t


def spring_root_keyframe(
    state: RootSpringState,
    nav_cmd: np.ndarray,
    *,
    t_s: float = TARGET_HORIZON_S,
    cfg: dict[str, Any] | None = None,
) -> RootKeyframe:
    """Place the planner keyframe at x(t) from Eq. 8 (IC-correct form)."""
    cfg = cfg or load_spring_cfg()
    c_pos = float(cfg["position_damping_ln2_multiplier"]) * LN2
    c_head = float(cfg["heading_damping_ln2_multiplier"]) * LN2
    pos_t, heading_t = velocity_target(state, nav_cmd, horizon_s=TARGET_HORIZON_S)
    x = critically_damped(float(state.pos_xy_m[0]), float(state.vel_xy_mps[0]), float(pos_t[0]), t_s, c_pos)
    y = critically_damped(float(state.pos_xy_m[1]), float(state.vel_xy_mps[1]), float(pos_t[1]), t_s, c_pos)
    heading = critically_damped_heading(
        float(state.heading_rad), float(state.yaw_rate_rad_s), heading_t, t_s, c_head
    )
    return RootKeyframe(
        pos_xy_m=np.array([x, y], dtype=np.float64),
        heading_rad=heading,
        t_s=float(t_s),
    )


def spring_root_trajectory(
    state: RootSpringState,
    nav_cmd: np.ndarray,
    t_s: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the spring at each time. ``t_s`` is seconds, shape (T,)."""
    times = np.asarray(t_s, dtype=np.float64).reshape(-1)
    xy = np.empty((times.shape[0], 2), dtype=np.float64)
    heading = np.empty(times.shape[0], dtype=np.float64)
    for i, t in enumerate(times):
        kf = spring_root_keyframe(state, nav_cmd, t_s=float(t), cfg=cfg)
        xy[i] = kf.pos_xy_m
        heading[i] = kf.heading_rad
    return xy, heading
