"""SONIC §3.3 root spring. No simulator imports."""

from __future__ import annotations

import numpy as np
import pytest

from wbc.planner import KinematicPlanner
from wbc.spring import (
    HEADING_DAMPING_C,
    LN2,
    MAX_NAV_SPEED_MPS,
    POSITION_DAMPING_C,
    NavCommandError,
    RootSpringState,
    clamp_nav_speed_mps,
    critically_damped,
    critically_damped_heading,
    load_spring_cfg,
    paper_eq8_as_typeset,
    spring_root_keyframe,
    wrap_pi,
)


def _rest_at_origin() -> RootSpringState:
    return RootSpringState(
        pos_xy_m=np.zeros(2),
        heading_rad=0.0,
        vel_xy_mps=np.zeros(2),
        yaw_rate_rad_s=0.0,
    )


def test_yaml_locks_paper_coefficients() -> None:
    cfg = load_spring_cfg()
    assert cfg["not_dexhand2_contact"] is True
    assert cfg["not_table_s4"] is True
    assert POSITION_DAMPING_C == pytest.approx(5.0 * LN2)
    assert HEADING_DAMPING_C == pytest.approx(20.0 * LN2)
    assert HEADING_DAMPING_C == pytest.approx(4.0 * POSITION_DAMPING_C)
    assert MAX_NAV_SPEED_MPS == 6.0


def test_critically_damped_initial_conditions() -> None:
    x0, v0, x_t = 1.0, 0.25, 4.0
    assert critically_damped(x0, v0, x_t, 0.0, POSITION_DAMPING_C) == pytest.approx(x0)
    # Finite difference matches v0.
    dt = 1e-6
    x1 = critically_damped(x0, v0, x_t, dt, POSITION_DAMPING_C)
    assert (x1 - x0) / dt == pytest.approx(v0, abs=1e-4)


def test_paper_typeset_is_not_the_runtime() -> None:
    x0, v0, x_t = 1.0, 0.0, 4.0
    typeset0 = paper_eq8_as_typeset(x0, v0, x_t, 0.0, POSITION_DAMPING_C)
    assert typeset0 == pytest.approx(x_t - x0)
    assert typeset0 != pytest.approx(x0)
    assert critically_damped(x0, v0, x_t, 0.0, POSITION_DAMPING_C) == pytest.approx(x0)


def test_rest_at_target_stays() -> None:
    for t in (0.0, 0.5, 1.0, 4.0):
        assert critically_damped(2.0, 0.0, 2.0, t, POSITION_DAMPING_C) == pytest.approx(2.0)


def test_no_overshoot_from_rest() -> None:
    times = np.linspace(0.0, 6.0, 61)
    xs = [critically_damped(0.0, 0.0, 1.0, t, POSITION_DAMPING_C) for t in times]
    assert min(xs) >= -1e-12
    assert max(xs) <= 1.0 + 1e-12
    assert xs[-1] == pytest.approx(1.0, abs=1e-3)


def test_heading_wraps_shortest_arc() -> None:
    theta0 = 3.0
    theta_t = 3.0 + 0.4  # crosses π (≈3.1416)
    out = critically_damped_heading(theta0, 0.0, theta_t, 1.0, HEADING_DAMPING_C)
    assert abs(wrap_pi(out - theta0)) < 0.4 + 1e-9
    assert abs(out) <= np.pi + 1e-9


def test_speed_clamp_is_planar_not_per_axis() -> None:
    vx, vy = clamp_nav_speed_mps(6.0, 6.0)
    assert np.hypot(vx, vy) == pytest.approx(6.0)
    assert vx == pytest.approx(vy)
    vx0, vy0 = clamp_nav_speed_mps(3.0, 4.0)
    assert (vx0, vy0) == (3.0, 4.0)


def test_reverse_six_mps_is_smoothed() -> None:
    """Paper's 'abruptly reversing 6 m/s to −6 m/s' example."""
    state = RootSpringState(
        pos_xy_m=np.zeros(2),
        heading_rad=0.0,
        vel_xy_mps=np.array([6.0, 0.0]),
        yaw_rate_rad_s=0.0,
    )
    kf = spring_root_keyframe(state, np.array([-6.0, 0.0, 0.0]), t_s=1.0)
    ballistic_target = -6.0
    ballistic_current = 6.0
    x = float(kf.pos_xy_m[0])
    assert ballistic_target < x < ballistic_current
    assert x > ballistic_target + 1.0  # spring must not dump the full −6 m in 1 s


def test_nav_cmd_rejects_bad_shape_and_nan() -> None:
    state = _rest_at_origin()
    with pytest.raises(NavCommandError):
        spring_root_keyframe(state, np.zeros(4))
    with pytest.raises(NavCommandError):
        spring_root_keyframe(state, np.array([0.0, np.nan, 0.0]))


def test_planner_nav_root_rate() -> None:
    planner = KinematicPlanner(horizon_s=1.6)
    ref = planner.plan_nav_root(_rest_at_origin(), np.array([1.0, 0.0, 0.0]))
    assert ref.rate_hz == 10
    assert ref.n_steps == int(round(1.6 * 10)) + 1
    np.testing.assert_allclose(ref.pos_xy_m[0], [0.0, 0.0], atol=1e-12)
    assert ref.keyframe.t_s == 1.0
    assert ref.pos_xy_m[-1, 0] > 0.0
    assert abs(ref.heading_rad[-1]) < 1e-12
