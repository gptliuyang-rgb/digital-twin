"""L1a 10 Hz kinematic planner. No simulator imports."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import CommandVector, command_layout
from vla.adapters.rotation import matrix_to_rot6d, rot6d_to_matrix, rpy_to_matrix
from wbc.planner import HORIZON_S_RANGE, KinematicPlanner, clamp_horizon_s, cubic_hermite, slerp_rot6d_series
from wbc.teleop import FivePointCommand


def test_horizon_clamped() -> None:
    assert clamp_horizon_s(0.1) == HORIZON_S_RANGE[0]
    assert clamp_horizon_s(9.0) == HORIZON_S_RANGE[1]
    assert clamp_horizon_s(1.6) == 1.6


def test_cubic_hermite_reproduces_line() -> None:
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([[0.0], [1.0], [2.0]])
    q = np.array([0.5, 1.5])
    out = cubic_hermite(t, y, q)
    np.testing.assert_allclose(out.ravel(), [0.5, 1.5], atol=1e-9)


def test_slerp_identity_stays_identity() -> None:
    ident = matrix_to_rot6d(np.eye(3))
    t = np.array([0.0, 1.0])
    r = np.vstack([ident, ident])
    out = slerp_rot6d_series(t, r, np.array([0.25, 0.75]))
    for row in out:
        np.testing.assert_allclose(rot6d_to_matrix(row), np.eye(3), atol=1e-9)


def test_plan_rate_and_hand_linear() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.2, 0.0, 0.0]
    b.left_hand_q[:] = 0.4
    planner = KinematicPlanner(horizon_s=1.6, pos_interp="linear")
    ref = planner.plan([a, b])
    assert ref.rate_hz == 10
    assert ref.n_steps == int(round(1.6 * 10)) + 1
    layout = command_layout()
    lo, hi = layout["left_wrist_pos"]
    np.testing.assert_allclose(ref.commands[0, lo:lo + 3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(ref.commands[-1, lo:lo + 3], [0.2, 0.0, 0.0], atol=1e-9)
    hq = layout["left_hand_q"]
    assert ref.commands[-1, hq[0]] == pytest.approx(0.4)
    mid = ref.commands[ref.n_steps // 2, hq[0]]
    assert 0.0 < mid < 0.4


def test_plan_rotation_slerp_no_flip() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.head_rot6d[:] = matrix_to_rot6d(rpy_to_matrix(0.0, 0.0, 0.4))
    planner = KinematicPlanner(horizon_s=0.8, pos_interp="linear")
    ref = planner.plan([a, b])
    layout = command_layout()
    lo, hi = layout["head_rot6d"]
    mats = [rot6d_to_matrix(row) for row in ref.commands[:, lo:hi]]
    # Consecutive geodesic steps stay well below π (no quaternion flip).
    for r0, r1 in zip(mats, mats[1:], strict=False):
        c = np.clip((float(np.trace(r0.T @ r1)) - 1.0) / 2.0, -1.0, 1.0)
        assert np.arccos(c) < 0.2


def test_plan_with_elbows() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    ea = FivePointCommand(a, np.zeros(3), np.zeros(3))
    eb = FivePointCommand(b, np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]))
    ref = KinematicPlanner(horizon_s=0.8, pos_interp="linear").plan([a, b], elbows=[ea, eb])
    assert ref.elbows is not None
    np.testing.assert_allclose(ref.elbows[-1], [0.1, 0.0, 0.0, 0.0, 0.1, 0.0], atol=1e-9)
