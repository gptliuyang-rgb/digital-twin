"""SONIC §3.5 100 Hz operator-input loop. No simulator imports."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import CommandVector, command_dim, command_layout
from vla.adapters.rotation import matrix_to_rot6d, rot6d_to_matrix, rpy_to_matrix
from wbc.operator import (
    ALLOWED_SOURCES,
    FORBIDDEN_SOURCES,
    OperatorHold,
    OperatorInputError,
    downsample_stride,
    hold_last,
    ingest_operator,
    load_operator_cfg,
    operator_to_hybrid_tokens,
    operator_to_nav_spring,
    operator_to_planner,
    operator_to_policy_tokens,
    operator_to_stream,
    refuse_operator_source,
)
from wbc.planner import slerp_rot6d_series
from wbc.spring import RootSpringState, spring_root_keyframe
from wbc.stream import OPERATOR_INPUT_HZ, PLANNER_HZ, POLICY_HZ, STREAM_HZ, stream_factor
from wbc.teleop import TELEOP_5POINT, TeleopModeIncompatible


def _ramp_window(*, horizon_s: float = 1.6, source: str = "vr_3point", nav_end: float = 0.0):
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.2, 0.0, 0.0]
    b.left_hand_q[:] = 0.4
    b.nav_cmd[:] = [nav_end, 0.0, 0.0]
    n = int(round(horizon_s * OPERATOR_INPUT_HZ)) + 1
    t = np.linspace(0.0, horizon_s, n)
    alpha = (t / horizon_s).reshape(-1, 1)
    commands = (1.0 - alpha) * a.to_flat_vector() + alpha * b.to_flat_vector()
    elbows = None
    if source == "vr_5point":
        elbows = np.zeros((n, 6), dtype=np.float64)
        elbows[:, 0] = t / horizon_s * 0.1
    return ingest_operator(commands, source=source, t_s=t, elbows=elbows)


def test_yaml_locks_paper_rates() -> None:
    cfg = load_operator_cfg()
    assert cfg["not_dexhand2_contact"] is True
    assert cfg["not_table_s4"] is True
    assert cfg["not_hand_mit_ring"] is True
    assert cfg["not_pico_sdk"] is True
    assert cfg["nav_resample"] == "hold_last"
    assert tuple(cfg["sources"]["allowed"]) == ALLOWED_SOURCES
    assert tuple(cfg["sources"]["forbidden"]) == FORBIDDEN_SOURCES
    assert OPERATOR_INPUT_HZ == 100
    assert stream_factor(100) == 5
    assert downsample_stride(100, 50) == 2
    assert downsample_stride(100, 10) == 10
    assert downsample_stride(100, 100) == 1


def test_vla_source_refused() -> None:
    with pytest.raises(OperatorInputError, match="VLA"):
        refuse_operator_source("vla")
    with pytest.raises(OperatorInputError, match="VLA"):
        refuse_operator_source("groot")
    with pytest.raises(OperatorInputError, match="not in"):
        refuse_operator_source("pico")


def test_case_a_refused() -> None:
    with pytest.raises(OperatorInputError, match="command_schema_v1"):
        ingest_operator(np.zeros((4, 50)), source="vr_3point")


def test_nonuniform_grid_refused() -> None:
    a = CommandVector.zeros()
    rows = np.stack([a.to_flat_vector(), a.to_flat_vector(), a.to_flat_vector()])
    with pytest.raises(OperatorInputError, match="uniform 100 Hz"):
        ingest_operator(rows, source="keyboard", t_s=np.array([0.0, 0.01, 0.025]))


def test_1p6s_window_lengths() -> None:
    window = _ramp_window(horizon_s=1.6)
    assert window.rate_hz == 100
    assert window.n_steps == 161
    streamed = operator_to_stream(window, pos_interp="linear")
    assert streamed.rate_hz == STREAM_HZ
    assert streamed.src_hz == OPERATOR_INPUT_HZ
    assert streamed.n_steps == 801
    policy = operator_to_policy_tokens(window)
    assert policy.shape[0] == 81
    planned = operator_to_planner(window)
    assert planned.rate_hz == PLANNER_HZ
    assert planned.n_steps == 17
    # Nested grids: 10 Hz ⊂ 100 Hz ⊂ 500 Hz; 50 Hz ⊂ 100 Hz.
    np.testing.assert_allclose(streamed.t_s[::5], window.t_s, atol=1e-12)
    np.testing.assert_allclose(window.t_s[::2], np.linspace(0.0, 1.6, 81), atol=1e-12)
    np.testing.assert_allclose(planned.t_s, window.t_s[::10], atol=1e-12)


def test_pose_endpoints_and_hand_ride_clock() -> None:
    window = _ramp_window(horizon_s=0.8)
    streamed = operator_to_stream(window, pos_interp="linear")
    layout = command_layout()
    lo, _ = layout["left_wrist_pos"]
    np.testing.assert_allclose(streamed.commands[0, lo : lo + 3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(streamed.commands[-1, lo : lo + 3], [0.2, 0.0, 0.0], atol=1e-9)
    hq = layout["left_hand_q"][0]
    assert streamed.commands[-1, hq] == pytest.approx(0.4)


def test_nav_cmd_is_hold_not_hermite() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.nav_cmd[:] = [6.0, 0.0, 0.0]
    n = 21  # 0.2 s at 100 Hz
    t = np.linspace(0.0, 0.2, n)
    rows = np.stack([a.to_flat_vector()] * 10 + [b.to_flat_vector()] * 11)
    window = ingest_operator(rows, source="gamepad", t_s=t)
    streamed = operator_to_stream(window, pos_interp="cubic_hermite")
    layout = command_layout()
    lo, hi = layout["nav_cmd"]
    # Step at t=0.1 s must not be cubic-smoothed on the 500 Hz ring.
    held = hold_last(window.t_s, window.commands[:, lo:hi], streamed.t_s)
    np.testing.assert_allclose(streamed.commands[:, lo:hi], held, atol=1e-12)
    mid = streamed.t_s.searchsorted(0.1)
    assert streamed.commands[mid - 1, lo] == pytest.approx(0.0)
    assert streamed.commands[mid, lo] == pytest.approx(6.0)


def test_nav_spring_uses_held_command_not_hermite() -> None:
    window = _ramp_window(horizon_s=1.0, nav_end=0.0)
    # Override last-half nav so start vs latest differ; spring uses start.
    layout = command_layout()
    lo, hi = layout["nav_cmd"]
    window.commands[window.n_steps // 2 :, lo] = 6.0
    state = RootSpringState(
        pos_xy_m=np.zeros(2),
        heading_rad=0.0,
        vel_xy_mps=np.zeros(2),
        yaw_rate_rad_s=0.0,
    )
    spring = operator_to_nav_spring(window, state, nav_at="start")
    assert spring.rate_hz == 500
    kf = spring_root_keyframe(state, window.commands[0, lo:hi], t_s=1.0)
    np.testing.assert_allclose(spring.pos_xy_m[-1], kf.pos_xy_m, atol=1e-12)


def test_rotation_slerp_no_flip() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.head_rot6d[:] = matrix_to_rot6d(rpy_to_matrix(0.0, 0.0, 0.4))
    n = int(round(0.8 * 100)) + 1
    t = np.linspace(0.0, 0.8, n)
    rows = np.stack([a.to_flat_vector()] * n)
    layout = command_layout()
    lo, hi = layout["head_rot6d"]
    rows[:, lo:hi] = slerp_rot6d_series(
        np.array([0.0, 0.8]), np.vstack([a.head_rot6d, b.head_rot6d]), t
    )
    window = ingest_operator(rows, source="vr_3point", t_s=t)
    streamed = operator_to_stream(window, pos_interp="linear")
    mats = [rot6d_to_matrix(row) for row in streamed.commands[:, lo:hi]]
    for r0, r1 in zip(mats, mats[1:], strict=False):
        c = np.clip((float(np.trace(r0.T @ r1)) - 1.0) / 2.0, -1.0, 1.0)
        assert np.arccos(c) < 0.05


def test_hybrid_tokens_3point_dim() -> None:
    window = _ramp_window(horizon_s=0.2)
    tokens = operator_to_hybrid_tokens(window)
    assert tokens.shape == (11, 21)  # 0.2 s → 21 operator ticks → 11 at 50 Hz; token 21


def test_hybrid_tokens_5point_dim() -> None:
    window = _ramp_window(horizon_s=0.2, source="vr_5point")
    tokens = operator_to_hybrid_tokens(window)
    assert tokens.shape == (11, 27)


def test_five_point_mismatch_refused() -> None:
    window = _ramp_window(horizon_s=0.2, source="vr_3point")
    with pytest.raises(TeleopModeIncompatible):
        operator_to_hybrid_tokens(window, mode=TELEOP_5POINT)
    a = CommandVector.zeros()
    rows = np.stack([a.to_flat_vector(), a.to_flat_vector()])
    with pytest.raises(OperatorInputError, match="elbows"):
        ingest_operator(rows, source="vr_5point")


def test_live_hold_500hz_read() -> None:
    hold = OperatorHold(source="keyboard")
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.05, 0.0, 0.0]
    hold.push(0.00, a)
    hold.push(0.01, b)
    row = hold.read(0.012)  # 500 Hz tick between 100 Hz samples
    layout = command_layout()
    lo, _ = layout["left_wrist_pos"]
    np.testing.assert_allclose(row[lo : lo + 3], [0.05, 0.0, 0.0], atol=1e-12)
    with pytest.raises(OperatorInputError, match="100 Hz grid"):
        hold.push(0.025, b)
    with pytest.raises(OperatorInputError, match="command_schema_v1"):
        hold.push(0.02, np.zeros(50))


def test_downsample_non_divisor_refused() -> None:
    with pytest.raises(OperatorInputError, match="integer multiple"):
        downsample_stride(100, 30)
    with pytest.raises(OperatorInputError, match="Upsample"):
        downsample_stride(50, 100)


def test_policy_stride_is_not_averaged() -> None:
    window = _ramp_window(horizon_s=0.2)
    policy = operator_to_policy_tokens(window)
    np.testing.assert_allclose(policy, window.commands[::2], atol=1e-12)
    assert command_dim() == window.commands.shape[1]
    assert POLICY_HZ == 50
