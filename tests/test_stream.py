"""SONIC §3.5 500 Hz command stream. No simulator imports."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import CommandVector, command_layout
from vla.adapters.rotation import matrix_to_rot6d, rot6d_to_matrix, rpy_to_matrix
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.planner import KinematicPlanner, PlannedRef
from wbc.spring import RootSpringState, spring_root_keyframe
from wbc.stream import (
    OPERATOR_INPUT_HZ,
    PLANNER_HZ,
    POLICY_HZ,
    STREAM_HZ,
    CommandStreamError,
    PolicyPdHold,
    load_stream_cfg,
    refuse_policy_action_as_decoder_run,
    refuse_policy_action_hermite,
    require_t800_pd_action,
    stream_factor,
    stream_nav_spring,
    stream_planned_ref,
    stream_policy_action_zoh,
    stream_policy_tokens,
)


def test_yaml_locks_paper_rates() -> None:
    cfg = load_stream_cfg()
    assert cfg["not_dexhand2_contact"] is True
    assert cfg["not_table_s4"] is True
    assert cfg["not_hand_mit_ring"] is True
    assert STREAM_HZ == 500
    assert POLICY_HZ == 50
    assert PLANNER_HZ == 10
    assert OPERATOR_INPUT_HZ == 100
    assert stream_factor(10) == 50
    assert stream_factor(50) == 10
    assert stream_factor(100) == 5
    assert cfg["factor_operator_to_stream"] == 5
    assert cfg["nav_eval"] == "closed_form_eq8"
    assert cfg["adr"] == "ADR-054"
    assert cfg["policy_action_pd_is_zoh"] is True
    assert cfg["not_policy_action_hermite"] is True
    assert cfg["policy_action_feeds_500hz_pd_after_stash"] is True
    assert cfg["not_policy_action_from_decoder_onnx"] is True
    assert cfg["not_policy_action_same_tick_decoder_obs"] is True
    assert cfg["pd_action_dim"] == 25
    assert cfg["g1_action_dim_forbidden"] == 29
    assert cfg["policy_action_hold"] == "zoh"


def test_non_integer_src_hz_refused() -> None:
    with pytest.raises(CommandStreamError, match="does not divide"):
        stream_factor(7)


def test_planner_10hz_to_500hz_length_and_endpoints() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.2, 0.0, 0.0]
    b.left_hand_q[:] = 0.4
    ref = KinematicPlanner(horizon_s=1.6, pos_interp="linear").plan([a, b])
    assert ref.rate_hz == 10
    streamed = stream_planned_ref(ref, pos_interp="linear")
    assert streamed.rate_hz == 500
    assert streamed.src_hz == 10
    assert streamed.n_steps == (ref.n_steps - 1) * 50 + 1
    layout = command_layout()
    lo, _ = layout["left_wrist_pos"]
    np.testing.assert_allclose(streamed.commands[0, lo : lo + 3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(streamed.commands[-1, lo : lo + 3], [0.2, 0.0, 0.0], atol=1e-9)
    hq = layout["left_hand_q"][0]
    assert streamed.commands[-1, hq] == pytest.approx(0.4)
    # 10 Hz samples are a subset of the 500 Hz grid.
    np.testing.assert_allclose(streamed.t_s[::50], ref.t_s, atol=1e-12)


def test_wrong_planner_rate_refused() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    good = KinematicPlanner(horizon_s=0.8, pos_interp="linear").plan([a, b])
    bad = PlannedRef(
        t_s=good.t_s,
        commands=good.commands,
        elbows=None,
        rate_hz=20,
        horizon_s=good.horizon_s,
    )
    with pytest.raises(CommandStreamError, match="10 Hz"):
        stream_planned_ref(bad)


def test_policy_50hz_to_500hz_length() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.1, 0.0, 0.0]
    chunk = np.stack([a.to_flat_vector(), b.to_flat_vector()])
    out = stream_policy_tokens(chunk, pos_interp="linear")
    assert out.rate_hz == 500
    assert out.src_hz == 50
    assert out.n_steps == 11  # (2-1)*10+1
    layout = command_layout()
    lo, _ = layout["left_wrist_pos"]
    np.testing.assert_allclose(out.commands[-1, lo : lo + 3], [0.1, 0.0, 0.0], atol=1e-9)


def test_case_a_refused() -> None:
    with pytest.raises(CommandStreamError, match="command_schema_v1"):
        stream_policy_tokens(np.zeros((4, 50)), pos_interp="linear")


def test_rotation_slerp_no_flip() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.head_rot6d[:] = matrix_to_rot6d(rpy_to_matrix(0.0, 0.0, 0.4))
    ref = KinematicPlanner(horizon_s=0.8, pos_interp="linear").plan([a, b])
    streamed = stream_planned_ref(ref, pos_interp="linear")
    layout = command_layout()
    lo, hi = layout["head_rot6d"]
    mats = [rot6d_to_matrix(row) for row in streamed.commands[:, lo:hi]]
    for r0, r1 in zip(mats, mats[1:], strict=False):
        c = np.clip((float(np.trace(r0.T @ r1)) - 1.0) / 2.0, -1.0, 1.0)
        assert np.arccos(c) < 0.05


def test_enum_nearest_not_averaged() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.tool_trigger = 1
    ref = KinematicPlanner(horizon_s=0.8, pos_interp="linear").plan([a, b])
    streamed = stream_planned_ref(ref, pos_interp="linear")
    trig = command_layout()["tool_trigger"][0]
    assert set(np.unique(streamed.commands[:, trig]).tolist()) <= {0.0, 1.0}


def test_nav_spring_is_closed_form_not_hermite() -> None:
    state = RootSpringState(
        pos_xy_m=np.zeros(2),
        heading_rad=0.0,
        vel_xy_mps=np.array([6.0, 0.0]),
        yaw_rate_rad_s=0.0,
    )
    nav = np.array([-6.0, 0.0, 0.0])
    streamed = stream_nav_spring(state, nav, horizon_s=1.0)
    assert streamed.rate_hz == 500
    assert streamed.n_steps == 501
    np.testing.assert_allclose(streamed.pos_xy_m[0], state.pos_xy_m, atol=1e-12)
    # Every 500 Hz sample matches Eq. 8, not a 10 Hz Hermite resample.
    for i in (0, 50, 250, 500):
        kf = spring_root_keyframe(state, nav, t_s=float(streamed.t_s[i]))
        np.testing.assert_allclose(streamed.pos_xy_m[i], kf.pos_xy_m, atol=1e-12)
        assert streamed.heading_rad[i] == pytest.approx(kf.heading_rad, abs=1e-12)
    # Reverse 6→−6 at 1 s stays strictly between ballistic −6 and +6.
    assert -6.0 < float(streamed.pos_xy_m[-1, 0]) < 6.0


def test_25d_policy_action_not_command_schema_spline() -> None:
    with pytest.raises(CommandStreamError, match="stream_policy_action_zoh"):
        stream_policy_tokens(np.zeros((2, 25)), pos_interp="linear")


def test_command_schema_not_pd_zoh() -> None:
    with pytest.raises(CommandStreamError, match="stream_policy_tokens"):
        stream_policy_action_zoh(np.zeros((2, 75)))


def test_policy_action_zoh_two_steps() -> None:
    a0 = np.zeros(25)
    a0[0] = 1.0
    a1 = np.zeros(25)
    a1[0] = 2.0
    out = stream_policy_action_zoh(np.stack([a0, a1]))
    assert out.rate_hz == 500
    assert out.src_hz == 50
    assert out.hold == "zoh"
    assert out.n_steps == 11
    np.testing.assert_allclose(out.q_des_rad[:10, 0], 1.0)
    np.testing.assert_allclose(out.q_des_rad[10, 0], 2.0)
    np.testing.assert_allclose(out.t_s[::10], [0.0, 0.02], atol=1e-12)


def test_policy_action_zoh_single_period() -> None:
    a = np.zeros(25)
    a[0] = 3.5
    out = stream_policy_action_zoh(a)
    assert out.n_steps == 10
    np.testing.assert_allclose(out.q_des_rad[:, 0], 3.5)


def test_policy_action_hermite_and_onnx_refused() -> None:
    with pytest.raises(CommandStreamError, match="ZOH"):
        refuse_policy_action_hermite()
    with pytest.raises(CommandStreamError, match="decoder ONNX"):
        refuse_policy_action_as_decoder_run()


def test_pd_hold_zoh_and_refusals() -> None:
    hold = PolicyPdHold()
    np.testing.assert_allclose(hold.read(), 0.0)
    a = np.zeros(25)
    a[0] = 4.0
    hold.push(a, t_s=0.0)
    np.testing.assert_allclose(hold.read(0.018)[0], 4.0)
    with pytest.raises(CommandStreamError, match="before the last"):
        hold.read(-0.001)
    with pytest.raises(G1CheckpointIncompatible):
        require_t800_pd_action(np.zeros(29))
    with pytest.raises(CommandStreamError, match="planner qpos"):
        require_t800_pd_action(np.zeros(32))
    with pytest.raises(CommandStreamError, match="DexHand2"):
        require_t800_pd_action(np.zeros(45))
