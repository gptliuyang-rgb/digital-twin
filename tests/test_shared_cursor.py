"""Encoder look-ahead and planner context share PlannerPlayback.current_frame."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    G1_DECODER_INPUT_DIM,
    G1_N_DOF,
    G1_PLANNER_QPOS_DIM,
    HISTORY_FRAMES,
    PLANNER_CONTEXT_FRAMES,
    PLANNER_LOOKAHEAD_STEPS_50HZ,
    T800_N_LOWER_BODY_DOF,
    TOKEN_DIM,
    decoder_history_dim,
    t800_planner_qpos_dim,
)
from wbc.gather import HardwareSnapshot, ObsGather, ObsGatherError
from wbc.motion_ref import MotionFrame, MotionHold, MotionRefError, look_ahead_indices
from wbc.pd_plant import GAINS_SOURCE
from wbc.pd_stand import pd_stand_kp_kd
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerOnnxError, PlannerQpos, pack_qpos
from wbc.playback import PlannerPlayback, PlaybackError, refuse_motion_cursor_mix
from wbc.shared_cursor import (
    SHARED_CURSOR_YAML,
    DecoderControlTick,
    SharedCursorError,
    SharedPlaybackCursor,
    bind_encoder_hold,
    decoder_joint_history,
    decoder_last_action_history,
    encoder_look_ahead_indices_from_playback,
    load_shared_cursor_cfg,
    qpos_clip_to_motion_frames,
    refuse_decoder_from_planner_qpos,
    refuse_g1_decoder_onnx,
    refuse_last_action_from_decoder_onnx,
    refuse_last_action_from_planner_qpos,
    refuse_motion_cursor_in_playback,
    refuse_policy_action_from_decoder_onnx,
    refuse_policy_action_same_tick_into_obs,
    refuse_policy_pd_as_decoder_run,
    refuse_policy_pd_hermite,
    refuse_policy_pd_physics_as_decoder_run,
    refuse_policy_pd_physics_finite_diff_dq,
    refuse_policy_pd_physics_hermite,
    refuse_policy_pd_physics_invent_imu,
    refuse_policy_pd_physics_q_onto_this_tick_decoder,
    refuse_policy_pd_plant_as_decoder_run,
    refuse_policy_pd_plant_finite_diff_dq,
    refuse_policy_pd_plant_hermite,
    refuse_run_shared_cursor_onnx,
    sample_planner_context_from_playback,
    validate_t800_last_action,
)
from wbc.stream import CommandStreamError


def _row(*, q0: float = 0.0, x_m: float = 0.0) -> np.ndarray:
    q = np.zeros(25)
    q[0] = q0
    return pack_qpos(
        PlannerQpos(
            root_pos_m=np.array([x_m, 0.0, 1.03]),
            root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            q_rad=q,
        )
    )


def _clip(*, n: int, q0: float = 0.0) -> np.ndarray:
    return np.stack([_row(q0=q0 + 0.01 * i, x_m=0.01 * i) for i in range(n)], axis=0)


def _motor(*, lower: float = 0.0) -> np.ndarray:
    q = np.zeros(25)
    q[:T800_N_LOWER_BODY_DOF] = lower
    return q


def _hw(*, q0: float = 0.0, t_s: float = 0.0) -> HardwareSnapshot:
    q = np.zeros(25)
    q[0] = q0
    return HardwareSnapshot(
        t_s=t_s,
        q_hw=q,
        dq_hw=np.zeros(25),
        omega_imu=np.zeros(3),
        imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(25),
    )


def _token() -> np.ndarray:
    return np.linspace(0.0, 1.0, TOKEN_DIM)


def test_yaml_locks_official_table() -> None:
    cfg = load_shared_cursor_cfg()
    assert cfg["adr"] == "ADR-056"
    assert cfg["encoder_reads_playback_current_frame"] is True
    assert cfg["planner_context_reads_playback_current_frame"] is True
    assert cfg["decoder_assembles_on_shared_tick"] is True
    assert cfg["decoder_history_is_hardware_not_clip"] is True
    assert cfg["last_action_is_caller_supplied_policy"] is True
    assert cfg["not_last_action_from_decoder_onnx"] is True
    assert cfg["not_last_action_from_planner_qpos"] is True
    assert cfg["policy_action_feeds_next_tick_last_action"] is True
    assert cfg["not_policy_action_same_tick_decoder_obs"] is True
    assert cfg["not_policy_action_from_decoder_onnx"] is True
    assert cfg["policy_action_feeds_500hz_pd_after_stash"] is True
    assert cfg["policy_action_pd_is_zoh"] is True
    assert cfg["not_policy_action_hermite"] is True
    assert cfg["pd_plant_on_same_tick"] is True
    assert cfg["pd_plant_gains_are_pd_stand_bringup"] is True
    assert cfg["not_sonic_tracking_gains"] is True
    assert cfg["pd_plant_dq_des_is_zero"] is True
    assert cfg["not_pd_plant_from_decoder_onnx"] is True
    assert cfg["not_pd_plant_hermite"] is True
    assert cfg["not_pd_plant_finite_diff_dq"] is True
    assert cfg["not_pd_tau_onto_decoder_obs"] is True
    assert cfg["pd_physics_on_same_tick"] is True
    assert cfg["pd_physics_is_optional"] is True
    assert cfg["not_pd_physics_from_decoder_onnx"] is True
    assert cfg["not_pd_physics_hermite"] is True
    assert cfg["not_pd_physics_finite_diff_dq"] is True
    assert cfg["not_pd_physics_q_onto_this_tick_decoder"] is True
    assert cfg["not_pd_physics_invent_imu"] is True
    assert cfg["not_pd_physics_from_planner_qpos"] is True
    assert cfg["pd_physics_n_steps_per_tick"] == 10
    assert cfg["pd_physics_timestep_s"] == 0.002
    assert cfg["not_decoder_from_planner_qpos"] is True
    assert cfg["not_g1_decoder_onnx"] is True
    assert cfg["not_motion_cursor_mix"] is True
    assert cfg["not_onnx_run"] is True
    assert cfg["control_hz"] == 50
    assert cfg["n_dof"] == 25
    assert cfg["action_dim"] == 25
    assert cfg["g1_action_dim_forbidden"] == 29
    assert cfg["expected_qpos_dim"] == t800_planner_qpos_dim() == 32
    assert cfg["lookahead_steps_50hz"] == PLANNER_LOOKAHEAD_STEPS_50HZ
    assert cfg["encoder_look_ahead_n_frames"] == HISTORY_FRAMES
    assert cfg["encoder_look_ahead_step"] == 5
    assert cfg["planner_context_frames"] == PLANNER_CONTEXT_FRAMES
    assert cfg["token_dim"] == TOKEN_DIM
    assert cfg["expected_decoder_dim"] == decoder_history_dim(25) == 874
    assert cfg["g1_decoder_dim_forbidden"] == G1_DECODER_INPUT_DIM


def test_first_tick_encoder_and_planner_share_frame_1() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    tick = cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    assert tick.current_frame == 1
    assert cur.current_frame == 1
    assert gather.motion.cursor == 1
    assert cur.encoder_look_ahead_indices() == look_ahead_indices(1, 10, 5, 16)
    assert cur.encoder_look_ahead_indices() == [1, 6, 11, 15, 15, 15, 15, 15, 15, 15]
    enc = cur.assemble_encoder("t800")
    assert enc.shape == (842,)
    q_hist = enc[4:254].reshape(10, 25)
    np.testing.assert_allclose(
        q_hist[:, 0], [0.01, 0.06, 0.11, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    ctx = cur.sample_planner_context()
    assert ctx.shape == (1, 4, 32)
    assert ctx[0, 0, 7] == pytest.approx(0.03)


def test_play_false_holds_both_readers() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    paused = cur.tick(_motor(), locomotion_mode=0, play=False)
    assert paused.current_frame == 1
    assert gather.motion.cursor == 1
    assert cur.encoder_look_ahead_indices()[0] == 1
    ctx = cur.sample_planner_context()
    assert ctx[0, 0, 7] == pytest.approx(0.03)


def test_last_frame_encoder_repeats_planner_raises() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    cur.tick(_motor(), locomotion_mode=2, new_qpos=_clip(n=16))
    for _ in range(14):
        cur.tick(_motor(), locomotion_mode=2)
    hold = cur.tick(_motor(lower=0.20), locomotion_mode=2)
    assert hold.current_frame == 15
    assert hold.clamped
    assert hold.skipped_reason == "not_idle"
    assert gather.motion.cursor == 15
    assert cur.encoder_look_ahead_indices() == [15] * 10
    enc = cur.assemble_encoder("t800")
    q_hist = enc[4:254].reshape(10, 25)
    np.testing.assert_allclose(q_hist[:, 0], np.full(10, 0.15))
    with pytest.raises(PlannerOnnxError, match="through index"):
        cur.sample_planner_context()


def test_empty_clip_does_not_invent() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    tick = cur.tick(_motor(), locomotion_mode=0)
    assert tick.skipped_reason == "empty_clip"
    assert gather.motion.n_frames == 0
    with pytest.raises(SharedCursorError, match="empty"):
        cur.encoder_look_ahead_indices()
    with pytest.raises(PlannerOnnxError, match="invent"):
        cur.sample_planner_context()
    with pytest.raises(MotionRefError, match="invent"):
        cur.assemble_encoder("t800")


def test_new_clip_resyncs_cursor() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    nxt = cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16, q0=1.0))
    assert nxt.assigned
    assert nxt.current_frame == 1
    assert gather.motion.cursor == 1
    assert cur.encoder_look_ahead_indices()[0] == 1
    enc = cur.assemble_encoder("t800")
    assert enc.shape == (842,)
    # Successful blend resets current_frame to 0 then the same tick advances to 1.
    # Do not assert raw q0=1.01 — ADR-047 cross-fades the old clip.


def test_qpos_hands_and_g1_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        qpos_clip_to_motion_frames(np.zeros((4, G1_PLANNER_QPOS_DIM)))
    with pytest.raises(SharedCursorError, match="32"):
        qpos_clip_to_motion_frames(np.zeros((4, 45)))
    with pytest.raises(G1CheckpointIncompatible):
        refuse_run_shared_cursor_onnx("planner_sonic.onnx")
    with pytest.raises(PlannerOnnxBlocked, match="874"):
        refuse_run_shared_cursor_onnx()
    with pytest.raises(G1CheckpointIncompatible, match="874"):
        refuse_run_shared_cursor_onnx("model_decoder.onnx")
    with pytest.raises(G1CheckpointIncompatible, match="874"):
        refuse_g1_decoder_onnx("model_decoder.onnx")
    with pytest.raises(SharedCursorError, match="HardwareHold"):
        refuse_decoder_from_planner_qpos()
    with pytest.raises(PlaybackError, match="MotionCursor"):
        refuse_motion_cursor_in_playback()
    with pytest.raises(PlaybackError, match="MotionCursor"):
        SharedPlaybackCursor().bind_motion_cursor(None)  # type: ignore[arg-type]
    with pytest.raises(PlaybackError, match="MotionCursor"):
        refuse_motion_cursor_mix()


def test_bind_encoder_hold_matches_playback_cursor() -> None:
    playback = PlannerPlayback()
    playback.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    hold = MotionHold()
    bind_encoder_hold(playback, hold)
    assert hold.cursor == playback.current_frame == 1
    window = hold.look_ahead(10, 5)
    assert [float(f.q_ref_rad[0]) for f in window] == [
        0.01,
        0.06,
        0.11,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
    ]


def test_sample_planner_context_uses_playback_current_frame() -> None:
    playback = PlannerPlayback()
    playback.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    ctx = sample_planner_context_from_playback(playback)
    assert ctx.shape == (1, 4, 32)
    standalone = sample_planner_context_from_playback(playback)
    np.testing.assert_allclose(ctx, standalone)
    idxs = encoder_look_ahead_indices_from_playback(playback)
    assert idxs[0] == playback.current_frame


def test_missing_dq_is_zeros_not_finite_diff() -> None:
    frames = qpos_clip_to_motion_frames(_clip(n=4))
    assert len(frames) == 4
    assert isinstance(frames[0], MotionFrame)
    np.testing.assert_allclose(frames[1].dq_ref_rad_s, np.zeros(25))
    np.testing.assert_allclose(frames[1].q_ref_rad[0], 0.01)


def test_yaml_flag_required(tmp_path) -> None:
    raw = yaml.safe_load(SHARED_CURSOR_YAML.read_text(encoding="utf-8"))
    raw["not_motion_cursor_mix"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SharedCursorError, match="not_motion_cursor_mix"):
        load_shared_cursor_cfg(path)


def test_control_tick_assembles_decoder_874_from_hardware_not_clip() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=7.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = _token()
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
    )
    assert isinstance(step, DecoderControlTick)
    assert step.current_frame == 1
    assert step.playback.current_frame == 1
    assert step.logger_len == 1
    assert step.decoder_obs.shape == (874,)
    np.testing.assert_allclose(step.decoder_obs[:TOKEN_DIM], token)
    q_hist = decoder_joint_history(step.decoder_obs)
    np.testing.assert_allclose(q_hist[:9], 0.0)
    np.testing.assert_allclose(q_hist[-1, 0], 7.0)
    # Clip hinge at frame 1 is 0.01 — decoder must not have copied it.
    assert cur.encoder_look_ahead_indices()[0] == 1
    enc = cur.assemble_encoder("t800")
    q_enc = enc[4:254].reshape(10, 25)
    np.testing.assert_allclose(q_enc[0, 0], 0.01)
    assert not np.isclose(q_hist[-1, 0], q_enc[0, 0])


def test_ten_control_ticks_fill_decoder_history_independent_of_encoder() -> None:
    gather = ObsGather()
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    last = None
    for i in range(10):
        gather.push_hw(_hw(q0=float(i), t_s=i / 50.0))
        last = cur.control_tick(
            _motor(),
            locomotion_mode=0,
            token=token,
            new_qpos=_clip(n=16) if i == 0 else None,
            t_s=i / 50.0,
        )
    assert last is not None
    assert last.logger_len == 10
    assert last.current_frame == 10
    q_hist = decoder_joint_history(last.decoder_obs)
    np.testing.assert_allclose(q_hist[:, 0], np.arange(10.0))
    assert cur.encoder_look_ahead_indices() == look_ahead_indices(10, 10, 5, 16)


def test_play_false_holds_cursor_but_decoder_ring_still_advances() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(_motor(), locomotion_mode=0, token=token, new_qpos=_clip(n=16), t_s=0.0)
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    paused = cur.control_tick(_motor(), locomotion_mode=0, token=token, play=False, t_s=0.02)
    assert paused.current_frame == 1
    assert paused.playback.skipped_reason == "play_false"
    assert paused.logger_len == 2
    q_hist = decoder_joint_history(paused.decoder_obs)
    np.testing.assert_allclose(q_hist[-1, 0], 2.0)
    assert cur.encoder_look_ahead_indices()[0] == 1


def test_tick_without_token_does_not_invent_decoder() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    tick = cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    assert tick.current_frame == 1
    assert cur.last_decoder_obs is None
    assert len(gather.logger) == 0


def test_assemble_decoder_requires_gather_hardware_and_token() -> None:
    cur = SharedPlaybackCursor()
    with pytest.raises(SharedCursorError, match="ObsGather"):
        cur.assemble_decoder(np.zeros(TOKEN_DIM))
    gather = ObsGather()
    cur = SharedPlaybackCursor(gather=gather)
    with pytest.raises(ObsGatherError, match="empty"):
        cur.assemble_decoder(np.zeros(TOKEN_DIM))
    gather.push_hw(_hw())
    with pytest.raises(SharedCursorError, match="token dim"):
        cur.assemble_decoder(np.zeros(8))


def test_g1_decoder_vector_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        decoder_joint_history(np.zeros(G1_DECODER_INPUT_DIM))
    with pytest.raises(G1CheckpointIncompatible):
        decoder_last_action_history(np.zeros(G1_DECODER_INPUT_DIM))


def _action(*, a0: float) -> np.ndarray:
    a = np.zeros(25)
    a[0] = a0
    return a


def test_control_tick_pushes_caller_last_action_into_decoder_ring() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=7.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = _token()
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        last_action=_action(a0=3.5),
    )
    assert step.last_action.shape == (25,)
    np.testing.assert_allclose(step.last_action[0], 3.5)
    a_hist = decoder_last_action_history(step.decoder_obs)
    np.testing.assert_allclose(a_hist[:9], 0.0)
    np.testing.assert_allclose(a_hist[-1, 0], 3.5)
    q_hist = decoder_joint_history(step.decoder_obs)
    np.testing.assert_allclose(q_hist[-1, 0], 7.0)
    assert not np.isclose(a_hist[-1, 0], q_hist[-1, 0])
    enc = cur.assemble_encoder("t800")
    q_enc = enc[4:254].reshape(10, 25)
    np.testing.assert_allclose(q_enc[0, 0], 0.01)
    assert not np.isclose(a_hist[-1, 0], q_enc[0, 0])


def test_ten_ticks_last_action_history_independent_of_q_and_clip() -> None:
    gather = ObsGather()
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    last = None
    for i in range(10):
        gather.push_hw(_hw(q0=float(i), t_s=i / 50.0))
        last = cur.control_tick(
            _motor(),
            locomotion_mode=0,
            token=token,
            new_qpos=_clip(n=16) if i == 0 else None,
            t_s=i / 50.0,
            last_action=_action(a0=10.0 + i),
        )
    assert last is not None
    q_hist = decoder_joint_history(last.decoder_obs)
    a_hist = decoder_last_action_history(last.decoder_obs)
    np.testing.assert_allclose(q_hist[:, 0], np.arange(10.0))
    np.testing.assert_allclose(a_hist[:, 0], np.arange(10.0, 20.0))
    assert cur.encoder_look_ahead_indices() == look_ahead_indices(10, 10, 5, 16)


def test_omit_last_action_keeps_hardware_hold_does_not_invent_onnx() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=np.zeros(TOKEN_DIM),
        new_qpos=_clip(n=16),
        t_s=0.0,
    )
    np.testing.assert_allclose(step.last_action, np.zeros(25))
    a_hist = decoder_last_action_history(step.decoder_obs)
    np.testing.assert_allclose(a_hist[-1], 0.0)


def test_play_false_still_logs_caller_last_action() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        last_action=_action(a0=1.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    paused = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        play=False,
        t_s=0.02,
        last_action=_action(a0=8.0),
    )
    assert paused.current_frame == 1
    assert paused.logger_len == 2
    a_hist = decoder_last_action_history(paused.decoder_obs)
    np.testing.assert_allclose(a_hist[-1, 0], 8.0)


def test_tick_without_token_still_applies_last_action() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    tick = cur.tick(
        _motor(),
        locomotion_mode=0,
        new_qpos=_clip(n=16),
        last_action=_action(a0=4.0),
    )
    assert tick.current_frame == 1
    assert cur.last_decoder_obs is None
    assert len(gather.logger) == 0
    np.testing.assert_allclose(gather.hw.read().last_action[0], 4.0)


def test_last_action_g1_hands_qpos_onnx_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        validate_t800_last_action(np.zeros(G1_N_DOF))
    with pytest.raises(G1CheckpointIncompatible):
        validate_t800_last_action(np.zeros(G1_PLANNER_QPOS_DIM))
    with pytest.raises(SharedCursorError, match="planner qpos"):
        validate_t800_last_action(np.zeros(32))
    with pytest.raises(SharedCursorError, match="DexHand2"):
        validate_t800_last_action(np.zeros(45))
    with pytest.raises(SharedCursorError, match="NaN"):
        validate_t800_last_action(np.array([np.nan] + [0.0] * 24))
    with pytest.raises(SharedCursorError, match="planner clip"):
        refuse_last_action_from_planner_qpos()
    with pytest.raises(SharedCursorError, match="fake decoder ONNX"):
        refuse_last_action_from_decoder_onnx()
    gather = ObsGather()
    cur = SharedPlaybackCursor(gather=gather)
    with pytest.raises(ObsGatherError, match="empty"):
        cur.apply_last_action(_action(a0=1.0))
    cur_bare = SharedPlaybackCursor()
    with pytest.raises(SharedCursorError, match="ObsGather"):
        cur_bare.apply_last_action(_action(a0=1.0))


def test_policy_action_is_delayed_one_tick_into_last_action() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=7.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = _token()
    first = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=3.5),
    )
    np.testing.assert_allclose(first.last_action, 0.0)
    assert first.policy_action is not None
    np.testing.assert_allclose(first.policy_action[0], 3.5)
    a_hist = decoder_last_action_history(first.decoder_obs)
    np.testing.assert_allclose(a_hist[-1], 0.0)
    q_hist = decoder_joint_history(first.decoder_obs)
    np.testing.assert_allclose(q_hist[-1, 0], 7.0)
    gather.push_hw(_hw(q0=8.0, t_s=0.02))
    second = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
        policy_action=_action(a0=4.5),
    )
    np.testing.assert_allclose(second.last_action[0], 3.5)
    assert second.policy_action is not None
    np.testing.assert_allclose(second.policy_action[0], 4.5)
    a_hist = decoder_last_action_history(second.decoder_obs)
    np.testing.assert_allclose(a_hist[-1, 0], 3.5)
    q_hist = decoder_joint_history(second.decoder_obs)
    np.testing.assert_allclose(q_hist[-1, 0], 8.0)
    assert not np.isclose(a_hist[-1, 0], q_hist[-1, 0])


def test_ten_ticks_policy_action_history_is_shifted_vs_q() -> None:
    gather = ObsGather()
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    last = None
    for i in range(10):
        gather.push_hw(_hw(q0=float(i), t_s=i / 50.0))
        last = cur.control_tick(
            _motor(),
            locomotion_mode=0,
            token=token,
            new_qpos=_clip(n=16) if i == 0 else None,
            t_s=i / 50.0,
            policy_action=_action(a0=10.0 + i),
        )
    assert last is not None
    q_hist = decoder_joint_history(last.decoder_obs)
    a_hist = decoder_last_action_history(last.decoder_obs)
    np.testing.assert_allclose(q_hist[:, 0], np.arange(10.0))
    expected_a = np.concatenate(([0.0], np.arange(10.0, 19.0)))
    np.testing.assert_allclose(a_hist[:, 0], expected_a)
    assert last.policy_action is not None
    np.testing.assert_allclose(last.policy_action[0], 19.0)
    np.testing.assert_allclose(cur.pending_policy_action[0], 19.0)
    assert cur.encoder_look_ahead_indices() == look_ahead_indices(10, 10, 5, 16)


def test_omit_policy_action_keeps_stash_feeding_last_action() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=6.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    held = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
    )
    np.testing.assert_allclose(held.last_action[0], 6.0)
    a_hist = decoder_last_action_history(held.decoder_obs)
    np.testing.assert_allclose(a_hist[-1, 0], 6.0)
    assert held.policy_action is not None
    np.testing.assert_allclose(held.policy_action[0], 6.0)


def test_play_false_still_delays_policy_action() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=1.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    paused = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        play=False,
        t_s=0.02,
        policy_action=_action(a0=8.0),
    )
    assert paused.current_frame == 1
    assert paused.logger_len == 2
    a_hist = decoder_last_action_history(paused.decoder_obs)
    np.testing.assert_allclose(a_hist[-1, 0], 1.0)
    assert paused.policy_action is not None
    np.testing.assert_allclose(paused.policy_action[0], 8.0)


def test_tick_without_token_still_stashes_policy_action() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    tick = cur.tick(
        _motor(),
        locomotion_mode=0,
        new_qpos=_clip(n=16),
        policy_action=_action(a0=4.0),
    )
    assert tick.current_frame == 1
    assert cur.last_decoder_obs is None
    assert len(gather.logger) == 0
    np.testing.assert_allclose(gather.hw.read().last_action, 0.0)
    np.testing.assert_allclose(cur.pending_policy_action[0], 4.0)
    gather.push_hw(_hw(q0=1.0, t_s=0.02))
    cur.tick(_motor(), locomotion_mode=0, t_s=0.02)
    np.testing.assert_allclose(gather.hw.read().last_action[0], 4.0)


def test_same_tick_last_action_overrides_delayed_stash() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=9.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
        last_action=_action(a0=1.5),
        policy_action=_action(a0=2.5),
    )
    np.testing.assert_allclose(step.last_action[0], 1.5)
    assert step.policy_action is not None
    np.testing.assert_allclose(step.policy_action[0], 2.5)
    a_hist = decoder_last_action_history(step.decoder_obs)
    np.testing.assert_allclose(a_hist[-1, 0], 1.5)


def test_policy_action_g1_hands_qpos_onnx_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        SharedPlaybackCursor().stash_policy_action(np.zeros(G1_N_DOF))
    with pytest.raises(SharedCursorError, match="planner qpos"):
        SharedPlaybackCursor().stash_policy_action(np.zeros(32))
    with pytest.raises(SharedCursorError, match="DexHand2"):
        SharedPlaybackCursor().stash_policy_action(np.zeros(45))
    with pytest.raises(SharedCursorError, match="next 50 Hz"):
        refuse_policy_action_from_decoder_onnx()
    with pytest.raises(SharedCursorError, match="next"):
        refuse_policy_action_same_tick_into_obs()
    with pytest.raises(CommandStreamError, match="ZOH"):
        refuse_policy_pd_hermite()
    with pytest.raises(CommandStreamError, match="decoder ONNX"):
        refuse_policy_pd_as_decoder_run()


def test_policy_action_is_zoh_held_on_pd_after_stash() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=7.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = _token()
    first = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=3.5),
    )
    np.testing.assert_allclose(first.last_action, 0.0)
    np.testing.assert_allclose(first.pd_q_des[0], 3.5)
    kp, kd = pd_stand_kp_kd()
    assert first.pd_gains_source == GAINS_SOURCE
    np.testing.assert_allclose(first.pd_tau_nm[0], kp[0] * (3.5 - 7.0) - kd[0] * 0.0)
    a_hist = decoder_last_action_history(first.decoder_obs)
    np.testing.assert_allclose(a_hist[-1], 0.0)
    period = cur.pd_hold.zoh_period(t0_s=0.0)
    assert period.n_steps == 10
    assert period.hold == "zoh"
    np.testing.assert_allclose(period.q_des_rad[:, 0], 3.5)
    gather.push_hw(_hw(q0=8.0, t_s=0.02))
    second = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
        policy_action=_action(a0=4.5),
    )
    np.testing.assert_allclose(second.last_action[0], 3.5)
    np.testing.assert_allclose(second.pd_q_des[0], 4.5)
    np.testing.assert_allclose(second.pd_tau_nm[0], kp[0] * (4.5 - 8.0))
    a_hist = decoder_last_action_history(second.decoder_obs)
    np.testing.assert_allclose(a_hist[-1, 0], 3.5)


def test_ten_ticks_pd_holds_latest_a_t_not_decoder_history() -> None:
    gather = ObsGather()
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    last = None
    for i in range(10):
        gather.push_hw(_hw(q0=float(i), t_s=i / 50.0))
        last = cur.control_tick(
            _motor(),
            locomotion_mode=0,
            token=token,
            new_qpos=_clip(n=16) if i == 0 else None,
            t_s=i / 50.0,
            policy_action=_action(a0=10.0 + i),
        )
    assert last is not None
    a_hist = decoder_last_action_history(last.decoder_obs)
    expected_a = np.concatenate(([0.0], np.arange(10.0, 19.0)))
    np.testing.assert_allclose(a_hist[:, 0], expected_a)
    np.testing.assert_allclose(last.pd_q_des[0], 19.0)
    np.testing.assert_allclose(cur.pd_hold.read()[0], 19.0)
    kp, _kd = pd_stand_kp_kd()
    np.testing.assert_allclose(last.pd_tau_nm[0], kp[0] * (19.0 - 9.0))


def test_omit_policy_action_keeps_pd_hold() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=6.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    held = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
    )
    np.testing.assert_allclose(held.last_action[0], 6.0)
    np.testing.assert_allclose(held.pd_q_des[0], 6.0)
    kp, _kd = pd_stand_kp_kd()
    np.testing.assert_allclose(held.pd_tau_nm[0], kp[0] * (6.0 - 2.0))


def test_play_false_still_pushes_pd_after_stash() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=1.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    paused = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        play=False,
        t_s=0.02,
        policy_action=_action(a0=8.0),
    )
    assert paused.current_frame == 1
    np.testing.assert_allclose(paused.last_action[0], 1.0)
    np.testing.assert_allclose(paused.pd_q_des[0], 8.0)


def test_tick_without_token_still_pushes_pd_after_stash() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    cur.tick(
        _motor(),
        locomotion_mode=0,
        new_qpos=_clip(n=16),
        policy_action=_action(a0=4.0),
        t_s=0.0,
    )
    np.testing.assert_allclose(gather.hw.read().last_action, 0.0)
    np.testing.assert_allclose(cur.pd_hold.read()[0], 4.0)


def test_same_tick_last_action_does_not_write_pd() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=9.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
        last_action=_action(a0=1.5),
        policy_action=_action(a0=2.5),
    )
    np.testing.assert_allclose(step.last_action[0], 1.5)
    np.testing.assert_allclose(step.pd_q_des[0], 2.5)
    kp, _kd = pd_stand_kp_kd()
    np.testing.assert_allclose(step.pd_tau_nm[0], kp[0] * (2.5 - 2.0))


def test_stash_alone_does_not_write_pd() -> None:
    cur = SharedPlaybackCursor()
    cur.stash_policy_action(_action(a0=7.0))
    np.testing.assert_allclose(cur.pd_hold.read(), 0.0)
    np.testing.assert_allclose(cur.pd_plant.read(), 0.0)
    np.testing.assert_allclose(cur.pending_policy_action[0], 7.0)


def test_pd_g1_hands_qpos_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        SharedPlaybackCursor().push_policy_pd(np.zeros(G1_N_DOF))
    with pytest.raises(CommandStreamError, match="planner qpos"):
        SharedPlaybackCursor().push_policy_pd(np.zeros(32))
    with pytest.raises(CommandStreamError, match="DexHand2"):
        SharedPlaybackCursor().push_policy_pd(np.zeros(45))
    with pytest.raises(CommandStreamError, match="ZOH"):
        refuse_policy_pd_plant_hermite()
    with pytest.raises(CommandStreamError, match="decoder ONNX"):
        refuse_policy_pd_plant_as_decoder_run()
    with pytest.raises(CommandStreamError, match="finite-diff"):
        refuse_policy_pd_plant_finite_diff_dq()
    with pytest.raises(CommandStreamError, match="ZOH"):
        refuse_policy_pd_physics_hermite()
    with pytest.raises(CommandStreamError, match="decoder ONNX"):
        refuse_policy_pd_physics_as_decoder_run()
    with pytest.raises(CommandStreamError, match="finite-diff"):
        refuse_policy_pd_physics_finite_diff_dq()
    with pytest.raises(CommandStreamError, match="before"):
        refuse_policy_pd_physics_q_onto_this_tick_decoder()
    with pytest.raises(CommandStreamError, match="invent IMU"):
        refuse_policy_pd_physics_invent_imu()


def test_tau_does_not_enter_decoder_obs() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=7.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=_token(),
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=3.5),
    )
    a_hist = decoder_last_action_history(step.decoder_obs)
    np.testing.assert_allclose(a_hist[-1], 0.0)
    assert not np.allclose(step.pd_tau_nm, 0.0)
    assert step.pd_gains_source == GAINS_SOURCE
    assert step.pd_physics_n_steps == 0


class _IncrementPhysics:
    """Test double. Not MuJoCo. Increments q[0] by 0.001 each 2 ms substep."""

    n_dof = 25
    timestep_s = 0.002

    def __init__(self, q0: float = 0.0) -> None:
        self.q = np.zeros(25)
        self.q[0] = q0
        self.dq = np.zeros(25)
        self.n_applies = 0
        self.last_tau = np.zeros(25)

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.dq.copy()

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.last_tau = np.asarray(tau_nm, dtype=np.float64).reshape(25).copy()
        self.q = self.q.copy()
        self.q[0] += 0.001
        self.n_applies += 1
        return self.read_q_dq()


def test_optional_physics_ten_substeps_do_not_rewrite_this_tick_decoder() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=2.0, t_s=0.0))
    physics = _IncrementPhysics(q0=2.0)
    cur = SharedPlaybackCursor(gather=gather, physics=physics)
    token = np.zeros(TOKEN_DIM)
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=6.0),
    )
    q_hist = decoder_joint_history(step.decoder_obs)
    np.testing.assert_allclose(q_hist[-1, 0], 2.0)
    assert step.pd_physics_n_steps == 10
    assert physics.n_applies == 10
    np.testing.assert_allclose(step.pd_physics_q[0], 2.01)
    np.testing.assert_allclose(gather.hw.read().q_hw[0], 2.01)
    np.testing.assert_allclose(gather.hw.read().omega_imu, 0.0)
    kp, _kd = pd_stand_kp_kd()
    np.testing.assert_allclose(step.pd_tau_nm[0], kp[0] * (6.0 - 2.0))
    assert step.pd_physics_tau_nm[0] != pytest.approx(step.pd_tau_nm[0])
    a_hist = decoder_last_action_history(step.decoder_obs)
    np.testing.assert_allclose(a_hist[-1], 0.0)


def test_omit_physics_keeps_adr_055() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=2.0, t_s=0.0))
    cur = SharedPlaybackCursor(gather=gather)
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=_token(),
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=6.0),
    )
    assert step.pd_physics_n_steps == 0
    np.testing.assert_allclose(step.pd_physics_q, 0.0)
    np.testing.assert_allclose(gather.hw.read().q_hw[0], 2.0)


def test_stash_alone_does_not_write_physics() -> None:
    physics = _IncrementPhysics()
    cur = SharedPlaybackCursor(physics=physics)
    cur.stash_policy_action(_action(a0=7.0))
    assert physics.n_applies == 0
    assert cur.last_pd_physics is None


def test_same_tick_last_action_does_not_write_physics_q_des() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(q0=1.0, t_s=0.0))
    physics = _IncrementPhysics(q0=1.0)
    cur = SharedPlaybackCursor(gather=gather, physics=physics)
    token = np.zeros(TOKEN_DIM)
    cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        new_qpos=_clip(n=16),
        t_s=0.0,
        policy_action=_action(a0=9.0),
    )
    gather.push_hw(_hw(q0=2.0, t_s=0.02))
    physics.q[:] = 0.0
    physics.q[0] = 2.0
    step = cur.control_tick(
        _motor(),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
        last_action=_action(a0=1.5),
        policy_action=_action(a0=2.5),
    )
    np.testing.assert_allclose(step.last_action[0], 1.5)
    np.testing.assert_allclose(step.pd_q_des[0], 2.5)
    assert step.pd_physics_n_steps == 10
    assert physics.n_applies == 20

