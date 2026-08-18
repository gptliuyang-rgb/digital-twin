"""Encoder look-ahead and planner context share PlannerPlayback.current_frame."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    G1_PLANNER_QPOS_DIM,
    HISTORY_FRAMES,
    PLANNER_CONTEXT_FRAMES,
    PLANNER_LOOKAHEAD_STEPS_50HZ,
    T800_N_LOWER_BODY_DOF,
    t800_planner_qpos_dim,
)
from wbc.gather import HardwareSnapshot, ObsGather
from wbc.motion_ref import MotionFrame, MotionHold, MotionRefError, look_ahead_indices
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerOnnxError, PlannerQpos, pack_qpos
from wbc.playback import PlannerPlayback, PlaybackError, refuse_motion_cursor_mix
from wbc.shared_cursor import (
    SHARED_CURSOR_YAML,
    SharedCursorError,
    SharedPlaybackCursor,
    bind_encoder_hold,
    encoder_look_ahead_indices_from_playback,
    load_shared_cursor_cfg,
    qpos_clip_to_motion_frames,
    refuse_motion_cursor_in_playback,
    refuse_run_shared_cursor_onnx,
    sample_planner_context_from_playback,
)


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


def _hw() -> HardwareSnapshot:
    return HardwareSnapshot(
        t_s=0.0,
        q_hw=np.zeros(25),
        dq_hw=np.zeros(25),
        omega_imu=np.zeros(3),
        imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(25),
    )


def test_yaml_locks_official_table() -> None:
    cfg = load_shared_cursor_cfg()
    assert cfg["adr"] == "ADR-050"
    assert cfg["encoder_reads_playback_current_frame"] is True
    assert cfg["planner_context_reads_playback_current_frame"] is True
    assert cfg["not_motion_cursor_mix"] is True
    assert cfg["not_onnx_run"] is True
    assert cfg["control_hz"] == 50
    assert cfg["n_dof"] == 25
    assert cfg["expected_qpos_dim"] == t800_planner_qpos_dim() == 32
    assert cfg["lookahead_steps_50hz"] == PLANNER_LOOKAHEAD_STEPS_50HZ
    assert cfg["encoder_look_ahead_n_frames"] == HISTORY_FRAMES
    assert cfg["encoder_look_ahead_step"] == 5
    assert cfg["planner_context_frames"] == PLANNER_CONTEXT_FRAMES


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
    with pytest.raises(PlannerOnnxBlocked, match="25"):
        refuse_run_shared_cursor_onnx()
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
