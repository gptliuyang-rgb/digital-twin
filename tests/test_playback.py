"""SONIC 50 Hz planner last-frame playback cursor. No ONNX, no simulator."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    G1_N_DOF,
    G1_PLANNER_QPOS_DIM,
    PLANNER_BLEND_FRAMES,
    PLANNER_CONTROL_HZ,
    PLANNER_IDLE_MODE,
    PLAYBACK_CONTROL_HZ,
    T800_N_LOWER_BODY_DOF,
    t800_planner_qpos_dim,
)
from wbc.idle_readapt import IdleReadaptState
from wbc.planner_blend import PlannerBlendError
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerOnnxError, PlannerQpos, pack_qpos
from wbc.playback import (
    PLAYBACK_YAML,
    PlannerPlayback,
    PlaybackError,
    clamp_frame,
    load_playback_cfg,
    refuse_command_schema_as_clip,
    refuse_motion_cursor_mix,
    refuse_reference_motion_loop,
    refuse_run_playback_onnx,
)


def _row(*, q0: float = 0.0, x_m: float = 0.0, upper: float = 0.4) -> np.ndarray:
    q = np.zeros(25)
    q[0] = q0
    q[T800_N_LOWER_BODY_DOF:] = upper
    return pack_qpos(
        PlannerQpos(
            root_pos_m=np.array([x_m, 0.0, 1.03]),
            root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            q_rad=q,
        )
    )


def _clip(*, n: int, q0: float = 0.0) -> np.ndarray:
    return np.stack([_row(q0=q0 + 0.01 * i, x_m=0.01 * i) for i in range(n)], axis=0)


def _motor(*, lower: float = 0.0, upper: float = 0.4) -> np.ndarray:
    q = np.zeros(25)
    q[:T800_N_LOWER_BODY_DOF] = lower
    q[T800_N_LOWER_BODY_DOF:] = upper
    return q


def test_yaml_locks_official_table() -> None:
    cfg = load_playback_cfg()
    assert cfg["adr"] == "ADR-049"
    assert cfg["not_onnx_run"] is True
    assert cfg["not_reference_motion_loop"] is True
    assert cfg["control_hz"] == PLAYBACK_CONTROL_HZ == PLANNER_CONTROL_HZ == 50
    assert cfg["n_dof"] == 25
    assert cfg["expected_qpos_dim"] == t800_planner_qpos_dim() == 32
    assert cfg["idle_mode"] == PLANNER_IDLE_MODE
    assert cfg["blend_num_frames"] == PLANNER_BLEND_FRAMES


def test_clamp_formula() -> None:
    assert clamp_frame(0, 16) == (1, False)
    assert clamp_frame(14, 16) == (15, False)
    assert clamp_frame(15, 16) == (15, True)
    assert clamp_frame(0, 1) == (0, True)
    with pytest.raises(PlaybackError, match="empty"):
        clamp_frame(0, 0)


def test_first_assign_then_advance_same_tick() -> None:
    cur = PlannerPlayback()
    tick = cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    assert tick.assigned
    assert tick.first_copy
    assert tick.current_frame == 1
    assert not tick.clamped
    assert tick.idle is None
    assert tick.q_rad is not None
    assert tick.q_rad[0] == pytest.approx(0.01)


def test_reaches_last_then_clamps_and_readapts() -> None:
    cur = PlannerPlayback()
    cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    for _ in range(13):
        tick = cur.tick(_motor(), locomotion_mode=0)
        assert not tick.clamped
    last_without_clamp = cur.tick(_motor(), locomotion_mode=0)
    assert last_without_clamp.current_frame == 15
    assert not last_without_clamp.clamped
    assert last_without_clamp.idle is None
    hold = cur.tick(_motor(lower=0.20), locomotion_mode=0)
    assert hold.clamped
    assert hold.current_frame == 15
    assert hold.idle is not None
    assert hold.idle.first_store
    assert hold.idle.state is IdleReadaptState.ADAPTING
    assert hold.q_rad is not None
    assert hold.q_rad[0] == pytest.approx(0.98 * 0.15 + 0.02 * 0.20)
    np.testing.assert_allclose(hold.q_rad[12:], np.full(13, 0.4))


def test_walk_clamps_without_readapt() -> None:
    cur = PlannerPlayback()
    first = cur.tick(_motor(), locomotion_mode=2, new_qpos=_clip(n=2))
    assert first.current_frame == 1
    assert not first.clamped
    last = cur.tick(_motor(lower=0.20), locomotion_mode=2)
    assert last.clamped
    assert last.current_frame == 1
    assert last.skipped_reason == "not_idle"
    assert last.idle is None
    assert last.q_rad is not None
    assert last.q_rad[0] == pytest.approx(0.01)


def test_squat_does_not_enter() -> None:
    cur = PlannerPlayback()
    cur.tick(_motor(), locomotion_mode=4, new_qpos=_clip(n=1))
    assert cur.current_frame == 0
    tick = cur.tick(_motor(lower=0.20), locomotion_mode=4)
    assert tick.clamped
    assert tick.skipped_reason == "not_idle"


def test_play_false_holds_frame() -> None:
    cur = PlannerPlayback()
    cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=8))
    assert cur.current_frame == 1
    paused = cur.tick(_motor(lower=0.20), locomotion_mode=0, play=False)
    assert paused.skipped_reason == "play_false"
    assert paused.current_frame == 1
    assert not paused.clamped
    resumed = cur.tick(_motor(), locomotion_mode=0, play=True)
    assert resumed.current_frame == 2


def test_one_frame_clip_clamps_on_first_tick() -> None:
    cur = PlannerPlayback()
    tick = cur.tick(_motor(lower=0.20), locomotion_mode=0, new_qpos=_clip(n=1))
    assert tick.first_copy
    assert tick.clamped
    assert tick.current_frame == 0
    assert tick.idle is not None
    assert tick.idle.state is IdleReadaptState.ADAPTING


def test_new_clip_clears_idle_store() -> None:
    cur = PlannerPlayback()
    cur.tick(_motor(lower=0.20), locomotion_mode=0, new_qpos=_clip(n=1))
    assert cur.idle.stored
    nxt = cur.tick(_motor(lower=0.20), locomotion_mode=0, new_qpos=_clip(n=8, q0=1.0))
    assert nxt.assigned
    assert not cur.idle.stored
    assert nxt.current_frame == 1
    assert nxt.idle is None


def test_late_assign_skips_without_clearing_store() -> None:
    cur = PlannerPlayback()
    cur.tick(_motor(), locomotion_mode=0, new_qpos=_clip(n=16))
    for _ in range(14):
        cur.tick(_motor(), locomotion_mode=0)
    hold = cur.tick(_motor(lower=0.20), locomotion_mode=0)
    assert hold.clamped
    assert cur.idle.stored
    late = cur.tick(
        _motor(lower=0.20),
        locomotion_mode=0,
        new_qpos=_clip(n=2),
        gen_frame=0,
    )
    assert late.assign_skipped
    assert not late.assigned
    assert cur.idle.stored
    assert late.clamped
    assert late.current_frame == 15


def test_empty_clip_skips() -> None:
    tick = PlannerPlayback().tick(_motor(), locomotion_mode=0)
    assert tick.skipped_reason == "empty_clip"
    assert tick.n_frames == 0


def test_g1_and_hands_refused() -> None:
    cur = PlannerPlayback()
    with pytest.raises(G1CheckpointIncompatible):
        cur.tick(np.zeros(G1_N_DOF), locomotion_mode=0, new_qpos=np.zeros((4, G1_PLANNER_QPOS_DIM)))
    with pytest.raises(PlannerBlendError, match="32"):
        cur.tick(np.zeros(45), locomotion_mode=0, new_qpos=np.zeros((4, 45)))
    with pytest.raises(G1CheckpointIncompatible):
        refuse_run_playback_onnx("planner_sonic.onnx")
    with pytest.raises(PlannerOnnxBlocked, match="25"):
        refuse_run_playback_onnx()
    with pytest.raises(PlaybackError, match="loop"):
        refuse_reference_motion_loop()
    with pytest.raises(PlaybackError, match="MotionCursor"):
        refuse_motion_cursor_mix()
    with pytest.raises(PlannerOnnxError, match="75"):
        refuse_command_schema_as_clip(np.zeros(75))


def test_yaml_flag_required(tmp_path) -> None:
    raw = yaml.safe_load(PLAYBACK_YAML.read_text(encoding="utf-8"))
    raw["not_reference_motion_loop"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PlaybackError, match="not_reference_motion_loop"):
        load_playback_cfg(path)
