"""Wire encoder look-ahead and planner context to PlannerPlayback.current_frame.

Numpy only — no simulator, no onnxruntime. Does **not** run encoder or
planner ONNX. Does **not** mix ``MotionCursor`` into ``PlannerPlayback``.

ADR-049 owns ``current_frame``. This module is the shared reader:

* encoder ``MotionHold.cursor`` ← ``playback.current_frame``
  (ADR-042 last-frame-repeat look-ahead)
* ``sample_context_from_50hz(..., current_frame=playback.current_frame)``
  (ADR-046; short windows raise, no last-frame-repeat)

Hands still bypass WBC. ADR-018 interpolators stay the runtime L1a.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import (
    G1_N_DOF,
    G1_PLANNER_QPOS_DIM,
    HISTORY_FRAMES,
    PLANNER_CONTEXT_FRAMES,
    PLANNER_CONTROL_HZ,
    PLANNER_IDLE_MODE,
    PLANNER_LOOKAHEAD_STEPS_50HZ,
    PLAYBACK_CONTROL_HZ,
    load_t800_sonic,
    planner_qpos_dim,
    t800_planner_qpos_dim,
)
from wbc.gather import ObsGather
from wbc.motion_ref import MotionCursor, MotionFrame, MotionHold, look_ahead_indices
from wbc.planner_onnx import (
    PlannerOnnxBlocked,
    PlannerOnnxError,
    sample_context_from_50hz,
    unpack_qpos,
)
from wbc.playback import (
    PlannerPlayback,
    PlaybackTick,
    refuse_motion_cursor_mix,
    refuse_run_playback_onnx,
)

SHARED_CURSOR_YAML = Path(__file__).with_name("shared_cursor.yaml")


class SharedCursorError(ValueError):
    """Empty clip, G1/hands mix-up, or MotionCursor bound into playback."""


def load_shared_cursor_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or SHARED_CURSOR_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("shared_cursor.yaml must be a mapping")
    flags = (
        "not_dexhand2_contact",
        "not_table_s4",
        "not_hand_mit_ring",
        "not_invented_latency",
        "not_pico_sdk",
        "not_invented_clip",
        "not_g1_planner_onnx",
        "not_g1_encoder_onnx",
        "not_interpolator_substitute",
        "not_onnx_run",
        "not_reference_motion_loop",
        "not_motion_cursor_mix",
        "not_wrist_pose_augmentation_sampler",
        "encoder_reads_playback_current_frame",
        "planner_context_reads_playback_current_frame",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise SharedCursorError(f"shared_cursor.yaml must keep {key}: true")
    n = int(raw["n_dof"])
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=n)
    if n != int(load_t800_sonic()["n_revolute"]):
        raise SharedCursorError(f"n_dof must stay T800 25, got {n}")
    if int(raw["expected_qpos_dim"]) != planner_qpos_dim(n):
        raise SharedCursorError("expected_qpos_dim must be 32")
    if int(raw["g1_qpos_dim_forbidden"]) != G1_PLANNER_QPOS_DIM:
        raise SharedCursorError("g1_qpos_dim_forbidden must stay 36")
    if int(raw["g1_n_dof_forbidden"]) != G1_N_DOF:
        raise SharedCursorError("g1_n_dof_forbidden must stay 29")
    if int(raw["idle_mode"]) != PLANNER_IDLE_MODE:
        raise SharedCursorError("idle_mode must stay 0 (C++ IDLE only)")
    if int(raw["control_hz"]) != PLAYBACK_CONTROL_HZ:
        raise SharedCursorError("control_hz must stay 50")
    if int(raw["lookahead_steps_50hz"]) != PLANNER_LOOKAHEAD_STEPS_50HZ:
        raise SharedCursorError("lookahead_steps_50hz must stay 2")
    if int(raw["encoder_look_ahead_n_frames"]) != HISTORY_FRAMES:
        raise SharedCursorError("encoder_look_ahead_n_frames must stay 10")
    if int(raw["encoder_look_ahead_step"]) != 5:
        raise SharedCursorError("encoder_look_ahead_step must stay 5 (default YAML)")
    if int(raw["planner_context_frames"]) != PLANNER_CONTEXT_FRAMES:
        raise SharedCursorError(f"planner_context_frames must stay {PLANNER_CONTEXT_FRAMES}")
    return raw


def qpos_clip_to_motion_frames(
    qpos: np.ndarray,
    *,
    dq: np.ndarray | None = None,
    n_dof: int | None = None,
) -> list[MotionFrame]:
    """Unpack a 50 Hz T800 qpos clip. Missing dq is zeros, not finite-diff."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    arr = np.asarray(qpos, dtype=np.float64)
    if arr.ndim != 2:
        raise SharedCursorError(f"qpos must be (T, 32), got {arr.shape}")
    if arr.shape[1] == G1_PLANNER_QPOS_DIM:
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    expected = planner_qpos_dim(n)
    if arr.shape[1] != expected:
        raise SharedCursorError(
            f"qpos last-dim {arr.shape[1]} != T800 {expected} "
            "(hands bypass WBC; do not concatenate DexHand2 q)"
        )
    if arr.shape[0] < 1:
        raise SharedCursorError("empty clip; do not invent a stand pose")
    d_arr = None
    if dq is not None:
        d_arr = np.asarray(dq, dtype=np.float64)
        if d_arr.shape != (arr.shape[0], n):
            raise SharedCursorError(f"dq shape {d_arr.shape} != ({arr.shape[0]}, {n})")
    frames: list[MotionFrame] = []
    for i in range(arr.shape[0]):
        row = unpack_qpos(arr[i], n_dof=n)
        vel = np.zeros(n, dtype=np.float64) if d_arr is None else d_arr[i].copy()
        frames.append(
            MotionFrame(
                q_ref_rad=row.q_rad.copy(),
                dq_ref_rad_s=vel,
                root_pos_m=row.root_pos_m.copy(),
                root_rot_wxyz=row.root_rot_wxyz.copy(),
            )
        )
    return frames


def bind_encoder_hold(playback: PlannerPlayback, hold: MotionHold) -> MotionHold:
    """Point ``hold.cursor`` at ``playback.current_frame``. Empty clip clears."""
    n = int(playback.n_dof)
    if hold.n_dof != n:
        raise SharedCursorError(f"hold n_dof {hold.n_dof} != playback {n}")
    if playback.n_frames < 1 or playback.qpos is None:
        hold.clear()
        return hold
    frames = qpos_clip_to_motion_frames(playback.qpos, dq=playback.dq, n_dof=n)
    hold.push_sequence(frames, cursor=int(playback.current_frame))
    return hold


def encoder_look_ahead_indices_from_playback(
    playback: PlannerPlayback,
    *,
    n_frames: int = HISTORY_FRAMES,
    step: int = 5,
) -> list[int]:
    """ADR-042 indices starting at the playback cursor. Last frame repeats."""
    if playback.n_frames < 1:
        raise SharedCursorError("empty clip; do not invent encoder look-ahead")
    return look_ahead_indices(int(playback.current_frame), n_frames, step, playback.n_frames)


def sample_planner_context_from_playback(
    playback: PlannerPlayback,
    *,
    look_ahead: int = PLANNER_LOOKAHEAD_STEPS_50HZ,
) -> np.ndarray:
    """ADR-046 context from the same ``current_frame``. Short windows raise."""
    if playback.n_frames < 1 or playback.qpos is None:
        raise PlannerOnnxError("empty 50 Hz motion; do not invent a stand clip")
    frames = qpos_clip_to_motion_frames(playback.qpos, dq=playback.dq, n_dof=playback.n_dof)
    return sample_context_from_50hz(
        frames,
        current_frame=int(playback.current_frame),
        look_ahead=look_ahead,
        n_dof=playback.n_dof,
    )


def refuse_motion_cursor_in_playback() -> None:
    """MotionCursor stays a sibling. Binding it into playback is ADR-049 refusal."""
    refuse_motion_cursor_mix()


def refuse_run_shared_cursor_onnx(path: str | Path | None = None) -> None:
    """Never execute encoder or planner weights from this wiring."""
    try:
        refuse_run_playback_onnx(path)
    except PlannerOnnxBlocked as exc:
        raise PlannerOnnxBlocked(
            f"{exc} Shared cursor still does not run ONNX. "
            f"T800 hinges are {int(load_t800_sonic()['n_revolute'])}-D; "
            f"qpos is {t800_planner_qpos_dim()}-D; control_hz is {PLANNER_CONTROL_HZ}."
        ) from exc


@dataclass
class SharedPlaybackCursor:
    """50 Hz loop: tick playback, then encoder + planner context read that frame.

    ``MotionCursor`` is not stored here. Pass a motion_lib through
    ``MotionCursor`` separately if you need encoder look-ahead over a
    validated clip that is *not* the planner animation.
    """

    playback: PlannerPlayback = field(default_factory=PlannerPlayback)
    gather: ObsGather | None = None

    @property
    def current_frame(self) -> int:
        return int(self.playback.current_frame)

    def tick(
        self,
        q_motor: np.ndarray | None = None,
        *,
        locomotion_mode: int,
        play: bool | None = None,
        new_qpos: np.ndarray | None = None,
        gen_frame: int | None = None,
        new_dq: np.ndarray | None = None,
    ) -> PlaybackTick:
        tick = self.playback.tick(
            q_motor,
            locomotion_mode=locomotion_mode,
            play=play,
            new_qpos=new_qpos,
            gen_frame=gen_frame,
            new_dq=new_dq,
        )
        self.sync()
        return tick

    def sync(self) -> None:
        if self.gather is not None:
            bind_encoder_hold(self.playback, self.gather.motion)

    def assemble_encoder(self, mode: str | None = None) -> np.ndarray:
        if self.gather is None:
            raise SharedCursorError("assemble_encoder needs an ObsGather")
        self.sync()
        return self.gather.assemble_encoder(mode)

    def sample_planner_context(
        self, *, look_ahead: int = PLANNER_LOOKAHEAD_STEPS_50HZ
    ) -> np.ndarray:
        return sample_planner_context_from_playback(self.playback, look_ahead=look_ahead)

    def encoder_look_ahead_indices(self, *, n_frames: int = HISTORY_FRAMES, step: int = 5) -> list[int]:
        return encoder_look_ahead_indices_from_playback(self.playback, n_frames=n_frames, step=step)

    def bind_motion_cursor(self, cursor: MotionCursor) -> None:
        del cursor
        refuse_motion_cursor_in_playback()
