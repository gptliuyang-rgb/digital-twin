"""Official SONIC 50 Hz planner last-frame playback cursor.

Numpy only — no simulator, no onnxruntime. Does **not** run
``planner_sonic.onnx``. ADR-047 8-frame blend and ADR-048 idle
ADAPTING/RECOVERING stay the callees. ADR-018 interpolators stay the
runtime L1a.

C++ source (NVlabs/GR00T-WholeBodyControl ``g1_deploy_onnx_ref.cpp``
``CurrentFrameAdvancement``), planner-active branch, end of each 50 Hz
tick:

1. If a new 50 Hz clip is available, 8-frame cross-fade. Success resets
   ``current_frame`` to 0 and clears ``idle_readapt_stored_``.
2. If ``timesteps > 0`` and ``play``: ``new_frame = current_frame + 1``,
   clamp to last, then idle readapt only when ``LocomotionMode::IDLE``.
   ``current_frame = new_frame``.

Named-clip loop-to-zero is the *reference-motion* branch and is refused.
Hands still bypass WBC. ADR-050 readers (encoder look-ahead, planner
context) and ADR-051 (decoder 874-D HardwareHold gather) consume
``current_frame`` from this object; they do not own it.
``MotionCursor`` stays a sibling over motion_lib.
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
    PLANNER_BLEND_FRAMES,
    PLANNER_CONTROL_HZ,
    PLANNER_IDLE_MODE,
    PLANNER_LOOKAHEAD_STEPS_50HZ,
    PLAYBACK_CONTROL_HZ,
    load_t800_sonic,
    planner_qpos_dim,
    t800_planner_qpos_dim,
)
from wbc.idle_readapt import IdleReadapt, IdleReadaptTick
from wbc.planner_blend import BlendResult, cross_fade_qpos
from wbc.planner_onnx import (
    PlannerOnnxBlocked,
    PlannerOnnxError,
    PlannerQpos,
    pack_qpos,
    refuse_g1_planner_onnx,
    refuse_run_planner_onnx,
    unpack_qpos,
)

PLAYBACK_YAML = Path(__file__).with_name("playback.yaml")


class PlaybackError(ValueError):
    """Wrong qpos width, G1 clip, invented loop, or empty-advance mix-up."""


@dataclass(frozen=True)
class PlaybackTick:
    qpos: np.ndarray | None
    q_rad: np.ndarray | None
    current_frame: int
    n_frames: int
    at_last_frame: bool
    clamped: bool
    play: bool
    assigned: bool
    first_copy: bool
    assign_skipped: bool
    idle: IdleReadaptTick | None
    skipped_reason: str | None


def load_playback_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or PLAYBACK_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("playback.yaml must be a mapping")
    flags = (
        "not_dexhand2_contact",
        "not_table_s4",
        "not_hand_mit_ring",
        "not_invented_latency",
        "not_pico_sdk",
        "not_invented_clip",
        "not_g1_planner_onnx",
        "not_interpolator_substitute",
        "not_onnx_run",
        "not_reference_motion_loop",
        "not_wrist_pose_augmentation_sampler",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise PlaybackError(f"playback.yaml must keep {key}: true")
    n = int(raw["n_dof"])
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=n)
    if n != int(load_t800_sonic()["n_revolute"]):
        raise PlaybackError(f"n_dof must stay T800 25, got {n}")
    if int(raw["expected_qpos_dim"]) != planner_qpos_dim(n):
        raise PlaybackError("expected_qpos_dim must be 32")
    if int(raw["g1_qpos_dim_forbidden"]) != G1_PLANNER_QPOS_DIM:
        raise PlaybackError("g1_qpos_dim_forbidden must stay 36")
    if int(raw["g1_n_dof_forbidden"]) != G1_N_DOF:
        raise PlaybackError("g1_n_dof_forbidden must stay 29")
    if int(raw["idle_mode"]) != PLANNER_IDLE_MODE:
        raise PlaybackError("idle_mode must stay 0 (C++ IDLE only)")
    if int(raw["control_hz"]) != PLAYBACK_CONTROL_HZ:
        raise PlaybackError("control_hz must stay 50")
    if int(raw["blend_num_frames"]) != PLANNER_BLEND_FRAMES:
        raise PlaybackError(f"blend_num_frames must stay {PLANNER_BLEND_FRAMES}")
    if int(raw["lookahead_steps_50hz"]) != PLANNER_LOOKAHEAD_STEPS_50HZ:
        raise PlaybackError("lookahead_steps_50hz must stay 2")
    return raw


def clamp_frame(current_frame: int, n_frames: int) -> tuple[int, bool]:
    """C++ ``new_frame = current + 1`` then clamp to ``timesteps - 1``.

    Returns ``(new_frame, clamped)``. Empty clips raise — do not invent a stand.
    """
    n = int(n_frames)
    if n < 1:
        raise PlaybackError("empty clip; do not invent a stand pose")
    if int(current_frame) < 0:
        raise PlaybackError("current_frame must be >= 0")
    new_frame = int(current_frame) + 1
    clamped = new_frame >= n
    if clamped:
        new_frame = n - 1
    return new_frame, clamped


@dataclass
class PlannerPlayback:
    """Stateful 50 Hz planner cursor. Owns ``current_frame`` + last-frame hold."""

    current_frame: int = 0
    qpos: np.ndarray | None = None
    dq: np.ndarray | None = None
    play: bool = True
    idle: IdleReadapt = field(default_factory=IdleReadapt)
    n_dof: int = field(default_factory=lambda: int(load_t800_sonic()["n_revolute"]))

    @property
    def n_frames(self) -> int:
        if self.qpos is None:
            return 0
        return int(self.qpos.shape[0])

    def current_row(self) -> np.ndarray | None:
        if self.qpos is None or self.n_frames < 1:
            return None
        idx = int(np.clip(self.current_frame, 0, self.n_frames - 1))
        return self.qpos[idx].copy()

    def assign(
        self,
        new_qpos: np.ndarray,
        *,
        gen_frame: int | None = None,
        new_dq: np.ndarray | None = None,
    ) -> BlendResult:
        """C++ blend/copy. Success resets the cursor and clears idle storage."""
        n = int(self.n_dof)
        if n == G1_N_DOF:
            refuse_g1_checkpoint(n_dof=n)
        result = cross_fade_qpos(
            self.qpos,
            new_qpos,
            current_frame=self.current_frame,
            gen_frame=gen_frame,
            old_dq=self.dq,
            new_dq=new_dq,
            n_dof=n,
        )
        if result.skipped:
            return result
        self.qpos = result.qpos.copy()
        self.dq = None if result.dq is None else result.dq.copy()
        self.current_frame = int(result.current_frame)
        self.idle.notify_new_clip()
        return result

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
        """One 50 Hz ``CurrentFrameAdvancement``: optional assign, then clamp."""
        if play is not None:
            self.play = bool(play)
        assigned = False
        first_copy = False
        assign_skipped = False
        if new_qpos is not None:
            blend = self.assign(new_qpos, gen_frame=gen_frame, new_dq=new_dq)
            assigned = not blend.skipped
            first_copy = blend.first_copy
            assign_skipped = blend.skipped
        if self.n_frames < 1:
            return PlaybackTick(
                qpos=None,
                q_rad=None,
                current_frame=self.current_frame,
                n_frames=0,
                at_last_frame=False,
                clamped=False,
                play=self.play,
                assigned=assigned,
                first_copy=first_copy,
                assign_skipped=assign_skipped,
                idle=None,
                skipped_reason="empty_clip",
            )
        if not self.play:
            row = self.current_row()
            assert row is not None
            return PlaybackTick(
                qpos=row,
                q_rad=unpack_qpos(row, n_dof=self.n_dof).q_rad.copy(),
                current_frame=self.current_frame,
                n_frames=self.n_frames,
                at_last_frame=self.current_frame >= self.n_frames - 1,
                clamped=False,
                play=False,
                assigned=assigned,
                first_copy=first_copy,
                assign_skipped=assign_skipped,
                idle=None,
                skipped_reason="play_false",
            )
        new_frame, clamped = clamp_frame(self.current_frame, self.n_frames)
        idle_tick: IdleReadaptTick | None = None
        skipped_reason: str | None = None
        if clamped:
            if int(locomotion_mode) != PLANNER_IDLE_MODE:
                skipped_reason = "not_idle"
            elif q_motor is None:
                skipped_reason = "no_motor"
            else:
                last = unpack_qpos(self.qpos[new_frame], n_dof=self.n_dof)
                idle_tick = self.idle.tick(
                    last.q_rad,
                    q_motor,
                    locomotion_mode=int(locomotion_mode),
                    at_last_frame=True,
                    play=True,
                )
                self._write_hinges(new_frame, idle_tick.q_planner)
        self.current_frame = new_frame
        row = self.current_row()
        assert row is not None
        return PlaybackTick(
            qpos=row,
            q_rad=unpack_qpos(row, n_dof=self.n_dof).q_rad.copy(),
            current_frame=self.current_frame,
            n_frames=self.n_frames,
            at_last_frame=self.current_frame >= self.n_frames - 1,
            clamped=clamped,
            play=True,
            assigned=assigned,
            first_copy=first_copy,
            assign_skipped=assign_skipped,
            idle=idle_tick,
            skipped_reason=skipped_reason,
        )

    def _write_hinges(self, frame_index: int, q_rad: np.ndarray) -> None:
        assert self.qpos is not None
        row = unpack_qpos(self.qpos[frame_index], n_dof=self.n_dof)
        self.qpos[frame_index] = pack_qpos(
            PlannerQpos(
                root_pos_m=row.root_pos_m,
                root_rot_wxyz=row.root_rot_wxyz,
                q_rad=np.asarray(q_rad, dtype=np.float64).reshape(-1),
            ),
            n_dof=self.n_dof,
        )


def refuse_reference_motion_loop() -> None:
    """Planner-active path clamps. Named-clip reset-to-zero is a different branch."""
    raise PlaybackError(
        "Planner playback clamps to the last frame. Do not loop named clips "
        "to frame 0 here (that is the C++ reference-motion branch, not the "
        "planner-active CurrentFrameAdvancement path)."
    )


def refuse_run_playback_onnx(path: str | Path | None = None) -> None:
    """Never execute planner weights from the playback cursor."""
    if path is not None:
        refuse_g1_planner_onnx(path)
    try:
        refuse_run_planner_onnx()
    except PlannerOnnxBlocked as exc:
        raise PlannerOnnxBlocked(
            f"{exc} 50 Hz last-frame playback still does not run ONNX. "
            f"T800 hinges are {int(load_t800_sonic()['n_revolute'])}-D; "
            f"qpos is {t800_planner_qpos_dim()}-D; control_hz is {PLANNER_CONTROL_HZ}."
        ) from exc


def refuse_command_schema_as_clip(flat: np.ndarray) -> None:
    """75-D command_schema is not a planner animation. Use ADR-018 for that."""
    arr = np.asarray(flat)
    last = int(arr.shape[-1]) if arr.ndim >= 1 else 0
    if last == G1_PLANNER_QPOS_DIM:
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    raise PlannerOnnxError(
        f"command last-dim {last} is not T800 planner qpos {t800_planner_qpos_dim()}. "
        "ADR-018 interpolators stay the runtime L1a; this cursor only plays 50 Hz qpos."
    )


def refuse_motion_cursor_mix() -> None:
    """Encoder MotionCursor is look-ahead over a motion_lib, not this planner hold."""
    raise PlaybackError(
        "wbc.motion_ref.MotionCursor is the encoder look-ahead (ADR-042). "
        "Do not mix last-frame idle readapt into that cursor. Use PlannerPlayback."
    )
