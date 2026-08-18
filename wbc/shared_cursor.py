"""Wire encoder look-ahead, planner context, and decoder gather to one 50 Hz tick.

Numpy only — no simulator, no onnxruntime. Does **not** run encoder,
planner, or decoder ONNX. Does **not** mix ``MotionCursor`` into
``PlannerPlayback``. Does **not** copy planner clip qpos into the
decoder vector.

ADR-049 owns ``current_frame``. This module is the shared reader:

* encoder ``MotionHold.cursor`` ← ``playback.current_frame``
  (ADR-042 last-frame-repeat look-ahead)
* ``sample_context_from_50hz(..., current_frame=playback.current_frame)``
  (ADR-046; short windows raise, no last-frame-repeat)
* ``ObsGather.control_tick`` ← same 50 Hz tick, ``HardwareHold``
  (ADR-041 / paper S7; 874-D grouped history)

Decoder history is robot proprioception. Token is an external 64-D —
do not invent it from ONNX. ``last_action`` is a caller-supplied 25-D
previous policy output pushed into ``HardwareHold`` on the same 50 Hz
tick (ADR-052). Do not invent it from decoder ONNX or copy clip qpos.
Hands still bypass WBC. ADR-018 interpolators stay the runtime L1a.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import (
    G1_DECODER_INPUT_DIM,
    G1_N_DOF,
    G1_PLANNER_QPOS_DIM,
    HISTORY_FRAMES,
    PLANNER_CONTEXT_FRAMES,
    PLANNER_CONTROL_HZ,
    PLANNER_IDLE_MODE,
    PLANNER_LOOKAHEAD_STEPS_50HZ,
    PLAYBACK_CONTROL_HZ,
    TOKEN_DIM,
    decoder_history_dim,
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
        "not_g1_decoder_onnx",
        "not_decoder_from_planner_qpos",
        "encoder_reads_playback_current_frame",
        "planner_context_reads_playback_current_frame",
        "decoder_assembles_on_shared_tick",
        "decoder_history_is_hardware_not_clip",
        "last_action_is_caller_supplied_policy",
        "not_last_action_from_decoder_onnx",
        "not_last_action_from_planner_qpos",
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
    if int(raw["token_dim"]) != TOKEN_DIM:
        raise SharedCursorError(f"token_dim must stay {TOKEN_DIM}")
    expected_dec = decoder_history_dim(n)
    if int(raw["expected_decoder_dim"]) != expected_dec:
        raise SharedCursorError(f"expected_decoder_dim must stay {expected_dec}")
    if int(raw["g1_decoder_dim_forbidden"]) != G1_DECODER_INPUT_DIM:
        raise SharedCursorError("g1_decoder_dim_forbidden must stay 994")
    if int(raw["action_dim"]) != n:
        raise SharedCursorError(f"action_dim must stay T800 {n}, got {raw['action_dim']}")
    if int(raw["g1_action_dim_forbidden"]) != G1_N_DOF:
        raise SharedCursorError("g1_action_dim_forbidden must stay 29")
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
    """Never execute encoder, planner, or decoder weights from this wiring."""
    if path is not None:
        lowered = str(path).lower().replace("\\", "/")
        if "model_decoder" in lowered or "decoder.onnx" in lowered:
            refuse_g1_decoder_onnx(path)
    try:
        refuse_run_playback_onnx(path)
    except PlannerOnnxBlocked as exc:
        raise PlannerOnnxBlocked(
            f"{exc} Shared cursor still does not run ONNX. "
            f"T800 hinges are {int(load_t800_sonic()['n_revolute'])}-D; "
            f"qpos is {t800_planner_qpos_dim()}-D; decoder is "
            f"{decoder_history_dim(int(load_t800_sonic()['n_revolute']))}-D; "
            f"control_hz is {PLANNER_CONTROL_HZ}."
        ) from exc


def refuse_g1_decoder_onnx(path: str | Path | None = None) -> None:
    """Official model_decoder.onnx is Unitree G1 994-D. Do not load it on T800."""
    label = str(path) if path is not None else "model_decoder.onnx"
    n = int(load_t800_sonic()["n_revolute"])
    raise G1CheckpointIncompatible(
        f"{label} is a G1 decoder (ONNX input {G1_DECODER_INPUT_DIM}-D). "
        f"T800 decoder input is {decoder_history_dim(n)}-D. Retrain; do not load G1 ONNX."
    )


def refuse_decoder_from_planner_qpos() -> None:
    """Decoder 874-D is HardwareHold proprioception, not clip hinges."""
    raise SharedCursorError(
        "decoder 874-D is HardwareHold proprioception (paper S7), not planner "
        "clip qpos. Do not copy playback.qpos hinges into ObsGather.control_tick. "
        "Token_state stays an external 64-D; do not invent it from ONNX."
    )


def decoder_joint_history(obs: np.ndarray, *, n_dof: int = 25) -> np.ndarray:
    """Grouped YAML q-history block: shape (10, n_dof). Token/ω sit in front."""
    n = int(n_dof)
    vec = np.asarray(obs, dtype=np.float64).reshape(-1)
    expected = decoder_history_dim(n)
    if vec.shape == (G1_DECODER_INPUT_DIM,):
        refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
    if vec.shape != (expected,):
        raise SharedCursorError(f"decoder obs dim {vec.shape} != T800 {expected}")
    start = TOKEN_DIM + 3 * HISTORY_FRAMES
    return vec[start : start + n * HISTORY_FRAMES].reshape(HISTORY_FRAMES, n)


def decoder_last_action_history(obs: np.ndarray, *, n_dof: int = 25) -> np.ndarray:
    """Grouped YAML last-action history: shape (10, n_dof). After ω/q/dq."""
    n = int(n_dof)
    vec = np.asarray(obs, dtype=np.float64).reshape(-1)
    expected = decoder_history_dim(n)
    if vec.shape == (G1_DECODER_INPUT_DIM,):
        refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
    if vec.shape != (expected,):
        raise SharedCursorError(f"decoder obs dim {vec.shape} != T800 {expected}")
    start = TOKEN_DIM + 3 * HISTORY_FRAMES + 2 * n * HISTORY_FRAMES
    return vec[start : start + n * HISTORY_FRAMES].reshape(HISTORY_FRAMES, n)


def validate_t800_last_action(last_action: np.ndarray, *, n_dof: int = 25) -> np.ndarray:
    """T800 PPO/decoder action is 25-D. G1 29-D, planner qpos, and hands refused."""
    n = int(n_dof)
    a = np.asarray(last_action, dtype=np.float64).reshape(-1)
    if a.shape == (G1_N_DOF,):
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if a.shape == (G1_PLANNER_QPOS_DIM,):
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    if a.shape == (planner_qpos_dim(n),):
        raise SharedCursorError(
            f"last_action dim {a.shape[0]} is planner qpos, not the {n}-D policy "
            "action. Do not copy playback.qpos into HardwareHold.last_action."
        )
    if a.shape[0] == 45:
        raise SharedCursorError(
            "last_action dim 45 looks like DexHand2 concat; hands bypass WBC"
        )
    if a.shape != (n,):
        raise SharedCursorError(
            f"last_action dim {a.shape} != T800 {n} (PPO/decoder action). "
            "Do not invent a decoder ONNX output."
        )
    if not np.isfinite(a).all():
        raise SharedCursorError("last_action contains NaN/Inf")
    return a.copy()


def refuse_last_action_from_planner_qpos() -> None:
    """last_action is the previous 25-D policy output, not clip hinges."""
    raise SharedCursorError(
        "last_action is the previous 25-D policy output (PPO/decoder), not "
        "planner clip qpos. Do not copy playback.qpos hinges into "
        "HardwareHold.last_action."
    )


def refuse_last_action_from_decoder_onnx() -> None:
    """Do not fill last_action from a fake decoder ONNX vector."""
    raise SharedCursorError(
        "do not fill last_action from a fake decoder ONNX output. "
        "Caller supplies the previous 25-D policy action when one exists. "
        "Until a T800 decoder exists, omit last_action and keep HardwareHold."
    )


@dataclass(frozen=True)
class DecoderControlTick:
    """One 50 Hz control thread sample: playback + decoder 874-D.

    ``decoder_obs`` is grouped YAML order (token | ω | q | dq | a | g).
    Encoder look-ahead and planner context still read ``current_frame``.
    ``last_action`` is the 25-D vector that entered the ring this tick
    (caller-supplied policy output, or HardwareHold's existing value).
    """

    playback: PlaybackTick
    decoder_obs: np.ndarray
    current_frame: int
    logger_len: int
    last_action: np.ndarray


@dataclass
class SharedPlaybackCursor:
    """50 Hz loop: tick playback, then encoder / planner / decoder share that clock.

    ``MotionCursor`` is not stored here. Pass a motion_lib through
    ``MotionCursor`` separately if you need encoder look-ahead over a
    validated clip that is *not* the planner animation.

    Decoder history comes from ``HardwareHold``, not from the planner clip.
    ``tick(..., token=)`` is optional so ADR-050 callers stay valid; the
    official combined entry is ``control_tick`` (token required).
    ``last_action`` is optional so ADR-051 callers stay valid; omit it to
    keep HardwareHold's existing 25-D (startup zeros are padding, not an
    invented decoder output).
    """

    playback: PlannerPlayback = field(default_factory=PlannerPlayback)
    gather: ObsGather | None = None
    last_decoder_obs: np.ndarray | None = None

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
        token: np.ndarray | None = None,
        t_s: float | None = None,
        last_action: np.ndarray | None = None,
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
        if last_action is not None:
            self.apply_last_action(last_action)
        if token is not None:
            self.assemble_decoder(token, t_s=t_s)
        return tick

    def apply_last_action(self, last_action: np.ndarray) -> np.ndarray:
        """Push a caller-supplied 25-D policy action into HardwareHold.

        Does not run decoder ONNX. Does not copy planner clip qpos.
        Requires a hardware snapshot already on the hold.
        """
        if self.gather is None:
            raise SharedCursorError("apply_last_action needs an ObsGather")
        a = validate_t800_last_action(last_action, n_dof=self.playback.n_dof)
        self.gather.hw.push_last_action(a, n_dof=int(self.playback.n_dof))
        return a

    def sync(self) -> None:
        if self.gather is not None:
            bind_encoder_hold(self.playback, self.gather.motion)

    def assemble_decoder(self, token: np.ndarray, *, t_s: float | None = None) -> np.ndarray:
        """Paper S7 874-D from HardwareHold. Does not read clip qpos."""
        if self.gather is None:
            raise SharedCursorError("assemble_decoder needs an ObsGather")
        tok = np.asarray(token, dtype=np.float64).reshape(-1)
        if tok.shape != (TOKEN_DIM,):
            raise SharedCursorError(
                f"token dim {tok.shape} != {TOKEN_DIM}. Token_state is external; "
                "do not invent a 64-D vector from decoder ONNX."
            )
        obs = self.gather.control_tick(tok, t_s=t_s)
        if obs.shape == (G1_DECODER_INPUT_DIM,):
            refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
        self.last_decoder_obs = obs
        return obs

    def control_tick(
        self,
        q_motor: np.ndarray | None = None,
        *,
        locomotion_mode: int,
        token: np.ndarray,
        play: bool | None = None,
        new_qpos: np.ndarray | None = None,
        gen_frame: int | None = None,
        new_dq: np.ndarray | None = None,
        t_s: float | None = None,
        last_action: np.ndarray | None = None,
    ) -> DecoderControlTick:
        """Official 50 Hz order: advance playback, bind encoder, last_action, decoder.

        ``play == false`` holds ``current_frame`` (and therefore encoder /
        planner readers) but still logs hardware into the 50 Hz decoder ring.
        ``last_action`` is the previous 25-D policy output. Omit it to keep
        HardwareHold (do not invent a decoder ONNX vector).
        """
        playback = self.tick(
            q_motor,
            locomotion_mode=locomotion_mode,
            play=play,
            new_qpos=new_qpos,
            gen_frame=gen_frame,
            new_dq=new_dq,
            token=token,
            t_s=t_s,
            last_action=last_action,
        )
        if self.last_decoder_obs is None:
            raise SharedCursorError("control_tick did not assemble a decoder vector")
        if self.gather is None:
            raise SharedCursorError("control_tick needs an ObsGather")
        return DecoderControlTick(
            playback=playback,
            decoder_obs=self.last_decoder_obs,
            current_frame=int(playback.current_frame),
            logger_len=len(self.gather.logger),
            last_action=self.gather.hw.read().last_action.copy(),
        )

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
