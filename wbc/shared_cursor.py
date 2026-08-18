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
tick (ADR-052). ``policy_action`` is this tick's 25-D output; it is
stashed and becomes ``last_action`` on the **next** 50 Hz tick
(ADR-053, official ``a_{t-1}``). After that stash, the same ``a_t`` is
ZOH-held onto the 500 Hz PD ring (ADR-054) and the joint-PD plant
evaluates ``τ = Kp(a_t − q) − Kd q̇`` with EngineAI ``pd_stand``
bring-up gains (ADR-055). If a ``JointPdPhysics`` backend is attached,
the same tick applies that τ for 10 physics substeps at 1/500 s
(ADR-056). q/dq are measured from the backend, not interpolated.
This tick's decoder 874-D stays pre-physics. IMU is not invented.
Do not invent policy_action from decoder ONNX or copy clip qpos.
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
from wbc.gather import ObsGather, ObsGatherError
from wbc.motion_ref import MotionCursor, MotionFrame, MotionHold, look_ahead_indices
from wbc.pd_physics import (
    PHYSICS_N_STEPS,
    PHYSICS_TIMESTEP_S,
    JointPdPhysics,
    PdPhysicsPeriod,
    refuse_pd_physics_as_decoder_run,
    refuse_pd_physics_finite_diff_dq,
    refuse_pd_physics_hermite,
    refuse_pd_physics_invent_imu,
    refuse_pd_physics_q_onto_this_tick_decoder,
    run_zoh_period,
)
from wbc.pd_plant import (
    GAINS_SOURCE,
    PolicyPdPlant,
    refuse_pd_plant_as_decoder_run,
    refuse_pd_plant_finite_diff_dq,
    refuse_pd_plant_hermite,
)
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
from wbc.stream import PolicyPdHold, refuse_policy_action_as_decoder_run, refuse_policy_action_hermite

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
        "policy_action_feeds_next_tick_last_action",
        "not_policy_action_same_tick_decoder_obs",
        "not_policy_action_from_decoder_onnx",
        "policy_action_feeds_500hz_pd_after_stash",
        "policy_action_pd_is_zoh",
        "not_policy_action_hermite",
        "pd_plant_on_same_tick",
        "pd_plant_gains_are_pd_stand_bringup",
        "not_sonic_tracking_gains",
        "pd_plant_dq_des_is_zero",
        "not_pd_plant_from_decoder_onnx",
        "not_pd_plant_hermite",
        "not_pd_plant_finite_diff_dq",
        "not_pd_tau_onto_decoder_obs",
        "pd_physics_on_same_tick",
        "pd_physics_is_optional",
        "not_pd_physics_from_decoder_onnx",
        "not_pd_physics_hermite",
        "not_pd_physics_finite_diff_dq",
        "not_pd_physics_q_onto_this_tick_decoder",
        "not_pd_physics_invent_imu",
        "not_pd_physics_from_planner_qpos",
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
    if int(raw["pd_physics_n_steps_per_tick"]) != PHYSICS_N_STEPS:
        raise SharedCursorError("pd_physics_n_steps_per_tick must stay 10")
    if abs(float(raw["pd_physics_timestep_s"]) - PHYSICS_TIMESTEP_S) > 1e-12:
        raise SharedCursorError("pd_physics_timestep_s must stay 0.002")
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


def refuse_policy_action_from_decoder_onnx() -> None:
    """Do not fill policy_action from a fake decoder ONNX vector."""
    raise SharedCursorError(
        "do not fill policy_action from a fake decoder ONNX output. "
        "Caller supplies this tick's 25-D policy action when one exists. "
        "It is stashed and becomes last_action on the next 50 Hz tick. "
        "Until a T800 decoder exists, omit policy_action."
    )


def refuse_policy_action_same_tick_into_obs() -> None:
    """policy_action is a_t. It must not enter this tick's decoder last_action slot."""
    raise SharedCursorError(
        "policy_action is this tick's 25-D output (a_t). Official C++ writes "
        "it into last_action on the *next* 50 Hz gather. Do not push it into "
        "this tick's decoder obs. Pass last_action= for same-tick ADR-052."
    )


def refuse_policy_pd_hermite() -> None:
    """PD q_des is ZOH. Re-export of stream.refuse_policy_action_hermite."""
    refuse_policy_action_hermite()


def refuse_policy_pd_as_decoder_run() -> None:
    """PD q_des is not a decoder ONNX vector."""
    refuse_policy_action_as_decoder_run()


def refuse_policy_pd_plant_hermite() -> None:
    """PD plant τ is ZOH. Re-export of pd_plant.refuse_pd_plant_hermite."""
    refuse_pd_plant_hermite()


def refuse_policy_pd_plant_as_decoder_run() -> None:
    """PD plant τ is not a decoder ONNX vector."""
    refuse_pd_plant_as_decoder_run()


def refuse_policy_pd_plant_finite_diff_dq() -> None:
    """Do not invent 500 Hz dq from a 50 Hz q snapshot."""
    refuse_pd_plant_finite_diff_dq()


def refuse_policy_pd_physics_hermite() -> None:
    """Physics q_des is ZOH. Re-export of pd_physics.refuse_pd_physics_hermite."""
    refuse_pd_physics_hermite()


def refuse_policy_pd_physics_as_decoder_run() -> None:
    """Physics τ is not a decoder ONNX vector."""
    refuse_pd_physics_as_decoder_run()


def refuse_policy_pd_physics_finite_diff_dq() -> None:
    """Do not invent 500 Hz dq from a 50 Hz q snapshot."""
    refuse_pd_physics_finite_diff_dq()


def refuse_policy_pd_physics_q_onto_this_tick_decoder() -> None:
    """Physics q must not rewrite this tick's 874-D."""
    refuse_pd_physics_q_onto_this_tick_decoder()


def refuse_policy_pd_physics_invent_imu() -> None:
    """Do not fill IMU from physics."""
    refuse_pd_physics_invent_imu()


@dataclass(frozen=True)
class DecoderControlTick:
    """One 50 Hz control thread sample: playback + decoder 874-D.

    ``decoder_obs`` is grouped YAML order (token | ω | q | dq | a | g).
    Encoder look-ahead and planner context still read ``current_frame``.
    ``last_action`` is the 25-D vector that entered the ring this tick
    (ADR-052 same-tick, ADR-053 delayed ``a_{t-1}``, or HardwareHold).
    ``policy_action`` is this tick's stashed ``a_t`` (None if omitted).
    ``pd_q_des`` is the 25-D currently ZOH-held on the 500 Hz PD ring
    (ADR-054; ``a_t`` this tick, zeros if never pushed).
    ``pd_tau_nm`` is ``Kp(a_t − q) − Kd q̇`` from EngineAI pd_stand
    bring-up (ADR-055). ``dq_des`` is 0. τ does not enter decoder obs.
    ``pd_physics_*`` is the optional 10-substep plant (ADR-056). n_steps
    is 0 when no backend is attached. Physics q does not rewrite this
    tick's decoder obs.
    """

    playback: PlaybackTick
    decoder_obs: np.ndarray
    current_frame: int
    logger_len: int
    last_action: np.ndarray
    policy_action: np.ndarray | None = None
    pd_q_des: np.ndarray = field(default_factory=lambda: np.zeros(25))
    pd_tau_nm: np.ndarray = field(default_factory=lambda: np.zeros(25))
    pd_gains_source: str = GAINS_SOURCE
    pd_physics_n_steps: int = 0
    pd_physics_q: np.ndarray = field(default_factory=lambda: np.zeros(25))
    pd_physics_tau_nm: np.ndarray = field(default_factory=lambda: np.zeros(25))


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
    invented decoder output). ``policy_action`` is optional so ADR-052
    callers stay valid; it is stashed and applied as ``last_action`` on
    the **next** tick (official ``a_{t-1}``). After that stash the same
    ``a_t`` is ZOH-held onto ``pd_hold`` (ADR-054 500 Hz PD) and the
    joint-PD plant evaluates τ (ADR-055). If ``physics`` is attached,
    the same tick applies that τ for 10 substeps (ADR-056). Do not
    invent a decoder ONNX vector for either slot. ``last_action=``
    does **not** write the PD ring, the plant, or physics.
    """

    playback: PlannerPlayback = field(default_factory=PlannerPlayback)
    gather: ObsGather | None = None
    last_decoder_obs: np.ndarray | None = None
    pd_hold: PolicyPdHold = field(default_factory=PolicyPdHold)
    pd_plant: PolicyPdPlant = field(default_factory=PolicyPdPlant)
    physics: JointPdPhysics | None = None
    last_pd_physics: PdPhysicsPeriod | None = None
    _pending_last_action: np.ndarray | None = None
    _tau_50hz: np.ndarray | None = None

    @property
    def current_frame(self) -> int:
        return int(self.playback.current_frame)

    @property
    def pending_policy_action(self) -> np.ndarray | None:
        """Stashed ``a_t`` that will become ``last_action`` on the next tick."""
        if self._pending_last_action is None:
            return None
        return self._pending_last_action.copy()

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
        policy_action: np.ndarray | None = None,
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
        elif self._pending_last_action is not None:
            self.apply_last_action(self._pending_last_action)
        if token is not None:
            self.assemble_decoder(token, t_s=t_s)
        if policy_action is not None:
            self.stash_policy_action(policy_action)
            self.push_policy_pd(policy_action, t_s=t_s)
        self._maybe_apply_pd_plant()
        self._tau_50hz = self.pd_plant.read()
        self._maybe_step_physics(t_s=t_s)
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

    def stash_policy_action(self, policy_action: np.ndarray) -> np.ndarray:
        """Keep this tick's 25-D output to feed last_action on the next tick.

        Does not write into this tick's decoder obs. Does not run ONNX.
        Does not copy planner clip qpos. Does **not** itself write the
        500 Hz PD ring — ``tick`` / ``control_tick`` push PD *after* this.
        """
        a = validate_t800_last_action(policy_action, n_dof=self.playback.n_dof)
        self._pending_last_action = a
        return a

    def push_policy_pd(self, policy_action: np.ndarray, t_s: float | None = None) -> np.ndarray:
        """ZOH-hold ``a_t`` on the 500 Hz PD ring. Call after stash.

        Does not write decoder last_action. Does not run ONNX. Hands bypass.
        """
        if self.pd_hold.n_dof != int(self.playback.n_dof):
            raise SharedCursorError(
                f"pd_hold n_dof {self.pd_hold.n_dof} != playback {self.playback.n_dof}"
            )
        return self.pd_hold.push(policy_action, t_s=t_s)

    def apply_pd_plant(self) -> np.ndarray:
        """τ = Kp(a_t − q) − Kd q̇ from HardwareHold and PolicyPdHold.

        Does not write decoder last_action. Does not run ONNX. Hands bypass.
        ``last_action=`` does not change q_des. Requires a hardware snapshot.
        """
        if self.gather is None:
            raise SharedCursorError("apply_pd_plant needs an ObsGather")
        if self.pd_plant.n_dof != int(self.playback.n_dof):
            raise SharedCursorError(
                f"pd_plant n_dof {self.pd_plant.n_dof} != playback {self.playback.n_dof}"
            )
        snap = self.gather.hw.read()
        return self.pd_plant.torque(snap.q_hw, snap.dq_hw, self.pd_hold.read())

    def _maybe_apply_pd_plant(self) -> np.ndarray | None:
        if self.gather is None:
            return None
        try:
            return self.apply_pd_plant()
        except ObsGatherError:
            return None

    def step_physics(self, *, t_s: float | None = None) -> PdPhysicsPeriod:
        """Apply ZOH q_des for 10 measured 500 Hz physics substeps.

        Does not rewrite this tick's decoder obs. Does not run ONNX.
        Does not invent IMU. ``last_action=`` does not change q_des.
        After the period, q/dq are pushed into HardwareHold for the
        *next* gather.
        """
        if self.physics is None:
            raise SharedCursorError("step_physics needs a JointPdPhysics backend")
        if self.pd_plant.n_dof != int(self.playback.n_dof):
            raise SharedCursorError(
                f"pd_plant n_dof {self.pd_plant.n_dof} != playback {self.playback.n_dof}"
            )
        if int(self.physics.n_dof) != int(self.playback.n_dof):
            raise SharedCursorError(
                f"physics n_dof {self.physics.n_dof} != playback {self.playback.n_dof}"
            )
        stamp = 0.0 if t_s is None else float(t_s)
        period = run_zoh_period(
            self.pd_plant, self.physics, self.pd_hold.read(), t0_s=stamp
        )
        self.last_pd_physics = period
        if self.gather is not None:
            try:
                self.gather.hw.push_joints(
                    period.q_rad[-1],
                    period.dq_rad_s[-1],
                    n_dof=int(self.playback.n_dof),
                    t_s=float(period.t_s[-1]),
                )
            except ObsGatherError:
                pass
        return period

    def _maybe_step_physics(self, *, t_s: float | None = None) -> PdPhysicsPeriod | None:
        if self.physics is None:
            return None
        return self.step_physics(t_s=t_s)

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
        policy_action: np.ndarray | None = None,
    ) -> DecoderControlTick:
        """Official 50 Hz order: advance playback, bind encoder, last_action, decoder.

        ``play == false`` holds ``current_frame`` (and therefore encoder /
        planner readers) but still logs hardware into the 50 Hz decoder ring.
        ``last_action`` is the previous 25-D policy output (same-tick). Omit
        it to apply a stashed ``policy_action`` from the previous tick, or
        keep HardwareHold if none is stashed (do not invent a decoder ONNX
        vector). ``policy_action`` is this tick's output and is applied as
        ``last_action`` on the next tick. After that stash the same
        ``a_t`` is ZOH-held onto the 500 Hz PD ring and the joint-PD
        plant evaluates τ with pd_stand bring-up gains (not SONIC tracking).
        If ``physics`` is attached, the same tick applies that τ for 10
        substeps at 1/500 s. Physics q does not rewrite this tick's decoder.
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
            policy_action=policy_action,
        )
        if self.last_decoder_obs is None:
            raise SharedCursorError("control_tick did not assemble a decoder vector")
        if self.gather is None:
            raise SharedCursorError("control_tick needs an ObsGather")
        pending = self.pending_policy_action
        phys = None if self.physics is None else self.last_pd_physics
        tau_50 = self._tau_50hz if self._tau_50hz is not None else self.pd_plant.read()
        return DecoderControlTick(
            playback=playback,
            decoder_obs=self.last_decoder_obs,
            current_frame=int(playback.current_frame),
            logger_len=len(self.gather.logger),
            last_action=self.gather.hw.read().last_action.copy(),
            policy_action=pending,
            pd_q_des=self.pd_hold.read(),
            pd_tau_nm=tau_50,
            pd_gains_source=self.pd_plant.gains_source,
            pd_physics_n_steps=0 if phys is None else phys.n_steps,
            pd_physics_q=np.zeros(self.playback.n_dof) if phys is None else phys.q_rad[-1].copy(),
            pd_physics_tau_nm=np.zeros(self.playback.n_dof) if phys is None else phys.tau_nm[-1].copy(),
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
