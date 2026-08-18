"""Official SONIC idle-mode ADAPTING / RECOVERING readapt.

Numpy only — no simulator, no onnxruntime. Does **not** run
``planner_sonic.onnx``. ADR-047 8-frame blend stays a sibling.
ADR-018 interpolators stay the runtime L1a.

C++ source (NVlabs/GR00T-WholeBodyControl ``g1_deploy_onnx_ref.cpp``):

* Runs at 50 Hz only after playback clamps to the last planner frame,
  ``operator_state.play``, and ``LocomotionMode::IDLE`` (0).
* First qualifying tick stores lower-body planner targets and sets IDLE.
* ``avg_error`` is mean |q_planner − q_motor| over lower-body hinges.
* IDLE: error > 0.10 → ADAPTING; error < 0.045 → RECOVERING.
* ADAPTING: error < 0.05 → IDLE. Blend ``0.98 q + 0.02 q_motor``.
* RECOVERING: error > 0.10 → ADAPTING. Blend ``0.98 q + 0.02 q_original``.
* No ``kRecoverStop``. Upper body is never written. Hands bypass WBC.

A successful ADR-047 clip assign clears the stored flag
(``idle_readapt_stored_ = false``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import (
    G1_N_DOF,
    IDLE_READAPT_ADAPT_STOP_RAD,
    IDLE_READAPT_ADAPT_TRIGGER_RAD,
    IDLE_READAPT_BLEND_KEEP,
    IDLE_READAPT_BLEND_TO,
    IDLE_READAPT_RECOVER_TRIGGER_RAD,
    PLANNER_IDLE_MODE,
    T800_N_LOWER_BODY_DOF,
    load_t800_sonic,
)
from wbc.planner_onnx import PlannerOnnxBlocked, refuse_g1_planner_onnx, refuse_run_planner_onnx

IDLE_READAPT_YAML = Path(__file__).with_name("idle_readapt.yaml")


class IdleReadaptError(ValueError):
    """Wrong hinge width, G1 29-D, invented recover-stop, or blend mix-in."""


class IdleReadaptState(IntEnum):
    """C++ ``IdleReadaptState``."""

    IDLE = 0
    ADAPTING = 1
    RECOVERING = 2


@dataclass(frozen=True)
class IdleReadaptTick:
    q_planner: np.ndarray
    state: IdleReadaptState
    applied: bool
    stored: bool
    skipped_reason: str | None
    avg_error_rad: float | None
    first_store: bool


def load_idle_readapt_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or IDLE_READAPT_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("idle_readapt.yaml must be a mapping")
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
        "not_planner_blend",
        "not_recover_stop",
        "not_wrist_pose_augmentation_sampler",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise IdleReadaptError(f"idle_readapt.yaml must keep {key}: true")
    n = int(raw["n_dof"])
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=n)
    if n != int(load_t800_sonic()["n_revolute"]):
        raise IdleReadaptError(f"n_dof must stay T800 25, got {n}")
    if int(raw["n_lower_body_dof"]) != T800_N_LOWER_BODY_DOF:
        raise IdleReadaptError("n_lower_body_dof must stay 12 (J00–J11)")
    if int(raw["g1_n_dof_forbidden"]) != G1_N_DOF:
        raise IdleReadaptError("g1_n_dof_forbidden must stay 29")
    if int(raw["idle_mode"]) != PLANNER_IDLE_MODE:
        raise IdleReadaptError("idle_mode must stay 0 (C++ IDLE only)")
    if float(raw["k_adapt_trigger_rad"]) != IDLE_READAPT_ADAPT_TRIGGER_RAD:
        raise IdleReadaptError("k_adapt_trigger_rad must stay 0.10")
    if float(raw["k_adapt_stop_rad"]) != IDLE_READAPT_ADAPT_STOP_RAD:
        raise IdleReadaptError("k_adapt_stop_rad must stay 0.05")
    if float(raw["k_recover_trigger_rad"]) != IDLE_READAPT_RECOVER_TRIGGER_RAD:
        raise IdleReadaptError("k_recover_trigger_rad must stay 0.045")
    if float(raw["blend_keep"]) != IDLE_READAPT_BLEND_KEEP:
        raise IdleReadaptError("blend_keep must stay 0.98")
    if float(raw["blend_to"]) != IDLE_READAPT_BLEND_TO:
        raise IdleReadaptError("blend_to must stay 0.02")
    if list(raw["lower_body_slice"]) != [0, T800_N_LOWER_BODY_DOF]:
        raise IdleReadaptError("lower_body_slice must be [0, 12]")
    if "k_recover_stop_rad" in raw:
        raise IdleReadaptError("do not invent k_recover_stop_rad")
    return raw


def lower_body_indices(*, n_dof: int | None = None) -> np.ndarray:
    """T800 J00–J11. Same count as G1 lower body; not a G1 index copy."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=n)
    return np.arange(T800_N_LOWER_BODY_DOF, dtype=np.int64)


def mean_abs_lower_body_error(q_planner: np.ndarray, q_motor: np.ndarray) -> float:
    """C++ ``avg_error``: mean |planner − motor| on lower-body hinges."""
    a = _as_t800_hinges(q_planner)
    b = _as_t800_hinges(q_motor)
    idx = lower_body_indices()
    return float(np.mean(np.abs(a[idx] - b[idx])))


def transition(state: IdleReadaptState, avg_error_rad: float) -> IdleReadaptState:
    """C++ switch. Strict ``>`` / ``<``. No recover-stop."""
    err = float(avg_error_rad)
    if not np.isfinite(err):
        raise IdleReadaptError("avg_error contains NaN/Inf")
    if state is IdleReadaptState.IDLE:
        if err > IDLE_READAPT_ADAPT_TRIGGER_RAD:
            return IdleReadaptState.ADAPTING
        if err < IDLE_READAPT_RECOVER_TRIGGER_RAD:
            return IdleReadaptState.RECOVERING
        return IdleReadaptState.IDLE
    if state is IdleReadaptState.ADAPTING:
        if err < IDLE_READAPT_ADAPT_STOP_RAD:
            return IdleReadaptState.IDLE
        return IdleReadaptState.ADAPTING
    if state is IdleReadaptState.RECOVERING:
        if err > IDLE_READAPT_ADAPT_TRIGGER_RAD:
            return IdleReadaptState.ADAPTING
        return IdleReadaptState.RECOVERING
    raise IdleReadaptError(f"unknown idle-readapt state {state!r}")


def blend_last_frame(
    q_planner: np.ndarray,
    *,
    state: IdleReadaptState,
    q_motor: np.ndarray,
    q_original: np.ndarray,
) -> np.ndarray:
    """Write only lower-body hinges. Upper body and root are not in this vector."""
    out = _as_t800_hinges(q_planner).copy()
    if state is IdleReadaptState.IDLE:
        return out
    idx = lower_body_indices()
    keep = IDLE_READAPT_BLEND_KEEP
    to = IDLE_READAPT_BLEND_TO
    if abs(keep + to - 1.0) > 1e-12:
        raise IdleReadaptError("blend_keep + blend_to must stay 1")
    if state is IdleReadaptState.ADAPTING:
        src = _as_t800_hinges(q_motor)
        out[idx] = keep * out[idx] + to * src[idx]
        return out
    if state is IdleReadaptState.RECOVERING:
        src = _as_t800_hinges(q_original)
        out[idx] = keep * out[idx] + to * src[idx]
        return out
    raise IdleReadaptError(f"unknown idle-readapt state {state!r}")


def _as_t800_hinges(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    n = int(load_t800_sonic()["n_revolute"])
    if arr.size == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if arr.size == n + 20:
        raise IdleReadaptError(
            f"hinge dim {arr.size} includes DexHand2 (20). Hands bypass WBC."
        )
    if arr.size != n:
        raise IdleReadaptError(
            f"hinge dim {arr.size} != T800 {n} "
            "(this path is last-frame JointPositions, not 32-D qpos)"
        )
    if not np.isfinite(arr).all():
        raise IdleReadaptError("hinges contain NaN/Inf")
    return arr


@dataclass
class IdleReadapt:
    """Stateful last-frame hold. ``notify_new_clip`` matches C++ stored=false."""

    state: IdleReadaptState = IdleReadaptState.IDLE
    stored: bool = False
    original_lower: np.ndarray | None = None

    def notify_new_clip(self) -> None:
        """C++ ``idle_readapt_stored_ = false`` after a successful clip assign."""
        self.stored = False
        self.original_lower = None

    def tick(
        self,
        q_planner: np.ndarray,
        q_motor: np.ndarray,
        *,
        locomotion_mode: int,
        at_last_frame: bool,
        play: bool = True,
    ) -> IdleReadaptTick:
        planner = _as_t800_hinges(q_planner)
        motor = _as_t800_hinges(q_motor)
        if not play:
            return self._skip(planner, "play_false")
        if not at_last_frame:
            return self._skip(planner, "not_last_frame")
        if int(locomotion_mode) != PLANNER_IDLE_MODE:
            return self._skip(planner, "not_idle")
        first_store = False
        if not self.stored:
            idx = lower_body_indices()
            self.original_lower = planner[idx].copy()
            self.stored = True
            self.state = IdleReadaptState.IDLE
            first_store = True
        if self.original_lower is None:
            raise IdleReadaptError("stored flag set without original targets")
        avg = mean_abs_lower_body_error(planner, motor)
        self.state = transition(self.state, avg)
        original = planner.copy()
        original[lower_body_indices()] = self.original_lower
        out = blend_last_frame(
            planner, state=self.state, q_motor=motor, q_original=original
        )
        applied = self.state is not IdleReadaptState.IDLE
        return IdleReadaptTick(
            q_planner=out,
            state=self.state,
            applied=applied,
            stored=True,
            skipped_reason=None,
            avg_error_rad=avg,
            first_store=first_store,
        )

    def _skip(self, planner: np.ndarray, reason: str) -> IdleReadaptTick:
        return IdleReadaptTick(
            q_planner=planner.copy(),
            state=self.state,
            applied=False,
            stored=self.stored,
            skipped_reason=reason,
            avg_error_rad=None,
            first_store=False,
        )


def refuse_recover_stop() -> None:
    """C++ has no kRecoverStop. RECOVERING does not fall through to IDLE."""
    raise IdleReadaptError(
        "There is no kRecoverStop. RECOVERING stays RECOVERING until "
        "avg_error > kAdaptTrigger (0.10 rad) moves it to ADAPTING."
    )


def refuse_planner_blend_mix() -> None:
    """8-frame clip cross-fade is ADR-047, not this last-frame hold."""
    raise IdleReadaptError(
        "Idle ADAPTING/RECOVERING is not the 8-frame planner blend. "
        "Do not mix kAdaptTrigger into wbc/planner_blend.py."
    )


def refuse_run_idle_readapt_onnx(path: str | Path | None = None) -> None:
    """Never execute planner weights from the idle-readapt module."""
    if path is not None:
        refuse_g1_planner_onnx(path)
    try:
        refuse_run_planner_onnx()
    except PlannerOnnxBlocked as exc:
        n = int(load_t800_sonic()["n_revolute"])
        raise PlannerOnnxBlocked(
            f"{exc} Idle-mode ADAPTING/RECOVERING still does not run ONNX. "
            f"T800 hinges are {n}-D; lower-body slice is {T800_N_LOWER_BODY_DOF}."
        ) from exc
