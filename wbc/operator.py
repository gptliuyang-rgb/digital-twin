"""SONIC §3.5 100 Hz operator-input loop. Numpy only — no simulator.

He et al., arXiv:2511.07820v3 §3.5 four concurrent loops: planner 10 Hz,
policy 50 Hz, operator input 100 Hz, command stream 500 Hz. This module is
the 100 Hz VR / gamepad / keyboard / network ring.

Paper: switching interfaces changes the active encoder; no retraining.
VLA (5–10 Hz) is not an operator source — it goes through L1a (ADR-018).
Hands still bypass WBC; they ride this clock so the DexHand2 1 kHz MIT
backend can subsample. ``nav_cmd`` is hold-last onto 500 Hz (ADR-038/039);
do not Hermite vx/vy/wz.

Not Table S4. Not pad–cardboard. Not a T800+Hand weld (ADR-009).
Not a PICO / CloudXR / Isaac Teleop import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import CommandVector, HandSpec, command_dim, command_layout, load_hand_spec
from wbc.dims import load_t800_sonic
from wbc.planner import PlannedRef
from wbc.spring import RootSpringRef, RootSpringState
from wbc.stream import (
    OPERATOR_INPUT_HZ,
    PLANNER_HZ,
    POLICY_HZ,
    STREAM_HZ,
    CommandStream,
    CommandStreamError,
    stream_commands,
    stream_nav_spring,
)
from wbc.teleop import (
    TELEOP_3POINT,
    TELEOP_5POINT,
    FivePointCommand,
    command_to_vr_3point,
    command_to_vr_5point,
    pack_hybrid_encoder_cmd,
    refuse_teleop_mode_mismatch,
)

OPERATOR_YAML = Path(__file__).with_name("operator.yaml")
ALLOWED_SOURCES = ("keyboard", "gamepad", "vr_3point", "vr_5point", "network")
FORBIDDEN_SOURCES = ("vla", "groot", "pi05", "policy")


class OperatorInputError(ValueError):
    """Wrong rate, source, dimension, or non-aligned downsample."""


def load_operator_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or OPERATOR_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("operator.yaml must be a mapping")
    if raw.get("not_dexhand2_contact") is not True:
        raise ValueError("operator.yaml must keep not_dexhand2_contact: true")
    if raw.get("not_table_s4") is not True:
        raise ValueError("operator.yaml must keep not_table_s4: true")
    if raw.get("not_hand_mit_ring") is not True:
        raise ValueError("operator.yaml must keep not_hand_mit_ring: true")
    if raw.get("not_pico_sdk") is not True:
        raise ValueError("operator.yaml must keep not_pico_sdk: true")
    if int(raw["operator_input_hz"]) != OPERATOR_INPUT_HZ:
        raise ValueError("operator_input_hz must stay 100 (SONIC §3.5)")
    if int(raw["command_stream_hz"]) != STREAM_HZ:
        raise ValueError("command_stream_hz must stay 500 (SONIC §3.5)")
    if int(raw["policy_hz"]) != POLICY_HZ:
        raise ValueError("policy_hz must stay 50 (SONIC §3.5)")
    if int(raw["planner_hz"]) != PLANNER_HZ:
        raise ValueError("planner_hz must stay 10 (SONIC §3.5)")
    if int(raw["factor_operator_to_stream"]) != STREAM_HZ // OPERATOR_INPUT_HZ:
        raise ValueError("factor_operator_to_stream must stay 5")
    if int(raw["factor_operator_to_policy"]) != OPERATOR_INPUT_HZ // POLICY_HZ:
        raise ValueError("factor_operator_to_policy must stay 2")
    if int(raw["factor_operator_to_planner"]) != OPERATOR_INPUT_HZ // PLANNER_HZ:
        raise ValueError("factor_operator_to_planner must stay 10")
    if str(raw.get("nav_resample")) != "hold_last":
        raise ValueError("nav_resample must stay hold_last (do not Hermite nav_cmd)")
    allowed = tuple(raw["sources"]["allowed"])
    forbidden = tuple(raw["sources"]["forbidden"])
    if allowed != ALLOWED_SOURCES:
        raise ValueError(f"sources.allowed drifted: {allowed}")
    if forbidden != FORBIDDEN_SOURCES:
        raise ValueError(f"sources.forbidden drifted: {forbidden}")
    sonic = load_t800_sonic()
    if int(sonic["operator_input_hz"]) != OPERATOR_INPUT_HZ:
        raise ValueError("t800_sonic.yaml operator_input_hz drifted from 100")
    return raw


def refuse_operator_source(source: str) -> str:
    """Paper interfaces only. VLA is 5–10 Hz and must not enter this ring."""
    key = str(source).strip().replace("-", "_").lower()
    if key in ("3point", "vr3point"):
        key = "vr_3point"
    if key in ("5point", "vr5point"):
        key = "vr_5point"
    if key in FORBIDDEN_SOURCES:
        raise OperatorInputError(
            f"source={source!r} is a VLA / policy clock, not the 100 Hz operator ring. "
            "Send those chunks through L1a (ADR-018) then the 500 Hz PD stream (ADR-039)."
        )
    if key not in ALLOWED_SOURCES:
        raise OperatorInputError(
            f"source={source!r} is not in {ALLOWED_SOURCES}. "
            "Paper §3.5 switches keyboard / gamepad / VR / network by encoder, not by inventing a rate."
        )
    return key


def downsample_stride(src_hz: int, dst_hz: int) -> int:
    """Integer stride from a faster grid onto a slower one (100→50 = 2, 100→10 = 10)."""
    src = int(src_hz)
    dst = int(dst_hz)
    if src < 1 or dst < 1:
        raise OperatorInputError("src_hz and dst_hz must be ≥ 1")
    if src < dst:
        raise OperatorInputError(
            f"downsample_stride expects src_hz ≥ dst_hz, got {src}→{dst}. "
            "Upsample with stream_factor instead."
        )
    if src % dst != 0:
        raise OperatorInputError(
            f"src_hz={src} is not an integer multiple of dst_hz={dst}. "
            "Refuse a rounded downsample."
        )
    return src // dst


def _require_uniform_grid(t_s: np.ndarray, *, hz: int) -> np.ndarray:
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    if t.shape[0] < 2:
        raise OperatorInputError("operator window needs ≥2 samples")
    dt = np.diff(t)
    expected = 1.0 / float(hz)
    if not np.allclose(dt, expected, atol=1e-9, rtol=0.0):
        raise OperatorInputError(
            f"operator samples must sit on a uniform {hz} Hz grid (dt={expected} s)"
        )
    return t


def _require_command_rows(commands: np.ndarray, spec: HandSpec) -> np.ndarray:
    arr = np.asarray(commands, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise OperatorInputError(f"commands must be (N, D), got {arr.shape}")
    expected = command_dim(spec)
    if arr.shape[1] != expected:
        raise OperatorInputError(
            "100 Hz operator ring requires command_schema_v1 rows. "
            f"got dim {arr.shape[1]} (expected {expected}). "
            "If this is Case A, call CaseAToCommandSchema.convert(..., apply_fk=True) first."
        )
    if arr.shape[0] < 2:
        raise OperatorInputError("operator window needs ≥2 samples")
    return arr


def hold_last(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Left-neighbour hold. ``y_src`` is (N,) or (N, D). Not a spline."""
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    squeeze = False
    if y_src.ndim == 1:
        y_src = y_src.reshape(-1, 1)
        squeeze = True
    if t_src.shape[0] != y_src.shape[0]:
        raise OperatorInputError("t_src / y_src length mismatch")
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, t_src.shape[0] - 1)
    out = y_src[idx]
    return out[:, 0] if squeeze else out


@dataclass(frozen=True)
class OperatorWindow:
    """Uniform 100 Hz operator samples. Simulator-free."""

    t_s: np.ndarray
    commands: np.ndarray  # (T, command_dim)
    elbows: np.ndarray | None
    rate_hz: int
    source: str
    horizon_s: float

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


def ingest_operator(
    commands: np.ndarray,
    *,
    source: str,
    t_s: np.ndarray | None = None,
    elbows: np.ndarray | None = None,
    spec: HandSpec | None = None,
    hz: int = OPERATOR_INPUT_HZ,
) -> OperatorWindow:
    """Accept a batch of 100 Hz command_schema rows. Refuse VLA and Case A."""
    src = refuse_operator_source(source)
    spec = spec or load_hand_spec()
    arr = _require_command_rows(commands, spec)
    if t_s is None:
        t = np.arange(arr.shape[0], dtype=np.float64) / float(hz)
    else:
        t = _require_uniform_grid(t_s, hz=hz)
        if t.shape[0] != arr.shape[0]:
            raise OperatorInputError("t_s / commands length mismatch")
    elbow_out = None
    if elbows is not None:
        e = np.asarray(elbows, dtype=np.float64)
        if e.ndim != 2 or e.shape[0] != arr.shape[0] or e.shape[1] != 6:
            raise OperatorInputError(f"elbows must be (N, 6), got {e.shape}")
        elbow_out = e
        if src != "vr_5point":
            raise OperatorInputError(
                f"source={src} cannot carry 5-point elbows. "
                "A 5-point hybrid encoder is a retrain (ADR-017)."
            )
    elif src == "vr_5point":
        raise OperatorInputError("vr_5point operator window requires elbows (N, 6)")
    return OperatorWindow(
        t_s=t,
        commands=arr,
        elbows=elbow_out,
        rate_hz=int(hz),
        source=src,
        horizon_s=float(t[-1] - t[0]),
    )


def stride_window(window: OperatorWindow, *, dst_hz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Aligned subset of a 100 Hz window onto 50 Hz or 10 Hz. No rotation average."""
    if int(window.rate_hz) != OPERATOR_INPUT_HZ:
        raise OperatorInputError(f"stride_window expects 100 Hz, got {window.rate_hz} Hz")
    stride = downsample_stride(window.rate_hz, dst_hz)
    t = window.t_s[::stride]
    cmd = window.commands[::stride]
    elbows = None if window.elbows is None else window.elbows[::stride]
    if t.shape[0] < 2:
        raise OperatorInputError(
            f"downsample 100→{dst_hz} Hz needs ≥2 destination samples "
            f"(window has {window.n_steps} operator ticks)"
        )
    return t, cmd, elbows


def operator_to_stream(
    window: OperatorWindow,
    *,
    spec: HandSpec | None = None,
    pos_interp: str = "cubic_hermite",
) -> CommandStream:
    """100 Hz operator window → 500 Hz PD ring. Poses spline; nav_cmd holds."""
    if int(window.rate_hz) != OPERATOR_INPUT_HZ:
        raise OperatorInputError(
            f"operator_to_stream expects a 100 Hz window, got {window.rate_hz} Hz"
        )
    spec = spec or load_hand_spec()
    try:
        streamed = stream_commands(
            window.commands,
            window.t_s,
            src_hz=OPERATOR_INPUT_HZ,
            spec=spec,
            pos_interp=pos_interp,
            elbows=window.elbows,
        )
    except CommandStreamError as exc:
        raise OperatorInputError(str(exc)) from exc
    layout = command_layout(spec)
    lo, hi = layout["nav_cmd"]
    streamed.commands[:, lo:hi] = hold_last(window.t_s, window.commands[:, lo:hi], streamed.t_s)
    return streamed


def operator_to_policy_tokens(window: OperatorWindow) -> np.ndarray:
    """100 Hz → 50 Hz by taking every 2nd aligned sample."""
    _, cmd, _ = stride_window(window, dst_hz=POLICY_HZ)
    return cmd


def operator_to_planner(window: OperatorWindow) -> PlannedRef:
    """100 Hz → 10 Hz L1a waypoints by taking every 10th aligned sample."""
    t, cmd, elbows = stride_window(window, dst_hz=PLANNER_HZ)
    return PlannedRef(
        t_s=t,
        commands=cmd,
        elbows=elbows,
        rate_hz=PLANNER_HZ,
        horizon_s=float(t[-1] - t[0]),
    )


def operator_to_hybrid_tokens(window: OperatorWindow, *, mode: str | None = None) -> np.ndarray:
    """50 Hz hybrid-encoder tokens. Hands never enter the token (ADR-012)."""
    src_mode = TELEOP_5POINT if window.source == "vr_5point" else TELEOP_3POINT
    run_mode = mode or src_mode
    refuse_teleop_mode_mismatch(src_mode, run_mode)
    rows = operator_to_policy_tokens(window)
    spec = load_hand_spec()
    tokens = []
    for i, row in enumerate(rows):
        cmd = CommandVector.from_flat_vector(row, spec=spec)
        if run_mode == TELEOP_5POINT:
            if window.elbows is None:
                raise OperatorInputError("5-point hybrid tokens need elbows")
            _, _, elbows = stride_window(window, dst_hz=POLICY_HZ)
            assert elbows is not None
            fp = FivePointCommand(cmd, elbows[i, :3], elbows[i, 3:])
            pos, quat = command_to_vr_5point(fp)
        else:
            pos, quat = command_to_vr_3point(cmd)
        tokens.append(pack_hybrid_encoder_cmd(pos, quat, mode=run_mode))
    return np.stack(tokens, axis=0)


def operator_to_nav_spring(
    window: OperatorWindow,
    state: RootSpringState,
    *,
    nav_at: str = "start",
) -> RootSpringRef:
    """Eq. 8 at 500 Hz from one held operator nav_cmd. Not a Hermite of vx.

    Live deployment re-seeds the spring each 100 Hz tick with the robot state
    at that tick. This helper is the single-command diagnostic over the window
    (same contract as ADR-038 ``stream_nav_spring``).
    """
    layout = command_layout()
    lo, hi = layout["nav_cmd"]
    if nav_at == "start":
        nav = window.commands[0, lo:hi]
    elif nav_at == "latest":
        nav = window.commands[-1, lo:hi]
    else:
        raise OperatorInputError("nav_at must be 'start' or 'latest'")
    return stream_nav_spring(state, nav, horizon_s=window.horizon_s, dst_hz=STREAM_HZ)


class OperatorHold:
    """Live 100 Hz push / 500 Hz hold-last readout. No PICO SDK."""

    def __init__(self, *, source: str) -> None:
        self.source = refuse_operator_source(source)
        self._t_s: float | None = None
        self._row: np.ndarray | None = None
        self._elbows: np.ndarray | None = None
        self.spec = load_hand_spec()

    def push(
        self,
        t_s: float,
        command: np.ndarray | CommandVector,
        *,
        elbows: np.ndarray | None = None,
    ) -> None:
        if isinstance(command, CommandVector):
            row = command.to_flat_vector()
        else:
            row = np.asarray(command, dtype=np.float64).reshape(-1)
        expected = command_dim(self.spec)
        if row.shape != (expected,):
            raise OperatorInputError(
                f"operator push dim {row.shape} != command_schema_v1 {expected}"
            )
        t = float(t_s)
        if not np.isfinite(t):
            raise OperatorInputError("t_s must be finite")
        if self._t_s is not None:
            dt = t - self._t_s
            expected_dt = 1.0 / float(OPERATOR_INPUT_HZ)
            if abs(dt - expected_dt) > 1e-9:
                raise OperatorInputError(
                    f"operator push dt={dt} s, expected {expected_dt} s (100 Hz grid)"
                )
        if elbows is not None:
            e = np.asarray(elbows, dtype=np.float64).reshape(6)
            if self.source != "vr_5point":
                raise OperatorInputError(f"source={self.source} cannot push elbows")
            self._elbows = e
        elif self.source == "vr_5point":
            raise OperatorInputError("vr_5point push requires elbows")
        self._t_s = t
        self._row = row.copy()

    def read(self, t_s: float) -> np.ndarray:
        """Hold-last command at an arbitrary 500 Hz timestamp."""
        if self._row is None or self._t_s is None:
            raise OperatorInputError("operator hold is empty; push a 100 Hz sample first")
        t = float(t_s)
        if t + 1e-12 < self._t_s:
            raise OperatorInputError("cannot read operator hold before the last push")
        return self._row.copy()
