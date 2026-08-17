"""SONIC §3.5 500 Hz command stream. Numpy only — no simulator.

He et al., arXiv:2511.07820v3 §3.5 four concurrent loops: planner 10 Hz,
policy 50 Hz, operator 100 Hz, command stream 500 Hz. This module is the
500 Hz PD instruction ring.

ADR-021 ``sonic_command_hz: 50`` is the *policy/token* rate. It is not this
ring. Hands still bypass WBC; they ride the 500 Hz clock so the DexHand2
1 kHz MIT backend can subsample. Nav uses Eq. 8 evaluated at 500 Hz
timestamps (ADR-038), not a Hermite resample of the 10 Hz spring samples.

Not Table S4. Not pad–cardboard. Not a T800+Hand weld (ADR-009 / ADR-039).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import HandSpec, command_dim, load_hand_spec
from wbc.dims import load_t800_sonic
from wbc.planner import PlannedRef, cubic_hermite, interpolate_command_matrix
from wbc.spring import RootSpringRef, RootSpringState, spring_root_trajectory

STREAM_YAML = Path(__file__).with_name("stream.yaml")
STREAM_HZ = 500
POLICY_HZ = 50
PLANNER_HZ = 10
OPERATOR_INPUT_HZ = 100


class CommandStreamError(ValueError):
    """Wrong rate, non-integer factor, or wrong command dimension."""


def load_stream_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or STREAM_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("stream.yaml must be a mapping")
    if raw.get("not_dexhand2_contact") is not True:
        raise ValueError("stream.yaml must keep not_dexhand2_contact: true")
    if raw.get("not_table_s4") is not True:
        raise ValueError("stream.yaml must keep not_table_s4: true")
    if raw.get("not_hand_mit_ring") is not True:
        raise ValueError("stream.yaml must keep not_hand_mit_ring: true")
    if int(raw["command_stream_hz"]) != STREAM_HZ:
        raise ValueError("command_stream_hz must stay 500 (SONIC §3.5)")
    if int(raw["policy_hz"]) != POLICY_HZ:
        raise ValueError("policy_hz must stay 50 (SONIC §3.5)")
    if int(raw["planner_hz"]) != PLANNER_HZ:
        raise ValueError("planner_hz must stay 10 (SONIC §3.5)")
    if int(raw["operator_input_hz"]) != OPERATOR_INPUT_HZ:
        raise ValueError("operator_input_hz must stay 100 (SONIC §3.5, recorded)")
    if int(raw["factor_planner_to_stream"]) != STREAM_HZ // PLANNER_HZ:
        raise ValueError("factor_planner_to_stream must stay 50")
    if int(raw["factor_policy_to_stream"]) != STREAM_HZ // POLICY_HZ:
        raise ValueError("factor_policy_to_stream must stay 10")
    if str(raw.get("nav_eval")) != "closed_form_eq8":
        raise ValueError("nav_eval must stay closed_form_eq8 (do not Hermite the spring)")
    sonic = load_t800_sonic()
    if int(sonic["command_stream_hz"]) != STREAM_HZ:
        raise ValueError("t800_sonic.yaml command_stream_hz drifted from 500")
    if int(sonic["control_rate_hz"]) != POLICY_HZ:
        raise ValueError("t800_sonic.yaml control_rate_hz drifted from 50")
    if int(sonic["planner_hz"]) != PLANNER_HZ:
        raise ValueError("t800_sonic.yaml planner_hz drifted from 10")
    return raw


def stream_factor(src_hz: int, *, dst_hz: int = STREAM_HZ) -> int:
    """Integer factor from a SONIC loop onto the 500 Hz PD ring.

    Non-integer ratios are refused — do not round a 7 Hz source silently.
    """
    src = int(src_hz)
    dst = int(dst_hz)
    if src < 1 or dst < 1:
        raise CommandStreamError("src_hz and dst_hz must be ≥ 1")
    ratio = dst / src
    factor = int(round(ratio))
    if abs(ratio - factor) > 1e-9 or factor < 1:
        raise CommandStreamError(
            f"src_hz={src} does not divide command_stream_hz={dst}. "
            "Refuse a rounded upsample; resample with an integer-factor source."
        )
    return factor


def _require_command_rows(commands: np.ndarray, spec: HandSpec) -> np.ndarray:
    arr = np.asarray(commands, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise CommandStreamError(f"commands must be (N, D), got {arr.shape}")
    expected = command_dim(spec)
    if arr.shape[1] != expected:
        raise CommandStreamError(
            "500 Hz stream requires command_schema_v1 rows. "
            f"got dim {arr.shape[1]} (expected {expected}). "
            "If this is Case A, call CaseAToCommandSchema.convert(..., apply_fk=True) first."
        )
    if arr.shape[0] < 2:
        raise CommandStreamError("stream needs ≥2 source steps")
    return arr


def stream_times_s(t_src: np.ndarray, *, src_hz: int, dst_hz: int = STREAM_HZ) -> np.ndarray:
    """Destination timestamps spanning the same interval as ``t_src``."""
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    if t_src.shape[0] < 2:
        raise CommandStreamError("need ≥2 source timestamps")
    factor = stream_factor(src_hz, dst_hz=dst_hz)
    n_dst = (t_src.shape[0] - 1) * factor + 1
    return np.linspace(float(t_src[0]), float(t_src[-1]), n_dst)


@dataclass(frozen=True)
class CommandStream:
    """500 Hz PD instruction window. Simulator-free."""

    t_s: np.ndarray
    commands: np.ndarray  # (T, command_dim)
    elbows: np.ndarray | None
    rate_hz: int
    src_hz: int
    horizon_s: float

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


def stream_commands(
    commands: np.ndarray,
    t_src: np.ndarray,
    *,
    src_hz: int,
    spec: HandSpec | None = None,
    pos_interp: str = "cubic_hermite",
    elbows: np.ndarray | None = None,
    dst_hz: int = STREAM_HZ,
) -> CommandStream:
    """Interpolate command_schema rows from ``src_hz`` onto the 500 Hz ring."""
    spec = spec or load_hand_spec()
    arr = _require_command_rows(commands, spec)
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    if t_src.shape[0] != arr.shape[0]:
        raise CommandStreamError("t_src / commands length mismatch")
    t_dst = stream_times_s(t_src, src_hz=src_hz, dst_hz=dst_hz)
    out = interpolate_command_matrix(arr, t_src, t_dst, spec, pos_interp=pos_interp)
    elbow_out = None
    if elbows is not None:
        e = np.asarray(elbows, dtype=np.float64)
        if e.ndim != 2 or e.shape[0] != arr.shape[0] or e.shape[1] != 6:
            raise CommandStreamError(f"elbows must be (N, 6), got {e.shape}")
        if pos_interp == "linear":
            elbow_out = np.stack([np.interp(t_dst, t_src, e[:, d]) for d in range(6)], axis=1)
        else:
            elbow_out = cubic_hermite(t_src, e, t_dst)
    horizon_s = float(t_dst[-1] - t_dst[0])
    return CommandStream(
        t_s=t_dst,
        commands=out,
        elbows=elbow_out,
        rate_hz=int(dst_hz),
        src_hz=int(src_hz),
        horizon_s=horizon_s,
    )


def stream_planned_ref(
    ref: PlannedRef,
    *,
    spec: HandSpec | None = None,
    pos_interp: str = "cubic_hermite",
) -> CommandStream:
    """10 Hz L1a window → 500 Hz PD ring. Same kernel as the planner."""
    if int(ref.rate_hz) != PLANNER_HZ:
        raise CommandStreamError(
            f"stream_planned_ref expects a 10 Hz PlannedRef, got {ref.rate_hz} Hz"
        )
    return stream_commands(
        ref.commands,
        ref.t_s,
        src_hz=PLANNER_HZ,
        spec=spec,
        pos_interp=pos_interp,
        elbows=ref.elbows,
    )


def stream_policy_tokens(
    commands: np.ndarray,
    *,
    spec: HandSpec | None = None,
    pos_interp: str = "cubic_hermite",
    infer_hz: int = POLICY_HZ,
) -> CommandStream:
    """50 Hz SONIC policy/token rows → 500 Hz PD ring.

    ``infer_hz`` must stay 50 unless a test labels another integer divisor.
    This is not ADR-021's VLA→50 Hz upsample.
    """
    spec = spec or load_hand_spec()
    arr = _require_command_rows(commands, spec)
    t_src = np.arange(arr.shape[0], dtype=np.float64) / float(infer_hz)
    return stream_commands(
        arr,
        t_src,
        src_hz=int(infer_hz),
        spec=spec,
        pos_interp=pos_interp,
    )


def stream_nav_spring(
    state: RootSpringState,
    nav_cmd: np.ndarray,
    *,
    horizon_s: float,
    dst_hz: int = STREAM_HZ,
) -> RootSpringRef:
    """Evaluate Eq. 8 at 500 Hz. Do not Hermite-resample a 10 Hz sampling.

    Hands are not in this ref (ADR-038). Upper-body command_schema still goes
    through :func:`stream_planned_ref`.
    """
    h = float(horizon_s)
    if not np.isfinite(h) or h < 0.0:
        raise CommandStreamError("horizon_s must be finite and ≥ 0")
    n_steps = int(round(h * dst_hz)) + 1
    t_dst = np.linspace(0.0, h, n_steps)
    xy, heading = spring_root_trajectory(state, nav_cmd, t_dst)
    from wbc.spring import spring_root_keyframe

    keyframe = spring_root_keyframe(state, nav_cmd, t_s=1.0)
    return RootSpringRef(
        t_s=t_dst,
        pos_xy_m=xy,
        heading_rad=heading,
        rate_hz=int(dst_hz),
        horizon_s=h,
        keyframe=keyframe,
    )
