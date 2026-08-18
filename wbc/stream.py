"""SONIC §3.5 500 Hz command stream. Numpy only — no simulator.

He et al., arXiv:2511.07820v3 §3.5 four concurrent loops: planner 10 Hz,
policy 50 Hz, operator 100 Hz, command stream 500 Hz. This module is the
500 Hz PD instruction ring.

ADR-021 ``sonic_command_hz: 50`` is the *policy/token* rate. It is not this
ring. Hands still bypass WBC; they ride the 500 Hz clock so the DexHand2
1 kHz MIT backend can subsample. Nav uses Eq. 8 evaluated at 500 Hz
timestamps (ADR-038), not a Hermite resample of the 10 Hz spring samples.

ADR-054: the 25-D ``policy_action`` (T800 PPO/decoder joint targets) is
**zero-order held** onto this ring on the same 50 Hz tick *after* ADR-053
stash. That vector is ``a_t`` for PD. Decoder ``last_action`` still sees
``a_{t-1}``. ADR-055 evaluates ``τ = Kp(a_t − q) − Kd q̇`` with EngineAI
``pd_stand`` bring-up (not SONIC tracking). Do **not** Hermite it
(Hermite is for command_schema poses). Do **not** treat it as a decoder
ONNX run.

Not Table S4. Not pad–cardboard. Not a T800+Hand weld (ADR-009 / ADR-039).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import HandSpec, command_dim, load_hand_spec
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import G1_N_DOF, G1_PLANNER_QPOS_DIM, load_t800_sonic, planner_qpos_dim
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
        raise ValueError("operator_input_hz must stay 100 (SONIC §3.5, ADR-040)")
    if int(raw["factor_planner_to_stream"]) != STREAM_HZ // PLANNER_HZ:
        raise ValueError("factor_planner_to_stream must stay 50")
    if int(raw["factor_policy_to_stream"]) != STREAM_HZ // POLICY_HZ:
        raise ValueError("factor_policy_to_stream must stay 10")
    if int(raw["factor_operator_to_stream"]) != STREAM_HZ // OPERATOR_INPUT_HZ:
        raise ValueError("factor_operator_to_stream must stay 5")
    if str(raw.get("nav_eval")) != "closed_form_eq8":
        raise ValueError("nav_eval must stay closed_form_eq8 (do not Hermite the spring)")
    pd_flags = (
        "policy_action_pd_is_zoh",
        "not_policy_action_hermite",
        "policy_action_feeds_500hz_pd_after_stash",
        "not_policy_action_from_decoder_onnx",
        "not_policy_action_same_tick_decoder_obs",
        "pd_plant_on_same_tick",
        "pd_plant_gains_are_pd_stand_bringup",
        "not_sonic_tracking_gains",
        "pd_plant_dq_des_is_zero",
        "not_pd_plant_from_decoder_onnx",
        "not_pd_plant_hermite",
        "not_pd_plant_finite_diff_dq",
        "not_pd_tau_onto_decoder_obs",
    )
    for key in pd_flags:
        if raw.get(key) is not True:
            raise ValueError(f"stream.yaml must keep {key}: true")
    sonic = load_t800_sonic()
    n = int(sonic["n_revolute"])
    if int(raw["pd_action_dim"]) != n:
        raise ValueError(f"pd_action_dim must stay T800 {n}")
    if int(raw["g1_action_dim_forbidden"]) != G1_N_DOF:
        raise ValueError("g1_action_dim_forbidden must stay 29")
    if int(sonic["command_stream_hz"]) != STREAM_HZ:
        raise ValueError("t800_sonic.yaml command_stream_hz drifted from 500")
    if int(sonic["control_rate_hz"]) != POLICY_HZ:
        raise ValueError("t800_sonic.yaml control_rate_hz drifted from 50")
    if int(sonic["planner_hz"]) != PLANNER_HZ:
        raise ValueError("t800_sonic.yaml planner_hz drifted from 10")
    if int(sonic["operator_input_hz"]) != OPERATOR_INPUT_HZ:
        raise ValueError("t800_sonic.yaml operator_input_hz drifted from 100")
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
    n_pd = int(load_t800_sonic()["n_revolute"])
    if arr.shape[1] == n_pd:
        raise CommandStreamError(
            f"got dim {n_pd} (T800 policy_action / PD q_des). "
            "Use stream_policy_action_zoh (ADR-054 ZOH), not stream_policy_tokens "
            "(command_schema spline). Do not Hermite joint targets."
        )
    if arr.shape[1] != expected:
        raise CommandStreamError(
            "500 Hz stream requires command_schema_v1 rows. "
            f"got dim {arr.shape[1]} (expected {expected}). "
            "If this is Case A, call CaseAToCommandSchema.convert(..., apply_fk=True) first. "
            "If this is T800 25-D policy_action, call stream_policy_action_zoh."
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


def refuse_policy_action_hermite() -> None:
    """25-D PD q_des is ZOH. Cubic Hermite is for command_schema poses (ADR-039)."""
    n = int(load_t800_sonic()["n_revolute"])
    raise CommandStreamError(
        f"T800 {n}-D policy_action onto the 500 Hz PD ring is zero-order hold "
        "(ZOH, ADR-054). Do not cubic-Hermite joint targets. Hermite/SLERP stays on "
        "command_schema_v1 rows via stream_policy_tokens / stream_planned_ref."
    )


def refuse_policy_action_as_decoder_run() -> None:
    """PD q_des is the caller-supplied a_t. It is not a decoder ONNX output."""
    raise CommandStreamError(
        "do not treat the 25-D PD q_des as a decoder ONNX run. Caller supplies "
        "policy_action; this ring only ZOH-holds it after the 50 Hz stash. "
        "Until a T800 decoder exists, omit policy_action and keep startup zeros."
    )


def require_t800_pd_action(action: np.ndarray, *, n_dof: int | None = None) -> np.ndarray:
    """T800 PPO/decoder action is 25-D joint q_des. G1 / qpos / hands refused."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape == (G1_N_DOF,):
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if a.shape == (G1_PLANNER_QPOS_DIM,):
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    if a.shape == (planner_qpos_dim(n),):
        raise CommandStreamError(
            f"PD q_des dim {a.shape[0]} is planner qpos, not the {n}-D policy "
            "action. Do not copy playback.qpos onto the 500 Hz PD ring."
        )
    if a.shape[0] == 45:
        raise CommandStreamError(
            "PD q_des dim 45 looks like DexHand2 concat; hands bypass WBC"
        )
    expected_cmd = command_dim(load_hand_spec())
    if a.shape == (expected_cmd,):
        raise CommandStreamError(
            "PD q_des is T800 25-D joint targets, not command_schema_v1. "
            "Use stream_policy_tokens for 75-D teleop rows."
        )
    if a.shape != (n,):
        raise CommandStreamError(
            f"PD q_des dim {a.shape} != T800 {n} (PPO/decoder action). "
            "Do not invent a decoder ONNX output."
        )
    if not np.isfinite(a).all():
        raise CommandStreamError("PD q_des contains NaN/Inf")
    return a.copy()


@dataclass(frozen=True)
class PolicyPdStream:
    """500 Hz ZOH window of T800 25-D policy_action. Simulator-free."""

    t_s: np.ndarray
    q_des_rad: np.ndarray  # (T, n_dof)
    rate_hz: int
    src_hz: int
    horizon_s: float
    hold: str

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


def stream_policy_action_zoh(
    actions: np.ndarray,
    *,
    infer_hz: int = POLICY_HZ,
    dst_hz: int = STREAM_HZ,
    t0_s: float = 0.0,
) -> PolicyPdStream:
    """50 Hz T800 25-D policy_action → 500 Hz PD q_des by zero-order hold.

    Not command_schema spline. Not cubic Hermite. Not a decoder ONNX run.
    Hands still bypass WBC.
    """
    n = int(load_t800_sonic()["n_revolute"])
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise CommandStreamError(f"policy_action must be (N, {n}), got {arr.shape}")
    if arr.shape[1] == command_dim(load_hand_spec()):
        raise CommandStreamError(
            "command_schema_v1 rows use stream_policy_tokens, not "
            "stream_policy_action_zoh (ADR-054 ZOH of 25-D joint targets)"
        )
    if arr.shape[0] < 1:
        raise CommandStreamError("policy_action ZOH needs ≥1 source step")
    rows = np.stack([require_t800_pd_action(row, n_dof=n) for row in arr], axis=0)
    factor = stream_factor(int(infer_hz), dst_hz=int(dst_hz))
    if rows.shape[0] == 1:
        n_dst = factor
        t_dst = float(t0_s) + np.arange(n_dst, dtype=np.float64) / float(dst_hz)
        held = np.repeat(rows, n_dst, axis=0)
    else:
        n_dst = (rows.shape[0] - 1) * factor + 1
        t_dst = float(t0_s) + np.linspace(
            0.0, (rows.shape[0] - 1) / float(infer_hz), n_dst
        )
        idx = np.minimum(np.arange(n_dst) // factor, rows.shape[0] - 1)
        held = rows[idx]
    horizon_s = float(t_dst[-1] - t_dst[0]) if n_dst > 1 else 0.0
    return PolicyPdStream(
        t_s=t_dst,
        q_des_rad=held,
        rate_hz=int(dst_hz),
        src_hz=int(infer_hz),
        horizon_s=horizon_s,
        hold="zoh",
    )


class PolicyPdHold:
    """Live 50 Hz push / 500 Hz hold-last readout of 25-D PD q_des.

    Startup zeros are padding, not an invented decoder ONNX vector.
    Official C++ sends ``a_t`` to PD on the same 50 Hz tick that produced it.
    Decoder ``last_action`` still sees ``a_{t-1}`` (ADR-053).
    """

    def __init__(self, *, n_dof: int | None = None) -> None:
        n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
        if n == G1_N_DOF:
            refuse_g1_checkpoint(n_dof=n)
        self.n_dof = n
        self._q_des = np.zeros(n, dtype=np.float64)
        self._t_s: float | None = None

    def push(self, policy_action: np.ndarray, t_s: float | None = None) -> np.ndarray:
        a = require_t800_pd_action(policy_action, n_dof=self.n_dof)
        if t_s is not None:
            t = float(t_s)
            if not np.isfinite(t):
                raise CommandStreamError("t_s must be finite")
            if self._t_s is not None and t + 1e-12 < self._t_s:
                raise CommandStreamError("cannot push PD q_des earlier than the last sample")
            self._t_s = t
        self._q_des = a
        return a

    def read(self, t_s: float | None = None) -> np.ndarray:
        """Hold-last q_des. Omit ``t_s`` to read the current hold."""
        if t_s is not None:
            t = float(t_s)
            if not np.isfinite(t):
                raise CommandStreamError("t_s must be finite")
            if self._t_s is not None and t + 1e-12 < self._t_s:
                raise CommandStreamError("cannot read PD hold before the last push")
        return self._q_des.copy()

    def zoh_period(self, *, t0_s: float | None = None, dst_hz: int = STREAM_HZ) -> PolicyPdStream:
        """One 50 Hz period of the current hold on the 500 Hz grid (10 samples)."""
        t0 = float(self._t_s if t0_s is None and self._t_s is not None else (t0_s or 0.0))
        return stream_policy_action_zoh(
            self._q_des.reshape(1, -1), infer_hz=POLICY_HZ, dst_hz=int(dst_hz), t0_s=t0
        )
