"""1 kHz MIT command ring for DexHand2. Numpy only — no simulator.

Official Wuji Hand 2 (docs.wuji.tech overview):

    τ = kp (qd − q) + kd (dqd − dq) + τ_ff

at 1000 Hz × 20 axes. ``command_schema_v1`` ``left_hand_q`` / ``right_hand_q``
bypass WBC (``hand_slice``). This module zero-order-holds those 20-D
targets for ``stream_factor = 20`` substeps on each 50 Hz tick.

It does **not** live in ``wbc/``. ``wbc/*.yaml`` keep ``not_hand_mit_ring:
true``. Do not write hand q into ``last_action``, ``policy_action``, or
decoder 874-D. Do not concatenate T800 25-D + DexHand2 20-D (45-D).

Gains are official MJCF ``sim_kp`` / ``sim_kv``, labelled
``mjcf_gen1_carryover_not_hand2_sysid``. ``hardware_kp`` / ``hardware_kd``
stay REQUIRED_INPUT. ``command_latency_ms`` stays REQUIRED_INPUT — do
not enable a delay deque until measured. τ is clipped to MJCF
``sim_actuator_forcerange_nm`` (sim-only; not a payload rating).
``grasp_success_rate`` stays JSON null. Combined T800+Hand stays
PolicyEvalBlocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from hand.mit import mit_torque
from interface.schema import HandSpec, load_hand_spec

MIT_RING_YAML = Path(__file__).with_name("mit_ring.yaml")
GAINS_SOURCE = "mjcf_gen1_carryover_not_hand2_sysid"
HAND_STREAM_HZ = 1000
POLICY_HZ = 50
HAND_N_STEPS = HAND_STREAM_HZ // POLICY_HZ
HAND_TIMESTEP_S = 1.0 / HAND_STREAM_HZ
T800_N_DOF = 25
G1_N_DOF = 29
PLANNER_QPOS_DIM = 32
T800_PLUS_HAND_DIM = 45
COMMAND_SCHEMA_DIM = 75


class MitRingError(ValueError):
    """Wrong dim, invented latency/gains, or writing hand q into WBC."""


class HandMitPhysics(Protocol):
    """1 kHz torque plant. Implemented in ``sim/`` / backends or a test double.

    ``n_dof`` is DexHand2 20. ``timestep_s`` must be ``1/1000``.
    Do not weld the hand onto T800 here.
    """

    n_dof: int
    timestep_s: float

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured hinge q (rad) and dq (rad/s). Not interpolated."""
        ...

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Write τ (N·m), step one ``timestep_s``, return measured q/dq."""
        ...


def load_mit_ring_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or MIT_RING_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("mit_ring.yaml must be a mapping")
    flags = (
        "not_dexhand2_contact",
        "not_table_s4",
        "hand_mit_ring",
        "not_wbc_last_action",
        "not_wbc_policy_action",
        "not_decoder_obs",
        "not_concat_t800_plus_hand",
        "hand_q_is_zoh",
        "not_hand_q_hermite",
        "dq_des_is_zero",
        "gains_are_mjcf_gen1_carryover",
        "not_hardware_kp",
        "not_invented_latency",
        "not_force_mode_without_hardware_tau",
        "mit_ring_on_same_tick",
        "mit_ring_is_optional",
        "not_mit_ring_from_decoder_onnx",
        "not_mit_ring_finite_diff_dq",
        "clip_tau_to_sim_forcerange",
        "not_sim_forcerange_as_payload_rating",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise MitRingError(f"mit_ring.yaml must keep {key}: true")
    spec = load_hand_spec()
    n = int(spec.n_active_dof)
    if int(raw["n_active_dof"]) != n:
        raise MitRingError(f"n_active_dof must stay {n}")
    if int(raw["hand_stream_hz"]) != HAND_STREAM_HZ:
        raise MitRingError("hand_stream_hz must stay 1000")
    if int(raw["control_hz"]) != POLICY_HZ:
        raise MitRingError("control_hz must stay 50")
    if int(raw["mit_n_steps_per_tick"]) != HAND_N_STEPS:
        raise MitRingError("mit_n_steps_per_tick must stay 20")
    if abs(float(raw["mit_timestep_s"]) - HAND_TIMESTEP_S) > 1e-12:
        raise MitRingError("mit_timestep_s must stay 0.001")
    if int(raw["factor_policy_to_hand_stream"]) != HAND_N_STEPS:
        raise MitRingError("factor_policy_to_hand_stream must stay 20")
    if str(raw.get("gains_source")) != GAINS_SOURCE:
        raise MitRingError(f"gains_source must stay {GAINS_SOURCE}")
    if int(raw["g1_n_dof_forbidden"]) != G1_N_DOF:
        raise MitRingError("g1_n_dof_forbidden must stay 29")
    if int(raw["t800_n_dof_forbidden"]) != T800_N_DOF:
        raise MitRingError("t800_n_dof_forbidden must stay 25")
    if int(raw["planner_qpos_dim_forbidden"]) != PLANNER_QPOS_DIM:
        raise MitRingError("planner_qpos_dim_forbidden must stay 32")
    if int(raw["t800_plus_hand_dim_forbidden"]) != T800_PLUS_HAND_DIM:
        raise MitRingError("t800_plus_hand_dim_forbidden must stay 45")
    if int(raw["command_schema_dim_forbidden"]) != COMMAND_SCHEMA_DIM:
        raise MitRingError("command_schema_dim_forbidden must stay 75")
    return raw


def sim_kp_kd(spec: HandSpec | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Official MJCF kp/kv in ``joint_order``. Not Hand 2 sys-id."""
    spec = spec or load_hand_spec()
    kp_map = spec.raw["sim_kp"]
    kv_map = spec.raw["sim_kv"]
    kp = np.array([float(kp_map[name]) for name in spec.joint_order], dtype=np.float64)
    kd = np.array([float(kv_map[name]) for name in spec.joint_order], dtype=np.float64)
    return kp, kd


def sim_forcerange_nm(spec: HandSpec | None = None) -> np.ndarray:
    """MJCF ``forcerange``. Sim-only. Not a hardware payload rating."""
    spec = spec or load_hand_spec()
    raw = spec.raw["sim_actuator_forcerange_nm"]
    return np.array([float(raw[name]) for name in spec.joint_order], dtype=np.float64)


def require_hand_q(vec: np.ndarray, *, n_dof: int | None = None, name: str = "q") -> np.ndarray:
    """DexHand2 20-D. T800 / G1 / qpos / concat / command_schema refused."""
    spec = load_hand_spec()
    n = int(n_dof if n_dof is not None else spec.n_active_dof)
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    dim = int(arr.shape[0])
    if dim == T800_N_DOF:
        raise MitRingError(
            f"{name} dim {dim} is T800 WBC action. DexHand2 q is {n}-D and bypasses WBC."
        )
    if dim == G1_N_DOF:
        raise MitRingError(f"{name} dim {dim} is G1 29-D. DexHand2 q is {n}-D.")
    if dim == PLANNER_QPOS_DIM:
        raise MitRingError(f"{name} dim {dim} is planner qpos. Do not copy clip hinges into the hand ring.")
    if dim == T800_PLUS_HAND_DIM:
        raise MitRingError(
            f"{name} dim {dim} concatenates T800 25-D + DexHand2 20-D. "
            "Hands bypass WBC; pass a 20-D left_hand_q or right_hand_q."
        )
    if dim == COMMAND_SCHEMA_DIM:
        raise MitRingError(
            f"{name} dim {dim} is the full command_schema vector. "
            "Pass left_hand_q / right_hand_q slices, not the 75-D flat command."
        )
    if dim != n:
        raise MitRingError(f"{name} dim {dim} != DexHand2 {n}")
    if not np.isfinite(arr).all():
        raise MitRingError(f"{name} contains NaN/Inf")
    return arr.copy()


def require_hand_timestep_1khz(timestep_s: float) -> float:
    """Backend dt must match the 1 kHz MIT ring. Do not resample."""
    dt = float(timestep_s)
    if abs(dt - HAND_TIMESTEP_S) > 1e-9:
        raise MitRingError(
            f"hand physics timestep {dt} s != 1/{HAND_STREAM_HZ} s. "
            "Do not resample the 1 kHz MIT ring onto another dt."
        )
    return dt


def refuse_hand_q_hermite() -> None:
    """Hand q_des is ZOH. Cubic Hermite is for command_schema poses."""
    raise MitRingError(
        "DexHand2 20-D MIT ring uses ZOH q_des (ADR-058). Do not cubic-Hermite "
        "finger targets or invent dq_des. Hermite/SLERP stays on command_schema_v1 wrists."
    )


def refuse_hand_q_onto_wbc_last_action() -> None:
    """Hands bypass WBC. last_action stays T800 25-D."""
    raise MitRingError(
        "do not write left_hand_q / right_hand_q into last_action or policy_action. "
        "WBC action_dim stays T800 25. DexHand2 is the L1c bypass (ADR-058)."
    )


def refuse_mit_ring_as_decoder_run() -> None:
    """MIT τ is not a decoder ONNX output."""
    raise MitRingError(
        "do not treat the 20-D MIT torque as a decoder ONNX run. Caller supplies "
        "command_schema left_hand_q / right_hand_q; this ring ZOH-holds them."
    )


def refuse_mit_ring_finite_diff_dq() -> None:
    """Do not invent 1 kHz dq by differentiating a 50 Hz q snapshot."""
    raise MitRingError(
        "do not finite-diff dq from a 50 Hz hand q snapshot. Pass measured physics "
        "dq, or hold-last. Interpolating proprioception onto 1 kHz is refused."
    )


def refuse_hardware_kp() -> None:
    """hardware_kp is REQUIRED_INPUT. This ring uses MJCF gen-1 sim_kp."""
    raise MitRingError(
        "hardware_kp / hardware_kd are REQUIRED_INPUT. This ring uses official "
        f"MJCF sim_kp/sim_kv labelled {GAINS_SOURCE}. Do not invent Hand 2 sys-id."
    )


def refuse_invented_latency() -> None:
    """command_latency_ms is REQUIRED_INPUT. Do not enable a guessed delay."""
    raise MitRingError(
        "command_latency_ms is REQUIRED_INPUT. Do not enable the delay deque "
        "on the 1 kHz MIT ring until it is measured."
    )


def refuse_force_mode_without_hardware_tau() -> None:
    """hand_mode=force needs a measured torque map."""
    raise MitRingError(
        "hand_mode=force needs motor_max_torque_nm (REQUIRED_INPUT). "
        "Do not invent a current-to-tau map. Keep mode=position and pass q."
    )


def refuse_sim_forcerange_as_payload_rating() -> None:
    """MJCF forcerange is not E3 / not a committed load spec."""
    raise MitRingError(
        "sim_actuator_forcerange_nm is MJCF sim-only. Do not publish it as a "
        "payload rating. motor_max_torque_nm stays REQUIRED_INPUT (E3)."
    )


@dataclass
class HandMitPlant:
    """τ = kp (qd − q) + kd (0 − dq) + τ_ff, clipped to MJCF forcerange."""

    spec: HandSpec
    kp: np.ndarray
    kd: np.ndarray
    forcerange_nm: np.ndarray
    gains_source: str = GAINS_SOURCE
    dq_des_rad_s: float = 0.0

    @classmethod
    def from_spec(cls, spec: HandSpec | None = None) -> HandMitPlant:
        spec = spec or load_hand_spec()
        kp, kd = sim_kp_kd(spec)
        return cls(spec=spec, kp=kp, kd=kd, forcerange_nm=sim_forcerange_nm(spec))

    @property
    def n_dof(self) -> int:
        return int(self.spec.n_active_dof)

    def torque(
        self,
        q_rad: np.ndarray,
        dq_rad_s: np.ndarray,
        q_des_rad: np.ndarray,
        *,
        dq_des_rad_s: np.ndarray | None = None,
        tau_ff_nm: np.ndarray | None = None,
    ) -> np.ndarray:
        n = self.n_dof
        q = require_hand_q(q_rad, n_dof=n, name="q")
        dq = require_hand_q(dq_rad_s, n_dof=n, name="dq")
        q_des = require_hand_q(q_des_rad, n_dof=n, name="q_des")
        if dq_des_rad_s is None:
            dq_des = np.zeros(n, dtype=np.float64)
        else:
            dq_des = require_hand_q(dq_des_rad_s, n_dof=n, name="dq_des")
        if tau_ff_nm is None:
            tau_ff = np.zeros(n, dtype=np.float64)
        else:
            tau_ff = require_hand_q(tau_ff_nm, n_dof=n, name="tau_ff")
        tau = mit_torque(q, dq, q_des, dq_des, tau_ff, self.kp, self.kd)
        return np.clip(tau, -self.forcerange_nm, self.forcerange_nm)


@dataclass(frozen=True)
class MitRingPeriod:
    """One 50 Hz period: 20 measured 1 kHz (q, dq, τ) samples for one hand."""

    t_s: np.ndarray
    q_des_rad: np.ndarray
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    tau_nm: np.ndarray
    rate_hz: int
    src_hz: int
    horizon_s: float
    hold: str
    gains_source: str
    dq_des_rad_s: float
    timestep_s: float
    side: str

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


def run_mit_period(
    plant: HandMitPlant,
    physics: HandMitPhysics,
    q_des_rad: np.ndarray,
    *,
    t0_s: float = 0.0,
    side: str = "right",
) -> MitRingPeriod:
    """Apply ZOH q_des for 20 physics substeps. Measured q/dq, not interpolated."""
    n = int(plant.n_dof)
    if int(physics.n_dof) != n:
        raise MitRingError(f"physics n_dof {physics.n_dof} != plant {n}")
    dt = require_hand_timestep_1khz(physics.timestep_s)
    q_des = require_hand_q(q_des_rad, n_dof=n, name="q_des")
    n_steps = HAND_N_STEPS
    q, dq = physics.read_q_dq()
    q = require_hand_q(q, n_dof=n, name="q")
    dq = require_hand_q(dq, n_dof=n, name="dq")
    qs = np.zeros((n_steps, n), dtype=np.float64)
    dqs = np.zeros((n_steps, n), dtype=np.float64)
    taus = np.zeros((n_steps, n), dtype=np.float64)
    ts = np.zeros(n_steps, dtype=np.float64)
    t0 = float(t0_s)
    for i in range(n_steps):
        tau = plant.torque(q, dq, q_des)
        q, dq = physics.apply_tau_and_step(tau)
        q = require_hand_q(q, n_dof=n, name="q")
        dq = require_hand_q(dq, n_dof=n, name="dq")
        taus[i] = tau
        qs[i] = q
        dqs[i] = dq
        ts[i] = t0 + (i + 1) * dt
    return MitRingPeriod(
        t_s=ts,
        q_des_rad=np.repeat(q_des.reshape(1, -1), n_steps, axis=0),
        q_rad=qs,
        dq_rad_s=dqs,
        tau_nm=taus,
        rate_hz=HAND_STREAM_HZ,
        src_hz=POLICY_HZ,
        horizon_s=float(n_steps * dt),
        hold="zoh",
        gains_source=GAINS_SOURCE,
        dq_des_rad_s=0.0,
        timestep_s=dt,
        side=side,
    )
