"""Close ADR-055 τ onto a 500 Hz physics plant. Numpy only — no simulator.

He et al., arXiv:2511.07820v3 §3.5: the 500 Hz command stream feeds joint PD.

    τ_t = Kp (a_t − q_t) − Kd q̇_t

``a_t`` is the T800 25-D ``policy_action`` zero-order held on the PD ring
(ADR-054). ADR-055 evaluates that formula from the 50 Hz snapshot.
This module runs the same formula for ``stream_factor(50) == 10``
substeps on a physics backend. ``wbc/`` does not import MuJoCo; the
backend lives in ``sim/``.

q/dq are **measured** from the backend each substep. Do not interpolate
a 50 Hz snapshot. Do not finite-diff dq. This tick's decoder 874-D stays
pre-physics. IMU is not invented. Omit ``push_hw`` after the first
tick to close the loop (ADR-057): next gather q/dq follow the plant;
IMU stays on the last full snapshot. Gains are EngineAI ``pd_stand``
bring-up (ADR-022), **not** SONIC tracking. Hands still bypass WBC.
Not a decoder ONNX run. Not Table S4. Not pad–cardboard. Not a
T800+Hand weld (ADR-009 / ADR-039).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from wbc.dims import G1_N_DOF, load_t800_sonic
from wbc.pd_plant import (
    GAINS_SOURCE,
    PdPlantError,
    PolicyPdPlant,
    PolicyPdPlantStream,
    require_t800_pd_state,
)
from wbc.stream import STREAM_HZ, require_t800_pd_action, stream_factor

PD_PHYSICS_YAML = Path(__file__).with_name("pd_physics.yaml")
PHYSICS_TIMESTEP_S = 1.0 / STREAM_HZ
PHYSICS_N_STEPS = stream_factor(50)


class PdPhysicsError(PdPlantError):
    """Wrong dim, invented dq, or treating physics τ as a decoder ONNX run."""


class JointPdPhysics(Protocol):
    """500 Hz torque plant. Implemented in ``sim/`` (MuJoCo) or a test double.

    Must not live in ``wbc/`` as a MuJoCo import. ``n_dof`` is T800 25.
    ``timestep_s`` must be ``1/500``. Do not weld DexHand2.
    """

    n_dof: int
    timestep_s: float

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured hinge q (rad) and dq (rad/s). Not interpolated."""
        ...

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Write τ (N·m), step one ``timestep_s``, return measured q/dq."""
        ...


def load_pd_physics_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or PD_PHYSICS_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pd_physics.yaml must be a mapping")
    flags = (
        "not_dexhand2_contact",
        "not_table_s4",
        "not_hand_mit_ring",
        "policy_action_pd_is_zoh",
        "not_policy_action_hermite",
        "policy_action_feeds_500hz_pd_after_stash",
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
        "not_policy_action_from_decoder_onnx",
        "not_policy_action_same_tick_decoder_obs",
        "pd_physics_feeds_next_tick_decoder_q",
        "omit_push_hw_keeps_imu",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise PdPhysicsError(f"pd_physics.yaml must keep {key}: true")
    n = int(load_t800_sonic()["n_revolute"])
    if int(raw["pd_action_dim"]) != n:
        raise PdPhysicsError(f"pd_action_dim must stay T800 {n}")
    if int(raw["g1_action_dim_forbidden"]) != G1_N_DOF:
        raise PdPhysicsError("g1_action_dim_forbidden must stay 29")
    if int(raw["command_stream_hz"]) != STREAM_HZ:
        raise PdPhysicsError("command_stream_hz must stay 500")
    if int(raw["pd_physics_n_steps_per_tick"]) != PHYSICS_N_STEPS:
        raise PdPhysicsError("pd_physics_n_steps_per_tick must stay 10")
    if abs(float(raw["pd_physics_timestep_s"]) - PHYSICS_TIMESTEP_S) > 1e-12:
        raise PdPhysicsError("pd_physics_timestep_s must stay 0.002")
    if str(raw.get("gains_source")) != GAINS_SOURCE:
        raise PdPhysicsError(f"gains_source must stay {GAINS_SOURCE}")
    if int(raw["factor_policy_to_stream"]) != PHYSICS_N_STEPS:
        raise PdPhysicsError("factor_policy_to_stream must stay 10")
    return raw


def refuse_pd_physics_hermite() -> None:
    """Physics q_des is ZOH. Cubic Hermite is for command_schema poses."""
    n = int(load_t800_sonic()["n_revolute"])
    raise PdPhysicsError(
        f"T800 {n}-D PD physics uses ZOH q_des (ADR-054/055/056). Do not cubic-Hermite "
        "joint targets or invent dq_des. Hermite/SLERP stays on command_schema_v1."
    )


def refuse_pd_physics_as_decoder_run() -> None:
    """Physics τ is not a decoder ONNX output."""
    raise PdPhysicsError(
        "do not treat the 25-D PD physics torque as a decoder ONNX run. Caller "
        "supplies policy_action; ADR-054 ZOH-holds it; ADR-055 evaluates "
        "τ = Kp(a_t − q) − Kd q̇; this backend applies that τ for 10 substeps. "
        "Until a T800 decoder exists, omit policy_action and keep startup-zero q_des."
    )


def refuse_pd_physics_finite_diff_dq() -> None:
    """Do not invent 500 Hz dq by differentiating a 50 Hz q snapshot."""
    raise PdPhysicsError(
        "do not finite-diff dq from a 50 Hz q snapshot. Pass measured physics "
        "dq, or hold-last the 50 Hz HardwareSnapshot. Interpolating "
        "proprioception onto 500 Hz is refused."
    )


def refuse_pd_physics_q_onto_this_tick_decoder() -> None:
    """Physics q is applied after decoder assemble. Do not rewrite 874-D."""
    raise PdPhysicsError(
        "this tick's decoder 874-D is assembled from HardwareHold *before* "
        "the 500 Hz physics period. Do not copy physics q into this tick's "
        "decoder obs. Joints may be pushed for the *next* gather."
    )


def refuse_pd_physics_invent_imu() -> None:
    """Do not fill IMU from a guessed latency or from undocumented body rates."""
    raise PdPhysicsError(
        "do not invent IMU from physics. push_joints copies q/dq only and "
        "keeps the previous omega/quat. Measured IMU latency stays REQUIRED_INPUT."
    )


def refuse_omit_push_hw_invent_imu() -> None:
    """Omitting push_hw copies q/dq only. Do not invent IMU from body rates."""
    raise PdPhysicsError(
        "omitting push_hw after a physics period copies q/dq only. "
        "Do not invent IMU from physics body rates. Measured IMU latency "
        "stays REQUIRED_INPUT."
    )


def require_physics_timestep_500hz(timestep_s: float) -> float:
    """MuJoCo / backend dt must match the 500 Hz PD ring. Do not resample."""
    dt = float(timestep_s)
    if abs(dt - PHYSICS_TIMESTEP_S) > 1e-9:
        raise PdPhysicsError(
            f"physics timestep {dt} s != 1/{STREAM_HZ} s. "
            "Do not resample the 500 Hz PD ring onto another dt."
        )
    return dt


def require_t800_tau_nm(vec: np.ndarray, *, n_dof: int | None = None) -> np.ndarray:
    """T800 25-D torque. G1 / qpos / hands / command_schema refused."""
    return require_t800_pd_state(vec, n_dof=n_dof, name="tau")


@dataclass(frozen=True)
class PdPhysicsPeriod:
    """One 50 Hz period: 10 measured 500 Hz (q, dq, τ) samples."""

    t_s: np.ndarray
    q_des_rad: np.ndarray  # (T, n_dof)
    q_rad: np.ndarray  # (T, n_dof) after each substep
    dq_rad_s: np.ndarray  # (T, n_dof) after each substep
    tau_nm: np.ndarray  # (T, n_dof) applied at the start of each substep
    rate_hz: int
    src_hz: int
    horizon_s: float
    hold: str
    gains_source: str
    dq_des_rad_s: float
    timestep_s: float

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


def run_zoh_period(
    plant: PolicyPdPlant,
    physics: JointPdPhysics,
    q_des_rad: np.ndarray,
    *,
    t0_s: float = 0.0,
) -> PdPhysicsPeriod:
    """Apply ZOH q_des for 10 physics substeps. Measured q/dq, not interpolated.

    ``τ`` at substep i uses ``physics.read_q_dq()`` (or the q/dq returned from
    substep i-1). Gains are the plant's pd_stand bring-up. ``dq_des`` is 0.
    """
    n = int(plant.n_dof)
    if int(physics.n_dof) != n:
        raise PdPhysicsError(f"physics n_dof {physics.n_dof} != plant {n}")
    dt = require_physics_timestep_500hz(physics.timestep_s)
    q_des = require_t800_pd_action(q_des_rad, n_dof=n)
    n_steps = PHYSICS_N_STEPS
    q, dq = physics.read_q_dq()
    q = require_t800_pd_state(q, n_dof=n, name="q")
    dq = require_t800_pd_state(dq, n_dof=n, name="dq")
    qs = np.zeros((n_steps, n), dtype=np.float64)
    dqs = np.zeros((n_steps, n), dtype=np.float64)
    taus = np.zeros((n_steps, n), dtype=np.float64)
    ts = np.zeros(n_steps, dtype=np.float64)
    t0 = float(t0_s)
    for i in range(n_steps):
        tau = plant.torque(q, dq, q_des)
        tau = require_t800_tau_nm(tau, n_dof=n)
        q, dq = physics.apply_tau_and_step(tau)
        q = require_t800_pd_state(q, n_dof=n, name="q")
        dq = require_t800_pd_state(dq, n_dof=n, name="dq")
        taus[i] = tau
        qs[i] = q
        dqs[i] = dq
        ts[i] = t0 + (i + 1) * dt
    return PdPhysicsPeriod(
        t_s=ts,
        q_des_rad=np.repeat(q_des.reshape(1, -1), n_steps, axis=0),
        q_rad=qs,
        dq_rad_s=dqs,
        tau_nm=taus,
        rate_hz=STREAM_HZ,
        src_hz=50,
        horizon_s=float(n_steps * dt),
        hold="zoh",
        gains_source=GAINS_SOURCE,
        dq_des_rad_s=0.0,
        timestep_s=dt,
    )


def as_plant_stream(period: PdPhysicsPeriod) -> PolicyPdPlantStream:
    """View a physics period as the simulator-free plant stream type."""
    return PolicyPdPlantStream(
        t_s=np.asarray(period.t_s, dtype=np.float64).copy(),
        q_des_rad=np.asarray(period.q_des_rad, dtype=np.float64).copy(),
        q_rad=np.asarray(period.q_rad, dtype=np.float64).copy(),
        dq_rad_s=np.asarray(period.dq_rad_s, dtype=np.float64).copy(),
        tau_nm=np.asarray(period.tau_nm, dtype=np.float64).copy(),
        rate_hz=int(period.rate_hz),
        src_hz=int(period.src_hz),
        horizon_s=float(period.horizon_s),
        hold=str(period.hold),
        gains_source=str(period.gains_source),
        dq_des_rad_s=float(period.dq_des_rad_s),
    )
