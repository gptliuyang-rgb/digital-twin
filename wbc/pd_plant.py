"""500 Hz joint-PD command plant. Numpy only — no simulator.

He et al., arXiv:2511.07820v3 §3.5: the 500 Hz command stream feeds joint PD.

    τ_t = Kp (a_t − q_t) − Kd q̇_t

``a_t`` is the T800 25-D ``policy_action`` zero-order held on the PD ring
(ADR-054). ``dq_des`` is 0 under ZOH — do not Hermite a velocity target.
Gains are EngineAI ``pd_stand`` bring-up (ADR-022), **not** SONIC tracking.
Hands still bypass WBC. Not a decoder ONNX run. Not Table S4. Not pad–cardboard.
Not a T800+Hand weld (ADR-009 / ADR-039).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import command_dim, load_hand_spec
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import G1_N_DOF, G1_PLANNER_QPOS_DIM, load_t800_sonic, planner_qpos_dim
from wbc.pd_stand import pd_stand_kp_kd
from wbc.stream import (
    STREAM_HZ,
    CommandStreamError,
    PolicyPdStream,
    require_t800_pd_action,
    stream_factor,
    stream_policy_action_zoh,
)

PD_PLANT_YAML = Path(__file__).with_name("pd_plant.yaml")
GAINS_SOURCE = "pd_stand_bringup_not_sonic"


class PdPlantError(CommandStreamError):
    """Wrong dim, invented dq, or treating τ as a decoder ONNX run."""


def load_pd_plant_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or PD_PLANT_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pd_plant.yaml must be a mapping")
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
        "not_policy_action_from_decoder_onnx",
        "not_policy_action_same_tick_decoder_obs",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise PdPlantError(f"pd_plant.yaml must keep {key}: true")
    n = int(load_t800_sonic()["n_revolute"])
    if int(raw["pd_action_dim"]) != n:
        raise PdPlantError(f"pd_action_dim must stay T800 {n}")
    if int(raw["g1_action_dim_forbidden"]) != G1_N_DOF:
        raise PdPlantError("g1_action_dim_forbidden must stay 29")
    if int(raw["command_stream_hz"]) != STREAM_HZ:
        raise PdPlantError("command_stream_hz must stay 500")
    if str(raw.get("gains_source")) != GAINS_SOURCE:
        raise PdPlantError(f"gains_source must stay {GAINS_SOURCE}")
    if int(raw["factor_policy_to_stream"]) != stream_factor(50):
        raise PdPlantError("factor_policy_to_stream must stay 10")
    return raw


def refuse_pd_plant_hermite() -> None:
    """τ is evaluated on ZOH q_des. Cubic Hermite is for command_schema poses."""
    n = int(load_t800_sonic()["n_revolute"])
    raise PdPlantError(
        f"T800 {n}-D PD plant uses ZOH q_des (ADR-054/055). Do not cubic-Hermite "
        "joint targets or invent dq_des. Hermite/SLERP stays on command_schema_v1."
    )


def refuse_pd_plant_as_decoder_run() -> None:
    """τ is not a decoder ONNX output. Caller supplies a_t; this plant only applies PD."""
    raise PdPlantError(
        "do not treat the 25-D PD torque as a decoder ONNX run. Caller supplies "
        "policy_action; ADR-054 ZOH-holds it; this plant evaluates "
        "τ = Kp(a_t − q) − Kd q̇ with pd_stand bring-up gains. Until a T800 decoder "
        "exists, omit policy_action and keep startup-zero q_des."
    )


def refuse_pd_plant_finite_diff_dq() -> None:
    """Do not invent 500 Hz dq by differentiating a 50 Hz q snapshot."""
    raise PdPlantError(
        "do not finite-diff dq from a 50 Hz q snapshot. Pass measured dq, or "
        "hold-last the 50 Hz HardwareSnapshot. Interpolating proprioception "
        "onto 500 Hz is refused."
    )


def require_t800_pd_state(vec: np.ndarray, *, n_dof: int | None = None, name: str = "q") -> np.ndarray:
    """T800 25-D joint state. G1 / qpos / hands / command_schema refused."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    a = np.asarray(vec, dtype=np.float64).reshape(-1)
    if a.shape == (G1_N_DOF,):
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if a.shape == (G1_PLANNER_QPOS_DIM,):
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    if a.shape == (planner_qpos_dim(n),):
        raise PdPlantError(
            f"{name} dim {a.shape[0]} is planner qpos, not the {n}-D joint state. "
            "Do not copy playback.qpos into the PD plant."
        )
    if a.shape[0] == 45:
        raise PdPlantError(f"{name} dim 45 looks like DexHand2 concat; hands bypass WBC")
    expected_cmd = command_dim(load_hand_spec())
    if a.shape == (expected_cmd,):
        raise PdPlantError(
            f"{name} is T800 25-D joints, not command_schema_v1. "
            "Hands and teleop poses do not enter this plant."
        )
    if a.shape != (n,):
        raise PdPlantError(f"{name} dim {a.shape} != T800 {n}")
    if not np.isfinite(a).all():
        raise PdPlantError(f"{name} contains NaN/Inf")
    return a.copy()


def joint_pd_torque_nm(
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    q_des_rad: np.ndarray,
    *,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    n_dof: int | None = None,
) -> np.ndarray:
    """τ = Kp (a_t − q) − Kd q̇  with dq_des = 0 (ZOH).

    Default Kp/Kd are EngineAI ``pd_stand`` bring-up, labelled
    ``pd_stand_bringup_not_sonic``. Passing other gains is allowed only
    for the MuJoCo env that already loaded the same vectors; do not pass
    SONIC tracking gains here and claim they are bring-up.
    """
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    q = require_t800_pd_state(q_rad, n_dof=n, name="q")
    dq = require_t800_pd_state(dq_rad_s, n_dof=n, name="dq")
    q_des = require_t800_pd_action(q_des_rad, n_dof=n)
    if kp is None or kd is None:
        stand_kp, stand_kd = pd_stand_kp_kd()
        kp = stand_kp if kp is None else kp
        kd = stand_kd if kd is None else kd
    kp_v = np.asarray(kp, dtype=np.float64).reshape(-1)
    kd_v = np.asarray(kd, dtype=np.float64).reshape(-1)
    if kp_v.shape != (n,) or kd_v.shape != (n,):
        raise PdPlantError(f"kp/kd must be ({n},), got {kp_v.shape}/{kd_v.shape}")
    if not np.isfinite(kp_v).all() or not np.isfinite(kd_v).all():
        raise PdPlantError("kp/kd contain NaN/Inf")
    return kp_v * (q_des - q) - kd_v * dq


@dataclass(frozen=True)
class PolicyPdPlantStream:
    """500 Hz window of τ from ZOH q_des. Simulator-free."""

    t_s: np.ndarray
    q_des_rad: np.ndarray  # (T, n_dof)
    q_rad: np.ndarray  # (T, n_dof)
    dq_rad_s: np.ndarray  # (T, n_dof)
    tau_nm: np.ndarray  # (T, n_dof)
    rate_hz: int
    src_hz: int
    horizon_s: float
    hold: str
    gains_source: str
    dq_des_rad_s: float

    @property
    def n_steps(self) -> int:
        return int(self.t_s.shape[0])


def _hold_or_stream(
    vec: np.ndarray,
    n_steps: int,
    n_dof: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim == 1:
        row = require_t800_pd_state(arr, n_dof=n_dof, name=name)
        return np.repeat(row.reshape(1, -1), n_steps, axis=0)
    if arr.ndim != 2:
        raise PdPlantError(f"{name} must be ({n_dof},) or (T, {n_dof}), got {arr.shape}")
    if arr.shape[0] != n_steps or arr.shape[1] != n_dof:
        if arr.shape[0] == n_steps and arr.shape[1] in (G1_N_DOF, 32, 45, command_dim(load_hand_spec())):
            require_t800_pd_state(arr[0], n_dof=n_dof, name=name)
        raise PdPlantError(
            f"{name} stream shape {arr.shape} != ({n_steps}, {n_dof}). "
            "Do not interpolate a 50 Hz snapshot onto 500 Hz."
        )
    return np.stack([require_t800_pd_state(row, n_dof=n_dof, name=name) for row in arr], axis=0)


def apply_policy_pd_stream(
    stream: PolicyPdStream,
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    *,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
) -> PolicyPdPlantStream:
    """Evaluate τ on a ZOH q_des window.

    ``q_rad`` / ``dq_rad_s`` are hold-last ``(n,)`` from the 50 Hz snapshot,
    or caller-supplied ``(T, n)`` at 500 Hz. Do not interpolate. Do not
    finite-diff dq.
    """
    if stream.hold != "zoh":
        raise PdPlantError(f"PD plant requires ZOH q_des, got hold={stream.hold!r}")
    if stream.rate_hz != STREAM_HZ:
        raise PdPlantError(f"PD plant requires 500 Hz stream, got {stream.rate_hz}")
    n = int(load_t800_sonic()["n_revolute"])
    if stream.q_des_rad.shape[1] != n:
        raise PdPlantError(f"q_des dim {stream.q_des_rad.shape} != (T, {n})")
    q = _hold_or_stream(q_rad, stream.n_steps, n, "q")
    dq = _hold_or_stream(dq_rad_s, stream.n_steps, n, "dq")
    tau = np.stack(
        [
            joint_pd_torque_nm(q[i], dq[i], stream.q_des_rad[i], kp=kp, kd=kd, n_dof=n)
            for i in range(stream.n_steps)
        ],
        axis=0,
    )
    return PolicyPdPlantStream(
        t_s=np.asarray(stream.t_s, dtype=np.float64).copy(),
        q_des_rad=np.asarray(stream.q_des_rad, dtype=np.float64).copy(),
        q_rad=q,
        dq_rad_s=dq,
        tau_nm=tau,
        rate_hz=int(stream.rate_hz),
        src_hz=int(stream.src_hz),
        horizon_s=float(stream.horizon_s),
        hold="zoh",
        gains_source=GAINS_SOURCE,
        dq_des_rad_s=0.0,
    )


class PolicyPdPlant:
    """Evaluate τ at 500 Hz from ZOH q_des and caller-supplied proprioception.

    Kp/Kd are frozen from EngineAI pd_stand. Not SONIC tracking.
    """

    def __init__(self, *, n_dof: int | None = None) -> None:
        n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
        if n == G1_N_DOF:
            refuse_g1_checkpoint(n_dof=n)
        self.n_dof = n
        self.kp, self.kd = pd_stand_kp_kd()
        if self.kp.shape != (n,) or self.kd.shape != (n,):
            raise PdPlantError(f"pd_stand kp/kd must be ({n},)")
        self.gains_source = GAINS_SOURCE
        self._tau = np.zeros(n, dtype=np.float64)

    def torque(self, q_rad: np.ndarray, dq_rad_s: np.ndarray, q_des_rad: np.ndarray) -> np.ndarray:
        tau = joint_pd_torque_nm(
            q_rad, dq_rad_s, q_des_rad, kp=self.kp, kd=self.kd, n_dof=self.n_dof
        )
        self._tau = tau
        return tau.copy()

    def read(self) -> np.ndarray:
        return self._tau.copy()

    def zoh_period(
        self,
        q_rad: np.ndarray,
        dq_rad_s: np.ndarray,
        q_des_rad: np.ndarray,
        *,
        t0_s: float = 0.0,
        dst_hz: int = STREAM_HZ,
    ) -> PolicyPdPlantStream:
        """One 50 Hz period: 10 identical 500 Hz samples (hold-last q/dq)."""
        q_des = require_t800_pd_action(q_des_rad, n_dof=self.n_dof)
        stream = stream_policy_action_zoh(
            q_des.reshape(1, -1), infer_hz=50, dst_hz=int(dst_hz), t0_s=float(t0_s)
        )
        out = apply_policy_pd_stream(stream, q_rad, dq_rad_s, kp=self.kp, kd=self.kd)
        self._tau = out.tau_nm[-1].copy()
        return out
