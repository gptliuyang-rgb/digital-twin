"""Apply MIT hybrid τ = kp(qd−q)+kd(dqd−dq)+τ_ff onto unit-gain <motor> actuators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand.controller import mit_torque


@dataclass
class MitActuatorSet:
    ids: np.ndarray
    qadr: np.ndarray
    dadr: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_lim: np.ndarray
    names: tuple[str, ...]


def lookup_hand_actuators(model, gains: list[dict], side: str | None = None) -> MitActuatorSet:
    names = []
    ids = []
    kp = []
    kd = []
    tau = []
    qadr = []
    dadr = []
    prefix = None if side is None else ("l_" if side == "left" else "r_")
    for g in gains:
        name = g["name"]
        if prefix and not name.startswith(prefix):
            continue
        try:
            act_id = int(model.actuator(name).id)
        except Exception:
            continue
        names.append(name)
        ids.append(act_id)
        kp.append(float(g["kp"]))
        kd.append(float(g["kd"]))
        tau.append(float(g["tau_lim"]))
        jnt_id = int(model.actuator_trnid[act_id, 0])
        qadr.append(int(model.jnt_qposadr[jnt_id]))
        dadr.append(int(model.jnt_dofadr[jnt_id]))
    if not ids:
        raise KeyError("no hand MIT actuators matched")
    return MitActuatorSet(
        ids=np.asarray(ids, dtype=np.int32),
        qadr=np.asarray(qadr, dtype=np.int32),
        dadr=np.asarray(dadr, dtype=np.int32),
        kp=np.asarray(kp, dtype=np.float64),
        kd=np.asarray(kd, dtype=np.float64),
        tau_lim=np.asarray(tau, dtype=np.float64),
        names=tuple(names),
    )


def apply_mit(
    model,
    data,
    plant: MitActuatorSet,
    q_des: np.ndarray,
    dq_des: np.ndarray | None = None,
    tau_ff: np.ndarray | None = None,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
) -> np.ndarray:
    q = data.qpos[plant.qadr]
    dq = data.qvel[plant.dadr]
    dq_des = np.zeros_like(q) if dq_des is None else np.asarray(dq_des, dtype=np.float64)
    tau_ff = np.zeros_like(q) if tau_ff is None else np.asarray(tau_ff, dtype=np.float64)
    tau = mit_torque(q, dq, q_des, dq_des, tau_ff, plant.kp * kp_scale, plant.kd * kd_scale)
    tau = np.clip(tau, -plant.tau_lim, plant.tau_lim)
    for i, act_id in enumerate(plant.ids):
        lo, hi = model.actuator_ctrlrange[act_id]
        data.ctrl[act_id] = float(np.clip(tau[i], lo, hi))
    return tau
