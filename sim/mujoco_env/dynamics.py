"""Gravity-compensated joint PD plus payload Jacobian feedforward.

The pinned-base T800 plant is torque-actuated (unit-gain motors). A tick is:

    τ = kp (q* − q) − kd q̇ + τ_bias + Jᵀ F_payload

``τ_bias`` is MuJoCo ``qfrc_bias`` (gravity + Coriolis/centrifugal) after
``mj_forward``. That cancels the robot's own dynamics so the PD sees a
nearly-linear error system (computed-torque with q̈* = 0).

A weld constraint is **not** in the RNE tree, so a carried carton does not
appear in ``qfrc_bias``. ``Jᵀ [0, 0, m g]`` at the welded body is the
first-order support wrench for that extra mass. Hands stay on the MIT side
channel; this module only writes T800 ``motor_J*`` controls.
"""

from __future__ import annotations

import numpy as np


def rate_limit_qpos(q_now: np.ndarray, q_tgt: np.ndarray, max_dq: float) -> np.ndarray:
    """Clamp a joint-space step so PD does not see a one-tick IK jump."""
    now = np.asarray(q_now, dtype=np.float64)
    tgt = np.asarray(q_tgt, dtype=np.float64)
    return now + np.clip(tgt - now, -max_dq, max_dq)


def apply_body_pd(
    model,
    data,
    act_ids: np.ndarray,
    q_des_full: np.ndarray,
    kp: float,
    kd: float,
    *,
    gravity_comp: bool = True,
    max_err_rad: float = 0.25,
) -> None:
    """Write motor ctrl for T800 actuators. Call after ``mj_forward`` if using bias."""
    q_des = np.asarray(q_des_full, dtype=np.float64)
    for act_id in act_ids:
        jnt = int(model.actuator_trnid[act_id, 0])
        qadr = int(model.jnt_qposadr[jnt])
        dadr = int(model.jnt_dofadr[jnt])
        err = float(np.clip(q_des[qadr] - data.qpos[qadr], -max_err_rad, max_err_rad))
        tau = kp * err - kd * data.qvel[dadr]
        if gravity_comp:
            tau += float(data.qfrc_bias[dadr])
        lo, hi = model.actuator_ctrlrange[act_id]
        data.ctrl[act_id] = float(np.clip(tau, lo, hi))


def apply_payload_support(
    model,
    data,
    act_ids: np.ndarray,
    body_name: str,
    mass_kg: float,
    *,
    g: float = 9.81,
) -> None:
    """Add Jᵀ [0, 0, m g] so the arm supports a welded payload against gravity.

    Call after ``apply_body_pd`` and ``mj_forward``. ``mass_kg <= 0`` is a no-op.
    """
    if mass_kg <= 0.0:
        return
    import mujoco

    bid = int(model.body(body_name).id)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, bid)
    tau_nv = jacp.T @ np.array([0.0, 0.0, mass_kg * g], dtype=np.float64)
    for act_id in act_ids:
        jnt = int(model.actuator_trnid[act_id, 0])
        dadr = int(model.jnt_dofadr[jnt])
        lo, hi = model.actuator_ctrlrange[act_id]
        data.ctrl[act_id] = float(np.clip(data.ctrl[act_id] + tau_nv[dadr], lo, hi))
