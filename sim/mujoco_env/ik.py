"""Damped-least-squares position IK on a subset of MuJoCo joints."""

from __future__ import annotations

import numpy as np


def joint_dof_columns(model, joint_names: list[str]) -> np.ndarray:
    cols = []
    for name in joint_names:
        jid = int(model.joint(name).id)
        cols.append(int(model.jnt_dofadr[jid]))
    return np.asarray(cols, dtype=np.int32)


def dls_ik_pos(
    model,
    data,
    body_name: str,
    target_pos: np.ndarray,
    joint_names: list[str],
    *,
    damping: float = 0.05,
    iters: int = 8,
    step: float = 0.25,
    max_delta_rad: float = 0.08,
) -> dict:
    import mujoco

    body_id = int(model.body(body_name).id)
    cols = joint_dof_columns(model, joint_names)
    jacp = np.zeros((3, model.nv))
    last_err = 0.0
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        mujoco.mj_jacBody(model, data, jacp, None, body_id)
        err = np.asarray(target_pos, dtype=np.float64) - data.xpos[body_id]
        last_err = float(np.linalg.norm(err))
        if last_err < 2e-3:
            break
        j = jacp[:, cols]
        h = j @ j.T + damping * np.eye(3)
        dq = j.T @ np.linalg.solve(h, err)
        dq = np.clip(dq, -max_delta_rad, max_delta_rad)
        for col, delta in zip(cols, dq, strict=True):
            jnt = int(np.where(model.jnt_dofadr == col)[0][0])
            qadr = int(model.jnt_qposadr[jnt])
            lo, hi = model.jnt_range[jnt]
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + step * delta, lo, hi))
    mujoco.mj_forward(model, data)
    return {"err_m": last_err, "pos": data.xpos[body_id].copy()}


def smooth_move_wrist(
    model,
    data,
    body_name: str,
    target_pos: np.ndarray,
    joint_names: list[str],
    *,
    alpha: float = 0.12,
) -> dict:
    """One small IK step toward target — call every control tick for smooth motion."""
    import mujoco

    bid = int(model.body(body_name).id)
    pos = data.xpos[bid].copy()
    waypoint = pos + alpha * (np.asarray(target_pos, dtype=np.float64) - pos)
    return dls_ik_pos(
        model,
        data,
        body_name,
        waypoint,
        joint_names,
        damping=0.08,
        iters=4,
        step=0.35,
        max_delta_rad=0.05,
    )
