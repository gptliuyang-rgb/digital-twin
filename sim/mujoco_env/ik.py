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
    damping: float = 1e-2,
    iters: int = 12,
    step: float = 0.6,
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
        if last_err < 1e-4:
            break
        j = jacp[:, cols]
        h = j @ j.T + damping * np.eye(3)
        dq = j.T @ np.linalg.solve(h, err)
        for col, delta in zip(cols, dq, strict=True):
            # map dof column back to qpos (hinge/slide: 1-1)
            jnt = int(np.where(model.jnt_dofadr == col)[0][0])
            qadr = int(model.jnt_qposadr[jnt])
            lo, hi = model.jnt_range[jnt]
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + step * delta, lo, hi))
    mujoco.mj_forward(model, data)
    return {"err_m": last_err, "pos": data.xpos[body_id].copy()}
