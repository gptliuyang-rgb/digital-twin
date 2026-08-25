"""Equality welds for constraint grasps (not Coulomb friction).

Enabling a weld snapshots the current relative pose so the constraint does not
yank the bodies. MuJoCo 3 weld ``eq_data`` is:

    [anchor xyz | relpose pos xyz | relpose quat wxyz | torquescale]

This is still ``policy_eval_forbidden`` — not an E1/E2 friction grasp.
"""

from __future__ import annotations

import numpy as np

from sim.mujoco_env.kinematic_assist import mat_to_quat_wxyz


def equality_id(model, name: str) -> int:
    return int(model.equality(name).id)


def weld_relpose(model, data, eq_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Pose of body2 in body1's frame (matches weld ``relpose``)."""
    eid = equality_id(model, eq_name)
    b1 = int(model.eq_obj1id[eid])
    b2 = int(model.eq_obj2id[eid])
    p1 = np.asarray(data.xpos[b1], dtype=np.float64)
    r1 = np.asarray(data.xmat[b1], dtype=np.float64).reshape(3, 3)
    p2 = np.asarray(data.xpos[b2], dtype=np.float64)
    r2 = np.asarray(data.xmat[b2], dtype=np.float64).reshape(3, 3)
    rel_p = r1.T @ (p2 - p1)
    rel_q = mat_to_quat_wxyz(r1.T @ r2)
    return rel_p, rel_q


def set_weld_active(model, data, eq_name: str, active: bool) -> None:
    """If activating, write current relative pose into ``model.eq_data`` then enable."""
    eid = equality_id(model, eq_name)
    if active:
        rel_p, rel_q = weld_relpose(model, data, eq_name)
        row = np.array(model.eq_data[eid], dtype=np.float64, copy=True)
        row[0:3] = 0.0
        row[3:6] = rel_p
        row[6:10] = rel_q
        if row.shape[0] > 10:
            row[10] = 1.0 if abs(row[10]) < 1e-12 else row[10]
        model.eq_data[eid] = row
        data.eq_active[eid] = True
    else:
        data.eq_active[eid] = False


def has_equality(model, name: str) -> bool:
    try:
        model.equality(name)
        return True
    except Exception:
        return False
