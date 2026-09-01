"""Body poses expressed in the T800 robot base (``LINK_BASE``).

Pinned-base sim: ``LINK_BASE`` coincides with world. On a floating robot the
same helper still returns DexHand frames in the pelvis/base, which is the
parent of ``robot_heading_frame`` after yaw normalization (see frames.yaml).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assets.combined.flange import se3_inv, se3_mul, se3_to_pos_quat

BASE_BODY = "LINK_BASE"
CHAIN_BODIES = (
    "LINK_BASE",
    "LINK_WRIST_END_L",
    "l_mount",
    "l_wrist",
    "LINK_WRIST_END_R",
    "r_mount",
    "r_wrist",
)


def _body_id(model, name: str) -> int:
    return int(model.body(name).id)


def body_se3_world(model, data, name: str) -> np.ndarray:
    bid = _body_id(model, name)
    rot = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    pos = np.asarray(data.xpos[bid], dtype=np.float64)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def pose_in_base(model, data, body: str, *, parent: str = BASE_BODY) -> dict[str, Any]:
    """``body`` pose in ``parent`` (default ``LINK_BASE``)."""
    t = se3_mul(se3_inv(body_se3_world(model, data, parent)), body_se3_world(model, data, body))
    pos, quat = se3_to_pos_quat(t)
    rot = t[:3, :3]
    return {
        "body": body,
        "parent": parent,
        "pos_m": pos.tolist(),
        "quat_wxyz": quat.tolist(),
        "x_axis": rot[:, 0].tolist(),
        "y_axis": rot[:, 1].tolist(),
        "z_axis": rot[:, 2].tolist(),
    }


def relative_se3(model, data, parent: str, child: str) -> np.ndarray:
    return se3_mul(se3_inv(body_se3_world(model, data, parent)), body_se3_world(model, data, child))


def alignment_snapshot(model, data, *, yaml_chain: dict[str, Any] | None = None) -> dict[str, Any]:
    """Live FK check: YAML ``T_wrist_end_palm`` vs MuJoCo ``l_wrist`` in ``LINK_WRIST_END_L``."""
    import mujoco

    from assets.combined.flange import wrist_end_to_palm_se3

    mujoco.mj_forward(model, data)
    bodies = {}
    for name in CHAIN_BODIES:
        try:
            bodies[name] = pose_in_base(model, data, name)
        except Exception as exc:  # noqa: BLE001
            bodies[name] = {"error": str(exc)}

    yaml_t = wrist_end_to_palm_se3()
    live_t = relative_se3(model, data, "LINK_WRIST_END_L", "l_wrist")
    err = float(np.linalg.norm(live_t - yaml_t))
    pos_err = float(np.linalg.norm(live_t[:3, 3] - yaml_t[:3, 3]))
    rot_err = float(np.linalg.norm(live_t[:3, :3] - yaml_t[:3, :3], ord="fro"))
    out: dict[str, Any] = {
        "parent": BASE_BODY,
        "bodies_in_base": bodies,
        "yaml_T_wrist_end_palm_matches_left_mjcf": {
            "pos_err_m": pos_err,
            "rot_fro_err": rot_err,
            "se3_abs_max": err,
            "ok": pos_err < 2e-4 and rot_err < 2e-3,
        },
        "note": (
            "T_base_palm uses T800 FK for T_base_wrist_end. YAML only stores the "
            "constant T_wrist_end_palm. Identity flange ⇒ palm axes follow the "
            "dummy elbow-yaw frame (ADR-001/006)."
        ),
    }
    if yaml_chain is not None:
        out["yaml_chain"] = yaml_chain
    return out
