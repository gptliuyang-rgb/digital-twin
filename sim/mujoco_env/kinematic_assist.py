"""Kinematic assists for uncalibrated digital-twin demos.

These are **not** contact-validated grasps. They keep props visually attached during
carry/scan playback until E1/E2 friction/stiffness exist (ADR-004/006).
"""

from __future__ import annotations

import numpy as np


def body_pos_mat(model, data, name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = int(model.body(name).id)
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def free_joint_qpos_slice(model, body_name: str) -> slice:
    jnt = int(model.body(body_name).jntadr[0])
    if jnt < 0:
        raise ValueError(f"{body_name} has no joint")
    qadr = int(model.jnt_qposadr[jnt])
    jtype = int(model.jnt_type[jnt])
    if jtype == 0:  # mjJNT_FREE
        return slice(qadr, qadr + 7)
    if jtype in (2, 3):  # hinge/slide
        return slice(qadr, qadr + 1)
    raise ValueError(f"unsupported joint type {jtype} on {body_name}")


def set_free_body_pose(model, data, body_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    sl = free_joint_qpos_slice(model, body_name)
    data.qpos[sl] = np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat_wxyz, dtype=np.float64)])
    jnt = int(model.body(body_name).jntadr[0])
    dadr = int(model.jnt_dofadr[jnt])
    nv = 6 if int(model.jnt_type[jnt]) == 0 else 1
    data.qvel[dadr : dadr + nv] = 0.0


def set_mocap_pose(model, data, body_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    mid = int(model.body(body_name).mocapid[0])
    if mid < 0:
        raise ValueError(f"{body_name} is not a mocap body")
    data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64)
    data.mocap_quat[mid] = np.asarray(quat_wxyz, dtype=np.float64)


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion wxyz (MuJoCo convention)."""
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([(0.25 * s), (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def follow_box_between_wrists(model, data, box_name: str = "box_0", z_offset_m: float = -0.06) -> None:
    lp, _ = body_pos_mat(model, data, "l_wrist")
    rp, _ = body_pos_mat(model, data, "r_wrist")
    center = 0.5 * (lp + rp)
    center[2] += z_offset_m
    set_free_body_pose(model, data, box_name, center, np.array([1.0, 0.0, 0.0, 0.0]))


def lerp_free_body_toward(
    model,
    data,
    body_name: str,
    target_pos: np.ndarray,
    *,
    alpha: float = 0.12,
    quat_wxyz: np.ndarray | None = None,
) -> None:
    pos, _ = body_pos_mat(model, data, body_name)
    new = pos + alpha * (np.asarray(target_pos, dtype=np.float64) - pos)
    q = quat_wxyz if quat_wxyz is not None else np.array([1.0, 0.0, 0.0, 0.0])
    set_free_body_pose(model, data, body_name, new, q)


def stack_center_on_pallet(pallet_pos: np.ndarray, pallet_height_m: float, box_half_z_m: float) -> np.ndarray:
    center = np.asarray(pallet_pos, dtype=np.float64).copy()
    center[2] = float(pallet_pos[2]) + pallet_height_m / 2.0 + box_half_z_m
    return center


def place_box_on_pallet(
    model,
    data,
    box_name: str,
    pallet_pos: np.ndarray,
    *,
    pallet_height_m: float = 0.144,
    box_half_z_m: float = 0.09,
) -> None:
    center = stack_center_on_pallet(pallet_pos, pallet_height_m, box_half_z_m)
    set_free_body_pose(model, data, box_name, center, np.array([1.0, 0.0, 0.0, 0.0]))


def attach_gun_to_right_wrist(model, data, gun_body: str = "scan_gun") -> None:
    pos, R = body_pos_mat(model, data, "r_wrist")
    # Handle sits slightly forward/down in wrist frame; barrel extends -Z local.
    handle = pos + R @ np.array([0.04, -0.02, -0.05], dtype=np.float64)
    barrel_dir = R @ np.array([0.0, 0.0, -1.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    x = barrel_dir / (np.linalg.norm(barrel_dir) + 1e-9)
    z = np.cross(x, up)
    zn = np.linalg.norm(z)
    if zn < 1e-6:
        z = np.array([0.0, 1.0, 0.0])
    else:
        z /= zn
    y = np.cross(z, x)
    Rg = np.column_stack([x, y, z])
    set_mocap_pose(model, data, gun_body, handle, mat_to_quat_wxyz(Rg))


def aim_gun_at_site(model, data, gun_body: str, site_name: str, *, standoff_m: float = 0.16) -> None:
    """Place mocap scan gun so gun_tcp sits at standoff from a QR site (demo-only)."""
    import mujoco

    mujoco.mj_forward(model, data)
    sid = int(model.site(site_name).id)
    qr = data.site_xpos[sid].copy()
    tcp_target = qr + np.array([-standoff_m, 0.0, 0.0], dtype=np.float64)
    look = qr - tcp_target
    look /= np.linalg.norm(look) + 1e-9
    # MuJoCo gun mesh/barrel extends along local -Z; build R with -Z = look.
    z = -look
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    xn = np.linalg.norm(x)
    if xn < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        xn = np.linalg.norm(x)
    x /= xn
    y = np.cross(z, x)
    Rg = np.column_stack([x, y, z])
    gun_sid = int(model.site("gun_tcp").id)
    tcp_local = model.site_pos[gun_sid].copy()
    body_id = int(model.body(gun_body).id)
    grip = tcp_target - Rg @ tcp_local
    set_mocap_pose(model, data, gun_body, grip, mat_to_quat_wxyz(Rg))
