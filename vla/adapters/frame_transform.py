"""World / base / heading-frame SE(3) helpers."""

from __future__ import annotations

import numpy as np

from vla.adapters.rotation import matrix_to_rot6d, rot6d_to_matrix


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def heading_rotation_z(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def world_pose_to_heading(
    pos_world: np.ndarray,
    rot6d_world: np.ndarray,
    pelvis_pos_world: np.ndarray,
    pelvis_yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    r_h = heading_rotation_z(pelvis_yaw_rad)
    rel = np.asarray(pos_world, dtype=np.float64) - np.asarray(pelvis_pos_world, dtype=np.float64)
    pos_h = r_h.T @ rel
    rot_h = r_h.T @ rot6d_to_matrix(rot6d_world)
    return pos_h, matrix_to_rot6d(rot_h)
