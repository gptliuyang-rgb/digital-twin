"""Rotation conversions. 6D follows Zhou et al. 2019 (two columns of R, Gram-Schmidt)."""

from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero vector in 6D decode")
    return v / n


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = np.asarray(rot6d, dtype=np.float64).reshape(6)[:3]
    a2 = np.asarray(rot6d, dtype=np.float64).reshape(6)[3:]
    b1 = _normalize(a1)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = _normalize(a2_proj)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return np.concatenate([r[:, 0], r[:, 1]])


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def slerp_matrices(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    q0 = matrix_to_quaternion_wxyz(r0)
    q1 = matrix_to_quaternion_wxyz(r1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        q = q / np.linalg.norm(q)
        return quaternion_wxyz_to_matrix(q)
    theta = np.arccos(dot)
    q = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return quaternion_wxyz_to_matrix(q)
