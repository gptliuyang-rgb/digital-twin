from __future__ import annotations

import numpy as np

from vla.adapters.rotation import (
    matrix_to_quaternion_wxyz,
    matrix_to_rot6d,
    quaternion_wxyz_to_matrix,
    rot6d_to_matrix,
    rpy_to_matrix,
)


def test_rot6d_roundtrip() -> None:
    rng = np.random.default_rng(0)
    err = []
    for _ in range(10000):
        rpy = rng.uniform(-np.pi, np.pi, size=3)
        r = rpy_to_matrix(*rpy)
        r2 = rot6d_to_matrix(matrix_to_rot6d(r))
        err.append(np.linalg.norm(r2 - r))
    assert max(err) < 1e-6


def test_quat_roundtrip() -> None:
    rng = np.random.default_rng(1)
    for _ in range(10000):
        r = rpy_to_matrix(*rng.uniform(-np.pi, np.pi, size=3))
        r2 = quaternion_wxyz_to_matrix(matrix_to_quaternion_wxyz(r))
        assert np.linalg.norm(r2 - r) < 1e-6
