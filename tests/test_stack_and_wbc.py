from __future__ import annotations

import numpy as np
import pytest

from eval.stack_metrics import point_in_support, release_quality, stack_stable
from runtime.wbc_safety import WbcSafetyFilter
from vla.adapters.rotation import matrix_to_rot6d


def test_stack_stable_identity() -> None:
    xy = np.zeros((50, 2))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (50, 1))
    support = np.array([[-0.2, -0.2], [0.2, 0.2]])
    gate = stack_stable(xy, quat, support, dt_s=0.1)
    assert gate.stable
    assert gate.max_drift_m == pytest.approx(0.0)


def test_stack_unstable_when_com_leaves_support() -> None:
    xy = np.zeros((50, 2))
    xy[:, 0] = 0.5
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (50, 1))
    support = np.array([[-0.1, -0.1], [0.1, 0.1]])
    gate = stack_stable(xy, quat, support, dt_s=0.1)
    assert not gate.support_ok
    assert not gate.stable


def test_release_quality_gates() -> None:
    ok = release_quality(0.01, 0.001)
    assert ok.velocity_ok and ok.height_ok
    bad = release_quality(0.2, 0.02)
    assert not bad.velocity_ok and not bad.height_ok


def test_point_in_support() -> None:
    box = np.array([[-1.0, -1.0], [1.0, 1.0]])
    assert point_in_support([0, 0], box)
    assert not point_in_support([2, 0], box)


def test_wbc_filter_rejects_wrist_jump() -> None:
    ident = matrix_to_rot6d(np.eye(3))
    filt = WbcSafetyFilter()
    ok = filt.filter(
        head_pos=np.zeros(3),
        head_rot6d=ident,
        left_wrist_pos=np.zeros(3),
        left_wrist_rot6d=ident,
        right_wrist_pos=np.zeros(3),
        right_wrist_rot6d=ident,
        pelvis_height=0.55,
        nav_cmd=np.zeros(3),
        dt_s=0.02,
    )
    assert ok.accepted
    jump = filt.filter(
        head_pos=np.zeros(3),
        head_rot6d=ident,
        left_wrist_pos=np.array([0.2, 0, 0]),
        left_wrist_rot6d=ident,
        right_wrist_pos=np.zeros(3),
        right_wrist_rot6d=ident,
        pelvis_height=0.55,
        nav_cmd=np.zeros(3),
        dt_s=0.02,
    )
    assert not jump.accepted
    assert jump.reason == "jump_left_wrist_pos"
    np.testing.assert_allclose(jump.left_wrist_pos, 0.0)
