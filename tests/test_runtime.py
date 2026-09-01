from __future__ import annotations

import numpy as np

from runtime.latency_comp import delayed_index, pick_delayed_action
from runtime.safety_filter import CartesianJumpFilter, SafetyFilter
from runtime.temporal_ensemble import TemporalEnsemble
from vla.adapters.rotation import matrix_to_rot6d, rpy_to_matrix


def test_safety_nan_and_jump() -> None:
    n = 4
    filt = SafetyFilter(np.full(n, -1.0), np.full(n, 1.0), max_delta_q=0.2, max_abs_qdot=10.0)
    ok = filt.filter(np.zeros(n), 0.01)
    assert ok.accepted
    bad = filt.filter(np.array([np.nan, 0, 0, 0]), 0.01)
    assert not bad.accepted
    jump = filt.filter(np.full(n, 0.9), 0.01)
    assert not jump.accepted


def test_cartesian_wrist_jump() -> None:
    filt = CartesianJumpFilter(max_delta_m=0.05)
    assert filt.filter(np.zeros(3)).accepted
    assert not filt.filter(np.array([0.08, 0.0, 0.0])).accepted


def test_latency_index() -> None:
    assert delayed_index(0.02, 0.06, 16) == 3
    chunk = np.arange(16 * 2).reshape(16, 2).astype(float)
    np.testing.assert_array_equal(pick_delayed_action(chunk, 0.02, 0.06), chunk[3])


def test_temporal_ensemble_no_rot_flip() -> None:
    ident = matrix_to_rot6d(np.eye(3))
    other = matrix_to_rot6d(rpy_to_matrix(0.0, 0.0, 0.4))
    dim = 6
    h = 4
    ens = TemporalEnsemble(h, alpha=0.2, rot6d_slices=[(0, 6)])
    c0 = np.tile(ident, (h, 1))
    c1 = np.tile(other, (h, 1))
    ens.push(c0)
    ens.push(c1)
    v = ens.value_at(0)
    assert v.shape == (dim,)
    assert np.isfinite(v).all()
