from __future__ import annotations

import numpy as np
import pytest

from hand.backends.mock_backend import MockBackend
from hand.controller import DexHand2Controller, SafetyLimits
from hand.coupling import Coupling
from hand.grasp_primitives import GraspLibrary
from hand.mit import mit_torque
from interface.schema import load_hand_spec


def _ctrl() -> DexHand2Controller:
    spec = load_hand_spec()
    backend = MockBackend(spec.n_active_dof)
    safety = SafetyLimits(max_delta_q_rad=2.0, velocity_limit_rad_s=10.0)
    return DexHand2Controller(spec, backend, safety=safety)


def test_identity_coupling() -> None:
    spec = load_hand_spec()
    c = Coupling.from_spec(spec)
    q = np.linspace(0.01, 0.2, spec.n_active_dof)
    np.testing.assert_allclose(c.active_to_full(q), q)
    np.testing.assert_allclose(c.full_to_active(q), q)
    assert c.jacobian_full_wrt_active().shape == (spec.n_active_dof, spec.n_active_dof)


def test_mit_formula() -> None:
    q = np.array([0.1])
    tau = mit_torque(
        q, np.array([0.0]), np.array([0.2]), np.array([0.0]), np.array([0.01]), np.array([2.0]), np.array([0.1])
    )
    assert tau[0] == pytest.approx(0.21)


def test_set_targets_and_state() -> None:
    ctrl = _ctrl()
    q = np.zeros(ctrl.spec.n_active_dof)
    q[0] = 0.2
    assert ctrl.set_joint_targets(q)
    st = ctrl.get_state()
    assert st.q_rad[0] == pytest.approx(0.2)


def test_nan_rejected() -> None:
    ctrl = _ctrl()
    q = np.zeros(ctrl.spec.n_active_dof)
    q[1] = np.nan
    assert not ctrl.set_joint_targets(q)
    assert ctrl.last_rejected


def test_jump_rejected() -> None:
    spec = load_hand_spec()
    backend = MockBackend(spec.n_active_dof)
    ctrl = DexHand2Controller(spec, backend, safety=SafetyLimits(max_delta_q_rad=0.05, velocity_limit_rad_s=10.0))
    q = np.zeros(spec.n_active_dof)
    q[0] = 1.0
    assert not ctrl.set_joint_targets(q)
    assert ctrl.last_rejected


def test_limit_clamp() -> None:
    spec = load_hand_spec()
    backend = MockBackend(spec.n_active_dof)
    ctrl = DexHand2Controller(spec, backend, safety=SafetyLimits(max_delta_q_rad=5.0, velocity_limit_rad_s=10.0))
    q = np.full(ctrl.spec.n_active_dof, 10.0)
    assert ctrl.set_joint_targets(q)
    hi = ctrl.spec.limits_vector()[:, 1]
    np.testing.assert_allclose(ctrl.get_state().q_rad, hi)


def test_primitives() -> None:
    spec = load_hand_spec()
    lib = GraspLibrary(spec)
    for name in ("open", "power_grasp", "pinch", "gun_grip", "flat_support"):
        q = lib.q_active(name, 0.5)
        assert q.shape == (20,)
        assert np.isfinite(q).all()
    np.testing.assert_allclose(lib.q_active("open", 1.0), 0)
