"""1 kHz DexHand2 MIT ring. No simulator imports. Hands bypass WBC."""

from __future__ import annotations

import numpy as np
import pytest

from hand.mit_ring import (
    GAINS_SOURCE,
    HAND_N_STEPS,
    HAND_TIMESTEP_S,
    HandMitPlant,
    MitRingError,
    MitRingPeriod,
    load_mit_ring_cfg,
    refuse_force_mode_without_hardware_tau,
    refuse_hand_q_hermite,
    refuse_hand_q_onto_wbc_last_action,
    refuse_hardware_kp,
    refuse_invented_latency,
    refuse_mit_ring_as_decoder_run,
    refuse_mit_ring_finite_diff_dq,
    refuse_sim_forcerange_as_payload_rating,
    require_hand_q,
    require_hand_timestep_1khz,
    run_mit_period,
    sim_forcerange_nm,
    sim_kp_kd,
)
from interface.schema import CommandVector
from runtime.hand_bypass import (
    hand_targets_from_command,
    require_wbc_action_untouched,
    run_bimanual_period,
)
from wbc.stream import load_stream_cfg


class _HoldHand:
    n_dof = 20
    timestep_s = 0.001

    def __init__(self, q0: float = 0.1) -> None:
        self.q = np.zeros(20)
        self.q[0] = q0
        self.dq = np.zeros(20)
        self.n_applies = 0
        self.last_tau = np.zeros(20)

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.dq.copy()

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.last_tau = np.asarray(tau_nm, dtype=np.float64).copy()
        self.n_applies += 1
        return self.read_q_dq()


def test_yaml_locks_mit_ring() -> None:
    cfg = load_mit_ring_cfg()
    assert cfg["adr"] == "ADR-058"
    assert cfg["hand_mit_ring"] is True
    assert cfg["not_wbc_last_action"] is True
    assert cfg["not_concat_t800_plus_hand"] is True
    assert cfg["hand_q_is_zoh"] is True
    assert cfg["not_hardware_kp"] is True
    assert cfg["not_invented_latency"] is True
    assert cfg["mit_n_steps_per_tick"] == HAND_N_STEPS == 20
    assert cfg["mit_timestep_s"] == HAND_TIMESTEP_S == 0.001
    assert cfg["gains_source"] == GAINS_SOURCE
    assert cfg["n_active_dof"] == 20


def test_wbc_yaml_still_bypasses_hands() -> None:
    stream = load_stream_cfg()
    assert stream["not_hand_mit_ring"] is True


def test_hold_physics_twenty_identical_tau() -> None:
    plant = HandMitPlant.from_spec()
    physics = _HoldHand(q0=0.1)
    q_des = np.zeros(20)
    q_des[0] = 0.3
    period = run_mit_period(plant, physics, q_des, t0_s=0.0, side="right")
    assert period.n_steps == 20
    assert period.rate_hz == 1000
    assert period.hold == "zoh"
    assert period.gains_source == GAINS_SOURCE
    assert period.dq_des_rad_s == 0.0
    assert physics.n_applies == 20
    expected0 = plant.kp[0] * (0.3 - 0.1)
    np.testing.assert_allclose(period.tau_nm[:, 0], expected0)
    np.testing.assert_allclose(period.q_rad[:, 0], 0.1)


def test_tau_clipped_to_sim_forcerange() -> None:
    plant = HandMitPlant.from_spec()
    physics = _HoldHand(q0=0.0)
    q_des = np.full(20, 10.0)
    period = run_mit_period(plant, physics, q_des)
    fr = sim_forcerange_nm()
    assert np.all(np.abs(period.tau_nm) <= fr + 1e-12)


def test_wrong_timestep_refused() -> None:
    with pytest.raises(MitRingError, match="1/1000"):
        require_hand_timestep_1khz(0.002)


def test_wrong_dims_refused() -> None:
    with pytest.raises(MitRingError, match="T800 WBC"):
        require_hand_q(np.zeros(25))
    with pytest.raises(MitRingError, match="G1 29"):
        require_hand_q(np.zeros(29))
    with pytest.raises(MitRingError, match="planner qpos"):
        require_hand_q(np.zeros(32))
    with pytest.raises(MitRingError, match="concatenates"):
        require_hand_q(np.zeros(45))
    with pytest.raises(MitRingError, match="command_schema"):
        require_hand_q(np.zeros(75))
    with pytest.raises(MitRingError, match="ZOH"):
        refuse_hand_q_hermite()
    with pytest.raises(MitRingError, match="last_action"):
        refuse_hand_q_onto_wbc_last_action()
    with pytest.raises(MitRingError, match="decoder ONNX"):
        refuse_mit_ring_as_decoder_run()
    with pytest.raises(MitRingError, match="finite-diff"):
        refuse_mit_ring_finite_diff_dq()
    with pytest.raises(MitRingError, match="hardware_kp"):
        refuse_hardware_kp()
    with pytest.raises(MitRingError, match="command_latency_ms"):
        refuse_invented_latency()
    with pytest.raises(MitRingError, match="motor_max_torque_nm"):
        refuse_force_mode_without_hardware_tau()
    with pytest.raises(MitRingError, match="payload rating"):
        refuse_sim_forcerange_as_payload_rating()


def test_bimanual_command_bypass() -> None:
    cmd = CommandVector.zeros()
    cmd.left_hand_q[0] = 0.2
    cmd.right_hand_q[0] = 0.4
    left_q, right_q = hand_targets_from_command(cmd)
    assert left_q[0] == pytest.approx(0.2)
    assert right_q[0] == pytest.approx(0.4)
    left = _HoldHand(q0=0.0)
    right = _HoldHand(q0=0.0)
    period = run_bimanual_period(cmd, left, right)
    assert period.n_steps == 20
    assert period.tool_trigger == 0
    kp, _kd = sim_kp_kd()
    np.testing.assert_allclose(period.left.tau_nm[:, 0], kp[0] * 0.2)
    np.testing.assert_allclose(period.right.tau_nm[:, 0], kp[0] * 0.4)
    untouched = require_wbc_action_untouched(np.zeros(25))
    assert untouched.shape == (25,)
    with pytest.raises(MitRingError, match="T800 25"):
        require_wbc_action_untouched(np.zeros(45))


def test_force_mode_refused_on_command() -> None:
    cmd = CommandVector.zeros()
    cmd.left_hand_mode = 2
    with pytest.raises(MitRingError, match="motor_max_torque_nm"):
        hand_targets_from_command(cmd)


def test_period_type() -> None:
    period = MitRingPeriod(
        t_s=np.linspace(0.001, 0.02, 20),
        q_des_rad=np.zeros((20, 20)),
        q_rad=np.zeros((20, 20)),
        dq_rad_s=np.zeros((20, 20)),
        tau_nm=np.zeros((20, 20)),
        rate_hz=1000,
        src_hz=50,
        horizon_s=0.02,
        hold="zoh",
        gains_source=GAINS_SOURCE,
        dq_des_rad_s=0.0,
        timestep_s=0.001,
        side="right",
    )
    assert period.n_steps == 20
    kp, kd = sim_kp_kd()
    assert kp.shape == (20,)
    assert kd.shape == (20,)
