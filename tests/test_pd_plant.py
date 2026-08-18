"""500 Hz joint-PD plant on ZOH-held 25-D policy_action. No simulator imports."""

from __future__ import annotations

import numpy as np
import pytest

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.pd_plant import (
    GAINS_SOURCE,
    PdPlantError,
    PolicyPdPlant,
    apply_policy_pd_stream,
    joint_pd_torque_nm,
    load_pd_plant_cfg,
    refuse_pd_plant_as_decoder_run,
    refuse_pd_plant_finite_diff_dq,
    refuse_pd_plant_hermite,
    require_t800_pd_state,
)
from wbc.pd_stand import pd_stand_kp_kd
from wbc.stream import stream_policy_action_zoh


def test_yaml_locks_pd_plant() -> None:
    cfg = load_pd_plant_cfg()
    assert cfg["adr"] == "ADR-055"
    assert cfg["pd_plant_on_same_tick"] is True
    assert cfg["pd_plant_gains_are_pd_stand_bringup"] is True
    assert cfg["not_sonic_tracking_gains"] is True
    assert cfg["pd_plant_dq_des_is_zero"] is True
    assert cfg["not_pd_plant_from_decoder_onnx"] is True
    assert cfg["not_pd_plant_hermite"] is True
    assert cfg["not_pd_plant_finite_diff_dq"] is True
    assert cfg["not_pd_tau_onto_decoder_obs"] is True
    assert cfg["gains_source"] == GAINS_SOURCE
    assert cfg["pd_action_dim"] == 25
    assert cfg["command_stream_hz"] == 500
    assert cfg["factor_policy_to_stream"] == 10


def test_torque_matches_pd_stand_formula() -> None:
    kp, kd = pd_stand_kp_kd()
    q = np.zeros(25)
    q[0] = 7.0
    dq = np.zeros(25)
    dq[0] = 0.5
    q_des = np.zeros(25)
    q_des[0] = 3.5
    tau = joint_pd_torque_nm(q, dq, q_des)
    expected = kp * (q_des - q) - kd * dq
    np.testing.assert_allclose(tau, expected)
    assert tau[0] == pytest.approx(kp[0] * (3.5 - 7.0) - kd[0] * 0.5)


def test_zoh_period_is_ten_identical_samples() -> None:
    plant = PolicyPdPlant()
    q = np.zeros(25)
    q[0] = 1.0
    dq = np.zeros(25)
    q_des = np.zeros(25)
    q_des[0] = 6.0
    period = plant.zoh_period(q, dq, q_des, t0_s=0.0)
    assert period.n_steps == 10
    assert period.rate_hz == 500
    assert period.hold == "zoh"
    assert period.gains_source == GAINS_SOURCE
    assert period.dq_des_rad_s == 0.0
    np.testing.assert_allclose(period.q_des_rad[:, 0], 6.0)
    np.testing.assert_allclose(period.q_rad[:, 0], 1.0)
    expected0 = plant.kp[0] * (6.0 - 1.0)
    np.testing.assert_allclose(period.tau_nm[:, 0], expected0)
    np.testing.assert_allclose(plant.read()[0], expected0)


def test_zoh_stream_two_actions() -> None:
    a0 = np.zeros(25)
    a0[0] = 1.0
    a1 = np.zeros(25)
    a1[0] = 2.0
    stream = stream_policy_action_zoh(np.stack([a0, a1]))
    q = np.zeros(25)
    dq = np.zeros(25)
    out = apply_policy_pd_stream(stream, q, dq)
    assert out.n_steps == 11
    kp, _kd = pd_stand_kp_kd()
    np.testing.assert_allclose(out.tau_nm[:10, 0], kp[0] * 1.0)
    np.testing.assert_allclose(out.tau_nm[10, 0], kp[0] * 2.0)


def test_caller_500hz_q_is_not_interpolated() -> None:
    plant = PolicyPdPlant()
    q_des = np.zeros(25)
    q_des[0] = 2.0
    q_stream = np.zeros((10, 25))
    q_stream[:, 0] = np.linspace(0.0, 0.9, 10)
    dq = np.zeros(25)
    period = plant.zoh_period(q_stream, dq, q_des)
    expected = plant.kp[0] * (2.0 - q_stream[:, 0])
    np.testing.assert_allclose(period.tau_nm[:, 0], expected)


def test_wrong_length_q_stream_refused() -> None:
    plant = PolicyPdPlant()
    q_des = np.zeros(25)
    with pytest.raises(PdPlantError, match="Do not interpolate"):
        plant.zoh_period(np.zeros((9, 25)), np.zeros(25), q_des)


def test_g1_hands_qpos_command_schema_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        require_t800_pd_state(np.zeros(29))
    with pytest.raises(PdPlantError, match="planner qpos"):
        require_t800_pd_state(np.zeros(32))
    with pytest.raises(PdPlantError, match="DexHand2"):
        require_t800_pd_state(np.zeros(45))
    with pytest.raises(PdPlantError, match="command_schema"):
        require_t800_pd_state(np.zeros(75))
    with pytest.raises(PdPlantError, match="ZOH"):
        refuse_pd_plant_hermite()
    with pytest.raises(PdPlantError, match="decoder ONNX"):
        refuse_pd_plant_as_decoder_run()
    with pytest.raises(PdPlantError, match="finite-diff"):
        refuse_pd_plant_finite_diff_dq()


def test_g1_action_as_q_des_refused() -> None:
    q = np.zeros(25)
    dq = np.zeros(25)
    with pytest.raises(G1CheckpointIncompatible):
        joint_pd_torque_nm(q, dq, np.zeros(29))
