"""500 Hz physics plant on ZOH-held 25-D policy_action. No simulator imports."""

from __future__ import annotations

import numpy as np
import pytest

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.pd_physics import (
    GAINS_SOURCE,
    PHYSICS_N_STEPS,
    PHYSICS_TIMESTEP_S,
    PdPhysicsError,
    PdPhysicsPeriod,
    as_plant_stream,
    load_pd_physics_cfg,
    refuse_omit_push_hw_invent_imu,
    refuse_pd_physics_as_decoder_run,
    refuse_pd_physics_finite_diff_dq,
    refuse_pd_physics_hermite,
    refuse_pd_physics_invent_imu,
    refuse_pd_physics_q_onto_this_tick_decoder,
    require_physics_timestep_500hz,
    require_t800_tau_nm,
    run_zoh_period,
)
from wbc.pd_plant import PdPlantError, PolicyPdPlant
from wbc.pd_stand import pd_stand_kp_kd


class _HoldPhysics:
    n_dof = 25
    timestep_s = 0.002

    def __init__(self, q0: float = 1.0) -> None:
        self.q = np.zeros(25)
        self.q[0] = q0
        self.dq = np.zeros(25)
        self.n_applies = 0

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.dq.copy()

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del tau_nm
        self.n_applies += 1
        return self.read_q_dq()


def test_yaml_locks_pd_physics() -> None:
    cfg = load_pd_physics_cfg()
    assert cfg["adr"] == "ADR-057"
    assert cfg["pd_physics_on_same_tick"] is True
    assert cfg["pd_physics_is_optional"] is True
    assert cfg["not_pd_physics_from_decoder_onnx"] is True
    assert cfg["not_pd_physics_hermite"] is True
    assert cfg["not_pd_physics_finite_diff_dq"] is True
    assert cfg["not_pd_physics_q_onto_this_tick_decoder"] is True
    assert cfg["not_pd_physics_invent_imu"] is True
    assert cfg["pd_physics_feeds_next_tick_decoder_q"] is True
    assert cfg["omit_push_hw_keeps_imu"] is True
    assert cfg["pd_physics_n_steps_per_tick"] == PHYSICS_N_STEPS == 10
    assert cfg["pd_physics_timestep_s"] == PHYSICS_TIMESTEP_S == 0.002
    assert cfg["gains_source"] == GAINS_SOURCE
    assert cfg["pd_action_dim"] == 25


def test_hold_physics_ten_identical_tau() -> None:
    plant = PolicyPdPlant()
    physics = _HoldPhysics(q0=1.0)
    q_des = np.zeros(25)
    q_des[0] = 6.0
    period = run_zoh_period(plant, physics, q_des, t0_s=0.0)
    assert period.n_steps == 10
    assert period.rate_hz == 500
    assert period.hold == "zoh"
    assert period.gains_source == GAINS_SOURCE
    assert period.dq_des_rad_s == 0.0
    assert physics.n_applies == 10
    expected0 = plant.kp[0] * (6.0 - 1.0)
    np.testing.assert_allclose(period.tau_nm[:, 0], expected0)
    np.testing.assert_allclose(period.q_rad[:, 0], 1.0)
    stream = as_plant_stream(period)
    np.testing.assert_allclose(stream.tau_nm[:, 0], expected0)


def test_wrong_timestep_refused() -> None:
    with pytest.raises(PdPhysicsError, match="1/500"):
        require_physics_timestep_500hz(0.001)


def test_g1_hands_qpos_tau_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        require_t800_tau_nm(np.zeros(29))
    with pytest.raises(PdPlantError, match="planner qpos"):
        require_t800_tau_nm(np.zeros(32))
    with pytest.raises(PdPlantError, match="DexHand2"):
        require_t800_tau_nm(np.zeros(45))
    with pytest.raises(PdPlantError, match="command_schema"):
        require_t800_tau_nm(np.zeros(75))
    with pytest.raises(PdPhysicsError, match="ZOH"):
        refuse_pd_physics_hermite()
    with pytest.raises(PdPhysicsError, match="decoder ONNX"):
        refuse_pd_physics_as_decoder_run()
    with pytest.raises(PdPhysicsError, match="finite-diff"):
        refuse_pd_physics_finite_diff_dq()
    with pytest.raises(PdPhysicsError, match="before"):
        refuse_pd_physics_q_onto_this_tick_decoder()
    with pytest.raises(PdPhysicsError, match="invent IMU"):
        refuse_pd_physics_invent_imu()
    with pytest.raises(PdPhysicsError, match="omitting push_hw"):
        refuse_omit_push_hw_invent_imu()


def test_period_type() -> None:
    period = PdPhysicsPeriod(
        t_s=np.linspace(0.002, 0.02, 10),
        q_des_rad=np.zeros((10, 25)),
        q_rad=np.zeros((10, 25)),
        dq_rad_s=np.zeros((10, 25)),
        tau_nm=np.zeros((10, 25)),
        rate_hz=500,
        src_hz=50,
        horizon_s=0.02,
        hold="zoh",
        gains_source=GAINS_SOURCE,
        dq_des_rad_s=0.0,
        timestep_s=0.002,
    )
    assert period.n_steps == 10
    kp, _kd = pd_stand_kp_kd()
    assert kp.shape == (25,)
