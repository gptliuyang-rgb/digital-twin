"""1 kHz DexHand2 MuJoCo HandMitPhysics. Combined T800+Hand stays blocked."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from hand.mit_ring import (
    GAINS_SOURCE,
    HAND_N_STEPS,
    HAND_TIMESTEP_S,
    HandMitPlant,
    MitRingError,
    require_hand_q,
    run_mit_period,
)
from interface.schema import CommandVector, load_hand_spec
from runtime.hand_bypass import require_wbc_action_untouched, run_bimanual_period
from sim.mujoco_env.hand_mit_physics import (
    fixture_mjcf,
    load_hand_mit_physics_cfg,
    official_inertia_mjcf,
    official_mjcf_present,
    plant_mass_kg,
    refuse_body_500hz_dt,
    refuse_combined_robot,
    refuse_official_002s_dt,
    refuse_official_position_actuators,
    refuse_treat_fixture_as_cad,
    resolve_source,
)
from wbc.stream import load_stream_cfg


def test_yaml_locks_hand_mit_physics() -> None:
    cfg = load_hand_mit_physics_cfg()
    assert cfg["adr"] == "ADR-060"
    assert cfg["hand_mit_physics"] is True
    assert cfg["not_t800_weld"] is True
    assert cfg["not_official_position_actuators"] is True
    assert cfg["not_body_500hz_dt"] is True
    assert cfg["not_official_002s_dt"] is True
    assert cfg["not_hardware_kp"] is True
    assert cfg["not_invented_latency"] is True
    assert cfg["not_treat_fixture_as_cad"] is True
    assert cfg["official_inertias_from_skeleton_mjcf"] is True
    assert cfg["not_soft_body_in_inertia_plant"] is True
    assert cfg["mit_timestep_s"] == HAND_TIMESTEP_S == 0.001
    assert cfg["n_active_dof"] == 20
    assert cfg["gains_source"] == GAINS_SOURCE
    assert cfg["fixture_inertias_are_placeholders"] is True


def test_wbc_yaml_still_bypasses_hands() -> None:
    stream = load_stream_cfg()
    assert stream["not_hand_mit_ring"] is True


def test_combined_robot_refused() -> None:
    with pytest.raises(PolicyEvalBlocked, match="weld"):
        refuse_combined_robot()
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    with pytest.raises(PolicyEvalBlocked, match="weld"):
        HandMitMujocoEnv(include_t800=True)


def test_official_position_and_500hz_refused() -> None:
    with pytest.raises(MitRingError, match="<position>"):
        refuse_official_position_actuators()
    with pytest.raises(MitRingError, match="1/1000"):
        refuse_body_500hz_dt()
    with pytest.raises(MitRingError, match="1/1000"):
        refuse_official_002s_dt()
    with pytest.raises(MitRingError, match="placeholders"):
        refuse_treat_fixture_as_cad()
    with pytest.raises(MitRingError, match="T800 WBC"):
        require_hand_q(np.zeros(25), name="tau")


def test_fixture_xml_has_20_motors_and_1ms() -> None:
    xml = fixture_mjcf(side="right")
    assert xml.count("<motor ") == 20
    assert "<position " not in xml
    assert 'timestep="0.001"' in xml
    assert 'gravity="0 0 0"' in xml
    assert "r_THJ0" in xml
    assert "r_LFJ3" in xml
    assert "LINK_BASE" not in xml
    assert "t800" not in xml.lower()


def test_auto_source_never_loads_position_xml() -> None:
    src = resolve_source("auto")
    assert src in {"fixture", "official_inertia"}
    assert src != "official_position"
    if official_mjcf_present():
        assert src == "official_inertia"
    else:
        assert src == "fixture"


def test_fixture_timestep_is_1khz() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    env = HandMitMujocoEnv(source="fixture")
    env.reset()
    assert env.n_dof == 20
    assert env.timestep_s == pytest.approx(HAND_TIMESTEP_S)
    assert env.source == "fixture"
    assert env.xml_note == "fixture_20dof_motor_1khz"


def test_apply_tau_wrong_dims_refused() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    env = HandMitMujocoEnv(source="fixture")
    env.reset()
    with pytest.raises(MitRingError, match="T800 WBC"):
        env.apply_tau_nm(np.zeros(25))
    with pytest.raises(MitRingError, match="G1 29"):
        env.apply_tau_nm(np.zeros(29))
    with pytest.raises(MitRingError, match="concatenates"):
        env.apply_tau_nm(np.zeros(45))
    with pytest.raises(MitRingError, match="command_schema"):
        env.apply_tau_nm(np.zeros(75))


def test_apply_tau_and_step_moves_hinge() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    env = HandMitMujocoEnv(source="fixture")
    env.reset()
    tau = np.zeros(20)
    tau[0] = 0.05
    q0 = env.get_q().copy()
    q, dq = env.apply_tau_and_step(tau)
    assert q.shape == (20,)
    assert dq.shape == (20,)
    assert np.isfinite(q).all()
    assert not np.allclose(q, q0)


def test_mit_period_on_fixture_tracks_q_des() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    env = HandMitMujocoEnv(source="fixture")
    env.reset()
    plant = HandMitPlant.from_spec()
    q_des = np.zeros(20)
    q_des[0] = 0.5
    last = None
    for i in range(10):
        last = run_mit_period(plant, env, q_des, t0_s=i / 50.0, side="right")
    assert last is not None
    assert last.n_steps == HAND_N_STEPS
    assert last.rate_hz == 1000
    assert last.hold == "zoh"
    assert last.gains_source == GAINS_SOURCE
    assert last.q_rad[-1, 0] > 0.1
    assert last.q_rad[-1, 0] < 0.6
    untouched = require_wbc_action_untouched(np.zeros(25))
    assert untouched.shape == (25,)


def test_bimanual_period_two_fixtures() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    left = HandMitMujocoEnv(side="left", source="fixture")
    right = HandMitMujocoEnv(side="right", source="fixture")
    left.reset()
    right.reset()
    cmd = CommandVector.zeros()
    cmd.left_hand_q[0] = 0.3
    cmd.right_hand_q[0] = 0.4
    period = run_bimanual_period(cmd, left, right)
    assert period.n_steps == 20
    assert period.left.q_rad[-1, 0] != pytest.approx(0.0, abs=1e-6)
    assert period.right.q_rad[-1, 0] != pytest.approx(0.0, abs=1e-6)
    with pytest.raises(MitRingError, match="T800 25"):
        require_wbc_action_untouched(np.zeros(45))


def test_official_inertia_xml_is_motor_1khz_meshfree() -> None:
    if not official_mjcf_present():
        pytest.skip("wuji-description not cloned")
    xml = official_inertia_mjcf(side="right")
    assert xml.count("<motor ") == 20
    assert "<position " not in xml
    assert 'timestep="0.001"' in xml
    assert 'gravity="0 0 0"' in xml
    assert 'timestep="0.002"' not in xml
    assert 'type="mesh"' not in xml
    assert "<asset>" not in xml
    assert "<contact>" not in xml
    assert "r_THJ0" in xml
    assert "r_LFJ3" in xml
    assert "LINK_BASE" not in xml
    from assets.dexhand2.build.gen_derived import parse_inertial_mass_kg

    mass = parse_inertial_mass_kg(xml)
    spec = load_hand_spec()
    assert abs(mass - spec.skeleton_mass_kg) < 1e-3


def test_official_position_source_still_refused() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    with pytest.raises(MitRingError, match="<position>"):
        HandMitMujocoEnv(source="official_position")


@pytest.mark.skipif(not official_mjcf_present(), reason="wuji-description not cloned")
def test_mit_period_on_official_inertia_tracks_and_differs_from_fixture() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    spec = load_hand_spec()
    official = HandMitMujocoEnv(source="official_inertia")
    official.reset()
    assert official.source == "official_inertia"
    assert official.xml_note == "official_inertia_meshfree_mit_1khz"
    assert official.timestep_s == pytest.approx(HAND_TIMESTEP_S)
    assert abs(plant_mass_kg(official.model) - spec.skeleton_mass_kg) < 1e-3

    fixture = HandMitMujocoEnv(source="fixture")
    fixture.reset()
    assert plant_mass_kg(fixture.model) < 0.3

    plant = HandMitPlant.from_spec()
    q_des = np.zeros(20)
    q_des[0] = 0.5
    last_o = None
    last_f = None
    for i in range(10):
        last_o = run_mit_period(plant, official, q_des, t0_s=i / 50.0, side="right")
        last_f = run_mit_period(plant, fixture, q_des, t0_s=i / 50.0, side="right")
    assert last_o is not None and last_f is not None
    assert last_o.n_steps == HAND_N_STEPS
    assert last_o.q_rad[-1, 0] > 0.05
    assert last_o.q_rad[-1, 0] < 0.6
    assert last_f.q_rad[-1, 0] > 0.1
    # Placeholder vs CAD inertias must not produce the same q trajectory.
    assert not np.allclose(last_o.q_rad[-1], last_f.q_rad[-1], atol=1e-6)
    with pytest.raises(MitRingError, match="placeholders"):
        refuse_treat_fixture_as_cad()
