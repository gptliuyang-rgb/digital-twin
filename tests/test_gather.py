"""SONIC §S7 YAML-driven observation gathering. No simulator imports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import G1_DECODER_INPUT_DIM, TOKEN_DIM
from wbc.gather import (
    GATHER_YAML,
    HardwareHold,
    HardwareSnapshot,
    ObsGather,
    ObsGatherError,
    StateLogger,
    compile_observations,
    grouped_history_to_interleaved,
    interleaved_history_to_grouped,
    load_gather_cfg,
    parse_history_name,
    remap_identity,
    to_policy_snapshot,
)
from wbc.observation import ProprioHistory
from wbc.stream import OPERATOR_INPUT_HZ, PLANNER_HZ, POLICY_HZ, STREAM_HZ


def _identity_quat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def _hw(*, t_s: float, q0: float = 0.0, omega_z: float = 0.0, a0: float = 0.0) -> HardwareSnapshot:
    n = 25
    q = np.zeros(n)
    q[0] = q0
    a = np.zeros(n)
    a[0] = a0
    omega = np.array([0.0, 0.0, omega_z])
    return HardwareSnapshot(
        t_s=t_s,
        q_hw=q,
        dq_hw=np.zeros(n),
        omega_imu=omega,
        imu_quat_wxyz=_identity_quat(),
        last_action=a,
    )


def test_yaml_locks_paper_rates_and_t800_dim() -> None:
    cfg = load_gather_cfg()
    assert cfg["not_dexhand2_contact"] is True
    assert cfg["not_table_s4"] is True
    assert cfg["not_hand_mit_ring"] is True
    assert cfg["not_invented_latency"] is True
    assert cfg["not_pico_sdk"] is True
    assert cfg["policy_hz"] == POLICY_HZ == 50
    assert cfg["command_stream_hz"] == STREAM_HZ == 500
    assert cfg["operator_input_hz"] == OPERATOR_INPUT_HZ == 100
    assert cfg["planner_hz"] == PLANNER_HZ == 10
    assert cfg["n_dof"] == 25
    assert cfg["sensor_delay_ticks"] == 0
    assert cfg["expected_total_dim"] == 874
    assert cfg["g1_total_dim_forbidden"] == G1_DECODER_INPUT_DIM
    slots, total = compile_observations(cfg)
    assert total == 874
    assert [s.name for s in slots] == [
        "token_state",
        "his_base_angular_velocity_10frame_step1",
        "his_body_joint_positions_10frame_step1",
        "his_body_joint_velocities_10frame_step1",
        "his_last_actions_10frame_step1",
        "his_gravity_dir_10frame_step1",
    ]
    assert [s.dim for s in slots] == [64, 30, 250, 250, 250, 30]
    assert slots[0].offset == 0
    assert slots[1].offset == 64
    assert slots[-1].offset == 64 + 30 + 250 + 250 + 250


def test_parse_history_name() -> None:
    assert parse_history_name("his_body_joint_positions_10frame_step1") == (
        "his_body_joint_positions",
        10,
        1,
    )
    assert parse_history_name("token_state") is None


def test_g1_dof_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        remap_identity(np.zeros(29), 25)


def test_hand_concat_refused() -> None:
    with pytest.raises(ObsGatherError, match="DexHand2"):
        remap_identity(np.zeros(45), 25)


def test_hand_observation_name_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(GATHER_YAML.read_text(encoding="utf-8"))
    raw["observations"] = [{"name": "left_hand_q", "enabled": True}]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_gather_cfg(path)
    with pytest.raises(ObsGatherError, match="DexHand2"):
        compile_observations(cfg)


def test_invented_delay_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(GATHER_YAML.read_text(encoding="utf-8"))
    raw["sensor_delay_ticks"] = 2
    path = tmp_path / "delay.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ObsGatherError, match="measured"):
        load_gather_cfg(path)


def test_startup_zero_pad_and_grouped_layout() -> None:
    gather = ObsGather()
    gather.push_hw(_hw(t_s=0.0, q0=0.3, omega_z=0.1, a0=-0.2))
    token = np.linspace(0.0, 1.0, TOKEN_DIM)
    obs = gather.control_tick(token, t_s=0.0)
    assert obs.shape == (874,)
    np.testing.assert_allclose(obs[:64], token)
    # Oldest 9 frames zero; current frame is last 3 of each block.
    omega = obs[64 : 64 + 30].reshape(10, 3)
    q = obs[64 + 30 : 64 + 30 + 250].reshape(10, 25)
    a = obs[64 + 30 + 250 + 250 : 64 + 30 + 250 + 250 + 250].reshape(10, 25)
    grav = obs[-30:].reshape(10, 3)
    np.testing.assert_allclose(omega[:9], 0.0)
    np.testing.assert_allclose(omega[-1], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(q[:9], 0.0)
    np.testing.assert_allclose(q[-1, 0], 0.3)
    np.testing.assert_allclose(a[-1, 0], -0.2)
    np.testing.assert_allclose(grav[-1], [0.0, 0.0, -1.0])


def test_ten_ticks_fill_history() -> None:
    gather = ObsGather()
    token = np.zeros(TOKEN_DIM)
    for i in range(10):
        gather.push_hw(_hw(t_s=i / 50.0, q0=float(i)))
        obs = gather.control_tick(token, t_s=i / 50.0)
    q = obs[64 + 30 : 64 + 30 + 250].reshape(10, 25)
    np.testing.assert_allclose(q[:, 0], np.arange(10.0))


def test_latest_data_wins_between_50hz_ticks() -> None:
    gather = ObsGather()
    token = np.zeros(TOKEN_DIM)
    # 500 Hz pushes; only the last snapshot before the control tick is logged.
    for k in range(10):
        gather.push_hw(_hw(t_s=k / 500.0, q0=float(k)))
    obs = gather.control_tick(token, t_s=0.0)
    q = obs[64 + 30 : 64 + 30 + 250].reshape(10, 25)
    np.testing.assert_allclose(q[-1, 0], 9.0)
    assert len(gather.logger) == 1


def test_grouped_interleaved_roundtrip() -> None:
    gather = ObsGather()
    token = np.arange(TOKEN_DIM, dtype=np.float64)
    hist = ProprioHistory()
    grouped = None
    for i in range(10):
        q = np.zeros(25)
        q[0] = float(i)
        a = np.zeros(25)
        a[0] = -float(i)
        omega = np.array([0.0, 0.0, 0.05 * i])
        gather.push_hw(_hw(t_s=i / 50.0, q0=float(i), omega_z=0.05 * i, a0=-float(i)))
        grouped = gather.control_tick(token, t_s=i / 50.0)
        hist.push(q, np.zeros(25), omega, a, 0.0)
    assert grouped is not None
    interleaved = hist.decoder_input(token)
    converted = grouped_history_to_interleaved(grouped, 25)
    np.testing.assert_allclose(converted, interleaved)
    back = interleaved_history_to_grouped(converted, 25)
    np.testing.assert_allclose(back, grouped)
    assert converted.shape == (874,)
    assert not np.allclose(grouped, interleaved)


def test_hardware_hold_empty() -> None:
    hold = HardwareHold()
    with pytest.raises(ObsGatherError, match="empty"):
        hold.read()


def test_to_policy_snapshot_heading() -> None:
    snap = to_policy_snapshot(_hw(t_s=0.0, omega_z=0.2), 25)
    np.testing.assert_allclose(snap.omega_heading, [0.0, 0.0, 0.2])
    np.testing.assert_allclose(snap.gravity_heading, [0.0, 0.0, -1.0])


def test_csv_optional(tmp_path: Path) -> None:
    logger = StateLogger(capacity=16, n_dof=25, csv_dir=tmp_path)
    gather = ObsGather()
    gather.push_hw(_hw(t_s=0.0, q0=1.0))
    policy = to_policy_snapshot(gather.hw.read(), 25)
    logger.push(policy)
    logger.close()
    q_csv = (tmp_path / "q.csv").read_text(encoding="utf-8").strip().splitlines()
    assert q_csv[0].startswith("t_s")
    assert len(q_csv) == 2


def test_push_last_action_overwrites_without_inventing_q() -> None:
    hold = HardwareHold()
    hold.push(_hw(t_s=0.1, q0=2.0, a0=0.0))
    a = np.zeros(25)
    a[0] = 4.5
    hold.push_last_action(a)
    snap = hold.read()
    np.testing.assert_allclose(snap.q_hw[0], 2.0)
    np.testing.assert_allclose(snap.last_action[0], 4.5)
    assert snap.t_s == pytest.approx(0.1)
    with pytest.raises(G1CheckpointIncompatible):
        hold.push_last_action(np.zeros(29))
    with pytest.raises(ObsGatherError, match="DexHand2"):
        hold.push_last_action(np.zeros(45))
    empty = HardwareHold()
    with pytest.raises(ObsGatherError, match="empty"):
        empty.push_last_action(np.zeros(25))
