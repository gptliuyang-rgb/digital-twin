"""SONIC v1.1 encoder: heading-normalized 10frame_step5, no root_z, no G1 1751-D ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from interface.schema import CommandVector
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    G1_ENCODER_ONNX_DIM_V1_1,
    encoder_motion_dim,
    t800_encoder_onnx_dim_low_latency,
    t800_encoder_onnx_dim_v1_1,
)
from wbc.gather import (
    GATHER_LOW_LATENCY_YAML,
    GATHER_SONIC_V1_1_YAML,
    GATHER_YAML,
    HardwareSnapshot,
    ObsGather,
    ObsGatherError,
    compile_encoder_observations,
    is_full_ori_name,
    is_heading_ori_name,
    load_gather_cfg,
)
from wbc.motion_ref import (
    MotionFrame,
    heading_corrected_rel_rot6d,
    identity_rot6d,
    look_ahead_indices,
    refuse_g1_encoder_onnx,
)


def _identity_quat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def _hw(*, t_s: float = 0.0, yaw_rad: float = 0.0, pitch_rad: float = 0.0) -> HardwareSnapshot:
    # wxyz: yaw about Z, then pitch about Y. Tests use one axis at a time.
    if abs(pitch_rad) > 1e-12:
        half = pitch_rad / 2.0
        quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    else:
        half = yaw_rad / 2.0
        quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    n = 25
    return HardwareSnapshot(
        t_s=t_s,
        q_hw=np.zeros(n),
        dq_hw=np.zeros(n),
        omega_imu=np.zeros(3),
        imu_quat_wxyz=quat,
        last_action=np.zeros(n),
    )


def _frame(*, q0: float, z_m: float = 1.03, dq0: float = 0.0, yaw_rad: float = 0.0) -> MotionFrame:
    q = np.zeros(25)
    q[0] = q0
    dq = np.zeros(25)
    dq[0] = dq0
    half = yaw_rad / 2.0
    quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    pos = np.array([0.0, 0.0, z_m])
    return MotionFrame(q_ref_rad=q, dq_ref_rad_s=dq, root_pos_m=pos, root_rot_wxyz=quat)


def _window(n: int = 50) -> list[MotionFrame]:
    return [_frame(q0=float(i), z_m=1.03 + 0.001 * i, dq0=0.1 * i) for i in range(n)]


def test_yaml_encoder_locks_t800_831_heading_not_g1_1751() -> None:
    cfg = load_gather_cfg(GATHER_SONIC_V1_1_YAML)
    assert cfg["encoder_variant"] == "sonic_v1_1"
    assert cfg["not_smpl_encoder"] is True
    assert cfg["not_4frame_smpl"] is True
    assert cfg["not_low_latency"] is True
    assert cfg["not_wrist_pose_augmentation_sampler"] is True
    assert cfg["n_wrist_dof"] == 0
    assert cfg["encoder_motion_window_dim"] == 560
    assert cfg["expected_encoder_dim"] == 831
    assert cfg["g1_encoder_onnx_dim_forbidden"] == G1_ENCODER_ONNX_DIM_V1_1
    assert encoder_motion_dim(25, include_root_z=False) == 560
    assert t800_encoder_onnx_dim_v1_1() == 831
    assert t800_encoder_onnx_dim_low_latency() == 831
    slots, total = compile_encoder_observations(cfg)
    assert total == 831
    names = [s.name for s in slots]
    assert names == [
        "encoder_mode_4",
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_anchor_orientation_heading_10frame_step5",
        "motion_anchor_orientation_heading",
        "motion_joint_positions_lowerbody_10frame_step5",
        "motion_joint_velocities_lowerbody_10frame_step5",
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
    ]
    ll_cfg = load_gather_cfg(GATHER_LOW_LATENCY_YAML)
    ll_names = [s.name for s in compile_encoder_observations(ll_cfg)[0]]
    assert names != ll_names
    assert [s.dim for s in slots] == [4, 250, 250, 60, 6, 120, 120, 9, 12]
    assert [s.offset for s in slots] == [0, 4, 254, 504, 564, 570, 690, 810, 819]
    modes = {m["name"]: m for m in cfg["encoder"]["encoder_modes"]}
    assert modes["t800"]["mode_id"] == 0
    assert "motion_anchor_orientation_heading_10frame_step5" in modes["t800"]["required_observations"]
    assert "motion_anchor_orientation_heading" in modes["teleop"]["required_observations"]
    assert all(is_heading_ori_name(n) for n in names if "anchor_orientation" in n)
    assert not any(is_full_ori_name(n) for n in names)


def test_look_ahead_stays_step5() -> None:
    assert look_ahead_indices(0, 10, 5, 50) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


def test_heading_strips_robot_pitch_full_does_not() -> None:
    half = np.pi / 12.0  # 15° → 30° pitch about Y
    robot = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    ref = _identity_quat()
    heading = heading_corrected_rel_rot6d(robot, ref, mode="heading")
    full = heading_corrected_rel_rot6d(robot, ref, mode="full")
    np.testing.assert_allclose(heading, identity_rot6d(), atol=1e-9)
    assert np.linalg.norm(full - identity_rot6d()) > 0.1


def test_assemble_v1_1_t800_mode_uses_heading_ori() -> None:
    gather = ObsGather(cfg=load_gather_cfg(GATHER_SONIC_V1_1_YAML))
    gather.push_hw(_hw(pitch_rad=np.pi / 6))
    gather.push_motion(_window(50), cursor=0)
    enc = gather.assemble_encoder("t800")
    assert enc.shape == (831,)
    np.testing.assert_allclose(enc[:4], [0.0, 0.0, 0.0, 0.0])
    q = enc[4:254].reshape(10, 25)
    np.testing.assert_allclose(q[:, 0], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    ori = enc[504:564].reshape(10, 6)
    ori_now = enc[564:570]
    np.testing.assert_allclose(ori, np.tile(identity_rot6d(), (10, 1)), atol=1e-9)
    np.testing.assert_allclose(ori_now, 0.0)  # teleop-only slot
    np.testing.assert_allclose(enc[570:831], 0.0)


def test_default_full_ori_keeps_robot_pitch() -> None:
    gather = ObsGather(cfg=load_gather_cfg(GATHER_YAML))
    gather.push_hw(_hw(pitch_rad=np.pi / 6))
    gather.push_motion(_window(50), cursor=0)
    enc = gather.assemble_encoder("t800")
    ori = enc[521:581].reshape(10, 6)
    assert np.linalg.norm(ori[0] - identity_rot6d()) > 0.1


def test_v1_1_teleop_fills_lowerbody_and_vr() -> None:
    gather = ObsGather(cfg=load_gather_cfg(GATHER_SONIC_V1_1_YAML))
    gather.push_hw(_hw())
    gather.push_motion(_window(50), cursor=0)
    cmd = CommandVector.zeros()
    cmd.left_wrist_pos = np.array([0.1, 0.2, 0.3])
    cmd.right_wrist_pos = np.array([0.4, 0.5, 0.6])
    cmd.head_pos = np.array([0.7, 0.8, 0.9])
    gather.push_vr_3point(cmd)
    enc = gather.assemble_encoder("teleop")
    assert enc.shape == (831,)
    np.testing.assert_allclose(enc[:4], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(enc[4:504], 0.0)
    np.testing.assert_allclose(enc[504:564], 0.0)
    np.testing.assert_allclose(enc[564:570], identity_rot6d())
    lower_q = enc[570:690].reshape(10, 12)
    np.testing.assert_allclose(lower_q[:, 0], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    np.testing.assert_allclose(enc[810:819], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    np.testing.assert_allclose(enc[819:831], np.array([1.0, 0.0, 0.0, 0.0] * 3))


def test_v1_1_teleop_without_vr_does_not_invent_pico_pose() -> None:
    gather = ObsGather(cfg=load_gather_cfg(GATHER_SONIC_V1_1_YAML))
    gather.push_hw(_hw())
    gather.push_motion(_window(50), cursor=0)
    with pytest.raises(Exception, match="PICO"):
        gather.assemble_encoder("teleop")


def test_smpl_and_wrists_10frame_step1_refused() -> None:
    cfg = load_gather_cfg(GATHER_SONIC_V1_1_YAML)
    cfg["encoder"] = dict(cfg["encoder"])
    cfg["encoder"]["encoder_observations"] = [
        {"name": "smpl_joints_10frame_step1", "enabled": True}
    ]
    cfg["expected_encoder_dim"] = 1
    with pytest.raises(ObsGatherError, match="SMPL"):
        compile_encoder_observations(cfg)
    cfg["encoder"]["encoder_observations"] = [
        {"name": "motion_joint_positions_wrists_10frame_step1", "enabled": True}
    ]
    with pytest.raises(ObsGatherError, match="wrist"):
        compile_encoder_observations(cfg)


def test_g1_v1_1_onnx_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible, match="heading"):
        refuse_g1_encoder_onnx("sonic_v1_1/model_encoder.onnx")
    with pytest.raises(G1CheckpointIncompatible, match="831"):
        refuse_g1_encoder_onnx("gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx")


def test_heading_names_refused_on_default_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load(GATHER_YAML.read_text(encoding="utf-8"))
    raw["encoder"]["encoder_observations"].append(
        {"name": "motion_anchor_orientation_heading_10frame_step5", "enabled": True}
    )
    path = tmp_path / "mix_heading.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ObsGatherError, match="heading"):
        load_gather_cfg(path)


def test_full_ori_refused_on_v1_1_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load(GATHER_SONIC_V1_1_YAML.read_text(encoding="utf-8"))
    raw["encoder"]["encoder_observations"].append(
        {"name": "motion_anchor_orientation_10frame_step5", "enabled": True}
    )
    path = tmp_path / "mix_full.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ObsGatherError, match="heading"):
        load_gather_cfg(path)


def test_root_z_refused_on_v1_1_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load(GATHER_SONIC_V1_1_YAML.read_text(encoding="utf-8"))
    raw["encoder"]["encoder_observations"].append(
        {"name": "motion_root_z_position_10frame_step5", "enabled": True}
    )
    path = tmp_path / "z.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ObsGatherError, match="root_z"):
        load_gather_cfg(path)


def test_step1_refused_on_v1_1_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load(GATHER_SONIC_V1_1_YAML.read_text(encoding="utf-8"))
    raw["encoder"]["encoder_observations"].append(
        {"name": "motion_joint_positions_10frame_step1", "enabled": True}
    )
    path = tmp_path / "step1.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ObsGatherError, match="mix"):
        load_gather_cfg(path)


def test_decoder_dim_unchanged_on_v1_1() -> None:
    gather = ObsGather(cfg=load_gather_cfg(GATHER_SONIC_V1_1_YAML))
    gather.push_hw(_hw())
    gather.push_motion(_window(50), cursor=0)
    dec = gather.control_tick(np.zeros(64), t_s=0.0)
    assert dec.shape == (874,)
