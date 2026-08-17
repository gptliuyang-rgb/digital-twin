"""SONIC encoder motion_* 10frame_step5 look-ahead. No simulator, no invented clip."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import G1_ENCODER_MOTION_DIM, TOKEN_DIM, encoder_motion_dim
from wbc.gather import (
    GATHER_YAML,
    HardwareSnapshot,
    ObsGather,
    ObsGatherError,
    compile_encoder_observations,
    compile_observations,
    load_gather_cfg,
    parse_history_name,
)
from wbc.motion_ref import (
    MotionCursor,
    MotionFrame,
    MotionHold,
    MotionRefError,
    heading_corrected_rel_rot6d,
    identity_rot6d,
    look_ahead_indices,
    refuse_g1_encoder_onnx,
)


def _identity_quat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def _hw(*, t_s: float = 0.0, yaw_rad: float = 0.0) -> HardwareSnapshot:
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


def test_yaml_encoder_locks_t800_570_not_g1_650() -> None:
    cfg = load_gather_cfg()
    assert cfg["not_invented_clip"] is True
    assert cfg["not_g1_encoder_onnx"] is True
    assert cfg["n_wrist_dof"] == 0
    assert cfg["n_lower_body_dof"] == 12
    assert cfg["expected_encoder_dim"] == 570
    assert cfg["g1_encoder_dim_forbidden"] == G1_ENCODER_MOTION_DIM
    assert encoder_motion_dim(25) == 570
    assert encoder_motion_dim(29) == 650
    slots, total = compile_encoder_observations(cfg)
    assert total == 570
    assert [s.name for s in slots] == [
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_anchor_orientation_10frame_step5",
        "motion_root_z_position_10frame_step5",
    ]
    assert [s.dim for s in slots] == [250, 250, 60, 10]
    decoder_slots, decoder_dim = compile_observations(cfg)
    assert decoder_dim == 874
    assert decoder_slots[0].name == "token_state"


def test_parse_motion_history_name() -> None:
    assert parse_history_name("motion_joint_positions_10frame_step5") == (
        "motion_joint_positions",
        10,
        5,
    )
    assert parse_history_name("motion_anchor_orientation_heading_10frame_step5") == (
        "motion_anchor_orientation_heading",
        10,
        5,
    )


def test_empty_hold_does_not_invent_a_clip() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    with pytest.raises(MotionRefError, match="invent"):
        gather.assemble_encoder()


def test_look_ahead_step5_and_last_frame_repeat() -> None:
    assert look_ahead_indices(0, 10, 5, 50) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    assert look_ahead_indices(0, 10, 5, 3) == [0, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    hold = MotionHold()
    hold.push_sequence(_window(3), cursor=0)
    window = hold.look_ahead(10, 5)
    assert [float(f.q_ref_rad[0]) for f in window] == [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]


def test_assemble_encoder_grouped_layout() -> None:
    gather = ObsGather()
    gather.push_hw(_hw())
    gather.push_motion(_window(50), cursor=0)
    enc = gather.assemble_encoder()
    assert enc.shape == (570,)
    q = enc[:250].reshape(10, 25)
    dq = enc[250:500].reshape(10, 25)
    ori = enc[500:560].reshape(10, 6)
    z = enc[560:570]
    np.testing.assert_allclose(q[:, 0], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    np.testing.assert_allclose(dq[:, 0], 0.1 * q[:, 0])
    np.testing.assert_allclose(ori, np.tile(identity_rot6d(), (10, 1)))
    np.testing.assert_allclose(z, 1.03 + 0.001 * q[:, 0])
    token = np.zeros(TOKEN_DIM)
    dec = gather.control_tick(token, t_s=0.0)
    assert dec.shape == (874,)


def test_anchor_heading_correction_is_relative_rot6d() -> None:
    robot = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # +90° yaw
    ref = _identity_quat()
    six = heading_corrected_rel_rot6d(robot, ref, mode="full")
    # R_z(90).T @ I = R_z(-90) → columns [0,-1,0] and [1,0,0]
    np.testing.assert_allclose(six, [0.0, -1.0, 0.0, 1.0, 0.0, 0.0], atol=1e-9)
    gather = ObsGather()
    gather.push_hw(_hw(yaw_rad=np.pi / 2))
    frames = [_frame(q0=0.0, yaw_rad=0.0) for _ in range(50)]
    gather.push_motion(frames, cursor=0)
    enc = gather.assemble_encoder()
    ori = enc[500:560].reshape(10, 6)
    np.testing.assert_allclose(ori[0], [0.0, -1.0, 0.0, 1.0, 0.0, 0.0], atol=1e-9)


def test_g1_motion_frame_refused() -> None:
    hold = MotionHold()
    bad = MotionFrame(
        q_ref_rad=np.zeros(29),
        dq_ref_rad_s=np.zeros(29),
        root_pos_m=np.zeros(3),
        root_rot_wxyz=_identity_quat(),
    )
    with pytest.raises(G1CheckpointIncompatible):
        hold.push_sequence([bad])


def test_hand_concat_motion_refused() -> None:
    hold = MotionHold()
    bad = MotionFrame(
        q_ref_rad=np.zeros(45),
        dq_ref_rad_s=np.zeros(45),
        root_pos_m=np.zeros(3),
        root_rot_wxyz=_identity_quat(),
    )
    with pytest.raises(MotionRefError, match="DexHand2"):
        hold.push_sequence([bad])


def test_wrist_and_smpl_names_refused() -> None:
    cfg = load_gather_cfg()
    cfg["encoder"] = dict(cfg["encoder"])
    cfg["encoder"]["encoder_observations"] = [
        {"name": "motion_joint_positions_wrists_10frame_step1", "enabled": True}
    ]
    with pytest.raises(ObsGatherError, match="wrist"):
        compile_encoder_observations(cfg)
    cfg["encoder"]["encoder_observations"] = [
        {"name": "smpl_joints_10frame_step5", "enabled": True}
    ]
    with pytest.raises(ObsGatherError, match="SMPL"):
        compile_encoder_observations(cfg)


def test_g1_mode_name_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(GATHER_YAML.read_text(encoding="utf-8"))
    raw["encoder"]["encoder_modes"] = [{"name": "g1", "mode_id": 0, "required_observations": []}]
    path = tmp_path / "g1mode.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ObsGatherError, match="G1 mode"):
        load_gather_cfg(path)


def test_fps_30_clip_is_not_resampled() -> None:
    lib = {
        "fps": 30.0,
        "root_pos": np.zeros((10, 3)),
        "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (10, 1)),
        "dof_pos": np.zeros((10, 25)),
    }
    with pytest.raises(MotionRefError, match="resample"):
        MotionCursor(lib)


def test_g1_encoder_onnx_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible, match="570"):
        refuse_g1_encoder_onnx("model_encoder.onnx")


def test_lowerbody_is_first_twelve_joints() -> None:
    raw = yaml.safe_load(GATHER_YAML.read_text(encoding="utf-8"))
    raw["encoder"]["encoder_observations"] = [
        {"name": "motion_joint_positions_lowerbody_10frame_step5", "enabled": True}
    ]
    raw["expected_encoder_dim"] = 120
    cfg = dict(raw)
    slots, dim = compile_encoder_observations(cfg)
    assert dim == 120
    assert slots[0].dim == 120
    gather = ObsGather()
    gather.encoder_slots = slots
    gather.encoder_dim = dim
    frames = []
    for i in range(50):
        q = np.arange(25, dtype=np.float64) + i
        frames.append(
            MotionFrame(
                q_ref_rad=q,
                dq_ref_rad_s=np.zeros(25),
                root_pos_m=np.array([0.0, 0.0, 1.03]),
                root_rot_wxyz=_identity_quat(),
            )
        )
    gather.push_motion(frames, cursor=0)
    enc = gather.assemble_encoder()
    block = enc.reshape(10, 12)
    np.testing.assert_allclose(block[0], np.arange(12.0))
    np.testing.assert_allclose(block[1], np.arange(12.0) + 5.0)
