"""URDF FK, T800 dummy-wrist, and heading-frame 3-point packing."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from interface.schema import REPO_ROOT, CommandVector, load_frames
from sim.urdf_fk import UrdfTree, parse_urdf_joints
from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import G1_DECODER_INPUT_DIM, assert_t800_config, decoder_history_dim
from wbc.observation import heading_gravity, policy_proprio
from wbc.retarget import assert_retarget_ready, retarget_blockers, status_report
from wbc.teleop import command_to_vr_3point, vr_3point_to_wbc_fields

TOY_URDF = """
<robot name="toy">
  <link name="base"/>
  <link name="elbow"/>
  <link name="wrist"/>
  <joint name="j_yaw" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="elbow"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="j_fixed" type="fixed">
    <origin xyz="0.029436 0.0124855 -0.13222151" rpy="0 0 0"/>
    <parent link="elbow"/>
    <child link="wrist"/>
  </joint>
</robot>
"""

T800_URDF = (
    REPO_ROOT
    / "third_party"
    / "engineai-native-sdk"
    / "assets"
    / "resource"
    / "robot"
    / "t800"
    / "urdf"
    / "serial_t800.urdf"
)


def test_toy_fk_fixed_wrist_offset() -> None:
    tree = UrdfTree.from_urdf_text(TOY_URDF, root_link="elbow")
    pos, rot = tree.fk_link("wrist", {})
    np.testing.assert_allclose(pos, [0.029436, 0.0124855, -0.13222151], atol=1e-9)
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-9)


def test_toy_fk_revolute_yaw() -> None:
    tree = UrdfTree.from_urdf_text(TOY_URDF, root_link="base")
    pos0, _ = tree.fk_link("elbow", {"j_yaw": 0.0})
    np.testing.assert_allclose(pos0, [0.0, 0.0, 0.1], atol=1e-9)
    pos90, _ = tree.fk_link("wrist", {"j_yaw": np.pi / 2})
    # Fixed offset in elbow frame, elbow yawed +90° about Z.
    expected = np.array([-0.0124855, 0.029436, 0.1 - 0.13222151])
    np.testing.assert_allclose(pos90, expected, atol=1e-9)


@pytest.mark.skipif(not T800_URDF.is_file(), reason="T800 URDF not cloned")
def test_t800_dummy_wrist_matches_weld_recipe() -> None:
    from assets.combined.assemble import dummy_wrist_from_elbow

    recipe = dummy_wrist_from_elbow()
    text = T800_URDF.read_text(encoding="utf-8")
    joints = {j.name: j for j in parse_urdf_joints(text)}
    for side in ("left", "right"):
        rec = recipe[side]
        joint = joints[rec["joint"]]
        np.testing.assert_allclose(joint.origin_xyz_m, rec["pos_m"], atol=1e-9)
        local = UrdfTree.from_urdf_text(text, root_link=rec["parent"])
        pos, _ = local.fk_link(rec["child"], {})
        np.testing.assert_allclose(pos, rec["pos_m"], atol=1e-9)


@pytest.mark.skipif(not T800_URDF.is_file(), reason="T800 URDF not cloned")
def test_t800_adapter_zero_pose_finite() -> None:
    from vla.adapters.joint_to_wrist_adapter import JointToWristAdapter

    frames = load_frames()["frames"]
    adapter = JointToWristAdapter(side="right", urdf_path=T800_URDF)
    pose = adapter(np.zeros(5))
    assert pose.link == frames["right_wrist"]["t800_link"]
    assert np.isfinite(pose.pos_m).all()
    assert pose.rot6d.shape == (6,)


@pytest.mark.skipif(not T800_URDF.is_file(), reason="T800 URDF not cloned")
def test_t800_elbow_fk_is_pitch_body() -> None:
    from vla.adapters.joint_to_wrist_adapter import JointToWristAdapter

    frames = load_frames()["frames"]
    adapter = JointToWristAdapter(side="right", urdf_path=T800_URDF)
    elbow = adapter.elbow_pose(np.zeros(5))
    wrist = adapter(np.zeros(5))
    assert elbow.link == frames["right_elbow"]["t800_link"]
    assert elbow.link == "LINK_ELBOW_PITCH_R"
    assert np.isfinite(elbow.pos_m).all()
    # Dummy wrist sits on the forearm, not at the elbow joint.
    assert float(np.linalg.norm(wrist.pos_m - elbow.pos_m)) > 0.05


def test_vr_3point_is_left_right_head_not_schema_order() -> None:
    cmd = CommandVector.zeros()
    cmd.head_pos[:] = [1.0, 0.0, 0.0]
    cmd.left_wrist_pos[:] = [0.0, 1.0, 0.0]
    cmd.right_wrist_pos[:] = [0.0, 0.0, 1.0]
    ident = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    cmd.head_rot6d[:] = ident
    cmd.left_wrist_rot6d[:] = ident
    cmd.right_wrist_rot6d[:] = ident
    pos, quat = command_to_vr_3point(cmd)
    np.testing.assert_allclose(pos[0:3], [0.0, 1.0, 0.0])  # left first
    np.testing.assert_allclose(pos[3:6], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(pos[6:9], [1.0, 0.0, 0.0])  # head last
    back = vr_3point_to_wbc_fields(pos, quat)
    np.testing.assert_allclose(back["head_pos"], cmd.head_pos)
    np.testing.assert_allclose(back["left_wrist_pos"], cmd.left_wrist_pos)
    np.testing.assert_allclose(back["head_rot6d"], ident, atol=1e-9)


def test_decoder_dims_g1_vs_t800() -> None:
    cfg = assert_t800_config()
    assert cfg["n_revolute"] == 25
    assert decoder_history_dim(25) == 874
    assert decoder_history_dim(29) == G1_DECODER_INPUT_DIM
    assert cfg["decoder_input_dim"] != G1_DECODER_INPUT_DIM


def test_refuse_g1_checkpoint() -> None:
    with pytest.raises(G1CheckpointIncompatible, match="G1"):
        refuse_g1_checkpoint(checkpoint_meta={"robot": "g1_29dof", "n_dof": 29})
    with pytest.raises(G1CheckpointIncompatible, match="994"):
        refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
    refuse_g1_checkpoint(checkpoint_meta={"robot": "t800", "n_dof": 25})


def test_heading_gravity_yaw_90() -> None:
    g = heading_gravity(np.pi / 2)
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-9)
    proprio = policy_proprio(
        np.zeros(25), np.zeros(25), np.array([0.1, 0.0, 0.0]), np.zeros(25), 0.0, n_dof=25
    )
    assert proprio.shape == (25 + 25 + 3 + 3 + 25,)


def test_retarget_blocked_without_flange() -> None:
    blockers = retarget_blockers()
    assert any("t800_wrist_to_hand_mount" in b for b in blockers)
    with pytest.raises(PolicyEvalBlocked, match="flange|REQUIRED_INPUT|blocked"):
        assert_retarget_ready()
    report = status_report()
    assert report["g1_finetune"] == "forbidden"
    assert report["n_revolute"] == 25


def test_pd_stand_length_is_25() -> None:
    cfg = assert_t800_config()
    kp = [x for group in cfg["pd_stand"]["stiffness"] for x in group]
    kd = [x for group in cfg["pd_stand"]["damping"] for x in group]
    q = [x for group in cfg["pd_stand"]["desired_joint_position"] for x in group]
    assert len(kp) == 25
    assert len(kd) == 25
    assert len(q) == 25
    assert cfg["foot_frame"]["decision"] == "mjcf_link_foot_at_ankle_roll"
