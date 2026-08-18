"""T800 SONIC PPO recipe, clip filter, kinematic Sim2Sim, ONNX contract."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from eval.l2_sim2sim import run as sim2sim_run
from interface.schema import CommandVector, load_hand_spec
from sim.urdf_fk import (
    KINEMATICS_YAML,
    UrdfTree,
    fk_bodies_pelvis,
    load_t800_kinematics,
    mjcf_joint_limits_from_kinematics,
)
from vla.adapters.case_a import CaseAToCommandSchema, HeadNavCommand
from vla.adapters.rotation import matrix_to_rot6d
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.export_onnx import OnnxExportBlocked, expected_io, refuse_export
from wbc.filter import parse_mjcf_joint_limits
from wbc.gmr.filter_lib import filter_motion_library, official_mjcf_limits_rad
from wbc.gmr.synthetic_clip import synthetic_stand_clip
from wbc.ppo.recipe import PpoLaunchBlocked, load_ppo_recipe, refuse_g1_action_dim, refuse_ppo_launch
from wbc.ppo.rewards import (
    action_rate_penalty,
    joint_limit_penalty,
    tracking_reward_terms,
    weighted_tracking_return,
)
from wbc.teleop import FivePointCommand, command_to_vr_3point, command_to_vr_5point, pack_hybrid_encoder_cmd


def test_mjcf_range_is_radians_not_actuatorfrcrange() -> None:
    limits = mjcf_joint_limits_from_kinematics()
    assert limits.shape == (25, 2)
    hip = limits[0]
    np.testing.assert_allclose(hip, [-3.316, 2.269], atol=1e-6)
    assert float(np.max(np.abs(limits))) < 20.0


def test_parse_mjcf_does_not_latch_actuatorfrcrange() -> None:
    xml = (
        '<joint name="J00_HIP_PITCH_L" pos="0 0 0" axis="0 1 0" '
        'range="-3.316 2.269" actuatorfrcrange="-415 415"/>'
    )
    lim = parse_mjcf_joint_limits(xml, ["J00_HIP_PITCH_L"])
    np.testing.assert_allclose(lim[0], [-3.316, 2.269], atol=1e-9)


def test_ppo_recipe_action_dim_is_25() -> None:
    recipe = load_ppo_recipe()
    assert recipe["network"]["action_dim"] == 25
    assert recipe["network"]["paper_g1_action_dim"] == 29
    assert recipe["hyperparams"]["g1_last_pt_finetune"] == "forbidden"
    assert recipe["domain_rand"]["not_dexhand2_contact"] is True
    assert recipe["rewards"]["tracking"]["body_pos_rel"]["scale_m"] == 0.3
    assert recipe["rewards"]["tracking"]["ee_pos"]["weight"] == 2.0
    friction = recipe["domain_rand"]["physical"]["static_friction"]
    spec = load_hand_spec()
    assert spec.raw.get("friction_vs_cardboard_static") != friction


def test_refuse_g1_action_dim() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        refuse_g1_action_dim(29)
    refuse_g1_action_dim(25)


def test_refuse_ppo_launch_while_p0_open() -> None:
    with pytest.raises(PpoLaunchBlocked, match="blocked"):
        refuse_ppo_launch()
    with pytest.raises(PpoLaunchBlocked):
        refuse_ppo_launch(teleop_mode="vr_3point")


def test_five_point_overlay_keeps_action_dim_and_refuses_launch() -> None:
    from wbc.teleop import TELEOP_3POINT, TELEOP_5POINT, TeleopModeIncompatible, refuse_teleop_mode_mismatch

    three = load_ppo_recipe()
    assert three["teleop_mode"] == TELEOP_3POINT
    assert three["five_point_overlay"] is False
    five = load_ppo_recipe(teleop_mode=TELEOP_5POINT)
    assert five["teleop_mode"] == TELEOP_5POINT
    assert five["network"]["action_dim"] == 25
    assert five["network"]["hybrid_encoder_cmd_dim_5point"] == 27
    assert five["five_point_overlay"] is True
    with pytest.raises(PpoLaunchBlocked, match="blocked"):
        refuse_ppo_launch(teleop_mode=TELEOP_5POINT)
    with pytest.raises(TeleopModeIncompatible):
        refuse_teleop_mode_mismatch(three["teleop_mode"], TELEOP_5POINT)


def test_identity_tracking_rewards_are_one() -> None:
    z3 = np.zeros(3)
    b = np.zeros((6, 3))
    ee = np.zeros((5, 3))
    terms = tracking_reward_terms(
        root_pos_p=z3,
        root_pos_g=z3,
        root_ori_p=z3,
        root_ori_g=z3,
        body_pos_rel_p=b,
        body_pos_rel_g=b,
        body_ori_rel_p=b,
        body_ori_rel_g=b,
        body_lin_p=b,
        body_lin_g=b,
        body_ang_p=b,
        body_ang_g=b,
        ee_pos_p=ee,
        ee_pos_g=ee,
    )
    assert all(v == pytest.approx(1.0) for v in terms.values())
    total = weighted_tracking_return(terms)
    assert total == pytest.approx(0.5 + 0.5 + 1 + 1 + 1 + 1 + 2)


def test_body_pos_kernel_matches_paper_scale() -> None:
    # one body, 0.3 m error, scale 0.3 → exp(-1) 
    z3 = np.zeros(3)
    b_p = np.array([[0.3, 0.0, 0.0]])
    b_g = np.zeros((1, 3))
    ee = np.zeros((5, 3))
    terms = tracking_reward_terms(
        root_pos_p=z3,
        root_pos_g=z3,
        root_ori_p=z3,
        root_ori_g=z3,
        body_pos_rel_p=b_p,
        body_pos_rel_g=b_g,
        body_ori_rel_p=np.zeros((1, 3)),
        body_ori_rel_g=np.zeros((1, 3)),
        body_lin_p=np.zeros((1, 3)),
        body_lin_g=np.zeros((1, 3)),
        body_ang_p=np.zeros((1, 3)),
        body_ang_g=np.zeros((1, 3)),
        ee_pos_p=ee,
        ee_pos_g=ee,
    )
    assert terms["body_pos_rel"] == pytest.approx(float(np.exp(-1.0)), rel=1e-12)


def test_penalties() -> None:
    a0 = np.zeros(25)
    a1 = np.ones(25)
    p = action_rate_penalty(a1, a0)
    assert p == pytest.approx(-0.1 * 25.0)
    limits = official_mjcf_limits_rad()
    q = np.zeros(25)
    assert joint_limit_penalty(q, limits) == 0.0
    q[0] = limits[0, 1] + 0.5
    assert joint_limit_penalty(q, limits) == pytest.approx(-10.0)


def test_filter_library_keeps_stand_drops_limit() -> None:
    stand = synthetic_stand_clip(n_frames=20, fps=50.0, amplitude_rad=0.02)
    ok = filter_motion_library(stand)
    assert ok.n_keep == 1
    bad = dict(stand)
    q = np.asarray(bad["dof_pos"]).copy()
    q[5, 0] = 10.0  # far past hip pitch rad limit
    bad["dof_pos"] = q
    dropped = filter_motion_library({"clips": [stand, bad]})
    assert dropped.n_keep == 1
    assert dropped.n_drop == 1


def test_filter_refuses_g1_clip() -> None:
    lib = synthetic_stand_clip(n_frames=10, n_dof=29)
    with pytest.raises(G1CheckpointIncompatible):
        filter_motion_library(lib)


def test_sim2sim_identity_zero_error() -> None:
    report = sim2sim_run(perturb_wrist_joint_rad=0.0)
    assert report["grasp_success_rate"] is None
    assert report["mpjpe_m"] == pytest.approx(0.0, abs=1e-12)
    assert report["wrist_tracking_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert report["local_tracking_success"] is True
    noisy = sim2sim_run(perturb_wrist_joint_rad=0.2)
    assert noisy["wrist_tracking_error_m"] > 1e-4
    assert noisy["grasp_success_rate"] is None


def test_case_a_fk_from_kinematics_yaml() -> None:
    assert KINEMATICS_YAML.is_file()
    adapter = CaseAToCommandSchema.from_t800_urdf()
    vec = np.zeros(50)
    vec[0] = 0.1
    cmd = adapter.convert(
        vec,
        apply_fk=True,
        head_nav=HeadNavCommand(
            pelvis_height_m=0.55,
            nav_cmd_mps=np.zeros(3),
            loco_mode=0,
            tool_trigger=0,
            source="unit_test_kinematics_yaml",
            head_pos_m=np.array([0.0, 0.0, 0.4]),
            head_rot6d=matrix_to_rot6d(np.eye(3)),
        ),
    )
    assert np.isfinite(cmd.left_wrist_pos).all()
    # Non-zero shoulder pitch must move the dummy wrist off the q=0 pose.
    zero = adapter.convert(
        np.zeros(50),
        apply_fk=True,
        head_nav=HeadNavCommand(
            pelvis_height_m=0.55,
            nav_cmd_mps=np.zeros(3),
            loco_mode=0,
            tool_trigger=0,
            source="unit_test_kinematics_yaml",
            head_pos_m=np.array([0.0, 0.0, 0.4]),
            head_rot6d=matrix_to_rot6d(np.eye(3)),
        ),
    )
    assert not np.allclose(cmd.left_wrist_pos, zero.left_wrist_pos)


def test_kinematics_fk_dummy_wrist() -> None:
    kin = load_t800_kinematics()
    tree = UrdfTree.from_kinematics_yaml(KINEMATICS_YAML, root_link=kin["root_link"])
    poses = fk_bodies_pelvis(
        tree,
        np.zeros(25),
        joint_order=kin["joint_order"],
        bodies=kin["tracked_bodies"],
    )
    assert poses["pelvis"][0].shape == (3,)
    np.testing.assert_allclose(poses["pelvis"][0], 0.0, atol=1e-12)
    # Dummy wrist is a fixed offset from elbow yaw; at q=0 it is not the origin.
    assert np.linalg.norm(poses["left_wrist"][0]) > 0.05


def test_onnx_export_refused() -> None:
    io = expected_io()
    assert io["decoder"]["model_decoder.onnx"]["input_dim"] == 874
    assert io["decoder"]["model_decoder.onnx"]["g1_input_dim_forbidden"] == 994
    with pytest.raises((OnnxExportBlocked, PolicyEvalBlocked, G1CheckpointIncompatible)):
        refuse_export()
    with pytest.raises(G1CheckpointIncompatible):
        refuse_export(checkpoint_meta={"robot": "g1", "n_dof": 29})


def test_hybrid_encoder_pack() -> None:
    cmd = CommandVector.zeros()
    pos, quat = command_to_vr_3point(cmd)
    packed = pack_hybrid_encoder_cmd(pos, quat, mode="vr_3point")
    assert packed.shape == (21,)
    fp = FivePointCommand(cmd, left_elbow_pos=np.zeros(3), right_elbow_pos=np.zeros(3))
    pos5, quat5 = command_to_vr_5point(fp)
    assert pack_hybrid_encoder_cmd(pos5, quat5, mode="vr_5point").shape == (27,)
    with pytest.raises(ValueError):
        pack_hybrid_encoder_cmd(pos, quat, mode="vr_5point")
