"""Physics Sim2Sim on T800 MJCF. Grasp-success stays null. No combined robot."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from assets.combined.assemble import PolicyEvalBlocked
from interface.schema import REPO_ROOT
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key
from sim.mujoco_env.t800_env import OFFICIAL_MJCF, T800MujocoEnv, refuse_combined_robot
from wbc.dims import decoder_history_dim
from wbc.pd_stand import pd_stand_kp_kd, pd_stand_q_des_rad


def test_pd_stand_is_25() -> None:
    kp, kd = pd_stand_kp_kd()
    assert kp.shape == (25,)
    assert kd.shape == (25,)
    assert kp[0] == pytest.approx(1080.0)
    assert kd[-1] == pytest.approx(1.0)
    q = pd_stand_q_des_rad()
    assert q.shape == (25,)
    assert q[0] == pytest.approx(-0.105)


def test_foot_frame_offset_is_documented() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "assets/engineai/meta/t800_frame_offsets.yaml").read_text(encoding="utf-8")
    )
    assert raw["foot"]["delta_z_m"] == pytest.approx(0.06453)
    assert raw["foot"]["decision"] == "mjcf_link_foot_at_ankle_roll"
    assert raw["wrist"]["match"] is True


def test_combined_hands_refused() -> None:
    with pytest.raises(PolicyEvalBlocked, match="weld"):
        refuse_combined_robot()
    with pytest.raises(PolicyEvalBlocked):
        T800MujocoEnv(include_hands=True)


def test_fixture_pinned_compiles_and_tracks() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_physics_sim2sim import run

    env = T800MujocoEnv(source="fixture", pinned_base=True)
    assert env.model.nq == 25
    assert env.model.nu == 25
    env.reset()
    stats = env.hold(0.2)
    assert np.isfinite(stats["qvel_rms_rad_s"])
    report = run(source="fixture", pinned_base=True, n_frames=12, amplitude_rad=0.03)
    assert report["grasp_success_rate"] is None
    refuse_grasp_success_key(report)
    assert report["combined_robot"] == "PolicyEvalBlocked"
    assert report["decoder_input_dim"] == decoder_history_dim(25)
    assert report["joint_mae_rad"] < 0.15
    assert report["fk_consistency_no_feet_m"] < 0.02
    assert report["foot_urdf_mjcf_delta_z_m"] < 0.005
    assert report["source"] == "fixture"
    assert report["joint_jitter"]["name"] == "q_jit_+0.1"
    assert report["joint_jitter"]["kind"] == "target_motion_joint_jitter"
    assert report["joint_jitter"]["offset_rad"] == 0.1
    assert report["joint_jitter"]["not_a_sonic_gate"] is True
    assert report["joint_jitter"]["not_default_joint_pos_offset"] is True
    assert report["joint_jitter"]["grasp_success_rate"] is None
    names = [c["name"] for c in report["joint_jitter_sweep"]]
    assert names == ["q_jit_-0.1", "q_jit_+0.1"]
    assert report["joint_jitter_sweep_summary"]["n_cases"] == 2
    assert report["joint_jitter_sweep_summary"]["not_a_sonic_gate"] is True
    assert all(c["kind"] == "target_motion_joint_jitter" for c in report["joint_jitter_sweep"])
    assert all(c["grasp_success_rate"] is None for c in report["joint_jitter_sweep"])
    by_off = {c["name"]: c["offset_rad"] for c in report["joint_jitter_sweep"]}
    assert by_off["q_jit_-0.1"] == -0.1
    assert by_off["q_jit_+0.1"] == 0.1
    assert report["pos_jitter"]["name"] == "pos_+x"
    assert report["pos_jitter"]["kind"] == "target_motion_pos_jitter"
    assert report["pos_jitter"]["offset_m"] == [0.05, 0.0, 0.0]
    assert report["pos_jitter"]["not_a_height_ori_gate"] is True
    assert report["pos_jitter"]["not_a_sonic_gate"] is True
    assert report["pos_jitter"]["grasp_success_rate"] is None
    assert [c["name"] for c in report["pos_jitter_sweep"]] == [
        "pos_-x",
        "pos_+x",
        "pos_-y",
        "pos_+y",
        "pos_-z",
        "pos_+z",
    ]
    assert report["pos_jitter_sweep_summary"]["n_cases"] == 6
    assert report["pos_jitter_sweep_summary"]["not_a_height_ori_gate"] is True
    assert report["ori_jitter"]["name"] == "ori_+yaw"
    assert report["ori_jitter"]["kind"] == "target_motion_ori_jitter"
    assert report["ori_jitter"]["offset_rad"] == [0.0, 0.0, 0.2]
    assert report["ori_jitter"]["not_a_height_ori_gate"] is True
    assert report["ori_jitter"]["grasp_success_rate"] is None
    assert [c["name"] for c in report["ori_jitter_sweep"]] == [
        "ori_-roll",
        "ori_+roll",
        "ori_-pitch",
        "ori_+pitch",
        "ori_-yaw",
        "ori_+yaw",
    ]
    assert report["ori_jitter_sweep_summary"]["n_cases"] == 6
    # Negative control: joints stay, root metrics move. Not a 0.25 m / 1.0 rad gate.
    assert abs(report["pos_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["pos_jitter"]["root_pos_err_m"] == pytest.approx(0.05, abs=0.02)
    assert abs(report["ori_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["ori_jitter"]["pelvis_ori_err_rad"] == pytest.approx(0.2, abs=0.05)
    assert report["lin_vel_jitter"]["name"] == "lin_vel_+x"
    assert report["lin_vel_jitter"]["kind"] == "target_motion_lin_vel_jitter"
    assert report["lin_vel_jitter"]["offset_mps"] == [0.5, 0.0, 0.0]
    assert report["lin_vel_jitter"]["not_a_sonic_gate"] is True
    assert report["lin_vel_jitter"]["not_root_push"] is True
    assert report["lin_vel_jitter"]["grasp_success_rate"] is None
    assert [c["name"] for c in report["lin_vel_jitter_sweep"]] == [
        "lin_vel_-x",
        "lin_vel_+x",
        "lin_vel_-y",
        "lin_vel_+y",
        "lin_vel_-z",
        "lin_vel_+z",
    ]
    assert report["lin_vel_jitter_sweep_summary"]["n_cases"] == 6
    assert report["ang_vel_jitter"]["name"] == "ang_vel_+yaw"
    assert report["ang_vel_jitter"]["kind"] == "target_motion_ang_vel_jitter"
    assert report["ang_vel_jitter"]["offset_rad_s"] == [0.0, 0.0, 0.78]
    assert report["ang_vel_jitter"]["not_root_push"] is True
    assert [c["name"] for c in report["ang_vel_jitter_sweep"]] == [
        "ang_vel_-roll",
        "ang_vel_+roll",
        "ang_vel_-pitch",
        "ang_vel_+pitch",
        "ang_vel_-yaw",
        "ang_vel_+yaw",
    ]
    assert report["ang_vel_jitter_sweep_summary"]["n_cases"] == 6
    assert abs(report["lin_vel_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["lin_vel_jitter"]["target_linvel_mean_mps"][0] == pytest.approx(0.85, abs=1e-6)
    assert abs(report["ang_vel_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["ang_vel_jitter"]["target_angvel_mean_rad_s"][2] == pytest.approx(1.03, abs=1e-6)


@pytest.mark.skipif(not OFFICIAL_MJCF.is_file(), reason="engineai Native SDK not cloned")
def test_official_pinned_pd_and_fk() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_physics_sim2sim import run

    env = T800MujocoEnv(source="official", pinned_base=True)
    assert env.model.nq == 25
    assert env.model.nu == 25
    env.reset()
    report = run(source="official", pinned_base=True, n_frames=16, amplitude_rad=0.03)
    assert report["grasp_success_rate"] is None
    refuse_grasp_success_key(report)
    assert report["source"] == "official"
    assert report["gains_source"] == "pd_stand_bringup_not_sonic"
    assert report["decoder_input_dim"] == 874
    assert report["fk_consistency_no_feet_m"] < 0.005
    assert report["foot_urdf_mjcf_delta_z_m"] == pytest.approx(0.06453, abs=1e-3)
    assert report["joint_mae_rad"] < 0.15
    # G1 6 cm figure is recorded, not used as a gate.
    assert report["paper_g1_wrist_err_m_not_a_gate"] == 0.06
    assert report["joint_jitter_sweep_summary"]["n_cases"] == 2
    assert report["joint_jitter_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["joint_jitter"]["kind"] == "target_motion_joint_jitter"
    assert report["joint_jitter"]["not_default_joint_pos_offset"] is True
    assert report["joint_jitter"]["grasp_success_rate"] is None
    # Jittered tracking is a diagnostic, not a SONIC fail.
    assert report["joint_jitter"]["not_a_sonic_gate"] is True
    assert report["pos_jitter_sweep_summary"]["n_cases"] == 6
    assert report["pos_jitter"]["not_a_height_ori_gate"] is True
    assert report["pos_jitter"]["grasp_success_rate"] is None
    assert report["ori_jitter_sweep_summary"]["n_cases"] == 6
    assert report["ori_jitter"]["not_a_height_ori_gate"] is True
    assert report["ori_jitter"]["not_a_sonic_gate"] is True
    assert abs(report["pos_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["pos_jitter"]["root_pos_err_m"] == pytest.approx(0.05, abs=0.02)
    assert abs(report["ori_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["ori_jitter"]["pelvis_ori_err_rad"] == pytest.approx(0.2, abs=0.05)
    assert report["lin_vel_jitter_sweep_summary"]["n_cases"] == 6
    assert report["lin_vel_jitter"]["not_root_push"] is True
    assert report["lin_vel_jitter"]["grasp_success_rate"] is None
    assert report["ang_vel_jitter_sweep_summary"]["n_cases"] == 6
    assert report["ang_vel_jitter"]["not_a_sonic_gate"] is True
    assert abs(report["lin_vel_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["lin_vel_jitter"]["target_linvel_mean_mps"][0] == pytest.approx(0.85, abs=1e-6)
    assert abs(report["ang_vel_jitter"]["joint_mae_rad"] - report["joint_mae_rad"]) < 0.02
    assert report["ang_vel_jitter"]["target_angvel_mean_rad_s"][2] == pytest.approx(1.03, abs=1e-6)


def test_clip_joint_jitter_is_uniform_and_refuses_free_base() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_physics_sim2sim import apply_clip_joint_jitter, evaluate_joint_jitter
    from wbc.gmr.synthetic_clip import synthetic_stand_clip

    ref = synthetic_stand_clip(n_frames=8, fps=50.0, amplitude_rad=0.03)
    jittered = apply_clip_joint_jitter(ref, 0.1)
    delta = np.asarray(jittered["dof_pos"]) - np.asarray(ref["dof_pos"])
    assert delta == pytest.approx(np.full_like(delta, 0.1))
    assert jittered["root_pos"] is ref["root_pos"]
    env = T800MujocoEnv(source="fixture", pinned_base=True)
    env.assert_hinges_in_mjcf_range(np.zeros(25), what="zero")
    with pytest.raises(ValueError, match="outside"):
        env.assert_hinges_in_mjcf_range(np.full(25, 2.0), what="huge")
    with pytest.raises(ValueError, match="finite"):
        env.assert_hinges_in_mjcf_range(np.array([np.nan] * 25), what="nan")
    free = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=True)
    with pytest.raises(ValueError, match="pinned_base"):
        evaluate_joint_jitter(
            free,
            ref,
            offset_rad=0.1,
            case_name="q_jit_+0.1",
            height_fail_m=0.25,
            ori_fail_rad=1.0,
        )


def test_clip_pos_ori_jitter_is_root_only_and_refuses_free_base() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_physics_sim2sim import (
        apply_clip_ori_jitter,
        apply_clip_pos_jitter,
        evaluate_ori_jitter,
        evaluate_pos_jitter,
    )
    from wbc.gmr.synthetic_clip import synthetic_stand_clip

    ref = synthetic_stand_clip(n_frames=8, fps=50.0, amplitude_rad=0.03)
    pos = apply_clip_pos_jitter(ref, [0.05, 0.0, 0.0])
    delta = np.asarray(pos["root_pos"]) - np.asarray(ref["root_pos"])
    assert delta == pytest.approx(np.tile([0.05, 0.0, 0.0], (8, 1)))
    assert np.asarray(pos["dof_pos"]) == pytest.approx(np.asarray(ref["dof_pos"]))
    assert pos["root_rot"] is ref["root_rot"]
    ori = apply_clip_ori_jitter(ref, [0.0, 0.0, 0.2])
    assert np.asarray(ori["dof_pos"]) == pytest.approx(np.asarray(ref["dof_pos"]))
    assert np.asarray(ori["root_pos"]) == pytest.approx(np.asarray(ref["root_pos"]))
    assert ori["root_rot"] is not ref["root_rot"]
    free = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=True)
    with pytest.raises(ValueError, match="pinned_base"):
        evaluate_pos_jitter(
            free,
            ref,
            offset_m=[0.05, 0.0, 0.0],
            case_name="pos_+x",
            height_fail_m=0.25,
            ori_fail_rad=1.0,
        )
    with pytest.raises(ValueError, match="pinned_base"):
        evaluate_ori_jitter(
            free,
            ref,
            offset_rad=[0.0, 0.0, 0.2],
            case_name="ori_+yaw",
            height_fail_m=0.25,
            ori_fail_rad=1.0,
        )


def test_clip_vel_jitter_refuses_stand_clip_and_free_base() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_physics_sim2sim import (
        apply_clip_ang_vel_jitter,
        apply_clip_lin_vel_jitter,
        evaluate_ang_vel_jitter,
        evaluate_lin_vel_jitter,
    )
    from wbc.gmr.synthetic_clip import (
        SYNTHETIC_WALK_ANGVEL_RAD_S,
        SYNTHETIC_WALK_LINVEL_MPS,
        synthetic_stand_clip,
        synthetic_walk_clip,
    )

    stand = synthetic_stand_clip(n_frames=8, fps=50.0, amplitude_rad=0.03)
    with pytest.raises(ValueError, match="degenerate"):
        apply_clip_lin_vel_jitter(stand, [0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        apply_clip_ang_vel_jitter(stand, [0.0, 0.0, 0.78])
    walk = synthetic_walk_clip(n_frames=8, fps=50.0, amplitude_rad=0.03)
    lin = apply_clip_lin_vel_jitter(walk, [0.5, 0.0, 0.0])
    delta = np.asarray(lin["root_linvel"]) - np.asarray(walk["root_linvel"])
    assert delta == pytest.approx(np.tile([0.5, 0.0, 0.0], (8, 1)))
    assert np.asarray(lin["dof_pos"]) == pytest.approx(np.asarray(walk["dof_pos"]))
    assert np.asarray(lin["root_pos"]) == pytest.approx(np.asarray(walk["root_pos"]))
    assert lin["root_linvel"][0].tolist()[0] == pytest.approx(
        SYNTHETIC_WALK_LINVEL_MPS[0] + 0.5
    )
    ang = apply_clip_ang_vel_jitter(walk, [0.0, 0.0, 0.78])
    assert np.asarray(ang["dof_pos"]) == pytest.approx(np.asarray(walk["dof_pos"]))
    assert ang["root_angvel"][0].tolist()[2] == pytest.approx(
        SYNTHETIC_WALK_ANGVEL_RAD_S[2] + 0.78
    )
    free = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=True)
    with pytest.raises(ValueError, match="pinned_base"):
        evaluate_lin_vel_jitter(
            free,
            walk,
            offset_mps=[0.5, 0.0, 0.0],
            case_name="lin_vel_+x",
            height_fail_m=0.25,
            ori_fail_rad=1.0,
        )
    with pytest.raises(ValueError, match="pinned_base"):
        evaluate_ang_vel_jitter(
            free,
            walk,
            offset_rad_s=[0.0, 0.0, 0.78],
            case_name="ang_vel_+yaw",
            height_fail_m=0.25,
            ori_fail_rad=1.0,
        )
