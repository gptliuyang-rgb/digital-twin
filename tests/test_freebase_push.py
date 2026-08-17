"""Lean vs fall labels, lateral root-push, and freejoint air-drop."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from eval.lean_classify import classify_posture
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key
from sim.mujoco_env.t800_env import OFFICIAL_MJCF, T800MujocoEnv, refuse_combined_robot
from wbc.foot_frame import FOOT_FRAME_DECISION


def test_classify_posture_labels_the_official_hold() -> None:
    # PHASE_FREEBASE.md: 3 s official hold, pelvis 0.864 m, tilt 0.72 rad, fallen=false.
    assert classify_posture(0.864, 0.72) == "leaned"
    assert classify_posture(1.03, 0.05) == "upright"
    assert classify_posture(0.30, 0.10) == "fallen"
    assert classify_posture(0.90, 0.85) == "fallen"
    assert classify_posture(0.864, 0.80) == "leaned"  # fall is strictly greater than 0.80
    assert classify_posture(0.864, 0.801) == "fallen"


def test_pinned_base_refuses_root_linvel() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=True)
    with pytest.raises(ValueError, match="pinned_base"):
        env.apply_root_linvel(np.array([0.0, 0.5, 0.0]))
    with pytest.raises(ValueError, match="pinned_base"):
        env.apply_root_force_n(np.array([0.0, 10.0, 0.0]))
    with pytest.raises(ValueError, match="pinned_base"):
        env.apply_root_angvel(np.array([0.0, 0.0, 0.78]))
    with pytest.raises(ValueError, match="pinned_base"):
        env.apply_root_torque_nm(np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="pinned_base"):
        env.root_ang_inertia_kgm2()


def test_fixture_wbc_slide_friction_sets_floor_and_feet() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=True)
    ids = env.wbc_slide_friction_geom_ids()
    names = [env._geom_name(i).lower() for i in ids]
    bodies = [env._body_name(int(env.model.geom_bodyid[i])).upper() for i in ids]
    assert env.n_plane >= 1
    assert any("floor" in n for n in names)
    assert any("FOOT" in b for b in bodies)
    backup = env.geom_friction_copy()
    applied = env.set_wbc_slide_friction(0.3)
    assert applied["not_pad_cardboard"] is True
    assert applied["not_dexhand2_contact"] is True
    assert applied["mujoco_contact"] == "elementwise_max_of_two_geoms"
    assert any("FOOT" in b.upper() for b in applied["body_names"])
    for gid in ids:
        assert env.model.geom_friction[gid, 0] == pytest.approx(0.3)
    floor_gid = next(i for i, n in zip(ids, names, strict=True) if "floor" in n)
    assert env.model.geom_friction[floor_gid, 1] == pytest.approx(0.005)
    assert env.model.geom_friction[floor_gid, 2] == pytest.approx(0.0001)
    env.restore_geom_friction(backup)
    assert env.model.geom_friction[floor_gid, 0] == pytest.approx(1.0)
    air = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=False)
    with pytest.raises(ValueError, match="floor plane"):
        air.set_wbc_slide_friction(0.3)
    with pytest.raises(ValueError, match="mu_slide"):
        env.set_wbc_slide_friction(0.0)


def test_fixture_base_com_offset_adds_to_compiled_ipos() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=True)
    compiled = env.compiled_base_ipos_m()
    assert compiled.tolist() == [0.0, 0.0, 0.0]
    bid = env.root_body_id()
    assert env._body_name(bid) == "LINK_BASE"
    applied = env.set_base_com_offset([0.075, 0.0, 0.0])
    assert applied["not_wrist_com"] is True
    assert applied["not_pad_cardboard"] is True
    assert applied["additive_to_compiled_ipos"] is True
    assert applied["mujoco_channel"] == "body_ipos[LINK_BASE]"
    assert applied["offset_m"] == [0.075, 0.0, 0.0]
    assert applied["applied_ipos_m"] == [0.075, 0.0, 0.0]
    assert applied["body_mass_kg"] == pytest.approx(1.0)
    assert env.model.body_ipos[bid].tolist() == pytest.approx([0.075, 0.0, 0.0])
    env.restore_base_ipos()
    assert env.model.body_ipos[bid].tolist() == pytest.approx([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        env.set_base_com_offset([float("nan"), 0.0, 0.0])


def test_fixture_push_suite_is_not_a_sonic_gate() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_freebase_push import run
    from eval.l3_isaac_bind import try_bind_and_step

    report = run(source="fixture")
    refuse_grasp_success_key(report)
    assert report["grasp_success_rate"] is None
    assert report["combined_robot"] == "PolicyEvalBlocked"
    assert report["not_a_sonic_gate"] is True
    assert report["local_tracking_success"] is False
    assert report["hold"]["posture"] in ("upright", "leaned", "fallen")
    assert report["hold"]["local_tracking_success"] is False
    assert report["lateral_push"]["kind"] == "one_shot_qvel"
    assert report["lateral_push"]["name"] == "+y"
    assert report["lateral_push"]["lin_vel_mps"] == [0.0, 0.5, 0.0]
    assert report["lateral_push"]["injected"] is True
    assert "pre_push_posture" in report["lateral_push"]
    assert report["lateral_push"]["pre_push_posture"] in ("upright", "leaned", "fallen")
    assert report["lateral_push"]["posture"] in ("upright", "leaned", "fallen")
    names = [c["name"] for c in report["push_sweep"]]
    assert names == ["-x", "+x", "-y", "+y"]
    assert report["push_sweep_summary"]["n_cases"] == 4
    assert report["push_sweep_summary"]["not_a_sonic_gate"] is True
    by_name = {c["name"]: c["lin_vel_mps"] for c in report["push_sweep"]}
    assert by_name["-x"] == [-0.5, 0.0, 0.0]
    assert by_name["+x"] == [0.5, 0.0, 0.0]
    assert by_name["-y"] == [0.0, -0.5, 0.0]
    assert by_name["+y"] == [0.0, 0.5, 0.0]
    assert all(c["kind"] == "one_shot_qvel" for c in report["push_sweep"])
    assert all(c["not_a_sonic_gate"] for c in report["push_sweep"])
    force_names = [c["name"] for c in report["force_sweep"]]
    assert force_names == [
        "-x_T1.0s",
        "-x_T3.0s",
        "+x_T1.0s",
        "+x_T3.0s",
        "-y_T1.0s",
        "-y_T3.0s",
        "+y_T1.0s",
        "+y_T3.0s",
    ]
    assert report["force_sweep_summary"]["n_cases"] == 8
    assert report["force_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["sustained_force"]["name"] == "+y_T1.0s"
    assert report["sustained_force"]["kind"] == "sustained_force"
    assert report["sustained_force"]["duration_s"] == 1.0
    assert report["sustained_force"]["lin_vel_mps"] == [0.0, 0.5, 0.0]
    assert report["sustained_force"]["force_formula"] == "F = m * v / T"
    assert report["sustained_force"]["mass_kg"] > 0.0
    expected_fy = report["sustained_force"]["mass_kg"] * 0.5 / 1.0
    assert report["sustained_force"]["force_n"] == pytest.approx([0.0, expected_fy, 0.0])
    assert all(c["kind"] == "sustained_force" for c in report["force_sweep"])
    assert all(c["not_a_sonic_gate"] for c in report["force_sweep"])
    assert all(c["grasp_success_rate"] is None for c in report["force_sweep"])
    t3 = next(c for c in report["force_sweep"] if c["name"] == "+y_T3.0s")
    assert t3["duration_s"] == 3.0
    assert t3["force_n"] == pytest.approx(
        [0.0, report["sustained_force"]["mass_kg"] * 0.5 / 3.0, 0.0]
    )
    angvel_names = [c["name"] for c in report["angvel_sweep"]]
    assert angvel_names == ["-roll", "+roll", "-pitch", "+pitch", "-yaw", "+yaw"]
    assert report["angvel_sweep_summary"]["n_cases"] == 6
    assert report["angvel_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["angvel_push"]["name"] == "+yaw"
    assert report["angvel_push"]["kind"] == "one_shot_qvel"
    assert report["angvel_push"]["ang_vel_rad_s"] == [0.0, 0.0, 0.78]
    assert all(c["kind"] == "one_shot_qvel" for c in report["angvel_sweep"])
    assert all(c["not_a_sonic_gate"] for c in report["angvel_sweep"])
    assert all(c["grasp_success_rate"] is None for c in report["angvel_sweep"])
    torque_names = [c["name"] for c in report["torque_sweep"]]
    assert torque_names == [
        "-roll_T1.0s",
        "-roll_T3.0s",
        "+roll_T1.0s",
        "+roll_T3.0s",
        "-pitch_T1.0s",
        "-pitch_T3.0s",
        "+pitch_T1.0s",
        "+pitch_T3.0s",
        "-yaw_T1.0s",
        "-yaw_T3.0s",
        "+yaw_T1.0s",
        "+yaw_T3.0s",
    ]
    assert report["torque_sweep_summary"]["n_cases"] == 12
    assert report["torque_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["sustained_torque"]["name"] == "+yaw_T1.0s"
    assert report["sustained_torque"]["kind"] == "sustained_torque"
    assert report["sustained_torque"]["duration_s"] == 1.0
    assert report["sustained_torque"]["ang_vel_rad_s"] == [0.0, 0.0, 0.78]
    assert report["sustained_torque"]["torque_formula"] == "tau = I @ omega / T"
    inertia = np.asarray(report["sustained_torque"]["inertia_kgm2"], dtype=np.float64)
    assert inertia.shape == (3, 3)
    expected_tau = inertia @ np.array([0.0, 0.0, 0.78]) / 1.0
    assert report["sustained_torque"]["torque_nm"] == pytest.approx(expected_tau.tolist())
    assert all(c["kind"] == "sustained_torque" for c in report["torque_sweep"])
    assert all(c["not_a_sonic_gate"] for c in report["torque_sweep"])
    assert all(c["grasp_success_rate"] is None for c in report["torque_sweep"])
    yaw_t3 = next(c for c in report["torque_sweep"] if c["name"] == "+yaw_T3.0s")
    assert yaw_t3["duration_s"] == 3.0
    inertia_t3 = np.asarray(yaw_t3["inertia_kgm2"], dtype=np.float64)
    expected_tau_t3 = inertia_t3 @ np.array([0.0, 0.0, 0.78]) / 3.0
    assert yaw_t3["torque_nm"] == pytest.approx(expected_tau_t3.tolist())
    vert_names = [c["name"] for c in report["vertical_sweep"]]
    assert vert_names == ["-z", "+z"]
    assert report["vertical_sweep_summary"]["n_cases"] == 2
    assert report["vertical_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["vertical_push"]["name"] == "+z"
    assert report["vertical_push"]["kind"] == "one_shot_qvel"
    assert report["vertical_push"]["lin_vel_mps"] == [0.0, 0.0, 0.2]
    assert all(c["kind"] == "one_shot_qvel" for c in report["vertical_sweep"])
    assert all(c["not_a_sonic_gate"] for c in report["vertical_sweep"])
    assert all(c["grasp_success_rate"] is None for c in report["vertical_sweep"])
    vforce_names = [c["name"] for c in report["vertical_force_sweep"]]
    assert vforce_names == ["-z_T1.0s", "-z_T3.0s", "+z_T1.0s", "+z_T3.0s"]
    assert report["vertical_force_sweep_summary"]["n_cases"] == 4
    assert report["vertical_force_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["sustained_vertical"]["name"] == "+z_T1.0s"
    assert report["sustained_vertical"]["kind"] == "sustained_force"
    assert report["sustained_vertical"]["duration_s"] == 1.0
    assert report["sustained_vertical"]["lin_vel_mps"] == [0.0, 0.0, 0.2]
    assert report["sustained_vertical"]["force_formula"] == "F = m * v / T"
    expected_fz = report["sustained_vertical"]["mass_kg"] * 0.2 / 1.0
    assert report["sustained_vertical"]["force_n"] == pytest.approx([0.0, 0.0, expected_fz])
    assert all(c["kind"] == "sustained_force" for c in report["vertical_force_sweep"])
    assert all(c["not_a_sonic_gate"] for c in report["vertical_force_sweep"])
    assert all(c["grasp_success_rate"] is None for c in report["vertical_force_sweep"])
    z_t3 = next(c for c in report["vertical_force_sweep"] if c["name"] == "+z_T3.0s")
    assert z_t3["duration_s"] == 3.0
    assert z_t3["force_n"] == pytest.approx(
        [0.0, 0.0, report["sustained_vertical"]["mass_kg"] * 0.2 / 3.0]
    )
    friction_names = [c["name"] for c in report["friction_sweep"]]
    assert friction_names == ["mu_s_0.3", "mu_s_1.6"]
    assert report["friction_sweep_summary"]["n_cases"] == 2
    assert report["friction_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["friction_hold"]["name"] == "mu_s_0.3"
    assert report["friction_hold"]["kind"] == "wbc_floor_slide_friction"
    assert report["friction_hold"]["mu_slide"] == 0.3
    assert report["friction_hold"]["not_pad_cardboard"] is True
    assert report["friction_hold"]["dynamic_friction_not_mapped"] is True
    assert report["friction_hold"]["restitution_not_mapped"] is True
    assert all(c["kind"] == "wbc_floor_slide_friction" for c in report["friction_sweep"])
    assert all(c["not_a_sonic_gate"] for c in report["friction_sweep"])
    assert all(c["grasp_success_rate"] is None for c in report["friction_sweep"])
    by_mu = {c["name"]: c["mu_slide"] for c in report["friction_sweep"]}
    assert by_mu["mu_s_0.3"] == 0.3
    assert by_mu["mu_s_1.6"] == 1.6
    assert "mu_slide" not in report["hold"]
    assert report["airdrop"]["n_plane"] == 0
    assert report["airdrop"]["nq"] == 32
    assert report["airdrop"]["freejoint_moved"] is True
    assert report["airdrop"]["drop_m"] > 0.2
    assert report["foot_frame"] == FOOT_FRAME_DECISION
    with pytest.raises(PolicyEvalBlocked):
        refuse_combined_robot()

    bind = try_bind_and_step()
    refuse_grasp_success_key(bind)
    assert bind["grasp_success_rate"] is None
    assert bind["reset_step_ran"] is False
    assert bind["runtime"] == "unavailable"
    assert bind["status"] == "isaac_bind_unavailable"


@pytest.mark.skipif(not OFFICIAL_MJCF.is_file(), reason="engineai Native SDK not cloned")
def test_official_push_and_airdrop_report_honestly() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_freebase_push import run

    report = run(source="official")
    refuse_grasp_success_key(report)
    assert report["grasp_success_rate"] is None
    assert report["not_a_sonic_gate"] is True
    assert report["local_tracking_success"] is False
    assert report["gains_source"] == "pd_stand_bringup_not_sonic"
    assert report["hold"]["posture"] in ("upright", "leaned", "fallen")
    # Official 3 s hold previously leaned ~0.72 rad; do not treat that as a stand.
    if report["hold"]["end_tilt_rad"] > 0.20 and not report["hold"]["fallen"]:
        assert report["hold"]["posture"] == "leaned"
    assert report["airdrop"]["freejoint_moved"] is True
    assert report["airdrop"]["n_plane"] == 0
    assert np.isfinite(report["lateral_push"]["end_tilt_rad"])
    assert report["lateral_push"]["fall_rate"] in (0.0, 1.0)
    # Push is applied to the leaned hold (settle_s = hold_s), not to t=0.5 s upright.
    if report["hold"]["posture"] == "leaned":
        assert report["lateral_push"]["pre_push_posture"] == "leaned"
        assert all(c["pre_push_posture"] == "leaned" for c in report["push_sweep"])
    assert report["push_sweep_summary"]["n_cases"] == 4
    assert report["push_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["force_sweep_summary"]["n_cases"] == 8
    assert report["force_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["sustained_force"]["kind"] == "sustained_force"
    if report["hold"]["posture"] == "leaned":
        assert report["sustained_force"]["pre_push_posture"] == "leaned"
        assert all(c["pre_push_posture"] == "leaned" for c in report["force_sweep"])
    assert report["angvel_sweep_summary"]["n_cases"] == 6
    assert report["angvel_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["torque_sweep_summary"]["n_cases"] == 12
    assert report["torque_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["sustained_torque"]["kind"] == "sustained_torque"
    if report["hold"]["posture"] == "leaned":
        assert report["angvel_push"]["pre_push_posture"] == "leaned"
        assert report["sustained_torque"]["pre_push_posture"] == "leaned"
        assert all(c["pre_push_posture"] == "leaned" for c in report["angvel_sweep"])
        assert all(c["pre_push_posture"] == "leaned" for c in report["torque_sweep"])
        assert report["vertical_push"]["pre_push_posture"] == "leaned"
        assert report["sustained_vertical"]["pre_push_posture"] == "leaned"
        assert all(c["pre_push_posture"] == "leaned" for c in report["vertical_sweep"])
        assert all(c["pre_push_posture"] == "leaned" for c in report["vertical_force_sweep"])
    assert report["vertical_sweep_summary"]["n_cases"] == 2
    assert report["vertical_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["vertical_force_sweep_summary"]["n_cases"] == 4
    assert report["vertical_force_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["friction_sweep_summary"]["n_cases"] == 2
    assert report["friction_sweep_summary"]["not_a_sonic_gate"] is True
    assert report["friction_hold"]["kind"] == "wbc_floor_slide_friction"
    assert report["friction_hold"]["not_pad_cardboard"] is True
    # Fall on any axis, duration, or friction extremum is a diagnostic, not a SONIC fail.
    assert report["not_a_sonic_gate"] is True
