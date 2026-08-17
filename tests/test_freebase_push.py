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
    # Fall on any axis is a diagnostic, not a SONIC fail.
    assert report["not_a_sonic_gate"] is True
