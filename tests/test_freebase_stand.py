"""Free-base T800 PD stand. Fall is reported, never a SONIC gate."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from sim.isaaclab_env.probe import isaac_runtime_status
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key
from sim.mujoco_env.t800_env import OFFICIAL_MJCF, T800MujocoEnv, refuse_combined_robot
from wbc.foot_frame import FOOT_FRAME_DECISION


def test_isaac_probe_is_safe_without_isaac() -> None:
    status = isaac_runtime_status()
    assert status["grasp_success_rate"] is None
    assert status["combined_robot"] == "PolicyEvalBlocked"
    assert status["runtime"] in ("bound", "unavailable")


def test_combined_robot_still_blocked() -> None:
    with pytest.raises(PolicyEvalBlocked):
        refuse_combined_robot()


def test_fixture_freebase_compiles_and_moves() -> None:
    pytest.importorskip("mujoco")
    pinned = T800MujocoEnv(source="fixture", pinned_base=True)
    free_air = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=False)
    free_floor = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=True)
    assert pinned.model.nq == 25
    assert free_air.model.nq == 32
    assert free_floor.expected_nq() == 32
    assert free_air.n_plane == 0
    assert free_floor.n_plane >= 1
    pinned.reset()
    free_air.reset()
    pinned.hold(0.4)
    free_air.hold(0.4)
    z_pinned = float(pinned.body_pose("pelvis")[0][2])
    z_air = float(free_air.body_pose("pelvis")[0][2])
    # No floor ⇒ gravity must move the freejoint. With a floor the 1 kg boxes may hold.
    assert z_pinned == pytest.approx(1.03, abs=0.02)
    assert z_air < z_pinned - 0.2


def test_fixture_freebase_report_is_not_a_sonic_gate() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_freebase_stand import run

    report = run(source="fixture", hold_s=0.6)
    refuse_grasp_success_key(report)
    assert report["grasp_success_rate"] is None
    assert report["combined_robot"] == "PolicyEvalBlocked"
    assert report["not_a_sonic_gate"] is True
    assert report["local_tracking_success"] is False
    assert report["foot_frame"] == FOOT_FRAME_DECISION
    assert report["pinned_base"] is False
    assert report["nq"] == 32
    assert report["fall_rate"] in (0.0, 1.0)
    assert report["n_plane"] >= 1
    assert report["end_foot"]["foot_frame"] == FOOT_FRAME_DECISION
    assert report["posture"] in ("upright", "leaned", "fallen")
    env = T800MujocoEnv(source="fixture", pinned_base=False, add_floor=True)
    env.reset()
    diag = env.foot_diagnostics()
    assert "left_mjcf_foot_z_m" in diag
    assert "left_urdf_sole_z_m" in diag


@pytest.mark.skipif(not OFFICIAL_MJCF.is_file(), reason="engineai Native SDK not cloned")
def test_official_freebase_pd_stand_reports_honestly() -> None:
    pytest.importorskip("mujoco")
    from eval.l2_freebase_stand import run

    env = T800MujocoEnv(source="official", pinned_base=False, add_floor=True)
    assert env.model.nq == 32
    assert env.n_plane >= 1
    env.reset()
    env.set_q(np.zeros(25))
    env._mujoco.mj_forward(env.model, env.data)
    feet = env.foot_diagnostics()
    # URDF sole is 64.53 mm below the MJCF foot body at identity orientation.
    assert feet["left_mjcf_foot_z_m"] - feet["left_urdf_sole_z_m"] == pytest.approx(0.06453, abs=5e-3)
    report = run(source="official", hold_s=1.0)
    refuse_grasp_success_key(report)
    assert report["grasp_success_rate"] is None
    assert report["not_a_sonic_gate"] is True
    assert report["local_tracking_success"] is False
    assert report["q_des_source"] == "official_pd_stand_desired_joint_position"
    assert report["gains_source"] == "pd_stand_bringup_not_sonic"
    assert report["foot_frame"] == FOOT_FRAME_DECISION
    # Do not gate on fallen vs stood: bring-up PD is not SONIC.
    assert report["fall_rate"] in (0.0, 1.0)
    assert np.isfinite(report["end_pelvis_z_m"])
