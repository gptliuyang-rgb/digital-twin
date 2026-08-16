"""Physics Sim2Sim on T800 MJCF. Grasp-success stays null. No combined robot."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key
from sim.mujoco_env.t800_env import OFFICIAL_MJCF, T800MujocoEnv, refuse_combined_robot
from wbc.dims import decoder_history_dim
from wbc.pd_stand import pd_stand_kp_kd


def test_pd_stand_is_25() -> None:
    kp, kd = pd_stand_kp_kd()
    assert kp.shape == (25,)
    assert kd.shape == (25,)
    assert kp[0] == pytest.approx(1080.0)
    assert kd[-1] == pytest.approx(1.0)


def test_foot_frame_offset_is_documented() -> None:
    import yaml
    from interface.schema import REPO_ROOT

    raw = yaml.safe_load(
        (REPO_ROOT / "assets/engineai/meta/t800_frame_offsets.yaml").read_text(encoding="utf-8")
    )
    assert raw["foot"]["delta_z_m"] == pytest.approx(0.06453)
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
