"""Assembly, MIT plant, L2.2 micro-env, industrial pipeline. Needs official clones + mujoco."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from assets.combined.assemble import (
    assemble_mjcf,
    assemble_urdf,
    combined_paths,
    convert_position_actuators,
    kinematic_bringup,
)
from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, official_mjcf
from interface.schema import REPO_ROOT

mujoco = pytest.importorskip("mujoco")


def _have_t800() -> bool:
    try:
        from assets.engineai.paths import t800_mjcf

        return t800_mjcf().is_file()
    except FileNotFoundError:
        return False


def test_kinematic_bringup_is_explicitly_forbidden_for_policy() -> None:
    block = kinematic_bringup()
    assert block["enabled"] is True
    assert block["policy_eval_forbidden"] is True
    info = combined_paths()
    assert info["policy_eval_forbidden"] is True
    assert info["mount_ready"] is False


def test_convert_position_to_motor_preserves_forcerange() -> None:
    path = official_mjcf("right", with_mount=True)
    if not path.is_file():
        pytest.skip("wuji-description not cloned")
    xml, gains = convert_position_actuators(path.read_text(encoding="utf-8"))
    assert "<position " not in xml
    assert xml.count("<motor ") == 20
    assert len(gains) == 20
    assert gains[0]["name"] == "r_THJ0"
    assert gains[0]["kp"] > 0
    assert gains[0]["kd"] > 0


def test_assemble_mjcf_compiles_both_hands() -> None:
    if not official_mjcf("left", with_mount=True).is_file() or not _have_t800():
        pytest.skip("official T800/Hand 2 trees not cloned")
    xml, manifest = assemble_mjcf(pad_spheres=False)
    assert manifest["policy_eval_forbidden"] is True
    assert manifest["flange"] == "kinematic_bringup_identity"
    assert manifest["flange_policy_eval_forbidden"] is True
    assert manifest["n_hand_actuators"] == 40
    model = mujoco.MjModel.from_xml_string(xml)
    names = {model.body(i).name for i in range(model.nbody)}
    assert "l_mount" in names and "r_mount" in names
    assert "l_wrist" in names and "r_wrist" in names
    sites = {model.site(i).name for i in range(model.nsite)}
    assert "frame_LINK_BASE_x" in sites
    assert "frame_l_wrist_z" in sites
    assert "frame_LINK_WRIST_END_R_y" in sites
    # pinned base: no freejoint on the robot (nq = 25 body + 40 hand)
    assert model.nq == 65
    assert model.nu == 65
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()


def test_assemble_urdf_welds_mounts() -> None:
    if not official_mjcf("right", with_mount=True).is_file() or not _have_t800():
        pytest.skip("official trees not cloned")
    urdf = assemble_urdf()
    assert "weld_left_hand_kinematic_bringup" in urdf
    assert "weld_right_hand_kinematic_bringup" in urdf
    assert "l_mount" in urdf and "r_mount" in urdf


def test_yaml_flange_matches_assembled_left_wrist() -> None:
    if not official_mjcf("left", with_mount=True).is_file() or not _have_t800():
        pytest.skip("official trees not cloned")
    from assets.combined.flange import wrist_end_to_palm_se3
    from sim.mujoco_env.frames import relative_se3

    xml, _manifest = assemble_mjcf(pad_spheres=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    live = relative_se3(model, data, "LINK_WRIST_END_L", "l_wrist")
    yaml_t = wrist_end_to_palm_se3()
    assert float(np.linalg.norm(live[:3, 3] - yaml_t[:3, 3])) < 2e-4


def test_l22_micro_metrics_have_no_success_rate() -> None:
    if not official_mjcf("right", with_mount=True).is_file():
        pytest.skip("wuji-description not cloned")
    from sim.mujoco_env.grasp_micro import run_micro_episode

    m = run_micro_episode(0.8, 0.01, n_close=30, n_hold=30, lift_m=0.05)
    assert m.finite
    payload = m.__dict__
    assert "success" not in payload
    assert m.slip_m >= 0.0


def test_l22_scan_grid_nine_cells() -> None:
    if not official_mjcf("right", with_mount=True).is_file():
        pytest.skip("wuji-description not cloned")
    from sim.mujoco_env.grasp_micro import run_scan_grid

    cfg = yaml.safe_load((REPO_ROOT / "eval" / "configs" / "l2_scan_grid.yaml").read_text(encoding="utf-8"))
    # Shrink to 2×2 in CI time; still proves the grid runner.
    cfg["friction_static"] = cfg["friction_static"][:2]
    cfg["solref_timeconst_s"] = cfg["solref_timeconst_s"][:2]
    rows = run_scan_grid(cfg, n_close=20, n_hold=20)
    assert len(rows) == 4
    assert all("grasp_success_rate" not in r for r in rows)
    assert all(r["label"] == "SCAN_PLACEHOLDER" for r in rows)


def test_industrial_pipeline_walks_phases() -> None:
    if not official_mjcf("left", with_mount=True).is_file() or not _have_t800():
        pytest.skip("official trees not cloned")
    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.tasks.industrial_pipeline import PHASES, run_industrial_pipeline

    env = CombinedMujocoEnv(scene="industrial")
    env.reset()
    from sim.mujoco_env.scene import SceneSpec

    spec = SceneSpec()
    names = {env.model.geom(i).name for i in range(env.model.ngeom)}
    assert "scan_gun_window" in names
    assert "scan_gun_trigger" in names
    assert "scan_gun_bumper" in names
    assert "scan_gun_housing" in names
    assert "scan_gun_col_body" in names
    assert "scan_gun_col_grip" in names
    env.model.equality("weld_box_grasp")
    env.model.equality("weld_box_grasp_r")
    env.model.equality("weld_gun_grasp")
    tcp = env.model.site_pos[int(env.model.site("gun_tcp").id)]
    assert 0.10 <= float(tcp[0]) <= 0.15
    jnt = int(env.model.body("scan_gun").jntadr[0])
    assert int(env.model.jnt_type[jnt]) == 0  # freejoint, not mocap
    gun0 = env.xpos("scan_gun")
    box0 = env.xpos("box_0")
    # Housing origin sits on the pick-bench top, not inside the slab or the carton.
    assert gun0[2] >= spec.table_height_m + 0.02
    assert abs(gun0[2] - spec.table_height_m) < 0.08
    assert float(np.linalg.norm(gun0[:2] - box0[:2])) > 0.10

    result = run_industrial_pipeline(env, steps_per_phase=40)
    assert result.policy_eval_forbidden
    assert result.kinematic_assist
    assert result.finite
    assert "done" in result.phases
    assert result.phases[0] == PHASES[0]
    assert result.n_steps > 0
    assert result.scan_geometry_ok
    assert result.scan_distance_m is not None
    assert 0.05 <= result.scan_distance_m <= 0.35
    env.model.body("box_0")
    env.model.body("scan_gun")
    env.model.site("gun_tcp")


def test_l2_report_omits_computed_success_rate(tmp_path: Path) -> None:
    import json
    import sys

    from eval.l2_mujoco_closedloop import main

    out = tmp_path / "l2.json"
    old = sys.argv
    try:
        sys.argv = ["l2", "--skip-physics", "--out", str(out)]
        main()
    finally:
        sys.argv = old
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_uncalibrated"
    assert "grasp_success_rate" not in report
    assert report["policy_eval_forbidden"] is True


def test_gen_derived_left_when_cloned() -> None:
    if not (DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/mjcf/left.xml").is_file():
        pytest.skip("wuji-description not cloned")
    from assets.dexhand2.build.gen_derived import generate

    path = generate("left")
    assert path.is_file()
    assert "l_thumb_tip_pad_0" in path.read_text(encoding="utf-8") or "pad_" in path.read_text(encoding="utf-8")
