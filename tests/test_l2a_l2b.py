"""L2a/L2b units: gates, E3, wrist jump, flex box, gain scan, pad metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from assets.objects.flex import FlexBoxSpec, flex_box_xml, should_split
from eval.gates import assert_no_grasp_success_rate, find_forbidden_keys
from hand.calibration.fit_params import fit_e3
from hand.calibration.synthetic import E3_FIRST_SLIP_KG, E3_MAX_HELD_KG, write_synthetic_csvs
from runtime.safety_filter import CartesianJumpFilter


def test_gates_reject_success_rate_key() -> None:
    with pytest.raises(AssertionError, match="ADR-004"):
        assert_no_grasp_success_rate({"l2": {"grasp_success_rate": 0.9}})
    assert find_forbidden_keys({"uncalibrated_slip_m": 0.01}) == []


def test_fit_e3_synthetic(tmp_path: Path) -> None:
    paths = write_synthetic_csvs(tmp_path, seed=0)
    out = fit_e3(paths["e3"])
    assert out["n_hold"] > 0
    assert out["e3_max_held_mass_kg"] == pytest.approx(E3_MAX_HELD_KG)
    assert out["e3_first_slip_mass_kg"] == pytest.approx(E3_FIRST_SLIP_KG)
    assert out["do_not_treat_as_payload_rating"] is True
    assert "motor_max_torque_nm" not in out


def test_wrist_jump_filter() -> None:
    filt = CartesianJumpFilter(max_delta_m=0.05)
    ok = filt.filter(np.zeros(3))
    assert ok.accepted
    jump = filt.filter(np.array([0.2, 0.0, 0.0]))
    assert not jump.accepted
    assert jump.reason == "wrist_jump"
    nan = filt.filter(np.array([np.nan, 0.0, 0.0]))
    assert not nan.accepted


def test_flex_box_splits_above_40cm() -> None:
    assert should_split((0.45, 0.30, 0.30))
    assert not should_split((0.18, 0.14, 0.12))
    xml = flex_box_xml(
        FlexBoxSpec(size_m=(0.50, 0.40, 0.30), mass_kg=8.0, pos_m=(0.0, 0.0, 0.5), name="box_flex")
    )
    assert xml.count("<joint ") == 8
    assert "box_flex_3" in xml


def test_flex_box_compiles_when_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    inner = flex_box_xml(
        FlexBoxSpec(size_m=(0.50, 0.40, 0.30), mass_kg=8.0, pos_m=(0.0, 0.0, 0.4), name="box_flex")
    )
    xml = (
        '<mujoco model="flex"><compiler angle="radian"/><worldbody>'
        '<geom name="floor" type="plane" size="1 1 0.05"/>'
        f"{inner}</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()


def test_gain_scan_corners_when_cloned() -> None:
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from assets.dexhand2.build.ingest_official import official_mjcf
    from interface.schema import REPO_ROOT
    from sim.mujoco_env.grasp_micro import run_gain_scan

    if not official_mjcf("right", with_mount=True).is_file():
        pytest.skip("wuji-description not cloned")
    cfg = yaml.safe_load((REPO_ROOT / "eval" / "configs" / "l2_scan_grid.yaml").read_text(encoding="utf-8"))
    rows = run_gain_scan(cfg, n_close=12, n_hold=12, mode="corners")
    assert len(rows) == 4
    assert all(r["label"] == "GAIN_SCAN_PLACEHOLDER" for r in rows)
    assert all("grasp_success_rate" not in r for r in rows)


def test_bimanual_micro_when_cloned() -> None:
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from assets.dexhand2.build.ingest_official import official_mjcf
    from sim.mujoco_env.bimanual_box import run_bimanual_episode

    if not official_mjcf("left", with_mount=True).is_file():
        pytest.skip("wuji-description not cloned")
    m = run_bimanual_episode(0.8, 0.01, n_close=20, n_hold=20, lift_m=0.04)
    assert m.finite
    assert m.policy_eval_forbidden
    assert m.slip_m >= 0.0


def test_pad_metrics_when_cloned() -> None:
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from assets.dexhand2.build.ingest_official import official_mjcf
    from assets.dexhand2.build.pad_metrics import measure_side

    if not official_mjcf("right").is_file():
        pytest.skip("wuji-description not cloned")
    report = measure_side("right")
    assert report["n_tip_sites"] == 5
    assert report["derived_mean_pad_sphere_m"] is not None
    n_with_pads = sum(1 for r in report["derived"] if r["n_pad_spheres"] >= 1)
    assert n_with_pads == 5, [(r["site"], r["n_pad_spheres"]) for r in report["derived"]]


def test_physics_industrial_flag_smoke_when_cloned() -> None:
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from assets.dexhand2.build.ingest_official import official_mjcf
    from assets.engineai.paths import t800_mjcf
    from sim.tasks.physics_industrial import run_physics_industrial

    try:
        have_t800 = t800_mjcf().is_file()
    except FileNotFoundError:
        have_t800 = False
    if not official_mjcf("left", with_mount=True).is_file() or not have_t800:
        pytest.skip("official trees not cloned")
    result = run_physics_industrial(steps_per_phase=4, substeps=2, settle_steps=20)
    assert result.kinematic_assist is False
    assert result.constraint_weld is True
    assert result.policy_eval_forbidden
    assert "done" in result.phases
    assert "grasp_success_rate" not in result.__dict__
    assert result.finite
