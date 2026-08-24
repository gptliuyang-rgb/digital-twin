"""E1/E2 fit → overlay → MuJoCo replay. Live spec stays REQUIRED_INPUT."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import yaml

from hand.calibration.apply_fragment import guard_live_spec_write, merge_fragment
from hand.calibration.contact_mujoco import (
    apply_contact_to_xml,
    contact_is_calibrated,
    solref_timeconst_from_stiffness,
)
from hand.calibration.fit_params import fit_e1, fit_e2
from hand.calibration.synthetic import (
    K_TRUE_N_PER_M,
    MU_D_TRUE,
    MU_S_TRUE,
    write_synthetic_csvs,
)
from interface.schema import HAND_SPEC_PATH, REQUIRED_INPUT_TOKEN, load_hand_spec, load_yaml


def test_solref_proposal_matches_protocol_formula() -> None:
    tc = solref_timeconst_from_stiffness(2500.0, 0.03)
    assert tc == pytest.approx(2.0 * math.pi / math.sqrt(2500.0 / 0.03), rel=1e-12)


def test_fit_e1_e2_from_synthetic_csvs(tmp_path: Path) -> None:
    paths = write_synthetic_csvs(tmp_path, seed=0)
    e1 = fit_e1(paths["e1"])
    e2 = fit_e2(paths["e2"])
    assert e1["n_static"] == 30
    assert e1["friction_vs_cardboard_static"] == pytest.approx(MU_S_TRUE, rel=0.08)
    assert e1["friction_vs_cardboard_dynamic"] == pytest.approx(MU_D_TRUE, rel=0.12)
    assert e2["n"] == 15
    assert e2["normal_stiffness_n_per_m"] == pytest.approx(K_TRUE_N_PER_M, rel=0.12)
    assert e2["solref_timeconst_s"] > 0


def test_fit_e2_uses_precomputed_k_column(tmp_path: Path) -> None:
    csv = tmp_path / "e2.csv"
    csv.write_text(
        "trial,finger,skin,disp_m,force_n,k_n_per_m,batch,fw\n"
        "1,index,on,,,1800,batch,fw\n"
        "2,index,on,,,2200,batch,fw\n",
        encoding="utf-8",
    )
    out = fit_e2(csv)
    assert out["n"] == 2
    assert out["normal_stiffness_n_per_m"] == pytest.approx(2000.0)


def test_live_spec_stays_uncalibrated() -> None:
    spec = load_hand_spec()
    assert spec.raw["friction_vs_cardboard_static"] == REQUIRED_INPUT_TOKEN
    assert spec.raw["normal_stiffness_n_per_m"] == REQUIRED_INPUT_TOKEN
    assert not contact_is_calibrated(spec.raw)


def test_apply_fragment_writes_overlay_not_live(tmp_path: Path) -> None:
    paths = write_synthetic_csvs(tmp_path / "csv", seed=1)
    fragment = {**fit_e1(paths["e1"]), **fit_e2(paths["e2"]), "do_not_treat_as_committed_hardware": True}
    base = load_yaml(HAND_SPEC_PATH)
    overlay = merge_fragment(base, fragment)
    assert overlay["friction_vs_cardboard_static"] != REQUIRED_INPUT_TOKEN
    assert load_yaml(HAND_SPEC_PATH)["friction_vs_cardboard_static"] == REQUIRED_INPUT_TOKEN
    assert contact_is_calibrated(overlay)


def test_apply_refuses_live_spec_without_commit_flag() -> None:
    fragment = {
        "friction_vs_cardboard_static": 0.7,
        "do_not_treat_as_committed_hardware": True,
    }
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        guard_live_spec_write(HAND_SPEC_PATH, fragment, commit_live=False, accept_synthetic=False)
    with pytest.raises(SystemExit, match="synthetic"):
        guard_live_spec_write(HAND_SPEC_PATH, fragment, commit_live=True, accept_synthetic=False)
    assert load_yaml(HAND_SPEC_PATH)["friction_vs_cardboard_static"] == REQUIRED_INPUT_TOKEN


def test_patch_pad_geoms_sets_friction_solref() -> None:
    xml = '<geom name="r_index_finger_tip_pad_0" type="sphere" size="0.008" condim="4"/>'
    patched = apply_contact_to_xml(
        xml,
        {
            "friction_vs_cardboard_static": 0.75,
            "friction_vs_cardboard_dynamic": 0.55,
            "normal_stiffness_n_per_m": 2500.0,
            "m_eff_kg": 0.03,
        },
    )
    assert "friction=" in patched
    assert "solref=" in patched
    assert "0.750000" in patched


def test_validate_sim_refuses_uncalibrated_live_spec() -> None:
    from hand.calibration.validate_sim import main as validate_main

    old = sys.argv
    try:
        sys.argv = ["validate_sim"]
        with pytest.raises(SystemExit, match="REQUIRED_INPUT"):
            validate_main()
    finally:
        sys.argv = old


def test_validate_sim_replay_on_overlay(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    from hand.calibration.validate_sim import run_validate

    paths = write_synthetic_csvs(tmp_path / "csv", seed=0)
    fragment = {**fit_e1(paths["e1"]), **fit_e2(paths["e2"])}
    overlay = merge_fragment(load_yaml(HAND_SPEC_PATH), fragment)
    report = run_validate(overlay)
    assert report["finite"]
    assert report["e1"]["ok"]
    assert report["e2"]["finite"]
    assert report["grasp_success_rate"] is None
    assert report["human_must_accept_solref"] is True


def test_l2_overlay_ready_but_no_success_rate(tmp_path: Path) -> None:
    from eval.l2_mujoco_closedloop import main

    paths = write_synthetic_csvs(tmp_path / "csv", seed=0)
    fragment = {
        **fit_e1(paths["e1"]),
        **fit_e2(paths["e2"]),
        "do_not_treat_as_committed_hardware": True,
        "calibration_kind": "synthetic_dry_run",
    }
    overlay = merge_fragment(load_yaml(HAND_SPEC_PATH), fragment)
    spec_path = tmp_path / "overlay.yaml"
    spec_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    out = tmp_path / "l2.json"
    old = sys.argv
    try:
        sys.argv = ["l2", "--skip-physics", "--spec", str(spec_path), "--out", str(out)]
        main()
    finally:
        sys.argv = old
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["contact_calibrated"] is True
    assert report["uncalibrated"] is False
    assert "grasp_success_rate" not in report
    assert report["policy_eval_forbidden"] is True
    assert load_hand_spec().raw["friction_vs_cardboard_static"] == REQUIRED_INPUT_TOKEN


def test_gen_derived_overlay_writes_pad_friction() -> None:
    from assets.dexhand2.build.gen_derived import generate
    from assets.dexhand2.build.ingest_official import official_mjcf

    if not official_mjcf("right").is_file():
        pytest.skip("wuji-description not cloned")
    paths = write_synthetic_csvs()
    fragment = {
        **fit_e1(paths["e1"]),
        **fit_e2(paths["e2"]),
        "do_not_treat_as_committed_hardware": True,
    }
    overlay = merge_fragment(load_yaml(HAND_SPEC_PATH), fragment)
    spec_path = paths["e1"].parent.parent / "generated" / "overlay_for_test.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    uncal = generate("right")
    uncal_text = uncal.read_text(encoding="utf-8")
    cal = generate("right", spec_path=spec_path)
    cal_text = cal.read_text(encoding="utf-8")
    assert "_pad_0" in uncal_text
    assert "friction=" not in uncal_text.split('name="r_thumb_tip_pad_0"', 1)[1].split("/>", 1)[0]
    assert "friction=" in cal_text.split('name="r_thumb_tip_pad_0"', 1)[1].split("/>", 1)[0]
    assert "solref=" in cal_text
    assert load_hand_spec().raw["friction_vs_cardboard_static"] == REQUIRED_INPUT_TOKEN
