"""GMR T800 export, motion_lib schema, and clip filter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from interface.schema import REPO_ROOT
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import load_t800_sonic
from wbc.filter import filter_motion_clip, parse_mjcf_joint_limits
from wbc.gmr.export import (
    BODY_MAP_PATH,
    export_all,
    ik_config,
    load_body_map,
    refuse_pm01_torso_name,
    sonic_tracked_bodies,
    validate_robot_bodies,
)
from wbc.gmr.motion_lib import MotionLibError, validate_motion_lib

T800_MJCF = (
    REPO_ROOT
    / "third_party"
    / "engineai-native-sdk"
    / "assets"
    / "resource"
    / "robot"
    / "t800"
    / "xml"
    / "serial_links.xml"
)


def test_sonic_tracked_bodies_match_contract() -> None:
    tracked = sonic_tracked_bodies()
    sonic = load_t800_sonic()["tracked_bodies"]
    assert tracked["left_wrist"] == "LINK_WRIST_END_L"
    assert tracked["right_wrist"] == "LINK_WRIST_END_R"
    assert tracked["head"] == "LINK_HEAD_YAW"
    assert tracked["pelvis"] == "LINK_BASE"
    assert tracked == {k: sonic[k] for k in tracked}


def test_ik_config_refuses_pm01_body_names() -> None:
    # Empty overlay = body_map copy (scale 1.0). A written tpose_offsets.yaml is a later pass.
    ik = ik_config("smplx", tpose={})
    refuse_pm01_torso_name(ik)
    assert "LINK_WAIST_YAW" in ik["ik_match_table1"]
    assert "LINK_WRIST_END_L" in ik["ik_match_table1"]
    assert "LINK_HEAD_YAW" in ik["ik_match_table1"]
    assert ik["human_scale_table"]["pelvis"] == 1.0
    assert load_body_map()["quat_offset_status"] == "uncalibrated_copied_from_gmr_pm01"
    with pytest.raises(ValueError, match="LINK_WAIST_YAW"):
        refuse_pm01_torso_name({"ik_match_table1": {"LINK_TORSO_YAW": []}, "ik_match_table2": {}})


def test_export_writes_json(tmp_path: Path) -> None:
    written = export_all(tmp_path)
    smplx = json.loads(Path(written["smplx"]).read_text(encoding="utf-8"))
    bvh = json.loads(Path(written["bvh_lafan1"]).read_text(encoding="utf-8"))
    assert smplx["robot_root_name"] == "LINK_BASE"
    assert bvh["human_root_name"] == "Hips"
    assert smplx["ik_match_table1"]["LINK_WRIST_END_L"][0] == "left_wrist"
    committed = BODY_MAP_PATH.with_name("smplx_to_t800.json")
    if committed.is_file():
        gold = json.loads(committed.read_text(encoding="utf-8"))
        assert gold["ik_match_table1"].keys() == smplx["ik_match_table1"].keys()


@pytest.mark.skipif(not T800_MJCF.is_file(), reason="T800 MJCF not cloned")
def test_gmr_bodies_exist_in_official_mjcf() -> None:
    missing = validate_robot_bodies()
    assert missing == []
    text = T800_MJCF.read_text(encoding="utf-8")
    assert 'name="LINK_WAIST_YAW"' in text
    assert 'name="LINK_TORSO_YAW"' not in text


@pytest.mark.skipif(not T800_MJCF.is_file(), reason="T800 MJCF not cloned")
def test_filter_uses_official_joint_limits() -> None:
    cfg = load_t800_sonic()
    text = T800_MJCF.read_text(encoding="utf-8")
    limits = parse_mjcf_joint_limits(text, cfg["joint_order"])
    assert limits.shape == (25, 2)
    np.testing.assert_allclose(limits[0], [-3.316, 2.269], atol=1e-6)
    q = np.zeros((20, 25))
    ok = filter_motion_clip(q, joint_limits_rad=limits, dt_s=0.05)
    assert ok.keep
    q_bad = q.copy()
    q_bad[3, 0] = limits[0, 1] + 0.5
    bad = filter_motion_clip(q_bad, joint_limits_rad=limits, dt_s=0.05)
    assert not bad.keep
    assert any("joint_limit" in r for r in bad.reasons)


def test_filter_foot_penetration_and_nan() -> None:
    limits = np.tile(np.array([-1.0, 1.0]), (4, 1))
    q = np.zeros((10, 4))
    feet = np.ones((10, 2)) * 0.02
    feet[4, 0] = -0.02
    res = filter_motion_clip(q, joint_limits_rad=limits, foot_height_m=feet, dt_s=0.05)
    assert not res.keep
    q_nan = q.copy()
    q_nan[0, 0] = np.nan
    res2 = filter_motion_clip(q_nan, joint_limits_rad=limits, dt_s=0.05)
    assert not res2.keep


def test_motion_lib_refuses_g1_dof() -> None:
    lib = {
        "fps": 30.0,
        "root_pos": np.zeros((8, 3)),
        "root_rot": np.tile([0, 0, 0, 1.0], (8, 1)),
        "dof_pos": np.zeros((8, 29)),
    }
    with pytest.raises(G1CheckpointIncompatible):
        validate_motion_lib(lib)
    lib25 = dict(lib)
    lib25["dof_pos"] = np.zeros((8, 25))
    info = validate_motion_lib(lib25)
    assert info["n_dof"] == 25
    with pytest.raises(MotionLibError):
        validate_motion_lib({"fps": 30.0})


def test_body_map_has_head_and_no_scale_copy() -> None:
    raw = load_body_map()
    assert raw["human_scale_status"] == "uncalibrated_identity"
    roles = {b["role"] for b in raw["smplx"]["bodies"]}
    assert "head" in roles
    assert "left_wrist" in roles
