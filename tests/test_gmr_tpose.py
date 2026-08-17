"""GMR T-pose composition and T800/PM01 q=0 frame checks."""

from __future__ import annotations

import numpy as np
import pytest

from wbc.gmr.export import ik_config, load_body_map, load_tpose_offsets, params_overlay
from wbc.gmr.tpose import (
    PM01_IK,
    PM01_MJCF,
    T800_MJCF,
    compose_offset,
    geodesic_deg,
    quat_mul,
    quat_wxyz_to_matrix,
)

EYE3 = np.eye(3)


def test_compose_offset_identity_frames_keeps_pm01_quat() -> None:
    off = np.array([0.5, -0.5, -0.5, -0.5])
    got = compose_offset(off, EYE3, EYE3)
    np.testing.assert_allclose(np.abs(got), np.abs(off), atol=1e-9)


def test_compose_offset_applies_relative_body_rotation() -> None:
    # 90° about Z on T800 only.
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    r_t800 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    off = np.array([1.0, 0.0, 0.0, 0.0])
    got = compose_offset(off, EYE3, r_t800)
    r_got = quat_wxyz_to_matrix(got)
    np.testing.assert_allclose(r_got, r_t800, atol=1e-9)
    assert geodesic_deg(EYE3, r_got) == pytest.approx(90.0, abs=1e-6)


def test_quat_mul_matches_matrix_product() -> None:
    a = np.array([0.5, -0.5, -0.5, -0.5])
    b = np.array([0.70710678, 0.0, 0.0, 0.70710678])
    r = quat_wxyz_to_matrix(a) @ quat_wxyz_to_matrix(b)
    np.testing.assert_allclose(quat_wxyz_to_matrix(quat_mul(a, b)), r, atol=1e-7)


@pytest.mark.skipif(
    not T800_MJCF.is_file() or not PM01_MJCF.is_file() or not PM01_IK.is_file(),
    reason="T800/PM01 MJCF or GMR PM01 IK not cloned",
)
def test_tpose_pass_head_matches_pelvis_and_scale_not_one() -> None:
    from wbc.gmr.tpose import compute_tpose

    report = compute_tpose()
    assert report["head_vs_pelvis_deg"] < 1e-6
    assert report["max_t800_vs_pm01_frame_delta_deg"] < 1e-6
    head = report["smplx"]["bodies"]["LINK_HEAD_YAW"]["table1_quat_wxyz"]
    pelvis = report["smplx"]["bodies"]["LINK_BASE"]["table1_quat_wxyz"]
    np.testing.assert_allclose(np.abs(head), np.abs(pelvis), atol=1e-8)
    assert report["smplx"]["human_scale_table"]["pelvis"] != pytest.approx(1.0, abs=1e-6)
    assert report["smplx"]["human_scale_table"]["left_wrist"] != pytest.approx(1.0, abs=1e-6)
    assert report["lengths_m"]["t800_hip_to_foot"] > report["lengths_m"]["pm01_hip_to_foot"]


@pytest.mark.skipif(load_tpose_offsets() is None, reason="tpose_offsets.yaml not written")
def test_export_uses_tpose_head_and_scale() -> None:
    tpose = load_tpose_offsets()
    assert tpose is not None
    ik = ik_config("smplx", load_body_map(), tpose)
    head = np.array(ik["ik_match_table1"]["LINK_HEAD_YAW"][4])
    pelvis = np.array(ik["ik_match_table1"]["LINK_BASE"][4])
    np.testing.assert_allclose(np.abs(head), np.abs(pelvis), atol=1e-8)
    assert ik["human_scale_table"]["pelvis"] != pytest.approx(1.0, abs=1e-6)
    overlay = params_overlay(load_body_map(), tpose)
    assert overlay["quat_offset_status"] == "tpose_composed_pm01_with_t800_q0_frames"
    assert overlay["human_scale_status"] == "tpose_scaled_from_pm01_link_lengths"
