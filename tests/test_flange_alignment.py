"""Flange SE(3) composition and active-flange resolver (no MuJoCo required)."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.flange import (
    R_to_quat_wxyz,
    active_flange,
    cad_flange_complete,
    chain_yaml,
    mount_to_wrist_se3,
    quat_wxyz_to_R,
    rgb_axis_site_xml,
    se3,
    se3_inv,
    se3_mul,
    se3_to_pos_quat,
    wrist_end_to_palm_se3,
)


def test_identity_quat_is_I() -> None:
    np.testing.assert_allclose(quat_wxyz_to_R([1, 0, 0, 0]), np.eye(3), atol=1e-12)


def test_quat_roundtrip_90deg_about_z() -> None:
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    r = quat_wxyz_to_R(q)
    np.testing.assert_allclose(r[:, 0], [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(R_to_quat_wxyz(r), q, atol=1e-9)


def test_se3_inv_is_inverse() -> None:
    t = se3([0.003, 0.0, -0.0285], [1, 0, 0, 0])
    i = se3_mul(se3_inv(t), t)
    np.testing.assert_allclose(i, np.eye(4), atol=1e-12)


def test_active_flange_is_identity_while_cad_missing() -> None:
    flange = active_flange()
    assert flange.kind == "kinematic_bringup_identity"
    assert flange.policy_eval_forbidden is True
    assert flange.cad_ready is False
    assert cad_flange_complete() is False
    np.testing.assert_allclose(flange.se3(), np.eye(4), atol=1e-12)


def test_wrist_end_to_palm_is_flange_times_official_mount() -> None:
    expected = se3_mul(active_flange().se3(), mount_to_wrist_se3())
    got = wrist_end_to_palm_se3()
    np.testing.assert_allclose(got, expected, atol=1e-12)
    pos, quat = se3_to_pos_quat(got)
    # Identity flange ⇒ palm offset is the official mount→wrist translation.
    np.testing.assert_allclose(pos, [0.003, 0.00025016, -0.0285], atol=1e-8)
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-4)


def test_chain_yaml_names_robot_base() -> None:
    chain = chain_yaml()
    assert chain["base_body"] == "LINK_BASE"
    assert "T_base_wrist_end" in chain["formula"]
    assert chain["T_flange"]["kind"] == "kinematic_bringup_identity"
    assert chain["policy_eval_forbidden"] is True
    assert chain["t800_wrist_body"]["left"] == "LINK_WRIST_END_L"
    assert chain["hand_root"]["right"] == "r_wrist"


def test_cad_flange_wins_when_filled() -> None:
    raw = {
        "t800_wrist_to_hand_mount": {
            "pos_m": [0.01, 0.0, 0.0],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "kinematic_bringup_identity": {
            "enabled": True,
            "pos_m": [0.0, 0.0, 0.0],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "hand_mount_to_wrist": {
            "pos_m": [0.0, 0.0, 0.0],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "humanoid": {"left_wrist_link": "LINK_WRIST_END_L", "right_wrist_link": "LINK_WRIST_END_R"},
    }
    flange = active_flange(raw)
    assert flange.kind == "cad"
    assert flange.policy_eval_forbidden is False
    assert flange.cad_ready is True
    np.testing.assert_allclose(flange.pos_m, [0.01, 0.0, 0.0])


def test_rgb_axis_sites_are_visual_only() -> None:
    xml = "\n".join(rgb_axis_site_xml("l_wrist"))
    assert "frame_l_wrist_x" in xml
    assert "contype" not in xml
    assert "conaffinity" not in xml
    assert 'group="4"' in xml


def test_rgb_axis_sites_compile_in_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    sites = "\n".join(rgb_axis_site_xml("l_wrist"))
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody><body name='b'>{sites}</body></worldbody></mujoco>"
    )
    names = {model.site(i).name for i in range(model.nsite)}
    assert "frame_l_wrist_x" in names
    assert "frame_l_wrist_z" in names
