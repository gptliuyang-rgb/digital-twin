"""ADR-024: SONIC/GMR feet are MJCF LINK_FOOT_*, not the URDF sole."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import load_frames
from wbc.dims import assert_t800_config
from wbc.foot_frame import (
    FOOT_FRAME_DECISION,
    MJCF_COLLISION_BOX_Z_M,
    URDF_SOLE_IN_MJCF_FOOT_M,
    FootFrameError,
    assert_foot_frame,
    refuse_urdf_foot_as_sonic_body,
    urdf_sole_world_m,
)
from wbc.gmr.export import sonic_tracked_bodies
from wbc.pd_stand import pd_stand_q_des_rad


def test_foot_frame_contract() -> None:
    block = assert_foot_frame()
    assert block["decision"] == FOOT_FRAME_DECISION
    cfg = assert_t800_config()
    assert cfg["tracked_bodies"]["left_foot"] == "LINK_FOOT_L"
    assert cfg["tracked_bodies"]["right_foot"] == "LINK_FOOT_R"
    np.testing.assert_allclose(block["urdf_sole_in_mjcf_foot_m"], URDF_SOLE_IN_MJCF_FOOT_M)
    assert float(block["mjcf_collision_box_z_in_foot_m"]) == pytest.approx(MJCF_COLLISION_BOX_Z_M)
    frames = load_frames()["frames"]
    assert frames["left_foot"]["t800_link"] == "LINK_FOOT_L"
    assert frames["left_foot"]["source"] == FOOT_FRAME_DECISION


def test_gmr_tracks_mjcf_foot_not_urdf_sole() -> None:
    tracked = sonic_tracked_bodies()
    assert tracked["left_foot"] == "LINK_FOOT_L"
    assert tracked["right_foot"] == "LINK_FOOT_R"
    with pytest.raises(FootFrameError, match="64.53"):
        refuse_urdf_foot_as_sonic_body("J_FIXED_FOOT_L")
    with pytest.raises(FootFrameError):
        refuse_urdf_foot_as_sonic_body("urdf_LINK_FOOT_R")


def test_urdf_sole_offset_at_identity() -> None:
    pos = np.array([0.1, -0.2, 0.5])
    rot = np.eye(3)
    sole = urdf_sole_world_m(pos, rot)
    np.testing.assert_allclose(sole, pos + np.array(URDF_SOLE_IN_MJCF_FOOT_M))
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sole_yaw = urdf_sole_world_m(pos, yaw90)
    np.testing.assert_allclose(sole_yaw[2], pos[2] + URDF_SOLE_IN_MJCF_FOOT_M[2])


def test_pd_stand_q_des_is_official_25() -> None:
    q = pd_stand_q_des_rad()
    assert q.shape == (25,)
    np.testing.assert_allclose(q[0:6], [-0.105, 0.105, 0.263, 0.145, -0.051, -0.074])
    np.testing.assert_allclose(q[-2:], [0.0, 0.0])
