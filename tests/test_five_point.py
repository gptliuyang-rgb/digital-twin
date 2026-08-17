"""5-point elbow teleop. command_schema_v1 stays 75-D."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import (
    CommandVector,
    command_dim,
    five_point_command_dim,
    load_five_point_schema,
    load_frames,
)
from runtime.wbc_safety import WbcSafetyFilter
from wbc.dims import assert_t800_config, hybrid_encoder_cmd_dim
from wbc.teleop import (
    TELEOP_3POINT,
    TELEOP_5POINT,
    FivePointCommand,
    TeleopModeIncompatible,
    command_to_vr_3point,
    command_to_vr_5point,
    refuse_teleop_mode_mismatch,
    vr_5point_to_fields,
)


def test_v1_command_dim_unchanged() -> None:
    spec_dim = command_dim()
    assert spec_dim == 75
    assert five_point_command_dim() == 81
    schema = load_five_point_schema()
    assert schema["extra_dim"] == 6
    assert schema["vr_5point_pos_dim"] == 15
    assert schema["vr_5point_orn_dim"] == 12


def test_hybrid_encoder_dims() -> None:
    cfg = assert_t800_config()
    assert hybrid_encoder_cmd_dim("vr_3point") == 21
    assert hybrid_encoder_cmd_dim("vr_5point") == 27
    assert cfg["optional_elbow_bodies"]["left_elbow"] == "LINK_ELBOW_PITCH_L"
    assert cfg["optional_elbow_bodies"]["right_elbow"] == "LINK_ELBOW_PITCH_R"
    frames = load_frames()["frames"]
    assert frames["left_elbow"]["t800_link"] == "LINK_ELBOW_PITCH_L"
    assert frames["left_elbow"]["t800_link"] != "LINK_ELBOW_YAW_L"


def test_vr_5point_appends_elbows_after_3point() -> None:
    cmd = CommandVector.zeros()
    cmd.head_pos[:] = [1.0, 0.0, 0.0]
    cmd.left_wrist_pos[:] = [0.0, 1.0, 0.0]
    cmd.right_wrist_pos[:] = [0.0, 0.0, 1.0]
    fp = FivePointCommand(cmd, left_elbow_pos=[0.1, 0.2, 0.3], right_elbow_pos=[0.4, 0.5, 0.6])
    pos3, quat3 = command_to_vr_3point(cmd)
    pos5, quat5 = command_to_vr_5point(fp)
    np.testing.assert_allclose(pos5[:9], pos3)
    np.testing.assert_allclose(pos5[9:12], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pos5[12:15], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(quat5, quat3)
    back = vr_5point_to_fields(pos5, quat5)
    np.testing.assert_allclose(back["head_pos"], cmd.head_pos)
    np.testing.assert_allclose(back["left_elbow_pos"], fp.left_elbow_pos)


def test_refuse_3point_ckpt_with_5point_runtime() -> None:
    with pytest.raises(TeleopModeIncompatible, match="retrain"):
        refuse_teleop_mode_mismatch(TELEOP_3POINT, TELEOP_5POINT)
    refuse_teleop_mode_mismatch("vr_3point", "3point")
    refuse_teleop_mode_mismatch("5point", TELEOP_5POINT)


def test_wbc_safety_rejects_elbow_jump() -> None:
    ident = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    filt = WbcSafetyFilter()
    kw = dict(
        head_pos=np.zeros(3),
        head_rot6d=ident,
        left_wrist_pos=np.zeros(3),
        left_wrist_rot6d=ident,
        right_wrist_pos=np.zeros(3),
        right_wrist_rot6d=ident,
        pelvis_height=0.55,
        nav_cmd=np.zeros(3),
        dt_s=0.02,
        left_elbow_pos=np.zeros(3),
        right_elbow_pos=np.zeros(3),
    )
    assert filt.filter(**kw).accepted
    kw["left_elbow_pos"] = np.array([0.2, 0.0, 0.0])
    held = filt.filter(**kw)
    assert not held.accepted
    assert held.reason == "jump_left_elbow_pos"
