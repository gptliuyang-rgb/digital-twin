"""SONIC planner ONNX contract: T800 32-D qpos, refuse G1 36-D / planner_sonic.onnx."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from interface.schema import CommandVector
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    G1_PLANNER_QPOS_DIM,
    PLANNER_ALLOWED_K,
    PLANNER_CONTEXT_FRAMES,
    assert_t800_config,
    planner_qpos_dim,
    t800_planner_qpos_dim,
)
from wbc.export_onnx import expected_io
from wbc.motion_ref import MotionFrame
from wbc.planner_onnx import (
    HEIGHT_DISABLED,
    PLANNER_ONNX_YAML,
    PlannerOnnxBlocked,
    PlannerOnnxError,
    PlannerQpos,
    command_to_planner_inputs,
    load_planner_onnx_cfg,
    loco_mode_to_planner_mode,
    nav_to_planner_dirs,
    pack_advanced_inputs,
    pack_context,
    pack_primary_inputs,
    pack_qpos,
    refuse_g1_planner_onnx,
    refuse_run_planner_onnx,
    resample_qpos_30_to_50,
    sample_context_from_50hz,
    truncate_planner_qpos,
    unpack_qpos,
)


def _qpos(*, q0: float = 0.0, x_m: float = 0.0, yaw_rad: float = 0.0) -> PlannerQpos:
    q = np.zeros(25)
    q[0] = q0
    half = yaw_rad / 2.0
    return PlannerQpos(
        root_pos_m=np.array([x_m, 0.0, 1.03]),
        root_rot_wxyz=np.array([np.cos(half), 0.0, 0.0, np.sin(half)]),
        q_rad=q,
    )


def _frame(*, q0: float = 0.0, x_m: float = 0.0) -> MotionFrame:
    q = np.zeros(25)
    q[0] = q0
    return MotionFrame(
        q_ref_rad=q,
        dq_ref_rad_s=np.zeros(25),
        root_pos_m=np.array([x_m, 0.0, 1.03]),
        root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_yaml_locks_t800_32_not_g1_36() -> None:
    cfg = load_planner_onnx_cfg()
    assert cfg["adr"] == "ADR-046"
    assert cfg["expected_qpos_dim"] == 32
    assert cfg["g1_qpos_dim_forbidden"] == 36
    assert cfg["context_shape"] == [1, 4, 32]
    assert cfg["not_g1_planner_onnx"] is True
    assert cfg["not_interpolator_substitute"] is True
    assert t800_planner_qpos_dim() == 32
    assert planner_qpos_dim(25) == 32
    with pytest.raises(ValueError, match="29"):
        planner_qpos_dim(29)
    sonic = assert_t800_config()
    assert sonic["planner_qpos_dim"] == 32
    assert sonic["g1_planner_qpos_dim"] == 36


def test_pack_qpos_refuses_g1_29() -> None:
    row = pack_qpos(_qpos(q0=0.2))
    assert row.shape == (32,)
    np.testing.assert_allclose(row[7], 0.2)
    got = unpack_qpos(row)
    np.testing.assert_allclose(got.q_rad[0], 0.2)
    with pytest.raises(G1CheckpointIncompatible):
        unpack_qpos(np.zeros(G1_PLANNER_QPOS_DIM))
    g1 = PlannerQpos(
        root_pos_m=np.zeros(3),
        root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        q_rad=np.zeros(29),
    )
    with pytest.raises(G1CheckpointIncompatible):
        pack_qpos(g1)


def test_pack_context_four_frames_and_empty_raises() -> None:
    ctx = pack_context([_qpos(q0=float(i)) for i in range(4)])
    assert ctx.shape == (1, PLANNER_CONTEXT_FRAMES, 32)
    assert ctx.dtype == np.float32
    with pytest.raises(PlannerOnnxError, match="invent"):
        pack_context([_qpos()])
    with pytest.raises(G1CheckpointIncompatible):
        pack_primary_inputs(
            np.zeros((1, 4, 36), dtype=np.float32),
            mode=0,
            movement_direction=np.array([1.0, 0.0, 0.0]),
            facing_direction=np.array([1.0, 0.0, 0.0]),
        )


def test_loco_mode_map_is_not_run() -> None:
    assert loco_mode_to_planner_mode(0) == 0
    assert loco_mode_to_planner_mode(1) == 1
    assert loco_mode_to_planner_mode(2) == 2
    with pytest.raises(PlannerOnnxError, match="command_schema"):
        loco_mode_to_planner_mode(3)


def test_nav_wz_is_not_facing() -> None:
    move, face, vel = nav_to_planner_dirs(
        np.array([0.6, 0.0, 1.2]),
        yaw_world_rad=0.0,
        facing="travel",
    )
    np.testing.assert_allclose(move[:2], [0.6, 0.0], atol=1e-9)
    np.testing.assert_allclose(face[:2], [0.6, 0.0], atol=1e-9)
    assert vel == pytest.approx(0.6)
    move_c, face_c, _ = nav_to_planner_dirs(
        np.array([0.0, 0.4, 0.0]),
        yaw_world_rad=0.0,
        facing="current",
    )
    np.testing.assert_allclose(move_c[:2], [0.0, 0.4], atol=1e-9)
    np.testing.assert_allclose(face_c[:2], [1.0, 0.0], atol=1e-9)
    _, _, vel0 = nav_to_planner_dirs(np.zeros(3), yaw_world_rad=0.0)
    assert vel0 == pytest.approx(-1.0)


def test_command_to_planner_inputs_walk() -> None:
    cmd = CommandVector.zeros()
    cmd.loco_mode = 2
    cmd.nav_cmd[:] = [0.35, 0.0, 0.25]
    cmd.pelvis_height = 0.55
    ctx = pack_context([_qpos() for _ in range(4)])
    packed = command_to_planner_inputs(cmd, ctx, yaw_world_rad=0.0)
    assert set(packed) == {
        "context_mujoco_qpos",
        "target_vel",
        "mode",
        "movement_direction",
        "facing_direction",
        "height",
        "random_seed",
        "has_specific_target",
        "specific_target_positions",
        "specific_target_headings",
        "allowed_pred_num_tokens",
    }
    assert packed["mode"][0] == 2
    assert packed["height"][0] == pytest.approx(HEIGHT_DISABLED)
    assert packed["target_vel"][0] == pytest.approx(0.35)
    assert packed["allowed_pred_num_tokens"].shape == (1, PLANNER_ALLOWED_K)
    assert packed["context_mujoco_qpos"].shape == (1, 4, 32)


def test_height_refused_on_walk_and_required_on_squat() -> None:
    ctx = pack_context([_qpos() for _ in range(4)])
    with pytest.raises(PlannerOnnxError, match="pelvis_height"):
        pack_primary_inputs(
            ctx,
            mode=2,
            movement_direction=np.array([1.0, 0.0, 0.0]),
            facing_direction=np.array([1.0, 0.0, 0.0]),
            height_m=0.55,
        )
    with pytest.raises(PlannerOnnxError, match="height_m"):
        pack_primary_inputs(
            ctx,
            mode=4,
            movement_direction=np.array([1.0, 0.0, 0.0]),
            facing_direction=np.array([1.0, 0.0, 0.0]),
            height_m=HEIGHT_DISABLED,
        )


def test_waypoints_refused_unless_flag() -> None:
    with pytest.raises(PlannerOnnxError, match="waypoints"):
        pack_advanced_inputs(
            has_specific_target=0,
            specific_target_positions=np.zeros((1, 4, 3)),
        )
    with pytest.raises(PlannerOnnxError, match="required"):
        pack_advanced_inputs(has_specific_target=1)


def test_resample_30_to_50_and_g1_refused() -> None:
    rows = np.stack([pack_qpos(_qpos(q0=0.0, x_m=0.0)), pack_qpos(_qpos(q0=0.3, x_m=0.3))], axis=0)
    out = resample_qpos_30_to_50(rows)
    assert out.shape[0] == int(2 * 50 / 30)  # floor = 3
    assert out.shape[1] == 32
    with pytest.raises(G1CheckpointIncompatible):
        resample_qpos_30_to_50(np.zeros((4, 36)))


def test_truncate_requires_token_multiple() -> None:
    rows = np.zeros((64, 32))
    got = truncate_planner_qpos(rows, 24)
    assert got.shape == (24, 32)
    with pytest.raises(PlannerOnnxError, match="tokens"):
        truncate_planner_qpos(rows, 25)
    with pytest.raises(G1CheckpointIncompatible):
        truncate_planner_qpos(np.zeros((64, 36)), 24)


def test_sample_context_from_50hz_needs_enough_frames() -> None:
    frames = [_frame(q0=float(i), x_m=0.01 * i) for i in range(8)]
    ctx = sample_context_from_50hz(frames, current_frame=0, look_ahead=2)
    assert ctx.shape == (1, 4, 32)
    with pytest.raises(PlannerOnnxError, match="invent"):
        sample_context_from_50hz([])
    with pytest.raises(PlannerOnnxError, match="through index"):
        sample_context_from_50hz(frames[:3], current_frame=0, look_ahead=2)


def test_refuse_g1_planner_onnx_and_run() -> None:
    with pytest.raises(G1CheckpointIncompatible, match="36"):
        refuse_g1_planner_onnx("planner/target_vel/V2/planner_sonic.onnx")
    with pytest.raises(G1CheckpointIncompatible):
        refuse_run_planner_onnx("planner_sonic.onnx")
    with pytest.raises(PlannerOnnxBlocked, match="32"):
        refuse_run_planner_onnx()


def test_export_io_includes_planner() -> None:
    io = expected_io()
    planner = io["planner"]["planner_sonic.onnx"]
    assert planner["qpos_dim"] == 32
    assert planner["g1_qpos_dim_forbidden"] == 36
    assert planner["context_shape"] == [1, 4, 32]


def test_yaml_flag_required(tmp_path) -> None:
    raw = yaml.safe_load(PLANNER_ONNX_YAML.read_text(encoding="utf-8"))
    raw["not_g1_planner_onnx"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PlannerOnnxError, match="not_g1_planner_onnx"):
        load_planner_onnx_cfg(path)
