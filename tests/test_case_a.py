"""Explicit Case A FK conversion. PolicyClient must still refuse 50-D."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import CommandVector, command_dim, load_hand_spec
from vla.adapters.action_space import ActionSpaceMismatch, diagnose_action_vector
from vla.adapters.case_a import (
    CaseAConversionError,
    CaseAToCommandSchema,
    HeadNavCommand,
    case_a_dim,
    case_a_slices,
    split_case_a,
)
from vla.adapters.rotation import matrix_to_rot6d
from vla.adapters.upsample import upsample_command_chunk
from vla.client.pipeline import DeployPipeline
from vla.client.policy_client import PolicyClient


def _head_nav() -> HeadNavCommand:
    return HeadNavCommand(
        pelvis_height_m=0.55,
        nav_cmd_mps=np.zeros(3),
        loco_mode=0,
        tool_trigger=0,
        source="unit_test_fixture",
        head_pos_m=np.array([0.0, 0.0, 0.4]),
        head_rot6d=matrix_to_rot6d(np.eye(3)),
    )


def _fk_left(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    pos = np.array([0.25 + 0.01 * q[0], 0.15, 0.05], dtype=np.float64)
    return pos, np.eye(3)


def _fk_right(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    pos = np.array([0.25 + 0.01 * q[0], -0.15, 0.05], dtype=np.float64)
    return pos, np.eye(3)


def _adapter() -> CaseAToCommandSchema:
    return CaseAToCommandSchema.from_injected_fk(_fk_left, _fk_right)


def test_case_a_layout_is_50() -> None:
    spec = load_hand_spec()
    sl = case_a_slices(spec)
    assert case_a_dim(spec) == 50
    assert sl["_total"][1] == 50
    assert sl["left_arm_q"] == (0, 5)
    assert sl["right_arm_q"] == (5, 10)
    assert sl["left_hand_q"] == (10, 30)
    assert sl["right_hand_q"] == (30, 50)
    report = diagnose_action_vector(dim=50)
    assert "CaseAToCommandSchema.convert" in (report.adapter or "")


def test_convert_requires_apply_fk_kwarg() -> None:
    adapter = _adapter()
    vec = np.zeros(50)
    with pytest.raises(TypeError):
        adapter.convert(vec, head_nav=_head_nav())  # type: ignore[call-arg]
    with pytest.raises(CaseAConversionError, match="apply_fk=True"):
        adapter.convert(vec, apply_fk=False, head_nav=_head_nav())


def test_convert_maps_wrists_and_fingers() -> None:
    spec = load_hand_spec()
    adapter = _adapter()
    vec = np.zeros(50)
    vec[0] = 2.0  # left shoulder pitch → left wrist x shift
    vec[5] = -1.0  # right
    vec[10] = 0.05  # first left finger
    cmd = adapter.convert(vec, apply_fk=True, head_nav=_head_nav())
    assert cmd.to_flat_vector().shape == (command_dim(spec),)
    np.testing.assert_allclose(cmd.left_wrist_pos, [0.27, 0.15, 0.05], atol=1e-9)
    np.testing.assert_allclose(cmd.right_wrist_pos, [0.24, -0.15, 0.05], atol=1e-9)
    assert cmd.left_hand_q[0] == pytest.approx(0.05)
    assert cmd.pelvis_height == pytest.approx(0.55)
    parts = split_case_a(vec)
    np.testing.assert_allclose(parts["left_arm_q"][0], 2.0)


def test_world_frame_requires_pelvis_pose() -> None:
    adapter = _adapter()
    with pytest.raises(CaseAConversionError, match="pelvis_pos_world_m"):
        adapter.convert(np.zeros(50), apply_fk=True, head_nav=_head_nav(), fk_frame="world")


def test_world_frame_heading_strip() -> None:
    adapter = _adapter()
    cmd = adapter.convert(
        np.zeros(50),
        apply_fk=True,
        head_nav=_head_nav(),
        fk_frame="world",
        pelvis_pos_world_m=np.zeros(3),
        pelvis_yaw_rad=0.0,
    )
    np.testing.assert_allclose(cmd.left_wrist_pos, [0.25, 0.15, 0.05], atol=1e-9)


def test_refuses_75d_and_unknown() -> None:
    adapter = _adapter()
    spec = load_hand_spec()
    with pytest.raises(CaseAConversionError):
        adapter.convert(CommandVector.zeros(spec).to_flat_vector(), apply_fk=True, head_nav=_head_nav())


def test_head_nav_requires_source() -> None:
    with pytest.raises(CaseAConversionError, match="source"):
        HeadNavCommand(
            pelvis_height_m=0.55,
            nav_cmd_mps=np.zeros(3),
            loco_mode=0,
            tool_trigger=0,
            source="",
            head_pos_m=np.zeros(3),
            head_rot6d=matrix_to_rot6d(np.eye(3)),
        )


def test_policy_client_still_refuses_case_a() -> None:
    spec = load_hand_spec()

    def infer(_obs: dict) -> np.ndarray:
        return np.zeros(50)

    with pytest.raises(ActionSpaceMismatch, match="apply_fk=True"):
        PolicyClient(infer, spec).step({})
    with pytest.raises(ActionSpaceMismatch):
        DeployPipeline(lambda _o: np.zeros((2, 50)), spec).step_chunk({})


def test_upsample_refuses_case_a() -> None:
    with pytest.raises(ActionSpaceMismatch, match="apply_fk=True"):
        upsample_command_chunk(np.zeros((4, 50)), model="groot_n1_6")


def test_l1_harness_refuses_without_flag() -> None:
    from eval.l1_case_a import evaluate_case_a

    with pytest.raises(CaseAConversionError, match="apply_fk=True"):
        evaluate_case_a(np.zeros((2, 50)), apply_fk=False, cfg={"head_nav_fixture": {}})


def test_upsample_after_explicit_fk() -> None:
    spec = load_hand_spec()
    adapter = _adapter()
    chunk_a = np.zeros((3, 50))
    chunk_a[1, 0] = 1.0
    chunk_a[2, 0] = 2.0
    rows = adapter.convert_chunk(chunk_a, apply_fk=True, head_nav=_head_nav())
    assert rows.shape == (3, command_dim(spec))
    out = upsample_command_chunk(rows, model="groot_n1_6", pos_interp="linear")
    # 10 Hz → 50 Hz, factor 5, N_out = (3-1)*5+1 = 11
    assert out.shape == (11, command_dim(spec))
    layout_end = rows[-1]
    np.testing.assert_allclose(out[-1], layout_end, atol=1e-9)
