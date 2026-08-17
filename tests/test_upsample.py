"""50 Hz upsample of command_schema_v1 chunks. SLERP on rotations, no Case A."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import CommandVector, command_layout
from vla.adapters.rotation import matrix_to_rot6d, rot6d_to_matrix, rpy_to_matrix
from vla.adapters.upsample import upsample_command_chunk


def test_groot_factor_length() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.2, 0.0, 0.0]
    chunk = np.stack([a.to_flat_vector(), b.to_flat_vector()])
    out = upsample_command_chunk(chunk, model="groot_n1_6", pos_interp="linear")
    assert out.shape[0] == 6  # (2-1)*5+1
    layout = command_layout()
    lo, _ = layout["left_wrist_pos"]
    np.testing.assert_allclose(out[0, lo : lo + 3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out[-1, lo : lo + 3], [0.2, 0.0, 0.0], atol=1e-9)


def test_pi05_factor_length() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    chunk = np.stack([a.to_flat_vector(), b.to_flat_vector()])
    out = upsample_command_chunk(chunk, model="pi05", pos_interp="linear")
    assert out.shape[0] == 11  # (2-1)*10+1


def test_rotation_slerp_no_flip() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.head_rot6d[:] = matrix_to_rot6d(rpy_to_matrix(0.0, 0.0, 0.4))
    chunk = np.stack([a.to_flat_vector(), b.to_flat_vector()])
    out = upsample_command_chunk(chunk, model="groot_n1_6", pos_interp="linear")
    layout = command_layout()
    lo, hi = layout["head_rot6d"]
    mats = [rot6d_to_matrix(row) for row in out[:, lo:hi]]
    for r0, r1 in zip(mats, mats[1:], strict=False):
        c = np.clip((float(np.trace(r0.T @ r1)) - 1.0) / 2.0, -1.0, 1.0)
        assert np.arccos(c) < 0.15


def test_enum_nearest_not_averaged() -> None:
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.tool_trigger = 1
    chunk = np.stack([a.to_flat_vector(), b.to_flat_vector()])
    out = upsample_command_chunk(chunk, model="groot_n1_6", pos_interp="linear")
    layout = command_layout()
    trig = layout["tool_trigger"][0]
    assert set(np.unique(out[:, trig]).tolist()) <= {0.0, 1.0}


def test_single_step_refused() -> None:
    with pytest.raises(ValueError, match="≥2"):
        upsample_command_chunk(CommandVector.zeros().to_flat_vector().reshape(1, -1), model="groot_n1_6")
