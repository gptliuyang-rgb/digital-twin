"""3-point teleop packing. SONIC order ≠ command_schema_v1 order.

SONIC `vr_3point_local_target` is [left_wrist xyz, right_wrist xyz, head xyz] (9).
SONIC `vr_3point_local_orn_target` is 3× quaternion wxyz (12).
command_schema_v1 stores head, left, right as rot6d.

This module is the only place that remaps the two. Hands stay in CommandVector
and never enter the WBC token.
"""

from __future__ import annotations

import numpy as np

from interface.schema import CommandVector
from vla.adapters.rotation import (
    matrix_to_quaternion_wxyz,
    matrix_to_rot6d,
    quaternion_wxyz_to_matrix,
    rot6d_to_matrix,
)
from wbc.dims import load_t800_sonic


def command_to_vr_3point(cmd: CommandVector) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos9, quat12) in SONIC left/right/head order, heading frame."""
    pos = np.concatenate([cmd.left_wrist_pos, cmd.right_wrist_pos, cmd.head_pos])
    quats = np.concatenate(
        [
            matrix_to_quaternion_wxyz(rot6d_to_matrix(cmd.left_wrist_rot6d)),
            matrix_to_quaternion_wxyz(rot6d_to_matrix(cmd.right_wrist_rot6d)),
            matrix_to_quaternion_wxyz(rot6d_to_matrix(cmd.head_rot6d)),
        ]
    )
    order = load_t800_sonic()["teleop_3point_order"]
    if order != ["left_wrist", "right_wrist", "head"]:
        raise ValueError(f"unexpected teleop_3point_order {order}")
    if pos.shape != (9,) or quats.shape != (12,):
        raise ValueError(f"vr_3point shapes {pos.shape} {quats.shape}")
    return pos, quats


def vr_3point_to_wbc_fields(pos9: np.ndarray, quat12: np.ndarray) -> dict[str, np.ndarray]:
    """Inverse map: SONIC 3-point → command_schema field dict (no hands)."""
    p = np.asarray(pos9, dtype=np.float64).reshape(9)
    q = np.asarray(quat12, dtype=np.float64).reshape(12)

    def rot6(i: int) -> np.ndarray:
        return matrix_to_rot6d(quaternion_wxyz_to_matrix(q[4 * i : 4 * i + 4]))

    return {
        "left_wrist_pos": p[0:3],
        "right_wrist_pos": p[3:6],
        "head_pos": p[6:9],
        "left_wrist_rot6d": rot6(0),
        "right_wrist_rot6d": rot6(1),
        "head_rot6d": rot6(2),
    }
