"""3-point and 5-point teleop packing. SONIC order ≠ command_schema_v1 order.

SONIC `vr_3point_local_target` is [left_wrist xyz, right_wrist xyz, head xyz] (9).
SONIC `vr_3point_local_orn_target` is 3× quaternion wxyz (12).
command_schema_v1 stores head, left, right as rot6d.

5-point (ADR-017) concatenates left_elbow xyz, right_elbow xyz after the 3-point
positions (15). Elbows are position-only; orientation stays the 3-point quats.

This module is the only place that remaps the two. Hands stay in CommandVector
and never enter the WBC token.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interface.schema import CommandVector, load_five_point_schema
from vla.adapters.rotation import (
    matrix_to_quaternion_wxyz,
    matrix_to_rot6d,
    quaternion_wxyz_to_matrix,
    rot6d_to_matrix,
)
from wbc.dims import load_t800_sonic

TELEOP_3POINT = "vr_3point"
TELEOP_5POINT = "vr_5point"


class TeleopModeIncompatible(RuntimeError):
    """Raised when a 3-point SONIC checkpoint is fed 5-point commands (or vice versa)."""


def refuse_teleop_mode_mismatch(checkpoint_mode: str, runtime_mode: str) -> None:
    ckpt = checkpoint_mode.replace("-", "_").lower()
    run = runtime_mode.replace("-", "_").lower()
    if ckpt in ("3point", "vr_3point"):
        ckpt = TELEOP_3POINT
    if ckpt in ("5point", "vr_5point"):
        ckpt = TELEOP_5POINT
    if run in ("3point", "vr_3point"):
        run = TELEOP_3POINT
    if run in ("5point", "vr_5point"):
        run = TELEOP_5POINT
    if ckpt != run:
        raise TeleopModeIncompatible(
            f"checkpoint teleop_mode={checkpoint_mode!r} ≠ runtime {runtime_mode!r}. "
            "A 5-point hybrid encoder is a retrain, not a concatenate-at-deploy hack."
        )


@dataclass
class FivePointCommand:
    """command_schema_v1 plus two elbow positions (heading frame, metres)."""

    cmd: CommandVector
    left_elbow_pos: np.ndarray
    right_elbow_pos: np.ndarray

    def __post_init__(self) -> None:
        self.left_elbow_pos = np.asarray(self.left_elbow_pos, dtype=np.float64).reshape(3)
        self.right_elbow_pos = np.asarray(self.right_elbow_pos, dtype=np.float64).reshape(3)
        if not np.isfinite(self.left_elbow_pos).all() or not np.isfinite(self.right_elbow_pos).all():
            raise ValueError("elbow positions contain NaN/Inf")

    def extra_vector(self) -> np.ndarray:
        return np.concatenate([self.left_elbow_pos, self.right_elbow_pos])


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


def command_to_vr_5point(fp: FivePointCommand) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos15, quat12). Elbows follow wrists+head; orn is still 3-point."""
    pos3, quat = command_to_vr_3point(fp.cmd)
    pos = np.concatenate([pos3, fp.left_elbow_pos, fp.right_elbow_pos])
    schema = load_five_point_schema()
    cfg = load_t800_sonic()
    if list(schema["teleop_5point_order"]) != list(cfg["teleop_5point_order"]):
        raise ValueError("5-point order drifted between schema and t800_sonic.yaml")
    if pos.shape != (int(schema["vr_5point_pos_dim"]),) or quat.shape != (int(schema["vr_5point_orn_dim"]),):
        raise ValueError(f"vr_5point shapes {pos.shape} {quat.shape}")
    return pos, quat


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


def vr_5point_to_fields(pos15: np.ndarray, quat12: np.ndarray) -> dict[str, np.ndarray]:
    p = np.asarray(pos15, dtype=np.float64).reshape(15)
    out = vr_3point_to_wbc_fields(p[:9], quat12)
    out["left_elbow_pos"] = p[9:12]
    out["right_elbow_pos"] = p[12:15]
    return out
