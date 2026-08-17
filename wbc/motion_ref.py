"""SONIC encoder motion-reference window. Numpy only — no simulator.

Official GEAR-SONIC encoder observations (nvlabs observation_config docs):
``motion_joint_positions_10frame_step5``,
``motion_joint_velocities_10frame_step5``,
``motion_anchor_orientation_10frame_step5``,
``motion_root_z_position_10frame_step5``.

Those are a **future look-ahead** at 50 Hz with step 5 (0.1 s), 10 frames,
0.9 s horizon. Past the clip end, the last frame repeats. This module does
**not** invent a BONES-SEED clip, does **not** resample 30 fps → 50 Hz, and
does **not** load G1 ``model_encoder.onnx``.

T800 dims replace G1 29 with 25: 250+250+60+10 = 570 (G1 650 is refused).
Low-latency g1/teleop names are ``*_10frame_step1`` (no root_z): T800 560
(G1 640 is refused). SMPL/wrist ``*_4frame_step1`` is refused.
Wrist ``motion_*_wrists_*`` is G1 6-DoF; T800 dummy wrists have 0 DoF.
Hands still bypass WBC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla.adapters.frame_transform import heading_rotation_z, yaw_from_quat_wxyz
from vla.adapters.rotation import matrix_to_rot6d, quaternion_wxyz_to_matrix
from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import (
    G1_ENCODER_MOTION_DIM,
    G1_N_DOF,
    G1_N_WRIST_DOF,
    HISTORY_FRAMES,
    T800_N_LOWER_BODY_DOF,
    T800_N_WRIST_DOF,
    encoder_motion_dim,
    load_t800_sonic,
)
from wbc.gmr.motion_lib import validate_motion_lib
from wbc.stream import POLICY_HZ


class MotionRefError(ValueError):
    """Empty hold, wrong fps, G1 clip, wrist DoF, or invented resampling."""


@dataclass
class MotionFrame:
    """One 50 Hz reference sample. Joints are T800 policy order (25)."""

    q_ref_rad: np.ndarray
    dq_ref_rad_s: np.ndarray
    root_pos_m: np.ndarray
    root_rot_wxyz: np.ndarray


def _as_frame(frame: MotionFrame, n_dof: int) -> MotionFrame:
    q = np.asarray(frame.q_ref_rad, dtype=np.float64).reshape(-1)
    dq = np.asarray(frame.dq_ref_rad_s, dtype=np.float64).reshape(-1)
    pos = np.asarray(frame.root_pos_m, dtype=np.float64).reshape(3)
    quat = np.asarray(frame.root_rot_wxyz, dtype=np.float64).reshape(4)
    if q.shape == (G1_N_DOF,) or dq.shape == (G1_N_DOF,):
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if q.shape != (n_dof,) or dq.shape != (n_dof,):
        raise MotionRefError(
            f"motion frame q/dq dim {q.shape}/{dq.shape} != T800 {n_dof} "
            "(hands bypass WBC; do not concatenate DexHand2 q)"
        )
    if not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise MotionRefError("motion frame contains NaN/Inf")
    n = float(np.linalg.norm(quat))
    if n < 1e-8:
        raise MotionRefError("motion root quaternion is zero")
    quat = quat / n
    return MotionFrame(q_ref_rad=q.copy(), dq_ref_rad_s=dq.copy(), root_pos_m=pos.copy(), root_rot_wxyz=quat)


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([w, x, y, z], dtype=np.float64)


def look_ahead_indices(cursor: int, n_frames: int, step: int, n_clip: int) -> list[int]:
    """Oldest-first future indices. Clamp to last frame (official last-frame-repeat)."""
    if n_frames < 1 or step < 1:
        raise MotionRefError("n_frames and step must be >= 1")
    if n_clip < 1:
        raise MotionRefError("clip is empty; do not invent a stand pose")
    if cursor < 0:
        raise MotionRefError("cursor must be >= 0")
    last = n_clip - 1
    return [min(cursor + i * step, last) for i in range(n_frames)]


def heading_corrected_rel_rot6d(
    robot_quat_wxyz: np.ndarray,
    ref_quat_wxyz: np.ndarray,
    *,
    mode: str = "full",
    refheading_quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Relative rotation as Zhou 6D (first two columns of R).

    Official: heading-corrected relative rotation from the robot's current
    base orientation to the reference motion orientation.

    * ``full`` — ``R_robot.T @ R_ref``
    * ``heading`` — yaw-only robot heading ``R_z(yaw_robot).T @ R_ref``
    * ``refheading`` — yaw from the first future ref frame
    """
    r_ref = quaternion_wxyz_to_matrix(ref_quat_wxyz)
    if mode == "full":
        r_left = quaternion_wxyz_to_matrix(robot_quat_wxyz)
    elif mode == "heading":
        r_left = heading_rotation_z(yaw_from_quat_wxyz(robot_quat_wxyz))
    elif mode == "refheading":
        if refheading_quat_wxyz is None:
            raise MotionRefError("refheading mode needs the first future ref quaternion")
        r_left = heading_rotation_z(yaw_from_quat_wxyz(refheading_quat_wxyz))
    else:
        raise MotionRefError(f"unknown anchor mode {mode!r}")
    return matrix_to_rot6d(r_left.T @ r_ref)


class MotionHold:
    """Caller-supplied 50 Hz reference sequence. Not a clip generator.

    Push a window that already lives at the policy rate. Empty hold raises
    instead of inventing a stand clip or BONES-SEED.
    """

    def __init__(self, *, n_dof: int | None = None) -> None:
        cfg = load_t800_sonic()
        self.n_dof = int(n_dof if n_dof is not None else cfg["n_revolute"])
        if self.n_dof == G1_N_DOF:
            refuse_g1_checkpoint(n_dof=G1_N_DOF)
        self._frames: list[MotionFrame] = []
        self._cursor = 0

    def push_sequence(self, frames: list[MotionFrame], *, cursor: int = 0) -> None:
        if not frames:
            raise MotionRefError("empty motion sequence; do not invent a clip")
        self._frames = [_as_frame(f, self.n_dof) for f in frames]
        if cursor < 0 or cursor >= len(self._frames):
            raise MotionRefError(f"cursor {cursor} out of range 0..{len(self._frames) - 1}")
        self._cursor = int(cursor)

    def look_ahead(self, n_frames: int, step: int) -> list[MotionFrame]:
        if not self._frames:
            raise MotionRefError(
                "motion hold is empty. Push a 50 Hz reference (planner / ZMQ / "
                "validated motion_lib). Do not invent BONES-SEED or a stand pose."
            )
        idxs = look_ahead_indices(self._cursor, n_frames, step, len(self._frames))
        return [self._frames[i] for i in idxs]


class MotionCursor:
    """Integer-frame playback over a caller-supplied 50 Hz motion_lib.

    ``fps`` must already be 50. Resampling 30 fps clips would invent frames.
    G1 29-DoF libraries are refused. Not BONES-SEED.
    """

    def __init__(self, lib: dict, *, n_dof: int | None = None) -> None:
        meta = validate_motion_lib(lib, n_dof=n_dof)
        fps = float(lib["fps"])
        if abs(fps - POLICY_HZ) > 1e-9:
            raise MotionRefError(
                f"motion_lib fps {fps} != {POLICY_HZ}. Do not resample here; "
                "the caller must supply a 50 Hz clip (official last-frame-repeat "
                "is tick-index, not time interpolation)."
            )
        self.n_dof = int(meta["n_dof"])
        self.n_frames = int(meta["n_frames"])
        dof = np.asarray(lib["dof_pos"], dtype=np.float64)
        root_pos = np.asarray(lib["root_pos"], dtype=np.float64)
        root_rot = np.asarray(lib["root_rot"], dtype=np.float64)
        dq = lib.get("dof_vel")
        if dq is None:
            vel = np.zeros_like(dof)
        else:
            vel = np.asarray(dq, dtype=np.float64)
            if vel.shape != dof.shape:
                raise MotionRefError(f"dof_vel shape {vel.shape} != dof_pos {dof.shape}")
        frames: list[MotionFrame] = []
        for i in range(self.n_frames):
            frames.append(
                MotionFrame(
                    q_ref_rad=dof[i],
                    dq_ref_rad_s=vel[i],
                    root_pos_m=root_pos[i],
                    root_rot_wxyz=xyzw_to_wxyz(root_rot[i]),
                )
            )
        self.hold = MotionHold(n_dof=self.n_dof)
        self.hold.push_sequence(frames, cursor=0)

    def set_cursor(self, frame_index: int) -> None:
        self.hold.push_sequence(self.hold._frames, cursor=frame_index)

    def look_ahead(self, n_frames: int, step: int) -> list[MotionFrame]:
        return self.hold.look_ahead(n_frames, step)


def refuse_g1_encoder_onnx(path: str | Path | None = None) -> None:
    """Official model_encoder.onnx is Unitree G1. Do not load it on T800."""
    label = str(path) if path is not None else "model_encoder.onnx"
    lowered = label.lower().replace("\\", "/")
    if "low_latency" in lowered:
        raise G1CheckpointIncompatible(
            f"{label} is a G1 low-latency encoder (ONNX input 1247-D). "
            "T800 low-latency encoder input is 831-D (10frame_step1, no "
            "SMPL/wrists/root_z). Retrain; do not load G1 ONNX."
        )
    raise G1CheckpointIncompatible(
        f"{label} is a G1 encoder (ONNX input 1751-D, motion window 650-D). "
        "T800 encoder input is 842-D (motion window 570-D). Retrain; do not load G1 ONNX."
    )


def refuse_wrist_observation(name: str) -> None:
    lowered = name.lower()
    if "wrist" in lowered:
        raise MotionRefError(
            f"observation {name!r} is a G1 wrist-DoF channel "
            f"({G1_N_WRIST_DOF} joints). T800 dummy LINK_WRIST_END_* has "
            f"{T800_N_WRIST_DOF} wrist joints (ADR-001). Do not pad zeros."
        )


def refuse_smpl_observation(name: str) -> None:
    if name.lower().startswith("smpl"):
        raise MotionRefError(
            f"observation {name!r} needs a clip with smpl_joint.csv / smpl_pose.csv. "
            "Do not invent SMPL joints on T800."
        )


def lower_body_slice(q_rad: np.ndarray, n_lower: int = T800_N_LOWER_BODY_DOF) -> np.ndarray:
    q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
    if q.shape[0] < n_lower:
        raise MotionRefError(f"q dim {q.shape} shorter than lower-body {n_lower}")
    return q[:n_lower].copy()


def default_encoder_dim(n_dof: int | None = None) -> int:
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    dim = encoder_motion_dim(n, n_frames=HISTORY_FRAMES)
    if dim == G1_ENCODER_MOTION_DIM:
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    return dim


def identity_rot6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
