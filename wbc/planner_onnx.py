"""SONIC kinematic planner ONNX I/O contract. Numpy only — no simulator.

Official G1 ``planner_sonic.onnx`` (nvlabs planner_onnx.html) takes
``context_mujoco_qpos`` of shape ``[1, 4, 36]`` (3 pos + 4 quat wxyz + 29
hinges) and emits 30 Hz MuJoCo qpos that the C++ stack resamples to 50 Hz.
T800 Native SDK is 25 revolute: last dim **32**. Loading the G1 ONNX would
smash 36 vs 32.

This module packs / validates that contract. It does **not** run ONNX, does
**not** invent a clip library, and does **not** replace ADR-018 interpolators.
Hands still bypass WBC. ``nav_cmd[2]`` (wz) is Eq. 8 heading rate, not
``facing_direction``. ``command_schema`` loco_mode 2 is walk, not official run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from interface.schema import CommandVector
from vla.adapters.frame_transform import heading_rotation_z
from vla.adapters.rotation import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    slerp_matrices,
)
from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import (
    G1_N_DOF,
    G1_PLANNER_QPOS_DIM,
    PLANNER_ALLOWED_K,
    PLANNER_CONTEXT_FRAMES,
    PLANNER_CONTROL_HZ,
    PLANNER_LOOKAHEAD_STEPS_50HZ,
    PLANNER_MAX_OUT_FRAMES,
    PLANNER_MAX_TOKENS,
    PLANNER_MIN_TOKENS,
    PLANNER_N_INPUTS_V2,
    PLANNER_N_MODES_V2,
    PLANNER_N_OUTPUTS,
    PLANNER_NATIVE_HZ,
    PLANNER_TOKEN_FRAMES,
    load_t800_sonic,
    planner_qpos_dim,
    t800_planner_qpos_dim,
)
from wbc.motion_ref import MotionFrame

PLANNER_ONNX_YAML = Path(__file__).with_name("planner_onnx.yaml")
ZERO_DIR_EPS = 1e-5
HEIGHT_DISABLED = -1.0
TARGET_VEL_MODE_DEFAULT = -1.0
LOCO_TO_PLANNER_MODE = {0: 0, 1: 1, 2: 2}  # stand/slow_walk/fast_walk → idle/slowWalk/walk
HEIGHT_MODES = frozenset({4, 5, 6, 7})
FacingMode = Literal["travel", "current"]


class PlannerOnnxError(ValueError):
    """Wrong qpos width, empty context, G1 36-D, or invented height/mode."""


class PlannerOnnxBlocked(RuntimeError):
    """No trained T800 planner ONNX exists. G1 planner_sonic.onnx is refused."""


@dataclass(frozen=True)
class PlannerQpos:
    """One MuJoCo freejoint qpos row. Joints are T800 policy order (25)."""

    root_pos_m: np.ndarray
    root_rot_wxyz: np.ndarray
    q_rad: np.ndarray

    @property
    def n_dof(self) -> int:
        return int(self.q_rad.shape[0])


def load_planner_onnx_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or PLANNER_ONNX_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("planner_onnx.yaml must be a mapping")
    flags = (
        "not_dexhand2_contact",
        "not_table_s4",
        "not_hand_mit_ring",
        "not_invented_latency",
        "not_pico_sdk",
        "not_invented_clip",
        "not_g1_planner_onnx",
        "not_interpolator_substitute",
        "not_wrist_pose_augmentation_sampler",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise PlannerOnnxError(f"planner_onnx.yaml must keep {key}: true")
    n = int(raw["n_dof"])
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=n)
    expected = planner_qpos_dim(n)
    if int(raw["expected_qpos_dim"]) != expected:
        raise PlannerOnnxError(f"expected_qpos_dim {raw['expected_qpos_dim']} != {expected}")
    if int(raw["g1_qpos_dim_forbidden"]) != G1_PLANNER_QPOS_DIM:
        raise PlannerOnnxError("g1_qpos_dim_forbidden must stay 36")
    if int(raw["allowed_k"]) != PLANNER_ALLOWED_K:
        raise PlannerOnnxError(f"allowed_k must be {PLANNER_ALLOWED_K}")
    if int(raw["n_inputs_v2"]) != PLANNER_N_INPUTS_V2:
        raise PlannerOnnxError("n_inputs_v2 must be 11 (official V2)")
    if int(raw["n_outputs"]) != PLANNER_N_OUTPUTS:
        raise PlannerOnnxError("n_outputs must be 2")
    if int(raw["n_modes_v2"]) != PLANNER_N_MODES_V2:
        raise PlannerOnnxError("n_modes_v2 must be 27")
    if list(raw["loco_mode_map"].values()) != [0, 1, 2]:
        raise PlannerOnnxError("loco_mode_map must be stand→idle, slow_walk→slowWalk, fast_walk→walk")
    if 3 in {int(v) for v in raw["loco_mode_map"].values()}:
        raise PlannerOnnxError("fast_walk must not map to official run (3)")
    return raw


def refuse_g1_planner_onnx(path: str | Path | None = None) -> None:
    """Official planner_sonic.onnx is Unitree G1 36-D qpos. Do not load it on T800."""
    label = str(path) if path is not None else "planner_sonic.onnx"
    raise G1CheckpointIncompatible(
        f"{label} is a G1 kinematic planner (context_mujoco_qpos [1,4,36], "
        f"29-DoF hinges). T800 planner qpos is {t800_planner_qpos_dim()}-D "
        "(7 + 25). Retrain; do not load G1 planner_sonic.onnx. "
        "ADR-018 interpolators stay the runtime L1a until a T800 planner exists."
    )


def pack_qpos(frame: PlannerQpos | MotionFrame, *, n_dof: int | None = None) -> np.ndarray:
    """Pack one T800 MuJoCo qpos row: xyz + quat wxyz + 25 hinges."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    if isinstance(frame, MotionFrame):
        pos = np.asarray(frame.root_pos_m, dtype=np.float64).reshape(3)
        quat = np.asarray(frame.root_rot_wxyz, dtype=np.float64).reshape(4)
        q = np.asarray(frame.q_ref_rad, dtype=np.float64).reshape(-1)
    else:
        pos = np.asarray(frame.root_pos_m, dtype=np.float64).reshape(3)
        quat = np.asarray(frame.root_rot_wxyz, dtype=np.float64).reshape(4)
        q = np.asarray(frame.q_rad, dtype=np.float64).reshape(-1)
    if q.shape == (G1_N_DOF,):
        refuse_g1_checkpoint(n_dof=G1_N_DOF, planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    if q.shape != (n,):
        raise PlannerOnnxError(
            f"planner q dim {q.shape} != T800 {n} "
            "(hands bypass WBC; do not concatenate DexHand2 q)"
        )
    nrm = float(np.linalg.norm(quat))
    if nrm < 1e-8:
        raise PlannerOnnxError("planner root quaternion is zero")
    quat = quat / nrm
    if not np.isfinite(pos).all() or not np.isfinite(q).all():
        raise PlannerOnnxError("planner qpos contains NaN/Inf")
    return np.concatenate([pos, quat, q])


def unpack_qpos(row: np.ndarray, *, n_dof: int | None = None) -> PlannerQpos:
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    vec = np.asarray(row, dtype=np.float64).reshape(-1)
    if vec.shape == (G1_PLANNER_QPOS_DIM,):
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    expected = planner_qpos_dim(n)
    if vec.shape != (expected,):
        raise PlannerOnnxError(f"qpos row dim {vec.shape} != T800 {expected} (G1 is 36)")
    return PlannerQpos(
        root_pos_m=vec[:3].copy(),
        root_rot_wxyz=vec[3:7].copy(),
        q_rad=vec[7:].copy(),
    )


def pack_context(
    frames: list[PlannerQpos | MotionFrame],
    *,
    n_dof: int | None = None,
) -> np.ndarray:
    """Official context_mujoco_qpos: ``[1, 4, 32]`` on T800. Empty raises."""
    if len(frames) != PLANNER_CONTEXT_FRAMES:
        raise PlannerOnnxError(
            f"planner context needs {PLANNER_CONTEXT_FRAMES} frames, got {len(frames)}. "
            "Do not invent a standing pose."
        )
    rows = np.stack([pack_qpos(f, n_dof=n_dof) for f in frames], axis=0)
    if rows.shape[-1] == G1_PLANNER_QPOS_DIM:
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    return rows.reshape(1, PLANNER_CONTEXT_FRAMES, rows.shape[-1]).astype(np.float32)


def sample_context_from_50hz(
    frames: list[MotionFrame],
    *,
    current_frame: int = 0,
    look_ahead: int = PLANNER_LOOKAHEAD_STEPS_50HZ,
    n_dof: int | None = None,
) -> np.ndarray:
    """Official context construction: 4 frames at 30 Hz from a 50 Hz motion.

    Starts at ``current_frame + look_ahead`` (default look-ahead = 2 at 50 Hz).
    Does not invent frames past the supplied window.
    """
    if not frames:
        raise PlannerOnnxError("empty 50 Hz motion; do not invent a stand clip")
    start = int(current_frame) + int(look_ahead)
    step = PLANNER_CONTROL_HZ / PLANNER_NATIVE_HZ  # 50/30
    needed = start + step * (PLANNER_CONTEXT_FRAMES - 1)
    if needed > len(frames) - 1 + 1e-9:
        raise PlannerOnnxError(
            f"need 50 Hz frames through index {needed:.3f}, have {len(frames)}. "
            "Do not last-frame-repeat a planner context."
        )
    sampled: list[MotionFrame] = []
    for k in range(PLANNER_CONTEXT_FRAMES):
        t = start + step * k
        sampled.append(_interp_motion_frame(frames, t, n_dof=n_dof))
    return pack_context(sampled, n_dof=n_dof)


def _interp_motion_frame(
    frames: list[MotionFrame], t: float, *, n_dof: int | None
) -> MotionFrame:
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, len(frames) - 1)
    u = float(t - i0)
    a = frames[i0]
    b = frames[i1]
    q = (1.0 - u) * np.asarray(a.q_ref_rad, dtype=np.float64) + u * np.asarray(
        b.q_ref_rad, dtype=np.float64
    )
    dq = (1.0 - u) * np.asarray(a.dq_ref_rad_s, dtype=np.float64) + u * np.asarray(
        b.dq_ref_rad_s, dtype=np.float64
    )
    pos = (1.0 - u) * np.asarray(a.root_pos_m, dtype=np.float64) + u * np.asarray(
        b.root_pos_m, dtype=np.float64
    )
    r = slerp_matrices(
        quaternion_wxyz_to_matrix(a.root_rot_wxyz),
        quaternion_wxyz_to_matrix(b.root_rot_wxyz),
        u,
    )
    return MotionFrame(
        q_ref_rad=q,
        dq_ref_rad_s=dq,
        root_pos_m=pos,
        root_rot_wxyz=matrix_to_quaternion_wxyz(r),
    )


def loco_mode_to_planner_mode(loco_mode: int) -> int:
    """command_schema {0,1,2} → official {idle, slowWalk, walk}. Not run."""
    if loco_mode not in LOCO_TO_PLANNER_MODE:
        raise PlannerOnnxError(
            f"loco_mode {loco_mode} is not in command_schema {{0,1,2}}. "
            "Do not invent squat/kneel/boxing from the 75-D vector."
        )
    return LOCO_TO_PLANNER_MODE[int(loco_mode)]


def nav_to_planner_dirs(
    nav_cmd: np.ndarray,
    *,
    yaw_world_rad: float,
    facing: FacingMode = "travel",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Heading-frame ``(vx, vy, wz)`` → world movement/facing + target_vel.

    ``wz`` is unused here (ADR-038 Eq. 8). Official near-zero movement
    (``< 1e-5``) falls back to facing with a small scale.
    """
    nav = np.asarray(nav_cmd, dtype=np.float64).reshape(3)
    if not np.isfinite(nav).all():
        raise PlannerOnnxError("nav_cmd contains NaN/Inf")
    r_h = heading_rotation_z(float(yaw_world_rad))
    move_xy = r_h @ np.array([nav[0], nav[1], 0.0], dtype=np.float64)
    speed = float(np.hypot(nav[0], nav[1]))
    heading = np.array([np.cos(yaw_world_rad), np.sin(yaw_world_rad), 0.0], dtype=np.float64)
    if facing == "current":
        face = heading
    elif facing == "travel":
        face = move_xy.copy()
        if float(np.linalg.norm(face[:2])) < ZERO_DIR_EPS:
            face = heading
    else:
        raise PlannerOnnxError(f"facing {facing!r} must be 'travel' or 'current'")
    if float(np.linalg.norm(move_xy[:2])) < ZERO_DIR_EPS:
        move_xy = face * 1e-4
    target_vel = speed if speed > 0.0 else TARGET_VEL_MODE_DEFAULT
    return move_xy.astype(np.float64), face.astype(np.float64), target_vel


def pack_primary_inputs(
    context: np.ndarray,
    *,
    mode: int,
    movement_direction: np.ndarray,
    facing_direction: np.ndarray,
    target_vel: float = TARGET_VEL_MODE_DEFAULT,
    height_m: float = HEIGHT_DISABLED,
) -> dict[str, np.ndarray]:
    """Six official primary tensors. Context last dim must be 32, not 36."""
    ctx = np.asarray(context, dtype=np.float32)
    if ctx.shape == (1, PLANNER_CONTEXT_FRAMES, G1_PLANNER_QPOS_DIM):
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    expected = (1, PLANNER_CONTEXT_FRAMES, t800_planner_qpos_dim())
    if ctx.shape != expected:
        raise PlannerOnnxError(f"context shape {ctx.shape} != {expected}")
    mode_i = int(mode)
    if mode_i < 0 or mode_i >= PLANNER_N_MODES_V2:
        raise PlannerOnnxError(f"planner mode {mode_i} outside V2 0..{PLANNER_N_MODES_V2 - 1}")
    h = float(height_m)
    if mode_i in HEIGHT_MODES and h < 0.0:
        raise PlannerOnnxError(
            f"official mode {mode_i} requires height_m >= 0; do not invent a squat height"
        )
    if mode_i not in HEIGHT_MODES and h >= 0.0:
        raise PlannerOnnxError(
            "height control is only for squat/kneel/lying. "
            "command_schema pelvis_height must not fill this channel for walk/idle."
        )
    move = np.asarray(movement_direction, dtype=np.float32).reshape(1, 3)
    face = np.asarray(facing_direction, dtype=np.float32).reshape(1, 3)
    return {
        "context_mujoco_qpos": ctx,
        "target_vel": np.array([float(target_vel)], dtype=np.float32),
        "mode": np.array([mode_i], dtype=np.int64),
        "movement_direction": move,
        "facing_direction": face,
        "height": np.array([h], dtype=np.float32),
    }


def pack_advanced_inputs(
    *,
    random_seed: int = 1234,
    has_specific_target: int = 0,
    specific_target_positions: np.ndarray | None = None,
    specific_target_headings: np.ndarray | None = None,
    allowed_pred_num_tokens: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Five official advanced tensors. Waypoints required only when the flag is 1."""
    flag = int(has_specific_target)
    if flag not in (0, 1):
        raise PlannerOnnxError("has_specific_target must be 0 or 1")
    if flag == 1:
        if specific_target_positions is None or specific_target_headings is None:
            raise PlannerOnnxError("specific targets required when has_specific_target=1")
        pos = np.asarray(specific_target_positions, dtype=np.float32).reshape(1, 4, 3)
        hdg = np.asarray(specific_target_headings, dtype=np.float32).reshape(1, 4)
    else:
        if specific_target_positions is not None or specific_target_headings is not None:
            raise PlannerOnnxError("do not pass waypoints when has_specific_target=0")
        pos = np.zeros((1, 4, 3), dtype=np.float32)
        hdg = np.zeros((1, 4), dtype=np.float32)
    if allowed_pred_num_tokens is None:
        mask = np.ones((1, PLANNER_ALLOWED_K), dtype=np.int64)
    else:
        mask = np.asarray(allowed_pred_num_tokens, dtype=np.int64).reshape(1, PLANNER_ALLOWED_K)
        if not np.isin(mask, (0, 1)).all():
            raise PlannerOnnxError("allowed_pred_num_tokens must be a 0/1 mask")
    return {
        "random_seed": np.array([int(random_seed)], dtype=np.int64),
        "has_specific_target": np.array([[flag]], dtype=np.int64),
        "specific_target_positions": pos,
        "specific_target_headings": hdg,
        "allowed_pred_num_tokens": mask,
    }


def command_to_planner_inputs(
    cmd: CommandVector,
    context: np.ndarray,
    *,
    yaw_world_rad: float,
    facing: FacingMode = "travel",
    random_seed: int = 1234,
) -> dict[str, np.ndarray]:
    """Pack V2's 11 tensors from ``command_schema_v1`` + a T800 context.

    ``pelvis_height`` is ignored (walk/idle disable official height).
    ``nav_cmd[2]`` is ignored (Eq. 8). Hands are not packed.
    """
    mode = loco_mode_to_planner_mode(int(cmd.loco_mode))
    move, face, vel = nav_to_planner_dirs(cmd.nav_cmd, yaw_world_rad=yaw_world_rad, facing=facing)
    primary = pack_primary_inputs(
        context,
        mode=mode,
        movement_direction=move,
        facing_direction=face,
        target_vel=vel,
        height_m=HEIGHT_DISABLED,
    )
    advanced = pack_advanced_inputs(random_seed=random_seed, has_specific_target=0)
    packed = {**primary, **advanced}
    if len(packed) != PLANNER_N_INPUTS_V2:
        raise AssertionError(f"V2 input count {len(packed)} != {PLANNER_N_INPUTS_V2}")
    return packed


def resample_qpos_30_to_50(qpos_30: np.ndarray, *, n_dof: int | None = None) -> np.ndarray:
    """Official deployment resample: linear pos/q, SLERP quat, floor(N*50/30)."""
    arr = np.asarray(qpos_30, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise PlannerOnnxError("batched planner output is not supported (official batch=1)")
        arr = arr[0]
    if arr.ndim != 2:
        raise PlannerOnnxError(f"qpos must be (N, D), got {arr.shape}")
    if arr.shape[-1] == G1_PLANNER_QPOS_DIM:
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    expected = planner_qpos_dim(n)
    if arr.shape[-1] != expected:
        raise PlannerOnnxError(f"qpos last dim {arr.shape[-1]} != T800 {expected}")
    n_in = int(arr.shape[0])
    if n_in < 2:
        raise PlannerOnnxError("need at least two 30 Hz frames to resample")
    n_out = int(n_in * PLANNER_CONTROL_HZ / PLANNER_NATIVE_HZ)  # official: rounded down
    out = np.zeros((n_out, expected), dtype=np.float64)
    for i in range(n_out):
        t = i * PLANNER_NATIVE_HZ / PLANNER_CONTROL_HZ
        i0 = min(int(np.floor(t)), n_in - 2)
        u = float(t - i0)
        a = unpack_qpos(arr[i0], n_dof=n)
        b = unpack_qpos(arr[i0 + 1], n_dof=n)
        pos = (1.0 - u) * a.root_pos_m + u * b.root_pos_m
        q = (1.0 - u) * a.q_rad + u * b.q_rad
        r = slerp_matrices(
            quaternion_wxyz_to_matrix(a.root_rot_wxyz),
            quaternion_wxyz_to_matrix(b.root_rot_wxyz),
            u,
        )
        out[i] = pack_qpos(
            PlannerQpos(root_pos_m=pos, root_rot_wxyz=matrix_to_quaternion_wxyz(r), q_rad=q),
            n_dof=n,
        )
    return out


def truncate_planner_qpos(qpos: np.ndarray, num_pred_frames: int) -> np.ndarray:
    """Official: ``mujoco_qpos`` is padded; only the first ``num_pred_frames`` are valid."""
    arr = np.asarray(qpos)
    if arr.ndim == 3:
        arr = arr[0]
    n = int(num_pred_frames)
    if n < 0 or n > arr.shape[0]:
        raise PlannerOnnxError(f"num_pred_frames {n} outside 0..{arr.shape[0]}")
    if arr.shape[-1] == G1_PLANNER_QPOS_DIM:
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    if n % PLANNER_TOKEN_FRAMES != 0:
        raise PlannerOnnxError(
            f"num_pred_frames {n} must be tokens*{PLANNER_TOKEN_FRAMES} "
            f"(official min {PLANNER_MIN_TOKENS * PLANNER_TOKEN_FRAMES}, "
            f"max {PLANNER_MAX_TOKENS * PLANNER_TOKEN_FRAMES})"
        )
    tokens = n // PLANNER_TOKEN_FRAMES
    if tokens < PLANNER_MIN_TOKENS or tokens > PLANNER_MAX_TOKENS:
        raise PlannerOnnxError(f"num_pred_tokens {tokens} outside {PLANNER_MIN_TOKENS}..{PLANNER_MAX_TOKENS}")
    return arr[:n].copy()


def refuse_run_planner_onnx(path: str | Path | None = None) -> None:
    """Never execute a planner ONNX in this repo until a T800 export exists."""
    if path is not None:
        refuse_g1_planner_onnx(path)
    raise PlannerOnnxBlocked(
        "No T800 planner ONNX to run. Official planner_sonic.onnx is G1 36-D. "
        "ADR-018 interpolators + Eq. 8 remain the runtime L1a. "
        f"T800 context is [1,4,{t800_planner_qpos_dim()}]; max padded 30 Hz frames "
        f"are {PLANNER_MAX_OUT_FRAMES}."
    )
