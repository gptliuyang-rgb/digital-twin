"""Validate a GMR / SONIC motion_lib pickle *schema* without running GMR.

Expected keys follow GMR's saved PKL:
  fps, root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T, n_dof)

A 29-DoF G1 library is refused on T800 (ADR-011).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import G1_N_DOF, load_t800_sonic

REQUIRED_KEYS = ("fps", "root_pos", "root_rot", "dof_pos")


class MotionLibError(ValueError):
    """Malformed or G1-shaped motion library."""


def validate_motion_lib(lib: Mapping[str, Any], *, n_dof: int | None = None) -> dict[str, Any]:
    cfg = load_t800_sonic()
    n = int(n_dof if n_dof is not None else cfg["n_revolute"])
    missing = [k for k in REQUIRED_KEYS if k not in lib]
    if missing:
        raise MotionLibError(f"motion_lib missing keys {missing}")
    fps = float(lib["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise MotionLibError(f"fps must be a positive finite Hz, got {fps}")
    root_pos = np.asarray(lib["root_pos"], dtype=np.float64)
    root_rot = np.asarray(lib["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(lib["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise MotionLibError(f"root_pos expected (T,3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise MotionLibError(f"root_rot expected (T,4) xyzw, got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise MotionLibError(f"dof_pos expected (T, n_dof), got {dof_pos.shape}")
    t = root_pos.shape[0]
    if root_rot.shape[0] != t or dof_pos.shape[0] != t:
        raise MotionLibError("root_pos / root_rot / dof_pos time dimensions disagree")
    if dof_pos.shape[1] == G1_N_DOF:
        refuse_g1_checkpoint(checkpoint_meta={"robot": "t800", "n_dof": G1_N_DOF})
    if dof_pos.shape[1] != n:
        raise MotionLibError(
            f"dof_pos last dim {dof_pos.shape[1]} != T800 n_revolute {n}. "
            "G1 29-DoF libraries are unloadable (ADR-011)."
        )
    if not np.isfinite(dof_pos).all():
        raise MotionLibError("dof_pos contains NaN/Inf")
    return {
        "n_frames": t,
        "n_dof": n,
        "fps": fps,
        "duration_s": t / fps,
        "robot": cfg["robot"],
    }
