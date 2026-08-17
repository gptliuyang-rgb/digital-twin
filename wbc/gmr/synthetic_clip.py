"""Tiny T800 motion_lib for pipeline tests. Not BONES-SEED and not a training set.

A 29-DoF G1-shaped clip is refused. Running GMR on BONES-SEED remains blocked
on flange SE(3) and wrist CoM (see wbc.retarget.retarget_blockers).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wbc.dims import load_t800_sonic
from wbc.filter import filter_motion_clip
from wbc.gmr.motion_lib import validate_motion_lib


def synthetic_stand_clip(
    *,
    n_frames: int = 60,
    fps: float = 30.0,
    n_dof: int | None = None,
    amplitude_rad: float = 0.02,
) -> dict[str, Any]:
    """Standing clip: LINK_BASE z=1.03 m (official MJCF default), small joint sinusoid."""
    cfg = load_t800_sonic()
    n = int(n_dof if n_dof is not None else cfg["n_revolute"])
    t = np.arange(n_frames, dtype=np.float64)
    phase = 2.0 * np.pi * t / max(n_frames, 1)
    q = amplitude_rad * np.sin(phase)[:, None] * np.ones((1, n))
    root_pos = np.zeros((n_frames, 3), dtype=np.float64)
    root_pos[:, 2] = 1.03
    root_rot = np.zeros((n_frames, 4), dtype=np.float64)
    root_rot[:, 3] = 1.0  # xyzw identity
    return {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": q,
        "source": "synthetic_stand_not_bones_seed",
        "robot": cfg["robot"],
    }


def validate_and_filter_clip(
    lib: dict[str, Any],
    *,
    joint_limits_rad: np.ndarray | None = None,
) -> dict[str, Any]:
    """Schema-check then clip-filter. Limits default to ±π (not hardware ratings)."""
    meta = validate_motion_lib(lib)
    q = np.asarray(lib["dof_pos"], dtype=np.float64)
    n = q.shape[1]
    limits = (
        np.asarray(joint_limits_rad, dtype=np.float64).reshape(n, 2)
        if joint_limits_rad is not None
        else np.column_stack([-np.pi * np.ones(n), np.pi * np.ones(n)])
    )
    dt_s = 1.0 / float(lib["fps"])
    filt = filter_motion_clip(q, joint_limits_rad=limits, dt_s=dt_s)
    return {
        **meta,
        "keep": filt.keep,
        "filter_reasons": filt.reasons,
        "source": lib.get("source"),
    }
