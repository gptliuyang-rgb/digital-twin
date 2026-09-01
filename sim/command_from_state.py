"""Pack a 75-D CommandVector from the combined MuJoCo twin (sim-only).

Wrist pose in this packing is the assembled hand wrist body (``l_wrist`` /
``r_wrist``), because that is what the industrial FSM IK tracks. It is **not**
the SONIC ``LINK_WRIST_END_*`` claim (ADR-001) and must not be used as a
policy-eval number (ADR-006).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from interface.schema import CommandVector, HandSpec, load_hand_spec
from vla.adapters.rotation import matrix_to_rot6d


def _body_pos_rot6d(model, data, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        bid = int(model.body(name).id)
    except Exception:
        return None
    pos = np.asarray(data.xpos[bid], dtype=np.float64).reshape(3).copy()
    mat = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3).copy()
    return pos, matrix_to_rot6d(mat)


def _site_pos_rot6d(model, data, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        sid = int(model.site(name).id)
    except Exception:
        return None
    pos = np.asarray(data.site_xpos[sid], dtype=np.float64).reshape(3).copy()
    mat = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
    return pos, matrix_to_rot6d(mat)


def command_from_env(
    env: Any,
    spec: HandSpec | None = None,
    *,
    tool_trigger: int = 0,
    phase: str = "",
) -> CommandVector:
    """Build a validated command from the current sim state."""
    spec = spec or load_hand_spec()
    cv = CommandVector.zeros(spec)
    model, data = env.model, env.data

    head = _body_pos_rot6d(model, data, "LINK_HEAD_YAW")
    if head is not None:
        cv.head_pos = head[0]
        cv.head_rot6d = head[1]

    # Body frames: industrial IK tracks l_wrist / r_wrist bodies, not sites.
    left = _body_pos_rot6d(model, data, "l_wrist") or _site_pos_rot6d(model, data, "l_wrist")
    right = _body_pos_rot6d(model, data, "r_wrist") or _site_pos_rot6d(model, data, "r_wrist")
    if left is not None:
        cv.left_wrist_pos = left[0]
        cv.left_wrist_rot6d = left[1]
    if right is not None:
        cv.right_wrist_pos = right[0]
        cv.right_wrist_rot6d = right[1]

    waist = _body_pos_rot6d(model, data, "LINK_WAIST_YAW")
    if waist is not None:
        cv.pelvis_height = float(waist[0][2])

    cv.left_hand_q = np.asarray(data.qpos[env.handles.left_hand.qadr], dtype=np.float64).copy()
    cv.right_hand_q = np.asarray(data.qpos[env.handles.right_hand.qadr], dtype=np.float64).copy()
    cv.left_hand_mode = 0
    cv.right_hand_mode = 0
    cv.tool_trigger = 1 if tool_trigger or phase == "scan" else 0
    cv.nav_cmd = np.zeros(3)
    cv.loco_mode = 0
    # Clamp hands into limits: kinematic assists can sit on the bound.
    limits = spec.limits_vector()
    cv.left_hand_q = np.clip(cv.left_hand_q, limits[:, 0], limits[:, 1])
    cv.right_hand_q = np.clip(cv.right_hand_q, limits[:, 0], limits[:, 1])
    cv.validate()
    return cv


def commands_to_chunks(commands: np.ndarray, horizon: int = 16) -> np.ndarray:
    """Trim to a whole number of chunks. Shape (N, H, D)."""
    arr = np.asarray(commands, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"commands must be (T, D), got {arr.shape}")
    n = (arr.shape[0] // horizon) * horizon
    if n == 0:
        raise ValueError(f"need at least {horizon} steps, got {arr.shape[0]}")
    return arr[:n].reshape(-1, horizon, arr.shape[1])
