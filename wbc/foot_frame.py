"""ADR-024: SONIC/GMR feet are official MJCF LINK_FOOT_* (ankle-roll origin).

The Native SDK URDF uses the same link names but welds them 64.53 mm below
``LINK_ANKLE_ROLL_*`` via ``J_FIXED_FOOT_*``. Mixing those two frames injects a
constant vertical bias into every GMR/SONIC track. This module is the check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml

from interface.schema import REPO_ROOT
from wbc.dims import load_t800_sonic

FOOT_FRAME_DECISION = "mjcf_link_foot_at_ankle_roll"
URDF_SOLE_IN_MJCF_FOOT_M = (0.0, 0.0, -0.06453)
MJCF_COLLISION_BOX_Z_M = -0.054
FORBIDDEN = "urdf_link_foot_as_sonic_body"
OFFSETS_PATH = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_frame_offsets.yaml"


class FootFrameError(ValueError):
    """Raised when a caller tries to use the URDF sole as a SONIC body."""


def load_frame_offsets() -> dict[str, Any]:
    raw = yaml.safe_load(OFFSETS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("t800_frame_offsets.yaml must be a mapping")
    return raw


def load_foot_frame(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_t800_sonic()
    block = dict(cfg["foot_frame"])
    return block


def urdf_sole_world_m(foot_pos_m: np.ndarray, foot_rot_3x3: np.ndarray) -> np.ndarray:
    """World position of the URDF sole, given the MJCF ``LINK_FOOT_*`` pose.

    ``sole = xpos_foot + R_foot @ [0, 0, -0.06453]``. Offset is official URDF,
    not a fitted contact height.
    """
    offset = np.asarray(URDF_SOLE_IN_MJCF_FOOT_M, dtype=np.float64)
    pos = np.asarray(foot_pos_m, dtype=np.float64).reshape(3)
    rot = np.asarray(foot_rot_3x3, dtype=np.float64).reshape(3, 3)
    return pos + rot @ offset


def refuse_urdf_foot_as_sonic_body(name: str) -> None:
    """URDF ``LINK_FOOT_*`` / ``J_FIXED_FOOT_*`` is the sole, not the SONIC body."""
    banned = {
        "urdf_link_foot",
        "J_FIXED_FOOT_L",
        "J_FIXED_FOOT_R",
        "urdf_LINK_FOOT_L",
        "urdf_LINK_FOOT_R",
    }
    if name in banned:
        raise FootFrameError(
            "SONIC/GMR must track MJCF LINK_FOOT_* at the ankle-roll origin "
            "(ADR-024). URDF J_FIXED_FOOT_* is a 64.53 mm sole offset."
        )


def assert_foot_frame(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_t800_sonic()
    block = load_foot_frame(cfg)
    if block.get("decision") != FOOT_FRAME_DECISION:
        raise FootFrameError(f"foot_frame.decision must be {FOOT_FRAME_DECISION!r}")
    if block.get("forbidden") != FORBIDDEN:
        raise FootFrameError("foot_frame.forbidden marker missing")
    tracked = cfg["tracked_bodies"]
    if tracked["left_foot"] != block["tracked_body_left"] or tracked["left_foot"] != "LINK_FOOT_L":
        raise FootFrameError("left_foot must be MJCF LINK_FOOT_L")
    if tracked["right_foot"] != block["tracked_body_right"] or tracked["right_foot"] != "LINK_FOOT_R":
        raise FootFrameError("right_foot must be MJCF LINK_FOOT_R")
    sole = [float(x) for x in block["urdf_sole_in_mjcf_foot_m"]]
    if sole != list(URDF_SOLE_IN_MJCF_FOOT_M):
        raise FootFrameError(f"urdf sole offset drifted: {sole}")
    box_z = float(block["mjcf_collision_box_z_in_foot_m"])
    if abs(box_z - MJCF_COLLISION_BOX_Z_M) > 1e-9:
        raise FootFrameError(f"MJCF collision box z drifted: {box_z}")
    offsets = load_frame_offsets()
    if offsets["foot"]["decision"] != FOOT_FRAME_DECISION:
        raise FootFrameError("t800_frame_offsets.yaml foot.decision must match ADR-024")
    if abs(float(offsets["foot"]["delta_z_m"]) - abs(URDF_SOLE_IN_MJCF_FOOT_M[2])) > 1e-9:
        raise FootFrameError("frame-offset delta_z_m drifted from URDF J_FIXED_FOOT")
    return block
