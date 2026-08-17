"""Batch-filter a T800 motion_lib. Uses committed MJCF ranges, not ±π."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import REPO_ROOT
from wbc.filter import ClipFilterResult, filter_motion_clip
from wbc.gmr.motion_lib import validate_motion_lib

KINEMATICS_YAML = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_kinematics.yaml"


@dataclass
class FilterLibraryReport:
    n_clips: int
    n_keep: int
    n_drop: int
    clips: list[dict[str, Any]] = field(default_factory=list)
    limits_source: str = "t800_kinematics.yaml mjcf_revolute_limits_rad"
    g1_refused: bool = False


def official_mjcf_limits_rad(path: Path | None = None) -> np.ndarray:
    """Official MJCF `range=` as (n_dof, 2) radians. Not actuatorfrcrange."""
    raw = yaml.safe_load((path or KINEMATICS_YAML).read_text(encoding="utf-8"))
    order = list(raw["joint_order"])
    lim_map = raw.get("mjcf_revolute_limits_rad") or {}
    limits = np.full((len(order), 2), np.nan)
    for i, name in enumerate(order):
        pair = lim_map[name]
        limits[i] = [float(pair[0]), float(pair[1])]
    if not np.isfinite(limits).all():
        raise ValueError("incomplete mjcf_revolute_limits_rad")
    if float(np.max(np.abs(limits))) > 20.0:
        raise ValueError(
            "MJCF joint limits look like actuatorfrcrange (N·m), not range= (rad). "
            "parse_mjcf_joint_limits must use \\brange="
        )
    return limits


def _as_clips(lib: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "clips" in lib:
        out = []
        for i, clip in enumerate(lib["clips"]):
            name = str(clip.get("name", f"clip{i}"))
            out.append((name, clip))
        return out
    return [(str(lib.get("name", "clip0")), lib)]


def filter_motion_library(
    lib: dict[str, Any],
    *,
    joint_limits_rad: np.ndarray | None = None,
    foot_height_m: np.ndarray | None = None,
) -> FilterLibraryReport:
    """Validate schema then drop limit / NaN / blow-up clips.

    Joint limits default to the committed official MJCF `range=` (radians).
    Passing ±π is not allowed as a silent default — that would hide limit hits.
    """
    limits = (
        np.asarray(joint_limits_rad, dtype=np.float64)
        if joint_limits_rad is not None
        else official_mjcf_limits_rad()
    )
    clips = _as_clips(lib)
    rows: list[dict[str, Any]] = []
    n_keep = 0
    for name, clip in clips:
        meta = validate_motion_lib(clip)
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        dt_s = 1.0 / float(clip["fps"])
        feet = clip.get("foot_height_m", foot_height_m)
        filt: ClipFilterResult = filter_motion_clip(
            q,
            joint_limits_rad=limits,
            foot_height_m=None if feet is None else np.asarray(feet, dtype=np.float64),
            dt_s=dt_s,
        )
        rows.append(
            {
                "name": name,
                "keep": filt.keep,
                "reasons": filt.reasons,
                "n_frames": meta["n_frames"],
                "duration_s": meta["duration_s"],
                "source": clip.get("source"),
            }
        )
        if filt.keep:
            n_keep += 1
    return FilterLibraryReport(
        n_clips=len(clips),
        n_keep=n_keep,
        n_drop=len(clips) - n_keep,
        clips=rows,
    )
