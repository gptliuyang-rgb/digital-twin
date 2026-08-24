"""Merge E1/E2 fragments into a spec copy. Never silently patch the live file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from interface.schema import HAND_SPEC_PATH

CONTACT_KEYS = (
    "fingertip_material",
    "fingertip_geometry_radius_m",
    "friction_vs_cardboard_static",
    "friction_vs_cardboard_dynamic",
    "normal_stiffness_n_per_m",
    "soft_body_batch_id",
    "skin_batch_id",
    "skin_present",
    "firmware_version",
    "solref_timeconst_s",
    "m_eff_kg",
    "calibration_kind",
    "do_not_treat_as_committed_hardware",
    "human_must_accept_solref",
)


def merge_fragment(base: dict[str, Any], fragment: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key in CONTACT_KEYS:
        if key in fragment and fragment[key] not in (None, ""):
            merged[key] = fragment[key]
    return merged


def guard_live_spec_write(
    out: Path,
    fragment: dict[str, Any],
    *,
    commit_live: bool,
    accept_synthetic: bool,
) -> None:
    writing_live = out.resolve() == HAND_SPEC_PATH.resolve()
    synthetic = bool(fragment.get("do_not_treat_as_committed_hardware")) or (
        fragment.get("calibration_kind") == "synthetic_dry_run"
    )
    if writing_live and not commit_live:
        raise SystemExit(
            "refusing to overwrite live dexhand2_spec.yaml; "
            "write an overlay (--out …/overlay.yaml) or pass --commit-live"
        )
    if writing_live and synthetic and not accept_synthetic:
        raise SystemExit(
            "fragment is marked synthetic / not hardware. "
            "Pass --i-accept-synthetic if you really want it on the live spec."
        )
