"""Posture labels for free-base T800 diagnostics.

Bring-up PD can settle into a large lean that is still below the fall tilt
threshold. That must not be read as a stand, and it is not a SONIC gate
(ADR-025 / ADR-026).
"""

from __future__ import annotations

from typing import Any, Literal

Posture = Literal["upright", "leaned", "fallen"]

# Copied from eval/configs/l2_freebase_stand.yaml. Diagnostic bands, not CAD.
DEFAULT_FALL_PELVIS_Z_M = 0.40
DEFAULT_FALL_TILT_RAD = 0.80
# Official 3 s PD hold reached ~0.72 rad (~41°) with fallen=false. That is leaned.
DEFAULT_LEAN_TILT_RAD = 0.20


def classify_posture(
    pelvis_z_m: float,
    tilt_rad: float,
    *,
    fall_pelvis_z_m: float = DEFAULT_FALL_PELVIS_Z_M,
    fall_tilt_rad: float = DEFAULT_FALL_TILT_RAD,
    lean_tilt_rad: float = DEFAULT_LEAN_TILT_RAD,
) -> Posture:
    """Return upright / leaned / fallen. Fallen wins over leaned."""
    z = float(pelvis_z_m)
    tilt = float(tilt_rad)
    if z < float(fall_pelvis_z_m) or tilt > float(fall_tilt_rad):
        return "fallen"
    if tilt > float(lean_tilt_rad):
        return "leaned"
    return "upright"


def fallen_from_cfg(pelvis_z_m: float, tilt_rad: float, cfg: dict[str, Any]) -> bool:
    fall = cfg["fall"]
    return classify_posture(
        pelvis_z_m,
        tilt_rad,
        fall_pelvis_z_m=float(fall["pelvis_z_m"]),
        fall_tilt_rad=float(fall["tilt_rad"]),
        lean_tilt_rad=float(cfg.get("lean", {}).get("tilt_rad", DEFAULT_LEAN_TILT_RAD)),
    ) == "fallen"


def posture_from_cfg(pelvis_z_m: float, tilt_rad: float, cfg: dict[str, Any]) -> Posture:
    fall = cfg["fall"]
    return classify_posture(
        pelvis_z_m,
        tilt_rad,
        fall_pelvis_z_m=float(fall["pelvis_z_m"]),
        fall_tilt_rad=float(fall["tilt_rad"]),
        lean_tilt_rad=float(cfg.get("lean", {}).get("tilt_rad", DEFAULT_LEAN_TILT_RAD)),
    )
