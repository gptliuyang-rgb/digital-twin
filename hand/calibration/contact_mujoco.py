"""Map E1/E2 numbers onto MuJoCo pad geoms. Pure functions — no simulator import.

``solref[0] ≈ 2π / sqrt(k / m_eff)`` is a *proposal*. A human must accept it
before copying the fragment into the live DexHand2 spec (PROTOCOL.md).
"""

from __future__ import annotations

import math
import re
from typing import Any

from interface.schema import REQUIRED_INPUT_TOKEN

# Distal pad + E2 fixture moving mass used only for the solref proposal.
# Not a measured hardware CoM.
DEFAULT_PAD_M_EFF_KG = 0.03

CONTACT_KEYS = (
    "friction_vs_cardboard_static",
    "friction_vs_cardboard_dynamic",
    "normal_stiffness_n_per_m",
)

_PAD_GEOM = re.compile(r'<geom name="([^"]+_pad_\d+)"([^>]*)/>')
_ATTR = re.compile(r'\s+(friction|solref|condim)="[^"]*"')


def _is_number(value: Any) -> bool:
    if value is None or value == REQUIRED_INPUT_TOKEN:
        return False
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def contact_is_calibrated(raw: dict[str, Any] | None) -> bool:
    """True when E1 μ and E2 k are numeric (overlay or live spec)."""
    if not raw:
        return False
    return all(_is_number(raw.get(key)) for key in CONTACT_KEYS)


def solref_timeconst_from_stiffness(
    k_n_per_m: float, m_eff_kg: float = DEFAULT_PAD_M_EFF_KG
) -> float:
    if k_n_per_m <= 0 or m_eff_kg <= 0:
        raise ValueError("k_n_per_m and m_eff_kg must be positive")
    return 2.0 * math.pi / math.sqrt(k_n_per_m / m_eff_kg)


def solref_timeconst_s(raw: dict[str, Any]) -> float:
    if _is_number(raw.get("solref_timeconst_s")):
        return float(raw["solref_timeconst_s"])
    k = float(raw["normal_stiffness_n_per_m"])
    m_eff = float(raw["m_eff_kg"]) if _is_number(raw.get("m_eff_kg")) else DEFAULT_PAD_M_EFF_KG
    return solref_timeconst_from_stiffness(k, m_eff)


def friction_attr(mu_s: float, mu_d: float | None = None) -> str:
    """MuJoCo ``friction='slide spin roll'``. Slide uses μ_s; spin is a small torsion."""
    slide = float(mu_s)
    if mu_d is not None and math.isfinite(float(mu_d)):
        slide = float(mu_s)  # MuJoCo has no separate μ_d; keep static as the cone.
    spin = max(slide * 0.1, 1e-4)
    roll = 0.001
    return f"{slide:.6f} {spin:.6f} {roll:.6f}"


def solref_attr(timeconst_s: float) -> str:
    tc = float(timeconst_s)
    return f"{tc:.6f} {tc * 2.0:.6f}"


def contact_xml_attrs(raw: dict[str, Any], *, condim: int = 4) -> dict[str, str] | None:
    if not contact_is_calibrated(raw):
        return None
    mu_s = float(raw["friction_vs_cardboard_static"])
    mu_d = float(raw["friction_vs_cardboard_dynamic"])
    return {
        "friction": friction_attr(mu_s, mu_d),
        "solref": solref_attr(solref_timeconst_s(raw)),
        "condim": str(int(condim)),
    }


def patch_pad_geoms(
    xml: str,
    *,
    friction: str,
    solref: str,
    condim: int | str = 4,
) -> str:
    """Set friction/solref/condim on every ``*_pad_*`` sphere injected by gen_derived."""

    def _repl(match: re.Match[str]) -> str:
        name, attrs = match.group(1), match.group(2)
        attrs = _ATTR.sub("", attrs)
        return (
            f'<geom name="{name}"{attrs} condim="{condim}" '
            f'friction="{friction}" solref="{solref}"/>'
        )

    return _PAD_GEOM.sub(_repl, xml)


def apply_contact_to_xml(xml: str, raw: dict[str, Any], *, condim: int = 4) -> str:
    attrs = contact_xml_attrs(raw, condim=condim)
    if attrs is None:
        return xml
    return patch_pad_geoms(
        xml, friction=attrs["friction"], solref=attrs["solref"], condim=attrs["condim"]
    )
