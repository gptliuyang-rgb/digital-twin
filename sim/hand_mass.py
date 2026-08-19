"""Soft-body mass that official skeleton MJCF/URDF does not contain.

Product mass 0.745 kg vs skeleton 0.6207 kg ⇒ Δ = 0.1243 kg. The delta is a
derived scalar, not a measured CoM. SONIC load-aware training is refused until
`com_in_wrist_frame_m` is filled.
"""

from __future__ import annotations

from dataclasses import dataclass

from interface.schema import REQUIRED_INPUT_TOKEN, HandSpec, SpecIncompleteError


@dataclass(frozen=True)
class HandMassBudget:
    skeleton_kg: float
    mount_kg: float
    product_kg: float
    soft_body_delta_kg: float
    two_hands_product_kg: float
    sonic_ready: bool


def mass_budget(spec: HandSpec) -> HandMassBudget:
    mount = float(spec.raw["mount_mass_kg"]["value"])
    delta = float(spec.raw["soft_body_mass_delta_kg"]["value"])
    com = spec.raw.get("com_in_wrist_frame_m")
    return HandMassBudget(
        skeleton_kg=spec.skeleton_mass_kg,
        mount_kg=mount,
        product_kg=spec.product_mass_kg,
        soft_body_delta_kg=delta,
        two_hands_product_kg=2.0 * spec.product_mass_kg,
        sonic_ready=com != REQUIRED_INPUT_TOKEN and com is not None,
    )


def require_sonic_mass(spec: HandSpec) -> None:
    budget = mass_budget(spec)
    if not budget.sonic_ready:
        raise SpecIncompleteError(
            ["com_in_wrist_frame_m"],
            spec_path=spec.path,
        )


def point_mass_mjcf(spec: HandSpec, body_name: str = "soft_body_delta") -> str:
    """Emit a point-mass geom. Refuses if wrist CoM is still REQUIRED_INPUT."""
    require_sonic_mass(spec)
    com = spec.raw["com_in_wrist_frame_m"]
    x, y, z = (float(v) for v in com)
    m = float(spec.raw["soft_body_mass_delta_kg"]["value"])
    return (
        f'<body name="{body_name}">\n'
        f'  <inertial pos="{x:.6f} {y:.6f} {z:.6f}" mass="{m:.6f}" diaginertia="1e-8 1e-8 1e-8"/>\n'
        f"</body>\n"
    )
