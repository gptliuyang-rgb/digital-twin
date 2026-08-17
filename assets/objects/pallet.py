"""EUR-pallet 1200×800×144 mm. Dimensions are the published EPAL standard, not a CAD scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EURO_PALLET_M = (1.20, 0.80, 0.144)
EURO_PALLET_MASS_KG = 25.0  # typical timber EUR; not a warehouse weighing


@dataclass(frozen=True)
class PalletSpec:
    size_m: tuple[float, float, float] = EURO_PALLET_M
    mass_kg: float = EURO_PALLET_MASS_KG
    height_offset_m: float = 0.0


def pallet_mjcf(spec: PalletSpec | None = None, name: str = "pallet") -> str:
    spec = spec or PalletSpec()
    sx, sy, sz = (c / 2.0 for c in spec.size_m)
    z = spec.height_offset_m
    return (
        f'<body name="{name}" pos="0 0 {z:.4f}">\n'
        f'  <geom type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" mass="{spec.mass_kg:.3f}" '
        f'rgba="0.55 0.35 0.15 1"/>\n'
        f"</body>\n"
    )


def support_xy_m(spec: PalletSpec | None = None) -> np.ndarray:
    spec = spec or PalletSpec()
    hx, hy = spec.size_m[0] / 2.0, spec.size_m[1] / 2.0
    return np.array([[-hx, -hy], [hx, hy]], dtype=np.float64)
