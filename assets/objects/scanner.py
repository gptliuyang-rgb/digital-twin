"""Handheld scanner placeholder. TCP offset stays REQUIRED_INPUT (CAD of the gun).

Working envelope numbers match sim.qr_scanner.ScanSpec (industrial handheld
typical: 5–30 cm, 45°, 0.3 m/s). They are task-gate constants, not a measured
gun datasheet — swap them when the physical model is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScannerPlaceholder:
    body_size_m: tuple[float, float, float] = (0.14, 0.07, 0.04)
    mass_kg: float = 0.22
    fov_half_angle_rad: float = float(np.deg2rad(20.0))
    d_max_m: float = 0.30


def scanner_mjcf(name: str = "scanner", model: ScannerPlaceholder | None = None) -> str:
    model = model or ScannerPlaceholder()
    sx, sy, sz = (s / 2.0 for s in model.body_size_m)
    d = model.d_max_m
    cone = d * float(np.tan(model.fov_half_angle_rad))
    return (
        f'<body name="{name}">\n'
        f'  <geom name="{name}_body" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
        f'mass="{model.mass_kg:.4f}" rgba="0.1 0.1 0.1 1"/>\n'
        f'  <site name="{name}_tcp" pos="0 0 {sz:.4f}" size="0.004" rgba="1 0.2 0.1 1"/>\n'
        f'  <geom name="{name}_fov" type="sphere" size="{cone:.4f}" pos="0 0 {d:.4f}" '
        f'contype="0" conaffinity="0" group="3" rgba="0.2 0.8 1 0.08"/>\n'
        f"</body>\n"
    )
