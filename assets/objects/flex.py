"""2×2 piecewise-rigid cardboard approximation (weak connecting joints).

Used when max(size) > 0.4 m. Four rigid tiles + slide joints. Not a continuum
shell. Contact parameters still come from the caller (E1/E2 or SCAN_PLACEHOLDER).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlexBoxSpec:
    size_m: tuple[float, float, float]
    mass_kg: float
    pos_m: tuple[float, float, float]
    name: str = "box_flex"
    friction: str = "0.8 0.08 0.001"
    solref: str = "0.01 0.02"
    rgba: str = "0.86 0.68 0.38 1"
    joint_stiffness: float = 40.0
    joint_damping: float = 2.0


def should_split(size_m: np.ndarray | tuple[float, float, float]) -> bool:
    return float(np.max(np.asarray(size_m, dtype=np.float64))) > 0.4


def flex_box_xml(spec: FlexBoxSpec) -> str:
    """Return a worldbody XML snippet: parent body + 2×2 tiles in XY."""
    sx, sy, sz = spec.size_m
    hx, hy, hz = sx / 4.0, sy / 4.0, sz / 2.0
    tile_mass = spec.mass_kg / 4.0
    ixx = tile_mass * (hy * 2 * hy * 2 + sz * sz) / 12.0
    iyy = tile_mass * (hx * 2 * hx * 2 + sz * sz) / 12.0
    izz = tile_mass * ((hx * 2) ** 2 + (hy * 2) ** 2) / 12.0
    px, py, pz = spec.pos_m
    tiles = []
    offsets = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    for i, (ox, oy) in enumerate(offsets):
        tx = ox * hx
        ty = oy * hy
        tiles.append(
            f'<body name="{spec.name}_{i}" pos="{tx:.6f} {ty:.6f} 0">'
            f'<inertial pos="0 0 0" mass="{tile_mass:.6f}" '
            f'diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>'
            f'<joint name="{spec.name}_jx_{i}" type="slide" axis="1 0 0" '
            f'stiffness="{spec.joint_stiffness}" damping="{spec.joint_damping}" range="-0.01 0.01"/>'
            f'<joint name="{spec.name}_jy_{i}" type="slide" axis="0 1 0" '
            f'stiffness="{spec.joint_stiffness}" damping="{spec.joint_damping}" range="-0.01 0.01"/>'
            f'<geom name="{spec.name}_g_{i}" type="box" size="{hx:.6f} {hy:.6f} {hz:.6f}" '
            f'rgba="{spec.rgba}" friction="{spec.friction}" solref="{spec.solref}" condim="4"/>'
            "</body>"
        )
    inner = "".join(tiles)
    return (
        f'<body name="{spec.name}" pos="{px:.6f} {py:.6f} {pz:.6f}">'
        '<freejoint/>'
        '<inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>'
        f"{inner}"
        "</body>"
    )
