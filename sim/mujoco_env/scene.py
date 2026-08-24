"""Industrial MuJoCo scene: floor, Euro pallet, boxes, handheld scan gun.

Contact friction/stiffness values passed in are SCAN_PLACEHOLDER, not E1/E2.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from assets.combined.mjcf_xml import dump_mjcf, find_or_create
from assets.objects.boxes import EURO_PALLET_M


@dataclass(frozen=True)
class SceneSpec:
    friction: float = 0.8
    solref_timeconst_s: float = 0.010
    n_boxes: int = 2
    box_size_m: tuple[float, float, float] = (0.28, 0.22, 0.18)
    box_mass_kg: float = 5.0
    table_height_m: float = 0.85
    # SCAN_PLACEHOLDER: gun barrel length until CAD of the scanner exists.
    gun_length_m: float = 0.12


def _box_inertial(mass: float, size: tuple[float, float, float]) -> str:
    sx, sy, sz = size
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    return (
        f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>'
    )


def attach_industrial_scene(robot_root: ET.Element, spec: SceneSpec | None = None) -> ET.Element:
    spec = spec or SceneSpec()
    world = find_or_create(robot_root, "worldbody")
    solref = f"{spec.solref_timeconst_s} {spec.solref_timeconst_s * 2}"
    friction = f"{spec.friction} {spec.friction * 0.1} 0.001"

    world.append(
        ET.fromstring(
            '<geom name="floor" type="plane" size="0 0 0.05" rgba="0.35 0.35 0.35 1" '
            f'friction="{friction}" solref="{solref}"/>'
        )
    )
    px, py, pz = EURO_PALLET_M
    world.append(
        ET.fromstring(
            f'<body name="pallet" pos="0.70 0 {pz / 2}">'
            f'<geom name="pallet_geom" type="box" size="{px / 2} {py / 2} {pz / 2}" '
            f'rgba="0.55 0.35 0.15 1" friction="{friction}" solref="{solref}"/>'
            "</body>"
        )
    )
    full = spec.box_size_m
    hx, hy, hz = (full[0] / 2, full[1] / 2, full[2] / 2)
    z0 = spec.table_height_m + hz
    for i in range(spec.n_boxes):
        y = -0.12 if i == 0 else 0.18
        x = 0.48
        inertial = _box_inertial(spec.box_mass_kg, full)
        qr_pos = f"{hx + 0.001} 0 0"
        world.append(
            ET.fromstring(
                f'<body name="box_{i}" pos="{x} {y} {z0}">'
                f"{inertial}"
                "<freejoint/>"
                f'<geom name="box_{i}_geom" type="box" size="{hx} {hy} {hz}" '
                f'rgba="0.82 0.64 0.35 1" friction="{friction}" solref="{solref}" condim="4"/>'
                f'<site name="box_{i}_qr" pos="{qr_pos}" size="0.02" rgba="0 0 0 1"/>'
                "</body>"
            )
        )
    gun_l = spec.gun_length_m
    world.append(
        ET.fromstring(
            f'<body name="scan_gun" pos="0.40 -0.35 {spec.table_height_m + 0.04}">'
            '<inertial pos="0 0 0" mass="0.18" diaginertia="0.0002 0.0002 0.00005"/>'
            "<freejoint/>"
            f'<geom name="scan_gun_body" type="capsule" fromto="0 0 0 0 0 {-gun_l}" '
            f'size="0.018" rgba="0.1 0.1 0.12 1" friction="{friction}"/>'
            f'<site name="gun_tcp" pos="0 0 {-gun_l}" size="0.008" rgba="0 1 0 1"/>'
            f'<camera name="gun_cam" pos="0 0 {-gun_l}" xyaxes="0 -1 0 0 0 1" fovy="60"/>'
            "</body>"
        )
    )
    return robot_root


def industrial_xml(robot_xml: str, spec: SceneSpec | None = None) -> str:
    text = robot_xml.lstrip()
    if text.startswith("<?xml"):
        text = text.split("?>", 1)[1]
    root = ET.fromstring(text)
    attach_industrial_scene(root, spec)
    return dump_mjcf(root)
