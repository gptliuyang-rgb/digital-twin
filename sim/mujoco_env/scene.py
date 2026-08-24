"""Industrial MuJoCo scene: floor, stack pad, boxes, handheld scan gun.

Contact friction/stiffness values passed in are SCAN_PLACEHOLDER, not E1/E2.
Layout is sized for T800 elbow reach (~0.30 m forward from pinned base).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from assets.combined.mjcf_xml import dump_mjcf, find_or_create


@dataclass(frozen=True)
class SceneSpec:
    friction: float = 0.8
    solref_timeconst_s: float = 0.010
    n_boxes: int = 1
    box_size_m: tuple[float, float, float] = (0.20, 0.16, 0.14)
    box_mass_kg: float = 2.0
    table_height_m: float = 0.90
    # Smaller than a full Euro pallet so the pad fits in the arm workspace.
    pad_size_m: tuple[float, float, float] = (0.36, 0.36, 0.06)
    # SCAN_PLACEHOLDER: gun barrel length until CAD of the scanner exists.
    gun_length_m: float = 0.14


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
            '<geom name="floor" type="plane" size="0 0 0.05" rgba="0.22 0.22 0.24 1" '
            f'friction="{friction}" solref="{solref}"/>'
        )
    )
    # Soft fill light so props stay readable in GIF / headless renders.
    world.append(
        ET.fromstring(
            '<light name="key" pos="0.4 -0.6 2.2" dir="0 0.2 -1" diffuse="0.9 0.9 0.85" '
            'specular="0.3 0.3 0.3" castshadow="true"/>'
        )
    )
    world.append(
        ET.fromstring(
            '<light name="fill" pos="-0.5 0.8 1.8" dir="0.3 -0.2 -1" diffuse="0.35 0.35 0.4" '
            'castshadow="false"/>'
        )
    )

    px, py, pz = spec.pad_size_m
    pad_z = spec.table_height_m - pz / 2.0
    # Stack pad (green) clearly left of pick table so carry motion is visible.
    world.append(
        ET.fromstring(
            f'<body name="pallet" pos="0.28 0.22 {pad_z}">'
            f'<geom name="pallet_geom" type="box" size="{px / 2} {py / 2} {pz / 2}" '
            f'rgba="0.30 0.55 0.35 1" friction="{friction}" solref="{solref}"/>'
            "</body>"
        )
    )

    full = spec.box_size_m
    hx, hy, hz = (full[0] / 2, full[1] / 2, full[2] / 2)
    z0 = spec.table_height_m + hz

    world.append(
        ET.fromstring(
            f'<body name="pick_table" pos="0.28 -0.22 {spec.table_height_m - 0.03}">'
            f'<geom name="pick_table_geom" type="box" size="0.16 0.14 0.03" '
            f'rgba="0.60 0.40 0.18 1" friction="{friction}" solref="{solref}"/>'
            "</body>"
        )
    )

    for i in range(spec.n_boxes):
        if i == 0:
            x, y = 0.28, -0.22
            rgba = "0.95 0.72 0.28 1"
        else:
            x, y = 0.28, 0.28
            rgba = "0.70 0.52 0.30 1"
        inertial = _box_inertial(spec.box_mass_kg, full)
        qr_pos = f"{hx + 0.001} 0 0"
        if i == 0:
            body_xml = (
                f'<body name="box_{i}" pos="{x} {y} {z0}">'
                f"{inertial}"
                "<freejoint/>"
                f'<geom name="box_{i}_geom" type="box" size="{hx} {hy} {hz}" '
                f'rgba="{rgba}" friction="{friction}" solref="{solref}" condim="4"/>'
                f'<site name="box_{i}_qr" pos="{qr_pos}" size="0.012" rgba="0.05 0.05 0.05 1"/>'
                f'<geom name="box_{i}_qr_marker" type="box" size="0.035 0.035 0.002" '
                f'pos="{qr_pos}" rgba="0.02 0.02 0.02 1" contype="0" conaffinity="0"/>'
                "</body>"
            )
        else:
            body_xml = (
                f'<body name="box_{i}" pos="{x} {y} {z0}">'
                f"{inertial}"
                f'<geom name="box_{i}_geom" type="box" size="{hx} {hy} {hz}" '
                f'rgba="{rgba}" friction="{friction}" solref="{solref}" condim="4"/>'
                f'<site name="box_{i}_qr" pos="{qr_pos}" size="0.012" rgba="0.05 0.05 0.05 1"/>'
                f'<geom name="box_{i}_qr_marker" type="box" size="0.035 0.035 0.002" '
                f'pos="{qr_pos}" rgba="0.02 0.02 0.02 1" contype="0" conaffinity="0"/>'
                "</body>"
            )
        world.append(ET.fromstring(body_xml))

    gun_l = spec.gun_length_m
    table_gun = f"0.18 -0.42 {spec.table_height_m + 0.05}"
    world.append(
        ET.fromstring(
            f'<body name="scan_gun" mocap="true" pos="{table_gun}">'
            '<inertial pos="0 0 0" mass="0.18" diaginertia="0.0002 0.0002 0.00005"/>'
            f'<geom name="scan_gun_body" type="capsule" fromto="0 0 0.02 0 0 {-gun_l}" '
            f'size="0.028" rgba="0.95 0.12 0.08 1" contype="0" conaffinity="0"/>'
            f'<geom name="scan_gun_grip" type="cylinder" fromto="0 0 0.02 0 0.06 0.02" '
            f'size="0.018" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>'
            f'<site name="gun_tcp" pos="0 0 {-gun_l}" size="0.014" rgba="0 1 0.2 1"/>'
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
