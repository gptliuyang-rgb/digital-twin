"""Industrial MuJoCo scene: grounded benches, carton, handheld scan gun.

Contact friction/stiffness are SCAN_PLACEHOLDER (not E1/E2). Layout fits T800
elbow reach (~0.30 m forward from the pinned base).
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
    box_size_m: tuple[float, float, float] = (0.18, 0.14, 0.12)
    box_mass_kg: float = 2.0
    # Top surface height of both benches (standing humanoid waist/chest work).
    table_height_m: float = 0.88
    # Top slab thickness; legs reach the floor.
    top_thickness_m: float = 0.05
    leg_size_m: float = 0.04
    # Stack-bench top footprint (not a floating Euro pallet slab).
    pad_size_m: tuple[float, float, float] = (0.40, 0.32, 0.04)
    # Barrel length for the placeholder pistol-style scanner (no vendor CAD yet).
    gun_barrel_m: float = 0.16
    gun_length_m: float = 0.16  # alias used by older callers


def _box_inertial(mass: float, size: tuple[float, float, float]) -> str:
    sx, sy, sz = size
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    return (
        f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>'
    )


def _bench_xml(
    *,
    name: str,
    cx: float,
    cy: float,
    top_h: float,
    half_x: float,
    half_y: float,
    top_t: float,
    leg: float,
    top_rgba: str,
    leg_rgba: str = "0.22 0.23 0.26 1",
    friction: str,
    solref: str,
) -> str:
    """Workbench with top, apron, four legs, and foot pads on the floor."""
    top_z = top_h - top_t / 2.0
    leg_h = top_h - top_t
    inset_x = half_x - leg * 1.1
    inset_y = half_y - leg * 1.1
    foot = leg * 1.35
    apron_t = 0.025
    geoms = [
        f'<geom name="{name}_top" type="box" size="{half_x} {half_y} {top_t / 2}" '
        f'pos="0 0 0" rgba="{top_rgba}" friction="{friction}" solref="{solref}"/>',
        # Skirt / apron under the top so the slab does not look like a floating plate.
        f'<geom name="{name}_apron" type="box" size="{half_x * 0.92} {half_y * 0.92} {apron_t / 2}" '
        f'pos="0 0 {-top_t / 2 - apron_t / 2}" rgba="0.18 0.18 0.20 1" '
        f'friction="{friction}" solref="{solref}"/>',
    ]
    for i, (lx, ly) in enumerate(
        ((-inset_x, -inset_y), (-inset_x, inset_y), (inset_x, -inset_y), (inset_x, inset_y))
    ):
        geoms.append(
            f'<geom name="{name}_leg{i}" type="box" size="{leg} {leg} {leg_h / 2}" '
            f'pos="{lx} {ly} {-top_t / 2 - leg_h / 2}" rgba="{leg_rgba}" '
            f'friction="{friction}" solref="{solref}"/>'
        )
        # Foot pad at z≈0 for a clear ground contact.
        geoms.append(
            f'<geom name="{name}_foot{i}" type="box" size="{foot} {foot} 0.012" '
            f'pos="{lx} {ly} {-top_z + 0.012}" rgba="0.12 0.12 0.14 1" '
            f'friction="{friction}" solref="{solref}"/>'
        )
    inner = "".join(geoms)
    return f'<body name="{name}" pos="{cx} {cy} {top_z}">{inner}</body>'


def _scan_gun_xml(*, barrel_m: float, table_pos: str) -> str:
    """Handheld barcode scanner placeholder: pistol grip + housing + barrel + window."""
    b = barrel_m
    tip = 0.055 + b
    return (
        f'<body name="scan_gun" mocap="true" pos="{table_pos}">'
        '<inertial pos="0.05 -0.04 0" mass="0.25" diaginertia="0.0005 0.0006 0.00025"/>'
        # Pistol grip
        '<geom name="scan_gun_grip" type="capsule" fromto="0.01 0.01 0 0.01 -0.11 0" '
        'size="0.020" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>'
        # Main housing
        '<geom name="scan_gun_housing" type="box" size="0.055 0.032 0.028" pos="0.03 0.0 0" '
        'rgba="0.16 0.17 0.20 1" contype="0" conaffinity="0"/>'
        # Amber / red scan engine window band
        '<geom name="scan_gun_band" type="box" size="0.014 0.034 0.030" pos="0.08 0 0" '
        'rgba="0.90 0.25 0.08 1" contype="0" conaffinity="0"/>'
        # Barrel
        f'<geom name="scan_gun_barrel" type="capsule" fromto="0.09 0 0 {tip - 0.01} 0 0" '
        'size="0.015" rgba="0.28 0.29 0.32 1" contype="0" conaffinity="0"/>'
        # Green optical nose
        f'<geom name="scan_gun_nose" type="cylinder" fromto="{tip - 0.012} 0 0 {tip} 0 0" '
        'size="0.018" rgba="0.10 0.85 0.40 1" contype="0" conaffinity="0"/>'
        # Trigger
        '<geom name="scan_gun_trigger" type="box" size="0.010 0.014 0.007" pos="0.025 -0.040 0" '
        'rgba="0.40 0.40 0.44 1" contype="0" conaffinity="0"/>'
        f'<site name="gun_tcp" pos="{tip} 0 0" size="0.009" rgba="0 1 0.3 1" '
        'xyaxes="0 1 0 0 0 1"/>'
        f'<camera name="gun_cam" pos="{tip} 0 0" xyaxes="0 -1 0 0 0 1" fovy="55"/>'
        "</body>"
    )


def attach_industrial_scene(robot_root: ET.Element, spec: SceneSpec | None = None) -> ET.Element:
    spec = spec or SceneSpec()
    world = find_or_create(robot_root, "worldbody")
    solref = f"{spec.solref_timeconst_s} {spec.solref_timeconst_s * 2}"
    friction = f"{spec.friction} {spec.friction * 0.1} 0.001"
    top_t = spec.top_thickness_m
    leg = spec.leg_size_m

    # Concrete-ish floor + soft ambient lighting (readable GIFs, less void).
    world.append(
        ET.fromstring(
            '<geom name="floor" type="plane" size="0 0 0.05" rgba="0.38 0.39 0.41 1" '
            f'friction="{friction}" solref="{solref}"/>'
        )
    )
    world.append(
        ET.fromstring(
            '<geom name="backdrop" type="plane" size="2 0 0.05" pos="-0.6 0 1.2" '
            'xyaxes="0 1 0 0 0 1" rgba="0.16 0.17 0.19 1" contype="0" conaffinity="0"/>'
        )
    )
    world.append(
        ET.fromstring(
            '<light name="key" pos="0.6 -0.8 2.4" dir="-0.15 0.25 -1" diffuse="0.95 0.93 0.88" '
            'specular="0.35 0.35 0.35" castshadow="true"/>'
        )
    )
    world.append(
        ET.fromstring(
            '<light name="fill" pos="-0.4 0.9 1.9" dir="0.25 -0.2 -1" diffuse="0.40 0.42 0.48" '
            'castshadow="false"/>'
        )
    )

    # --- pick bench (wood) @ robot right-front ---
    world.append(
        ET.fromstring(
            _bench_xml(
                name="pick_table",
                cx=0.30,
                cy=-0.24,
                top_h=spec.table_height_m,
                half_x=0.18,
                half_y=0.16,
                top_t=top_t,
                leg=leg,
                top_rgba="0.55 0.38 0.20 1",
                friction=friction,
                solref=solref,
            )
        )
    )

    # --- stack bench (green industrial) @ robot left-front ---
    px, py, _ = spec.pad_size_m
    world.append(
        ET.fromstring(
            _bench_xml(
                name="pallet",
                cx=0.30,
                cy=0.24,
                top_h=spec.table_height_m,
                half_x=px / 2,
                half_y=py / 2,
                top_t=top_t,
                leg=leg,
                top_rgba="0.32 0.48 0.36 1",
                friction=friction,
                solref=solref,
            )
        )
    )

    full = spec.box_size_m
    hx, hy, hz = (full[0] / 2, full[1] / 2, full[2] / 2)
    z0 = spec.table_height_m + hz

    for i in range(spec.n_boxes):
        x, y = (0.30, -0.24) if i == 0 else (0.30, 0.28)
        rgba = "0.86 0.68 0.38 1" if i == 0 else "0.70 0.52 0.30 1"
        inertial = _box_inertial(spec.box_mass_kg, full)
        # QR sticker on −X face (toward the robot / scanner approach).
        qr = f"{-hx - 0.001} 0 {hz * 0.15}"
        free = "<freejoint/>" if i == 0 else ""
        world.append(
            ET.fromstring(
                f'<body name="box_{i}" pos="{x} {y} {z0}">'
                f"{inertial}{free}"
                f'<geom name="box_{i}_geom" type="box" size="{hx} {hy} {hz}" '
                f'rgba="{rgba}" friction="{friction}" solref="{solref}" condim="4"/>'
                f'<geom name="box_{i}_seam" type="box" size="{hx * 0.98} 0.002 {hz * 0.98}" '
                f'pos="0 0 0" rgba="0.55 0.40 0.22 1" contype="0" conaffinity="0"/>'
                f'<site name="box_{i}_qr" pos="{qr}" size="0.010" rgba="0.05 0.05 0.05 1"/>'
                f'<geom name="box_{i}_qr_plate" type="box" size="0.032 0.032 0.0015" '
                f'pos="{qr}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>'
                f'<geom name="box_{i}_qr_ink" type="box" size="0.024 0.024 0.0018" '
                f'pos="{qr}" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>'
                "</body>"
            )
        )

    # Rest the scanner on the pick bench, barrel along +X, grip down.
    gun_z = spec.table_height_m + 0.04
    table_gun = f"0.22 -0.40 {gun_z}"
    barrel = spec.gun_barrel_m if spec.gun_barrel_m else spec.gun_length_m
    world.append(ET.fromstring(_scan_gun_xml(barrel_m=barrel, table_pos=table_gun)))
    return robot_root


def industrial_xml(robot_xml: str, spec: SceneSpec | None = None) -> str:
    text = robot_xml.lstrip()
    if text.startswith("<?xml"):
        text = text.split("?>", 1)[1]
    root = ET.fromstring(text)
    attach_industrial_scene(root, spec)
    return dump_mjcf(root)
