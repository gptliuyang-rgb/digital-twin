"""Industrial MuJoCo scene: grounded benches, carton, handheld barcode scanner.

Contact friction/stiffness are SCAN_PLACEHOLDER (not E1/E2). Layout fits T800
elbow reach (~0.30 m forward from the pinned base).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from assets.combined.mjcf_xml import dump_mjcf, find_or_create
from assets.objects.boxes import write_qr_texture
from assets.objects.scan_gun.generate import OUT_DIR as SCAN_GUN_DIR
from assets.objects.scan_gun.generate import TIP_X as SCAN_GUN_TIP_X
from interface.schema import REPO_ROOT


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
    # Origin → scan window along +X. Real handheld scanners are ~12 cm, not a rifle.
    gun_barrel_m: float = 0.13
    gun_length_m: float = 0.13  # alias used by older callers
    qr_payload: str = "BOX-0"
    use_flex_box: bool = False


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


def _vis(name: str, gtype: str, **kw: object) -> str:
    parts = [f'<geom name="{name}" type="{gtype}"']
    for key, value in kw.items():
        parts.append(f'{key}="{value}"')
    parts.append('contype="0" conaffinity="0" group="0"/>')
    return " ".join(parts)


_SCAN_GUN_MATERIALS = (
    '<material name="scan_gun_body" rgba="0.10 0.10 0.11 1" specular="0.28" shininess="0.25"/>',
    '<material name="scan_gun_accent" rgba="0.93 0.74 0.10 1" specular="0.18" shininess="0.15"/>',
    '<material name="scan_gun_glass" rgba="0.62 0.07 0.06 1" specular="0.65" shininess="0.55"/>',
    '<material name="scan_gun_metal" rgba="0.38 0.39 0.41 1" specular="0.45" shininess="0.40"/>',
    '<material name="scan_gun_rubber" rgba="0.07 0.07 0.08 1" specular="0.08" shininess="0.05"/>',
)


def _scan_gun_mesh_assets() -> list[str]:
    files = {
        "scan_gun_body_msh": SCAN_GUN_DIR / "body.stl",
        "scan_gun_accent_msh": SCAN_GUN_DIR / "accent.stl",
        "scan_gun_window_msh": SCAN_GUN_DIR / "window.stl",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "scan-gun STLs missing. Run: python3 -m assets.objects.scan_gun.generate\n"
            + "\n".join(missing)
        )
    return [
        f'<mesh name="{name}" file="{path.resolve().as_posix()}"/>' for name, path in files.items()
    ]


def _scan_gun_xml(*, barrel_m: float, table_pos: str) -> str:
    """Lofted industrial barcode-scanner mesh. Not vendor CAD.

    Gun frame is unchanged: optical axis +X, grip −Y, ``gun_tcp`` +Z = +X.
    ``barrel_m`` is accepted for API compatibility; the committed STL is ~13 cm.
    """
    _ = barrel_m
    tip = SCAN_GUN_TIP_X
    y0 = 0.006
    geoms = [
        _vis("scan_gun_housing", "mesh", mesh="scan_gun_body_msh", material="scan_gun_body"),
        _vis("scan_gun_bumper", "mesh", mesh="scan_gun_accent_msh", material="scan_gun_accent"),
        _vis("scan_gun_window", "mesh", mesh="scan_gun_window_msh", material="scan_gun_glass"),
        _vis(
            "scan_gun_trigger",
            "box",
            size="0.006 0.011 0.005",
            pos="0.038 -0.034 0",
            rgba="0.16 0.16 0.17 1",
        ),
        _vis(
            "scan_gun_led",
            "box",
            size="0.003 0.004 0.008",
            pos=f"{tip - 0.006:.4f} {y0 + 0.024:.4f} 0",
            rgba="0.15 0.85 0.35 1",
        ),
        _vis(
            "scan_gun_btn_a",
            "cylinder",
            fromto=f"0.042 {y0:.4f} 0.034 0.042 {y0:.4f} 0.041",
            size="0.0055",
            material="scan_gun_metal",
        ),
        _vis(
            "scan_gun_btn_b",
            "cylinder",
            fromto=f"0.062 {y0:.4f} 0.034 0.062 {y0:.4f} 0.041",
            size="0.0055",
            material="scan_gun_metal",
        ),
    ]
    inner = "".join(geoms)
    return (
        f'<body name="scan_gun" mocap="true" pos="{table_pos}">'
        '<inertial pos="0.04 -0.04 0" mass="0.28" diaginertia="0.00055 0.00065 0.00028"/>'
        f"{inner}"
        f'<site name="gun_tcp" pos="{tip:.4f} {y0} 0" size="0.003" rgba="0 0 0 0" '
        'xyaxes="0 1 0 0 0 1"/>'
        f'<camera name="gun_cam" pos="{tip:.4f} {y0} 0" xyaxes="0 -1 0 0 0 1" fovy="55"/>'
        "</body>"
    )


def _qr_texture_xml(payload: str) -> list[str]:
    path = REPO_ROOT / "assets" / "objects" / "generated" / f"qr_{payload}.png"
    write_qr_texture(payload, path, module_px=16)
    return [
        f'<texture name="qr_{payload}" type="2d" file="{path.resolve().as_posix()}"/>',
        f'<material name="qr_{payload}_mat" texture="qr_{payload}" texrepeat="1 1" reflectance="0"/>',
    ]


def attach_industrial_scene(robot_root: ET.Element, spec: SceneSpec | None = None) -> ET.Element:
    spec = spec or SceneSpec()
    asset = find_or_create(robot_root, "asset")
    qr_assets = _qr_texture_xml(spec.qr_payload)
    for xml in (*_scan_gun_mesh_assets(), *_SCAN_GUN_MATERIALS, *qr_assets):
        asset.append(ET.fromstring(xml))
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

    from assets.objects.flex import FlexBoxSpec, flex_box_xml, should_split

    full = spec.box_size_m
    hx, hy, hz = (full[0] / 2, full[1] / 2, full[2] / 2)
    z0 = spec.table_height_m + hz
    qr_mat = f"qr_{spec.qr_payload}_mat"

    for i in range(spec.n_boxes):
        x, y = (0.30, -0.24) if i == 0 else (0.30, 0.28)
        rgba = "0.86 0.68 0.38 1" if i == 0 else "0.70 0.52 0.30 1"
        qr = f"{-hx - 0.001} 0 {hz * 0.15}"
        use_flex = spec.use_flex_box or should_split(full)
        if use_flex and i == 0:
            world.append(
                ET.fromstring(
                    flex_box_xml(
                        FlexBoxSpec(
                            size_m=full,
                            mass_kg=spec.box_mass_kg,
                            pos_m=(x, y, z0),
                            name=f"box_{i}",
                            friction=friction,
                            solref=solref,
                            rgba=rgba,
                        )
                    )
                )
            )
            # QR plate stays a visual child of the parent body.
            parent = None
            for body in world.findall("body"):
                if body.get("name") == f"box_{i}":
                    parent = body
                    break
            if parent is not None:
                parent.append(
                    ET.fromstring(
                        f'<site name="box_{i}_qr" pos="{qr}" size="0.010" rgba="0.05 0.05 0.05 1"/>'
                    )
                )
                parent.append(
                    ET.fromstring(
                        f'<geom name="box_{i}_qr_plate" type="box" size="0.032 0.032 0.0015" '
                        f'pos="{qr}" material="{qr_mat}" contype="0" conaffinity="0"/>'
                    )
                )
            continue
        inertial = _box_inertial(spec.box_mass_kg, full)
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
                f'pos="{qr}" material="{qr_mat}" contype="0" conaffinity="0"/>'
                "</body>"
            )
        )

    # Rest the scanner on the pick-bench edge, window +X, grip hanging −Y off the apron.
    gun_z = spec.table_height_m + 0.038
    table_gun = f"0.24 -0.38 {gun_z}"
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
