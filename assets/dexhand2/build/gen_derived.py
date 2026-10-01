"""Derive Hand 2 MJCF variants. Official files are never overwritten.

Beta 2 official pads collide as convex hulls of `*_tip_sensor_frame.STL`.
This generator **replaces** those hulls with 2–3 spheres fitted to the same
mesh. That is mesh geometry, not the live soft pad (still REQUIRED_INPUT).

Variants
--------
pad_spheres
    Disable group-2 (collision) mesh geoms on pad bodies; inject spheres
    with condim=4. Keep every other official hull.
simplified
    pad_spheres plus disable distal-bone hulls so fingertip contact is the
    spheres only (gen-1 simplified.xml idea, Hand 2 pads). Wrist / phalanx
    hulls stay so box faces still hit the skeleton.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from assets.dexhand2.build.ingest_official import (
    DEFAULT_UPSTREAM,
    official_mjcf,
    parse_sites,
    revision_body_path,
)
from interface.schema import REPO_ROOT, load_hand_spec

DERIVED = REPO_ROOT / "assets" / "dexhand2" / "derived"
PAD_FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
# Distal-biased PCA stations + median local radius keep the fingertip *site*
# within 2 mm of a sphere surface on the official Beta 2 pad meshes.
N_SPHERES = 3
RADIUS_QUANTILE = 0.5
MAX_SITE_TO_SURFACE_M = 0.002
MAX_COLLISION_GEOMS = 60


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    # Binary STL: 80-byte header + uint32 count + 50-byte triangles.
    if len(data) >= 84 and data[:5] != b"solid":
        n = int.from_bytes(data[80:84], "little")
        verts = []
        off = 84
        for _ in range(n):
            for k in range(3):
                base = off + 12 + k * 12
                verts.append(np.frombuffer(data[base : base + 12], dtype=np.float32))
            off += 50
        return np.vstack(verts)
    text = data.decode("utf-8", errors="ignore")
    nums = []
    for line in text.splitlines():
        if "vertex" in line:
            nums.append([float(x) for x in line.split()[1:4]])
    return np.asarray(nums, dtype=np.float64)


def fit_pad_spheres(
    vertices: np.ndarray,
    *,
    n_spheres: int = N_SPHERES,
    radius_quantile: float = RADIUS_QUANTILE,
    site_m: np.ndarray | None = None,
) -> list[tuple[list[float], float]]:
    """Place n spheres along the first PCA axis of the pad mesh.

    The last station is biased distal so the official fingertip site sits on
    (or just inside) a small tip sphere instead of deep inside a bounding ball.
    Radii are the local ``radius_quantile`` of vertex distances — mesh geometry,
    not a measured live-pad radius.
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 8:
        raise ValueError(f"pad mesh has too few vertices: {pts.shape}")
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0]
    if site_m is not None and float(np.dot(np.asarray(site_m, dtype=np.float64) - center, axis)) < 0.0:
        axis = -axis
    proj = (pts - center) @ axis
    lo, hi = float(proj.min()), float(proj.max())
    span = max(hi - lo, 1e-9)
    spheres: list[tuple[list[float], float]] = []
    for i in range(n_spheres):
        frac = (i + 0.75) / n_spheres if i == n_spheres - 1 else (i + 0.5) / n_spheres
        t = lo + span * frac
        c = center + t * axis
        nearby = pts[np.abs(proj - t) < span / n_spheres]
        if nearby.shape[0] < 20:
            nearby = pts
        radial = np.linalg.norm(nearby - c, axis=1)
        radius = float(np.quantile(radial, radius_quantile))
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("fitted pad sphere radius is not a positive finite number")
        spheres.append((c.tolist(), radius))
    return spheres


def site_to_nearest_sphere_surface_m(
    site_m: np.ndarray, spheres: list[tuple[list[float], float]]
) -> float:
    """Signed distance: negative = site inside a sphere. Report abs for the gate."""
    site = np.asarray(site_m, dtype=np.float64)
    signed = [float(np.linalg.norm(site - np.asarray(c)) - r) for c, r in spheres]
    return float(min(signed, key=lambda d: abs(d)))


def disable_group2_mesh_collision(mjcf_text: str, mesh_name: str) -> str:
    """Turn a group-2 hull into visual-only (contype/conaffinity 0)."""
    pattern = rf'(<geom type="mesh" rgba="[^"]+" mesh="{re.escape(mesh_name)}" group="2")(\s*/>)'
    new, n = re.subn(pattern, r'\1 contype="0" conaffinity="0"\2', mjcf_text, count=1)
    if n != 1:
        raise ValueError(f"failed to disable group-2 collision for mesh {mesh_name}")
    return new


def inject_into_body(mjcf_text: str, body_name: str, xml_block: str) -> str:
    pattern = rf'(<body name="{re.escape(body_name)}"[^>]*>)(.*?)(</body>)'

    def repl(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2) + xml_block + match.group(3)

    new, n = re.subn(pattern, repl, mjcf_text, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"failed to inject into body {body_name}")
    return new


def sphere_geoms_xml(
    geom_prefix: str, spheres: list[tuple[list[float], float]], *, condim: int
) -> str:
    lines = []
    for i, (pos, radius) in enumerate(spheres):
        lines.append(
            f'<geom name="{geom_prefix}_pad_sphere_{i}" type="sphere" '
            f'size="{radius:.6f}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
            f'group="2" condim="{condim}" rgba="0.2 0.8 0.4 0.4"/>'
        )
    return "\n                " + "\n                ".join(lines) + "\n              "


def count_collision_geoms(mjcf_text: str) -> dict[str, int]:
    geoms = re.findall(r"<geom\b[^>]*>", mjcf_text)
    collision = []
    spheres = 0
    hulls = 0
    for tag in geoms:
        if 'contype="0"' in tag:
            continue
        collision.append(tag)
        if 'type="sphere"' in tag:
            spheres += 1
        elif 'type="mesh"' in tag:
            hulls += 1
    return {
        "n_collision_geoms": len(collision),
        "n_sphere_collision": spheres,
        "n_mesh_collision": hulls,
    }


def pad_stl_path(mesh_dir: Path, prefix: str, finger: str, revision: str) -> Path:
    if revision == "hand2_beta2":
        return mesh_dir / f"{prefix}_{finger}_tip_sensor_frame.STL"
    return mesh_dir / f"{prefix}_{finger}_tip.STL"


def _site_for_finger(sites: list[dict], prefix: str, finger: str) -> np.ndarray:
    name = f"{prefix}_{finger}_tip"
    for site in sites:
        if site["name"] == name:
            return np.asarray(site["pos"], dtype=np.float64)
    raise ValueError(f"missing fingertip site {name}")


def generate(
    side: str = "right",
    *,
    root: Path | None = None,
    revision: str | None = None,
    variant: str = "pad_spheres",
) -> dict[str, Any]:
    if variant not in {"pad_spheres", "simplified"}:
        raise ValueError(f"unknown variant {variant!r}")
    root = root or DEFAULT_UPSTREAM
    spec = load_hand_spec()
    rev = revision or spec.sim_model_revision
    src = official_mjcf(side, root=root, revision=rev)
    if not src.is_file():
        raise FileNotFoundError(f"official MJCF missing: {src}")
    text = src.read_text(encoding="utf-8")
    condim = int(spec.raw.get("derived_contact_condim", 4))
    mesh_dir = root / revision_body_path(rev, spec.raw) / "meshes" / side
    prefix = "r" if side == "right" else "l"
    sites = parse_sites(src)
    report_fingers: dict[str, Any] = {}
    for finger in PAD_FINGERS:
        stl = pad_stl_path(mesh_dir, prefix, finger, rev)
        if not stl.is_file():
            raise FileNotFoundError(
                f"pad mesh not found at {stl}. Cannot invent a sphere radius. "
                "Run scripts/bootstrap_resources.sh (git lfs)."
            )
        site = _site_for_finger(sites, prefix, finger)
        spheres = fit_pad_spheres(read_stl_vertices(stl), site_m=site)
        signed = site_to_nearest_sphere_surface_m(site, spheres)
        report_fingers[finger] = {
            "stl": stl.name,
            "site_m": site.tolist(),
            "spheres": [{"pos_m": pos, "radius_m": radius} for pos, radius in spheres],
            "site_to_surface_signed_m": signed,
            "site_to_surface_abs_m": abs(signed),
        }
        if rev == "hand2_beta2":
            mesh_name = f"{prefix}_{finger}_tip_sensor_frame"
            text = disable_group2_mesh_collision(text, mesh_name)
            if variant == "simplified":
                text = disable_group2_mesh_collision(text, f"{prefix}_{finger}_distal")
            body = f"{prefix}_{finger}_tip_sensor_frame"
            text = inject_into_body(text, body, sphere_geoms_xml(mesh_name, spheres, condim=condim))
        else:
            # Beta 1: pads do not collide. Overlay spheres at the distal body via the
            # fingertip site (same frame as *_tip.STL). Do not disable distal hulls
            # unless simplified — then the spheres become the only tip contact.
            site_name = f"{prefix}_{finger}_tip"
            if variant == "simplified":
                text = disable_group2_mesh_collision(text, f"{prefix}_{finger}_distal")
            geoms = sphere_geoms_xml(site_name, spheres, condim=condim)
            pattern = rf'(<site name="{re.escape(site_name)}"[^/]*/>)'
            text, n = re.subn(pattern, rf"\1{geoms}", text, count=1)
            if n != 1:
                raise ValueError(f"failed to overlay spheres at {site_name}")
    counts = count_collision_geoms(text)
    if counts["n_collision_geoms"] > MAX_COLLISION_GEOMS:
        raise ValueError(
            f"collision geom count {counts['n_collision_geoms']} exceeds budget {MAX_COLLISION_GEOMS}"
        )
    header = (
        f"<!-- GENERATED FROM {src.as_posix()} ({rev}/{variant}). Do not edit. "
        "Official file is untouched. Pad spheres replace official convex-hull pad "
        "collision. Mesh-fit only — not E1/E2, not live soft-pad radius. -->\n"
    )
    text = re.sub(r'meshdir="[^"]+"', f'meshdir="{mesh_dir.as_posix()}"', text)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / f"{side}_{rev}_{variant}.xml"
    out.write_text(header + text, encoding="utf-8")
    report = {
        "side": side,
        "revision": rev,
        "variant": variant,
        "source_mjcf": str(src),
        "output": str(out),
        "n_spheres_per_pad": N_SPHERES,
        "radius_quantile": RADIUS_QUANTILE,
        "note": "radii are official-mesh fits, not measured live-pad geometry",
        "fingers": report_fingers,
        "collision": counts,
        "max_site_to_surface_abs_m": max(item["site_to_surface_abs_m"] for item in report_fingers.values()),
        "gate_site_to_surface_m": MAX_SITE_TO_SURFACE_M,
        "site_gate_ok": max(item["site_to_surface_abs_m"] for item in report_fingers.values())
        <= MAX_SITE_TO_SURFACE_M,
    }
    report_path = DERIVED / f"{side}_{rev}_{variant}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def generate_default_set(root: Path | None = None) -> list[dict[str, Any]]:
    reports = []
    for side in ("right", "left"):
        for variant in ("pad_spheres", "simplified"):
            reports.append(generate(side, root=root, variant=variant))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="both", choices=["left", "right", "both"])
    parser.add_argument("--variant", default="all", choices=["pad_spheres", "simplified", "all"])
    args = parser.parse_args()
    sides = ("left", "right") if args.side == "both" else (args.side,)
    variants = ("pad_spheres", "simplified") if args.variant == "all" else (args.variant,)
    for side in sides:
        for variant in variants:
            report = generate(side, variant=variant)
            print(json.dumps({k: report[k] for k in ("output", "collision", "max_site_to_surface_abs_m", "site_gate_ok")}))


if __name__ == "__main__":
    main()
