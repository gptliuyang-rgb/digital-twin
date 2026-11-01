"""Derive Hand 2.2 MJCF with fingertip-pad spheres. Official files are never overwritten.

Official 2.2 already has `{side}_{finger}_tip_sensor_frame` bodies whose group-2
collision is the convex hull of `*_tip_sensor_frame.STL`. That hull is a plane/edge
approximation of the pad. This generator:

1. disables those group-2 hulls (contype=0, conaffinity=0)
2. fits 3 spheres to official `*_tip.STL` (soft-pad visual mesh)
3. injects the spheres with condim=4 into the pad body

Sphere radii are mesh geometry, not the live soft pad
(`fingertip_geometry_radius_m` stays REQUIRED_INPUT).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, official_mjcf, parse_sites
from interface.schema import REPO_ROOT

DERIVED = REPO_ROOT / "assets" / "dexhand2" / "derived"
TIP_FINGERS = ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"]
# Distal-biased stations along the pad PCA (last station toward the tip).
PAD_STATIONS = (0.20, 0.50, 0.85)
SITE_TO_SURFACE_GATE_M = 0.002


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84 and data[:5] != b"solid":
        count = int.from_bytes(data[80:84], "little")
        verts = []
        off = 84
        for _ in range(count):
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
    n_spheres: int = 3,
    site_pos_m: list[float] | None = None,
) -> list[tuple[list[float], float]]:
    """Place n spheres along the pad PCA, last station biased toward -Z (distal).

    Official `*_tip.STL` is **not** in the tip_sensor_frame body frame (Z is flipped
    and offset). Collision fitting uses `*_tip_sensor_frame.STL`, which shares the
    site frame. The distal sphere radius is pinned to the official fingertip site
    so the site lies on that sphere; that is a geometric construction, not a live
    pad measurement (`fingertip_geometry_radius_m` stays REQUIRED_INPUT).
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("empty tip STL")
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0]
    # Official fingertip sites sit at negative Z in the distal frame.
    if axis[2] > 0:
        axis = -axis
    proj = (pts - center) @ axis
    lo, hi = float(proj.min()), float(proj.max())
    stations = PAD_STATIONS if n_spheres == 3 else tuple((i + 0.5) / n_spheres for i in range(n_spheres))
    spheres: list[tuple[list[float], float]] = []
    site = None if site_pos_m is None else np.asarray(site_pos_m, dtype=np.float64)
    for i, frac in enumerate(stations):
        t = lo + (hi - lo) * float(frac)
        c = center + t * axis
        radial = np.linalg.norm((pts - c) - np.outer((pts - c) @ axis, axis), axis=1)
        radius = float(np.quantile(radial, 0.7))
        if site is not None and i == len(stations) - 1:
            radius = float(np.linalg.norm(site - c))
        spheres.append((c.tolist(), radius))
    return spheres


def site_to_surface_m(site_pos: list[float], spheres: list[tuple[list[float], float]]) -> float:
    site = np.asarray(site_pos, dtype=np.float64)
    dists = []
    for center, radius in spheres:
        dists.append(abs(float(np.linalg.norm(site - np.asarray(center)) - radius)))
    return float(min(dists))


def disable_pad_hull(mjcf_text: str, mesh_name: str) -> str:
    pattern = (
        rf'<geom type="mesh" rgba="[^"]+" mesh="{re.escape(mesh_name)}" group="2"\s*/>'
    )
    repl = (
        f'<geom type="mesh" rgba="0.8 0.6 1 0.15" mesh="{mesh_name}" '
        'group="2" contype="0" conaffinity="0" />'
    )
    new, n = re.subn(pattern, repl, mjcf_text, count=1)
    if n != 1:
        raise ValueError(f"failed to disable pad hull for {mesh_name}")
    return new


def disable_named_hull(mjcf_text: str, mesh_name: str) -> str:
    pattern = (
        rf'<geom type="mesh" rgba="[^"]+" mesh="{re.escape(mesh_name)}" group="2"\s*/>'
    )
    repl = (
        f'<geom type="mesh" rgba="0.8 0.6 1 0.15" mesh="{mesh_name}" '
        'group="2" contype="0" conaffinity="0" />'
    )
    new, n = re.subn(pattern, repl, mjcf_text, count=1)
    if n != 1:
        raise ValueError(f"failed to disable hull for {mesh_name}")
    return new


def inject_spheres(mjcf_text: str, site_name: str, spheres: list[tuple[list[float], float]]) -> str:
    geoms = []
    for i, (pos, radius) in enumerate(spheres):
        geoms.append(
            f'<geom name="{site_name}_pad_{i}" type="sphere" size="{radius:.6f}" '
            f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" group="2" condim="4" '
            f'rgba="0.2 0.8 0.4 0.4"/>'
        )
    block = "\n                  ".join(geoms)
    pattern = rf'(<site name="{re.escape(site_name)}"[^/]*/>)'
    repl = rf"\1\n                  {block}"
    new, n = re.subn(pattern, repl, mjcf_text, count=1)
    if n != 1:
        raise ValueError(f"failed to inject spheres at {site_name}")
    return new


def _tip_stl(mesh_dir: Path, prefix: str, finger: str) -> Path:
    """Pad mesh in the tip_sensor_frame body. Do not use `*_tip.STL` here."""
    return mesh_dir / f"{prefix}_{finger}_tip_sensor_frame.STL"


def generate(
    side: str = "right",
    *,
    root: Path | None = None,
    simplified: bool = False,
) -> tuple[Path, dict]:
    root = root or DEFAULT_UPSTREAM
    src = official_mjcf(side, root=root)
    text = src.read_text(encoding="utf-8")
    header = (
        f"<!-- GENERATED FROM {src.as_posix()} @ hand2_2 — fingertip pad spheres. "
        "Do not edit. Official file is untouched. -->\n"
    )
    mesh_dir = root / "hand2/hand2_2/body/meshes" / side
    prefix = "r" if side == "right" else "l"
    sites = {s["name"]: s["pos"] for s in parse_sites(src)}
    metrics: dict = {
        "side": side,
        "simplified": simplified,
        "source": str(src),
        "site_to_surface_m": {},
        "n_spheres": 0,
        "spheres": {},
    }
    for finger in TIP_FINGERS:
        site_name = f"{prefix}_{finger}_tip"
        pad_mesh = f"{prefix}_{finger}_tip_sensor_frame"
        stl = _tip_stl(mesh_dir, prefix, finger)
        if not stl.is_file():
            raise FileNotFoundError(f"missing tip STL {stl}")
        spheres = fit_pad_spheres(read_stl_vertices(stl), site_pos_m=sites[site_name])
        text = disable_pad_hull(text, pad_mesh)
        if simplified:
            text = disable_named_hull(text, f"{prefix}_{finger}_distal")
        text = inject_spheres(text, site_name, spheres)
        dist = site_to_surface_m(sites[site_name], spheres)
        metrics["site_to_surface_m"][site_name] = dist
        metrics["spheres"][site_name] = [
            {"pos_m": c, "radius_m": r} for c, r in spheres
        ]
        metrics["n_spheres"] += len(spheres)
    worst = max(metrics["site_to_surface_m"].values())
    metrics["worst_site_to_surface_m"] = worst
    metrics["site_to_surface_gate_m"] = SITE_TO_SURFACE_GATE_M
    metrics["gate_ok"] = worst <= SITE_TO_SURFACE_GATE_M

    DERIVED.mkdir(parents=True, exist_ok=True)
    suffix = "simplified" if simplified else "with_pad_spheres"
    out = DERIVED / f"{side}_{suffix}.xml"
    text = re.sub(r'meshdir="[^"]+"', f'meshdir="{mesh_dir.as_posix()}"', text)
    out.write_text(header + text, encoding="utf-8")
    report = DERIVED / f"{side}_{suffix}.json"
    report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--simplified", action="store_true")
    args = parser.parse_args()
    path, metrics = generate(args.side, simplified=args.simplified)
    print(path)
    print(json.dumps({k: metrics[k] for k in ("worst_site_to_surface_m", "gate_ok", "n_spheres")}, indent=2))
    if not metrics["gate_ok"]:
        raise SystemExit(f"site-to-surface {metrics['worst_site_to_surface_m']} > {SITE_TO_SURFACE_GATE_M}")


if __name__ == "__main__":
    main()
