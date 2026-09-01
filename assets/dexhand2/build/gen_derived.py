"""Derive a Hand 2 MJCF with fingertip-pad spheres. Official files are never overwritten.

Sphere radii are fitted from official `*_tip.STL` when present. That is mesh geometry,
not the live soft pad (still REQUIRED_INPUT).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from assets.dexhand2.build.ingest_official import (
    DEFAULT_UPSTREAM,
    official_mjcf,
    parse_sites,
    revision_body_path,
)
from interface.schema import REPO_ROOT, load_hand_spec

DERIVED = REPO_ROOT / "assets" / "dexhand2" / "derived"
TIP_FINGERS = ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"]


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    # Binary STL: 80-byte header + uint32 count + 50-byte triangles.
    if len(data) >= 84 and data[:5] != b"solid":
        n = int.from_bytes(data[80:84], "little")
        verts = []
        off = 84
        for _ in range(n):
            # skip normal (12) then 3 vertices (36) then attribute (2)
            for k in range(3):
                base = off + 12 + k * 12
                verts.append(np.frombuffer(data[base : base + 12], dtype=np.float32))
            off += 50
        return np.vstack(verts)
    # ASCII
    text = data.decode("utf-8", errors="ignore")
    nums = []
    for line in text.splitlines():
        if "vertex" in line:
            nums.append([float(x) for x in line.split()[1:4]])
    return np.asarray(nums, dtype=np.float64)


def fit_pad_spheres(vertices: np.ndarray, n_spheres: int = 3) -> list[tuple[list[float], float]]:
    """Place n spheres along the first PCA axis of the tip mesh, radius = rms radial size."""
    pts = np.asarray(vertices, dtype=np.float64)
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0]
    proj = (pts - center) @ axis
    lo, hi = proj.min(), proj.max()
    radii = []
    spheres = []
    for i in range(n_spheres):
        t = lo + (hi - lo) * (i + 0.5) / n_spheres
        c = center + t * axis
        radial = np.linalg.norm((pts - c) - np.outer((pts - c) @ axis, axis), axis=1)
        r = float(np.quantile(radial, 0.7))
        radii.append(r)
        spheres.append((c.tolist(), r))
    return spheres


def inject_spheres(mjcf_text: str, site_name: str, spheres: list[tuple[list[float], float]]) -> str:
    geoms = []
    for i, (pos, radius) in enumerate(spheres):
        geoms.append(
            f'<geom name="{site_name}_pad_{i}" type="sphere" size="{radius:.6f}" '
            f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" group="2" condim="4" '
            f'rgba="0.2 0.8 0.4 0.4"/>'
        )
    block = "\n              ".join(geoms)
    pattern = rf'(<site name="{re.escape(site_name)}"[^/]*/>)'
    repl = rf"\1\n              {block}"
    new, n = re.subn(pattern, repl, mjcf_text, count=1)
    if n != 1:
        raise ValueError(f"failed to inject spheres at {site_name}")
    return new


def generate(side: str = "right", root: Path | None = None, revision: str | None = None) -> Path:
    root = root or DEFAULT_UPSTREAM
    spec = load_hand_spec()
    rev = revision or spec.sim_model_revision
    src = official_mjcf(side, root=root, revision=rev)
    text = src.read_text(encoding="utf-8")
    header = (
        f"<!-- GENERATED FROM {src.as_posix()} ({rev}) — fingertip pad spheres. "
        "Do not edit. Official file is untouched. "
        "Beta 2 already collides official pad meshes (convex hull); "
        "these spheres are a simplified overlay, not a replacement for E1/E2. -->\n"
    )
    mesh_dir = root / revision_body_path(rev, spec.raw) / "meshes" / side
    prefix = "r" if side == "right" else "l"
    sites = parse_sites(src)
    for site in sites:
        finger = site["name"].replace(f"{prefix}_", "").replace("_tip", "")
        stl = mesh_dir / f"{prefix}_{finger}_tip.STL"
        if not stl.is_file():
            # pinky file is r_pinky_tip.STL; site is r_pinky_tip. Already handled.
            continue
        spheres = fit_pad_spheres(read_stl_vertices(stl))
        # Sites live in the distal frame. Tip STL is typically in the same distal frame
        # (Wuji ships tip meshes next to distal). If the STL origin differs, the
        # spheres will be wrong — PHASE 1 report must include site-to-sphere distance.
        text = inject_spheres(text, site["name"], spheres)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / f"{side}_with_pad_spheres.xml"
    # meshdir in official MJCF is relative; rewrite to absolute official meshes.
    text = re.sub(r'meshdir="[^"]+"', f'meshdir="{(mesh_dir).as_posix()}"', text)
    out.write_text(header + text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="right", choices=["left", "right"])
    args = parser.parse_args()
    path = generate(args.side)
    print(path)


if __name__ == "__main__":
    main()
