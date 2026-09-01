"""Shared fingertip-pad sphere inject for derived MJCF and the combined twin.

Official Hand 2 collides the distal convex hull, not ``*_tip.STL``. This module
fits 2–3 spheres from the pad STL, then **translates** the cluster so it sits on
the distal ``*_tip`` site (sim-only geometric proxy).

The tip STL vertex origin is not the site frame (≈40 mm offset on Wuji Beta 1).
Without that translation, pad surfaces land ~23 mm from the site. After
centroid-to-site + palmar surface offset, the site lies on the sphere surface
(|center−site| ≈ radius), which is the PHASE 1 2 mm *frame* gate — not live
pad thickness (ADR-004).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, official_mjcf

# Palmar in Wuji distal bodies is −Y (finger points −Z, nail +Y-ish).
_PALMAR_BODY = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_SITE_RE = re.compile(r'<site name="([^"]+)" pos="([^"]+)"')


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


def fit_pad_spheres(vertices: np.ndarray, n_spheres: int = 3) -> list[tuple[list[float], float]]:
    """Place n spheres along the first PCA axis of the tip mesh, radius = rms radial size."""
    pts = np.asarray(vertices, dtype=np.float64)
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0]
    proj = (pts - center) @ axis
    lo, hi = proj.min(), proj.max()
    spheres = []
    for i in range(n_spheres):
        t = lo + (hi - lo) * (i + 0.5) / n_spheres
        c = center + t * axis
        radial = np.linalg.norm((pts - c) - np.outer((pts - c) @ axis, axis), axis=1)
        r = float(np.quantile(radial, 0.7))
        spheres.append((c.tolist(), r))
    return spheres


def tip_finger_stem(site_name: str, side: str) -> str | None:
    """``r_index_finger_tip`` → ``index_finger``. Never ``str.replace("r_", ...)``.

    Replacing ``r_`` on the full name ate the ``r`` in ``finger_tip`` and skipped
    index/middle/ring pads.
    """
    prefix = "r_" if side == "right" else "l_"
    if not site_name.startswith(prefix) or not site_name.endswith("_tip"):
        return None
    return site_name[len(prefix) : -len("_tip")]


def parse_sites_text(xml_text: str) -> list[dict[str, Any]]:
    sites = []
    for match in _SITE_RE.finditer(xml_text):
        xyz = [float(x) for x in match.group(2).split()]
        sites.append({"name": match.group(1), "pos": xyz})
    return sites


def align_spheres_to_site(
    spheres: list[tuple[list[float], float]],
    site_pos: np.ndarray | list[float],
    *,
    surface: bool = True,
) -> list[tuple[list[float], float]]:
    """Rigidly translate a fitted cluster into the distal site frame.

    ``surface=True`` then offsets one mean-radius along body palmar (−Y) so the
    site lies on the sphere surface (PHASE 1 2 mm frame gate).
    """
    if not spheres:
        return []
    centers = np.asarray([s[0] for s in spheres], dtype=np.float64)
    radii = np.asarray([s[1] for s in spheres], dtype=np.float64)
    site = np.asarray(site_pos, dtype=np.float64).reshape(3)
    centers = centers + (site - centers.mean(axis=0))
    if surface:
        centers = centers + _PALMAR_BODY * float(np.mean(radii))
    return [(c.tolist(), float(r)) for c, r in zip(centers, radii, strict=True)]


def spheres_for_tip_stl(
    stl: Path,
    site_pos: np.ndarray | list[float],
    *,
    n_spheres: int = 3,
    surface: bool = True,
) -> list[tuple[list[float], float]]:
    raw = fit_pad_spheres(read_stl_vertices(stl), n_spheres=n_spheres)
    return align_spheres_to_site(raw, site_pos, surface=surface)


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


def inject_pads_for_side(
    xml_text: str,
    side: str,
    *,
    mesh_dir: Path | None = None,
    spec_raw: dict[str, Any] | None = None,
    surface: bool = True,
) -> tuple[str, int]:
    """Inject site-aligned pad spheres for every ``{l|r}_*_tip`` site. Returns (xml, n_sites)."""
    prefix = "r" if side == "right" else "l"
    mesh_dir = mesh_dir or (DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/meshes" / side)
    n = 0
    for site in parse_sites_text(xml_text):
        stem = tip_finger_stem(site["name"], side)
        if stem is None:
            continue
        stl = mesh_dir / f"{prefix}_{stem}_tip.STL"
        if not stl.is_file():
            continue
        spheres = spheres_for_tip_stl(stl, site["pos"], surface=surface)
        try:
            xml_text = inject_spheres(xml_text, site["name"], spheres)
        except ValueError:
            continue
        n += 1
    if spec_raw:
        from hand.calibration.contact_mujoco import apply_contact_to_xml

        xml_text = apply_contact_to_xml(xml_text, spec_raw)
    return xml_text, n


def official_mesh_dir(side: str, root: Path | None = None) -> Path:
    return (root or DEFAULT_UPSTREAM) / "hand2/hand2_beta1/body/meshes" / side


def official_mjcf_path(side: str, *, with_mount: bool = False, root: Path | None = None) -> Path:
    return official_mjcf(side, with_mount=with_mount, root=root)
