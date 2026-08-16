"""Derive Hand 2 MJCF variants. Official files are never overwritten.

Official gap (wuji-description integration docs):
  *_tip.STL ships but is not collision geometry; contact sits on the distal hull.

Those tip STLs are the distal bone mesh in a CAD frame (Z flipped). Pad spheres
are therefore fitted from the **distal** STL in the MJCF distal body frame, on
the palmar half of the fingertip — not from raw tip-STL coordinates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, official_mjcf, parse_sites
from interface.schema import REPO_ROOT

DERIVED = REPO_ROOT / "assets" / "dexhand2" / "derived"
FIT_CFG = Path(__file__).with_name("pad_fit.yaml")
FITTED_YAML = REPO_ROOT / "assets" / "dexhand2" / "meta" / "fitted_pad_spheres.yaml"
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def _load_fit_cfg() -> dict[str, Any]:
    return yaml.safe_load(FIT_CFG.read_text(encoding="utf-8"))


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84 and not data[:5].lower().startswith(b"solid"):
        n = int.from_bytes(data[80:84], "little")
        verts = []
        off = 84
        for _ in range(n):
            for k in range(3):
                base = off + 12 + k * 12
                verts.append(np.frombuffer(data[base : base + 12], dtype=np.float32))
            off += 50
        return np.vstack(verts).astype(np.float64)
    text = data.decode("utf-8", errors="ignore")
    nums = [[float(x) for x in line.split()[1:4]] for line in text.splitlines() if "vertex" in line]
    return np.asarray(nums, dtype=np.float64)


def quat_from_z_to(axis: np.ndarray) -> np.ndarray:
    """MuJoCo capsule default axis is +Z. Return wxyz rotating +Z onto `axis`."""
    z = np.array([0.0, 0.0, 1.0])
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c = float(np.clip(np.dot(z, a), -1.0, 1.0))
    if c > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    w = 1.0 + c
    xyz = np.cross(z, a)
    q = np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)
    return q / np.linalg.norm(q)


def fit_capsule(vertices: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(vertices, dtype=np.float64)
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0]
    proj = (pts - center) @ axis
    mid = center + axis * (0.5 * (proj.max() + proj.min()))
    half_m = 0.5 * float(proj.max() - proj.min())
    radial = np.linalg.norm((pts - mid) - np.outer((pts - mid) @ axis, axis), axis=1)
    radius_m = float(np.quantile(radial, 0.85))
    return {
        "pos_m": mid.tolist(),
        "radius_m": radius_m,
        "half_length_m": max(half_m - radius_m * 0.3, 1e-4),
        "quat_wxyz": quat_from_z_to(axis).tolist(),
    }


def fit_palmar_pad_spheres(vertices: np.ndarray, cfg: dict[str, Any]) -> list[tuple[list[float], float]]:
    """Fit n spheres to the palmar pulp of a distal-phalanx mesh (MJCF distal frame)."""
    pts = np.asarray(vertices, dtype=np.float64)
    n_spheres = int(cfg["n_spheres_per_finger"])
    z_thr = float(np.quantile(pts[:, 2], float(cfg["distal_z_quantile"])))
    far = pts[pts[:, 2] <= z_thr]
    if len(far) < 30:
        far = pts
    y_min, y_max = float(far[:, 1].min()), float(far[:, 1].max())
    palmar_negative = abs(y_min) >= abs(y_max)
    y_cut = float(np.quantile(far[:, 1], float(cfg["pulp_y_quantile"])))
    pulp = far[far[:, 1] <= y_cut] if palmar_negative else far[far[:, 1] >= y_cut]
    if len(pulp) < 20:
        pulp = far
    zs = pulp[:, 2]
    spheres: list[tuple[list[float], float]] = []
    min_r = float(cfg["min_radius_m"])
    rq = float(cfg["radius_quantile"])
    for i in range(n_spheres):
        lo = float(np.quantile(zs, i / n_spheres))
        hi = float(np.quantile(zs, (i + 1) / n_spheres))
        chunk = pulp[(zs >= lo) & (zs <= hi)]
        if len(chunk) < 8:
            chunk = pulp
        c = chunk.mean(axis=0)
        r = float(np.quantile(np.linalg.norm(chunk - c, axis=1), rq))
        spheres.append((c.tolist(), max(r, min_r)))
    spheres = _cover_palmar_vertex(pts, spheres, float(cfg["acceptance"]["palmar_vertex_to_sphere_surface_m"]))
    return spheres


def _cover_palmar_vertex(
    vertices: np.ndarray,
    spheres: list[tuple[list[float], float]],
    max_surface_m: float,
) -> list[tuple[list[float], float]]:
    """Expand the nearest sphere so the palmar-most distal vertex is on/inside it."""
    pts = np.asarray(vertices, dtype=np.float64)
    far = pts[pts[:, 2] <= np.quantile(pts[:, 2], 0.15)]
    palmar_negative = abs(float(far[:, 1].min())) >= abs(float(far[:, 1].max()))
    vertex = far[int(np.argmin(far[:, 1]))] if palmar_negative else far[int(np.argmax(far[:, 1]))]
    dists = [float(np.linalg.norm(vertex - np.asarray(c))) for c, _r in spheres]
    i = int(np.argmin(dists))
    c, r = spheres[i]
    need = dists[i]
    if need - r > max_surface_m:
        spheres[i] = (c, float(need))
    return spheres


def site_to_sphere_surface_m(site_pos: list[float], spheres: list[tuple[list[float], float]]) -> float:
    s = np.asarray(site_pos, dtype=np.float64)
    dists = []
    for c, r in spheres:
        d = float(np.linalg.norm(s - np.asarray(c)) - r)
        dists.append(abs(d) if d < 0 else d)  # inside counts as 0-ish; take |signed|
    # signed: negative = site inside sphere. Acceptance uses min unsigned distance to surface.
    signed = [float(np.linalg.norm(s - np.asarray(c)) - r) for c, r in spheres]
    return float(min(abs(x) for x in signed))


def palmar_vertex_to_sphere_surface_m(
    vertices: np.ndarray, spheres: list[tuple[list[float], float]]
) -> float:
    """Distance from the most palmar distal vertex to the nearest sphere surface."""
    pts = np.asarray(vertices, dtype=np.float64)
    far = pts[pts[:, 2] <= np.quantile(pts[:, 2], 0.15)]
    palmar_negative = abs(float(far[:, 1].min())) >= abs(float(far[:, 1].max()))
    vertex = far[int(np.argmin(far[:, 1]))] if palmar_negative else far[int(np.argmax(far[:, 1]))]
    signed = [float(np.linalg.norm(vertex - np.asarray(c)) - r) for c, r in spheres]
    return float(min(abs(x) for x in signed))


def inject_spheres(
    mjcf_text: str,
    site_name: str,
    spheres: list[tuple[list[float], float]],
    condim: int,
) -> str:
    geoms = []
    for i, (pos, radius) in enumerate(spheres):
        geoms.append(
            f'<geom name="{site_name}_pad_{i}" type="sphere" size="{radius:.6f}" '
            f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" group="2" condim="{condim}" '
            f'rgba="0.2 0.8 0.4 0.45"/>'
        )
    block = "\n              ".join(geoms)
    pattern = rf'(<site name="{re.escape(site_name)}"[^/]*/>)'
    new, n = re.subn(pattern, rf"\1\n              {block}", mjcf_text, count=1)
    if n != 1:
        raise ValueError(f"failed to inject spheres at {site_name}")
    return new


def disable_distal_hull(mjcf_text: str, prefix: str) -> str:
    for finger in FINGERS:
        mesh = f"{prefix}_{finger}_distal"

        def _disable(match: re.Match[str], mesh_name: str = mesh) -> str:
            tag = match.group(0)
            if "contype" in tag:
                return tag
            return tag[:-2] + ' contype="0" conaffinity="0"/>'

        mjcf_text, n = re.subn(
            rf'<geom type="mesh" rgba="[^"]+" mesh="{re.escape(mesh)}" group="2"\s*/>',
            _disable,
            mjcf_text,
            count=1,
        )
        if n != 1:
            raise ValueError(f"failed to disable distal hull collision for {mesh}")
    return mjcf_text


def convert_position_actuators_to_motor(mjcf_text: str) -> str:
    """Replace <position kp kv> with <motor> so the MIT law in the controller owns the gains."""

    def repl(match: re.Match[str]) -> str:
        name, joint, ctrl, frc = match.group(1), match.group(2), match.group(5), match.group(6)
        return (
            f'<motor name="{name}" joint="{joint}" ctrlrange="{ctrl}" forcerange="{frc}" '
            f'gear="1"/>'
        )

    new, n = re.subn(
        r'<position name="([^"]+)" joint="([^"]+)" kp="([^"]+)" kv="([^"]+)" '
        r'ctrlrange="([^"]+)" forcerange="([^"]+)"\s*/>',
        repl,
        mjcf_text,
    )
    if n != 20:
        raise ValueError(f"expected 20 position actuators to convert, got {n}")
    return new


def disable_non_wrist_mesh_collision(mjcf_text: str, prefix: str) -> str:
    """Simplified variant: only wrist hull + primitives collide."""

    def repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        mesh = match.group(1)
        if mesh == f"{prefix}_wrist":
            return tag
        if "contype" in tag:
            return tag
        return tag[:-2] + ' contype="0" conaffinity="0"/>'

    return re.sub(
        r'<geom type="mesh" rgba="[^"]+" mesh="([^"]+)" group="2"\s*/>',
        repl,
        mjcf_text,
    )


def inject_capsules_for_links(
    mjcf_text: str,
    prefix: str,
    mesh_dir: Path,
    skip: set[str],
) -> str:
    """Add one capsule per non-skipped mesh, parented at the same body as the visual mesh."""
    for stl in sorted(mesh_dir.glob(f"{prefix}_*.STL")):
        mesh_name = stl.stem
        if mesh_name in skip or mesh_name.endswith("_tip"):
            continue
        verts = read_stl_vertices(stl)
        cap = fit_capsule(verts)
        pos = cap["pos_m"]
        q = cap["quat_wxyz"]
        geom = (
            f'<geom name="{mesh_name}_capsule" type="capsule" '
            f'size="{cap["radius_m"]:.6f} {cap["half_length_m"]:.6f}" '
            f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
            f'group="2" condim="4" rgba="0.9 0.5 0.2 0.35"/>'
        )
        # Insert after the collision mesh geom of this link.
        pattern = rf'(<geom type="mesh" rgba="[^"]+" mesh="{re.escape(mesh_name)}"[^/]*/>)'
        mjcf_text, n = re.subn(pattern, rf"\1\n              {geom}", mjcf_text, count=1)
        if n != 1:
            # visual-only meshes (tips) may not have a group-2 geom
            continue
    return mjcf_text


def _rewrite_meshdir(text: str, mesh_dir: Path, out_path: Path) -> str:
    rel = Path(os_relpath(mesh_dir, out_path.parent))
    return re.sub(r'meshdir="[^"]+"', f'meshdir="{rel.as_posix()}"', text)


def os_relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target, start)


def generate(
    side: str = "right",
    root: Path | None = None,
    *,
    mit_motors: bool = True,
    simplified: bool = False,
) -> dict[str, Any]:
    cfg = _load_fit_cfg()
    root = root or DEFAULT_UPSTREAM
    src = official_mjcf(side, root=root)
    text = src.read_text(encoding="utf-8")
    mesh_dir = root / "hand2/hand2_beta1/body/meshes" / side
    prefix = "r" if side == "right" else "l"
    sites = parse_sites(src)
    per_finger: dict[str, Any] = {}
    for site in sites:
        raw = site["name"]
        if not raw.startswith(f"{prefix}_") or not raw.endswith("_tip"):
            raise ValueError(f"unexpected site name {raw}")
        finger = raw[len(prefix) + 1 : -4]  # strip '{p}_' and '_tip'
        distal_stl = mesh_dir / f"{prefix}_{finger}_distal.STL"
        if not distal_stl.is_file():
            raise FileNotFoundError(distal_stl)
        verts = read_stl_vertices(distal_stl)
        spheres = fit_palmar_pad_spheres(verts, cfg)
        per_finger[finger] = {
            "site": site["name"],
            "site_pos_m": site["pos"],
            "spheres": [{"pos_m": c, "radius_m": r} for c, r in spheres],
            "site_to_sphere_surface_m": site_to_sphere_surface_m(site["pos"], spheres),
            "palmar_vertex_to_sphere_surface_m": palmar_vertex_to_sphere_surface_m(verts, spheres),
            "official_site_to_distal_vertex_m": float(
                np.min(np.linalg.norm(verts - np.asarray(site["pos"]), axis=1))
            ),
        }
        text = inject_spheres(text, site["name"], spheres, int(cfg["condim"]))
    if cfg.get("disable_distal_hull_collision", True):
        text = disable_distal_hull(text, prefix)
    if simplified:
        text = disable_non_wrist_mesh_collision(text, prefix)
        skip = {f"{prefix}_wrist"} | {f"{prefix}_{f}_distal" for f in FINGERS}
        text = inject_capsules_for_links(text, prefix, mesh_dir, skip)
    if mit_motors:
        text = convert_position_actuators_to_motor(text)

    variant = "simplified" if simplified else "with_pad_spheres"
    if mit_motors:
        variant += "_mit"
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / f"{side}_{variant}.xml"
    header = (
        f"<!-- GENERATED FROM {src.as_posix()} @ pad_fit.yaml. "
        "Do not edit. Official file is untouched. Palmar pad spheres; "
        f"mit_motors={mit_motors} simplified={simplified}. -->\n"
    )
    out.write_text(header + _rewrite_meshdir(text, mesh_dir, out), encoding="utf-8")
    n_spheres = sum(len(v["spheres"]) for v in per_finger.values())
    n_capsules = len(re.findall(r'type="capsule"', text))
    report = {
        "side": side,
        "src": str(src),
        "out": str(out),
        "n_pad_spheres": n_spheres,
        "n_capsules": n_capsules,
        "mit_motors": mit_motors,
        "simplified": simplified,
        "fingers": per_finger,
        "gates": {
            "n_pad_spheres_ok": n_spheres <= int(cfg["acceptance"]["max_pad_spheres_per_hand"]),
            "palmar_fit_ok": all(
                v["palmar_vertex_to_sphere_surface_m"]
                <= float(cfg["acceptance"]["palmar_vertex_to_sphere_surface_m"])
                for v in per_finger.values()
            ),
        },
    }
    (DERIVED / f"{side}_{variant}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_fitted_yaml(reports: list[dict[str, Any]]) -> Path:
    payload = {
        "schema_version": "1.0",
        "source": "distal STL palmar-pulp fit (not live soft-pad geometry)",
        "note": (
            "fingertip_geometry_radius_m in dexhand2_spec.yaml stays REQUIRED_INPUT. "
            "These radii are collision-primitive sizes from the skeleton distal mesh."
        ),
        "hands": {},
    }
    for report in reports:
        payload["hands"][report["side"]] = {
            finger: {
                "spheres": data["spheres"],
                "site_to_sphere_surface_m": data["site_to_sphere_surface_m"],
                "palmar_vertex_to_sphere_surface_m": data["palmar_vertex_to_sphere_surface_m"],
                "official_site_to_distal_vertex_m": data["official_site_to_distal_vertex_m"],
            }
            for finger, data in report["fingers"].items()
        }
    FITTED_YAML.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return FITTED_YAML


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="both", choices=["left", "right", "both"])
    parser.add_argument("--no-mit", action="store_true")
    parser.add_argument("--simplified", action="store_true")
    args = parser.parse_args()
    sides = ["left", "right"] if args.side == "both" else [args.side]
    reports = []
    for side in sides:
        reports.append(generate(side, mit_motors=not args.no_mit, simplified=False))
        if args.simplified:
            reports.append(generate(side, mit_motors=not args.no_mit, simplified=True))
    path = write_fitted_yaml([r for r in reports if not r["simplified"]])
    print(json.dumps({"fitted_yaml": str(path), "reports": [r["out"] for r in reports]}, indent=2))


if __name__ == "__main__":
    main()
