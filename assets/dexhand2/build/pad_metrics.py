"""Fingertip site → nearest colliding geom distance (official vs pad-sphere derived).

Official Hand 2 MJCF collides the distal convex hull, not the pad STL. Derived
models add palmar spheres. PHASE 1 acceptance: derived pad distance should be
smaller than the official hull distance; target < 2 mm on the derived pads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from assets.dexhand2.build.gen_derived import DERIVED, generate
from assets.dexhand2.build.ingest_official import official_mjcf, parse_sites
from interface.schema import REPO_ROOT

TIP_SUFFIX = "_tip"
PAD_TOKEN = "_pad_"


def _tip_sites(side: str, mjcf: Path) -> list[str]:
    prefix = "r_" if side == "right" else "l_"
    names = []
    for site in parse_sites(mjcf):
        name = site["name"]
        if name.startswith(prefix) and name.endswith(TIP_SUFFIX):
            names.append(name)
    return names


def _geom_is_colliding(model, gid: int) -> bool:
    return int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0


def site_collision_distances(model, data, site_name: str) -> dict[str, Any]:
    """Distances (m) from a site to colliding geoms on the same body."""
    import mujoco

    sid = int(model.site(site_name).id)
    bid = int(model.site_bodyid[sid])
    p = np.asarray(data.site_xpos[sid], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != bid:
            continue
        if not _geom_is_colliding(model, gid):
            continue
        name = model.geom(gid).name
        gpos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
        gtype = int(model.geom_type[gid])
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius = float(model.geom_size[gid][0])
            dist_m = abs(float(np.linalg.norm(p - gpos)) - radius)
            kind = "sphere"
        else:
            dist_m = float(np.linalg.norm(p - gpos))
            kind = "other"
        rows.append(
            {
                "geom": name,
                "kind": kind,
                "dist_m": dist_m,
                "is_pad_sphere": PAD_TOKEN in name,
            }
        )
    pads = [r for r in rows if r["is_pad_sphere"]]
    hulls = [r for r in rows if not r["is_pad_sphere"]]
    nearest_pad = min((r["dist_m"] for r in pads), default=None)
    nearest_hull = min((r["dist_m"] for r in hulls), default=None)
    nearest = min((r["dist_m"] for r in rows), default=None)
    return {
        "site": site_name,
        "n_colliding_geoms": len(rows),
        "n_pad_spheres": len(pads),
        "nearest_m": nearest,
        "nearest_pad_sphere_m": nearest_pad,
        "nearest_non_pad_m": nearest_hull,
        "geoms": rows,
    }


def measure_side(side: str = "right") -> dict[str, Any]:
    import mujoco

    official = official_mjcf(side)
    if not official.is_file():
        raise FileNotFoundError(official)
    derived_path = generate(side)
    off_model = mujoco.MjModel.from_xml_path(official.as_posix())
    off_data = mujoco.MjData(off_model)
    mujoco.mj_forward(off_model, off_data)
    der_model = mujoco.MjModel.from_xml_path(derived_path.as_posix())
    der_data = mujoco.MjData(der_model)
    mujoco.mj_forward(der_model, der_data)

    sites = _tip_sites(side, official)
    official_rows = [site_collision_distances(off_model, off_data, s) for s in sites]
    derived_rows = [site_collision_distances(der_model, der_data, s) for s in sites]
    off_nearest = [r["nearest_m"] for r in official_rows if r["nearest_m"] is not None]
    der_pad = [r["nearest_pad_sphere_m"] for r in derived_rows if r["nearest_pad_sphere_m"] is not None]
    return {
        "side": side,
        "official_mjcf": official.as_posix(),
        "derived_mjcf": derived_path.as_posix(),
        "n_tip_sites": len(sites),
        "official": official_rows,
        "derived": derived_rows,
        "official_mean_nearest_m": float(np.mean(off_nearest)) if off_nearest else None,
        "derived_mean_pad_sphere_m": float(np.mean(der_pad)) if der_pad else None,
        "derived_max_pad_sphere_m": float(np.max(der_pad)) if der_pad else None,
        "derived_pad_under_2mm": bool(der_pad) and float(np.max(der_pad)) < 0.002,
        "policy_eval_forbidden": True,
        "note": (
            "Pad-sphere distance is mesh geometry of *_tip.STL, not the live soft pad. "
            "Official nearest geom is the distal hull (pads not colliding)."
        ),
    }


def measure_both() -> dict[str, Any]:
    sides = {}
    for side in ("right", "left"):
        try:
            sides[side] = measure_side(side)
        except FileNotFoundError as exc:
            sides[side] = {"status": "missing", "error": str(exc)}
    return {
        "kind": "phase1_pad_collision_baseline",
        "sides": sides,
        "derived_dir": DERIVED.as_posix(),
        "repo": REPO_ROOT.as_posix(),
    }


def write_report(out_json: Path, out_md: Path | None = None) -> dict[str, Any]:
    report = measure_both()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1000:.2f} mm"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PHASE 1 baseline — fingertip pad collision",
        "",
        "Generated by `python -m assets.dexhand2.build.pad_metrics`.",
        "Official files are not modified. Distances are **mesh** geometry, not live soft pads.",
        "",
        "| side | official mean nearest | derived mean pad sphere | derived max pad | < 2 mm |",
        "|---|---|---|---|---|",
    ]
    for side, block in report.get("sides", {}).items():
        if block.get("status") == "missing":
            lines.append(f"| {side} | missing | missing | missing | — |")
            continue
        lines.append(
            "| {side} | {off} | {pad} | {mx} | {ok} |".format(
                side=side,
                off=_fmt(block.get("official_mean_nearest_m")),
                pad=_fmt(block.get("derived_mean_pad_sphere_m")),
                mx=_fmt(block.get("derived_max_pad_sphere_m")),
                ok="yes" if block.get("derived_pad_under_2mm") else "no",
            )
        )
    lines.extend(
        [
            "",
            "Target: derived pad-sphere surface within 2 mm of the fingertip site.",
            "Official should be farther because pads are not collision geometry.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default="eval/report/generated/phase1_baseline.json",
    )
    parser.add_argument(
        "--md",
        default="docs/reports/PHASE_1_baseline.md",
    )
    args = parser.parse_args()
    report = write_report(Path(args.json), Path(args.md) if args.md else None)
    print(json.dumps({k: report["sides"].get(k, {}).get("derived_max_pad_sphere_m") for k in ("left", "right")}))


if __name__ == "__main__":
    main()
