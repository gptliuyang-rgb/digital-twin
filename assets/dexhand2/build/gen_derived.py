"""Derive a Hand 2 MJCF with fingertip-pad spheres. Official files are never overwritten.

Sphere radii are fitted from official `*_tip.STL` when present. Clusters are then
translated onto the distal `*_tip` site (see `pad_inject.py`). That is mesh
geometry, not the live soft pad. When the spec overlay has E1/E2 numbers, pad
geoms also get friction/solref; the live spec stays REQUIRED_INPUT until hardware
CSVs are accepted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, official_mjcf
from assets.dexhand2.build.pad_inject import (
    fit_pad_spheres,
    inject_pads_for_side,
    inject_spheres,
    official_mesh_dir,
    read_stl_vertices,
)
from interface.schema import REPO_ROOT

DERIVED = REPO_ROOT / "assets" / "dexhand2" / "derived"
TIP_FINGERS = ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"]

# Re-exports for callers that imported these from gen_derived.
__all__ = [
    "DERIVED",
    "TIP_FINGERS",
    "fit_pad_spheres",
    "generate",
    "inject_spheres",
    "read_stl_vertices",
]


def generate(side: str = "right", root: Path | None = None, spec_path: Path | None = None) -> Path:
    from interface.schema import load_hand_spec

    root = root or DEFAULT_UPSTREAM
    src = official_mjcf(side, root=root)
    text = src.read_text(encoding="utf-8")
    spec = load_hand_spec(spec_path) if spec_path is not None else load_hand_spec()
    header = (
        f"<!-- GENERATED FROM {src.as_posix()} — fingertip pad spheres, site-aligned. "
        "Do not edit. Official file is untouched. -->\n"
    )
    mesh_dir = official_mesh_dir(side, root=root)
    text, n_pads = inject_pads_for_side(text, side, mesh_dir=mesh_dir, spec_raw=spec.raw)
    if n_pads != 5:
        raise ValueError(f"{side} pad inject expected 5 fingertip sites, got {n_pads}")
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / f"{side}_with_pad_spheres.xml"
    text = re.sub(r'meshdir="[^"]+"', f'meshdir="{(mesh_dir).as_posix()}"', text)
    out.write_text(header + text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--spec", default="", help="Optional overlay spec with E1/E2 numbers")
    args = parser.parse_args()
    path = generate(args.side, spec_path=Path(args.spec) if args.spec else None)
    print(path)


if __name__ == "__main__":
    main()
