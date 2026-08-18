"""USD overlay: palmar pad spheres from fitted_pad_spheres.yaml.

Official Hand 2 USD is not overwritten. This file only emits a .usda layer that
adds sphere colliders under each distal Xform. Prim paths follow MJCF body names
(`{l|r}_{finger}_distal`), which the official USD uses as link prims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from assets.dexhand2.build.gen_derived import DERIVED, FINGERS, FITTED_YAML
from interface.schema import REPO_ROOT

USD_PRIM_MAP = REPO_ROOT / "assets" / "dexhand2" / "meta" / "usd_prim_map.yaml"


def load_fitted() -> dict[str, Any]:
    return yaml.safe_load(FITTED_YAML.read_text(encoding="utf-8"))


def load_prim_map() -> dict[str, Any]:
    return yaml.safe_load(USD_PRIM_MAP.read_text(encoding="utf-8"))


def distal_prim(side: str, finger: str, prim_map: dict[str, Any]) -> str:
    prefix = "r" if side == "right" else "l"
    template = prim_map["distal_prim_template"]
    return template.format(prefix=prefix, finger=finger)


def usda_for_hand(side: str, fitted: dict[str, Any] | None = None, prim_map: dict[str, Any] | None = None) -> str:
    fitted = fitted or load_fitted()
    prim_map = prim_map or load_prim_map()
    hand = fitted.get("hands", {}).get(side)
    if not hand:
        raise KeyError(f"fitted_pad_spheres.yaml has no '{side}' hand — will not mirror the other side")
    lines = [
        "#usda 1.0",
        "# GENERATED from assets/dexhand2/meta/fitted_pad_spheres.yaml. Official USD untouched.",
        "# These radii are skeleton-pulp collision primitives, not live soft-pad geometry.",
        "(",
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
    ]
    n = 0
    for finger in FINGERS:
        if finger not in hand:
            raise KeyError(f"missing finger {finger} on {side}")
        prim = distal_prim(side, finger, prim_map)
        lines.append(f'over "{prim}"')
        lines.append("{")
        for i, sph in enumerate(hand[finger]["spheres"]):
            pos = sph["pos_m"]
            r = float(sph["radius_m"])
            name = f"pad_{i}"
            lines.append(f'    def Sphere "{name}" (')
            lines.append('        prepend apiSchemas = ["PhysicsCollisionAPI"]')
            lines.append("    )")
            lines.append("    {")
            lines.append(f"        double radius = {r:.8f}")
            lines.append(f"        double3 xformOp:translate = ({pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.8f})")
            lines.append('        uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append("        bool physics:collisionEnabled = 1")
            lines.append("    }")
            n += 1
        lines.append("}")
        lines.append("")
    header = f"# n_pad_spheres={n} side={side}\n"
    return header + "\n".join(lines) + "\n"


def generate(side: str = "right") -> Path:
    DERIVED.mkdir(parents=True, exist_ok=True)
    text = usda_for_hand(side)
    out = DERIVED / f"{side}_pad_spheres.usda"
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="right", choices=["left", "right"])
    args = parser.parse_args()
    path = generate(args.side)
    print(path)


if __name__ == "__main__":
    main()
