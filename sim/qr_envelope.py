"""QR scan envelope: distance × incidence geometry heatmap (sim-only).

Decode cells are optional and require a rendered gun camera. Geometry uses
``simulate_scan`` with a dummy image so the envelope does not depend on GL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from interface.schema import REPO_ROOT
from sim.qr_scanner import ScanSpec, simulate_scan


def geometry_envelope(
    *,
    distances_m: np.ndarray | None = None,
    incidence_deg: np.ndarray | None = None,
    spec: ScanSpec | None = None,
) -> dict[str, Any]:
    spec = spec or ScanSpec(d_min_m=0.05, d_max_m=0.35, theta_max_rad=np.deg2rad(55.0))
    distances_m = (
        np.asarray(distances_m, dtype=np.float64)
        if distances_m is not None
        else np.linspace(0.04, 0.45, 9)
    )
    incidence_deg = (
        np.asarray(incidence_deg, dtype=np.float64)
        if incidence_deg is not None
        else np.linspace(0.0, 70.0, 8)
    )
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    gun = np.zeros(3)
    gun_z = np.array([0.0, 0.0, 1.0])
    grid = []
    n_ok = 0
    for d in distances_m:
        row = []
        for ang in incidence_deg:
            theta = float(np.deg2rad(ang))
            # QR in front of the gun; tilt the plate normal for incidence.
            qr = np.array([0.0, 0.0, float(d)])
            qr_n = np.array([np.sin(theta), 0.0, -np.cos(theta)])
            scan = simulate_scan(
                dummy,
                gun_tcp_pos_m=gun,
                gun_tcp_z=gun_z,
                qr_pos_m=qr,
                qr_normal=qr_n,
                rel_speed_m_s=0.0,
                spec=spec,
            )
            cell = {
                "distance_m": float(d),
                "incidence_deg": float(ang),
                "geometry_ok": bool(scan.geometry_ok),
                "decode_ok": None,
            }
            n_ok += int(scan.geometry_ok)
            row.append(cell)
            grid.append(cell)
        del row
    n = int(len(distances_m) * len(incidence_deg))
    return {
        "kind": "qr_geometry_envelope",
        "distances_m": distances_m.tolist(),
        "incidence_deg": incidence_deg.tolist(),
        "d_min_m": spec.d_min_m,
        "d_max_m": spec.d_max_m,
        "theta_max_deg": float(np.rad2deg(spec.theta_max_rad)),
        "n_cells": n,
        "n_geometry_ok": n_ok,
        "cells": grid,
        "policy_eval_forbidden": True,
        "note": (
            "Geometry-only envelope (dummy RGB, no decode). "
            "Industrial scan_decode_ok is attempted separately when GL works."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/report/generated/qr_envelope.json")
    args = parser.parse_args()
    report = geometry_envelope()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n_cells": report["n_cells"], "n_geometry_ok": report["n_geometry_ok"]}, indent=2))


if __name__ == "__main__":
    main()
