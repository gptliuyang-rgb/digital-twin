"""Synthetic QR scan-envelope heatmap. No Isaac renderer required.

A pinhole camera looks at a physical QR of known size. Success is a real decode
(OpenCV or pyzbar), gated by the same distance / incidence / speed rules as
`sim.qr_scanner.simulate_scan`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sim.qr_scanner import ScanSpec, decode_image, make_qr_png


def _project_qr(
    qr_rgb: np.ndarray,
    *,
    distance_m: float,
    incidence_rad: float,
    qr_size_m: float,
    fx_px: float,
    image_wh: tuple[int, int],
) -> np.ndarray:
    """Warp a QR onto a camera image under a pinhole + out-of-plane rotation."""
    w, h = image_wh
    canvas = np.full((h, w, 3), 180, dtype=np.uint8)
    apparent_px = fx_px * qr_size_m / max(distance_m, 1e-3)
    # Foreshorten along one axis by cos(incidence).
    width_px = max(8, int(round(apparent_px)))
    height_px = max(8, int(round(apparent_px * max(np.cos(incidence_rad), 0.05))))
    qr = Image.fromarray(qr_rgb).resize((width_px, height_px), Image.Resampling.NEAREST)
    arr = np.asarray(qr.convert("RGB"))
    y0 = (h - height_px) // 2
    x0 = (w - width_px) // 2
    y1, x1 = y0 + height_px, x0 + width_px
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return canvas
    canvas[y0:y1, x0:x1] = arr
    return canvas


def scan_envelope(
    payload: str = "BOX-DT-001",
    distances_m: np.ndarray | None = None,
    angles_deg: np.ndarray | None = None,
    qr_size_m: float = 0.04,
    spec: ScanSpec | None = None,
) -> dict:
    spec = spec or ScanSpec()
    distances_m = np.asarray(distances_m if distances_m is not None else np.linspace(0.03, 0.40, 12))
    angles_deg = np.asarray(angles_deg if angles_deg is not None else np.linspace(0.0, 70.0, 8))
    qr = np.asarray(make_qr_png(payload, box_size=12, border=4))
    fx = 600.0
    table = []
    n_ok = 0
    n_tot = 0
    for d in distances_m:
        row = []
        for ang in angles_deg:
            n_tot += 1
            theta = float(np.deg2rad(ang))
            rgb = _project_qr(
                qr,
                distance_m=float(d),
                incidence_rad=theta,
                qr_size_m=qr_size_m,
                fx_px=fx,
                image_wh=(640, 480),
            )
            geom = spec.d_min_m <= d <= spec.d_max_m and theta <= spec.theta_max_rad
            decoded = decode_image(rgb) if geom else None
            ok = geom and decoded == payload
            if ok:
                n_ok += 1
            row.append(
                {
                    "distance_m": float(d),
                    "angle_deg": float(ang),
                    "geometry_ok": bool(geom),
                    "decode_ok": decoded == payload,
                    "success": bool(ok),
                    "decoded": decoded,
                }
            )
        table.append(row)
    return {
        "payload": payload,
        "qr_size_m": qr_size_m,
        "distances_m": distances_m.tolist(),
        "angles_deg": angles_deg.tolist(),
        "cells": table,
        "success_rate": n_ok / n_tot if n_tot else 0.0,
        "n_ok": n_ok,
        "n_tot": n_tot,
        "note": "Synthetic pinhole projection, not RTX. Use to size the IBVS capture region.",
    }


def heatmap_markdown(report: dict) -> str:
    angles = report["angles_deg"]
    header = "| d\\θ deg | " + " | ".join(f"{a:.0f}" for a in angles) + " |"
    sep = "|---|" + "|".join("---" for _ in angles) + "|"
    lines = ["# QR scan envelope (synthetic)", "", header, sep]
    for row, d in zip(report["cells"], report["distances_m"], strict=True):
        cells = ["✓" if c["success"] else "·" for c in row]
        lines.append(f"| {d:.2f} m | " + " | ".join(cells) + " |")
    lines += ["", f"success_rate={report['success_rate']:.3f}  ({report['n_ok']}/{report['n_tot']})", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/report/generated/qr_envelope.json")
    parser.add_argument("--md", default="eval/report/generated/qr_envelope.md")
    args = parser.parse_args()
    report = scan_envelope()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.md).write_text(heatmap_markdown(report), encoding="utf-8")
    print(json.dumps({"success_rate": report["success_rate"], "n_ok": report["n_ok"], "n_tot": report["n_tot"]}))


if __name__ == "__main__":
    main()
