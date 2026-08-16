"""3×3 kp/kv scale grid from official gen-1 sim gains. Not Hand 2 system-id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from interface.schema import REPO_ROOT, HandSpec, load_hand_spec

GAIN_SCAN_PATH = REPO_ROOT / "eval" / "configs" / "gain_scan.yaml"


def load_gain_scan_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load((path or GAIN_SCAN_PATH).read_text(encoding="utf-8"))


def _vec(mapping: dict, order: list[str]) -> np.ndarray:
    return np.array([float(mapping[name]) for name in order], dtype=np.float64)


def gain_cells(spec: HandSpec | None = None, path: Path | None = None) -> list[dict]:
    spec = spec or load_hand_spec()
    cfg = load_gain_scan_cfg(path)
    kp0 = _vec(spec.raw["sim_kp"], spec.joint_order)
    kv0 = _vec(spec.raw["sim_kv"], spec.joint_order)
    cells = []
    for ks in cfg["kp_scale"]:
        for vs in cfg["kv_scale"]:
            cells.append(
                {
                    "kp_scale": float(ks),
                    "kv_scale": float(vs),
                    "kp": (kp0 * float(ks)).tolist(),
                    "kv": (kv0 * float(vs)).tolist(),
                    "status": spec.raw.get("sim_gains_status"),
                }
            )
    if len(cells) != int(cfg["n_cells"]):
        raise ValueError(f"expected {cfg['n_cells']} cells, got {len(cells)}")
    return cells
