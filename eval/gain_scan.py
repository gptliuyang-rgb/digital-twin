"""3×3 kp/kv scale grid from official gen-1 sim gains. Not Hand 2 system-id.

When MuJoCo and the derived MIT MJCF exist, each cell runs a hand-only hold
and records joint-velocity RMS. Grasp-success is always JSON null (ADR-004).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from interface.schema import REPO_ROOT, HandSpec, load_hand_spec
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key

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


def run_hold_scan(
    *,
    seconds: float = 1.0,
    side: str = "right",
    simplified: bool = True,
) -> dict:
    """Hold q=0 on each of 9 gain cells. Never reports grasp success."""
    cells = gain_cells()
    spec = load_hand_spec()
    uncal = spec.raw.get("hardware_kp") == "REQUIRED_INPUT"
    report: dict = {
        "n_cells": len(cells),
        "seconds": seconds,
        "side": side,
        "simplified": simplified,
        "grasp_success_rate": None,
        "gains_status": spec.raw.get("sim_gains_status"),
        "uncalibrated": uncal,
        "cells": [],
    }
    try:
        import mujoco  # noqa: F401
    except ImportError:
        report["status"] = "skipped_no_mujoco"
        refuse_grasp_success_key(report)
        return report
    from assets.dexhand2.build.ingest_official import official_mjcf
    from sim.mujoco_env.hand_env import HandOnlyMujocoEnv

    if not official_mjcf(side).is_file():
        report["status"] = "skipped_no_official_mjcf"
        refuse_grasp_success_key(report)
        return report

    env = HandOnlyMujocoEnv(side=side, derived=True, simplified=simplified)
    rms_vals: list[float] = []
    for cell in cells:
        env.controller.kp = np.asarray(cell["kp"], dtype=np.float64)
        env.controller.kd = np.asarray(cell["kv"], dtype=np.float64)
        env.reset()
        stats = env.hold(seconds)
        row = {
            "kp_scale": cell["kp_scale"],
            "kv_scale": cell["kv_scale"],
            "qvel_rms_rad_s": stats["qvel_rms_rad_s"],
        }
        report["cells"].append(row)
        rms_vals.append(float(stats["qvel_rms_rad_s"]))
    report["qvel_rms_rad_s_min"] = float(min(rms_vals))
    report["qvel_rms_rad_s_max"] = float(max(rms_vals))
    report["qvel_rms_rad_s_range"] = float(max(rms_vals) - min(rms_vals))
    report["xml"] = str(env.xml_path)
    report["status"] = "uncalibrated_gain_scan"
    refuse_grasp_success_key(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--out", default="eval/report/generated/gain_scan.json")
    args = parser.parse_args()
    report = run_hold_scan(seconds=args.seconds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in ("status", "n_cells", "grasp_success_rate") if k in report}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
