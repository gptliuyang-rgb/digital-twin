"""L2 closed-loop entry. Contact-uncalibrated specs must not report success rates.

When MuJoCo and official assets are present this runs:
  - L2.2 single-hand micro-env over the 3×3 μ/stiffness scan
  - L2.3 scripted industrial pipeline (dual-arm approach, stack, QR geometry)
Relative metrics only. grasp_success_rate is never written (ADR-004).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from assets.objects.boxes import sample_box
from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec


def _physics_bundle(cfg: dict, scan_cfg: dict | None) -> dict:
    try:
        import mujoco  # noqa: F401

        from sim.mujoco_env.grasp_micro import run_scan_grid
        from sim.tasks.industrial_pipeline import run_industrial_pipeline
    except ImportError as exc:
        return {"physics": "skipped", "reason": f"import_error:{exc}"}
    out: dict = {"physics": "ran_uncalibrated", "policy_eval_forbidden": True}
    if scan_cfg is not None:
        n_close = int(scan_cfg.get("n_hold_steps", 80) // 4)
        out["l2_2_micro_scan"] = run_scan_grid(scan_cfg, n_close=max(40, n_close), n_hold=max(40, n_close))
        out["l2_2_note"] = (
            "SCAN_PLACEHOLDER grid. Values are relative across μ/solref, not E1/E2. "
            "grasp_success_rate is omitted."
        )
    try:
        result = run_industrial_pipeline(steps_per_phase=int(cfg.get("industrial_steps_per_phase", 40)))
        out["l2_3_industrial"] = {
            "phases": result.phases,
            "ik_err_m": result.ik_err_m,
            "scan_geometry_ok": result.scan_geometry_ok,
            "scan_distance_m": result.scan_distance_m,
            "finite": result.finite,
            "n_steps": result.n_steps,
            "box0_z_range": (
                [min(result.box0_z), max(result.box0_z)] if result.box0_z else None
            ),
            "policy_eval_forbidden": True,
            "note": result.note,
        }
    except Exception as exc:  # pragma: no cover - asset/mesh issues
        out["l2_3_industrial"] = {"status": "failed", "error": str(exc)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval/configs/l2_mujoco.yaml")
    parser.add_argument("--scan-config", default="eval/configs/l2_scan_grid.yaml")
    parser.add_argument("--out", default="eval/report/generated/l2.json")
    parser.add_argument("--skip-physics", action="store_true")
    args = parser.parse_args()
    spec = load_hand_spec()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scan_path = Path(args.scan_config)
    scan_cfg = yaml.safe_load(scan_path.read_text(encoding="utf-8")) if scan_path.is_file() else None
    uncalibrated = spec.raw.get("friction_vs_cardboard_static") == REQUIRED_INPUT_TOKEN
    rng = np.random.default_rng(2)
    boxes = []
    for _ in range(8):
        box = sample_box(rng)
        boxes.append(
            {
                "size_m": box.size_m.tolist(),
                "mass_kg": box.mass_kg,
                "split_flex": box.split_flex,
            }
        )
    report = {
        "uncalibrated": uncalibrated,
        "warning": (
            "Contact parameters are REQUIRED_INPUT. grasp_success_rate is NOT reported."
        )
        if uncalibrated
        else "calibrated_spec_present",
        "n_envs_configured": cfg.get("n_envs"),
        "gates": cfg.get("gates"),
        "sample_boxes": boxes,
        "status": "blocked_uncalibrated" if uncalibrated else "ready",
        "policy_eval_forbidden": True,
    }
    if not args.skip_physics:
        report.update(_physics_bundle(cfg, scan_cfg))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in ("uncalibrated", "status", "warning") if k in report}
    if "physics" in report:
        summary["physics"] = report["physics"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
