#!/usr/bin/env python3
"""Run E1/E2 dry-run: CSV → fit → overlay spec → derived pads → validate_sim.

Never patches the live dexhand2_spec.yaml. Overlay + reports stay under
hand/calibration/results/<batch>/generated/.

Options
-------
--both-skins       Also fit the skin_off sibling CSVs and write a comparison JSON.
--with-l2          After validate, run eval.l2_mujoco_closedloop against the overlay.
--fit-tip-radius   Include fingertip_geometry_radius_m from official tip STLs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from hand.calibration.synthetic import (
    FRAGMENT_EXTRAS,
    SKIN_ON_DIR,
    SYNTHETIC_ROOT,
    write_synthetic_csvs,
)
from interface.schema import REPO_ROOT

GEN = SKIN_ON_DIR.parent / "generated"


def _run(args: list[str]) -> None:
    subprocess.check_call(args, cwd=REPO_ROOT)


def _run_fit(
    e1: Path,
    e2: Path,
    out: Path,
    *,
    fit_tip_radius: bool = False,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "hand.calibration.fit_params",
        "--e1",
        str(e1),
        "--e2",
        str(e2),
        "--out",
        str(out),
    ]
    if fit_tip_radius:
        cmd.append("--fit-tip-radius")
    _run(cmd)


def _fit_one(
    e1: Path,
    e2: Path,
    out_dir: Path,
    *,
    synthetic: bool,
    fit_tip_radius: bool = False,
    skip_derived: bool = False,
    skip_validate: bool = False,
    with_l2: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fragment_path = out_dir / "fragment.yaml"
    overlay_path = out_dir / "overlay.yaml"
    validate_path = out_dir / "validate_sim.json"
    l2_path = out_dir / "l2_overlay.json"

    _run_fit(e1, e2, fragment_path, fit_tip_radius=fit_tip_radius)

    fragment = yaml.safe_load(fragment_path.read_text(encoding="utf-8"))
    if synthetic:
        # Preserve STL-derived radius if already fitted, then apply other extras.
        stl_r = fragment.get("fingertip_geometry_radius_m")
        fragment.update(FRAGMENT_EXTRAS)
        if stl_r is not None:
            fragment["fingertip_geometry_radius_m"] = stl_r
        fragment["do_not_treat_as_committed_hardware"] = True
        fragment_path.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")

    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "apply_calibration_fragment.py"),
            "--fragment",
            str(fragment_path),
            "--out",
            str(overlay_path),
        ]
    )

    derived: list[str] = []
    if not skip_derived:
        for side in ("right", "left"):
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "assets.dexhand2.build.gen_derived",
                    "--side",
                    side,
                    "--spec",
                    str(overlay_path),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                derived.append(f"{side}:skipped:{proc.stderr.strip()[:200]}")
            else:
                derived.append(proc.stdout.strip() or side)

    validate_summary: dict = {}
    if not skip_validate:
        _run(
            [
                sys.executable,
                "-m",
                "hand.calibration.validate_sim",
                "--spec",
                str(overlay_path),
                "--out",
                str(validate_path),
            ]
        )
        validate_summary = json.loads(validate_path.read_text(encoding="utf-8"))

    l2_summary: dict = {}
    if with_l2:
        _run(
            [
                sys.executable,
                "-m",
                "eval.l2_mujoco_closedloop",
                "--spec",
                str(overlay_path),
                "--out",
                str(l2_path),
            ]
        )
        l2_summary = json.loads(l2_path.read_text(encoding="utf-8"))

    return {
        "e1_csv": str(e1),
        "e2_csv": str(e2),
        "fragment": str(fragment_path),
        "overlay": str(overlay_path),
        "derived": derived,
        "validate_ok": validate_summary.get("ok"),
        "e1_ok": (validate_summary.get("e1") or {}).get("ok"),
        "e2_ok": (validate_summary.get("e2") or {}).get("ok"),
        "mu_s": (validate_summary.get("e1") or {}).get("mu_s"),
        "k_n_per_m": (validate_summary.get("e2") or {}).get("k_e2_n_per_m"),
        "solref_s": (validate_summary.get("e2") or {}).get("solref_timeconst_s"),
        "l2_status": l2_summary.get("status"),
        "l2_physics": l2_summary.get("physics"),
        "l2_micro": l2_summary.get("l2_2_calibrated_micro"),
        "live_spec_unchanged": True,
        "do_not_treat_as_committed_hardware": True,
        "grasp_success_rate": None,
    }


def _compare(on: dict, off: dict) -> dict:
    """Delta between skin-on and skin-off fit values."""
    keys = ("mu_s", "k_n_per_m", "solref_s")
    delta = {}
    for k in keys:
        a, b = on.get(k), off.get(k)
        if a is not None and b is not None and a != 0:
            delta[k] = {"skin_on": round(a, 6), "skin_off": round(b, 6), "delta_pct": round(100 * (b - a) / a, 1)}
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1/E2 dry-run: CSV → fit → overlay → pad geoms → validate_sim"
    )
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic fragment extras")
    parser.add_argument("--e1", type=Path, default=None, help="skin_on E1 CSV")
    parser.add_argument("--e2", type=Path, default=None, help="skin_on E2 CSV")
    parser.add_argument("--e1-off", type=Path, default=None, help="skin_off E1 CSV (--both-skins)")
    parser.add_argument("--e2-off", type=Path, default=None, help="skin_off E2 CSV (--both-skins)")
    parser.add_argument("--out-dir", type=Path, default=GEN)
    parser.add_argument("--both-skins", action="store_true", help="Also fit skin_off and compare")
    parser.add_argument("--with-l2", action="store_true", help="Run micro episode via overlay spec")
    parser.add_argument("--skip-derived", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--fit-tip-radius",
        action="store_true",
        default=True,
        help="Read fingertip_geometry_radius_m from official *_tip.STL (default on)",
    )
    parser.add_argument("--no-fit-tip-radius", dest="fit_tip_radius", action="store_false")
    args = parser.parse_args()

    # Default to synthetic CSVs when no real CSVs are provided.
    if args.e1 is None or args.e2 is None:
        paths_on = write_synthetic_csvs(SKIN_ON_DIR, seed=0, skin="on")
        e1 = args.e1 or paths_on["e1"]
        e2 = args.e2 or paths_on["e2"]
        synthetic = True
    else:
        e1, e2 = args.e1, args.e2
        synthetic = args.synthetic or "synthetic" in str(e1)

    print("=== skin_on ===", flush=True)
    result_on = _fit_one(
        e1,
        e2,
        args.out_dir,
        synthetic=synthetic,
        fit_tip_radius=args.fit_tip_radius,
        skip_derived=args.skip_derived,
        skip_validate=args.skip_validate,
        with_l2=args.with_l2,
    )

    result_off: dict = {}
    if args.both_skins:
        if args.e1_off is None or args.e2_off is None:
            paths_off = write_synthetic_csvs(
                SYNTHETIC_ROOT / "skin_off", seed=1, skin="off", mu_s=0.58, mu_d=0.42, k_n_per_m=3800.0
            )
            e1_off = args.e1_off or paths_off["e1"]
            e2_off = args.e2_off or paths_off["e2"]
        else:
            e1_off, e2_off = args.e1_off, args.e2_off
        print("=== skin_off ===", flush=True)
        result_off = _fit_one(
            e1_off,
            e2_off,
            args.out_dir / "skin_off",
            synthetic=synthetic,
            fit_tip_radius=False,  # radius same regardless of skin
            skip_derived=True,  # skin_off derived not needed
            skip_validate=args.skip_validate,
            with_l2=False,
        )

    summary: dict = {
        "skin_on": result_on,
        "live_spec_unchanged": True,
        "do_not_treat_as_committed_hardware": True,
        "grasp_success_rate": None,
    }
    if result_off:
        summary["skin_off"] = result_off
        summary["skin_comparison"] = _compare(result_on, result_off)

    # Top-level pass/fail gates.
    summary["validate_ok"] = result_on.get("validate_ok")
    summary["e1_ok"] = result_on.get("e1_ok")
    summary["e2_ok"] = result_on.get("e2_ok")

    # Write a machine-readable summary beside the overlay.
    report_path = args.out_dir / "e1_e2_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
