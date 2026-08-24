#!/usr/bin/env python3
"""Run E1/E2 dry-run: CSV → fit → overlay spec → derived pads → validate_sim.

Never patches the live dexhand2_spec.yaml. Overlay + reports stay under
hand/calibration/results/<batch>/generated/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from hand.calibration.synthetic import FRAGMENT_EXTRAS, SKIN_ON_DIR, write_synthetic_csvs
from interface.schema import REPO_ROOT

GEN = SKIN_ON_DIR.parent / "generated"


def _run(args: list[str]) -> None:
    subprocess.check_call(args, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic fragment extras")
    parser.add_argument("--e1", type=Path, default=None)
    parser.add_argument("--e2", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=GEN)
    parser.add_argument("--with-l2", action="store_true")
    parser.add_argument("--skip-derived", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    if args.e1 is None or args.e2 is None:
        paths = write_synthetic_csvs()
        e1 = args.e1 or paths["e1"]
        e2 = args.e2 or paths["e2"]
        synthetic = True
    else:
        e1, e2 = args.e1, args.e2
        synthetic = args.synthetic or "synthetic" in str(e1)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fragment_path = out_dir / "fragment.yaml"
    overlay_path = out_dir / "overlay.yaml"
    validate_path = out_dir / "validate_sim.json"

    _run(
        [
            sys.executable,
            "-m",
            "hand.calibration.fit_params",
            "--e1",
            str(e1),
            "--e2",
            str(e2),
            "--out",
            str(fragment_path),
        ]
    )
    fragment = yaml.safe_load(fragment_path.read_text(encoding="utf-8"))
    if synthetic:
        fragment.update(FRAGMENT_EXTRAS)
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
    if not args.skip_derived:
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
    if not args.skip_validate:
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

    l2_path = out_dir / "l2_overlay.json"
    if args.with_l2:
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

    summary = {
        "e1_csv": str(e1),
        "e2_csv": str(e2),
        "fragment": str(fragment_path),
        "overlay": str(overlay_path),
        "derived": derived,
        "validate_ok": validate_summary.get("ok"),
        "e1_ok": (validate_summary.get("e1") or {}).get("ok"),
        "e2_ok": (validate_summary.get("e2") or {}).get("ok"),
        "live_spec_unchanged": True,
        "do_not_treat_as_committed_hardware": True,
        "grasp_success_rate": None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
