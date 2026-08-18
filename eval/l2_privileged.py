"""Privileged L2 harness. Never reports a numeric grasp_success_rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec
from sim.mujoco_env.privileged_l2 import (
    combined_robot_blocked,
    refuse_grasp_success_key,
    uncalibrated_contact,
)


def run() -> dict:
    spec = load_hand_spec()
    uncal = uncalibrated_contact() or spec.raw.get("friction_vs_cardboard_static") == REQUIRED_INPUT_TOKEN
    report: dict = {
        "uncalibrated": uncal,
        "grasp_success_rate": None,
        "status": "blocked_uncalibrated" if uncal else "contact_params_present",
        "warning": (
            "Contact parameters are REQUIRED_INPUT. grasp_success_rate is NOT reported."
        ),
        "pallet_drop": {"status": "skipped_no_mujoco"},
    }
    try:
        from sim.mujoco_env.privileged_l2 import PrivilegedL2Env

        env = PrivilegedL2Env()
        report["pallet_drop"] = env.drop_and_settle(settle_s=1.5)
    except ImportError:
        report["pallet_drop"] = {"status": "skipped_no_mujoco"}
    try:
        combined_robot_blocked()
    except Exception as exc:
        report["combined_robot"] = {"blocked": True, "reason": str(exc)}
    refuse_grasp_success_key(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/report/generated/l2_privileged.json")
    args = parser.parse_args()
    report = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("uncalibrated", "status", "grasp_success_rate")}, indent=2))


if __name__ == "__main__":
    main()
