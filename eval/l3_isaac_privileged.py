"""Privileged Isaac Lab harness. Never reports a numeric grasp_success_rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.isaaclab_env.privileged import PrivilegedIsaacCfg, privileged_report


def run() -> dict:
    return privileged_report(PrivilegedIsaacCfg())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/report/generated/l3_isaac_privileged.json")
    args = parser.parse_args()
    report = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("uncalibrated", "status", "grasp_success_rate")}, indent=2))


if __name__ == "__main__":
    main()
