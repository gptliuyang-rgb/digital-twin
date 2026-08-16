"""L2 closed-loop entry. Contact-uncalibrated specs must not report success rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from assets.objects.boxes import sample_box
from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval/configs/l2_mujoco.yaml")
    parser.add_argument("--out", default="eval/report/generated/l2.json")
    args = parser.parse_args()
    spec = load_hand_spec()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
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
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("uncalibrated", "status", "warning")}, indent=2))


if __name__ == "__main__":
    main()
