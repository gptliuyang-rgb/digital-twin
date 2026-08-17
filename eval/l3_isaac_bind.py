"""Try IsaacLabSceneRuntime.reset/step when Isaac Sim python is present.

CI without Isaac reports runtime=unavailable. Combined T800+Hand stays
PolicyEvalBlocked. grasp_success_rate stays JSON null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.isaaclab_env.privileged import IsaacLabUnavailable
from sim.isaaclab_env.probe import isaac_runtime_status
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key


def try_bind_and_step() -> dict[str, Any]:
    status = isaac_runtime_status()
    report: dict[str, Any] = {
        **status,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "reset_step_ran": False,
        "not_a_grasp_eval": True,
        "include_hands": False,
        "include_t800": False,
    }
    if status["runtime"] != "bound":
        report["status"] = "isaac_bind_unavailable"
        refuse_grasp_success_key(report)
        return report
    from assets.objects.boxes import sample_box
    from sim.isaaclab_env.scene_spec import IsaacLabSceneRuntime, PalletBoxSceneSpec

    spec = PalletBoxSceneSpec()
    box = sample_box(np.random.default_rng(0))
    try:
        runtime = IsaacLabSceneRuntime(spec, box)
        obs = runtime.reset()
        step_obs, _reward, _done, info = runtime.step(np.zeros(1))
    except IsaacLabUnavailable as exc:
        report["status"] = "isaac_bind_unavailable"
        report["error"] = str(exc)
        refuse_grasp_success_key(report)
        return report
    report["reset_step_ran"] = True
    report["reset_status"] = obs.get("status")
    report["step_status"] = step_obs.get("status")
    report["step_grasp_success_rate"] = info.get("grasp_success_rate")
    report["status"] = "isaac_bind_ok"
    refuse_grasp_success_key(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/report/generated/l3_isaac_bind.json")
    args = parser.parse_args()
    report = try_bind_and_step()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = (
        "status",
        "runtime",
        "reset_step_ran",
        "grasp_success_rate",
        "combined_robot",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
