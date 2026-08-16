"""L2 closed-loop entry. Contact-uncalibrated specs must not report success rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from assets.combined.assemble import mount_ready, weld_recipe
from assets.objects.boxes import sample_box
from eval.gain_scan import gain_cells
from eval.stack_metrics import release_quality, stack_stable
from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec
from sim.hand_mass import mass_budget


def _privileged_pallet() -> dict:
    try:
        from eval.l2_privileged import run

        return run()
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "error": str(exc), "grasp_success_rate": None}


def _optional_mujoco_tracking(spec) -> dict:
    try:
        import mujoco
    except ImportError:
        return {"status": "skipped_no_mujoco"}
    from assets.dexhand2.build.gen_derived import DERIVED, generate
    from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM

    if not (DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/mjcf/right.xml").is_file():
        return {"status": "skipped_no_official_mjcf"}
    xml = DERIVED / "right_with_pad_spheres_mit.xml"
    if not xml.is_file():
        generate("right", mit_motors=True, simplified=False)
    from hand.backends.mujoco_backend import MujocoBackend
    from hand.controller import DexHand2Controller, SafetyLimits
    from hand.grasp_primitives import GraspLibrary

    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    data = mujoco.MjData(model)
    acts = list(range(model.nu))
    backend = MujocoBackend(model, data, acts)
    safety = SafetyLimits(max_delta_q_rad=3.0, velocity_limit_rad_s=20.0)
    ctrl = DexHand2Controller(spec, backend, safety=safety)
    lib = GraspLibrary(spec)
    q_goal = lib.q_active("power_grasp", 0.4)
    ctrl.set_joint_targets(q_goal)
    nstep = int(1.0 / model.opt.timestep)
    for _ in range(nstep):
        ctrl.set_joint_targets(q_goal)
        mujoco.mj_step(model, data)
    err = float(np.linalg.norm(ctrl.get_state().q_rad - q_goal))
    return {
        "status": "ran_privileged_tracking",
        "xml": str(xml),
        "n_nu": int(model.nu),
        "q_goal_l2_err_rad": err,
        "note": "Joint tracking only. Not a grasp-success number.",
    }


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
    # Synthetic settle traj for the metric implementation (not a physics result).
    xy = np.zeros((50, 2))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (50, 1))
    support = np.array([[-0.4, -0.4], [0.4, 0.4]])
    gate = stack_stable(xy, quat, support, dt_s=0.1)
    rel = release_quality(0.01, 0.002)
    budget = mass_budget(spec)
    cells = [
        {k: cell[k] for k in ("kp_scale", "kv_scale", "status")}
        for cell in gain_cells(spec)
    ]
    report = {
        "uncalibrated": uncalibrated,
        "combined_eval_allowed": mount_ready(),
        "weld_eval_allowed": weld_recipe()["eval_allowed"],
        "sonic_mass_ready": budget.sonic_ready,
        "two_hands_product_kg": budget.two_hands_product_kg,
        "warning": (
            "Contact parameters are REQUIRED_INPUT. grasp_success_rate is NOT reported."
        )
        if uncalibrated
        else "calibrated_spec_present",
        "n_envs_configured": cfg.get("n_envs"),
        "gates": cfg.get("gates"),
        "sample_boxes": boxes,
        "gain_scan_cells": cells,
        "stack_metric_selftest": {
            "stable": gate.stable,
            "max_drift_m": gate.max_drift_m,
            "release_ok": rel.velocity_ok and rel.height_ok,
        },
        "privileged_tracking": _optional_mujoco_tracking(spec),
        "privileged_l2": _privileged_pallet(),
        "grasp_success_rate": None,
        "status": "blocked_uncalibrated" if uncalibrated else "ready",
    }
    from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key

    refuse_grasp_success_key(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("uncalibrated", "status", "warning", "combined_eval_allowed")}, indent=2))


if __name__ == "__main__":
    main()
