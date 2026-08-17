"""Free-base T800 PD stand. Report fall honestly. Not a SONIC pass/fail.

Pinned-base physics Sim2Sim (``eval/l2_physics_sim2sim.py``) is the CI tracking
gate. This harness lets the floating base move on a floor with EngineAI bring-up
PD. Without a trained SONIC policy the robot is expected to fall; if it happens
to hold, that is also not a tracker success. ``grasp_success_rate`` stays null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from assets.combined.assemble import PolicyEvalBlocked
from eval.lean_classify import fallen_from_cfg, posture_from_cfg
from interface.schema import REPO_ROOT
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key
from sim.mujoco_env.t800_env import T800MujocoEnv, pelvis_tilt_rad, refuse_combined_robot
from wbc.foot_frame import FOOT_FRAME_DECISION, assert_foot_frame


def evaluate_freebase_stand(
    env: T800MujocoEnv,
    *,
    hold_s: float,
    cfg: dict[str, Any],
    q_des: np.ndarray | None = None,
) -> dict[str, Any]:
    if env.pinned_base:
        raise ValueError("free-base stand requires pinned_base=False")
    if env.n_plane < 1:
        raise ValueError("free-base stand requires a floor plane")
    if q_des is None:
        q_des = env.default_q_des_rad()
        q_des_source = (
            "official_pd_stand_desired_joint_position"
            if env.source == "official"
            else "fixture_zeros"
        )
    else:
        q_des = np.asarray(q_des, dtype=np.float64).reshape(len(env.joint_order))
        q_des_source = "caller_supplied"
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    start = env.foot_diagnostics()
    pelvis0, _ = env.body_pose("pelvis")
    n = int(round(hold_s / float(env.model.opt.timestep)))
    time_to_fall_s: float | None = None
    z_hist: list[float] = []
    tilt_hist: list[float] = []
    for i in range(max(n, 1)):
        env.step(q_des)
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = float(i + 1) * float(env.model.opt.timestep)
    end = env.foot_diagnostics()
    pelvis, rot = env.body_pose("pelvis")
    end_tilt = pelvis_tilt_rad(rot)
    fallen = fallen_from_cfg(float(pelvis[2]), end_tilt, cfg)
    posture = posture_from_cfg(float(pelvis[2]), end_tilt, cfg)
    return {
        "hold_s": hold_s,
        "n_steps": n,
        "q_des_source": q_des_source,
        "start_pelvis_z_m": float(pelvis0[2]),
        "end_pelvis_z_m": float(pelvis[2]),
        "min_pelvis_z_m": float(np.min(z_hist)),
        "end_tilt_rad": float(end_tilt),
        "max_tilt_rad": float(np.max(tilt_hist)),
        "time_to_fall_s": time_to_fall_s,
        "posture": posture,
        "fallen": fallen,
        "fall_rate": 1.0 if fallen else 0.0,
        "start_foot": start,
        "end_foot": end,
        "n_plane": env.n_plane,
        "nq": int(env.model.nq),
        "expected_nq": env.expected_nq(),
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "bringup_pd_is_not_a_balance_controller": True,
        "grasp_success_rate": None,
        "success_rate_note": "free-base PD stand is not a grasp metric and not a SONIC gate",
    }


def run(*, source: str = "auto", hold_s: float | None = None) -> dict[str, Any]:
    cfg_path = REPO_ROOT / "eval" / "configs" / "l2_freebase_stand.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert_foot_frame()
    env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    metrics = evaluate_freebase_stand(env, hold_s=float(hold_s if hold_s is not None else cfg["hold_s"]), cfg=cfg)
    report = {
        "status": "freebase_pd_stand",
        "physics": "mujoco",
        "source": env.source,
        "xml_note": env.xml_note,
        "gains_source": env.gains_source,
        "pinned_base": env.pinned_base,
        "foot_frame": FOOT_FRAME_DECISION,
        "floor_friction": "official_wbc_collision_default_not_pad_cardboard",
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        **metrics,
    }
    refuse_grasp_success_key(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto", choices=("auto", "official", "fixture"))
    parser.add_argument("--hold-s", type=float, default=None)
    parser.add_argument("--out", default="eval/report/generated/l2_freebase_stand.json")
    args = parser.parse_args()
    try:
        refuse_combined_robot()
    except PolicyEvalBlocked:
        pass
    report = run(source=args.source, hold_s=args.hold_s)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = (
        "status",
        "source",
        "pinned_base",
        "fallen",
        "fall_rate",
        "time_to_fall_s",
        "end_pelvis_z_m",
        "not_a_sonic_gate",
        "foot_frame",
        "grasp_success_rate",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
