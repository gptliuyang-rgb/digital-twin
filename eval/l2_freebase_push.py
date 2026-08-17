"""Free-base T800 Table S4 root-push sweep and air-drop. Not a SONIC pass/fail.

The unperturbed 3 s PD hold can lean ~41° (0.72 rad) and still sit under the
0.80 rad fall tilt band. That is ``leaned``, not a stand, and not a tracker
success (ADR-026). ADR-027 sweeps Table S4 planar extrema (±X / ±Y at 0.5 m/s)
as one-shot root velocities after that hold. A single +Y hit can hide an
axis-dependent lean. Air-drop (no floor) proves the freejoint.
``grasp_success_rate`` stays JSON null. Combined T800+Hand stays PolicyEvalBlocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from assets.combined.assemble import PolicyEvalBlocked
from eval.l2_freebase_stand import evaluate_freebase_stand
from eval.lean_classify import fallen_from_cfg, posture_from_cfg
from interface.schema import REPO_ROOT
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key
from sim.mujoco_env.t800_env import T800MujocoEnv, pelvis_tilt_rad, refuse_combined_robot
from wbc.foot_frame import FOOT_FRAME_DECISION, assert_foot_frame
from wbc.pd_stand import pd_stand_q_des_rad
from wbc.ppo.table_s4 import axis_aligned_linvel_extrema_mps


def _q_des(env: T800MujocoEnv) -> np.ndarray:
    return pd_stand_q_des_rad() if env.source == "official" else np.zeros(len(env.joint_order))


def evaluate_lateral_push(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
    lin_vel_mps: list[float] | np.ndarray | None = None,
    case_name: str | None = None,
) -> dict[str, Any]:
    if env.pinned_base:
        raise ValueError("lateral push requires pinned_base=False")
    if env.n_plane < 1:
        raise ValueError("lateral push requires a floor plane")
    q_des = _q_des(env)
    if lin_vel_mps is None:
        lin_vel = np.asarray(cfg["push"]["lin_vel_mps"], dtype=np.float64).reshape(3)
        name = str(cfg["push"].get("name", "+y"))
    else:
        lin_vel = np.asarray(lin_vel_mps, dtype=np.float64).reshape(3)
        name = case_name or "custom"
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    dt = float(env.model.opt.timestep)
    n_settle = int(round(settle_s / dt))
    n_post = int(round(post_push_s / dt))
    z_hist: list[float] = []
    tilt_hist: list[float] = []
    time_to_fall_s: float | None = None
    injected = False
    t = 0.0
    for i in range(max(n_settle, 1)):
        env.step(q_des)
        t = float(i + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    env.apply_root_linvel(lin_vel)
    injected = True
    t_inject_s = t
    pre_z = float(z_hist[-1]) if z_hist else float("nan")
    pre_tilt = float(tilt_hist[-1]) if tilt_hist else float("nan")
    pre_posture = posture_from_cfg(pre_z, pre_tilt, cfg)
    for j in range(max(n_post, 1)):
        env.step(q_des)
        t = t_inject_s + float(j + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    pelvis, rot = env.body_pose("pelvis")
    end_tilt = pelvis_tilt_rad(rot)
    posture = posture_from_cfg(float(pelvis[2]), end_tilt, cfg)
    fallen = posture == "fallen"
    return {
        "settle_s": settle_s,
        "post_push_s": post_push_s,
        "inject_t_s": t_inject_s,
        "pre_push_pelvis_z_m": pre_z,
        "pre_push_tilt_rad": pre_tilt,
        "pre_push_posture": pre_posture,
        "name": name,
        "lin_vel_mps": [float(x) for x in lin_vel],
        "kind": cfg["push"]["kind"],
        "source": cfg["push"]["source"],
        "injected": injected,
        "end_pelvis_z_m": float(pelvis[2]),
        "min_pelvis_z_m": float(np.min(z_hist)),
        "end_tilt_rad": float(end_tilt),
        "max_tilt_rad": float(np.max(tilt_hist)),
        "posture": posture,
        "fallen": fallen,
        "fall_rate": 1.0 if fallen else 0.0,
        "time_to_fall_s": time_to_fall_s,
        "end_foot": env.foot_diagnostics(),
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "grasp_success_rate": None,
    }


def evaluate_airdrop(env: T800MujocoEnv, *, hold_s: float, min_drop_m: float) -> dict[str, Any]:
    if env.pinned_base:
        raise ValueError("air-drop requires pinned_base=False")
    if env.n_plane != 0:
        raise ValueError("air-drop requires add_floor=False (no plane)")
    q_des = _q_des(env)
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    pelvis0, _ = env.body_pose("pelvis")
    n = int(round(hold_s / float(env.model.opt.timestep)))
    for _ in range(max(n, 1)):
        env.step(q_des)
    pelvis, rot = env.body_pose("pelvis")
    drop_m = float(pelvis0[2] - pelvis[2])
    return {
        "hold_s": hold_s,
        "add_floor": False,
        "n_plane": env.n_plane,
        "nq": int(env.model.nq),
        "start_pelvis_z_m": float(pelvis0[2]),
        "end_pelvis_z_m": float(pelvis[2]),
        "drop_m": drop_m,
        "min_drop_m": min_drop_m,
        "end_tilt_rad": float(pelvis_tilt_rad(rot)),
        "freejoint_moved": drop_m > min_drop_m,
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "not_a_stand_rating": True,
        "grasp_success_rate": None,
        "note": "no floor: gravity must move the freejoint; not a stand rating",
    }


def evaluate_push_sweep(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
) -> list[dict[str, Any]]:
    """Run one-shot qvel at each Table S4 planar extremum. Not a SONIC gate."""
    axes = tuple(cfg.get("sweep", {}).get("axes", ["x", "y"]))
    cases = axis_aligned_linvel_extrema_mps(axes=axes)
    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(
            evaluate_lateral_push(
                env,
                cfg=cfg,
                settle_s=settle_s,
                post_push_s=post_push_s,
                lin_vel_mps=case["lin_vel_mps"],
                case_name=str(case["name"]),
            )
        )
    return out


def _sweep_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cases": len(cases),
        "names": [c["name"] for c in cases],
        "n_fallen": int(sum(1 for c in cases if c["fallen"])),
        "any_fallen": any(c["fallen"] for c in cases),
        "all_fallen": all(c["fallen"] for c in cases) if cases else False,
        "not_a_sonic_gate": True,
        "by_name": {
            c["name"]: {
                "lin_vel_mps": c["lin_vel_mps"],
                "pre_push_posture": c["pre_push_posture"],
                "posture": c["posture"],
                "fallen": c["fallen"],
                "time_to_fall_s": c["time_to_fall_s"],
                "end_pelvis_z_m": c["end_pelvis_z_m"],
                "end_tilt_rad": c["end_tilt_rad"],
            }
            for c in cases
        },
    }


def run(*, source: str = "auto") -> dict[str, Any]:
    cfg_path = REPO_ROOT / "eval" / "configs" / "l2_freebase_push.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert_foot_frame()
    hold_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    hold = evaluate_freebase_stand(hold_env, hold_s=float(cfg["hold_s"]), cfg=cfg)
    push_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    sweep = evaluate_push_sweep(
        push_env,
        cfg=cfg,
        settle_s=float(cfg["settle_s"]),
        post_push_s=float(cfg["post_push_s"]),
    )
    plus_y = next((c for c in sweep if c["name"] == "+y"), None)
    if plus_y is None:
        raise ValueError("Table S4 planar sweep must include +y (ADR-026 compatibility)")
    air_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=False)
    airdrop = evaluate_airdrop(
        air_env,
        hold_s=float(cfg["airdrop_s"]),
        min_drop_m=float(cfg["airdrop"]["min_drop_m"]),
    )
    report = {
        "status": "freebase_pd_push",
        "physics": "mujoco",
        "source": hold_env.source,
        "xml_note": hold_env.xml_note,
        "gains_source": hold_env.gains_source,
        "pinned_base": False,
        "foot_frame": FOOT_FRAME_DECISION,
        "floor_friction": "official_wbc_collision_default_not_pad_cardboard",
        "lean_tilt_rad": float(cfg["lean"]["tilt_rad"]),
        "fall": cfg["fall"],
        "hold": hold,
        "lateral_push": plus_y,
        "push_sweep": sweep,
        "push_sweep_summary": _sweep_summary(sweep),
        "airdrop": airdrop,
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "bringup_pd_is_not_a_balance_controller": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "success_rate_note": "lean vs fall vs planar sweep vs air-drop are diagnostics; not SONIC gates",
    }
    refuse_grasp_success_key(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto", choices=("auto", "official", "fixture"))
    parser.add_argument("--out", default="eval/report/generated/l2_freebase_push.json")
    args = parser.parse_args()
    try:
        refuse_combined_robot()
    except PolicyEvalBlocked:
        pass
    report = run(source=args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "status": report["status"],
        "source": report["source"],
        "hold_posture": report["hold"]["posture"],
        "hold_fallen": report["hold"]["fallen"],
        "hold_end_tilt_rad": report["hold"]["end_tilt_rad"],
        "push_posture": report["lateral_push"]["posture"],
        "push_pre_posture": report["lateral_push"]["pre_push_posture"],
        "push_fallen": report["lateral_push"]["fallen"],
        "push_sweep_names": report["push_sweep_summary"]["names"],
        "push_sweep_n_fallen": report["push_sweep_summary"]["n_fallen"],
        "airdrop_freejoint_moved": report["airdrop"]["freejoint_moved"],
        "airdrop_drop_m": report["airdrop"]["drop_m"],
        "not_a_sonic_gate": report["not_a_sonic_gate"],
        "grasp_success_rate": report["grasp_success_rate"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
