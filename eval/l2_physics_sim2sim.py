"""Physics Sim2Sim: PD-track a T800 clip in MuJoCo. No grasp-success number.

Pinned-base is the CI gate (joint MAE + FK consistency). Free-base reports
fall_rate; a trained SONIC policy is required before that number is meaningful.
Combined T800+Hand stays PolicyEvalBlocked. G1 ~6 cm wrist error is not a gate.
Table S4 target_motion.joint_jitter_rad extrema (ADR-034) are a pinned-base
clip offset, not a SONIC gate and not ADR-033 reset-qpos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from assets.combined.assemble import PolicyEvalBlocked
from eval.l2_sim2sim import EE_ROLES, _so3_geodesic_rad
from interface.schema import REPO_ROOT
from sim.mujoco_env.t800_env import ROOT_Z_M, T800MujocoEnv, refuse_combined_robot
from sim.urdf_fk import KINEMATICS_YAML, UrdfTree, fk_bodies_world, load_t800_kinematics
from wbc.gmr.synthetic_clip import synthetic_stand_clip
from wbc.observation import ProprioHistory
from wbc.ppo.table_s4 import target_motion_joint_jitter_extrema


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([w, x, y, z], dtype=np.float64)


def evaluate_physics(
    env: T800MujocoEnv,
    ref: dict[str, Any],
    *,
    height_fail_m: float,
    ori_fail_rad: float,
) -> dict[str, Any]:
    kin = load_t800_kinematics()
    tree = UrdfTree.from_kinematics_yaml(KINEMATICS_YAML, root_link=str(kin["root_link"]))
    order = list(kin["joint_order"])
    bodies = dict(kin["tracked_bodies"])
    q_ref = np.asarray(ref["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(ref["root_pos"], dtype=np.float64)
    root_rot = np.asarray(ref["root_rot"], dtype=np.float64)
    dt = float(env.model.opt.timestep)
    substeps = max(int(round((1.0 / float(ref["fps"])) / dt)), 1)

    env.reset()
    env.set_q(q_ref[0])
    env.set_root(root_pos[0], _xyzw_to_wxyz(root_rot[0]))
    env._mujoco.mj_forward(env.model, env.data)

    hist = ProprioHistory(n_dof=len(order))
    mpjpe: list[float] = []
    wrist: list[float] = []
    fk_consist: list[float] = []
    fk_consist_no_feet: list[float] = []
    foot_dz_frames: list[float] = []
    joint_mae: list[float] = []
    height_fail = 0
    ori_fail = 0
    last_a = np.zeros(len(order))

    for i in range(q_ref.shape[0]):
        q_des = q_ref[i]
        for _ in range(substeps):
            env.step(q_des)
        q_sim = env.get_q()
        joint_mae.append(float(np.mean(np.abs(q_sim - q_des))))
        fk_ref = fk_bodies_world(
            tree,
            q_des,
            joint_order=order,
            bodies=bodies,
            root_pos_m=root_pos[i],
            root_rot_xyzw=root_rot[i],
        )
        fk_sim = fk_bodies_world(
            tree,
            q_sim,
            joint_order=order,
            bodies=bodies,
            root_pos_m=root_pos[i] if env.pinned_base else env.body_pose("pelvis")[0],
            root_rot_xyzw=root_rot[i],
        )
        body_err = []
        consist = []
        consist_no_feet = []
        foot_dz = []
        for role in bodies:
            sim_p, _sim_r = env.body_pose(role)
            body_err.append(float(np.linalg.norm(sim_p - fk_ref[role][0])))
            delta = float(np.linalg.norm(sim_p - fk_sim[role][0]))
            consist.append(delta)
            if role in ("left_foot", "right_foot"):
                foot_dz.append(float(sim_p[2] - fk_sim[role][0][2]))
            else:
                consist_no_feet.append(delta)
        mpjpe.append(float(np.mean(body_err)))
        wrist.append(
            0.5
            * (
                float(np.linalg.norm(env.body_pose("left_wrist")[0] - fk_ref["left_wrist"][0]))
                + float(np.linalg.norm(env.body_pose("right_wrist")[0] - fk_ref["right_wrist"][0]))
            )
        )
        fk_consist.append(float(np.mean(consist)))
        fk_consist_no_feet.append(float(np.mean(consist_no_feet)))
        foot_dz_frames.append(float(np.mean(np.abs(foot_dz))))
        pelvis_p, pelvis_r = env.body_pose("pelvis")
        root_h = abs(float(pelvis_p[2] - fk_ref["pelvis"][0][2]))
        ee_h = max(abs(float(env.body_pose(r)[0][2] - fk_ref[r][0][2])) for r in EE_ROLES)
        if root_h > height_fail_m or ee_h > height_fail_m:
            height_fail += 1
        if _so3_geodesic_rad(pelvis_r, fk_ref["pelvis"][1]) > ori_fail_rad:
            ori_fail += 1
        omega = np.zeros(3) if env.pinned_base else np.asarray(env.data.qvel[3:6], dtype=np.float64)
        yaw = float(np.arctan2(pelvis_r[1, 0], pelvis_r[0, 0]))
        hist.push(q_sim, env.get_dq(), omega, last_a, yaw)
        last_a = q_des.copy()

    fallen = pelvis_p[2] < 0.4
    success = (height_fail + ori_fail) == 0 and not fallen
    decoder = hist.decoder_input(np.zeros(64))
    return {
        "n_frames": int(q_ref.shape[0]),
        "substeps": substeps,
        "mpjpe_m": float(np.mean(mpjpe)),
        "wrist_tracking_error_m": float(np.mean(wrist)),
        "fk_consistency_m": float(np.mean(fk_consist)),
        "fk_consistency_no_feet_m": float(np.mean(fk_consist_no_feet)),
        "foot_urdf_mjcf_delta_z_m": float(np.mean(foot_dz_frames)),
        "joint_mae_rad": float(np.mean(joint_mae)),
        "height_fail_frames": height_fail,
        "ori_fail_frames": ori_fail,
        "local_tracking_success": bool(success) if env.pinned_base else False,
        "fall_rate": 0.0 if env.pinned_base else (1.0 if fallen else 0.0),
        "pelvis_z_m": float(pelvis_p[2]),
        "decoder_input_dim": int(decoder.shape[0]),
        "grasp_success_rate": None,
        "success_rate_note": "physics PD tracking, not a grasp metric",
    }


def apply_clip_joint_jitter(ref: dict[str, Any], offset_rad: float) -> dict[str, Any]:
    """Uniform additive offset on every hinge of every clip frame.

    ``offset_rad`` must come from Table S4 ``target_motion.joint_jitter_rad``.
    Root pose is unchanged (pinned-base cannot follow a jittered root).
    """
    offset = float(offset_rad)
    if not np.isfinite(offset):
        raise ValueError(f"offset_rad must be finite, got {offset}")
    out = dict(ref)
    q = np.asarray(ref["dof_pos"], dtype=np.float64).copy()
    out["dof_pos"] = q + offset
    out["joint_jitter_rad"] = offset
    return out


def _assert_clip_in_range(env: T800MujocoEnv, ref: dict[str, Any], *, offset_rad: float) -> int:
    q = np.asarray(ref["dof_pos"], dtype=np.float64)
    limited = env.assert_hinges_in_mjcf_range(
        q.min(axis=0), what=f"target_motion joint_jitter {offset_rad:+g} rad clip-min"
    )
    env.assert_hinges_in_mjcf_range(
        q.max(axis=0), what=f"target_motion joint_jitter {offset_rad:+g} rad clip-max"
    )
    return len(limited)


def evaluate_joint_jitter(
    env: T800MujocoEnv,
    ref_base: dict[str, Any],
    *,
    offset_rad: float,
    case_name: str,
    height_fail_m: float,
    ori_fail_rad: float,
) -> dict[str, Any]:
    """PD-track one Table S4 joint-jitter extremum. Not a SONIC gate."""
    if not env.pinned_base:
        raise ValueError("target_motion joint jitter sweep requires pinned_base=True")
    ref = apply_clip_joint_jitter(ref_base, offset_rad)
    n_limited = _assert_clip_in_range(env, ref, offset_rad=offset_rad)
    metrics = evaluate_physics(
        env,
        ref,
        height_fail_m=height_fail_m,
        ori_fail_rad=ori_fail_rad,
    )
    return {
        **metrics,
        "name": case_name,
        "kind": "target_motion_joint_jitter",
        "offset_rad": float(offset_rad),
        "n_limited_joints_checked": n_limited,
        "table_s4_field": "target_motion.joint_jitter_rad",
        "mujoco_channel": "clip_dof_pos",
        "additive_to_clip_dof_pos": True,
        "not_default_joint_pos_offset": True,
        "not_per_joint_corner_grid": True,
        "not_root_push": True,
        "source": "He et al., SONIC, arXiv:2511.07820v3 Table S4 target_motion.joint_jitter_rad",
        "not_a_sonic_gate": True,
        "not_dexhand2_contact": True,
        "not_pad_cardboard": True,
        "restitution_not_mapped": True,
        "grasp_success_rate": None,
    }


def evaluate_joint_jitter_sweep(
    env: T800MujocoEnv,
    ref_base: dict[str, Any],
    *,
    height_fail_m: float,
    ori_fail_rad: float,
) -> list[dict[str, Any]]:
    """PD-track each Table S4 target_motion.joint_jitter_rad extremum."""
    out: list[dict[str, Any]] = []
    for case in target_motion_joint_jitter_extrema():
        out.append(
            evaluate_joint_jitter(
                env,
                ref_base,
                offset_rad=float(case["offset_rad"]),
                case_name=str(case["name"]),
                height_fail_m=height_fail_m,
                ori_fail_rad=ori_fail_rad,
            )
        )
    return out


def _jitter_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cases": len(cases),
        "names": [c["name"] for c in cases],
        "joint_mae_rad": {c["name"]: c["joint_mae_rad"] for c in cases},
        "local_tracking_success": {c["name"]: c["local_tracking_success"] for c in cases},
        "not_a_sonic_gate": True,
    }


def run(
    *,
    source: str = "auto",
    pinned_base: bool = True,
    n_frames: int = 40,
    amplitude_rad: float = 0.05,
) -> dict[str, Any]:
    cfg_path = REPO_ROOT / "eval" / "configs" / "l2_physics_sim2sim.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    env = T800MujocoEnv(source=source, pinned_base=pinned_base)
    ref = synthetic_stand_clip(n_frames=n_frames, fps=50.0, amplitude_rad=amplitude_rad)
    # Pinned world root matches official LINK_BASE z; clip already uses 1.03 m.
    if abs(float(ref["root_pos"][0, 2]) - ROOT_Z_M) > 1e-6:
        ref = dict(ref)
        root = np.asarray(ref["root_pos"]).copy()
        root[:, 2] = ROOT_Z_M
        ref["root_pos"] = root
    metrics = evaluate_physics(
        env,
        ref,
        height_fail_m=float(cfg["success_fail"]["root_or_ee_height_err_m"]),
        ori_fail_rad=float(cfg["success_fail"]["root_ori_err_rad"]),
    )
    report: dict[str, Any] = {
        "status": "physics_sim2sim",
        "physics": "mujoco",
        "source": env.source,
        "xml_note": env.xml_note,
        "gains_source": env.gains_source,
        "pinned_base": env.pinned_base,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "paper_g1_wrist_err_m_not_a_gate": 0.06,
        **metrics,
    }
    if env.pinned_base:
        sweep = evaluate_joint_jitter_sweep(
            env,
            ref,
            height_fail_m=float(cfg["success_fail"]["root_or_ee_height_err_m"]),
            ori_fail_rad=float(cfg["success_fail"]["root_ori_err_rad"]),
        )
        plus = next((c for c in sweep if c["name"] == "q_jit_+0.1"), None)
        if plus is None:
            raise ValueError(
                "Table S4 joint-jitter sweep must include q_jit_+0.1 (ADR-034 compatibility)"
            )
        report["joint_jitter"] = plus
        report["joint_jitter_sweep"] = sweep
        report["joint_jitter_sweep_summary"] = _jitter_summary(sweep)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto", choices=("auto", "official", "fixture"))
    parser.add_argument("--free-base", action="store_true")
    parser.add_argument("--out", default="eval/report/generated/l2_physics_sim2sim.json")
    args = parser.parse_args()
    try:
        refuse_combined_robot()
    except PolicyEvalBlocked:
        pass
    report = run(source=args.source, pinned_base=not args.free_base)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = (
        "status",
        "source",
        "pinned_base",
        "joint_mae_rad",
        "fk_consistency_no_feet_m",
        "foot_urdf_mjcf_delta_z_m",
        "wrist_tracking_error_m",
        "fall_rate",
        "grasp_success_rate",
        "decoder_input_dim",
    )
    summary = {k: report[k] for k in keys}
    if report.get("joint_jitter_sweep_summary"):
        summary["joint_jitter_sweep_names"] = report["joint_jitter_sweep_summary"]["names"]
        summary["joint_jitter_mae_rad"] = report["joint_jitter_sweep_summary"]["joint_mae_rad"]
        summary["joint_jitter_offset_rad"] = report["joint_jitter"]["offset_rad"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
