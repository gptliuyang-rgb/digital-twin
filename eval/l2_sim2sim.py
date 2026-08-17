"""Kinematic Sim2Sim: MPJPE / wrist error on T800 tracked bodies. No physics.

Identity tracker (q_pred = q_ref) must report ~0 error. This is not a grasp
metric and not the G1 real-world 6 cm wrist number (ADR-022).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import REPO_ROOT
from sim.urdf_fk import (
    KINEMATICS_YAML,
    UrdfTree,
    fk_bodies_world,
    load_t800_kinematics,
)
from wbc.gmr.motion_lib import validate_motion_lib
from wbc.gmr.synthetic_clip import synthetic_stand_clip

EE_ROLES = ("head", "left_wrist", "right_wrist", "left_foot", "right_foot")


def _so3_geodesic_rad(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r = r_a.T @ r_b
    tr = float(np.trace(r))
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))


def evaluate_clip(
    tree: UrdfTree,
    ref: dict[str, Any],
    pred_q: np.ndarray,
    *,
    kin: dict[str, Any],
    height_fail_m: float,
    ori_fail_rad: float,
) -> dict[str, Any]:
    validate_motion_lib(ref)
    q_ref = np.asarray(ref["dof_pos"], dtype=np.float64)
    q_pred = np.asarray(pred_q, dtype=np.float64)
    if q_pred.shape != q_ref.shape:
        raise ValueError(f"pred q {q_pred.shape} != ref {q_ref.shape}")
    order = list(kin["joint_order"])
    bodies = dict(kin["tracked_bodies"])
    t = q_ref.shape[0]
    mpjpe = []
    wrist = []
    height_fail = 0
    ori_fail = 0
    for i in range(t):
        root_p = ref["root_pos"][i]
        root_r = ref["root_rot"][i]
        g = fk_bodies_world(
            tree, q_ref[i], joint_order=order, bodies=bodies, root_pos_m=root_p, root_rot_xyzw=root_r
        )
        p = fk_bodies_world(
            tree, q_pred[i], joint_order=order, bodies=bodies, root_pos_m=root_p, root_rot_xyzw=root_r
        )
        body_err = [np.linalg.norm(p[role][0] - g[role][0]) for role in bodies]
        mpjpe.append(float(np.mean(body_err)))
        wrist.append(
            0.5
            * (
                float(np.linalg.norm(p["left_wrist"][0] - g["left_wrist"][0]))
                + float(np.linalg.norm(p["right_wrist"][0] - g["right_wrist"][0]))
            )
        )
        root_h = abs(float(p["pelvis"][0][2] - g["pelvis"][0][2]))
        ee_h = max(abs(float(p[r][0][2] - g[r][0][2])) for r in EE_ROLES)
        if root_h > height_fail_m or ee_h > height_fail_m:
            height_fail += 1
        if _so3_geodesic_rad(p["pelvis"][1], g["pelvis"][1]) > ori_fail_rad:
            ori_fail += 1
    success = (height_fail + ori_fail) == 0
    return {
        "n_frames": t,
        "mpjpe_m": float(np.mean(mpjpe)),
        "wrist_tracking_error_m": float(np.mean(wrist)),
        "height_fail_frames": height_fail,
        "ori_fail_frames": ori_fail,
        "local_tracking_success": success,
        "fall_rate": 0.0 if success else 1.0,
        "grasp_success_rate": None,
        "success_rate_note": "local motion tracking (SONIC §2.1), not a grasp metric",
    }


def run(*, perturb_wrist_joint_rad: float = 0.0) -> dict[str, Any]:
    kin = load_t800_kinematics()
    tree = UrdfTree.from_kinematics_yaml(KINEMATICS_YAML, root_link=str(kin["root_link"]))
    cfg_path = REPO_ROOT / "eval" / "configs" / "l2_sim2sim.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ref = synthetic_stand_clip(n_frames=30, fps=50.0)
    q_pred = np.asarray(ref["dof_pos"], dtype=np.float64).copy()
    if perturb_wrist_joint_rad:
        # J17 / J22 are elbow yaw (dummy-wrist parent). A joint offset moves the wrist.
        order = list(kin["joint_order"])
        q_pred[:, order.index("J17_ELBOW_YAW_L")] += float(perturb_wrist_joint_rad)
    metrics = evaluate_clip(
        tree,
        ref,
        q_pred,
        kin=kin,
        height_fail_m=float(cfg["success_fail"]["root_or_ee_height_err_m"]),
        ori_fail_rad=float(cfg["success_fail"]["root_ori_err_rad"]),
    )
    identity = abs(float(perturb_wrist_joint_rad)) < 1e-15
    return {
        "status": "kinematic_sim2sim",
        "physics": "none",
        "tracker": "identity" if identity else "perturbed_elbow_yaw",
        "perturb_wrist_joint_rad": float(perturb_wrist_joint_rad),
        "kinematics": str(KINEMATICS_YAML.relative_to(REPO_ROOT)),
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "paper_g1_wrist_err_m_not_a_gate": 0.06,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturb-rad", type=float, default=0.0)
    parser.add_argument("--out", default="eval/report/generated/l2_sim2sim.json")
    args = parser.parse_args()
    report = run(perturb_wrist_joint_rad=args.perturb_rad)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "mpjpe_m", "wrist_tracking_error_m", "grasp_success_rate")}, indent=2))


if __name__ == "__main__":
    main()
