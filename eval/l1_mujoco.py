"""L1 checks that use MuJoCo: joint limits, IK residual, ncon at grasp primitives.

Pinocchio/FCL are not required. Self-collision is reported as contact counts, not
a binary fail — power grasp may produce finger–finger contacts. Not a policy gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.l1_kinematic import check_coupling, check_joint_limits
from hand.grasp_primitives import GraspLibrary
from interface.schema import REPO_ROOT, load_hand_spec


def _set_both_hands(env, left_q: np.ndarray, right_q: np.ndarray) -> None:
    env.data.qpos[env.handles.left_hand.qadr] = np.asarray(left_q, dtype=np.float64)
    env.data.qpos[env.handles.right_hand.qadr] = np.asarray(right_q, dtype=np.float64)


def evaluate_l1_mujoco(*, qpos_dataset: np.ndarray | None = None) -> dict[str, Any]:
    import mujoco

    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.mujoco_env.ik import dls_ik_pos

    spec = load_hand_spec()
    lib = GraspLibrary(spec)
    env = CombinedMujocoEnv(scene="empty")
    env.reset()
    right_arm = env.handles.arm_joints["right"]

    primitive_ncon: dict[str, int] = {}
    primitive_limits: dict[str, dict] = {}
    for name, closure in (("open", 0.0), ("power_grasp", 0.85), ("gun_grip", 0.75)):
        q = lib.q_active(name, closure)
        _set_both_hands(env, q, q)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        primitive_ncon[name] = int(env.data.ncon)
        primitive_limits[name] = check_joint_limits(q.reshape(1, -1), spec)

    # IK: 3 cm in +Y from the rest pose (T800 default hang cannot lift +Z well).
    r0 = env.xpos("r_wrist").copy()
    tgt = r0 + np.array([0.0, 0.03, 0.0])
    ik = dls_ik_pos(
        env.model,
        env.data,
        "r_wrist",
        tgt,
        right_arm,
        damping=0.05,
        iters=32,
        step=0.55,
        max_delta_rad=0.12,
    )
    mujoco.mj_forward(env.model, env.data)

    q_active_sample = lib.q_active("open", 0.0)
    if qpos_dataset is not None and qpos_dataset.size:
        # Dataset is full-model qpos; check hand slices only.
        left = qpos_dataset[:, env.handles.left_hand.qadr]
        right = qpos_dataset[:, env.handles.right_hand.qadr]
        limits_ds = {
            "left": check_joint_limits(left, spec),
            "right": check_joint_limits(right, spec),
        }
    else:
        limits_ds = None

    report = {
        "kind": "l1_mujoco",
        "limits_primitives": primitive_limits,
        "coupling": check_coupling(q_active_sample, spec),
        "ik": {
            "site": "r_wrist",
            "offset_m": [0.0, 0.03, 0.0],
            "err_m": float(ik["err_m"]),
            "ok": bool(ik["err_m"] <= 0.03),
            "arm_joints": right_arm,
        },
        "ncon_at_primitive": primitive_ncon,
        "self_collision": {
            "backend": "mujoco_ncon",
            "note": (
                "ncon is total contacts in the empty-scene twin (may include "
                "unrelated geoms). Not an FCL self-collision fail."
            ),
            "open": primitive_ncon["open"],
            "power_grasp": primitive_ncon["power_grasp"],
            "gun_grip": primitive_ncon["gun_grip"],
        },
        "dataset_hand_limits": limits_ds,
        "policy_eval_forbidden": True,
        "note": "MuJoCo L1. IK residual is DLS on r_wrist, not Pinocchio.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/report/generated/l1_mujoco.json")
    parser.add_argument("--dataset", default="", help="Optional recorded npz with qpos")
    args = parser.parse_args()
    qpos = None
    if args.dataset:
        qpos = np.load(args.dataset)["qpos"]
    report = evaluate_l1_mujoco(qpos_dataset=qpos)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ik_err_m": report["ik"]["err_m"], "ncon": report["ncon_at_primitive"]}, indent=2))


if __name__ == "__main__":
    main()
