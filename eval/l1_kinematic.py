"""L1: kinematic checks without physics. IK is optional (Pinocchio)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assets.dexhand2.build.ingest_official import official_mjcf
from hand.coupling import Coupling
from interface.schema import REPO_ROOT, load_frames, load_hand_spec
from sim.urdf_fk import UrdfTree


def _t800_wrist_fk() -> dict | str:
    urdf = (
        REPO_ROOT
        / "third_party"
        / "engineai-native-sdk"
        / "assets"
        / "resource"
        / "robot"
        / "t800"
        / "urdf"
        / "serial_t800.urdf"
    )
    if not urdf.is_file():
        return "skipped_no_t800_urdf"
    frames = load_frames()["frames"]
    tree = UrdfTree.from_path(urdf, root_link=frames["pelvis"]["t800_link"])
    dummy = {}
    for side, parent in (("left", "LINK_ELBOW_YAW_L"), ("right", "LINK_ELBOW_YAW_R")):
        wrist = frames[f"{side}_wrist"]["t800_link"]
        local = UrdfTree.from_path(urdf, root_link=parent)
        pos, rot = local.fk_link(wrist, {})
        dummy[side] = {"parent": parent, "link": wrist, "pos_m": pos.tolist(), "rot_trace": float(np.trace(rot))}
    zero = {}
    for side in ("left", "right"):
        link = frames[f"{side}_wrist"]["t800_link"]
        pos, _rot = tree.fk_link(link, {})
        zero[side] = pos.tolist()
    return {"dummy_wrist_from_elbow": dummy, "zero_pose_wrist_in_base_m": zero}


def _mujoco_zero_sites() -> dict | str:
    try:
        import mujoco
    except ImportError:
        return "skipped_no_mujoco"
    path = official_mjcf("right")
    if not path.is_file():
        return "skipped_no_official_mjcf"
    model = mujoco.MjModel.from_xml_path(path.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sites = {}
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.endswith("_tip"):
            sites[name] = data.site_xpos[i].tolist()
    return {"n_qpos": int(model.nq), "n_nu": int(model.nu), "zero_pose_sites": sites}


def check_joint_limits(q: np.ndarray, spec, margin: float = 0.05) -> dict:
    limits = spec.limits_vector()
    span = limits[:, 1] - limits[:, 0]
    lo = limits[:, 0] + margin * span
    hi = limits[:, 1] - margin * span
    inside = np.all((q >= lo) & (q <= hi), axis=1) if q.ndim == 2 else np.all((q >= lo) & (q <= hi))
    return {"all_inside_margin": bool(np.all(inside)), "n": int(q.shape[0] if q.ndim == 2 else 1)}


def check_coupling(q_active: np.ndarray, spec) -> dict:
    c = Coupling.from_spec(spec)
    full = c.active_to_full(q_active)
    back = c.full_to_active(full)
    err = float(np.max(np.abs(back - q_active)))
    return {"max_roundtrip_rad": err, "ok": err < 1e-9}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval/configs/l1_kinematic.yaml")
    parser.add_argument("--out", default="eval/report/generated/l1.json")
    args = parser.parse_args()
    spec = load_hand_spec()
    rng = np.random.default_rng(1)
    limits = spec.limits_vector()
    q = rng.uniform(limits[:, 0] * 0.2, limits[:, 1] * 0.2, size=(100, spec.n_active_dof))
    report = {
        "limits": check_joint_limits(q, spec),
        "coupling": check_coupling(q[0], spec),
        "ik": "skipped_no_pinocchio",
        "self_collision": "skipped_no_fcl",
        "mujoco_fk": _mujoco_zero_sites(),
        "t800_wrist_fk": _t800_wrist_fk(),
        "note": "Geometry IK/FCL require Pinocchio/FCL. Limit + coupling + URDF FK always run when assets exist.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
