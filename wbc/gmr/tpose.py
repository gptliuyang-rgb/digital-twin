"""T800 GMR T-pose pass. No MuJoCo import (wbc/ is sim-free).

GMR applies ``R_target = R_human * R_offset`` (motion_retarget.offset_human_data).
At the XML default pose (q = 0) this implies

    R_offset_t800 = R_offset_pm01 * inv(R_pm01) * R_t800

for every body that exists on both robots. EngineAI T800 and GMR's PM01 XML
use body ``pos`` only (no body-level quat), so R_t800 ≈ R_pm01 ≈ I in the
pelvis frame — the copied PM01 offsets are then *verified*, not replaced.

Head is the exception: GMR's PM01 IK table has no head. Official configs that
do track a head (tienkung, hi) use the pelvis/spine offset when the head body
frame matches the pelvis frame at q = 0. We do the same after measuring that
match, instead of shipping identity.

Human scale is PM01's GMR table multiplied by T800/PM01 link-length ratios
measured at q = 0. That is still not a live actor T-pose against BONES-SEED.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import REPO_ROOT
from wbc.gmr.export import load_body_map

TPOSE_OFFSETS_PATH = Path(__file__).with_name("tpose_offsets.yaml")
T800_MJCF = (
    REPO_ROOT / "third_party/engineai-native-sdk/assets/resource/robot/t800/xml/serial_t800.xml"
)
PM01_MJCF = REPO_ROOT / "third_party/GMR/assets/engineai_pm01/xml/serial_pm_v2.xml"
PM01_IK = REPO_ROOT / "third_party/GMR/general_motion_retargeting/ik_configs/smplx_to_pm01.json"


def _pm01_scales() -> tuple[float, float]:
    """Leg/arm human_scale from GMR's official PM01 IK config. Not invented."""
    if not PM01_IK.is_file():
        raise FileNotFoundError(PM01_IK)
    raw = json.loads(PM01_IK.read_text(encoding="utf-8"))
    table = raw["human_scale_table"]
    return float(table["pelvis"]), float(table["left_wrist"])


# T800 GMR body → PM01 body used to borrow the human-frame offset.
T800_TO_PM01: dict[str, str] = {
    "LINK_BASE": "LINK_BASE",
    "LINK_HIP_ROLL_L": "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L": "LINK_KNEE_PITCH_L",
    "LINK_FOOT_L": "LINK_FOOT_L",
    "LINK_HIP_ROLL_R": "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R": "LINK_KNEE_PITCH_R",
    "LINK_FOOT_R": "LINK_FOOT_R",
    "LINK_WAIST_YAW": "LINK_TORSO_YAW",
    "LINK_SHOULDER_ROLL_L": "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_PITCH_L": "LINK_ELBOW_PITCH_L",
    "LINK_WRIST_END_L": "LINK_ELBOW_END_L",
    "LINK_SHOULDER_ROLL_R": "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_PITCH_R": "LINK_ELBOW_PITCH_R",
    "LINK_WRIST_END_R": "LINK_ELBOW_END_R",
    "LINK_HEAD_YAW": "LINK_HEAD_YAW",
}


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Shepperd's method. Returns wxyz, w >= 0."""
    m = np.asarray(rot, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q / (np.linalg.norm(q) + 1e-15)
    if q[0] < 0:
        q = -q
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz. Equivalent to R(a) @ R(b)."""
    wa, xa, ya, za = (float(v) for v in a)
    wb, xb, yb, zb = (float(v) for v in b)
    q = np.array(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dtype=np.float64,
    )
    q = q / (np.linalg.norm(q) + 1e-15)
    if q[0] < 0:
        q = -q
    return q


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def geodesic_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    rel = ra.T @ rb
    c = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def compose_offset(offset_pm01_wxyz: np.ndarray, r_pm01: np.ndarray, r_t800: np.ndarray) -> np.ndarray:
    """R_off_t800 = R_off_pm01 @ inv(R_pm01) @ R_t800  (GMR left-multiply convention)."""
    r_off = quat_wxyz_to_matrix(offset_pm01_wxyz)
    r_new = r_off @ r_pm01.T @ r_t800
    return matrix_to_quat_wxyz(r_new)


def _euler_matrix(angles: list[float], seq: str) -> np.ndarray:
    """MuJoCo compiler eulerseq: successive elemental rotations, intrinsic."""
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    rot = np.eye(3)
    for axis_name, ang in zip(seq.lower(), angles, strict=True):
        x, y, z = axes[axis_name]
        c, s = math.cos(ang), math.sin(ang)
        k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        rot = rot @ (np.eye(3) + s * k + (1.0 - c) * (k @ k))
    return rot


def _expand_includes(elem: ET.Element, base_dir: Path) -> None:
    for child in list(elem):
        if child.tag == "include":
            inc_path = (base_dir / child.attrib["file"]).resolve()
            inc_root = ET.parse(inc_path).getroot()
            _expand_includes(inc_root, inc_path.parent)
            idx = list(elem).index(child)
            elem.remove(child)
            inserted = []
            if inc_root.tag == "mujoco":
                for sub in list(inc_root):
                    if sub.tag == "worldbody":
                        inserted.extend(list(sub))
                    elif sub.tag == "body":
                        inserted.append(sub)
            else:
                inserted.append(inc_root)
            for node in inserted:
                elem.insert(idx, node)
                idx += 1
        else:
            _expand_includes(child, base_dir)


def load_mjcf_root(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    _expand_includes(root, path.parent)
    return root


def _compiler_eulerseq(root: ET.Element) -> str:
    compiler = root.find("compiler")
    if compiler is not None and "eulerseq" in compiler.attrib:
        return compiler.attrib["eulerseq"]
    return "xyz"


def _body_local(elem: ET.Element, eulerseq: str) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([float(x) for x in elem.attrib.get("pos", "0 0 0").split()], dtype=np.float64)
    if "quat" in elem.attrib:
        q = np.array([float(x) for x in elem.attrib["quat"].split()], dtype=np.float64)
        rot = quat_wxyz_to_matrix(q)
    elif "euler" in elem.attrib:
        ang = [float(x) for x in elem.attrib["euler"].split()]
        rot = _euler_matrix(ang, eulerseq)
    else:
        rot = np.eye(3)
    return pos, rot


def body_frames_q0(mjcf_path: Path) -> dict[str, dict[str, Any]]:
    """World pose of every <body> at q = 0 (joint motion is the identity)."""
    root = load_mjcf_root(mjcf_path)
    eulerseq = _compiler_eulerseq(root)
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"no worldbody in {mjcf_path}")
    out: dict[str, dict[str, Any]] = {}

    def walk(elem: ET.Element, p_w: np.ndarray, r_w: np.ndarray) -> None:
        for child in elem:
            if child.tag != "body":
                continue
            p_l, r_l = _body_local(child, eulerseq)
            r = r_w @ r_l
            p = p_w + r_w @ p_l
            name = child.attrib.get("name")
            if name:
                out[name] = {
                    "pos_m": p,
                    "rot": r,
                    "quat_wxyz": matrix_to_quat_wxyz(r),
                }
            walk(child, p, r)

    walk(world, np.zeros(3), np.eye(3))
    return out


def _dist(frames: dict[str, dict[str, Any]], a: str, b: str) -> float:
    return float(np.linalg.norm(frames[a]["pos_m"] - frames[b]["pos_m"]))


def require_clones() -> None:
    missing = [str(p) for p in (T800_MJCF, PM01_MJCF, PM01_IK) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "T-pose pass needs official MJCF clones (not committed):\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/bootstrap_resources.sh and clone GMR into third_party/GMR."
        )


def compute_tpose(body_map: dict[str, Any] | None = None) -> dict[str, Any]:
    require_clones()
    body_map = body_map or load_body_map()
    t800 = body_frames_q0(T800_MJCF)
    pm01 = body_frames_q0(PM01_MJCF)
    for name in T800_TO_PM01:
        if name not in t800:
            raise KeyError(f"T800 MJCF missing body {name}")
        pm_name = T800_TO_PM01[name]
        if pm_name not in pm01:
            raise KeyError(f"PM01 MJCF missing body {pm_name} (mapped from {name})")

    head_vs_pelvis_deg = geodesic_deg(t800["LINK_BASE"]["rot"], t800["LINK_HEAD_YAW"]["rot"])
    torso_vs_pelvis_deg = geodesic_deg(t800["LINK_BASE"]["rot"], t800["LINK_WAIST_YAW"]["rot"])

    sources: dict[str, dict[str, Any]] = {}
    max_frame_delta_deg = 0.0
    for src in ("smplx", "bvh_lafan1"):
        by_robot = {b["robot"]: b for b in body_map[src]["bodies"]}
        pelvis_off_t1 = np.array(by_robot["LINK_BASE"]["table1"]["quat_wxyz"], dtype=np.float64)
        pelvis_off_t2 = np.array(by_robot["LINK_BASE"]["table2"]["quat_wxyz"], dtype=np.float64)
        bodies_out = {}
        for robot, pm_name in T800_TO_PM01.items():
            entry = by_robot[robot]
            r_t = t800[robot]["rot"]
            r_p = pm01[pm_name]["rot"]
            delta = geodesic_deg(r_p, r_t)
            max_frame_delta_deg = max(max_frame_delta_deg, delta)
            if robot == "LINK_HEAD_YAW":
                # PM01 IK table has no head. Use pelvis offset after frame composition.
                off1 = compose_offset(pelvis_off_t1, t800["LINK_BASE"]["rot"], r_t)
                off2 = compose_offset(pelvis_off_t2, t800["LINK_BASE"]["rot"], r_t)
                method = "pelvis_offset_composed_with_head_vs_pelvis_q0"
            else:
                off1 = compose_offset(
                    np.array(entry["table1"]["quat_wxyz"], dtype=np.float64), r_p, r_t
                )
                off2 = compose_offset(
                    np.array(entry["table2"]["quat_wxyz"], dtype=np.float64), r_p, r_t
                )
                method = "pm01_offset_composed_with_t800_vs_pm01_q0"
            bodies_out[robot] = {
                "pm01_body": pm_name,
                "method": method,
                "table1_quat_wxyz": off1.tolist(),
                "table2_quat_wxyz": off2.tolist(),
                "copied_table1_quat_wxyz": [float(x) for x in entry["table1"]["quat_wxyz"]],
                "frame_delta_vs_pm01_deg": delta,
                "t800_q0_quat_wxyz": t800[robot]["quat_wxyz"].tolist(),
                "pm01_q0_quat_wxyz": pm01[pm_name]["quat_wxyz"].tolist(),
            }
        sources[src] = {"bodies": bodies_out}

    # Link lengths at q=0 (pelvis/base frame).
    t800_leg = 0.5 * (
        _dist(t800, "LINK_HIP_ROLL_L", "LINK_FOOT_L") + _dist(t800, "LINK_HIP_ROLL_R", "LINK_FOOT_R")
    )
    pm01_leg = 0.5 * (
        _dist(pm01, "LINK_HIP_ROLL_L", "LINK_FOOT_L") + _dist(pm01, "LINK_HIP_ROLL_R", "LINK_FOOT_R")
    )
    t800_arm = 0.5 * (
        _dist(t800, "LINK_SHOULDER_ROLL_L", "LINK_WRIST_END_L")
        + _dist(t800, "LINK_SHOULDER_ROLL_R", "LINK_WRIST_END_R")
    )
    pm01_arm = 0.5 * (
        _dist(pm01, "LINK_SHOULDER_ROLL_L", "LINK_ELBOW_END_L")
        + _dist(pm01, "LINK_SHOULDER_ROLL_R", "LINK_ELBOW_END_R")
    )
    ratio_leg = t800_leg / pm01_leg
    ratio_arm = t800_arm / pm01_arm
    pm01_leg_scale, pm01_arm_scale = _pm01_scales()
    scale_leg = pm01_leg_scale * ratio_leg
    scale_arm = pm01_arm_scale * ratio_arm

    smplx_scale = {
        "pelvis": scale_leg,
        "spine3": scale_leg,
        "left_hip": scale_leg,
        "right_hip": scale_leg,
        "left_knee": scale_leg,
        "right_knee": scale_leg,
        "left_foot": scale_leg,
        "right_foot": scale_leg,
        "left_shoulder": scale_arm,
        "right_shoulder": scale_arm,
        "left_elbow": scale_arm,
        "right_elbow": scale_arm,
        "left_wrist": scale_arm,
        "right_wrist": scale_arm,
        "head": scale_leg,
    }
    bvh_scale = {
        "Hips": scale_leg,
        "Spine2": scale_leg,
        "LeftUpLeg": scale_leg,
        "RightUpLeg": scale_leg,
        "LeftLeg": scale_leg,
        "RightLeg": scale_leg,
        "LeftFootMod": scale_leg,
        "RightFootMod": scale_leg,
        "LeftArm": scale_arm,
        "RightArm": scale_arm,
        "LeftForeArm": scale_arm,
        "RightForeArm": scale_arm,
        "LeftHand": scale_arm,
        "RightHand": scale_arm,
        "Head": scale_leg,
    }
    sources["smplx"]["human_scale_table"] = {k: float(v) for k, v in smplx_scale.items()}
    sources["bvh_lafan1"]["human_scale_table"] = {k: float(v) for k, v in bvh_scale.items()}

    return {
        "schema_version": "1.0",
        "quat_offset_status": "tpose_composed_pm01_with_t800_q0_frames",
        "human_scale_status": "tpose_scaled_from_pm01_link_lengths",
        "method": (
            "R_offset_t800 = R_offset_pm01 * inv(R_pm01_q0) * R_t800_q0. "
            "Head uses the pelvis offset composed with inv(R_pelvis)*R_head "
            "(GMR tienkung/hi convention when head frame matches pelvis at q=0)."
        ),
        "not_a_live_actor_tpose": True,
        "head_vs_pelvis_deg": head_vs_pelvis_deg,
        "torso_vs_pelvis_deg": torso_vs_pelvis_deg,
        "max_t800_vs_pm01_frame_delta_deg": max_frame_delta_deg,
        "lengths_m": {
            "t800_hip_to_foot": t800_leg,
            "pm01_hip_to_foot": pm01_leg,
            "t800_shoulder_to_wrist": t800_arm,
            "pm01_shoulder_to_elbow_end": pm01_arm,
            "ratio_leg": ratio_leg,
            "ratio_arm": ratio_arm,
        },
        "pm01_gmr_scale": {"leg": pm01_leg_scale, "arm": pm01_arm_scale},
        "smplx": sources["smplx"],
        "bvh_lafan1": sources["bvh_lafan1"],
        "t800_mjcf": str(T800_MJCF.relative_to(REPO_ROOT)),
        "pm01_mjcf": str(PM01_MJCF.relative_to(REPO_ROOT)),
    }


def write_tpose_offsets(report: dict[str, Any] | None = None, path: Path | None = None) -> Path:
    report = report or compute_tpose()
    path = path or TPOSE_OFFSETS_PATH
    path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return path


def load_tpose_offsets(path: Path | None = None) -> dict[str, Any] | None:
    path = path or TPOSE_OFFSETS_PATH
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write tpose_offsets.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = compute_tpose()
    if args.write:
        write_tpose_offsets(report)
    summary = {
        "quat_offset_status": report["quat_offset_status"],
        "human_scale_status": report["human_scale_status"],
        "head_vs_pelvis_deg": report["head_vs_pelvis_deg"],
        "max_t800_vs_pm01_frame_delta_deg": report["max_t800_vs_pm01_frame_delta_deg"],
        "lengths_m": report["lengths_m"],
        "smplx_head_table1": report["smplx"]["bodies"]["LINK_HEAD_YAW"]["table1_quat_wxyz"],
        "smplx_scale_pelvis": report["smplx"]["human_scale_table"]["pelvis"],
        "smplx_scale_wrist": report["smplx"]["human_scale_table"]["left_wrist"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
