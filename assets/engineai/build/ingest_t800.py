"""Parse EngineAI T800 / T800 Pro URDF joint tables. No invented DoF counts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from interface.schema import REPO_ROOT

SDK = REPO_ROOT / "third_party" / "engineai-native-sdk"
T800_URDF = SDK / "assets/resource/robot/t800/urdf/serial_t800.urdf"
T800PRO_URDF = SDK / "assets/resource/robot/t800pro/urdf/serial_t800pro.urdf"
OUT = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_joints.yaml"


def parse_revolute(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    joints = []
    for match in re.finditer(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', text, re.S):
        body = match.group(2)
        parent = re.search(r'<parent\s+link="([^"]+)"', body)
        child = re.search(r'<child\s+link="([^"]+)"', body)
        if not parent or not child:
            continue
        joints.append({"name": match.group(1), "parent": parent.group(1), "child": child.group(1)})
    return joints


def dummy_wrist_origins(urdf_text: str) -> dict:
    out = {}
    for side, joint in (("left", "J_FIXED_WAIST_L"), ("right", "J_FIXED_WAIST_R")):
        block = re.search(
            rf'<joint name="{joint}" type="fixed">(.*?)</joint>',
            urdf_text,
            re.S,
        )
        if not block:
            continue
        origin = re.search(r'<origin xyz="([^"]+)" rpy="([^"]+)"', block.group(1))
        parent = re.search(r'<parent link="([^"]+)"', block.group(1))
        child = re.search(r'<child link="([^"]+)"', block.group(1))
        if origin and parent and child:
            out[side] = {
                "joint": joint,
                "parent": parent.group(1),
                "child": child.group(1),
                "pos_m": [float(x) for x in origin.group(1).split()],
                "rpy_rad": [float(x) for x in origin.group(2).split()],
            }
    return out


def classify(name: str) -> str:
    u = name.upper()
    if any(k in u for k in ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY", "FINGER")):
        return "builtin_hand"
    if "WRIST" in u:
        return "wrist"
    if any(k in u for k in ("HIP", "KNEE", "ANKLE")):
        return "leg"
    if any(k in u for k in ("SHOULDER", "ELBOW")):
        return "arm"
    if any(k in u for k in ("WAIST", "TORSO")):
        return "waist"
    if "HEAD" in u or "NECK" in u:
        return "head"
    return "other"


def summarize(path: Path, robot: str) -> dict:
    joints = parse_revolute(path)
    by = {}
    for j in joints:
        by.setdefault(classify(j["name"]), []).append(j["name"])
    n_hand = len(by.get("builtin_hand", []))
    return {
        "robot": robot,
        "urdf": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "revolute_count": len(joints),
        "revolute_excluding_builtin_hands": len(joints) - n_hand,
        "groups": {k: v for k, v in by.items()},
    }


def build() -> dict:
    t800 = summarize(T800_URDF, "t800")
    pro = summarize(T800PRO_URDF, "t800pro") if T800PRO_URDF.is_file() else None
    dummy = dummy_wrist_origins(T800_URDF.read_text(encoding="utf-8")) if T800_URDF.is_file() else {}
    doc = {
        "schema_version": "1.0",
        "robot": "t800",
        "urdf": t800["urdf"],
        "revolute_count": t800["revolute_count"],
        "floating_base_link": "LINK_BASE",
        "end_effector_frames": {
            "left_foot": "LINK_FOOT_L",
            "right_foot": "LINK_FOOT_R",
            "waist": "LINK_WAIST_YAW",
            "left_wrist": "LINK_WRIST_END_L",
            "right_wrist": "LINK_WRIST_END_R",
            "head": "LINK_HEAD_YAW",
        },
        "arm_joints": {
            "left": [n for n in t800["groups"].get("arm", []) if n.endswith("_L")],
            "right": [n for n in t800["groups"].get("arm", []) if n.endswith("_R")],
        },
        "t800": t800,
        "t800pro": pro,
        "t800_dummy_wrist_from_elbow": dummy,
        "note": (
            "T800 Native SDK URDF has 25 revolute DoF and dummy LINK_WRIST_END_* "
            "(no wrist pitch/roll). T800 Pro URDF has wrist pitch/roll plus a 7-DoF "
            "built-in hand per side that we replace with Wuji Hand 2. Body DoF excluding "
            "those built-in hands is revolute_excluding_builtin_hands."
        ),
    }
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not T800_URDF.is_file():
        raise SystemExit(f"missing {T800_URDF}; run scripts/bootstrap_resources.sh")
    doc = build()
    if args.write:
        OUT.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(OUT)
    else:
        print(yaml.safe_dump({k: doc[k] for k in ("revolute_count", "t800", "t800pro", "note")}, sort_keys=False))


if __name__ == "__main__":
    main()
