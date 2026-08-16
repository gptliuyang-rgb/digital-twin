"""Emit GMR IK JSON + params overlay from body_map.yaml and t800_sonic.yaml.

Does not import MuJoCo. Does not download mocap. When ``tpose_offsets.yaml``
exists (ADR-015), quat offsets and human_scale_table come from the T-pose
pass; otherwise they stay labelled uncalibrated copies of GMR PM01.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from interface.schema import REPO_ROOT
from wbc.dims import load_t800_sonic

BODY_MAP_PATH = Path(__file__).with_name("body_map.yaml")
TPOSE_OFFSETS_PATH = Path(__file__).with_name("tpose_offsets.yaml")
OUT_DIR = Path(__file__).resolve().parent
SONIC_TRACKED_ROLES = ("pelvis", "head", "left_wrist", "right_wrist", "left_foot", "right_foot")


def load_body_map(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or BODY_MAP_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("body_map.yaml must be a mapping")
    return raw


def load_tpose_offsets(path: Path | None = None) -> dict[str, Any] | None:
    path = path or TPOSE_OFFSETS_PATH
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def _entry(body: dict[str, Any], table: str, *, quat_wxyz: list[float] | None = None) -> list[Any]:
    t = body[table]
    quat = quat_wxyz if quat_wxyz is not None else [float(x) for x in t["quat_wxyz"]]
    return [
        body["human"],
        int(t["pos_w"]),
        int(t["rot_w"]),
        [float(x) for x in t["pos_m"]],
        [float(x) for x in quat],
    ]


def ik_config(
    src: str,
    body_map: dict[str, Any] | None = None,
    tpose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a GMR ik_configs JSON object for smplx or bvh_lafan1."""
    body_map = body_map or load_body_map()
    if tpose is None:
        tpose = load_tpose_offsets()
    block = body_map[src]
    tpose_bodies = (tpose or {}).get(src, {}).get("bodies", {})
    table1 = {}
    table2 = {}
    for body in block["bodies"]:
        overlay = tpose_bodies.get(body["robot"], {})
        table1[body["robot"]] = _entry(body, "table1", quat_wxyz=overlay.get("table1_quat_wxyz"))
        table2[body["robot"]] = _entry(body, "table2", quat_wxyz=overlay.get("table2_quat_wxyz"))
    scale = (tpose or {}).get(src, {}).get("human_scale_table") or block["human_scale_table"]
    payload = {
        "robot_root_name": body_map["robot_root_name"],
        "human_root_name": block["human_root_name"],
        "ground_height": float(block["ground_height_m"]),
        "human_height_assumption": float(block["human_height_assumption_m"]),
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": {k: float(v) for k, v in scale.items()},
        "ik_match_table1": table1,
        "ik_match_table2": table2,
    }
    return payload


def params_overlay(
    body_map: dict[str, Any] | None = None,
    tpose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """How to register T800 in GMR params.py. Paths are repo-relative."""
    body_map = body_map or load_body_map()
    if tpose is None:
        tpose = load_tpose_offsets()
    key = body_map["gmr_robot_key"]
    return {
        "ROBOT_XML_DICT": {key: body_map["mjcf"]},
        "ROBOT_BASE_DICT": {key: body_map["robot_root_name"]},
        "IK_CONFIG_DICT": {
            "smplx": {key: "wbc/gmr/smplx_to_t800.json"},
            "bvh_lafan1": {key: "wbc/gmr/bvh_lafan1_to_t800.json"},
        },
        "VIEWER_CAM_DISTANCE_DICT": {key: 2.5},
        "note": (
            "Copy these entries into third_party/GMR/general_motion_retargeting/params.py "
            "or pass --ik_config / --robot xml explicitly. Do not add T800 under unitree_g1."
        ),
        "quat_offset_status": (tpose or {}).get("quat_offset_status", body_map["quat_offset_status"]),
        "human_scale_status": (tpose or {}).get("human_scale_status", body_map["human_scale_status"]),
        "n_revolute": int(body_map["n_revolute"]),
        "hand_bypass": True,
    }


def sonic_tracked_bodies(body_map: dict[str, Any] | None = None) -> dict[str, str]:
    """role → MJCF body, intersected with t800_sonic.yaml tracked_bodies."""
    body_map = body_map or load_body_map()
    sonic = load_t800_sonic()["tracked_bodies"]
    by_role = {b["role"]: b["robot"] for b in body_map["smplx"]["bodies"]}
    out = {}
    for role in SONIC_TRACKED_ROLES:
        robot = by_role[role]
        expected = sonic[role]
        if robot != expected:
            raise ValueError(
                f"GMR body for {role} is {robot} but t800_sonic.yaml tracks {expected}"
            )
        out[role] = robot
    return out


def validate_robot_bodies(
    body_map: dict[str, Any] | None = None,
    *,
    mjcf_text: str | None = None,
    urdf_text: str | None = None,
) -> list[str]:
    """Return missing body names. Empty list means every GMR body exists in the model."""
    body_map = body_map or load_body_map()
    names = {b["robot"] for src in ("smplx", "bvh_lafan1") for b in body_map[src]["bodies"]}
    missing: list[str] = []
    haystack = (mjcf_text or "") + "\n" + (urdf_text or "")
    if not haystack.strip():
        mjcf = REPO_ROOT / body_map["mjcf"]
        urdf = REPO_ROOT / body_map["urdf"]
        if mjcf.is_file():
            haystack += mjcf.read_text(encoding="utf-8")
        if urdf.is_file():
            haystack += urdf.read_text(encoding="utf-8")
    if not haystack.strip():
        return sorted(names)  # cannot confirm; treat as all missing
    for name in sorted(names):
        if f'name="{name}"' not in haystack:
            missing.append(name)
    return missing


def refuse_pm01_torso_name(ik: dict[str, Any]) -> None:
    """T800 must not ship PM01's LINK_TORSO_YAW — that body does not exist."""
    for table in ("ik_match_table1", "ik_match_table2"):
        if "LINK_TORSO_YAW" in ik.get(table, {}):
            raise ValueError("T800 GMR config must use LINK_WAIST_YAW, not LINK_TORSO_YAW")
        if "LINK_ELBOW_END_L" in ik.get(table, {}) or "LINK_ELBOW_END_R" in ik.get(table, {}):
            raise ValueError("T800 GMR config must use LINK_WRIST_END_*, not LINK_ELBOW_END_*")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_all(out_dir: Path | None = None) -> dict[str, str]:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body_map = load_body_map()
    tpose = load_tpose_offsets()
    sonic_tracked_bodies(body_map)
    written: dict[str, str] = {}
    for src, filename in (("smplx", "smplx_to_t800.json"), ("bvh_lafan1", "bvh_lafan1_to_t800.json")):
        ik = ik_config(src, body_map, tpose)
        refuse_pm01_torso_name(ik)
        path = out_dir / filename
        write_json(path, ik)
        written[src] = str(path)
    overlay = params_overlay(body_map, tpose)
    overlay_path = out_dir / "params_overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    written["params_overlay"] = str(overlay_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    written = export_all(args.out)
    missing = validate_robot_bodies()
    print(json.dumps({"written": written, "missing_bodies_in_cloned_model": missing}, indent=2))


if __name__ == "__main__":
    main()
