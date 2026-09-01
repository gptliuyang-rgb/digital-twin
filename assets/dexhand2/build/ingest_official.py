"""Parse official Wuji Hand 2 MJCF/URDF. Never invent joint names.

Default revision is `sim_model_revision` from dexhand2_spec.yaml (hand2_beta2).
Beta 1 remains loadable for drift comparison.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from interface.schema import REPO_ROOT, load_hand_spec, load_joint_map

DEFAULT_UPSTREAM = REPO_ROOT / "third_party" / "wuji-description"
PAD_FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def revision_body_path(revision: str, spec_raw: dict[str, Any] | None = None) -> Path:
    raw = spec_raw if spec_raw is not None else load_hand_spec().raw
    facts = (raw.get("revisions") or {}).get(revision)
    if not isinstance(facts, dict) or "body_path" not in facts:
        raise ValueError(f"unknown Hand 2 revision {revision!r}")
    return Path(str(facts["body_path"]))


def official_mjcf(
    side: str,
    *,
    with_mount: bool = False,
    root: Path | None = None,
    revision: str | None = None,
) -> Path:
    spec = load_hand_spec()
    rev = revision or spec.sim_model_revision
    name = f"{side}_with_mount.xml" if with_mount else f"{side}.xml"
    return (root or DEFAULT_UPSTREAM) / revision_body_path(rev, spec.raw) / "mjcf" / name


def official_urdf(
    side: str,
    *,
    with_mount: bool = False,
    root: Path | None = None,
    revision: str | None = None,
) -> Path:
    spec = load_hand_spec()
    rev = revision or spec.sim_model_revision
    name = f"{side}_with_mount.urdf" if with_mount else f"{side}.urdf"
    return (root or DEFAULT_UPSTREAM) / revision_body_path(rev, spec.raw) / "urdf" / name


def parse_mjcf_joints(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    joints = []
    for match in re.finditer(
        r'<joint name="([^"]+)"[^>]*range="([^"]+)"[^>]*actuatorfrcrange="([^"]+)"',
        text,
    ):
        lo, hi = (float(x) for x in match.group(2).split())
        fr = float(match.group(3).split()[1])
        joints.append({"name": match.group(1), "lower": lo, "upper": hi, "forcerange": fr})
    return joints


def parse_mjcf_actuators(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    acts = []
    for match in re.finditer(
        r'<position name="([^"]+)" joint="([^"]+)" kp="([^"]+)" kv="([^"]+)"',
        text,
    ):
        acts.append(
            {
                "actuator": match.group(1),
                "joint": match.group(2),
                "kp": float(match.group(3)),
                "kv": float(match.group(4)),
            }
        )
    return acts


def parse_sites(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    sites = []
    for match in re.finditer(r'<site name="([^"]+)" pos="([^"]+)"', text):
        xyz = [float(x) for x in match.group(2).split()]
        sites.append({"name": match.group(1), "pos": xyz})
    return sites


def parse_inertial_masses(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    masses: dict[str, float] = {}
    for match in re.finditer(
        r'<body name="([^"]+)"[^>]*>\s*<inertial[^>]*mass="([^"]+)"',
        text,
    ):
        masses[match.group(1)] = float(match.group(2))
    return masses


def parse_bodies(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r'<body name="([^"]+)"', text)


def urdf_mass_sum(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    return sum(float(x) for x in re.findall(r'<mass value="([^"]+)"', text))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _pad_checks(revision: str, facts: dict[str, Any], masses: dict[str, float], bodies: list[str]) -> list[str]:
    mismatches: list[str] = []
    expected_n = int(facts["n_pad_bodies"])
    pad_bodies = [name for name in bodies if name.endswith("_tip_sensor_frame")]
    if len(pad_bodies) != expected_n:
        mismatches.append(f"{revision} pad bodies {pad_bodies} != n_pad_bodies {expected_n}")
    pad_masses = facts.get("pad_mass_kg") or {}
    for finger in PAD_FINGERS:
        body = f"r_{finger}_tip_sensor_frame"
        if expected_n == 0:
            if body in masses:
                mismatches.append(f"{revision} unexpected pad body {body}")
            continue
        if body not in masses:
            mismatches.append(f"{revision} missing pad inertial {body}")
            continue
        want = float(pad_masses[finger])
        if abs(masses[body] - want) > 1e-6:
            mismatches.append(f"{revision} pad mass {body}={masses[body]} != {want}")
    distal_masses = facts.get("distal_mass_kg") or {}
    for finger, want in distal_masses.items():
        body = f"r_{finger}_distal"
        got = masses.get(body)
        if got is None:
            mismatches.append(f"{revision} missing distal {body}")
        elif abs(got - float(want)) > 1e-6:
            mismatches.append(f"{revision} distal mass {body}={got} != {want}")
    return mismatches


def ingest(root: Path | None = None, *, revision: str | None = None) -> dict:
    root = root or DEFAULT_UPSTREAM
    spec = load_hand_spec()
    joint_map = load_joint_map()
    rev = revision or spec.sim_model_revision
    facts = (spec.raw.get("revisions") or {}).get(rev)
    if not isinstance(facts, dict):
        raise ValueError(f"spec.revisions missing {rev}")
    mjcf_r = official_mjcf("right", root=root, revision=rev)
    if not mjcf_r.is_file():
        raise FileNotFoundError(
            f"Official MJCF not found at {mjcf_r}. Run scripts/bootstrap_resources.sh"
        )
    joints = parse_mjcf_joints(mjcf_r)
    acts = parse_mjcf_actuators(mjcf_r)
    sites = parse_sites(mjcf_r)
    bodies = parse_bodies(mjcf_r)
    masses = parse_inertial_masses(mjcf_r)
    by_joint = {j["name"]: j for j in joints}
    mismatches: list[str] = []
    for entry in joint_map:
        jname = entry.mjcf_joint("r")
        if jname not in by_joint:
            mismatches.append(f"missing mjcf joint {jname}")
            continue
        lo, hi = spec.joint_limits_rad[entry.canonical]
        got = by_joint[jname]
        if abs(got["lower"] - lo) > 1e-3 or abs(got["upper"] - hi) > 1e-3:
            mismatches.append(
                f"limit mismatch {jname}: spec=({lo},{hi}) mjcf=({got['lower']},{got['upper']})"
            )
    act_order = [a["joint"] for a in acts]
    expected = [e.mjcf_joint("r") for e in joint_map]
    if act_order != expected:
        mismatches.append(f"actuator order {act_order} != {expected}")
    urdf = official_urdf("right", root=root, revision=rev)
    mass = urdf_mass_sum(urdf)
    want_mass = float(facts["sim_no_mount_mass_kg"])
    if abs(mass - want_mass) > 1e-4:
        mismatches.append(f"mass {mass} != spec revision sim_no_mount_mass_kg {want_mass}")
    mjcf_mass = sum(masses.values())
    if abs(mjcf_mass - want_mass) > 1e-4:
        mismatches.append(f"mjcf inertial sum {mjcf_mass} != {want_mass}")
    if len(bodies) != int(facts["n_bodies_no_mount"]):
        mismatches.append(f"n_bodies {len(bodies)} != {facts['n_bodies_no_mount']}")
    mismatches.extend(_pad_checks(rev, facts, masses, bodies))
    model = re.search(r'<mujoco model="([^"]+)"', mjcf_r.read_text(encoding="utf-8"))
    want_model = facts.get("mjcf_model_right")
    if want_model and model and model.group(1) != want_model:
        mismatches.append(f"model {model.group(1)} != {want_model}")
    return {
        "revision": rev,
        "mjcf": str(mjcf_r),
        "n_joints": len(joints),
        "n_actuators": len(acts),
        "n_bodies": len(bodies),
        "n_pad_bodies": sum(1 for name in bodies if name.endswith("_tip_sensor_frame")),
        "sites": sites,
        "sim_mass_kg": mass,
        "pad_collision_in_official_model": bool(facts["pad_collision_in_official_model"]),
        "sha256": file_sha256(mjcf_r),
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def ingest_all(root: Path | None = None) -> dict[str, dict]:
    spec = load_hand_spec()
    reports = {}
    for rev in (spec.raw.get("revisions") or {}):
        try:
            reports[rev] = ingest(root, revision=rev)
        except FileNotFoundError as exc:
            reports[rev] = {"revision": rev, "ok": False, "mismatches": [str(exc)]}
    return reports


def main() -> None:
    reports = ingest_all()
    print(json.dumps(reports, indent=2))
    if any(not item["ok"] for item in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
