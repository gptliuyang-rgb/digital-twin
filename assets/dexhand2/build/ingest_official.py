"""Parse official Wuji Hand 2 MJCF/URDF. Never invent joint names."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from interface.schema import REPO_ROOT, load_hand_spec, load_joint_map

DEFAULT_UPSTREAM = REPO_ROOT / "third_party" / "wuji-description"
HAND2 = Path("hand2/hand2_beta1/body")


def official_mjcf(side: str, *, with_mount: bool = False, root: Path | None = None) -> Path:
    name = f"{side}_with_mount.xml" if with_mount else f"{side}.xml"
    return (root or DEFAULT_UPSTREAM) / HAND2 / "mjcf" / name


def official_urdf(side: str, *, with_mount: bool = False, root: Path | None = None) -> Path:
    name = f"{side}_with_mount.urdf" if with_mount else f"{side}.urdf"
    return (root or DEFAULT_UPSTREAM) / HAND2 / "urdf" / name


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


def urdf_mass_sum(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    return sum(float(x) for x in re.findall(r'<mass value="([^"]+)"', text))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def ingest(root: Path | None = None) -> dict:
    root = root or DEFAULT_UPSTREAM
    spec = load_hand_spec()
    joint_map = load_joint_map()
    mjcf_r = official_mjcf("right", root=root)
    if not mjcf_r.is_file():
        raise FileNotFoundError(
            f"Official MJCF not found at {mjcf_r}. Run scripts/bootstrap_resources.sh"
        )
    joints = parse_mjcf_joints(mjcf_r)
    acts = parse_mjcf_actuators(mjcf_r)
    sites = parse_sites(mjcf_r)
    by_joint = {j["name"]: j for j in joints}
    mismatches = []
    for entry in joint_map:
        jname = entry.mjcf_joint("r")
        if jname not in by_joint:
            mismatches.append(f"missing mjcf joint {jname}")
            continue
        lo, hi = spec.joint_limits_rad[entry.canonical]
        got = by_joint[jname]
        if abs(got["lower"] - lo) > 1e-3 or abs(got["upper"] - hi) > 1e-3:
            mismatches.append(f"limit mismatch {jname}: spec=({lo},{hi}) mjcf=({got['lower']},{got['upper']})")
    act_order = [a["joint"] for a in acts]
    expected = [e.mjcf_joint("r") for e in joint_map]
    if act_order != expected:
        mismatches.append(f"actuator order {act_order} != {expected}")
    mass = urdf_mass_sum(official_urdf("right", root=root))
    if abs(mass - spec.skeleton_mass_kg) > 1e-4:
        mismatches.append(f"mass {mass} != spec skeleton {spec.skeleton_mass_kg}")
    return {
        "mjcf": str(mjcf_r),
        "n_joints": len(joints),
        "n_actuators": len(acts),
        "sites": sites,
        "skeleton_mass_kg": mass,
        "sha256": file_sha256(mjcf_r),
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def main() -> None:
    report = ingest()
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
