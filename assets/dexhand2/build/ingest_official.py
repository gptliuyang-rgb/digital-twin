"""Parse official Wuji Hand 2.2 MJCF/URDF. Never invent joint names.

Beta 1 / Beta 2 packages were removed from wuji-description on 2026-10-12.
This loader reads hand2/hand2_2 (default) and checks hand2/hand2_1 joint identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from interface.schema import REPO_ROOT, load_hand_spec, load_joint_map

DEFAULT_UPSTREAM = REPO_ROOT / "third_party" / "wuji-description"
HAND2_2 = Path("hand2/hand2_2/body")
HAND2_1 = Path("hand2/hand2_1/body")
REMOVED_BETA_PATHS = (
    Path("hand2/hand2_beta1/body"),
    Path("hand2/hand2_beta2/body"),
)


def official_mjcf(
    side: str,
    *,
    variant: str = "default",
    revision: str = "2.2",
    root: Path | None = None,
) -> Path:
    base = HAND2_2 if revision == "2.2" else HAND2_1
    name = f"{side}_with_cover.xml" if variant == "with_cover" else f"{side}.xml"
    return (root or DEFAULT_UPSTREAM) / base / "mjcf" / name


def official_urdf(
    side: str,
    *,
    variant: str = "default",
    revision: str = "2.2",
    root: Path | None = None,
) -> Path:
    base = HAND2_2 if revision == "2.2" else HAND2_1
    name = f"{side}_with_cover.urdf" if variant == "with_cover" else f"{side}.urdf"
    return (root or DEFAULT_UPSTREAM) / base / "urdf" / name


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


def parse_mount_to_wrist(path: Path) -> dict[str, list[float]]:
    """Read {side}_wrist pose under {side}_mount from official MJCF."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<body name="[lr]_wrist" pos="([^"]+)" quat="([^"]+)"',
        text,
    )
    if match is None:
        raise ValueError(f"mount→wrist body not found in {path}")
    pos = [float(x) for x in match.group(1).split()]
    quat = [float(x) for x in match.group(2).split()]
    return {"pos_m": pos, "quat_wxyz": quat}


def urdf_mass_sum(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    return sum(float(x) for x in re.findall(r'<mass value="([^"]+)"', text))


def count_urdf_inertial_links(path: Path) -> int:
    """Count links that carry a mass tag. 2.2 also has massless tip/IMU frames."""
    text = path.read_text(encoding="utf-8")
    return len(re.findall(r'<mass value="', text))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _assert_removed_beta(root: Path) -> list[str]:
    notes = []
    for rel in REMOVED_BETA_PATHS:
        if (root / rel).exists():
            notes.append(f"unexpected leftover path {rel}")
    return notes


def ingest(root: Path | None = None) -> dict:
    root = root or DEFAULT_UPSTREAM
    spec = load_hand_spec()
    joint_map = load_joint_map()
    mjcf_r = official_mjcf("right", root=root)
    if not mjcf_r.is_file():
        raise FileNotFoundError(
            f"Official Hand 2.2 MJCF not found at {mjcf_r}. "
            "Run scripts/bootstrap_resources.sh (wuji-description ≥ v2026.10.12)."
        )
    joints = parse_mjcf_joints(mjcf_r)
    acts = parse_mjcf_actuators(mjcf_r)
    sites = parse_sites(mjcf_r)
    by_joint = {j["name"]: j for j in joints}
    mismatches: list[str] = []
    mismatches.extend(_assert_removed_beta(root))
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

    urdf_r = official_urdf("right", root=root)
    mass = urdf_mass_sum(urdf_r)
    n_links = count_urdf_inertial_links(urdf_r)
    if abs(mass - spec.skeleton_mass_kg) > 1e-4:
        mismatches.append(f"mass {mass} != spec skeleton {spec.skeleton_mass_kg}")

    cover_urdf = official_urdf("right", variant="with_cover", root=root)
    cover_mass = urdf_mass_sum(cover_urdf) if cover_urdf.is_file() else None
    expected_cover = float(spec.raw["sim_mass_kg"]["with_cover"]["value"])
    if cover_mass is not None and abs(cover_mass - expected_cover) > 1e-4:
        mismatches.append(f"cover mass {cover_mass} != spec {expected_cover}")

    mjcf_21 = official_mjcf("right", revision="2.1", root=root)
    names_22 = [j["name"] for j in joints]
    if mjcf_21.is_file():
        names_21 = [j["name"] for j in parse_mjcf_joints(mjcf_21)]
        if names_21 != names_22:
            mismatches.append("hand2_1 joint names differ from hand2_2")
        mass_21 = urdf_mass_sum(official_urdf("right", revision="2.1", root=root))
        if abs(mass_21 - mass) > 1e-4:
            mismatches.append(f"hand2_1 mass {mass_21} != hand2_2 {mass}")

    text = mjcf_r.read_text(encoding="utf-8")
    n_exclude = len(re.findall(r"<exclude ", text))
    n_group2 = len(re.findall(r'group="2"', text))
    n_pad_bodies = len(re.findall(r'tip_sensor_frame"', text))
    mount = parse_mount_to_wrist(mjcf_r)

    return {
        "mjcf": str(mjcf_r),
        "revision": "2.2",
        "n_joints": len(joints),
        "n_actuators": len(acts),
        "n_inertial_links": n_links,
        "n_contact_excludes": n_exclude,
        "n_group2_geoms": n_group2,
        "n_tip_sensor_frame_mentions": n_pad_bodies,
        "sites": sites,
        "skeleton_mass_kg": mass,
        "cover_mass_kg": cover_mass,
        "mount_to_wrist": mount,
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
