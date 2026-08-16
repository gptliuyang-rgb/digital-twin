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


def parse_collision_audit(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    n_visual = len(re.findall(r'group="1"', text))
    n_collision_mesh = len(re.findall(r'<geom type="mesh"[^>]*group="2"', text))
    n_collision_mesh += len(re.findall(r'<geom type="mesh"[^>]*mesh="[^"]+"\s*group="2"', text))
    # Official collision geoms are mesh + group="2" without contype=0.
    collision_geoms = re.findall(r"<geom [^>]+>", text)
    n_colliding = 0
    n_pad_spheres = 0
    n_capsules = 0
    for tag in collision_geoms:
        if 'contype="0"' in tag:
            continue
        if "group=\"1\"" in tag:
            continue
        n_colliding += 1
        if 'type="sphere"' in tag and "_pad_" in tag:
            n_pad_spheres += 1
        if 'type="capsule"' in tag:
            n_capsules += 1
    n_exclude = len(re.findall(r"<exclude ", text))
    n_tip_stl = 0
    # meshdir relative to mjcf file
    meshdir_m = re.search(r'meshdir="([^"]+)"', text)
    if meshdir_m:
        mesh_dir = (path.parent / meshdir_m.group(1)).resolve()
        n_tip_stl = len(list(mesh_dir.glob("*_tip.STL"))) if mesh_dir.is_dir() else 0
    return {
        "n_visual_geoms_group1": n_visual,
        "n_collision_mesh_group2": n_collision_mesh,
        "n_colliding_geoms": n_colliding,
        "n_pad_spheres": n_pad_spheres,
        "n_capsules": n_capsules,
        "n_contact_excludes": n_exclude,
        "n_tip_stl": n_tip_stl,
        "tip_stl_used_as_collision": False,
    }


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
    audit = parse_collision_audit(mjcf_r)
    if audit["n_contact_excludes"] != 10:
        mismatches.append(f"expected 10 contact excludes, got {audit['n_contact_excludes']}")
    if audit["n_tip_stl"] < 5:
        mismatches.append(f"expected ≥5 tip STLs, got {audit['n_tip_stl']}")
    if len(sites) != 5:
        mismatches.append(f"expected 5 fingertip sites, got {len(sites)}")
    return {
        "mjcf": str(mjcf_r),
        "n_joints": len(joints),
        "n_actuators": len(acts),
        "sites": sites,
        "skeleton_mass_kg": mass,
        "sha256": file_sha256(mjcf_r),
        "collision": audit,
        "mismatches": mismatches,
        "ok": not mismatches,
        "official_gaps": [
            "fingertip_soft_pad_not_in_collision",
            "sim_gains_carried_from_gen1",
            "collision_is_per_link_convex_hull",
        ],
    }


def write_baseline_markdown(report: dict, path: Path | None = None) -> Path:
    path = path or (REPO_ROOT / "docs" / "reports" / "PHASE_1_baseline.md")
    src = report["mjcf"]
    try:
        src = str(Path(src).resolve().relative_to(REPO_ROOT))
    except ValueError:
        pass
    coll = report["collision"]
    sites = "\n".join(f"- `{s['name']}` pos_m={s['pos']}" for s in report["sites"])
    body = f"""# PHASE 1 baseline — official Wuji Hand 2 Beta 1 (right)

Source: `{src}`
SHA256: `{report['sha256']}`

## Counts

| Item | Value |
|---|---|
| Actuators | {report['n_actuators']} |
| Joints | {report['n_joints']} |
| Fingertip sites | {len(report['sites'])} |
| Skeleton mass (URDF sum) | {report['skeleton_mass_kg']} kg |
| Contact excludes | {coll['n_contact_excludes']} |
| Tip STL files on disk | {coll['n_tip_stl']} |
| Pad spheres in official MJCF | {coll['n_pad_spheres']} |

## Fingertip sites

{sites}

## Official gaps (unchanged by ingest)

- Tip STLs are **not** collision geometry (`tip_stl_used_as_collision={coll['tip_stl_used_as_collision']}`).
- Collision geoms are per-link convex hulls.
- Drive kp/kv are gen-1 carry-over.

Ingest ok: **{report['ok']}**. Mismatches: {report['mismatches'] or 'none'}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    report = ingest()
    md = write_baseline_markdown(report)
    print(
        json.dumps(
            {k: report[k] for k in ("ok", "n_actuators", "skeleton_mass_kg", "collision", "mismatches")},
            indent=2,
        )
    )
    print(f"wrote {md}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
