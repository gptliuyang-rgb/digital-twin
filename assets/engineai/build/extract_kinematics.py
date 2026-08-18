"""Dump T800 joint kinematics from the official URDF + MJCF ranges.

The committed YAML is a kinematics-only fixture so Case A FK and Sim2Sim MPJPE
can run in CI without cloning the Native SDK. Numbers are copied from the
official files, not invented. Re-run after a SDK bump:

    python -m assets.engineai.build.extract_kinematics --write
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml

from assets.engineai.build.ingest_t800 import SDK, T800_URDF
from interface.schema import REPO_ROOT
from sim.urdf_fk import parse_urdf_joints
from wbc.dims import load_t800_sonic
from wbc.filter import parse_mjcf_joint_limits

OUT = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_kinematics.yaml"
T800_MJCF_LINKS = SDK / "assets/resource/robot/t800/xml/serial_links.xml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(urdf_path: Path | None = None, mjcf_path: Path | None = None) -> dict[str, Any]:
    urdf_path = urdf_path or T800_URDF
    mjcf_path = mjcf_path or T800_MJCF_LINKS
    if not urdf_path.is_file():
        raise FileNotFoundError(f"missing {urdf_path}; run scripts/bootstrap_resources.sh")
    cfg = load_t800_sonic()
    joints = parse_urdf_joints(urdf_path.read_text(encoding="utf-8"))
    order = list(cfg["joint_order"])
    by_name = {j.name: j for j in joints}
    missing = [n for n in order if n not in by_name]
    if missing:
        raise KeyError(f"URDF missing SONIC joint_order entries {missing}")
    limits: dict[str, list[float]] | None = None
    mjcf_sha = None
    if mjcf_path.is_file():
        lim = parse_mjcf_joint_limits(mjcf_path.read_text(encoding="utf-8"), order)
        limits = {name: [float(lim[i, 0]), float(lim[i, 1])] for i, name in enumerate(order)}
        mjcf_sha = _sha256(mjcf_path)
    dumped = []
    for j in joints:
        dumped.append(
            {
                "name": j.name,
                "type": j.joint_type,
                "parent": j.parent,
                "child": j.child,
                "origin_xyz_m": [float(x) for x in j.origin_xyz_m],
                "origin_rpy_rad": [float(x) for x in j.origin_rpy_rad],
                "axis": [float(x) for x in j.axis],
                "lower_rad": None if j.lower_rad is None else float(j.lower_rad),
                "upper_rad": None if j.upper_rad is None else float(j.upper_rad),
            }
        )
    return {
        "schema_version": "1.0",
        "robot": "t800",
        "root_link": cfg["floating_base_link"],
        "source_urdf": str(urdf_path.relative_to(REPO_ROOT)) if urdf_path.is_relative_to(REPO_ROOT) else str(urdf_path),
        "source_urdf_sha256": _sha256(urdf_path),
        "source_mjcf_links": (
            str(mjcf_path.relative_to(REPO_ROOT)) if mjcf_path.is_file() and mjcf_path.is_relative_to(REPO_ROOT) else None
        ),
        "source_mjcf_sha256": mjcf_sha,
        "n_revolute": int(cfg["n_revolute"]),
        "joint_order": order,
        "tracked_bodies": dict(cfg["tracked_bodies"]),
        "optional_elbow_bodies": dict(cfg.get("optional_elbow_bodies") or {}),
        "mjcf_revolute_limits_rad": limits,
        "joints": dumped,
        "note": (
            "Kinematics-only extract of the official Native SDK URDF. "
            "mjcf_revolute_limits_rad comes from serial_links.xml range=, not URDF <limit>. "
            "Not a mesh/inertial dump. Not a G1 model."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    doc = extract()
    if args.write:
        OUT.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(OUT)
        print(f"joints={len(doc['joints'])} revolute={doc['n_revolute']}")
    else:
        print(
            yaml.safe_dump(
                {k: doc[k] for k in ("n_revolute", "root_link", "source_urdf_sha256", "note")},
                sort_keys=False,
            )
        )


if __name__ == "__main__":
    main()
