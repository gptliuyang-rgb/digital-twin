"""Assemble EngineAI T800 + Wuji Hand 2 into one MJCF/URDF digital twin.

CAD flange SE(3) is still REQUIRED_INPUT. This welds `{l,r}_mount` onto
`LINK_WRIST_END_*` with the identity listed under `kinematic_bringup_identity`.
That weld is forbidden as a CAD substitute for SONIC / VLA / sim2real claims
(ADR-004, ADR-006).
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from assets.combined.mjcf_xml import dump_mjcf, find_body, find_or_create, load_mjcf
from assets.dexhand2.build.ingest_official import official_mjcf, official_urdf
from assets.engineai.paths import t800_mjcf, t800_root, t800_urdf
from hand.calibration.contact_mujoco import contact_is_calibrated
from interface.schema import REPO_ROOT, REQUIRED_INPUT_TOKEN, find_required_inputs

MOUNT = REPO_ROOT / "assets" / "dexhand2" / "meta" / "mount_transform.yaml"
GENERATED = REPO_ROOT / "assets" / "combined" / "generated"

_POSITION_ACT = re.compile(
    r"<position\s+name=\"([^\"]+)\"\s+joint=\"([^\"]+)\"\s+kp=\"([^\"]+)\"\s+"
    r"kv=\"([^\"]+)\"\s+ctrlrange=\"([^\"]+)\"\s+forcerange=\"([^\"]+)\"\s*/>",
    re.MULTILINE,
)


def mount_ready() -> bool:
    raw = yaml.safe_load(MOUNT.read_text(encoding="utf-8"))
    missing = find_required_inputs(raw.get("t800_wrist_to_hand_mount", {}))
    return not missing


def kinematic_bringup() -> dict[str, Any]:
    raw = yaml.safe_load(MOUNT.read_text(encoding="utf-8"))
    block = raw.get("kinematic_bringup_identity") or {}
    if not block.get("enabled"):
        raise SystemExit(
            "kinematic_bringup_identity.enabled is false and CAD flange is still "
            f"{REQUIRED_INPUT_TOKEN}. Fill t800_wrist_to_hand_mount or enable bring-up."
        )
    return block


def combined_paths() -> dict[str, Any]:
    return {
        "t800_urdf": str(t800_urdf()),
        "t800_mjcf": str(t800_mjcf()),
        "hand_right_mjcf": str(official_mjcf("right", with_mount=True)),
        "hand_left_mjcf": str(official_mjcf("left", with_mount=True)),
        "wrist_links": ["LINK_WRIST_END_L", "LINK_WRIST_END_R"],
        "mount_ready": mount_ready(),
        "kinematic_bringup": bool((yaml.safe_load(MOUNT.read_text(encoding="utf-8")) or {}).get(
            "kinematic_bringup_identity", {}
        ).get("enabled")),
        "policy_eval_forbidden": True,
    }


def convert_position_actuators(xml_text: str) -> tuple[str, list[dict[str, Any]]]:
    """Turn official <position> actuators into unit-gain <motor> plants. Collect kp/kv."""
    gains: list[dict[str, Any]] = []

    def _repl(match: re.Match[str]) -> str:
        name, joint, kp, kv, _ctrl, force = match.groups()
        gains.append(
            {
                "name": name,
                "joint": joint,
                "kp": float(kp),
                "kd": float(kv),
                "tau_lim": abs(float(force.split()[1])),
            }
        )
        return (
            f'<motor name="{name}" joint="{joint}" gear="1" ctrllimited="true" '
            f'ctrlrange="{force}" forcelimited="true" forcerange="{force}"/>'
        )

    new, n = _POSITION_ACT.subn(_repl, xml_text)
    if n == 0:
        raise ValueError("no <position> actuators found — official Hand 2 MJCF changed")
    return new, gains


def _inject_pad_spheres(xml_text: str, side: str, spec_raw: dict[str, Any] | None = None) -> str:
    from assets.dexhand2.build.gen_derived import (
        fit_pad_spheres,
        inject_spheres,
        parse_sites,
        read_stl_vertices,
    )
    from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM
    from hand.calibration.contact_mujoco import apply_contact_to_xml

    prefix = "r" if side == "right" else "l"
    mesh_dir = DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/meshes" / side
    src = official_mjcf(side, with_mount=True)
    for site in parse_sites(src):
        # site name is {l|r}_{finger}_tip
        finger = site["name"][len(prefix) + 1 :].replace("_tip", "")
        stl = mesh_dir / f"{prefix}_{finger}_tip.STL"
        if not stl.is_file():
            continue
        try:
            xml_text = inject_spheres(xml_text, site["name"], fit_pad_spheres(read_stl_vertices(stl)))
        except ValueError:
            continue
    if spec_raw:
        xml_text = apply_contact_to_xml(xml_text, spec_raw)
    return xml_text


def _load_hand_mjcf(
    side: str,
    *,
    mit_motors: bool,
    pad_spheres: bool,
    spec_raw: dict[str, Any] | None = None,
) -> tuple[ET.Element, list[dict[str, Any]]]:
    from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM

    path = official_mjcf(side, with_mount=True)
    text = path.read_text(encoding="utf-8")
    if pad_spheres:
        text = _inject_pad_spheres(text, side, spec_raw=spec_raw)
    gains: list[dict[str, Any]] = []
    if mit_motors:
        text, gains = convert_position_actuators(text)
    mesh_dir = (DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/meshes" / side).resolve()
    text = re.sub(r'meshdir="[^"]+"', f'meshdir="{mesh_dir.as_posix()}"', text)
    tmp = path.parent / f"_resolved_{side}_with_mount.xml"
    tmp.write_text(text, encoding="utf-8")
    try:
        root = load_mjcf(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return root, gains


def assemble_mjcf(
    *,
    pin_base: bool = True,
    mit_motors: bool = True,
    pad_spheres: bool = True,
    timestep_s: float = 0.001,
    spec_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    from interface.schema import load_hand_spec

    bringup = kinematic_bringup()
    spec_raw = load_hand_spec(spec_path).raw if spec_path is not None else load_hand_spec().raw
    pos = " ".join(str(x) for x in bringup["pos_m"])
    quat = " ".join(str(x) for x in bringup["quat_wxyz"])

    robot = load_mjcf(t800_mjcf())
    robot.set("model", "t800_dexhand2_kinematic_bringup")
    option = find_or_create(robot, "option")
    option.set("timestep", str(timestep_s))

    if pin_base:
        base = find_body(robot, "LINK_BASE")
        for joint in list(base):
            if joint.tag == "freejoint":
                base.remove(joint)

    for cls in ("collision_left_wrist_end", "collision_right_wrist_end"):
        for geom in robot.iter("geom"):
            if geom.get("class") == cls:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    all_gains: list[dict[str, Any]] = []
    asset = find_or_create(robot, "asset")
    actuator = find_or_create(robot, "actuator")
    contact = find_or_create(robot, "contact")

    for side, wrist_name, mount_name in (
        ("left", "LINK_WRIST_END_L", "l_mount"),
        ("right", "LINK_WRIST_END_R", "r_mount"),
    ):
        hand, gains = _load_hand_mjcf(
            side, mit_motors=mit_motors, pad_spheres=pad_spheres, spec_raw=spec_raw
        )
        all_gains.extend(gains)
        hand_asset = hand.find("asset")
        if hand_asset is not None:
            for child in list(hand_asset):
                asset.append(child)
        world = hand.find("worldbody")
        if world is None:
            raise ValueError(f"{side} hand MJCF has no worldbody")
        mount = None
        for body in list(world):
            if body.tag == "body" and body.get("name") == mount_name:
                mount = body
                break
        if mount is None:
            raise ValueError(f"{side} hand missing {mount_name}")
        mount.set("pos", pos)
        mount.set("quat", quat)
        find_body(robot, wrist_name).append(mount)
        hand_act = hand.find("actuator")
        if hand_act is not None:
            for child in list(hand_act):
                actuator.append(child)
        hand_contact = hand.find("contact")
        if hand_contact is not None:
            for child in list(hand_contact):
                contact.append(child)

    xml = dump_mjcf(robot)
    manifest = {
        "kind": "t800_dexhand2_combined",
        "policy_eval_forbidden": True,
        "flange": "kinematic_bringup_identity",
        "cad_ready": mount_ready(),
        "pin_base": pin_base,
        "mit_motors": mit_motors,
        "pad_spheres": pad_spheres,
        "contact_calibrated": contact_is_calibrated(spec_raw) if pad_spheres else False,
        "timestep_s": timestep_s,
        "hand_mit_gains": all_gains,
        "n_hand_actuators": len(all_gains),
        "paths": combined_paths(),
    }
    return xml, manifest


def assemble_urdf(*, disable_wrist_collision: bool = True) -> str:
    bringup = kinematic_bringup()
    xyz = " ".join(str(x) for x in bringup["pos_m"])
    # Identity quat → rpy 0 0 0. CAD rpy must replace this.
    rpy = "0 0 0"
    t800_path = t800_urdf()
    text = t800_path.read_text(encoding="utf-8")
    mesh_root = t800_root() / "meshes"
    text = re.sub(
        r'package://resource/robot/t800/meshes/([^"]+)',
        lambda m: str((mesh_root / m.group(1)).resolve()),
        text,
    )
    if disable_wrist_collision:
        text = re.sub(
            r"(<link name=\"LINK_WRIST_END_[LR]\">.*?</link>)",
            _strip_urdf_collision,
            text,
            flags=re.DOTALL,
        )

    extras = []
    for side, wrist, mount in (
        ("left", "LINK_WRIST_END_L", "l_mount"),
        ("right", "LINK_WRIST_END_R", "r_mount"),
    ):
        hand_path = official_urdf(side, with_mount=True)
        hand = hand_path.read_text(encoding="utf-8")
        inner = re.sub(r"^<\?xml[^>]*>", "", hand).strip()
        inner = re.sub(r"<robot[^>]*>", "", inner, count=1)
        inner = re.sub(r"</robot>\s*$", "", inner)
        inner = re.sub(
            r'filename="(\.\./meshes/[^"]+)"',
            lambda m, hp=hand_path: f'filename="{(hp.parent / m.group(1)).resolve()}"',
            inner,
        )
        extras.append(inner)
        extras.append(
            f"""
  <joint name="weld_{side}_hand_kinematic_bringup" type="fixed">
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <parent link="{wrist}"/>
    <child link="{mount}"/>
  </joint>
"""
        )
    if "</robot>" not in text:
        raise ValueError("T800 URDF missing closing </robot>")
    text = text.replace("</robot>", "\n".join(extras) + "\n</robot>")
    header = (
        "<!-- GENERATED T800 + DexHand2 URDF. Flange is kinematic_bringup_identity. "
        "Not valid for policy eval. -->\n"
    )
    return header + text


def _strip_urdf_collision(match: re.Match[str]) -> str:
    block = match.group(1)
    return re.sub(r"<collision>.*?</collision>", "", block, flags=re.DOTALL)


def write_generated(*, mjcf: bool = True, urdf: bool = True) -> dict[str, Any]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = combined_paths()
    if mjcf:
        xml, manifest = assemble_mjcf()
        mjcf_path = GENERATED / "t800_dexhand2.xml"
        mjcf_path.write_text(xml, encoding="utf-8")
        (GENERATED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        info["mjcf"] = str(mjcf_path)
        info["n_hand_actuators"] = manifest["n_hand_actuators"]
        info["policy_eval_forbidden"] = True
    if urdf:
        urdf_path = GENERATED / "t800_dexhand2.urdf"
        urdf_path.write_text(assemble_urdf(), encoding="utf-8")
        info["urdf"] = str(urdf_path)
    return info


def compile_mjcf(xml: str | None = None):
    """Compile the combined MJCF. Optional dependency on mujoco."""
    import mujoco

    if xml is None:
        xml, _ = assemble_mjcf()
    model = mujoco.MjModel.from_xml_string(xml)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", action="store_true", default=True)
    parser.add_argument("--no-mjcf", action="store_false", dest="mjcf")
    parser.add_argument("--urdf", action="store_true", default=True)
    parser.add_argument("--check-compile", action="store_true")
    args = parser.parse_args()
    if not kinematic_bringup().get("enabled") and not mount_ready():
        raise SystemExit(
            "assets/combined: t800_wrist_to_hand_mount is REQUIRED_INPUT and "
            "kinematic_bringup_identity is disabled."
        )
    info = write_generated(mjcf=args.mjcf, urdf=args.urdf)
    if args.check_compile:
        model = compile_mjcf(Path(info["mjcf"]).read_text(encoding="utf-8"))
        info["nq"] = int(model.nq)
        info["nu"] = int(model.nu)
        info["nbody"] = int(model.nbody)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
