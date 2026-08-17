"""Weld recipe for T800 + Hand 2. Policy eval is refused until flange SE(3) is CAD-measured.

A bring-up identity transform is *not* emitted. Identity would bake a constant
VLA wrist bias into every checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from interface.schema import REPO_ROOT, find_required_inputs, load_frames

MOUNT = REPO_ROOT / "assets" / "dexhand2" / "meta" / "mount_transform.yaml"
T800_URDF = (
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
T800PRO_URDF = (
    REPO_ROOT
    / "third_party"
    / "engineai-native-sdk"
    / "assets"
    / "resource"
    / "robot"
    / "t800pro"
    / "urdf"
    / "serial_t800pro.urdf"
)
HAND_RIGHT = (
    REPO_ROOT
    / "third_party"
    / "wuji-description"
    / "hand2"
    / "hand2_beta1"
    / "body"
    / "mjcf"
    / "right_with_mount.xml"
)
RECIPE_OUT = REPO_ROOT / "assets" / "combined" / "weld_recipe.yaml"


class PolicyEvalBlocked(RuntimeError):
    """Raised when a policy eval asks for a combined robot without a measured mount."""


def _mount_raw() -> dict:
    return yaml.safe_load(MOUNT.read_text(encoding="utf-8"))


def mount_ready() -> bool:
    raw = _mount_raw()
    missing = find_required_inputs(raw.get("t800_wrist_to_hand_mount", {}))
    return not missing


def dummy_wrist_from_elbow() -> dict:
    """Official T800 URDF fixed joint elbow-yaw → dummy LINK_WRIST_END_*. Not the hand flange."""
    return {
        "source": "third_party/engineai-native-sdk/assets/resource/robot/t800/urdf/serial_t800.urdf",
        "left": {
            "joint": "J_FIXED_WAIST_L",
            "parent": "LINK_ELBOW_YAW_L",
            "child": "LINK_WRIST_END_L",
            "pos_m": [0.029436, 0.0124855, -0.13222151],
            "rpy_rad": [0.0, 0.0, 0.0],
        },
        "right": {
            "joint": "J_FIXED_WAIST_R",
            "parent": "LINK_ELBOW_YAW_R",
            "child": "LINK_WRIST_END_R",
            "pos_m": [0.029436, -0.0124855, -0.13222151],
            "rpy_rad": [0.0, 0.0, 0.0],
        },
        "note": "Dummy 1 g sphere. Do not treat this as Wuji mount SE(3).",
    }


def weld_recipe() -> dict:
    raw = _mount_raw()
    frames = load_frames()
    missing = find_required_inputs(raw.get("t800_wrist_to_hand_mount", {}))
    return {
        "schema_version": "1.0",
        "eval_allowed": mount_ready(),
        "bringup_identity_forbidden_for_eval": True,
        "missing": missing,
        "humanoid": raw.get("humanoid"),
        "hand_mount_to_wrist": raw.get("hand_mount_to_wrist"),
        "t800_wrist_to_hand_mount": raw.get("t800_wrist_to_hand_mount"),
        "t800_dummy_wrist_from_elbow": dummy_wrist_from_elbow(),
        "sonic_wrist_links": {
            "t800": {
                "left": frames["frames"]["left_wrist"]["t800_link"],
                "right": frames["frames"]["right_wrist"]["t800_link"],
            },
            "t800pro": {
                "left": frames["frames"]["left_wrist"]["t800pro_link"],
                "right": frames["frames"]["right_wrist"]["t800pro_link"],
            },
        },
        "paths": {
            "t800_urdf": str(T800_URDF.relative_to(REPO_ROOT)),
            "t800pro_urdf": str(T800PRO_URDF.relative_to(REPO_ROOT)),
            "hand_right_mjcf": str(HAND_RIGHT.relative_to(REPO_ROOT)),
        },
    }


def write_weld_recipe(path: Path | None = None) -> Path:
    out = path or RECIPE_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(weld_recipe(), sort_keys=False), encoding="utf-8")
    return out


def assert_policy_eval_allowed(purpose: str = "policy_eval") -> None:
    if mount_ready():
        return
    raise PolicyEvalBlocked(
        f"{purpose} requires assets/dexhand2/meta/mount_transform.yaml "
        f"t800_wrist_to_hand_mount to be a CAD-measured SE(3). "
        f"Identity bring-up is forbidden (constant VLA wrist bias). "
        f"Still REQUIRED_INPUT: {find_required_inputs(_mount_raw().get('t800_wrist_to_hand_mount', {}))}"
    )


def combined_paths() -> dict:
    recipe = weld_recipe()
    return {
        "t800_urdf": recipe["paths"]["t800_urdf"],
        "hand_right_mjcf": recipe["paths"]["hand_right_mjcf"],
        "wrist_links": [
            recipe["sonic_wrist_links"]["t800"]["left"],
            recipe["sonic_wrist_links"]["t800"]["right"],
        ],
        "mount_ready": recipe["eval_allowed"],
        "eval_allowed": recipe["eval_allowed"],
    }


def main() -> None:
    path = write_weld_recipe()
    info = combined_paths()
    print(f"wrote {path}")
    if not info["mount_ready"]:
        raise SystemExit(
            "assets/combined: t800_wrist_to_hand_mount is REQUIRED_INPUT. "
            "Hand-side offset is known; T800 flange SE(3) is not. "
            "weld_recipe.yaml written for humans. See docs/HW_INTEGRATION.md"
        )
    print(info)


if __name__ == "__main__":
    main()
