"""Refuse to weld Hand 2 onto T800 until mount SE(3) is a measured CAD number."""

from __future__ import annotations

import yaml

from interface.schema import REPO_ROOT, find_required_inputs

MOUNT = REPO_ROOT / "assets" / "dexhand2" / "meta" / "mount_transform.yaml"
T800_URDF = REPO_ROOT / "third_party" / "engineai-native-sdk" / "assets" / "resource" / "robot" / "t800" / "urdf" / "serial_t800.urdf"


def mount_ready() -> bool:
    raw = yaml.safe_load(MOUNT.read_text(encoding="utf-8"))
    missing = find_required_inputs(raw.get("t800_wrist_to_hand_mount", {}))
    return not missing


def combined_paths() -> dict:
    return {
        "t800_urdf": str(T800_URDF),
        "hand_right_mjcf": str(
            REPO_ROOT / "third_party" / "wuji-description" / "hand2" / "hand2_beta1" / "body" / "mjcf" / "right_with_mount.xml"
        ),
        "wrist_links": ["LINK_WRIST_END_L", "LINK_WRIST_END_R"],
        "mount_ready": mount_ready(),
    }


def main() -> None:
    info = combined_paths()
    if not info["mount_ready"]:
        raise SystemExit(
            "assets/combined: t800_wrist_to_hand_mount is REQUIRED_INPUT. "
            "Hand-side offset is known; T800 flange SE(3) is not. See docs/HW_INTEGRATION.md"
        )
    print(info)


if __name__ == "__main__":
    main()
