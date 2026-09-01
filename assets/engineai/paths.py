"""Resolve cloned T800 / Wuji trees. Official files are never copied into git."""

from __future__ import annotations

from pathlib import Path

from interface.schema import REPO_ROOT

_T800_CANDIDATES = (
    REPO_ROOT / "third_party" / "engineai-native-sdk" / "assets" / "resource" / "robot" / "t800",
    REPO_ROOT / "third_party" / "engineai_robotics_native_sdk" / "assets" / "resource" / "robot" / "t800",
)

_WUJI_CANDIDATES = (
    REPO_ROOT / "third_party" / "wuji-description",
    REPO_ROOT / "third_party" / "wuji_description",
)


def _first_existing(paths: tuple[Path, ...], marker: str) -> Path:
    for root in paths:
        if (root / marker).is_file() or (root / marker).is_dir():
            return root
    raise FileNotFoundError(
        f"Missing upstream tree containing {marker}. Run scripts/bootstrap_resources.sh. Tried: {list(paths)}"
    )


def t800_root() -> Path:
    return _first_existing(_T800_CANDIDATES, "xml/serial_t800.xml")


def t800_mjcf() -> Path:
    return t800_root() / "xml" / "serial_t800.xml"


def t800_urdf() -> Path:
    return t800_root() / "urdf" / "serial_t800.urdf"


def wuji_root() -> Path:
    return _first_existing(_WUJI_CANDIDATES, "hand2/hand2_beta1/body/mjcf/right_with_mount.xml")
