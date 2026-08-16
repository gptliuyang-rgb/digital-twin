#!/usr/bin/env python3
"""Record upstream wuji-description checksums; fail CI on silent drift."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interface.schema import REPO_ROOT

UPSTREAM = REPO_ROOT / "third_party" / "wuji-description"
PIN = REPO_ROOT / "assets" / "dexhand2" / "meta" / "upstream_pin.json"
WATCH = [
    "hand2/hand2_beta1/body/mjcf/right.xml",
    "hand2/hand2_beta1/body/mjcf/left.xml",
    "hand2/hand2_beta1/body/urdf/right.urdf",
    "hand2/hand2_beta1/body/mjcf/right_with_mount.xml",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict:
    files = {}
    for rel in WATCH:
        path = UPSTREAM / rel
        files[rel] = sha256(path) if path.is_file() else None
    commit = None
    head = UPSTREAM / ".git" / "HEAD"
    if head.is_file():
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            ref_path = UPSTREAM / ".git" / ref.split(" ", 1)[1]
            commit = ref_path.read_text().strip() if ref_path.is_file() else None
        else:
            commit = ref
    return {"commit": commit, "files": files}


def main() -> None:
    current = snapshot()
    if not PIN.is_file():
        PIN.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"wrote {PIN}")
        return
    pinned = json.loads(PIN.read_text(encoding="utf-8"))
    drift = {k: (pinned["files"].get(k), current["files"].get(k)) for k in WATCH if pinned["files"].get(k) != current["files"].get(k)}
    if drift:
        raise SystemExit(f"upstream drift: {json.dumps(drift, indent=2)}")
    print("no upstream drift")


if __name__ == "__main__":
    main()
