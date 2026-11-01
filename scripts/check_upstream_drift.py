#!/usr/bin/env python3
"""Record upstream wuji-description checksums; fail CI on silent drift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interface.schema import REPO_ROOT

UPSTREAM = REPO_ROOT / "third_party" / "wuji-description"
PIN = REPO_ROOT / "assets" / "dexhand2" / "meta" / "upstream_pin.json"
WATCH = [
    "hand2/hand2_2/body/mjcf/right.xml",
    "hand2/hand2_2/body/mjcf/left.xml",
    "hand2/hand2_2/body/mjcf/right_with_cover.xml",
    "hand2/hand2_2/body/urdf/right.urdf",
    "hand2/hand2_2/body/urdf/right_with_cover.urdf",
    "hand2/hand2_1/body/mjcf/right.xml",
    "hand2/hand2_1/body/urdf/right.urdf",
]
REMOVED = [
    "hand2/hand2_beta1/body/mjcf/right.xml",
    "hand2/hand2_beta2/body/mjcf/right.xml",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    head = UPSTREAM / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        ref_path = UPSTREAM / ".git" / ref.split(" ", 1)[1]
        return ref_path.read_text().strip() if ref_path.is_file() else None
    return ref


def snapshot() -> dict:
    files = {}
    for rel in WATCH:
        path = UPSTREAM / rel
        files[rel] = sha256(path) if path.is_file() else None
    return {"commit": git_commit(), "files": files}


def main() -> None:
    if not UPSTREAM.is_dir():
        raise SystemExit("third_party/wuji-description missing; run scripts/bootstrap_resources.sh")
    leftovers = [rel for rel in REMOVED if (UPSTREAM / rel).is_file()]
    if leftovers:
        raise SystemExit(f"Beta 1/2 paths returned upstream: {leftovers}")
    current = snapshot()
    if not PIN.is_file():
        PIN.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PIN}")
        return
    pinned = json.loads(PIN.read_text(encoding="utf-8"))
    drift = {
        key: (pinned.get("files", {}).get(key), current["files"].get(key))
        for key in WATCH
        if pinned.get("files", {}).get(key) != current["files"].get(key)
    }
    if pinned.get("commit") and current["commit"] and pinned["commit"] != current["commit"]:
        drift["commit"] = (pinned["commit"], current["commit"])
    if drift:
        raise SystemExit(f"upstream drift: {json.dumps(drift, indent=2)}")
    print("no upstream drift")


if __name__ == "__main__":
    main()
