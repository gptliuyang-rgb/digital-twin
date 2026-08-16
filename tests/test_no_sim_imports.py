from __future__ import annotations

import ast
from pathlib import Path

from interface.schema import REPO_ROOT

FORBIDDEN = ("mujoco", "isaacsim", "omni", "isaaclab")
SCAN = [
    REPO_ROOT / "hand" / "controller.py",
    REPO_ROOT / "runtime",
    REPO_ROOT / "vla" / "client",
]


def _py_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return list(root.rglob("*.py"))


def test_shared_code_has_no_sim_imports() -> None:
    hits = []
    for root in SCAN:
        for path in _py_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name.split(".")[0]
                        if name in FORBIDDEN:
                            hits.append(f"{path}:{alias.name}")
                if isinstance(node, ast.ImportFrom) and node.module:
                    name = node.module.split(".")[0]
                    if name in FORBIDDEN:
                        hits.append(f"{path}:{node.module}")
    assert hits == []
