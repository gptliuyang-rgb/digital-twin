"""Eval report gates. grasp_success_rate is forbidden until live E1/E2 (ADR-004)."""

from __future__ import annotations

from typing import Any

FORBIDDEN_KEYS = frozenset(
    {
        "grasp_success_rate",
        "pick_success_rate",
        "success_rate",
    }
)


def find_forbidden_keys(obj: Any, *, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}"
            if key in FORBIDDEN_KEYS:
                hits.append(here)
            hits.extend(find_forbidden_keys(value, path=here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(find_forbidden_keys(value, path=f"{path}[{i}]"))
    return hits


def assert_no_grasp_success_rate(report: Any) -> None:
    hits = find_forbidden_keys(report)
    if hits:
        raise AssertionError(
            "ADR-004: grasp/pick success-rate keys are forbidden in eval reports: " + ", ".join(hits)
        )
