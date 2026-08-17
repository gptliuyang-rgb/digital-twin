"""Read labelled VLA chunk clocks. No invented GPU latency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from interface.schema import REPO_ROOT

CLOCK_PATH = REPO_ROOT / "vla" / "chunk_clock.yaml"


def load_chunk_clock(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or CLOCK_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("chunk_clock.yaml must be a mapping")
    return raw


def upsample_factor(model: str, *, command_hz: int | None = None) -> int:
    """Integer factor from VLA infer rate to SONIC's 50 Hz policy/token rate.

    Non-integer ratios are refused — use the L1a planner, do not round silently.
    Default ``command_hz`` is ``sonic_command_hz`` (50), not the 500 Hz PD ring.
    """
    cfg = load_chunk_clock()
    models = cfg["models"]
    if model not in models:
        raise KeyError(f"unknown VLA clock {model!r}; known={sorted(models)}")
    infer_hz = float(models[model]["infer_hz"])
    cmd_hz = float(command_hz if command_hz is not None else cfg["sonic_command_hz"])
    ratio = cmd_hz / infer_hz
    factor = int(round(ratio))
    if abs(ratio - factor) > 1e-9 or factor < 1:
        raise ValueError(
            f"{model} infer_hz={infer_hz} does not divide sonic_command_hz={cmd_hz}. "
            "Use wbc.planner.KinematicPlanner instead of a rounded upsample."
        )
    return factor
