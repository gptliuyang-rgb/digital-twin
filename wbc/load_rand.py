"""SONIC load-aware payload sampling. Refused until wrist CoM is measured."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from assets.combined.assemble import PolicyEvalBlocked, mount_ready
from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec
from sim.payload import Payload

LOAD_RAND_PATH = Path(__file__).with_name("load_rand.yaml")


def load_rand_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or LOAD_RAND_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("load_rand.yaml must be a mapping")
    return raw


def require_load_aware_ready(spec=None) -> None:
    spec = spec or load_hand_spec()
    if spec.raw.get("com_in_wrist_frame_m") == REQUIRED_INPUT_TOKEN:
        raise PolicyEvalBlocked(
            "SONIC load-aware payload sampling is blocked until com_in_wrist_frame_m "
            "is hang-tested (ADR-010). Do not attach 0–20 kg on the skeleton CoM."
        )
    if not mount_ready():
        raise PolicyEvalBlocked(
            "SONIC load-aware payload sampling is blocked until t800_wrist_to_hand_mount "
            "SE(3) is CAD-measured. Payload hangs after the dummy wrist."
        )


def sample_training_payloads(rng: np.random.Generator, spec=None) -> list[Payload]:
    """Two wrist payloads. Raises PolicyEvalBlocked while P0 CoM/flange are open."""
    require_load_aware_ready(spec)
    cfg = load_rand_cfg()
    lo_m, hi_m = (float(x) for x in cfg["mass_kg"])
    lo_c, hi_c = (float(x) for x in cfg["com_offset_m"])
    out: list[Payload] = []
    for side in cfg["attach_sides"]:
        out.append(
            Payload(
                mass_kg=float(rng.uniform(lo_m, hi_m)),
                com_offset_m=rng.uniform(lo_c, hi_c, size=3),
                side=str(side),
            )
        )
    return out


def ranges_match_sim_payload() -> bool:
    """Guard against load_rand.yaml drifting from sim/payload.py clamps."""
    cfg = load_rand_cfg()
    return list(cfg["mass_kg"]) == [0.0, 20.0] and list(cfg["com_offset_m"]) == [-0.15, 0.15]
