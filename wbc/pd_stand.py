"""Flatten EngineAI bring-up stand PD. Not SONIC tracking gains (ADR-022)."""

from __future__ import annotations

import numpy as np

from wbc.dims import load_t800_sonic


def pd_stand_kp_kd() -> tuple[np.ndarray, np.ndarray]:
    """Return (kp, kd) in SONIC joint_order. Source: t800_sonic.yaml pd_stand."""
    cfg = load_t800_sonic()
    block = cfg["pd_stand"]
    n = int(cfg["n_revolute"])
    kp = np.concatenate(
        [np.asarray(row, dtype=np.float64).reshape(-1) for row in block["stiffness"]]
    )
    kd = np.concatenate(
        [np.asarray(row, dtype=np.float64).reshape(-1) for row in block["damping"]]
    )
    if kp.shape != (n,) or kd.shape != (n,):
        raise ValueError(f"pd_stand flattened to {kp.shape}/{kd.shape}, expected ({n},)")
    return kp, kd
