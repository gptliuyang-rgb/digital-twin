"""Flatten EngineAI bring-up stand PD. Not SONIC tracking gains (ADR-022)."""

from __future__ import annotations

import numpy as np

from wbc.dims import load_t800_sonic


def _flatten_rows(rows: list[list[float]], n: int, name: str) -> np.ndarray:
    out = np.concatenate([np.asarray(row, dtype=np.float64).reshape(-1) for row in rows])
    if out.shape != (n,):
        raise ValueError(f"pd_stand {name} flattened to {out.shape}, expected ({n},)")
    return out


def pd_stand_kp_kd() -> tuple[np.ndarray, np.ndarray]:
    """Return (kp, kd) in SONIC joint_order. Source: t800_sonic.yaml pd_stand."""
    cfg = load_t800_sonic()
    block = cfg["pd_stand"]
    n = int(cfg["n_revolute"])
    kp = _flatten_rows(block["stiffness"], n, "stiffness")
    kd = _flatten_rows(block["damping"], n, "damping")
    return kp, kd


def pd_stand_q_des_rad() -> np.ndarray:
    """Official EngineAI ``desired_joint_position`` (rad). Not a SONIC reference."""
    cfg = load_t800_sonic()
    return _flatten_rows(cfg["pd_stand"]["desired_joint_position"], int(cfg["n_revolute"]), "q_des")
