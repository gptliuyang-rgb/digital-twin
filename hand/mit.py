"""MIT force-position hybrid. Shared by controller and sim backends."""

from __future__ import annotations

import numpy as np


def mit_torque(
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    q_des_rad: np.ndarray,
    dq_des_rad_s: np.ndarray,
    tau_ff: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """τ = kp (qd − q) + kd (dqd − dq) + τ_ff."""
    return kp * (q_des_rad - q_rad) + kd * (dq_des_rad_s - dq_rad_s) + tau_ff
