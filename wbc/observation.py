"""SONIC proprioception in the robot-heading frame.

s^p_t = (q, dq, ω, gravity_heading, a_{t-1})  — He et al., SONIC / GR00T-WBC.
Hands are not in this vector (bypass WBC).
"""

from __future__ import annotations

import numpy as np

from vla.adapters.frame_transform import heading_rotation_z
from wbc.dims import load_t800_sonic


def heading_gravity(pelvis_yaw_rad: float, gravity_world: np.ndarray | None = None) -> np.ndarray:
    """Gravity direction in the heading frame (SONIC v1.1 robot-heading-normalized)."""
    g = np.array([0.0, 0.0, -1.0], dtype=np.float64) if gravity_world is None else np.asarray(
        gravity_world, dtype=np.float64
    ).reshape(3)
    r_h = heading_rotation_z(pelvis_yaw_rad)
    return r_h.T @ g


def heading_angular_velocity(omega_world: np.ndarray, pelvis_yaw_rad: float) -> np.ndarray:
    r_h = heading_rotation_z(pelvis_yaw_rad)
    return r_h.T @ np.asarray(omega_world, dtype=np.float64).reshape(3)


def policy_proprio(
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    omega_world: np.ndarray,
    last_action: np.ndarray,
    pelvis_yaw_rad: float,
    *,
    n_dof: int | None = None,
) -> np.ndarray:
    """Concatenate one-step proprio. History stacking is the caller's job (10× step1)."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    q = np.asarray(q_rad, dtype=np.float64).reshape(n)
    dq = np.asarray(dq_rad_s, dtype=np.float64).reshape(n)
    a = np.asarray(last_action, dtype=np.float64).reshape(n)
    omega = heading_angular_velocity(omega_world, pelvis_yaw_rad)
    grav = heading_gravity(pelvis_yaw_rad)
    return np.concatenate([q, dq, omega, grav, a])
