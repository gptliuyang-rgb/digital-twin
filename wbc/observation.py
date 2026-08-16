"""SONIC proprioception in the robot-heading frame.

s^p_t = (q, dq, ω, gravity_heading, a_{t-1})  — He et al., SONIC / GR00T-WBC.
Hands are not in this vector (bypass WBC).
"""

from __future__ import annotations

import numpy as np

from vla.adapters.frame_transform import heading_rotation_z
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import (
    ANGVEL_DIM,
    G1_N_DOF,
    GRAVITY_DIM,
    HISTORY_FRAMES,
    TOKEN_DIM,
    decoder_history_dim,
    load_t800_sonic,
)


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
    """Concatenate one-step proprio as (q, dq, ω, g, a). History is ProprioHistory."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    q = np.asarray(q_rad, dtype=np.float64).reshape(n)
    dq = np.asarray(dq_rad_s, dtype=np.float64).reshape(n)
    a = np.asarray(last_action, dtype=np.float64).reshape(n)
    omega = heading_angular_velocity(omega_world, pelvis_yaw_rad)
    grav = heading_gravity(pelvis_yaw_rad)
    return np.concatenate([q, dq, omega, grav, a])


def decoder_step_dim(n_dof: int) -> int:
    """Per-frame decoder slice: ω(3) + q(n) + dq(n) + a(n) + g(3) = 3n+6."""
    return ANGVEL_DIM + GRAVITY_DIM + 3 * int(n_dof)


def pack_decoder_step(
    q_rad: np.ndarray,
    dq_rad_s: np.ndarray,
    omega_heading: np.ndarray,
    last_action: np.ndarray,
    gravity_heading: np.ndarray,
) -> np.ndarray:
    """SONIC v1.1 decoder frame order: (ω, q, dq, a, g). Not (q, dq, ω, g, a)."""
    q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
    dq = np.asarray(dq_rad_s, dtype=np.float64).reshape(-1)
    a = np.asarray(last_action, dtype=np.float64).reshape(-1)
    omega = np.asarray(omega_heading, dtype=np.float64).reshape(3)
    grav = np.asarray(gravity_heading, dtype=np.float64).reshape(3)
    if not (q.shape == dq.shape == a.shape):
        raise ValueError("q, dq, last_action must share n_dof")
    return np.concatenate([omega, q, dq, a, grav])


class ProprioHistory:
    """10-frame heading-normalized decoder input. Token is supplied at pack time.

    T800: 64 + 10×81 = 874. G1 29-DoF construction is refused (ADR-011).
    """

    def __init__(
        self,
        n_dof: int | None = None,
        *,
        history_frames: int = HISTORY_FRAMES,
        token_dim: int = TOKEN_DIM,
    ) -> None:
        n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
        if n == G1_N_DOF:
            refuse_g1_checkpoint(n_dof=n)
        self.n_dof = n
        self.history_frames = int(history_frames)
        self.token_dim = int(token_dim)
        self.step_dim = decoder_step_dim(n)
        self._buf = np.zeros((self.history_frames, self.step_dim), dtype=np.float64)

    def reset(self) -> None:
        self._buf[:] = 0.0

    def push(
        self,
        q_rad: np.ndarray,
        dq_rad_s: np.ndarray,
        omega_world: np.ndarray,
        last_action: np.ndarray,
        pelvis_yaw_rad: float,
    ) -> None:
        omega = heading_angular_velocity(omega_world, pelvis_yaw_rad)
        grav = heading_gravity(pelvis_yaw_rad)
        step = pack_decoder_step(q_rad, dq_rad_s, omega, last_action, grav)
        if step.shape != (self.step_dim,):
            raise ValueError(f"decoder step dim {step.shape} != {self.step_dim}")
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = step

    def decoder_input(self, token: np.ndarray) -> np.ndarray:
        tok = np.asarray(token, dtype=np.float64).reshape(-1)
        if tok.shape != (self.token_dim,):
            raise ValueError(f"token dim {tok.shape} != {self.token_dim}")
        packed = np.concatenate([tok, self._buf.reshape(-1)])
        expected = self.token_dim + self.history_frames * self.step_dim
        if packed.shape != (expected,):
            raise AssertionError(f"decoder input {packed.shape} != {expected}")
        if self.history_frames == HISTORY_FRAMES and self.token_dim == TOKEN_DIM:
            contract = decoder_history_dim(self.n_dof, token_dim=self.token_dim)
            if packed.shape != (contract,):
                raise AssertionError("decoder packing drifted from wbc.dims.decoder_history_dim")
        return packed
