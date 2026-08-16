from __future__ import annotations

import numpy as np

from hand.backends.base import HandBackend, HandState


class MockBackend(HandBackend):
    """In-memory plant: first-order hold of the last command. Used by unit tests."""

    def __init__(self, n_dof: int) -> None:
        self.n_dof = n_dof
        self.q = np.zeros(n_dof, dtype=np.float64)
        self.dq = np.zeros(n_dof, dtype=np.float64)
        self.tau = np.zeros(n_dof, dtype=np.float64)
        self.t_s = 0.0

    def write_mit(
        self,
        q_des_rad: np.ndarray,
        dq_des_rad_s: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        self.q = np.asarray(q_des_rad, dtype=np.float64).copy()
        self.dq = np.asarray(dq_des_rad_s, dtype=np.float64).copy()
        self.tau = np.asarray(tau_ff, dtype=np.float64).copy()
        self.t_s += 0.001

    def read_state(self) -> HandState:
        return HandState(q_rad=self.q.copy(), dq_rad_s=self.dq.copy(), tau_est=self.tau.copy(), timestamp_s=self.t_s)
