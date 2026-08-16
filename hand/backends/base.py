from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class HandState:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    tau_est: np.ndarray | None = None
    tactile: np.ndarray | None = None
    timestamp_s: float = 0.0


class HandBackend(ABC):
    """Simulator/hardware injection point. Controller must not import sim packages."""

    @abstractmethod
    def write_mit(
        self,
        q_des_rad: np.ndarray,
        dq_des_rad_s: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_state(self) -> HandState:
        raise NotImplementedError
