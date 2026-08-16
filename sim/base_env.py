"""Abstract env. Concrete simulators live in mujoco_env / isaaclab_env."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError
