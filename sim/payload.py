"""Runtime payload attachment for SONIC load-aware training.

Does not invent a default mass. Callers must pass mass_kg in [0, 20].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Payload:
    mass_kg: float
    com_offset_m: np.ndarray
    side: str  # left | right

    def __post_init__(self) -> None:
        if not 0.0 <= self.mass_kg <= 20.0:
            raise ValueError("payload mass_kg must be in [0, 20]")
        self.com_offset_m = np.asarray(self.com_offset_m, dtype=np.float64).reshape(3)
        if np.any(np.abs(self.com_offset_m) > 0.15 + 1e-9):
            raise ValueError("com_offset_m must lie in [-0.15, 0.15]^3 m")
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right")


def sample_payload(rng: np.random.Generator, side: str) -> Payload:
    mass = float(rng.uniform(0.0, 20.0))
    com = rng.uniform(-0.15, 0.15, size=3)
    return Payload(mass_kg=mass, com_offset_m=com, side=side)


def box_inertia_kgm2(mass_kg: float, size_m: np.ndarray) -> np.ndarray:
    sx, sy, sz = np.asarray(size_m, dtype=np.float64)
    ixx = mass_kg * (sy * sy + sz * sz) / 12.0
    iyy = mass_kg * (sx * sx + sz * sz) / 12.0
    izz = mass_kg * (sx * sx + sy * sy) / 12.0
    return np.diag([ixx, iyy, izz])
