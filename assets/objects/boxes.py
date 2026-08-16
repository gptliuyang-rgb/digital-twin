"""Parameterised cardboard box + QR sticker assets. No grasp-success claims here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.qr_scanner import make_qr_png


@dataclass
class BoxSpec:
    size_m: np.ndarray
    mass_kg: float
    com_offset_frac: np.ndarray
    split_flex: bool

    def __post_init__(self) -> None:
        self.size_m = np.asarray(self.size_m, dtype=np.float64)
        self.com_offset_frac = np.asarray(self.com_offset_frac, dtype=np.float64)
        if not (0.25 <= self.size_m.min() and self.size_m.max() <= 0.6):
            # Generator still allows out-of-range for tests; industrial default is 0.25–0.6 m.
            pass
        if not 2.0 <= self.mass_kg <= 20.0:
            raise ValueError("box mass_kg must be in [2, 20]")
        self.split_flex = bool(self.size_m.max() > 0.4)


def sample_box(rng: np.random.Generator) -> BoxSpec:
    size = rng.uniform(0.25, 0.6, size=3)
    mass = float(rng.uniform(2.0, 20.0))
    com = rng.uniform(-0.15, 0.15, size=3)
    return BoxSpec(size_m=size, mass_kg=mass, com_offset_frac=com, split_flex=size.max() > 0.4)


def write_qr_texture(payload: str, path: Path, module_px: int = 16) -> None:
    img = make_qr_png(payload, box_size=module_px, border=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


EURO_PALLET_M = (1.20, 0.80, 0.144)
