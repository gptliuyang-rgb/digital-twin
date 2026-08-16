"""Camera / IMU models. Calibration numbers stay REQUIRED_INPUT until filled."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml

from interface.schema import REPO_ROOT, find_required_inputs

CALIB_PATH = REPO_ROOT / "sim" / "sensors" / "calib_real.yaml"


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class FrameDelay:
    def __init__(self, delay_ms: float, rate_hz: float) -> None:
        depth = max(1, int(round((delay_ms / 1000.0) * rate_hz)))
        self._buf: deque[np.ndarray] = deque(maxlen=depth)

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(frame))
        return self._buf[0]


def jpeg_roundtrip(rgb: np.ndarray, quality: int = 80) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    return np.asarray(out.convert("RGB"))


def load_calib(path: Path | None = None) -> dict:
    with (path or CALIB_PATH).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    missing = find_required_inputs(raw)
    if missing:
        # Allowed: sensors are P2. Callers must not pretend they are calibrated.
        raw["_missing"] = missing
    return raw
