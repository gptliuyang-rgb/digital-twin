"""Pick the action in a chunk that corresponds to now + inference delay."""

from __future__ import annotations

import numpy as np


def delayed_index(dt_chunk_s: float, latency_s: float, horizon: int) -> int:
    if dt_chunk_s <= 0:
        raise ValueError("dt_chunk_s must be > 0")
    idx = int(round(latency_s / dt_chunk_s))
    return int(np.clip(idx, 0, horizon - 1))


def pick_delayed_action(chunk: np.ndarray, dt_chunk_s: float, latency_s: float) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float64)
    idx = delayed_index(dt_chunk_s, latency_s, arr.shape[0])
    return arr[idx].copy()
