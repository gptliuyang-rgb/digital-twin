"""Action-chunk temporal ensembling. Rotations averaged on SO(3), not in 6D Euclidean space."""

from __future__ import annotations

from collections import deque

import numpy as np

from vla.adapters.rotation import matrix_to_rot6d, rot6d_to_matrix, slerp_matrices


class TemporalEnsemble:
    def __init__(self, chunk_horizon: int, alpha: float = 0.3, rot6d_slices: list[tuple[int, int]] | None = None) -> None:
        self.horizon = chunk_horizon
        self.alpha = float(alpha)
        self.rot6d_slices = rot6d_slices or []
        self._chunks: deque[np.ndarray] = deque()

    def push(self, chunk: np.ndarray) -> None:
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != self.horizon:
            raise ValueError(f"chunk must be ({self.horizon}, dim), got {arr.shape}")
        self._chunks.appendleft(arr)

    def value_at(self, k: int = 0) -> np.ndarray:
        if not self._chunks:
            raise RuntimeError("no chunks")
        weights = []
        samples = []
        for i, chunk in enumerate(self._chunks):
            if k >= chunk.shape[0]:
                continue
            weights.append(np.exp(-self.alpha * i))
            samples.append(chunk[k])
        w = np.array(weights)
        w = w / w.sum()
        acc = np.zeros_like(samples[0])
        for weight, sample in zip(w, samples, strict=True):
            acc = acc + weight * sample
        # Overwrite rotation blocks with SO(3) weighted mean via successive slerp.
        for a, b in self.rot6d_slices:
            mats = [rot6d_to_matrix(s[a:b]) for s in samples]
            r = mats[0]
            # Normalized incremental slerp: not exact Riemannian mean, but flip-free.
            cum = w[0]
            for weight, mat in zip(w[1:], mats[1:], strict=True):
                t = weight / (cum + weight)
                r = slerp_matrices(r, mat, t)
                cum += weight
            acc[a:b] = matrix_to_rot6d(r)
        return acc
