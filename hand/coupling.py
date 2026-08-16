"""Underactuation map. Wuji Hand 2 is identity; the interface stays stable."""

from __future__ import annotations

import numpy as np

from interface.schema import HandSpec, load_hand_spec


class Coupling:
    """q_full = [q_active; C @ q_active] for linear coupling. Identity if none."""

    def __init__(self, n_active: int, n_passive: int, matrix: np.ndarray) -> None:
        self.n_active = n_active
        self.n_passive = n_passive
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(n_passive, n_active) if n_passive else np.zeros((0, n_active))

    @classmethod
    def from_spec(cls, spec: HandSpec | None = None) -> Coupling:
        spec = spec or load_hand_spec()
        if spec.coupling_type not in {"none", "identity"}:
            raise NotImplementedError(
                f"coupling_type={spec.coupling_type} is not implemented; Hand 2 is identity-only"
            )
        return cls(spec.n_active_dof, 0, np.zeros((0, spec.n_active_dof)))

    def active_to_full(self, q_active: np.ndarray) -> np.ndarray:
        q = np.asarray(q_active, dtype=np.float64).reshape(self.n_active)
        if self.n_passive == 0:
            return q.copy()
        return np.concatenate([q, self.matrix @ q])

    def full_to_active(self, q_full: np.ndarray) -> np.ndarray:
        q = np.asarray(q_full, dtype=np.float64).reshape(self.n_active + self.n_passive)
        return q[: self.n_active].copy()

    def jacobian_full_wrt_active(self) -> np.ndarray:
        top = np.eye(self.n_active)
        if self.n_passive == 0:
            return top
        return np.vstack([top, self.matrix])


IdentityCoupling = Coupling
