"""Policy client. Simulator-free. A bridge injects observations and consumes commands."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from interface.schema import CommandVector, HandSpec, load_hand_spec
from vla.adapters.action_space import require_command_schema_vector


class PolicyClient:
    def __init__(
        self,
        infer_fn: Callable[[dict], np.ndarray],
        spec: HandSpec | None = None,
    ) -> None:
        self.infer_fn = infer_fn
        self.spec = spec or load_hand_spec()

    def step(self, observation: dict) -> CommandVector:
        flat = np.asarray(self.infer_fn(observation), dtype=np.float64)
        if flat.ndim == 2:
            flat = flat[0]
        flat = require_command_schema_vector(flat, self.spec)
        return CommandVector.from_flat_vector(flat, self.spec)
