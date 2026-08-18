"""Deploy pipeline: infer → temporal ensemble → latency pick → CommandVector.

Simulator-free. Sim and real swap only the observation bridge sitting *outside*.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from interface.schema import CommandVector, HandSpec, command_layout, load_hand_spec
from runtime.latency_comp import delayed_index
from runtime.temporal_ensemble import TemporalEnsemble
from vla.adapters.action_space import require_command_schema_vector
from vla.client.policy_client import PolicyClient


def _rot6d_slices(spec: HandSpec) -> list[tuple[int, int]]:
    layout = command_layout(spec)
    return [layout[name] for name in ("head_rot6d", "left_wrist_rot6d", "right_wrist_rot6d")]


class DeployPipeline:
    def __init__(
        self,
        infer_fn: Callable[[dict], np.ndarray],
        spec: HandSpec | None = None,
        *,
        chunk_horizon: int = 16,
        dt_chunk_s: float = 0.1,
        latency_s: float = 0.0,
        ensemble_alpha: float = 0.3,
    ) -> None:
        self.spec = spec or load_hand_spec()
        self.client = PolicyClient(infer_fn, self.spec)
        self.horizon = chunk_horizon
        self.dt_chunk_s = dt_chunk_s
        self.latency_s = latency_s
        self.ensemble = TemporalEnsemble(
            chunk_horizon, alpha=ensemble_alpha, rot6d_slices=_rot6d_slices(self.spec)
        )

    def step_chunk(self, observation: dict, chunk: np.ndarray | None = None) -> CommandVector:
        if chunk is None:
            raw = np.asarray(self.client.infer_fn(observation), dtype=np.float64)
            chunk = raw if raw.ndim == 2 else raw.reshape(self.horizon, -1)
        require_command_schema_vector(np.asarray(chunk).reshape(-1, chunk.shape[-1])[0], self.spec)
        self.ensemble.push(chunk)
        k = delayed_index(self.dt_chunk_s, self.latency_s, self.horizon)
        return CommandVector.from_flat_vector(self.ensemble.value_at(k), self.spec)
