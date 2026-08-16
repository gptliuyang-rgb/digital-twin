"""In-process policy server. Same CommandVector contract as the real ZMQ/HTTP process.

Does not download GR00T / π0.5 weights. `kind=replay` is the only implemented backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from interface.schema import CommandVector, HandSpec, load_hand_spec
from wbc.checkpoint import refuse_g1_checkpoint


class PolicyServer:
    def __init__(
        self,
        kind: str,
        infer_fn: Callable[[dict], np.ndarray] | None = None,
        spec: HandSpec | None = None,
    ) -> None:
        if kind not in {"groot", "pi05", "replay"}:
            raise ValueError(kind)
        self.kind = kind
        self.spec = spec or load_hand_spec()
        self.infer_fn = infer_fn
        if kind in {"groot", "pi05"} and infer_fn is None:
            raise NotImplementedError(
                f"{kind} weight loading is out of scope for P1. "
                "Pass infer_fn, or use kind='replay'."
            )

    def infer(self, observation: dict) -> CommandVector:
        refuse_g1_checkpoint(checkpoint_meta=observation.get("wbc_checkpoint"))
        if self.infer_fn is None:
            raise NotImplementedError(self.kind)
        flat = np.asarray(self.infer_fn(observation), dtype=np.float64)
        if flat.ndim == 2:
            flat = flat[0]
        cmd = CommandVector.from_flat_vector(flat, self.spec)
        cmd.validate()
        return cmd

    def serve(self) -> None:
        raise NotImplementedError("process serve loop is wired at deploy time; use infer() in-process")
