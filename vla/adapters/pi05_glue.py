"""π0.5 (openpi) glue onto command_schema_v1.

GR00T N1.5/N1.6 share SONIC's decoupled WBC in GR00T-WholeBodyControl. π0.5 does
not. This module is the *only* π0.5 path: if a chunk's last dim is already
command_schema_v1, it passes through. Any other dim is diagnosed and refused.

This is not a trained π0.5 head and does not invent an openpi action layout.
"""

from __future__ import annotations

import numpy as np

from interface.schema import HandSpec, command_dim, load_hand_spec
from vla.adapters.action_space import (
    ActionSpaceMismatch,
    diagnose_action_vector,
    require_command_schema_vector,
)


def pi05_chunk_to_command_schema(
    chunk: np.ndarray,
    spec: HandSpec | None = None,
) -> np.ndarray:
    """Map a π0.5 action chunk (H, D) onto command_schema_v1 rows.

    Pass-through when D matches. Otherwise raise ActionSpaceMismatch with the
    A/B/C diagnosis so the caller writes an adapter or retrains — never a silent
    slice.
    """
    spec = spec or load_hand_spec()
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ActionSpaceMismatch(f"π0.5 chunk must be (H, D) or (D,), got {arr.shape}")
    expected = command_dim(spec)
    if arr.shape[1] != expected:
        report = diagnose_action_vector(dim=int(arr.shape[1]), spec=spec)
        raise ActionSpaceMismatch(
            "π0.5 is not native to SONIC. Glue only accepts command_schema_v1 "
            f"(dim {expected}); this chunk is dim {arr.shape[1]} "
            f"(case {report.case}, layout {report.layout_name}). "
            f"adapter={report.adapter}."
        )
    # Validate first row so a NaN/wrong-length slip still fails loud.
    require_command_schema_vector(arr[0], spec=spec)
    return arr
