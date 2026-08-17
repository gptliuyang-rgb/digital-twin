"""Upsample a command_schema_v1 chunk from VLA infer_hz to SONIC's 50 Hz policy/token rate.

Integer factors only (see chunk_clock.yaml). Zhou 6D is SLERP'd on SO(3), not
averaged. Case A chunks are refused — convert with apply_fk=True first.
The 500 Hz PD ring is ``wbc.stream`` (ADR-039), not this module.
"""

from __future__ import annotations

import numpy as np

from interface.schema import HandSpec, command_dim, load_hand_spec
from vla.adapters.action_space import ActionSpaceMismatch, diagnose_action_vector
from vla.adapters.chunk_clock import load_chunk_clock, upsample_factor
from wbc.planner import interpolate_command_matrix


def upsample_command_chunk(
    chunk: np.ndarray,
    *,
    model: str,
    spec: HandSpec | None = None,
    pos_interp: str = "cubic_hermite",
) -> np.ndarray:
    """Return (N_out, 75) at sonic_command_hz.

    N_out = (H - 1) * factor + 1, spanning the same duration as the H waypoints.
    """
    spec = spec or load_hand_spec()
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ActionSpaceMismatch(f"chunk must be (H, D), got {arr.shape}")
    expected = command_dim(spec)
    if arr.shape[1] != expected:
        report = diagnose_action_vector(dim=int(arr.shape[1]), spec=spec)
        raise ActionSpaceMismatch(
            "upsample requires command_schema_v1 rows. "
            f"got dim {arr.shape[1]} (case {report.case}, layout {report.layout_name}). "
            "If this is Case A, call CaseAToCommandSchema.convert(..., apply_fk=True) first."
        )
    if arr.shape[0] < 2:
        raise ValueError("upsample needs ≥2 chunk steps")
    factor = upsample_factor(model)
    clock = load_chunk_clock()
    infer_hz = float(clock["models"][model]["infer_hz"])
    h = arr.shape[0]
    t_src = np.arange(h, dtype=np.float64) / infer_hz
    n_dst = (h - 1) * factor + 1
    t_dst = np.linspace(float(t_src[0]), float(t_src[-1]), n_dst)
    return interpolate_command_matrix(arr, t_src, t_dst, spec, pos_interp=pos_interp)
