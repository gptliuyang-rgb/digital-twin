"""Thin wrapper around official wuji-retargeting when present; IK fallback otherwise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interface.schema import HandSpec, load_hand_spec


@dataclass
class RetargetResult:
    q_active: np.ndarray
    used_official: bool


def retarget_keypoints(
    keypoints_m: np.ndarray,
    spec: HandSpec | None = None,
    q_ref: np.ndarray | None = None,
) -> RetargetResult:
    """keypoints_m: (21, 3) MANO/VR landmarks in metres.

    Prefers official wuji-retargeting. Without it, returns q_ref (or zeros) rather
    than inventing an IK solution against an unmeasured scale factor.
    """
    spec = spec or load_hand_spec()
    n = spec.n_active_dof
    ref = np.zeros(n) if q_ref is None else np.asarray(q_ref, dtype=np.float64).reshape(n)
    try:
        import wuji_retargeting  # type: ignore
    except ImportError:
        return RetargetResult(q_active=ref, used_official=False)
    solver = getattr(wuji_retargeting, "retarget", None)
    if solver is None:
        return RetargetResult(q_active=ref, used_official=False)
    q = np.asarray(solver(np.asarray(keypoints_m, dtype=np.float64)), dtype=np.float64).reshape(n)
    return RetargetResult(q_active=q, used_official=True)
