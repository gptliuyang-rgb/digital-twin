"""FK adapter: arm joint angles → wrist SE(3) in a given frame.

Requires a kinematic backend (Pinocchio or MuJoCo). Without one, this module still
exposes the mapping contract so eval L0 can record the mismatch instead of guessing FK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla.adapters.rotation import matrix_to_rot6d


@dataclass
class WristPose:
    pos_m: np.ndarray
    rot6d: np.ndarray


class JointToWristAdapter:
    def __init__(self, fk_fn) -> None:
        self.fk_fn = fk_fn

    def __call__(self, q_arm: np.ndarray) -> WristPose:
        pos, rot = self.fk_fn(np.asarray(q_arm, dtype=np.float64))
        return WristPose(pos_m=np.asarray(pos, dtype=np.float64).reshape(3), rot6d=matrix_to_rot6d(rot))
