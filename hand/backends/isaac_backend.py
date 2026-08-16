from __future__ import annotations

from typing import Any

import numpy as np

from hand.backends.base import HandBackend, HandState


class IsaacBackend(HandBackend):
    """Isaac Lab / Isaac Sim articulation view. Optional; kept free of omni imports at module load."""

    def __init__(self, articulation: Any, dof_indices: np.ndarray) -> None:
        self.articulation = articulation
        self.dof_indices = np.asarray(dof_indices, dtype=np.int64)

    def write_mit(
        self,
        q_des_rad: np.ndarray,
        dq_des_rad_s: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        # Isaac Lab Articulation.set_joint_position_target + effort. Concrete API varies by version.
        if not hasattr(self.articulation, "set_joint_position_target"):
            raise NotImplementedError("Isaac articulation adapter is not bound")
        self.articulation.set_joint_position_target(q_des_rad, joint_ids=self.dof_indices)
        if hasattr(self.articulation, "set_joint_effort_target"):
            self.articulation.set_joint_effort_target(tau_ff, joint_ids=self.dof_indices)

    def read_state(self) -> HandState:
        if not hasattr(self.articulation, "data"):
            raise NotImplementedError("Isaac articulation adapter is not bound")
        q = np.asarray(self.articulation.data.joint_pos[:, self.dof_indices]).reshape(-1)
        dq = np.asarray(self.articulation.data.joint_vel[:, self.dof_indices]).reshape(-1)
        return HandState(q_rad=q, dq_rad_s=dq)
