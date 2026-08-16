from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hand.backends.base import HandBackend, HandState


class MujocoBackend(HandBackend):
    """Applies MIT-equivalent pd+ff through MuJoCo ctrl. Optional dependency."""

    def __init__(self, model: Any, data: Any, actuator_ids: Sequence[int]) -> None:
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("mujoco is required for MujocoBackend") from exc
        self.model = model
        self.data = data
        self.actuator_ids = list(actuator_ids)

    def write_mit(
        self,
        q_des_rad: np.ndarray,
        dq_des_rad_s: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        # Official Hand 2 MJCF uses <position> actuators (kp/kv already on the model).
        # We set ctrl = q_des. Feed-forward torque is added via qfrc_applied on the joint.
        import mujoco

        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = float(q_des_rad[i])
        for i, act_id in enumerate(self.actuator_ids):
            jnt_id = int(self.model.actuator_trnid[act_id, 0])
            dofadr = int(self.model.jnt_dofadr[jnt_id])
            self.data.qfrc_applied[dofadr] = float(tau_ff[i])
        _ = mujoco  # imported for type checkers; stepping is the caller's job

    def read_state(self) -> HandState:
        q = []
        dq = []
        for act_id in self.actuator_ids:
            jnt_id = int(self.model.actuator_trnid[act_id, 0])
            qadr = int(self.model.jnt_qposadr[jnt_id])
            dadr = int(self.model.jnt_dofadr[jnt_id])
            q.append(float(self.data.qpos[qadr]))
            dq.append(float(self.data.qvel[dadr]))
        return HandState(q_rad=np.array(q), dq_rad_s=np.array(dq))
