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
        # Combined MIT plant uses unit-gain <motor> actuators: ctrl = τ.
        motor_mode = self._motor_mode()
        if motor_mode:
            q = []
            dq = []
            for act_id in self.actuator_ids:
                jnt_id = int(self.model.actuator_trnid[act_id, 0])
                q.append(float(self.data.qpos[int(self.model.jnt_qposadr[jnt_id])]))
                dq.append(float(self.data.qvel[int(self.model.jnt_dofadr[jnt_id])]))
            q_arr = np.asarray(q)
            dq_arr = np.asarray(dq)
            tau = kp * (q_des_rad - q_arr) + kd * (dq_des_rad_s - dq_arr) + tau_ff
            for i, act_id in enumerate(self.actuator_ids):
                lo, hi = self.model.actuator_ctrlrange[act_id]
                self.data.ctrl[act_id] = float(np.clip(tau[i], lo, hi))
            return
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = float(q_des_rad[i])
        for i, act_id in enumerate(self.actuator_ids):
            jnt_id = int(self.model.actuator_trnid[act_id, 0])
            dofadr = int(self.model.jnt_dofadr[jnt_id])
            self.data.qfrc_applied[dofadr] = float(tau_ff[i])

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

    def _motor_mode(self) -> bool:
        import mujoco

        if not self.actuator_ids:
            return False
        act_id = int(self.actuator_ids[0])
        return int(self.model.actuator_biastype[act_id]) == int(mujoco.mjtBias.mjBIAS_NONE)
