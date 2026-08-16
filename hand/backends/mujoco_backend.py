from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hand.backends.base import HandBackend, HandState
from hand.mit import mit_torque


class MujocoBackend(HandBackend):
    """Applies MIT τ = kp(qd−q)+kd(dqd−dq)+τ_ff.

    Official MJCF uses <position> actuators (kp/kv baked into XML). Derived
    ``*_mit.xml`` uses <motor> actuators so *this* kp/kd vector is the plant.
    """

    def __init__(self, model: Any, data: Any, actuator_ids: Sequence[int]) -> None:
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("mujoco is required for MujocoBackend") from exc
        self.model = model
        self.data = data
        self.actuator_ids = list(actuator_ids)

    def _is_motor(self, act_id: int) -> bool:
        import mujoco

        # mjGAIN_FIXED + force/acc biastype ≈ motor. Position actuators use affine gain.
        return int(self.model.actuator_gaintype[act_id]) == int(mujoco.mjtGain.mjGAIN_FIXED)

    def write_mit(
        self,
        q_des_rad: np.ndarray,
        dq_des_rad_s: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        state = self.read_state()
        tau = mit_torque(state.q_rad, state.dq_rad_s, q_des_rad, dq_des_rad_s, tau_ff, kp, kd)
        for i, act_id in enumerate(self.actuator_ids):
            if self._is_motor(act_id):
                self.data.ctrl[act_id] = float(tau[i])
            else:
                self.data.ctrl[act_id] = float(q_des_rad[i])
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
