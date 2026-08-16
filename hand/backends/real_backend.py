"""Wuji SDK real backend.

Needs: pip install wuji-sdk, device on 192.168.1.0/24 (factory IPs in dexhand2_spec.yaml).
This module does not import wuji_sdk at top level so CI without hardware still collects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hand.backends.base import HandBackend, HandState
from interface.schema import load_joint_map


class RealBackend(HandBackend):
    def __init__(self, hand: Any | None = None, *, handedness: str | None = None) -> None:
        self._hand = hand
        self._handedness = handedness
        self._pub: Any = None
        self._map = load_joint_map()

    def connect(self) -> None:
        try:
            from wuji_sdk import Handedness, SdkManager
        except ImportError as exc:  # pragma: no cover
            raise NotImplementedError(
                "wuji-sdk is not installed. See https://docs.wuji.tech/docs/zh/wuji-sdk/latest/"
            ) from exc
        if self._hand is not None:
            return
        manager = SdkManager.instance()
        kwargs: dict[str, Any] = {"device_name": "wuji_hand_2"}
        if self._handedness:
            kwargs["handedness"] = Handedness.Right if self._handedness == "right" else Handedness.Left
        self._hand = manager.connect(**kwargs)
        self._pub = self._hand.joint_command().publish()

    def write_mit(
        self,
        q_des_rad: np.ndarray,
        dq_des_rad_s: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        if self._hand is None or self._pub is None:
            raise NotImplementedError("RealBackend.connect() was not called")
        from wuji_sdk import JointCommand

        # SDK joint_command expects exactly 20 JointCommand in sdk_index order.
        cmds = [
            JointCommand(position=float(q_des_rad[i]), velocity=float(dq_des_rad_s[i]), effort=float(tau_ff[i]))
            for i in range(len(self._map))
        ]
        self._pub.send(cmds)
        try:
            self._hand.mit_params().set(list(zip(kp.tolist(), kd.tolist(), strict=True)))
        except Exception:
            # Some firmware builds only accept a single (kp, kd) pair.
            self._hand.mit_params().set((float(kp.mean()), float(kd.mean())))

    def read_state(self) -> HandState:
        if self._hand is None:
            raise NotImplementedError("RealBackend.connect() was not called")
        n = len(self._map)
        q = np.zeros(n)
        dq = np.zeros(n)
        tau = np.zeros(n)
        sub = self._hand.joint_states().subscribe()
        frame = sub.recv()
        sub.close()
        if frame is None:
            return HandState(q_rad=q, dq_rad_s=dq, tau_est=tau)
        for joint in frame.joints:
            idx = int(joint.nid)
            if 0 <= idx < n:
                q[idx] = float(joint.position)
                dq[idx] = float(joint.velocity)
                tau[idx] = float(joint.effort)
        return HandState(q_rad=q, dq_rad_s=dq, tau_est=tau)
