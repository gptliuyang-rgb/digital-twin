"""MuJoCo env wrapping the combined T800 + DexHand2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from assets.combined.assemble import assemble_mjcf, kinematic_bringup
from interface.schema import REPO_ROOT
from sim.base_env import BaseEnv
from sim.mujoco_env.hand_mit import MitActuatorSet, apply_mit, lookup_hand_actuators
from sim.mujoco_env.scene import SceneSpec, industrial_xml


def _load_arm_joints() -> dict[str, list[str]]:
    raw = yaml.safe_load(
        (REPO_ROOT / "assets" / "engineai" / "meta" / "t800_joints.yaml").read_text(encoding="utf-8")
    )
    return {k: list(v) for k, v in raw["arm_joints"].items()}


@dataclass
class TwinHandles:
    left_hand: MitActuatorSet
    right_hand: MitActuatorSet
    body_actuator_ids: np.ndarray
    arm_joints: dict[str, list[str]]


class CombinedMujocoEnv(BaseEnv):
    """Pinned-base T800 + two DexHand2 plants. Optional industrial scene."""

    def __init__(
        self,
        *,
        scene: str = "empty",
        scene_spec: SceneSpec | None = None,
        timestep_s: float = 0.001,
    ) -> None:
        import mujoco

        xml, manifest = assemble_mjcf(timestep_s=timestep_s)
        if scene == "industrial":
            xml = industrial_xml(xml, scene_spec)
        self.manifest = manifest
        self.scene = scene
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        gains = manifest["hand_mit_gains"]
        self.handles = TwinHandles(
            left_hand=lookup_hand_actuators(self.model, gains, "left"),
            right_hand=lookup_hand_actuators(self.model, gains, "right"),
            body_actuator_ids=_t800_actuator_ids(self.model),
            arm_joints=_load_arm_joints(),
        )
        self._bringup = kinematic_bringup()

    def reset(self) -> dict[str, Any]:
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        import mujoco

        self.data.ctrl[:] = np.asarray(action, dtype=np.float64).reshape(self.model.nu)
        mujoco.mj_step(self.model, self.data)
        return self._obs()

    def step_mit(
        self,
        left_q: np.ndarray,
        right_q: np.ndarray,
        *,
        body_q_des: np.ndarray | None = None,
        body_kp: float = 40.0,
        body_kd: float = 2.0,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> dict[str, Any]:
        import mujoco

        apply_mit(self.model, self.data, self.handles.left_hand, left_q, kp_scale=kp_scale, kd_scale=kd_scale)
        apply_mit(self.model, self.data, self.handles.right_hand, right_q, kp_scale=kp_scale, kd_scale=kd_scale)
        if body_q_des is not None:
            _apply_body_pd(self.model, self.data, self.handles.body_actuator_ids, body_q_des, body_kp, body_kd)
        mujoco.mj_step(self.model, self.data)
        return self._obs()

    def xpos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def site_xpos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(name).id].copy()

    def _obs(self) -> dict[str, Any]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "time": float(self.data.time),
            "policy_eval_forbidden": True,
            "flange": self._bringup.get("note", "kinematic_bringup_identity"),
        }


def _t800_actuator_ids(model) -> np.ndarray:
    ids = []
    for i in range(model.nu):
        name = model.actuator(i).name
        if name.startswith("motor_J"):
            ids.append(i)
    return np.asarray(ids, dtype=np.int32)


def _apply_body_pd(model, data, act_ids: np.ndarray, q_des_full: np.ndarray, kp: float, kd: float) -> None:
    for act_id in act_ids:
        jnt = int(model.actuator_trnid[act_id, 0])
        qadr = int(model.jnt_qposadr[jnt])
        dadr = int(model.jnt_dofadr[jnt])
        tau = kp * (q_des_full[qadr] - data.qpos[qadr]) - kd * data.qvel[dadr]
        lo, hi = model.actuator_ctrlrange[act_id]
        data.ctrl[act_id] = float(np.clip(tau, lo, hi))
