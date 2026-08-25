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
        _stabilize_plant(self.model)
        if scene == "industrial":
            # Stiffer constraint solve for welds + elliptic friction on the cell.
            self.model.opt.iterations = max(int(self.model.opt.iterations), 60)
            self.model.opt.ls_iterations = max(int(self.model.opt.ls_iterations), 40)
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
        body_kp: float = 90.0,
        body_kd: float = 12.0,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        payload_body: str | None = None,
        payload_mass_kg: float = 0.0,
        max_err_rad: float = 0.25,
    ) -> dict[str, Any]:
        import mujoco

        from sim.mujoco_env.dynamics import apply_body_pd, apply_payload_support

        apply_mit(self.model, self.data, self.handles.left_hand, left_q, kp_scale=kp_scale, kd_scale=kd_scale)
        apply_mit(self.model, self.data, self.handles.right_hand, right_q, kp_scale=kp_scale, kd_scale=kd_scale)
        if body_q_des is not None:
            mujoco.mj_forward(self.model, self.data)
            apply_body_pd(
                self.model,
                self.data,
                self.handles.body_actuator_ids,
                body_q_des,
                body_kp,
                body_kd,
                gravity_comp=True,
                max_err_rad=max_err_rad,
            )
            if payload_body and payload_mass_kg > 0.0:
                apply_payload_support(
                    self.model,
                    self.data,
                    self.handles.body_actuator_ids,
                    payload_body,
                    payload_mass_kg,
                )
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


def _stabilize_plant(model) -> None:
    """Viscous damping + armature so massless coupled finger dofs do not explode.

    Official Hand 2 MJCF has DIP/IP mimic joints with ~0 inertia. Forward dynamics
    (mj_step) needs a little armature/damping; kinematic mj_forward does not.
    T800 serial joints also ship with empty damping in the included links file.
    """
    for j in range(model.njnt):
        name = model.joint(j).name or ""
        dadr = int(model.jnt_dofadr[j])
        nv = 6 if int(model.jnt_type[j]) == 0 else 1
        if int(model.jnt_type[j]) == 0:
            # Free props (carton, scan gun): linear + angular damping so they
            # settle on the bench instead of skating from residual penetration.
            for k in range(3):
                model.dof_damping[dadr + k] = max(float(model.dof_damping[dadr + k]), 0.8)
            for k in range(3, 6):
                model.dof_damping[dadr + k] = max(float(model.dof_damping[dadr + k]), 0.05)
            continue
        if name.startswith("J"):
            damp, arm = 0.6, 0.0
        elif name.startswith(("l_", "r_")) or "finger" in name or "thumb" in name:
            damp, arm = 0.04, 2e-4
        else:
            continue
        for k in range(nv):
            model.dof_damping[dadr + k] = max(float(model.dof_damping[dadr + k]), damp)
            if arm > 0.0:
                model.dof_armature[dadr + k] = max(float(model.dof_armature[dadr + k]), arm)
    for i in range(model.nv):
        if float(model.dof_invweight0[i]) > 1e3:
            model.dof_armature[i] = max(float(model.dof_armature[i]), 3e-4)
            model.dof_damping[i] = max(float(model.dof_damping[i]), 0.05)
