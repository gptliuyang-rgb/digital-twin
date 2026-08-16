"""Hand-only MuJoCo env. Combined T800+hand is refused (PolicyEvalBlocked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from assets.combined.assemble import PolicyEvalBlocked, mount_ready
from assets.dexhand2.build.gen_derived import DERIVED, generate
from assets.dexhand2.build.ingest_official import official_mjcf
from hand.backends.mujoco_backend import MujocoBackend
from hand.controller import DexHand2Controller, SafetyLimits
from interface.schema import load_hand_spec, load_joint_map
from sim.base_env import BaseEnv


def resolve_hand_xml(side: str = "right", *, derived: bool = True, simplified: bool = False) -> Path:
    if derived:
        variant = "simplified_mit" if simplified else "with_pad_spheres_mit"
        path = DERIVED / f"{side}_{variant}.xml"
        if not path.is_file():
            generate(side, mit_motors=True, simplified=simplified)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    path = official_mjcf(side)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class HandOnlyMujocoEnv(BaseEnv):
    """Single Wuji Hand 2. Does not weld onto T800."""

    def __init__(
        self,
        *,
        side: str = "right",
        derived: bool = True,
        simplified: bool = False,
        safety: SafetyLimits | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install mujoco  (or humanoid-dt[sim])") from exc
        if not derived and mount_ready():
            # Even with a measured flange we still do not silently load a combined robot here.
            pass
        self.xml_path = resolve_hand_xml(side, derived=derived, simplified=simplified)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path.as_posix())
        self.data = mujoco.MjData(self.model)
        self.spec = load_hand_spec()
        joint_map = load_joint_map()
        prefix = "r" if side == "right" else "l"
        act_ids = []
        for entry in joint_map:
            name = entry.mjcf_actuator(prefix[0])
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise KeyError(f"actuator {name} not in {self.xml_path}")
            act_ids.append(int(aid))
        backend = MujocoBackend(self.model, self.data, act_ids)
        self.controller = DexHand2Controller(
            self.spec,
            backend,
            safety=safety or SafetyLimits(max_delta_q_rad=1.0, velocity_limit_rad_s=10.0),
        )
        self._mujoco = mujoco

    def reset(self) -> dict[str, Any]:
        self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        self.controller.set_joint_targets(np.asarray(action, dtype=np.float64))
        self._mujoco.mj_step(self.model, self.data)
        return self._obs()

    def hold(self, seconds: float, q_des: np.ndarray | None = None) -> dict[str, float]:
        """Hold a posture; return joint-velocity RMS. Used as the no-jitter gate."""
        q = np.zeros(self.spec.n_active_dof) if q_des is None else np.asarray(q_des, dtype=np.float64)
        n = int(round(seconds / float(self.model.opt.timestep)))
        vel = []
        for _ in range(max(n, 1)):
            obs = self.step(q)
            vel.append(obs["dq_rad_s"])
        arr = np.asarray(vel)
        return {
            "qvel_rms_rad_s": float(np.sqrt(np.mean(arr**2))),
            "seconds": seconds,
            "n_steps": n,
            "xml": str(self.xml_path),
        }

    def _obs(self) -> dict[str, Any]:
        state = self.controller.get_state()
        return {"q_rad": state.q_rad, "dq_rad_s": state.dq_rad_s, "time_s": float(self.data.time)}


def refuse_combined_robot() -> None:
    raise PolicyEvalBlocked(
        "HandOnlyMujocoEnv will not load a T800+Hand combined model. "
        "Fill t800_wrist_to_hand_mount first, then use a dedicated combined env."
    )
