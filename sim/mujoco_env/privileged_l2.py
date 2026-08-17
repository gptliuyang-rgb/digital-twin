"""Privileged-state L2: box pose + joint state, no camera.

Grasp-success is never computed (ADR-004). Combined T800+Hand stays
PolicyEvalBlocked until the flange SE(3) is measured (ADR-009).

The box-on-pallet drop uses an explicit *fixture* floor friction so the
stack_stable metric has a physics self-test. That friction is not
pad–cardboard and must not be copied into dexhand2_spec.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from assets.combined.assemble import PolicyEvalBlocked, mount_ready
from assets.objects.boxes import BoxSpec, sample_box
from assets.objects.pallet import EURO_PALLET_M
from eval.stack_metrics import stack_stable
from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec
from sim.base_env import BaseEnv
from sim.payload import box_inertia_kgm2

# World-box fixture only. NOT fingertip–cardboard (those stay REQUIRED_INPUT).
FIXTURE_FLOOR_FRICTION = (0.8, 0.005, 0.0001)
FIXTURE_NOTE = "synthetic floor-box friction for stack_stable self-test; not pad-cardboard"


@dataclass
class PrivilegedObs:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    box_pos_m: np.ndarray
    box_quat_wxyz: np.ndarray
    wrist_pos_m: np.ndarray | None = None
    time_s: float = 0.0


def pallet_box_mjcf(box: BoxSpec, *, floor_friction: tuple[float, float, float] = FIXTURE_FLOOR_FRICTION) -> str:
    """Minimal MJCF: Euro pallet + one box. No hand. No invented pad contact."""
    sx, sy, sz = (0.5 * float(v) for v in box.size_m)
    px, py, pz = (0.5 * float(v) for v in EURO_PALLET_M)
    fr = " ".join(str(x) for x in floor_friction)
    inertia = box_inertia_kgm2(box.mass_kg, box.size_m)
    ixx, iyy, izz = float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2])
    com = box.com_offset_frac * box.size_m
    return f"""<mujoco model="privileged_l2_pallet">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <geom friction="{fr}"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0"/>
    <body name="pallet" pos="0 0 {pz}">
      <geom name="pallet_geom" type="box" size="{px} {py} {pz}" rgba="0.55 0.35 0.15 1"/>
    </body>
    <body name="box" pos="0 0 {2 * pz + sz + 0.05}">
      <freejoint name="box_free"/>
      <inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" mass="{box.mass_kg:.4f}"
                diaginertia="{ixx:.6e} {iyy:.6e} {izz:.6e}"/>
      <geom name="box_geom" type="box" size="{sx} {sy} {sz}" rgba="0.8 0.6 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def refuse_grasp_success_key(report: dict[str, Any]) -> None:
    if "grasp_success_rate" in report and report["grasp_success_rate"] is not None:
        raise AssertionError("privileged L2 must not emit a numeric grasp_success_rate")


class PrivilegedL2Env(BaseEnv):
    """Box-on-pallet privileged physics. Hands are not welded onto T800 here."""

    def __init__(self, box: BoxSpec | None = None, rng: np.random.Generator | None = None) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install mujoco  (or humanoid-dt[sim])") from exc
        if mount_ready():
            # Flange known does not authorize a combined-robot load in this env.
            pass
        self.spec = load_hand_spec()
        self.box = box or sample_box(rng or np.random.default_rng(0))
        self.xml = pallet_box_mjcf(self.box)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco

    def reset(self) -> dict[str, Any]:
        self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        del action
        self._mujoco.mj_step(self.model, self.data)
        return self._obs()

    def drop_and_settle(self, settle_s: float = 2.0) -> dict[str, Any]:
        self.reset()
        n = int(round(settle_s / float(self.model.opt.timestep)))
        xy = []
        quat = []
        for _ in range(max(n, 1)):
            obs = self.step(np.zeros(0))
            xy.append(obs["box_pos_m"][:2].copy())
            quat.append(obs["box_quat_wxyz"].copy())
        xy_arr = np.asarray(xy)
        quat_arr = np.asarray(quat)
        support = np.array(
            [
                [-EURO_PALLET_M[0] / 2, -EURO_PALLET_M[1] / 2],
                [EURO_PALLET_M[0] / 2, EURO_PALLET_M[1] / 2],
            ]
        )
        gate = stack_stable(xy_arr, quat_arr, support, dt_s=float(self.model.opt.timestep), settle_t_s=settle_s)
        return {
            "stack_stable": gate.stable,
            "max_drift_m": gate.max_drift_m,
            "max_tilt_rad": gate.max_tilt_rad,
            "fixture_friction": FIXTURE_FLOOR_FRICTION,
            "fixture_note": FIXTURE_NOTE,
            "grasp_success_rate": None,
            "status": "privileged_pallet_drop",
        }

    def _obs(self) -> dict[str, Any]:
        bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, "box")
        pos = self.data.xpos[bid].copy()
        quat = self.data.xquat[bid].copy()
        return {
            "box_pos_m": pos,
            "box_quat_wxyz": quat,
            "time_s": float(self.data.time),
            "q_rad": np.zeros(0),
            "dq_rad_s": np.zeros(0),
        }


def combined_robot_blocked() -> None:
    raise PolicyEvalBlocked(
        "PrivilegedL2Env will not load T800+Hand. Fill t800_wrist_to_hand_mount first."
    )


def uncalibrated_contact() -> bool:
    spec = load_hand_spec()
    return spec.raw.get("friction_vs_cardboard_static") == REQUIRED_INPUT_TOKEN
