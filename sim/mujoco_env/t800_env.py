"""T800 body-only MuJoCo env. Hands are never welded (PolicyEvalBlocked).

Two XML sources:
- official Native SDK ``serial_t800.xml`` (inertias + collision from EngineAI)
- kinematics fixture generated from ``t800_kinematics.yaml`` (placeholder mass;
  CI path when the SDK is not cloned)

Neither source writes DexHand2 contact parameters. Floor friction on the
free-base path is the official collision default, labelled as WBC floor, not
pad–cardboard.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from assets.combined.assemble import PolicyEvalBlocked
from interface.schema import REPO_ROOT
from sim.base_env import BaseEnv
from sim.urdf_fk import load_t800_kinematics, rpy_to_matrix
from wbc.dims import load_t800_sonic
from wbc.pd_stand import pd_stand_kp_kd

OFFICIAL_MJCF = (
    REPO_ROOT
    / "third_party"
    / "engineai-native-sdk"
    / "assets"
    / "resource"
    / "robot"
    / "t800"
    / "xml"
    / "serial_t800.xml"
)

# Fixture-only. Not CAD. Not a payload rating.
FIXTURE_MASS_KG = 1.0
FIXTURE_DIAGINERTIA_KGM2 = (0.01, 0.01, 0.01)
FIXTURE_KP = 50.0
FIXTURE_KD = 2.0
# Copied from official serial_t800.xml collision default. NOT pad–cardboard.
OFFICIAL_FLOOR_FRICTION = (1.0, 0.005, 0.0001)
ROOT_Z_M = 1.03  # official MJCF LINK_BASE pos z


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(r))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def rpy_to_quat_wxyz(rpy: list[float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(x) for x in np.asarray(rpy, dtype=np.float64).reshape(3))
    return _matrix_to_quat_wxyz(rpy_to_matrix(roll, pitch, yaw))


def fixture_mjcf(*, pinned_base: bool, add_floor: bool) -> str:
    """Mesh-free MJCF from committed kinematics. Inertias are placeholders."""
    kin = load_t800_kinematics()
    order = list(kin["joint_order"])
    limits = kin["mjcf_revolute_limits_rad"]
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for joint in kin["joints"]:
        children[str(joint["parent"])].append(joint)
    ixx, iyy, izz = FIXTURE_DIAGINERTIA_KGM2

    def emit_from(parent: str, indent: str) -> list[str]:
        lines: list[str] = []
        pad = indent + "  "
        for joint in children[parent]:
            child = str(joint["child"])
            pos = " ".join(f"{float(x):.8g}" for x in joint["origin_xyz_m"])
            quat = " ".join(f"{float(x):.8g}" for x in rpy_to_quat_wxyz(joint["origin_rpy_rad"]))
            lines.append(f'{indent}<body name="{child}" pos="{pos}" quat="{quat}">')
            lines.append(
                f'{pad}<inertial pos="0 0 0" mass="{FIXTURE_MASS_KG}" '
                f'diaginertia="{ixx} {iyy} {izz}"/>'
            )
            if child in ("LINK_FOOT_L", "LINK_FOOT_R"):
                lines.append(f'{pad}<geom name="{child}_box" type="box" size="0.10 0.05 0.02"/>')
            else:
                lines.append(
                    f'{pad}<geom name="{child}_sph" type="sphere" size="0.03" '
                    f'contype="0" conaffinity="0"/>'
                )
            if joint["type"] == "revolute":
                axis = " ".join(f"{float(x):.8g}" for x in joint["axis"])
                lo, hi = limits[joint["name"]]
                lines.append(
                    f'{pad}<joint name="{joint["name"]}" type="hinge" axis="{axis}" '
                    f'range="{float(lo)} {float(hi)}"/>'
                )
            lines.extend(emit_from(child, pad))
            lines.append(f"{indent}</body>")
        return lines

    floor = ""
    if add_floor:
        fr = " ".join(str(x) for x in OFFICIAL_FLOOR_FRICTION)
        floor = (
            f'    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" friction="{fr}"/>\n'
        )
    free = "" if pinned_base else "      <freejoint/>\n"
    motors = "\n".join(
        f'    <motor name="motor_{name}" joint="{name}" gear="1" ctrllimited="false"/>' for name in order
    )
    tree = "\n".join(emit_from("LINK_BASE", "      "))
    return f"""<mujoco model="t800_kinematics_fixture_not_official_inertial">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
{floor}    <body name="LINK_BASE" pos="0 0 {ROOT_Z_M}">
      <inertial pos="0 0 0" mass="{FIXTURE_MASS_KG}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="LINK_BASE_sph" type="sphere" size="0.05" contype="0" conaffinity="0"/>
{free}{tree}
    </body>
  </worldbody>
  <actuator>
{motors}
  </actuator>
</mujoco>
"""


def refuse_combined_robot() -> None:
    raise PolicyEvalBlocked(
        "T800MujocoEnv will not weld DexHand2. Fill t800_wrist_to_hand_mount first."
    )


def resolve_source(source: str) -> str:
    if source == "auto":
        return "official" if OFFICIAL_MJCF.is_file() else "fixture"
    if source in ("official", "fixture"):
        return source
    raise ValueError(f"unknown T800 mjcf source {source!r}")


class T800MujocoEnv(BaseEnv):
    """Body-only T800. ``include_hands=True`` raises PolicyEvalBlocked."""

    def __init__(
        self,
        *,
        source: str = "auto",
        pinned_base: bool = True,
        add_floor: bool | None = None,
        include_hands: bool = False,
    ) -> None:
        if include_hands:
            refuse_combined_robot()
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install mujoco  (or humanoid-dt[sim])") from exc
        self._mujoco = mujoco
        self.source = resolve_source(source)
        self.pinned_base = bool(pinned_base)
        self.add_floor = bool(self.pinned_base is False if add_floor is None else add_floor)
        self.cfg = load_t800_sonic()
        self.joint_order = list(self.cfg["joint_order"])
        self.tracked_bodies = dict(self.cfg["tracked_bodies"])
        self.model, self.data, self.xml_note = self._compile()
        self._index()
        if self.source == "official":
            self.kp, self.kd = pd_stand_kp_kd()
            self.gains_source = "pd_stand_bringup_not_sonic"
        else:
            n = len(self.joint_order)
            self.kp = np.full(n, FIXTURE_KP)
            self.kd = np.full(n, FIXTURE_KD)
            self.gains_source = "fixture_mild_pd_not_cad"

    def _compile(self) -> tuple[Any, Any, str]:
        mujoco = self._mujoco
        if self.source == "official":
            if not OFFICIAL_MJCF.is_file():
                raise FileNotFoundError(f"missing {OFFICIAL_MJCF}; run scripts/bootstrap_resources.sh")
            spec = mujoco.MjSpec.from_file(OFFICIAL_MJCF.as_posix())
            if self.pinned_base:
                base = spec.body("LINK_BASE")
                free = [j for j in base.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
                for joint in free:
                    spec.delete(joint)
            if self.add_floor:
                geom = spec.worldbody.add_geom()
                geom.name = "floor"
                geom.type = mujoco.mjtGeom.mjGEOM_PLANE
                geom.size = [5.0, 5.0, 0.1]
                geom.friction = list(OFFICIAL_FLOOR_FRICTION)
            model = spec.compile()
            note = "official_serial_t800.xml"
        else:
            xml = fixture_mjcf(pinned_base=self.pinned_base, add_floor=self.add_floor)
            model = mujoco.MjModel.from_xml_string(xml)
            note = "kinematics_fixture_placeholder_inertia"
        data = mujoco.MjData(model)
        return model, data, note

    def _index(self) -> None:
        mujoco = self._mujoco
        self._qadr: list[int] = []
        self._vadr: list[int] = []
        self._act: list[int] = []
        for name in self.joint_order:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint {name} missing from {self.xml_note}")
            self._qadr.append(int(self.model.jnt_qposadr[jid]))
            self._vadr.append(int(self.model.jnt_dofadr[jid]))
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
            if aid < 0:
                raise KeyError(f"actuator motor_{name} missing from {self.xml_note}")
            self._act.append(int(aid))
        self._body_id = {
            role: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link))
            for role, link in self.tracked_bodies.items()
        }
        missing = [role for role, bid in self._body_id.items() if bid < 0]
        if missing:
            raise KeyError(f"tracked bodies missing from MJCF: {missing}")

    def get_q(self) -> np.ndarray:
        return np.array([self.data.qpos[i] for i in self._qadr], dtype=np.float64)

    def get_dq(self) -> np.ndarray:
        return np.array([self.data.qvel[i] for i in self._vadr], dtype=np.float64)

    def set_q(self, q_rad: np.ndarray, dq_rad_s: np.ndarray | None = None) -> None:
        q = np.asarray(q_rad, dtype=np.float64).reshape(len(self.joint_order))
        for adr, val in zip(self._qadr, q, strict=True):
            self.data.qpos[adr] = val
        if dq_rad_s is None:
            for adr in self._vadr:
                self.data.qvel[adr] = 0.0
        else:
            dq = np.asarray(dq_rad_s, dtype=np.float64).reshape(len(self.joint_order))
            for adr, val in zip(self._vadr, dq, strict=True):
                self.data.qvel[adr] = val

    def set_root(self, pos_m: np.ndarray, quat_wxyz: np.ndarray) -> None:
        if self.pinned_base:
            return
        self.data.qpos[0:3] = np.asarray(pos_m, dtype=np.float64).reshape(3)
        self.data.qpos[3:7] = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        self.data.qvel[0:6] = 0.0

    def body_pose(self, role: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._body_id[role]
        return self.data.xpos[bid].copy(), self.data.xmat[bid].reshape(3, 3).copy()

    def apply_pd(self, q_des: np.ndarray) -> np.ndarray:
        q = self.get_q()
        dq = self.get_dq()
        tau = self.kp * (np.asarray(q_des, dtype=np.float64) - q) - self.kd * dq
        for aid, val in zip(self._act, tau, strict=True):
            self.data.ctrl[aid] = val
        return tau

    def reset(self) -> dict[str, Any]:
        self._mujoco.mj_resetData(self.model, self.data)
        if not self.pinned_base:
            self.set_root(np.array([0.0, 0.0, ROOT_Z_M]), np.array([1.0, 0.0, 0.0, 0.0]))
        self._mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        self.apply_pd(action)
        self._mujoco.mj_step(self.model, self.data)
        return self._obs()

    def hold(self, seconds: float, q_des: np.ndarray | None = None) -> dict[str, float]:
        q = np.zeros(len(self.joint_order)) if q_des is None else np.asarray(q_des, dtype=np.float64)
        n = int(round(seconds / float(self.model.opt.timestep)))
        vel = []
        for _ in range(max(n, 1)):
            obs = self.step(q)
            vel.append(obs["dq_rad_s"])
        arr = np.asarray(vel)
        return {
            "qvel_rms_rad_s": float(np.sqrt(np.mean(arr**2))),
            "joint_mae_rad": float(np.mean(np.abs(self.get_q() - q))),
            "seconds": seconds,
            "n_steps": n,
            "source": self.source,
            "pinned_base": self.pinned_base,
        }

    def _obs(self) -> dict[str, Any]:
        pelvis, _ = self.body_pose("pelvis")
        return {
            "q_rad": self.get_q(),
            "dq_rad_s": self.get_dq(),
            "pelvis_pos_m": pelvis,
            "time_s": float(self.data.time),
            "source": self.source,
        }
