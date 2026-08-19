"""1 kHz DexHand2 MuJoCo plant. Implements ``HandMitPhysics``. No T800 weld.

ADR-058 owns the numpy MIT ring (shared sim/real, no MuJoCo import).
This module is the optional physics backend that ring steps: 20-D τ,
``dt = 1/1000`` s, motor actuators. Official Hand 2 MJCF ships
``<position>`` actuators (gen-1 kp/kv baked in) at ``timestep=0.002``.
Those are refused here. ADR-060 rewrites the official skeleton into a
mesh-free ``<motor>`` plant that keeps CAD inertias, strips meshes and
contact, and sets ``dt = 0.001`` so the MIT law owns the gains.

Fixture XML is a CI stand-in (placeholder mass, gravity off, no contact).
It is **not** CAD. Combined T800+Hand stays ``PolicyEvalBlocked``.
``grasp_success_rate`` stays JSON null. ``hardware_kp`` /
``command_latency_ms`` stay REQUIRED_INPUT. Soft-body mass is not in
this plant (``com_in_wrist_frame_m`` still REQUIRED_INPUT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from assets.combined.assemble import PolicyEvalBlocked
from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, official_mjcf
from hand.mit_ring import (
    HAND_TIMESTEP_S,
    MIT_RING_YAML,
    HandMitPlant,
    MitRingError,
    require_hand_q,
    require_hand_timestep_1khz,
    sim_forcerange_nm,
)
from interface.schema import load_hand_spec, load_joint_map
from sim.base_env import BaseEnv

HAND_MIT_PHYSICS_YAML = Path(__file__).with_name("hand_mit_physics.yaml")

# Fixture-only. Not CAD. Not a payload rating. Not pad–cardboard.
FIXTURE_LINK_MASS_KG = 0.01
FIXTURE_DIAGINERTIA_KGM2 = (1.0e-6, 1.0e-6, 1.0e-6)
FIXTURE_CAPSULE_HALF_M = 0.03
FIXTURE_CAPSULE_RADIUS_M = 0.006


def load_hand_mit_physics_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or HAND_MIT_PHYSICS_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("hand_mit_physics.yaml must be a mapping")
    flags = (
        "not_dexhand2_contact",
        "not_table_s4",
        "hand_mit_ring",
        "hand_mit_physics",
        "not_wbc_last_action",
        "not_wbc_policy_action",
        "not_decoder_obs",
        "not_concat_t800_plus_hand",
        "not_t800_weld",
        "hand_q_is_zoh",
        "not_hand_q_hermite",
        "dq_des_is_zero",
        "gains_are_mjcf_gen1_carryover",
        "not_hardware_kp",
        "not_invented_latency",
        "not_force_mode_without_hardware_tau",
        "not_mit_ring_from_decoder_onnx",
        "not_mit_ring_finite_diff_dq",
        "clip_tau_to_sim_forcerange",
        "not_sim_forcerange_as_payload_rating",
        "not_official_position_actuators",
        "not_body_500hz_dt",
        "not_official_002s_dt",
        "fixture_inertias_are_placeholders",
        "fixture_zero_gravity_not_cad",
        "not_treat_fixture_as_cad",
        "official_inertias_from_skeleton_mjcf",
        "not_soft_body_in_inertia_plant",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise MitRingError(f"hand_mit_physics.yaml must keep {key}: true")
    spec = load_hand_spec()
    if int(raw["n_active_dof"]) != int(spec.n_active_dof):
        raise MitRingError(f"n_active_dof must stay {spec.n_active_dof}")
    if abs(float(raw["mit_timestep_s"]) - HAND_TIMESTEP_S) > 1e-12:
        raise MitRingError("mit_timestep_s must stay 0.001")
    if abs(float(raw["body_500hz_timestep_s_forbidden"]) - 0.002) > 1e-12:
        raise MitRingError("body_500hz_timestep_s_forbidden must stay 0.002")
    ring = yaml.safe_load(MIT_RING_YAML.read_text(encoding="utf-8"))
    if ring.get("not_wbc_last_action") is not True:
        raise MitRingError("mit_ring.yaml must keep not_wbc_last_action: true")
    return raw


def refuse_combined_robot() -> None:
    raise PolicyEvalBlocked(
        "HandMitMujocoEnv will not weld DexHand2 onto T800. "
        "Fill t800_wrist_to_hand_mount first, then use a dedicated combined env."
    )


def refuse_official_position_actuators() -> None:
    raise MitRingError(
        "official Hand 2 MJCF uses <position> actuators (gen-1 kp/kv baked in). "
        "HandMitMujocoEnv writes MIT τ onto <motor> ctrl. Use source="
        "'official_inertia' (mesh-free CAD inertias) or source='fixture'. "
        "Do not load the unconverted official XML."
    )


def refuse_body_500hz_dt() -> None:
    require_hand_timestep_1khz(0.002)


def refuse_official_002s_dt() -> None:
    """Official Hand 2 MJCF option timestep is 0.002 s. MIT ring is 0.001 s."""
    require_hand_timestep_1khz(0.002)


def refuse_treat_fixture_as_cad() -> None:
    raise MitRingError(
        "fixture inertias are placeholders (0.01 kg / 1e-6 kg·m²). "
        "Do not treat them as CAD. Use source='official_inertia' after cloning "
        "wuji-description (ADR-060). Soft-body CoM stays REQUIRED_INPUT."
    )


def resolve_source(source: str) -> str:
    """Never auto-select official_position (would raise on clone)."""
    if source == "auto":
        if official_mjcf("right").is_file():
            return "official_inertia"
        return "fixture"
    if source in ("official_mit", "official_position", "official_inertia", "fixture"):
        return source
    raise ValueError(f"unknown Hand 2 mjcf source {source!r}")


def official_inertia_mjcf(*, side: str = "right") -> str:
    """Mesh-free MIT XML with official skeleton inertias. Official file untouched."""
    from assets.dexhand2.build.gen_derived import to_meshfree_mit

    src = official_mjcf(side)
    if not src.is_file():
        raise MitRingError(
            f"official Hand 2 MJCF not cloned at {src}. "
            "Run scripts/bootstrap_resources.sh, or use source='fixture'."
        )
    text = src.read_text(encoding="utf-8")
    if "<position " not in text:
        raise MitRingError(f"{src} has no <position> actuators to convert")
    if 'timestep="0.002"' not in text:
        raise MitRingError(
            f"{src} is not the official timestep=0.002 XML; refusing to guess dt."
        )
    return to_meshfree_mit(text)


def plant_mass_kg(model: Any) -> float:
    """Sum of body masses excluding worldbody (index 0)."""
    masses = getattr(model, "body_mass", None)
    if masses is None:
        raise MitRingError("MjModel has no body_mass")
    return float(sum(float(m) for m in masses[1:]))


def fixture_mjcf(*, side: str = "right") -> str:
    """Mesh-free 20-DoF motor plant. Inertias are placeholders. Gravity off."""
    spec = load_hand_spec()
    joint_map = load_joint_map()
    prefix = "r" if side == "right" else "l"
    fr = sim_forcerange_nm(spec)
    armature = float(spec.raw["default_armature_kgm2"])
    ixx, iyy, izz = FIXTURE_DIAGINERTIA_KGM2
    bodies: list[str] = []
    motors: list[str] = []
    for i, entry in enumerate(joint_map):
        jname = entry.mjcf_joint(prefix)
        aname = entry.mjcf_actuator(prefix)
        lo, hi = spec.joint_limits_rad[entry.canonical]
        x_m = 0.012 * i
        bodies.append(
            f'      <body name="link_{entry.canonical}" pos="{x_m:.4g} 0 0">\n'
            f'        <inertial pos="0 0 {FIXTURE_CAPSULE_HALF_M}" mass="{FIXTURE_LINK_MASS_KG}" '
            f'diaginertia="{ixx} {iyy} {izz}"/>\n'
            f'        <joint name="{jname}" type="hinge" axis="0 1 0" '
            f'range="{lo} {hi}" armature="{armature}"/>\n'
            f'        <geom name="cap_{entry.canonical}" type="capsule" '
            f'fromto="0 0 0 0 0 {2 * FIXTURE_CAPSULE_HALF_M}" '
            f'size="{FIXTURE_CAPSULE_RADIUS_M}" contype="0" conaffinity="0"/>\n'
            f"      </body>"
        )
        motors.append(
            f'    <motor name="{aname}" joint="{jname}" gear="1" '
            f'ctrlrange="{-fr[i]} {fr[i]}" forcerange="{-fr[i]} {fr[i]}"/>'
        )
    tree = "\n".join(bodies)
    acts = "\n".join(motors)
    return f"""<mujoco model="dexhand2_mit_fixture_not_official_inertial">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="{prefix}_palm" pos="0 0 0.2">
      <inertial pos="0 0 0" mass="{FIXTURE_LINK_MASS_KG}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="{prefix}_palm_sph" type="sphere" size="0.02" contype="0" conaffinity="0"/>
{tree}
    </body>
  </worldbody>
  <actuator>
{acts}
  </actuator>
</mujoco>
"""


class HandMitMujocoEnv(BaseEnv):
    """Hand-only 1 kHz torque plant. ``include_t800=True`` raises PolicyEvalBlocked."""

    def __init__(
        self,
        *,
        side: str = "right",
        source: str = "auto",
        include_t800: bool = False,
    ) -> None:
        if include_t800:
            refuse_combined_robot()
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install mujoco  (or humanoid-dt[sim])") from exc
        self._mujoco = mujoco
        self.side = "right" if side == "right" else "left"
        self.spec = load_hand_spec()
        self.joint_map = load_joint_map()
        self.source = resolve_source(source)
        if self.source == "official_position":
            refuse_official_position_actuators()
        self.model, self.data, self.xml_note = self._compile()
        require_hand_timestep_1khz(float(self.model.opt.timestep))
        if abs(float(self.model.opt.gravity[2])) > 1e-9:
            raise MitRingError(
                "HandMitMujocoEnv gravity must stay off on this plant. "
                "Do not treat this as a pad–cardboard or free-fall eval."
            )
        self._index()
        self._require_motor_actuators()
        self.plant = HandMitPlant.from_spec(self.spec)

    def _compile(self) -> tuple[Any, Any, str]:
        mujoco = self._mujoco
        if self.source == "official_mit":
            from sim.mujoco_env.hand_env import resolve_hand_xml

            path = resolve_hand_xml(self.side, derived=True, simplified=False)
            model = mujoco.MjModel.from_xml_path(path.as_posix())
            model.opt.timestep = HAND_TIMESTEP_S
            model.opt.gravity[:] = 0.0
            note = path.name
        elif self.source == "official_inertia":
            xml = official_inertia_mjcf(side=self.side)
            model = mujoco.MjModel.from_xml_string(xml)
            note = "official_inertia_meshfree_mit_1khz"
        else:
            xml = fixture_mjcf(side=self.side)
            model = mujoco.MjModel.from_xml_string(xml)
            note = "fixture_20dof_motor_1khz"
        data = mujoco.MjData(model)
        return model, data, note

    def _index(self) -> None:
        mujoco = self._mujoco
        prefix = "r" if self.side == "right" else "l"
        self._act: list[int] = []
        self._qadr: list[int] = []
        self._dadr: list[int] = []
        for entry in self.joint_map:
            aname = entry.mjcf_actuator(prefix)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise KeyError(f"actuator {aname} not in {self.xml_note}")
            jnt_id = int(self.model.actuator_trnid[aid, 0])
            self._act.append(int(aid))
            self._qadr.append(int(self.model.jnt_qposadr[jnt_id]))
            self._dadr.append(int(self.model.jnt_dofadr[jnt_id]))

    def _require_motor_actuators(self) -> None:
        mujoco = self._mujoco
        fixed = int(mujoco.mjtGain.mjGAIN_FIXED)
        for aid in self._act:
            if int(self.model.actuator_gaintype[aid]) != fixed:
                refuse_official_position_actuators()

    @property
    def n_dof(self) -> int:
        return int(self.spec.n_active_dof)

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    def get_q(self) -> np.ndarray:
        return np.array([float(self.data.qpos[i]) for i in self._qadr], dtype=np.float64)

    def get_dq(self) -> np.ndarray:
        return np.array([float(self.data.qvel[i]) for i in self._dadr], dtype=np.float64)

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_q(), self.get_dq()

    def apply_tau_nm(self, tau_nm: np.ndarray) -> np.ndarray:
        """Write a DexHand2 20-D torque command. Does not recompute MIT.

        Used by ADR-059 to apply HandMitPlant τ onto this env. T800 / G1 /
        planner qpos / 45-D concat / 75-D command_schema refused.
        """
        tau = require_hand_q(tau_nm, n_dof=self.n_dof, name="tau")
        for aid, val in zip(self._act, tau, strict=True):
            self.data.ctrl[aid] = float(val)
        return tau

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Write τ, step one 1 kHz tick, return measured q/dq.

        Timestep must be 1/1000 s (ADR-059). Do not resample onto 0.002 s.
        Do not weld T800. Do not finite-diff dq.
        """
        require_hand_timestep_1khz(self.timestep_s)
        self.apply_tau_nm(tau_nm)
        self._mujoco.mj_step(self.model, self.data)
        return self.get_q(), self.get_dq()

    def reset(self) -> dict[str, Any]:
        self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        """One 1 kHz MIT substep. ``action`` is 20-D q_des (rad), not T800 a_t."""
        q, dq = self.read_q_dq()
        tau = self.plant.torque(q, dq, action)
        self.apply_tau_and_step(tau)
        return self._obs()

    def _obs(self) -> dict[str, Any]:
        return {
            "q_rad": self.get_q(),
            "dq_rad_s": self.get_dq(),
            "time_s": float(self.data.time),
            "source": self.source,
            "xml_note": self.xml_note,
        }


def official_mjcf_present() -> bool:
    return official_mjcf("right", root=DEFAULT_UPSTREAM).is_file()
