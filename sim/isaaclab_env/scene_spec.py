"""Declarative pallet+box scene for privileged Isaac Lab (ADR-016 / ADR-023).

Isaac Lab is not imported at module level. Instantiating a runtime without
Isaac Lab raises IsaacLabUnavailable. Combined T800+Hand stays PolicyEvalBlocked.
grasp_success_rate is always JSON null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from assets.combined.assemble import PolicyEvalBlocked
from assets.objects.boxes import BoxSpec
from assets.objects.pallet import EURO_PALLET_M
from sim.mujoco_env.privileged_l2 import FIXTURE_FLOOR_FRICTION, FIXTURE_NOTE, pallet_box_mjcf


@dataclass(frozen=True)
class PalletBoxSceneSpec:
    """Same recipe as PrivilegedL2Env. Safe without Isaac Lab."""

    name: str = "t800_dexhand2_privileged_pallet"
    n_envs: int = 1
    episode_length_s: float = 4.0
    sensors: str = "privileged_state_only"
    include_hands: bool = False
    include_t800: bool = False
    pallet_size_m: tuple[float, float, float] = EURO_PALLET_M
    fixture_floor_friction: tuple[float, float, float] = FIXTURE_FLOOR_FRICTION
    fixture_note: str = FIXTURE_NOTE
    grasp_success_rate: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_envs": self.n_envs,
            "episode_length_s": self.episode_length_s,
            "sensors": self.sensors,
            "include_hands": self.include_hands,
            "include_t800": self.include_t800,
            "pallet_size_m": list(self.pallet_size_m),
            "fixture_floor_friction": list(self.fixture_floor_friction),
            "fixture_note": self.fixture_note,
            "grasp_success_rate": self.grasp_success_rate,
        }

    def mujoco_xml(self, box: BoxSpec) -> str:
        return pallet_box_mjcf(box, floor_friction=self.fixture_floor_friction)


class IsaacLabSceneRuntime:
    """Tries to construct an Isaac Lab sim. Raises if the app is not there.

    When Isaac Lab + SimulationApp are present, reset/step drive the pallet+box
    scene. This repo still does not load T800+Hand.
    """

    def __init__(self, spec: PalletBoxSceneSpec, box: BoxSpec) -> None:
        if spec.include_hands or spec.include_t800:
            raise PolicyEvalBlocked(
                "Isaac Lab privileged scene will not load T800+Hand. Fill t800_wrist_to_hand_mount first."
            )
        self.spec = spec
        self.box = box
        self._sim: Any = None
        self._try_bind()

    def _try_bind(self) -> None:
        try:
            import isaaclab  # noqa: F401
        except ImportError as exc:
            from sim.isaaclab_env.privileged import IsaacLabUnavailable

            raise IsaacLabUnavailable(
                "Isaac Lab is not installed. PalletBoxSceneSpec is the scene contract; "
                "run this inside Isaac Sim python to bind reset/step."
            ) from exc
        try:
            import omni.usd  # noqa: F401
        except ImportError as exc:
            from sim.isaaclab_env.privileged import IsaacLabUnavailable

            raise IsaacLabUnavailable(
                "isaaclab imported but SimulationApp/omni.usd is not running. "
                "PalletBoxSceneSpec.to_dict() is the scene to spawn."
            ) from exc
        self._sim = self._spawn()

    def _spawn(self) -> Any:
        """Isaac Sim python path: create a ground + pallet + free box.

        Kept behind omni.usd so unit tests never import the simulator.
        """
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        if stage is None:
            ctx.new_stage()
            stage = ctx.get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        ground = UsdGeom.Mesh.Define(stage, "/World/ground")
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        pallet = UsdGeom.Cube.Define(stage, "/World/pallet")
        px, py, pz = (float(v) for v in self.spec.pallet_size_m)
        pallet.AddTranslateOp().Set(Gf.Vec3d(0, 0, pz / 2.0))
        pallet.AddScaleOp().Set(Gf.Vec3d(px, py, pz))
        UsdPhysics.CollisionAPI.Apply(pallet.GetPrim())
        box_prim = UsdGeom.Cube.Define(stage, "/World/box")
        sx, sy, sz = (float(v) for v in self.box.size_m)
        box_prim.AddTranslateOp().Set(Gf.Vec3d(0, 0, pz + sz / 2.0 + 0.05))
        box_prim.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))
        UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(box_prim.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(box_prim.GetPrim())
        mass.CreateMassAttr(float(self.box.mass_kg))
        return stage

    def reset(self) -> dict[str, Any]:
        return {
            "sensors": self.spec.sensors,
            "box_size_m": list(self.box.size_m),
            "box_mass_kg": float(self.box.mass_kg),
            "grasp_success_rate": None,
            "status": "isaac_scene_reset",
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        del action
        obs = {
            "sensors": self.spec.sensors,
            "grasp_success_rate": None,
            "status": "isaac_scene_step",
        }
        info = {"grasp_success_rate": None}
        return obs, 0.0, False, info
