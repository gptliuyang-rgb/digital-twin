"""Privileged Isaac Lab env spec. Isaac Lab is not imported at module level.

Same pallet+box recipe as ``sim.mujoco_env.privileged_l2``. Grasp-success is
always JSON null (ADR-004 / ADR-014 / ADR-016). Combined T800+Hand stays
PolicyEvalBlocked until the flange SE(3) is CAD-measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from assets.combined.assemble import PolicyEvalBlocked
from assets.objects.boxes import BoxSpec, sample_box
from assets.objects.pallet import EURO_PALLET_M
from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec
from sim.base_env import BaseEnv
from sim.mujoco_env.privileged_l2 import FIXTURE_FLOOR_FRICTION, FIXTURE_NOTE, refuse_grasp_success_key

# Do not import isaaclab / omni / isaacsim here. Unit tests import this module.


class IsaacLabUnavailable(ImportError):
    """Raised when the privileged Isaac env is constructed without Isaac Lab."""


@dataclass(frozen=True)
class PrivilegedIsaacCfg:
    """Declarative scene. Safe to instantiate without Isaac Lab."""

    name: str = "t800_dexhand2_privileged_pallet"
    n_envs: int = 1
    episode_length_s: float = 4.0
    sensors: str = "privileged_state_only"  # no RTX cameras in this env
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
            "status": "privileged_isaac_cfg",
        }


def refuse_combined_robot(cfg: PrivilegedIsaacCfg | None = None) -> None:
    cfg = cfg or PrivilegedIsaacCfg()
    if cfg.include_hands or cfg.include_t800:
        raise PolicyEvalBlocked(
            "Privileged Isaac env will not load T800+Hand. Fill t800_wrist_to_hand_mount first."
        )


class PrivilegedIsaacEnv(BaseEnv):
    """Constructs only when Isaac Lab is installed. Tests must not instantiate this."""

    def __init__(self, cfg: PrivilegedIsaacCfg | None = None, box: BoxSpec | None = None) -> None:
        refuse_combined_robot(cfg)
        try:
            import isaaclab  # noqa: F401
        except ImportError as exc:
            raise IsaacLabUnavailable(
                "Isaac Lab is not installed. Use PrivilegedIsaacCfg for unit tests, "
                "or pip-install Isaac Lab in the Isaac Sim python env."
            ) from exc
        self.cfg = cfg or PrivilegedIsaacCfg()
        self.box = box or sample_box(np.random.default_rng(0))
        self.spec = load_hand_spec()

    def reset(self) -> dict[str, Any]:
        raise IsaacLabUnavailable("Isaac Lab env reset is not wired in this repo yet")

    def step(self, action):
        raise IsaacLabUnavailable("Isaac Lab env step is not wired in this repo yet")


def privileged_report(cfg: PrivilegedIsaacCfg | None = None) -> dict[str, Any]:
    """Always-safe report: cfg dump + blocked combined robot + null grasp success."""
    cfg = cfg or PrivilegedIsaacCfg()
    refuse_combined_robot(cfg)
    spec = load_hand_spec()
    uncal = spec.raw.get("friction_vs_cardboard_static") == REQUIRED_INPUT_TOKEN
    report = {
        **cfg.to_dict(),
        "uncalibrated": uncal,
        "grasp_success_rate": None,
        "combined_robot": {"blocked": True, "reason": "PolicyEvalBlocked until flange SE(3)"},
        "isaaclab_loaded": False,
        "status": "blocked_uncalibrated" if uncal else "cfg_only_no_isaaclab",
    }
    refuse_grasp_success_key(report)
    return report
