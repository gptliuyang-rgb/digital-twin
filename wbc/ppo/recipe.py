"""Load the frozen T800 SONIC PPO recipe and refuse to launch while blockers remain.

This module never imports Isaac Lab. `make ppo-train` is supposed to exit non-zero
until flange SE(3) and wrist CoM are measured (ADR-022).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import G1_N_DOF, load_t800_sonic
from wbc.retarget import assert_retarget_ready, retarget_blockers
from wbc.teleop import refuse_teleop_mode_mismatch

PPO_DIR = Path(__file__).resolve().parent


class PpoLaunchBlocked(RuntimeError):
    """Raised instead of starting Isaac Lab PPO while human P0 inputs are open."""


def _load_yaml(name: str) -> dict[str, Any]:
    raw = yaml.safe_load((PPO_DIR / name).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    return raw


def load_ppo_recipe(*, teleop_mode: str | None = None) -> dict[str, Any]:
    net = _load_yaml("network.yaml")
    hyp = _load_yaml("hyperparams.yaml")
    rew = _load_yaml("rewards.yaml")
    dr = _load_yaml("domain_rand.yaml")
    cfg = load_t800_sonic()
    mode = teleop_mode or str(net["default_teleop_mode"])
    five_point = mode.replace("-", "_").lower() in ("vr_5point", "5point")
    if five_point:
        overlay = _load_yaml("network_5point.yaml")
        if int(overlay["action_dim"]) != int(net["action_dim"]):
            raise ValueError("5-point overlay must keep action_dim=25 (T800 revolute)")
        if int(overlay["paper_g1_action_dim"]) != G1_N_DOF:
            raise ValueError("5-point overlay must keep paper_g1_action_dim=29 as forbidden")
        if overlay.get("g1_last_pt_finetune", "forbidden") != "forbidden":
            raise ValueError("G1 last.pt fine-tune must stay forbidden on the 5-point overlay")
        if overlay.get("elbow_bodies", {}).get("left_elbow") == "LINK_ELBOW_YAW_L":
            raise ValueError("5-point overlay elbow must be LINK_ELBOW_PITCH_*")
        net = {**net, **overlay}
        net["default_teleop_mode"] = "vr_5point"
    if int(net["action_dim"]) != int(cfg["n_revolute"]):
        raise ValueError(
            f"ppo network action_dim {net['action_dim']} != t800 n_revolute {cfg['n_revolute']}"
        )
    if int(net["paper_g1_action_dim"]) != G1_N_DOF:
        raise ValueError("paper_g1_action_dim drifted from 29")
    if int(net["hybrid_encoder_cmd_dim_3point"]) != int(cfg["hybrid_encoder_cmd_dim_3point"]):
        raise ValueError("3-point hybrid encoder dim drifted")
    if int(net["hybrid_encoder_cmd_dim_5point"]) != int(cfg["hybrid_encoder_cmd_dim_5point"]):
        raise ValueError("5-point hybrid encoder dim drifted")
    if dr.get("not_dexhand2_contact") is not True:
        raise ValueError("domain_rand.yaml must keep not_dexhand2_contact: true")
    if hyp.get("g1_last_pt_finetune") != "forbidden":
        raise ValueError("G1 last.pt fine-tune must stay forbidden")
    return {
        "network": net,
        "hyperparams": hyp,
        "rewards": rew,
        "domain_rand": dr,
        "robot": cfg["robot"],
        "n_revolute": int(cfg["n_revolute"]),
        "teleop_mode": str(net["default_teleop_mode"]),
        "five_point_overlay": five_point,
        "blockers": retarget_blockers(),
    }


def refuse_g1_action_dim(action_dim: int) -> None:
    if int(action_dim) == G1_N_DOF:
        raise G1CheckpointIncompatible(
            f"PPO action_dim={action_dim} is G1. T800 action_dim is "
            f"{load_t800_sonic()['n_revolute']}. Retrain from scratch."
        )


def refuse_ppo_launch(
    *,
    teleop_mode: str | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hard stop. Does not construct an Isaac Lab env."""
    recipe = load_ppo_recipe(teleop_mode=teleop_mode)
    refuse_g1_checkpoint(checkpoint_meta=checkpoint_meta or {"robot": "t800", "n_dof": recipe["n_revolute"]})
    refuse_g1_action_dim(int(recipe["network"]["action_dim"]))
    mode = teleop_mode or recipe["teleop_mode"]
    refuse_teleop_mode_mismatch(recipe["teleop_mode"], mode)
    blockers = list(recipe["blockers"])
    try:
        assert_retarget_ready(checkpoint_meta=checkpoint_meta or {"robot": "t800", "n_dof": 25})
    except Exception as exc:
        raise PpoLaunchBlocked(
            "SONIC T800 PPO is blocked. Fill docs/SPEC_INTAKE.md P0 "
            f"(flange SE(3), wrist CoM). blockers={blockers}. {exc}"
        ) from exc
    raise PpoLaunchBlocked(
        "Retarget blockers are empty but Isaac Lab PPO is not wired in this repo (ADR-022). "
        "Do not copy G1 last.pt."
    )


def status_report() -> dict[str, Any]:
    recipe = load_ppo_recipe()
    return {
        "robot": recipe["robot"],
        "action_dim": recipe["network"]["action_dim"],
        "paper_g1_action_dim": recipe["network"]["paper_g1_action_dim"],
        "teleop_mode": recipe["teleop_mode"],
        "five_point_overlay": recipe["five_point_overlay"],
        "g1_finetune": recipe["hyperparams"]["g1_last_pt_finetune"],
        "not_dexhand2_contact": recipe["domain_rand"]["not_dexhand2_contact"],
        "blockers": recipe["blockers"],
        "launch": "refused",
        "isaac_lab": "not_imported",
    }
