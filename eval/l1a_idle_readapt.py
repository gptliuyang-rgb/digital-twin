"""Dump the official idle ADAPTING/RECOVERING table. Does not run ONNX.

grasp_success_rate stays JSON null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    IDLE_READAPT_ADAPT_STOP_RAD,
    IDLE_READAPT_ADAPT_TRIGGER_RAD,
    IDLE_READAPT_RECOVER_TRIGGER_RAD,
    T800_N_LOWER_BODY_DOF,
)
from wbc.idle_readapt import (
    IDLE_READAPT_YAML,
    IdleReadapt,
    IdleReadaptState,
    load_idle_readapt_cfg,
    refuse_run_idle_readapt_onnx,
    transition,
)
from wbc.planner_onnx import PlannerOnnxBlocked

OUT = Path("eval/report/generated/l1a_idle_readapt.json")


def _q(lower: float, upper: float = 0.4) -> np.ndarray:
    q = np.zeros(25)
    q[:T800_N_LOWER_BODY_DOF] = lower
    q[T800_N_LOWER_BODY_DOF:] = upper
    return q


def _dump() -> dict:
    cfg = load_idle_readapt_cfg()
    machine = IdleReadapt()
    adapt = machine.tick(_q(0.0), _q(0.20), locomotion_mode=0, at_last_frame=True)
    recover_idle = IdleReadapt()
    recover = recover_idle.tick(_q(0.0), _q(0.0), locomotion_mode=0, at_last_frame=True)
    walk = IdleReadapt().tick(_q(0.2), _q(0.2), locomotion_mode=2, at_last_frame=True)
    mid = IdleReadapt().tick(_q(0.2), _q(0.2), locomotion_mode=0, at_last_frame=False)
    g1_refused = False
    onnx_blocked = False
    try:
        refuse_run_idle_readapt_onnx("planner_sonic.onnx")
    except G1CheckpointIncompatible:
        g1_refused = True
    try:
        refuse_run_idle_readapt_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    return {
        "source": IDLE_READAPT_YAML.name,
        "adr": "ADR-048",
        "control_hz": cfg["control_hz"],
        "n_dof": cfg["n_dof"],
        "n_lower_body_dof": cfg["n_lower_body_dof"],
        "idle_mode": cfg["idle_mode"],
        "k_adapt_trigger_rad": IDLE_READAPT_ADAPT_TRIGGER_RAD,
        "k_adapt_stop_rad": IDLE_READAPT_ADAPT_STOP_RAD,
        "k_recover_trigger_rad": IDLE_READAPT_RECOVER_TRIGGER_RAD,
        "k_recover_stop_rad": None,
        "blend_keep": cfg["blend_keep"],
        "blend_to": cfg["blend_to"],
        "idle_at_0.10": int(transition(IdleReadaptState.IDLE, 0.10)),
        "idle_above_0.10": int(transition(IdleReadaptState.IDLE, 0.10 + 1e-12)),
        "recovering_at_zero_stays": int(transition(IdleReadaptState.RECOVERING, 0.0)),
        "first_tick_adapt_state": int(adapt.state),
        "first_tick_adapt_applied": adapt.applied,
        "first_tick_recover_state": int(recover.state),
        "walk_skipped": walk.skipped_reason,
        "mid_clip_skipped": mid.skipped_reason,
        "not_g1_planner_onnx": True,
        "g1_planner_onnx_refused": g1_refused,
        "onnx_run_blocked": onnx_blocked,
        "not_planner_blend": True,
        "not_recover_stop": True,
        "not_interpolator_substitute": True,
        "not_invented_clip": True,
        "not_pico_sdk": True,
        "not_dexhand2_contact": True,
        "not_table_s4": True,
        "not_hand_mit_ring": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "repo": str(REPO_ROOT),
    }


def main() -> None:
    payload = _dump()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
