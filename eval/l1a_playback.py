"""Dump the official 50 Hz planner playback cursor table. Does not run ONNX.

grasp_success_rate stays JSON null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import PLANNER_CONTROL_HZ, T800_N_LOWER_BODY_DOF
from wbc.idle_readapt import IdleReadaptState
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerQpos, pack_qpos
from wbc.playback import (
    PLAYBACK_YAML,
    PlannerPlayback,
    clamp_frame,
    load_playback_cfg,
    refuse_run_playback_onnx,
)

OUT = Path("eval/report/generated/l1a_playback.json")


def _row(i: int) -> np.ndarray:
    q = np.zeros(25)
    q[0] = 0.01 * i
    return pack_qpos(
        PlannerQpos(
            root_pos_m=np.array([0.01 * i, 0.0, 1.03]),
            root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            q_rad=q,
        )
    )


def _motor(lower: float) -> np.ndarray:
    q = np.zeros(25)
    q[:T800_N_LOWER_BODY_DOF] = lower
    return q


def _dump() -> dict:
    cfg = load_playback_cfg()
    cur = PlannerPlayback()
    first = cur.tick(_motor(0.0), locomotion_mode=0, new_qpos=np.stack([_row(i) for i in range(16)]))
    for _ in range(14):
        cur.tick(_motor(0.0), locomotion_mode=0)
    last = cur.tick(_motor(0.20), locomotion_mode=0)
    walk = PlannerPlayback()
    walk.tick(_motor(0.0), locomotion_mode=2, new_qpos=np.stack([_row(i) for i in range(2)]))
    walk.tick(_motor(0.0), locomotion_mode=2)
    walk_hold = walk.tick(_motor(0.20), locomotion_mode=2)
    paused = PlannerPlayback()
    paused.tick(_motor(0.0), locomotion_mode=0, new_qpos=np.stack([_row(i) for i in range(8)]))
    paused_tick = paused.tick(_motor(0.20), locomotion_mode=0, play=False)
    g1_refused = False
    onnx_blocked = False
    try:
        refuse_run_playback_onnx("planner_sonic.onnx")
    except G1CheckpointIncompatible:
        g1_refused = True
    try:
        refuse_run_playback_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    return {
        "source": PLAYBACK_YAML.name,
        "adr": "ADR-049",
        "control_hz": cfg["control_hz"],
        "n_dof": cfg["n_dof"],
        "expected_qpos_dim": cfg["expected_qpos_dim"],
        "idle_mode": cfg["idle_mode"],
        "blend_num_frames": cfg["blend_num_frames"],
        "clamp_from_0_n16": list(clamp_frame(0, 16)),
        "clamp_from_last_n16": list(clamp_frame(15, 16)),
        "first_tick_assigned": first.assigned,
        "first_tick_current_frame": first.current_frame,
        "last_tick_current_frame": last.current_frame,
        "last_tick_clamped": last.clamped,
        "last_tick_idle_state": None if last.idle is None else int(last.idle.state),
        "last_tick_idle_is_adapting": last.idle is not None
        and last.idle.state is IdleReadaptState.ADAPTING,
        "walk_hold_skipped": walk_hold.skipped_reason,
        "paused_skipped": paused_tick.skipped_reason,
        "paused_current_frame": paused_tick.current_frame,
        "policy_hz": PLANNER_CONTROL_HZ,
        "not_g1_planner_onnx": True,
        "g1_planner_onnx_refused": g1_refused,
        "onnx_run_blocked": onnx_blocked,
        "not_reference_motion_loop": True,
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
