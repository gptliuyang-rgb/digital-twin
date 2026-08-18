"""Dump the official 8-frame planner blend + replan table. Does not run ONNX.

grasp_success_rate stays JSON null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT, CommandVector
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import PLANNER_BLEND_FRAMES, PLANNER_CRAWLING_MODE, PLANNER_RUNNING_MODE
from wbc.planner_blend import (
    PLANNER_BLEND_YAML,
    PlannerReplanClock,
    blend_weight,
    cross_fade_qpos,
    load_planner_blend_cfg,
    movement_from_command,
    refuse_run_blend_onnx,
    replan_interval_s,
)
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerQpos, pack_qpos

OUT = Path("eval/report/generated/l1a_planner_blend.json")


def _row(i: int) -> np.ndarray:
    q = np.zeros(25)
    q[0] = float(i)
    return pack_qpos(
        PlannerQpos(
            root_pos_m=np.array([0.01 * i, 0.0, 1.03]),
            root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            q_rad=q,
        )
    )


def _dump() -> dict:
    cfg = load_planner_blend_cfg()
    cmd = CommandVector.zeros()
    cmd.loco_mode = 2
    cmd.nav_cmd[:] = [0.35, 0.0, 0.0]
    st = movement_from_command(cmd, yaw_world_rad=0.0)
    clock = PlannerReplanClock()
    first = clock.tick(st)
    weights = [blend_weight(i, 0) for i in range(PLANNER_BLEND_FRAMES + 1)]
    old = np.stack([_row(0) for _ in range(16)], axis=0)
    new = np.stack([_row(1) for _ in range(16)], axis=0)
    blended = cross_fade_qpos(old, new, current_frame=5, gen_frame=7)
    g1_refused = False
    onnx_blocked = False
    try:
        refuse_run_blend_onnx("planner_sonic.onnx")
    except G1CheckpointIncompatible:
        g1_refused = True
    try:
        refuse_run_blend_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    return {
        "source": PLANNER_BLEND_YAML.name,
        "adr": "ADR-047",
        "blend_num_frames": PLANNER_BLEND_FRAMES,
        "planner_dt_s": cfg["planner_dt_s"],
        "replan_interval_s": cfg["replan_interval_s"],
        "static_modes": cfg["static_modes"],
        "crawling_modes": cfg["crawling_modes"],
        "boxing_modes": cfg["boxing_modes"],
        "command_schema_fast_walk_mode": int(st.locomotion_mode),
        "command_schema_fast_walk_interval_s": replan_interval_s(st.locomotion_mode),
        "run_interval_s": replan_interval_s(PLANNER_RUNNING_MODE),
        "crawl_interval_s": replan_interval_s(PLANNER_CRAWLING_MODE),
        "elbow_crawl_interval_s": replan_interval_s(14),
        "first_tick_replans_from_idle": first.need_replan,
        "blend_weights_0_to_8": weights,
        "blend_start_frame_cur5_gen7": blended.blend_start_frame,
        "new_anim_length_cur5_gen7": blended.new_anim_length,
        "current_frame_after_blend": blended.current_frame,
        "not_g1_planner_onnx": True,
        "g1_planner_onnx_refused": g1_refused,
        "onnx_run_blocked": onnx_blocked,
        "not_interpolator_substitute": True,
        "not_idle_readapt": True,
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
