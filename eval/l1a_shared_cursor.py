"""Dump encoder look-ahead vs planner context sharing PlannerPlayback.current_frame.

Does not run ONNX. grasp_success_rate stays JSON null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import PLANNER_CONTROL_HZ, T800_N_LOWER_BODY_DOF
from wbc.gather import HardwareSnapshot, ObsGather
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerOnnxError, PlannerQpos, pack_qpos
from wbc.shared_cursor import (
    SHARED_CURSOR_YAML,
    SharedPlaybackCursor,
    load_shared_cursor_cfg,
    refuse_run_shared_cursor_onnx,
)

OUT = Path("eval/report/generated/l1a_shared_cursor.json")


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


def _hw() -> HardwareSnapshot:
    return HardwareSnapshot(
        t_s=0.0,
        q_hw=np.zeros(25),
        dq_hw=np.zeros(25),
        omega_imu=np.zeros(3),
        imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(25),
    )


def _dump() -> dict:
    cfg = load_shared_cursor_cfg()
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather)
    first = cur.tick(_motor(0.0), locomotion_mode=0, new_qpos=np.stack([_row(i) for i in range(16)]))
    enc = cur.assemble_encoder("t800")
    ctx = cur.sample_planner_context()
    first_encoder_idxs = cur.encoder_look_ahead_indices()
    first_encoder_cursor = gather.motion.cursor
    for _ in range(14):
        cur.tick(_motor(0.0), locomotion_mode=0)
    last = cur.tick(_motor(0.20), locomotion_mode=0)
    last_encoder_idxs = cur.encoder_look_ahead_indices()
    last_planner_raised = False
    try:
        cur.sample_planner_context()
    except PlannerOnnxError:
        last_planner_raised = True
    paused_gather = ObsGather()
    paused_gather.push_hw(_hw())
    paused = SharedPlaybackCursor(gather=paused_gather)
    paused.tick(_motor(0.0), locomotion_mode=0, new_qpos=np.stack([_row(i) for i in range(8)]))
    paused_tick = paused.tick(_motor(0.20), locomotion_mode=0, play=False)
    g1_refused = False
    onnx_blocked = False
    try:
        refuse_run_shared_cursor_onnx("planner_sonic.onnx")
    except G1CheckpointIncompatible:
        g1_refused = True
    try:
        refuse_run_shared_cursor_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    return {
        "source": SHARED_CURSOR_YAML.name,
        "adr": "ADR-051",
        "control_hz": cfg["control_hz"],
        "n_dof": cfg["n_dof"],
        "expected_qpos_dim": cfg["expected_qpos_dim"],
        "lookahead_steps_50hz": cfg["lookahead_steps_50hz"],
        "encoder_look_ahead_n_frames": cfg["encoder_look_ahead_n_frames"],
        "encoder_look_ahead_step": cfg["encoder_look_ahead_step"],
        "first_tick_current_frame": first.current_frame,
        "encoder_cursor_after_first_tick": first_encoder_cursor,
        "encoder_look_ahead_indices_at_frame_1": first_encoder_idxs,
        "planner_context_shape": list(ctx.shape),
        "planner_context_first_hinge": float(ctx[0, 0, 7]),
        "encoder_dim": int(enc.shape[0]),
        "last_tick_current_frame": last.current_frame,
        "last_tick_clamped": last.clamped,
        "last_encoder_look_ahead_indices": last_encoder_idxs,
        "last_planner_context_raised": last_planner_raised,
        "paused_current_frame": paused_tick.current_frame,
        "paused_encoder_cursor": paused_gather.motion.cursor,
        "policy_hz": PLANNER_CONTROL_HZ,
        "not_g1_planner_onnx": True,
        "g1_planner_onnx_refused": g1_refused,
        "onnx_run_blocked": onnx_blocked,
        "not_motion_cursor_mix": True,
        "not_invented_clip": True,
        "not_pico_sdk": True,
        "not_dexhand2_contact": True,
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
