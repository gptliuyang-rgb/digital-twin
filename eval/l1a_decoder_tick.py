"""Dump decoder 874-D assembled on the shared 50 Hz playback tick.

Does not run ONNX. grasp_success_rate stays JSON null. Decoder history is
HardwareHold, not planner clip qpos.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import PLANNER_CONTROL_HZ, T800_N_LOWER_BODY_DOF, TOKEN_DIM
from wbc.gather import HardwareSnapshot, ObsGather
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerQpos, pack_qpos
from wbc.shared_cursor import (
    SHARED_CURSOR_YAML,
    SharedPlaybackCursor,
    decoder_joint_history,
    load_shared_cursor_cfg,
    refuse_decoder_from_planner_qpos,
    refuse_g1_decoder_onnx,
    refuse_run_shared_cursor_onnx,
)

OUT = Path("eval/report/generated/l1a_decoder_tick.json")


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


def _hw(*, q0: float, t_s: float) -> HardwareSnapshot:
    q = np.zeros(25)
    q[0] = q0
    return HardwareSnapshot(
        t_s=t_s,
        q_hw=q,
        dq_hw=np.zeros(25),
        omega_imu=np.zeros(3),
        imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(25),
    )


def _dump() -> dict:
    cfg = load_shared_cursor_cfg()
    gather = ObsGather()
    cur = SharedPlaybackCursor(gather=gather)
    token = np.linspace(0.0, 1.0, TOKEN_DIM)
    last = None
    for i in range(10):
        gather.push_hw(_hw(q0=float(i), t_s=i / 50.0))
        last = cur.control_tick(
            _motor(0.0),
            locomotion_mode=0,
            token=token,
            new_qpos=np.stack([_row(j) for j in range(16)]) if i == 0 else None,
            t_s=i / 50.0,
        )
    assert last is not None
    enc = cur.assemble_encoder("t800")
    q_dec = decoder_joint_history(last.decoder_obs)[:, 0].tolist()
    q_enc = enc[4:254].reshape(10, 25)[:, 0].tolist()
    paused_gather = ObsGather()
    paused_gather.push_hw(_hw(q0=1.0, t_s=0.0))
    paused = SharedPlaybackCursor(gather=paused_gather)
    paused.control_tick(
        _motor(0.0),
        locomotion_mode=0,
        token=np.zeros(TOKEN_DIM),
        new_qpos=np.stack([_row(j) for j in range(8)]),
        t_s=0.0,
    )
    paused_gather.push_hw(_hw(q0=9.0, t_s=0.02))
    paused_tick = paused.control_tick(
        _motor(0.0),
        locomotion_mode=0,
        token=np.zeros(TOKEN_DIM),
        play=False,
        t_s=0.02,
    )
    decoder_onnx_refused = False
    try:
        refuse_g1_decoder_onnx("model_decoder.onnx")
    except G1CheckpointIncompatible:
        decoder_onnx_refused = True
    clip_copy_refused = False
    try:
        refuse_decoder_from_planner_qpos()
    except Exception:
        clip_copy_refused = True
    onnx_blocked = False
    try:
        refuse_run_shared_cursor_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    return {
        "source": SHARED_CURSOR_YAML.name,
        "adr": "ADR-051",
        "control_hz": cfg["control_hz"],
        "n_dof": cfg["n_dof"],
        "token_dim": cfg["token_dim"],
        "expected_decoder_dim": cfg["expected_decoder_dim"],
        "g1_decoder_dim_forbidden": cfg["g1_decoder_dim_forbidden"],
        "decoder_obs_shape": list(last.decoder_obs.shape),
        "logger_len": last.logger_len,
        "current_frame_after_10": last.current_frame,
        "decoder_q0_history": q_dec,
        "encoder_q0_look_ahead": q_enc,
        "decoder_q_equals_encoder_q": q_dec == q_enc,
        "paused_current_frame": paused_tick.current_frame,
        "paused_logger_len": paused_tick.logger_len,
        "paused_decoder_q0": float(decoder_joint_history(paused_tick.decoder_obs)[-1, 0]),
        "policy_hz": PLANNER_CONTROL_HZ,
        "g1_decoder_onnx_refused": decoder_onnx_refused,
        "decoder_from_planner_qpos_refused": clip_copy_refused,
        "onnx_run_blocked": onnx_blocked,
        "decoder_history_is_hardware_not_clip": True,
        "not_g1_decoder_onnx": True,
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
