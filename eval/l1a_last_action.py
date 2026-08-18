"""Dump caller-supplied 25-D last_action on the shared 50 Hz decoder tick.

Does not run ONNX. Does not invent a decoder output. grasp_success_rate
stays JSON null. last_action is HardwareHold, not planner clip qpos.
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
    decoder_last_action_history,
    load_shared_cursor_cfg,
    refuse_last_action_from_decoder_onnx,
    refuse_last_action_from_planner_qpos,
    refuse_run_shared_cursor_onnx,
)

OUT = Path("eval/report/generated/l1a_last_action.json")


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


def _action(*, a0: float) -> np.ndarray:
    a = np.zeros(25)
    a[0] = a0
    return a


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
            last_action=_action(a0=10.0 + i),
        )
    assert last is not None
    enc = cur.assemble_encoder("t800")
    q_dec = decoder_joint_history(last.decoder_obs)[:, 0].tolist()
    a_dec = decoder_last_action_history(last.decoder_obs)[:, 0].tolist()
    q_enc = enc[4:254].reshape(10, 25)[:, 0].tolist()
    omitted = ObsGather()
    omitted.push_hw(_hw(q0=1.0, t_s=0.0))
    omitted_cur = SharedPlaybackCursor(gather=omitted)
    omitted_tick = omitted_cur.control_tick(
        _motor(0.0),
        locomotion_mode=0,
        token=np.zeros(TOKEN_DIM),
        new_qpos=np.stack([_row(j) for j in range(8)]),
        t_s=0.0,
    )
    onnx_blocked = False
    try:
        refuse_run_shared_cursor_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    planner_copy_refused = False
    try:
        refuse_last_action_from_planner_qpos()
    except Exception:
        planner_copy_refused = True
    invented_onnx_refused = False
    try:
        refuse_last_action_from_decoder_onnx()
    except Exception:
        invented_onnx_refused = True
    g1_action_refused = False
    try:
        cur.apply_last_action(np.zeros(29))
    except G1CheckpointIncompatible:
        g1_action_refused = True
    return {
        "source": SHARED_CURSOR_YAML.name,
        "adr": "ADR-052",
        "control_hz": cfg["control_hz"],
        "n_dof": cfg["n_dof"],
        "action_dim": cfg["action_dim"],
        "g1_action_dim_forbidden": cfg["g1_action_dim_forbidden"],
        "token_dim": cfg["token_dim"],
        "expected_decoder_dim": cfg["expected_decoder_dim"],
        "decoder_obs_shape": list(last.decoder_obs.shape),
        "logger_len": last.logger_len,
        "current_frame_after_10": last.current_frame,
        "decoder_q0_history": q_dec,
        "decoder_a0_history": a_dec,
        "encoder_q0_look_ahead": q_enc,
        "decoder_a_equals_decoder_q": a_dec == q_dec,
        "decoder_a_equals_encoder_q": a_dec == q_enc,
        "omitted_last_action_is_zeros": bool(np.allclose(omitted_tick.last_action, 0.0)),
        "policy_hz": PLANNER_CONTROL_HZ,
        "g1_action_refused": g1_action_refused,
        "last_action_from_planner_qpos_refused": planner_copy_refused,
        "last_action_from_decoder_onnx_refused": invented_onnx_refused,
        "onnx_run_blocked": onnx_blocked,
        "last_action_is_caller_supplied_policy": True,
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
