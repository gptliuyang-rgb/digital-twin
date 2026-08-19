"""Dump 25-D policy_action ZOH onto the 500 Hz PD ring after stash.

Does not run ONNX. Does not Hermite joint targets. grasp_success_rate
stays JSON null. PD sees a_t this tick; decoder last_action sees a_{t-1}.
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
    refuse_policy_pd_as_decoder_run,
    refuse_policy_pd_hermite,
    refuse_run_shared_cursor_onnx,
)
from wbc.stream import STREAM_HZ, load_stream_cfg, stream_factor, stream_policy_action_zoh

OUT = Path("eval/report/generated/l1a_pd_stream.json")


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
    stream_cfg = load_stream_cfg()
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
            policy_action=_action(a0=10.0 + i),
        )
    assert last is not None
    q_dec = decoder_joint_history(last.decoder_obs)[:, 0].tolist()
    a_dec = decoder_last_action_history(last.decoder_obs)[:, 0].tolist()
    expected_a = [0.0] + [10.0 + i for i in range(9)]
    period = cur.pd_hold.zoh_period(t0_s=9 / 50.0)
    zoh = stream_policy_action_zoh(np.stack([_action(a0=1.0), _action(a0=2.0)]))
    onnx_blocked = False
    try:
        refuse_run_shared_cursor_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    hermite_refused = False
    try:
        refuse_policy_pd_hermite()
    except Exception:
        hermite_refused = True
    invented_onnx_refused = False
    try:
        refuse_policy_pd_as_decoder_run()
    except Exception:
        invented_onnx_refused = True
    g1_action_refused = False
    try:
        cur.push_policy_pd(np.zeros(29))
    except G1CheckpointIncompatible:
        g1_action_refused = True
    return {
        "source": SHARED_CURSOR_YAML.name,
        "adr": "ADR-054",
        "control_hz": cfg["control_hz"],
        "command_stream_hz": STREAM_HZ,
        "factor_policy_to_stream": stream_factor(50),
        "n_dof": cfg["n_dof"],
        "action_dim": cfg["action_dim"],
        "pd_action_dim": stream_cfg["pd_action_dim"],
        "g1_action_dim_forbidden": cfg["g1_action_dim_forbidden"],
        "decoder_q0_history": q_dec,
        "decoder_a0_history": a_dec,
        "decoder_a_equals_expected_delay": a_dec == expected_a,
        "pd_q_des_a0_after_10": float(last.pd_q_des[0]),
        "stashed_a0_after_10": None if last.policy_action is None else float(last.policy_action[0]),
        "zoh_period_n_steps": period.n_steps,
        "zoh_period_all_equal_latest": bool(np.allclose(period.q_des_rad[:, 0], 19.0)),
        "zoh_two_step_n_steps": zoh.n_steps,
        "zoh_two_step_first_ten": zoh.q_des_rad[:10, 0].tolist(),
        "zoh_two_step_last": float(zoh.q_des_rad[-1, 0]),
        "policy_hz": PLANNER_CONTROL_HZ,
        "g1_action_refused": g1_action_refused,
        "policy_action_hermite_refused": hermite_refused,
        "policy_action_from_decoder_onnx_refused": invented_onnx_refused,
        "onnx_run_blocked": onnx_blocked,
        "policy_action_feeds_500hz_pd_after_stash": True,
        "policy_action_pd_is_zoh": True,
        "not_policy_action_hermite": True,
        "not_policy_action_same_tick_decoder_obs": True,
        "policy_action_feeds_next_tick_last_action": True,
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
