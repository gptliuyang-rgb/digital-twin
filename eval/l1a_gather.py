"""Dump SONIC §S7 observation-gather layout. grasp_success_rate stays JSON null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.dims import TOKEN_DIM
from wbc.gather import HardwareSnapshot, ObsGather, compile_observations, load_gather_cfg
from wbc.stream import STREAM_HZ

OUT = Path("eval/report/generated/l1a_gather.json")


def main() -> None:
    cfg = load_gather_cfg()
    slots, total = compile_observations(cfg)
    gather = ObsGather()
    token = np.zeros(TOKEN_DIM)
    n = int(cfg["n_dof"])
    for i in range(int(cfg["history_frames"])):
        q = np.zeros(n)
        q[0] = float(i)
        snap = HardwareSnapshot(
            t_s=i / float(cfg["policy_hz"]),
            q_hw=q,
            dq_hw=np.zeros(n),
            omega_imu=np.zeros(3),
            imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            last_action=np.zeros(n),
        )
        gather.push_hw(snap)
        obs = gather.control_tick(token, t_s=snap.t_s)
    payload = {
        "source": "He et al., SONIC, arXiv:2511.07820v3 §S7",
        "adr": "ADR-041",
        "policy_hz": cfg["policy_hz"],
        "command_stream_hz": STREAM_HZ,
        "operator_input_hz": cfg["operator_input_hz"],
        "planner_hz": cfg["planner_hz"],
        "n_dof": n,
        "total_dim": total,
        "expected_total_dim": cfg["expected_total_dim"],
        "g1_total_dim_forbidden": cfg["g1_total_dim_forbidden"],
        "sensor_delay_ticks": cfg["sensor_delay_ticks"],
        "slots": [{"name": s.name, "offset": s.offset, "dim": s.dim} for s in slots],
        "history_q0": obs[64 + 30 : 64 + 30 + 250].reshape(10, 25)[:, 0].tolist(),
        "grasp_success_rate": None,
        "not_dexhand2_contact": True,
        "not_table_s4": True,
        "not_hand_mit_ring": True,
        "not_invented_latency": True,
        "not_pico_sdk": True,
        "combined_robot": "PolicyEvalBlocked",
        "repo": str(REPO_ROOT),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
