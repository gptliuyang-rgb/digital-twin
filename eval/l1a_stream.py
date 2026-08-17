"""Dump SONIC §3.5 500 Hz stream factors. grasp_success_rate stays JSON null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT, CommandVector
from wbc.planner import KinematicPlanner
from wbc.spring import RootSpringState
from wbc.stream import (
    STREAM_HZ,
    load_stream_cfg,
    stream_factor,
    stream_nav_spring,
    stream_planned_ref,
    stream_policy_tokens,
)

OUT = Path("eval/report/generated/l1a_stream.json")


def main() -> None:
    cfg = load_stream_cfg()
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.2, 0.0, 0.0]
    planner = KinematicPlanner(horizon_s=1.6, pos_interp="linear")
    ref = planner.plan([a, b])
    streamed = stream_planned_ref(ref, pos_interp="linear")
    tokens = np.stack([a.to_flat_vector(), b.to_flat_vector()])
    policy = stream_policy_tokens(tokens, pos_interp="linear")
    nav = stream_nav_spring(
        RootSpringState(
            pos_xy_m=np.zeros(2),
            heading_rad=0.0,
            vel_xy_mps=np.array([6.0, 0.0]),
            yaw_rate_rad_s=0.0,
        ),
        np.array([-6.0, 0.0, 0.0]),
        horizon_s=1.6,
    )
    payload = {
        "source": "He et al., SONIC, arXiv:2511.07820v3 §3.5",
        "adr": "ADR-039",
        "command_stream_hz": STREAM_HZ,
        "factor_planner_to_stream": stream_factor(10),
        "factor_policy_to_stream": stream_factor(50),
        "planner_n_steps": ref.n_steps,
        "stream_n_steps": streamed.n_steps,
        "policy_stream_n_steps": policy.n_steps,
        "nav_stream_hz": nav.rate_hz,
        "nav_stream_n_steps": nav.n_steps,
        "nav_eval": cfg["nav_eval"],
        "operator_input_hz_recorded": cfg["operator_input_hz"],
        "grasp_success_rate": None,
        "not_dexhand2_contact": True,
        "not_table_s4": True,
        "not_hand_mit_ring": True,
        "combined_robot": "PolicyEvalBlocked",
        "repo": str(REPO_ROOT),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
