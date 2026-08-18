"""Dump SONIC §3.5 100 Hz operator-loop factors. grasp_success_rate stays JSON null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT, CommandVector
from wbc.operator import (
    OPERATOR_INPUT_HZ,
    downsample_stride,
    ingest_operator,
    load_operator_cfg,
    operator_to_hybrid_tokens,
    operator_to_planner,
    operator_to_policy_tokens,
    operator_to_stream,
)
from wbc.stream import STREAM_HZ, stream_factor

OUT = Path("eval/report/generated/l1a_operator.json")


def main() -> None:
    cfg = load_operator_cfg()
    a = CommandVector.zeros()
    b = CommandVector.zeros()
    b.left_wrist_pos[:] = [0.2, 0.0, 0.0]
    n = int(round(1.6 * OPERATOR_INPUT_HZ)) + 1
    rows = np.stack([a.to_flat_vector(), b.to_flat_vector()])
    t_src = np.array([0.0, 1.6])
    # Dense 100 Hz window by linear blend of the two endpoints (eval dump only).
    t = np.linspace(0.0, 1.6, n)
    alpha = (t / 1.6).reshape(-1, 1)
    commands = (1.0 - alpha) * rows[0] + alpha * rows[1]
    window = ingest_operator(commands, source="vr_3point", t_s=t)
    streamed = operator_to_stream(window, pos_interp="linear")
    policy = operator_to_policy_tokens(window)
    planned = operator_to_planner(window)
    hybrid = operator_to_hybrid_tokens(window)
    payload = {
        "source": "He et al., SONIC, arXiv:2511.07820v3 §3.5",
        "adr": "ADR-040",
        "operator_input_hz": OPERATOR_INPUT_HZ,
        "command_stream_hz": STREAM_HZ,
        "factor_operator_to_stream": stream_factor(OPERATOR_INPUT_HZ),
        "factor_operator_to_policy": downsample_stride(OPERATOR_INPUT_HZ, 50),
        "factor_operator_to_planner": downsample_stride(OPERATOR_INPUT_HZ, 10),
        "operator_n_steps": window.n_steps,
        "stream_n_steps": streamed.n_steps,
        "policy_n_steps": int(policy.shape[0]),
        "planner_n_steps": planned.n_steps,
        "hybrid_token_dim": int(hybrid.shape[1]),
        "nav_resample": cfg["nav_resample"],
        "sources_allowed": list(cfg["sources"]["allowed"]),
        "grasp_success_rate": None,
        "not_dexhand2_contact": True,
        "not_table_s4": True,
        "not_hand_mit_ring": True,
        "not_pico_sdk": True,
        "combined_robot": "PolicyEvalBlocked",
        "t_src_endpoints": t_src.tolist(),
        "repo": str(REPO_ROOT),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
