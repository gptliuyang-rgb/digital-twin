"""Dump the T800 SONIC planner ONNX contract. Does not run G1 planner_sonic.onnx.

grasp_success_rate stays JSON null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT, CommandVector
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import G1_PLANNER_QPOS_DIM, t800_planner_qpos_dim
from wbc.planner_onnx import (
    PLANNER_ONNX_YAML,
    PlannerQpos,
    command_to_planner_inputs,
    load_planner_onnx_cfg,
    pack_context,
    pack_qpos,
    refuse_g1_planner_onnx,
    resample_qpos_30_to_50,
)

OUT = Path("eval/report/generated/l1a_planner_onnx.json")


def _qpos(i: int) -> PlannerQpos:
    q = np.zeros(25)
    q[0] = float(i)
    return PlannerQpos(
        root_pos_m=np.array([0.01 * i, 0.0, 1.03]),
        root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        q_rad=q,
    )


def _dump() -> dict:
    cfg = load_planner_onnx_cfg()
    ctx = pack_context([_qpos(i) for i in range(4)])
    cmd = CommandVector.zeros()
    cmd.loco_mode = 2
    cmd.nav_cmd[:] = [0.35, 0.0, 0.0]
    packed = command_to_planner_inputs(cmd, ctx, yaw_world_rad=0.0)
    rows30 = np.stack([pack_qpos(_qpos(i)) for i in range(6)], axis=0)
    rows50 = resample_qpos_30_to_50(rows30)
    g1_refused = False
    try:
        refuse_g1_planner_onnx("planner_sonic.onnx")
    except G1CheckpointIncompatible as exc:
        g1_refused = "36" in str(exc)
    return {
        "source": PLANNER_ONNX_YAML.name,
        "adr": "ADR-046",
        "n_dof": cfg["n_dof"],
        "qpos_dim": t800_planner_qpos_dim(),
        "g1_qpos_dim_forbidden": G1_PLANNER_QPOS_DIM,
        "context_shape": list(ctx.shape),
        "n_inputs_v2": len(packed),
        "mode_from_fast_walk": int(packed["mode"][0]),
        "height_disabled": float(packed["height"][0]),
        "target_vel_mps": float(packed["target_vel"][0]),
        "resample_30_to_50": {"n_in": int(rows30.shape[0]), "n_out": int(rows50.shape[0])},
        "loco_mode_map": cfg["loco_mode_map"],
        "not_g1_planner_onnx": True,
        "g1_planner_onnx_refused": g1_refused,
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
