"""ONNX export *contract* for a T800 SONIC tracker. Does not export weights.

Official GEAR-SONIC ONNX is G1 (decoder 994). A T800 export must be 874-D.
This module refuses G1 blobs and refuses to write files without a trained ckpt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import G1_DECODER_INPUT_DIM, decoder_history_dim, load_t800_sonic
from wbc.retarget import retarget_blockers


class OnnxExportBlocked(RuntimeError):
    """No trained T800 tracker is in this repo yet."""


def expected_io() -> dict[str, Any]:
    cfg = load_t800_sonic()
    n = int(cfg["n_revolute"])
    return {
        "robot": cfg["robot"],
        "n_dof": n,
        "encoder": {
            "model_encoder.onnx": {
                "note": "universal token encoder; cmd dim depends on teleop_mode",
                "vr_3point_cmd_dim": int(cfg["hybrid_encoder_cmd_dim_3point"]),
                "vr_5point_cmd_dim": int(cfg["hybrid_encoder_cmd_dim_5point"]),
            }
        },
        "decoder": {
            "model_decoder.onnx": {
                "input_dim": decoder_history_dim(n),
                "g1_input_dim_forbidden": G1_DECODER_INPUT_DIM,
                "output_dim": n,
            }
        },
        "observation_config.yaml": "must list T800 joint_order (25), not G1 (29)",
    }


def refuse_export(
    *,
    checkpoint_path: Path | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
) -> None:
    refuse_g1_checkpoint(checkpoint_meta=checkpoint_meta or {"robot": "t800", "n_dof": 25})
    meta = checkpoint_meta or {}
    if int(meta.get("decoder_input_dim", decoder_history_dim(25))) == G1_DECODER_INPUT_DIM:
        raise G1CheckpointIncompatible("refusing to wrap a G1 decoder ONNX as T800")
    blockers = retarget_blockers()
    if blockers:
        raise OnnxExportBlocked(
            "No T800 SONIC checkpoint to export. PPO is blocked on "
            f"{blockers}. Do not copy nvidia/GEAR-SONIC G1 ONNX."
        )
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        raise OnnxExportBlocked(
            "T800 last.pt / ONNX is not in this repo. Train after SPEC_INTAKE P0, then export."
        )
