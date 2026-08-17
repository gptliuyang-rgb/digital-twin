"""GMR / BONES-SEED retarget checklist. Does not download datasets or run GPU training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from assets.combined.assemble import PolicyEvalBlocked, assert_policy_eval_allowed, mount_ready
from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import assert_t800_config


def retarget_blockers() -> list[str]:
    """Human/hardware gates that must clear before a T800 SONIC train job is launched."""
    blockers: list[str] = []
    if not mount_ready():
        blockers.append("t800_wrist_to_hand_mount SE(3) is REQUIRED_INPUT")
    spec = load_hand_spec()
    if spec.raw.get("com_in_wrist_frame_m") == REQUIRED_INPUT_TOKEN:
        blockers.append("com_in_wrist_frame_m is REQUIRED_INPUT (ADR-010)")
    return blockers


def assert_retarget_ready(*, checkpoint_meta: dict[str, Any] | None = None) -> None:
    refuse_g1_checkpoint(checkpoint_meta=checkpoint_meta or {"robot": "t800", "n_dof": 25})
    assert_t800_config()
    if not mount_ready():
        raise PolicyEvalBlocked(
            "SONIC T800 retarget/train is blocked until the T800 flange SE(3) is CAD-measured. "
            "G1 fine-tune is forbidden."
        )
    assert_policy_eval_allowed("sonic_retarget")
    spec = load_hand_spec()
    if spec.raw.get("com_in_wrist_frame_m") == REQUIRED_INPUT_TOKEN:
        raise PolicyEvalBlocked(
            "SONIC load-aware training is blocked until com_in_wrist_frame_m is measured (ADR-010)."
        )


def pipeline_steps() -> list[dict[str, str]]:
    return [
        {"id": "skeleton", "detail": "Map T800 URDF into a SONIC robot config (wbc/t800_sonic.yaml)."},
        {
            "id": "gmr_export",
            "detail": "Emit GMR IK JSON from wbc/gmr/body_map.yaml (make gmr-export). Quat offsets uncalibrated.",
        },
        {"id": "gmr", "detail": "GMR-retarget BONES-SEED mocap onto the T800 skeleton; write motion_lib.pkl."},
        {"id": "filter", "detail": "Drop joint-limit, foot-penetration, and dynamically infeasible clips."},
        {"id": "payload", "detail": "Domain-randomize wrist payload 0–20 kg (sim/payload.py) during PPO."},
        {"id": "ppo", "detail": "Train in Isaac Lab with paper Table 1/2 rewards. Do not start from G1 last.pt."},
        {"id": "sim2sim", "detail": "Evaluate MPJPE on held-out clips in MuJoCo."},
        {"id": "export", "detail": "Export encoder+decoder ONNX with T800 observation_config (25 DoF)."},
    ]


def status_report() -> dict[str, Any]:
    cfg = assert_t800_config()
    return {
        "robot": cfg["robot"],
        "n_revolute": cfg["n_revolute"],
        "decoder_input_dim": cfg["decoder_input_dim"],
        "g1_decoder_input_dim": cfg["g1_decoder_input_dim"],
        "g1_finetune": "forbidden",
        "blockers": retarget_blockers(),
        "steps": pipeline_steps(),
        "config": str(Path(__file__).with_name("t800_sonic.yaml")),
        "gmr_body_map": str(Path(__file__).with_name("gmr") / "body_map.yaml"),
        "gmr_ik_smplx": str(Path(__file__).with_name("gmr") / "smplx_to_t800.json"),
    }
