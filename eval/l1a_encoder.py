"""Dump SONIC encoder layouts (default 842-D and low-latency 831-D). grasp_success_rate stays JSON null."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT, CommandVector
from wbc.dims import (
    G1_ENCODER_MOTION_DIM,
    G1_ENCODER_ONNX_DIM,
    G1_ENCODER_ONNX_DIM_LOW_LATENCY,
    TOKEN_DIM,
    encoder_motion_dim,
    t800_encoder_onnx_dim,
    t800_encoder_onnx_dim_low_latency,
)
from wbc.gather import (
    GATHER_LOW_LATENCY_YAML,
    GATHER_YAML,
    HardwareSnapshot,
    ObsGather,
    compile_encoder_observations,
    load_gather_cfg,
)
from wbc.motion_ref import MotionFrame, identity_rot6d, look_ahead_indices

OUT_DEFAULT = Path("eval/report/generated/l1a_encoder.json")
OUT_LOW_LATENCY = Path("eval/report/generated/l1a_encoder_low_latency.json")


def _window(n: int = 50) -> list[MotionFrame]:
    frames = []
    for i in range(n):
        q = np.zeros(25)
        q[0] = float(i)
        frames.append(
            MotionFrame(
                q_ref_rad=q,
                dq_ref_rad_s=np.zeros(25),
                root_pos_m=np.array([0.0, 0.0, 1.03]),
                root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
    return frames


def _dump(variant: str) -> dict:
    yaml_path = GATHER_YAML if variant == "default" else GATHER_LOW_LATENCY_YAML
    cfg = load_gather_cfg(yaml_path)
    slots, total = compile_encoder_observations(cfg)
    gather = ObsGather(cfg=cfg)
    n = int(cfg["n_dof"])
    snap = HardwareSnapshot(
        t_s=0.0,
        q_hw=np.zeros(n),
        dq_hw=np.zeros(n),
        omega_imu=np.zeros(3),
        imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(n),
    )
    gather.push_hw(snap)
    gather.push_motion(_window(50), cursor=0)
    cmd = CommandVector.zeros()
    cmd.left_wrist_pos = np.array([0.1, 0.0, 0.0])
    gather.push_vr_3point(cmd)
    enc_t800 = gather.assemble_encoder("t800")
    enc_teleop = gather.assemble_encoder("teleop")
    token = np.zeros(TOKEN_DIM)
    dec = gather.control_tick(token, t_s=0.0)
    modes = {m["name"]: m for m in cfg["encoder"]["encoder_modes"]}
    step = 5 if variant == "default" else 1
    include_z = variant == "default"
    t800_q = enc_t800[4 : 4 + 250].reshape(10, 25)[:, 0]
    if variant == "default":
        teleop_lower = enc_teleop[581:701].reshape(10, 12)[:, 0]
        teleop_vr = enc_teleop[821:830]
        g1_onnx = G1_ENCODER_ONNX_DIM
        formula = t800_encoder_onnx_dim(n_dof=n)
        adr = "ADR-043"
    else:
        teleop_lower = enc_teleop[570:690].reshape(10, 12)[:, 0]
        teleop_vr = enc_teleop[810:819]
        g1_onnx = G1_ENCODER_ONNX_DIM_LOW_LATENCY
        formula = t800_encoder_onnx_dim_low_latency(n_dof=n)
        adr = "ADR-044"
    return {
        "source": str(yaml_path.name),
        "adr": adr,
        "encoder_variant": variant,
        "n_dof": n,
        "encoder_dim": total,
        "expected_encoder_dim": cfg["expected_encoder_dim"],
        "encoder_motion_window_dim": cfg["encoder_motion_window_dim"],
        "g1_encoder_dim_forbidden": G1_ENCODER_MOTION_DIM,
        "g1_encoder_onnx_dim_forbidden": g1_onnx,
        "formula_motion_window_dim": encoder_motion_dim(n, include_root_z=include_z),
        "formula_onnx_dim": formula,
        "decoder_dim": int(dec.shape[0]),
        "look_ahead_indices": look_ahead_indices(0, 10, step, 50),
        "identity_rot6d": identity_rot6d().tolist(),
        "slots": [{"name": s.name, "offset": s.offset, "dim": s.dim} for s in slots],
        "t800_mode": {
            "mode_id": modes["t800"]["mode_id"],
            "required_observations": modes["t800"]["required_observations"],
            "encoder_mode_4": enc_t800[:4].tolist(),
            "q0_window": t800_q.tolist(),
        },
        "teleop_mode": {
            "mode_id": modes["teleop"]["mode_id"],
            "required_observations": modes["teleop"]["required_observations"],
            "encoder_mode_4": enc_teleop[:4].tolist(),
            "lowerbody_q0_window": teleop_lower.tolist(),
            "vr_3point_pos": teleop_vr.tolist(),
        },
        "not_invented_clip": True,
        "not_g1_encoder_onnx": True,
        "not_smpl_encoder": True,
        "not_pico_sdk": True,
        "not_dexhand2_contact": True,
        "not_table_s4": True,
        "not_hand_mit_ring": True,
        "not_invented_latency": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "repo": str(REPO_ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("default", "low_latency", "both"),
        default="both",
    )
    args = parser.parse_args()
    variants = ("default", "low_latency") if args.variant == "both" else (args.variant,)
    for variant in variants:
        payload = _dump(variant)
        out = OUT_DEFAULT if variant == "default" else OUT_LOW_LATENCY
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
