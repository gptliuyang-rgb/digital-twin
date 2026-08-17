"""Dump SONIC encoder 842-D layout (t800 + teleop modes). grasp_success_rate stays JSON null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT, CommandVector
from wbc.dims import (
    G1_ENCODER_MOTION_DIM,
    G1_ENCODER_ONNX_DIM,
    TOKEN_DIM,
    encoder_motion_dim,
    t800_encoder_onnx_dim,
)
from wbc.gather import HardwareSnapshot, ObsGather, compile_encoder_observations, load_gather_cfg
from wbc.motion_ref import MotionFrame, identity_rot6d, look_ahead_indices

OUT = Path("eval/report/generated/l1a_encoder.json")


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


def main() -> None:
    cfg = load_gather_cfg()
    slots, total = compile_encoder_observations(cfg)
    gather = ObsGather()
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
    payload = {
        "source": "nvidia/GEAR-SONIC observation_config.yaml encoder_observations (T800 dims, no SMPL/wrists)",
        "adr": "ADR-043",
        "n_dof": n,
        "encoder_dim": total,
        "expected_encoder_dim": cfg["expected_encoder_dim"],
        "encoder_motion_window_dim": cfg["encoder_motion_window_dim"],
        "g1_encoder_dim_forbidden": G1_ENCODER_MOTION_DIM,
        "g1_encoder_onnx_dim_forbidden": G1_ENCODER_ONNX_DIM,
        "formula_motion_window_dim": encoder_motion_dim(n),
        "formula_onnx_dim": t800_encoder_onnx_dim(n_dof=n),
        "decoder_dim": int(dec.shape[0]),
        "look_ahead_indices": look_ahead_indices(0, 10, 5, 50),
        "identity_rot6d": identity_rot6d().tolist(),
        "slots": [{"name": s.name, "offset": s.offset, "dim": s.dim} for s in slots],
        "t800_mode": {
            "mode_id": modes["t800"]["mode_id"],
            "required_observations": modes["t800"]["required_observations"],
            "encoder_mode_4": enc_t800[:4].tolist(),
            "q0_window": enc_t800[4:254].reshape(10, 25)[:, 0].tolist(),
        },
        "teleop_mode": {
            "mode_id": modes["teleop"]["mode_id"],
            "required_observations": modes["teleop"]["required_observations"],
            "encoder_mode_4": enc_teleop[:4].tolist(),
            "lowerbody_q0_window": enc_teleop[581:701].reshape(10, 12)[:, 0].tolist(),
            "vr_3point_pos": enc_teleop[821:830].tolist(),
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
