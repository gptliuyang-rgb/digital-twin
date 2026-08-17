"""Dump SONIC encoder motion_* 10frame_step5 layout. grasp_success_rate stays JSON null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.dims import G1_ENCODER_MOTION_DIM, TOKEN_DIM, encoder_motion_dim
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
    enc = gather.assemble_encoder()
    token = np.zeros(TOKEN_DIM)
    dec = gather.control_tick(token, t_s=0.0)
    q = enc[:250].reshape(10, 25)[:, 0].tolist()
    payload = {
        "source": "GEAR-SONIC observation_config.yaml encoder_observations (T800 dims)",
        "adr": "ADR-042",
        "n_dof": n,
        "encoder_dim": total,
        "expected_encoder_dim": cfg["expected_encoder_dim"],
        "g1_encoder_dim_forbidden": G1_ENCODER_MOTION_DIM,
        "formula_dim": encoder_motion_dim(n),
        "decoder_dim": int(dec.shape[0]),
        "look_ahead_indices": look_ahead_indices(0, 10, 5, 50),
        "identity_rot6d": identity_rot6d().tolist(),
        "slots": [{"name": s.name, "offset": s.offset, "dim": s.dim} for s in slots],
        "q0_window": q,
        "not_invented_clip": True,
        "not_g1_encoder_onnx": True,
        "not_dexhand2_contact": True,
        "not_table_s4": True,
        "not_hand_mit_ring": True,
        "not_invented_latency": True,
        "not_pico_sdk": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "repo": str(REPO_ROOT),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
