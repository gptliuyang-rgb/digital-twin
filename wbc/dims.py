"""SONIC observation dimensions as a function of body DoF. G1 numbers are not reused."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from interface.schema import REPO_ROOT

T800_SONIC_PATH = Path(__file__).with_name("t800_sonic.yaml")
TOKEN_DIM = 64
HISTORY_FRAMES = 10
ANGVEL_DIM = 3
GRAVITY_DIM = 3
ANCHOR_ORI_DIM = 6  # Zhou 6D: first two columns of R
ROOT_Z_DIM = 1
G1_N_DOF = 29
G1_DECODER_INPUT_DIM = 994
# Official encoder motion window: 10frame_step5 of (q, dq, ori6, z).
# G1  10*29 + 10*29 + 60 + 10 = 650. T800 10*25 + 10*25 + 60 + 10 = 570.
G1_ENCODER_MOTION_DIM = 650
# Official nvidia/GEAR-SONIC encoder ONNX input (HuggingFace observation_config.yaml
# after the unused-key patch). Pre-patch concat was 1762. Neither is a T800 dim.
G1_ENCODER_ONNX_DIM = 1751
G1_ENCODER_ONNX_DIM_UNPATCHED = 1762
G1_N_WRIST_DOF = 6
T800_N_WRIST_DOF = 0
T800_N_LOWER_BODY_DOF = 12  # J00–J11; G1 lower-body is also 12 hip/knee/ankle
ENCODER_MODE_4_DIM = 4  # mode_id + 3 zeros (official encoder_mode_4)
VR_3POINT_POS_DIM = 9
VR_3POINT_ORN_DIM = 12


def encoder_motion_dim(n_dof: int, *, n_frames: int = HISTORY_FRAMES) -> int:
    """Full-body motion window only: q_hist + dq_hist + ori6_hist + z_hist.

    This is **not** the multi-mode encoder ONNX input. T800 570 / G1 650.
    """
    n = int(n_dof)
    nf = int(n_frames)
    return nf * n + nf * n + ANCHOR_ORI_DIM * nf + ROOT_Z_DIM * nf


def t800_encoder_onnx_dim(*, n_dof: int = 25, n_frames: int = HISTORY_FRAMES) -> int:
    """T800 analogue of HuggingFace encoder_observations minus SMPL/wrists.

    Official G1 list is encoder_mode_4 + full q/dq 10frame_step5 + root_z
    (10-frame and current) + anchor ori (current and 10-frame) + lower-body
    q/dq 10frame_step5 + vr_3point pos/orn + smpl_* + wrists. SMPL and wrist
    channels are refused on T800 (ADR-001 / ADR-042). 25-DoF replaces 29:

    4 + 2*(10*25) + 10 + 1 + 6 + 60 + 2*(10*12) + 9 + 12 = 842
    """
    n = int(n_dof)
    if n == G1_N_DOF:
        raise ValueError("t800_encoder_onnx_dim must not be called with G1 29 DoF")
    nf = int(n_frames)
    n_lower = T800_N_LOWER_BODY_DOF
    return (
        ENCODER_MODE_4_DIM
        + nf * n
        + nf * n
        + nf * ROOT_Z_DIM
        + ROOT_Z_DIM
        + ANCHOR_ORI_DIM
        + nf * ANCHOR_ORI_DIM
        + nf * n_lower
        + nf * n_lower
        + VR_3POINT_POS_DIM
        + VR_3POINT_ORN_DIM
    )


def teleop_encoder_required_dim() -> int:
    """Official teleop mode required_observations packed size (not the ONNX input).

    encoder_mode_4 + lower q + lower dq + vr pos + vr orn + current anchor ori
    = 4 + 120 + 120 + 9 + 12 + 6 = 271. Unused superset slots are zero-filled.
    """
    nf = HISTORY_FRAMES
    return (
        ENCODER_MODE_4_DIM
        + nf * T800_N_LOWER_BODY_DOF
        + nf * T800_N_LOWER_BODY_DOF
        + VR_3POINT_POS_DIM
        + VR_3POINT_ORN_DIM
        + ANCHOR_ORI_DIM
    )


def load_t800_sonic(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or T800_SONIC_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("t800_sonic.yaml must be a mapping")
    return raw


def decoder_history_dim(n_dof: int, *, token_dim: int = TOKEN_DIM) -> int:
    """SONIC v1.1 decoder input: token + 10×(ω, q, dq, a, g)."""
    return token_dim + HISTORY_FRAMES * (ANGVEL_DIM + GRAVITY_DIM + 3 * int(n_dof))


def hybrid_encoder_cmd_dim(mode: str, cfg: dict[str, Any] | None = None) -> int:
    """Command token size into the hybrid encoder. Not the decoder history dim.

    3-point: 9 pos + 12 quat = 21. 5-point: 15 pos + 12 quat = 27 (elbows xyz only).
    """
    cfg = cfg or load_t800_sonic()
    if mode in ("vr_3point", "3point"):
        expected = int(cfg["vr_3point_pos_dim"]) + int(cfg["vr_3point_orn_dim"])
        if expected != int(cfg["hybrid_encoder_cmd_dim_3point"]):
            raise AssertionError(f"3-point encoder dim drifted: {expected}")
        return expected
    if mode in ("vr_5point", "5point"):
        expected = int(cfg["vr_5point_pos_dim"]) + int(cfg["vr_5point_orn_dim"])
        if expected != int(cfg["hybrid_encoder_cmd_dim_5point"]):
            raise AssertionError(f"5-point encoder dim drifted: {expected}")
        return expected
    raise ValueError(f"unknown teleop mode {mode!r}")


def assert_t800_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_t800_sonic()
    order = list(cfg["joint_order"])
    if len(order) != int(cfg["n_revolute"]):
        raise ValueError(f"joint_order {len(order)} != n_revolute {cfg['n_revolute']}")
    if len(set(order)) != len(order):
        raise ValueError("t800_sonic joint_order has duplicates")
    if int(cfg["n_revolute"]) == G1_N_DOF:
        raise ValueError("T800 config must not claim 29 DoF (that is G1)")
    expected = decoder_history_dim(int(cfg["n_revolute"]))
    g1 = decoder_history_dim(G1_N_DOF)
    if g1 != G1_DECODER_INPUT_DIM:
        raise AssertionError(f"G1 dim formula drifted: {g1} != {G1_DECODER_INPUT_DIM}")
    cfg = dict(cfg)
    cfg["decoder_input_dim"] = expected
    cfg["g1_decoder_input_dim"] = g1
    cfg["urdf_path"] = str(REPO_ROOT / cfg["source_urdf"])
    cfg["hybrid_encoder_cmd_dim_3point"] = hybrid_encoder_cmd_dim("vr_3point", cfg)
    cfg["hybrid_encoder_cmd_dim_5point"] = hybrid_encoder_cmd_dim("vr_5point", cfg)
    elbows = cfg.get("optional_elbow_bodies", {})
    if elbows.get("left_elbow") == "LINK_ELBOW_YAW_L":
        raise ValueError("5-point elbow must be LINK_ELBOW_PITCH_* (joint), not YAW (forearm)")
    foot = cfg.get("foot_frame", {})
    if foot.get("decision") != "mjcf_link_foot_at_ankle_roll":
        raise ValueError("foot_frame.decision must be mjcf_link_foot_at_ankle_roll (ADR-024)")
    if cfg["tracked_bodies"].get("left_foot") != "LINK_FOOT_L":
        raise ValueError("SONIC left_foot must be MJCF LINK_FOOT_L, not the URDF sole")
    if cfg["tracked_bodies"].get("right_foot") != "LINK_FOOT_R":
        raise ValueError("SONIC right_foot must be MJCF LINK_FOOT_R, not the URDF sole")
    if int(cfg["command_stream_hz"]) != 500:
        raise ValueError("command_stream_hz must be 500 (SONIC §3.5, ADR-039)")
    if int(cfg["control_rate_hz"]) != 50:
        raise ValueError("control_rate_hz must be 50 (SONIC policy/token rate)")
    if int(cfg["planner_hz"]) != 10:
        raise ValueError("planner_hz must be 10 (SONIC L1a)")
    if int(cfg["operator_input_hz"]) != 100:
        raise ValueError("operator_input_hz must be 100 (SONIC §3.5, ADR-040)")
    enc = encoder_motion_dim(int(cfg["n_revolute"]))
    if enc == G1_ENCODER_MOTION_DIM:
        raise ValueError("T800 encoder motion dim must not be G1 650")
    if int(cfg.get("encoder_motion_dim", enc)) != enc:
        raise ValueError(f"encoder_motion_dim {cfg.get('encoder_motion_dim')} != {enc}")
    onnx = t800_encoder_onnx_dim(n_dof=int(cfg["n_revolute"]))
    if onnx in (G1_ENCODER_ONNX_DIM, G1_ENCODER_ONNX_DIM_UNPATCHED, G1_ENCODER_MOTION_DIM):
        raise ValueError("T800 encoder ONNX dim must not be a G1 number")
    if int(cfg.get("n_wrist_dof", 0)) != 0:
        raise ValueError("n_wrist_dof must be 0 on T800")
    if int(cfg.get("n_lower_body_dof", T800_N_LOWER_BODY_DOF)) != T800_N_LOWER_BODY_DOF:
        raise ValueError("n_lower_body_dof must be 12 (J00–J11)")
    cfg["encoder_motion_dim"] = enc
    cfg["g1_encoder_motion_dim"] = G1_ENCODER_MOTION_DIM
    cfg["encoder_onnx_dim"] = onnx
    cfg["g1_encoder_onnx_dim"] = G1_ENCODER_ONNX_DIM
    return cfg
