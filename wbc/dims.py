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
G1_N_DOF = 29
G1_DECODER_INPUT_DIM = 994


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
    return cfg
