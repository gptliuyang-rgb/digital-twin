"""SONIC / T800 whole-body-control contracts. Simulator-free."""

from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import decoder_history_dim, hybrid_encoder_cmd_dim, load_t800_sonic
from wbc.observation import ProprioHistory, heading_gravity, pack_decoder_step, policy_proprio
from wbc.planner import KinematicPlanner
from wbc.spring import (
    HEADING_DAMPING_C,
    MAX_NAV_SPEED_MPS,
    POSITION_DAMPING_C,
    RootKeyframe,
    RootSpringRef,
    RootSpringState,
    critically_damped,
    spring_root_keyframe,
)
from wbc.teleop import (
    FivePointCommand,
    TeleopModeIncompatible,
    command_to_vr_3point,
    command_to_vr_5point,
    refuse_teleop_mode_mismatch,
    vr_3point_to_wbc_fields,
)

__all__ = [
    "FivePointCommand",
    "G1CheckpointIncompatible",
    "HEADING_DAMPING_C",
    "KinematicPlanner",
    "MAX_NAV_SPEED_MPS",
    "POSITION_DAMPING_C",
    "RootKeyframe",
    "RootSpringRef",
    "RootSpringState",
    "TeleopModeIncompatible",
    "critically_damped",
    "spring_root_keyframe",
    "command_to_vr_3point",
    "command_to_vr_5point",
    "decoder_history_dim",
    "ProprioHistory",
    "heading_gravity",
    "hybrid_encoder_cmd_dim",
    "load_t800_sonic",
    "pack_decoder_step",
    "policy_proprio",
    "refuse_g1_checkpoint",
    "refuse_teleop_mode_mismatch",
    "vr_3point_to_wbc_fields",
]
