"""SONIC / T800 whole-body-control contracts. Simulator-free."""

from wbc.checkpoint import G1CheckpointIncompatible, refuse_g1_checkpoint
from wbc.dims import decoder_history_dim, load_t800_sonic
from wbc.observation import heading_gravity, policy_proprio
from wbc.teleop import command_to_vr_3point, vr_3point_to_wbc_fields

__all__ = [
    "G1CheckpointIncompatible",
    "command_to_vr_3point",
    "decoder_history_dim",
    "heading_gravity",
    "load_t800_sonic",
    "policy_proprio",
    "refuse_g1_checkpoint",
    "vr_3point_to_wbc_fields",
]
