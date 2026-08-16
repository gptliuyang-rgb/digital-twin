"""L0 ckpt action-space diagnostic, decoder history, synthetic motion_lib, load-rand gate."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from eval.l0_ckpt_diagnose import diagnose
from eval.l0_offline_replay import evaluate_episode
from interface.schema import CommandVector, command_dim, load_hand_spec
from vla.adapters.action_space import (
    ActionSpaceMismatch,
    diagnose_action_vector,
    known_layouts,
    require_command_schema_vector,
)
from vla.adapters.chunk_clock import upsample_factor
from vla.adapters.pi05_glue import pi05_chunk_to_command_schema
from vla.client.policy_client import PolicyClient
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import decoder_history_dim
from wbc.gmr.synthetic_clip import synthetic_stand_clip, validate_and_filter_clip
from wbc.load_rand import ranges_match_sim_payload, require_load_aware_ready, sample_training_payloads
from wbc.observation import ProprioHistory, pack_decoder_step


def test_known_layouts_match_frozen_dims() -> None:
    spec = load_hand_spec()
    layouts = known_layouts(spec)
    assert layouts["command_schema_v1"]["dim"] == 75
    assert layouts["command_schema_v1"]["dim"] == command_dim(spec)
    assert layouts["t800_dual_arm_q_plus_hands"]["dim"] == 10 + 40
    assert layouts["t800_body_q"]["dim"] == 25
    assert layouts["g1_body_q"]["dim"] == 29
    assert layouts["sonic_vr_3point"]["dim"] == 21
    assert layouts["sonic_vr_5point"]["dim"] == 27
    assert layouts["command_schema_v1_5point"]["dim"] == 81


def test_case_b_aligned_75d() -> None:
    report = diagnose_action_vector(dim=75)
    assert report.case == "B"
    assert report.layout_name == "command_schema_v1"
    assert report.aligned_with_command_schema is True
    assert report.adapter is None


def test_case_a_needs_fk_adapter() -> None:
    report = diagnose_action_vector(dim=50)
    assert report.case == "A"
    assert report.layout_name == "t800_dual_arm_q_plus_hands"
    assert report.aligned_with_command_schema is False
    assert report.adapter is not None
    assert "CaseAToCommandSchema" in report.adapter


def test_case_c_from_modality_overrides_joint_dim() -> None:
    report = diagnose_action_vector(dim=50, modality={"action": {"left_arm_dq": {}, "right_arm_dq": {}}})
    assert report.case == "C"
    assert any("Case C" in n for n in report.notes)


def test_five_point_is_not_concat() -> None:
    report = diagnose_action_vector(dim=81)
    assert report.layout_name == "command_schema_v1_5point"
    assert report.aligned_with_command_schema is False
    assert any("ADR-017" in n for n in report.notes)


def test_g1_dim_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        diagnose_action_vector(dim=29)


def test_unknown_dim_refused() -> None:
    with pytest.raises(ActionSpaceMismatch, match="none of the frozen layouts"):
        diagnose_action_vector(dim=13)


def test_modality_json_action_dim() -> None:
    out = diagnose(modality={"action_dim": 75, "action": {"left_wrist_pos": {}, "head_rot6d": {}}})
    assert out["case"] == "B"
    assert out["l0_replay_allowed"] is True


def test_policy_client_rejects_case_a() -> None:
    spec = load_hand_spec()

    def infer(_obs: dict) -> np.ndarray:
        return np.zeros(50)

    client = PolicyClient(infer, spec)
    with pytest.raises(ActionSpaceMismatch, match="apply_fk=True"):
        client.step({})


def test_policy_client_accepts_schema() -> None:
    spec = load_hand_spec()
    cmd = CommandVector.zeros(spec)

    def infer(_obs: dict) -> np.ndarray:
        return cmd.to_flat_vector()

    out = PolicyClient(infer, spec).step({})
    np.testing.assert_allclose(out.to_flat_vector(), cmd.to_flat_vector())


def test_l0_replay_refuses_wrong_dim() -> None:
    spec = load_hand_spec()
    with pytest.raises(ActionSpaceMismatch):
        evaluate_episode(np.zeros((4, 8, 50)), np.zeros((4, 8, 50)), spec)


def test_pi05_glue_passthrough_and_refuse() -> None:
    spec = load_hand_spec()
    chunk = np.stack([CommandVector.zeros(spec).to_flat_vector() for _ in range(4)])
    out = pi05_chunk_to_command_schema(chunk, spec)
    assert out.shape == (4, 75)
    with pytest.raises(ActionSpaceMismatch, match="not native"):
        pi05_chunk_to_command_schema(np.zeros((4, 50)), spec)


def test_chunk_clock_integer_upsample() -> None:
    assert upsample_factor("groot_n1_6") == 5
    assert upsample_factor("pi05") == 10
    assert upsample_factor("sonic_teleop") == 1


def test_proprio_history_t800_decoder_dim() -> None:
    hist = ProprioHistory(n_dof=25)
    q = np.linspace(0.0, 0.1, 25)
    hist.push(q, np.zeros(25), np.array([0.2, 0.0, 0.0]), np.ones(25) * 0.01, 0.0)
    packed = hist.decoder_input(np.arange(64, dtype=np.float64))
    assert packed.shape == (decoder_history_dim(25),)
    assert packed.shape == (874,)
    np.testing.assert_allclose(packed[:64], np.arange(64, dtype=np.float64))
    last = packed[-hist.step_dim :]
    expected = pack_decoder_step(q, np.zeros(25), np.array([0.2, 0.0, 0.0]), np.ones(25) * 0.01, np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(last, expected)


def test_proprio_history_refuses_g1_dof() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        ProprioHistory(n_dof=29)


def test_synthetic_clip_filters_clean() -> None:
    lib = synthetic_stand_clip()
    report = validate_and_filter_clip(lib)
    assert report["keep"] is True
    assert report["n_dof"] == 25
    assert report["source"] == "synthetic_stand_not_bones_seed"


def test_synthetic_g1_clip_refused() -> None:
    lib = synthetic_stand_clip(n_dof=29)
    with pytest.raises(G1CheckpointIncompatible):
        validate_and_filter_clip(lib)


def test_load_rand_blocked_until_com() -> None:
    assert ranges_match_sim_payload() is True
    with pytest.raises(PolicyEvalBlocked, match="com_in_wrist_frame_m"):
        require_load_aware_ready()
    with pytest.raises(PolicyEvalBlocked):
        sample_training_payloads(np.random.default_rng(0))


def test_require_command_schema_roundtrip() -> None:
    spec = load_hand_spec()
    vec = CommandVector.zeros(spec).to_flat_vector()
    out = require_command_schema_vector(vec, spec)
    np.testing.assert_allclose(out, vec)
