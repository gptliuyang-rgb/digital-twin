"""Schema, joint-map, and command-vector tests. No simulator imports."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from interface.schema import (
    HAND_SPEC_PATH,
    CommandVector,
    SpecIncompleteError,
    collect_repo_required_inputs,
    command_dim,
    find_required_inputs,
    load_command_schema,
    load_frames,
    load_hand_spec,
    load_joint_map,
    load_yaml,
    validate_repo_spec,
)


def test_import_validate_lists_required_inputs() -> None:
    with pytest.raises(SpecIncompleteError) as excinfo:
        validate_repo_spec()
    message = str(excinfo.value)
    assert "friction_vs_cardboard_static" in message
    assert "normal_stiffness_n_per_m" in message
    assert "motor_max_torque_nm" in message
    assert "hardware_kp" in message
    assert "command_latency_ms" in message
    assert "hardware_revision" in message
    assert len(excinfo.value.missing) >= 5


def test_p0_required_input_count_is_bounded() -> None:
    spec = load_yaml(HAND_SPEC_PATH)
    p0_keys = {
        "friction_vs_cardboard_static",
        "friction_vs_cardboard_dynamic",
        "normal_stiffness_n_per_m",
        "motor_max_torque_nm",
        "hardware_kp",
        "hardware_kd",
        "command_latency_ms",
        "com_in_wrist_frame_m",
        "fingertip_geometry_radius_m",
        "fingertip_material",
        "max_delta_q_rad",
        "velocity_limit_rad_s",
        "hardware_revision",
        "hardware_has_tactile",
    }
    missing = set(find_required_inputs(spec))
    p0_missing = {item for item in missing if item.split(".")[0] in p0_keys or item in p0_keys}
    assert len(p0_missing) <= 12


def test_topology_from_official_docs() -> None:
    spec = load_hand_spec()
    assert spec.n_fingers == 5
    assert spec.n_active_dof == 20
    assert spec.n_total_dof == 20
    assert spec.coupling_type == "none"
    assert spec.sim_model_revision == "hand2_beta2"
    assert spec.product_mass_kg == pytest.approx(0.800)
    assert spec.sim_mass_kg == pytest.approx(0.6228)
    assert spec.skeleton_mass_kg == pytest.approx(0.6207)
    facts = spec.revision_facts()
    assert facts["n_pad_bodies"] == 5
    assert facts["pad_collision_in_official_model"] is True
    assert spec.raw["has_tactile"] is True
    assert spec.raw["tactile_layout"]["thumb_points"] == 40
    assert spec.raw["tactile_layout"]["other_finger_points"] == 34
    assert spec.raw["voltage_range_v"] == [11, 13]


def test_joint_order_unique_and_matches_map() -> None:
    spec = load_hand_spec()
    assert len(spec.joint_order) == spec.n_active_dof
    assert len(set(spec.joint_order)) == spec.n_active_dof
    entries = load_joint_map()
    assert [e.canonical for e in entries] == spec.joint_order
    assert [e.sdk_index for e in entries] == list(range(20))
    assert entries[0].mjcf_actuator("r") == "r_THJ0"
    assert entries[4].mjcf_joint("l") == "l_index_finger_mcp_flex"
    assert entries[19].sdk_label == "pinky_S4"


def test_command_schema_resolves_hand_dim() -> None:
    spec = load_hand_spec()
    schema = load_command_schema()
    assert schema["frame"] == "robot_heading_frame"
    assert schema["rotation_representation"] == "rot6d_zhou2019"
    assert command_dim(spec) == 32 + 2 * spec.n_active_dof + 3
    frames = load_frames()
    assert frames["frames"]["left_wrist"]["t800_link"] == "LINK_WRIST_END_L"


def test_flat_vector_roundtrip() -> None:
    spec = load_hand_spec()
    cmd = CommandVector.zeros(spec)
    cmd.left_hand_q[3] = 0.2
    cmd.right_hand_q[7] = -0.1
    cmd.tool_trigger = 1
    cmd.validate()
    flat = cmd.to_flat_vector()
    restored = CommandVector.from_flat_vector(flat, spec)
    np.testing.assert_allclose(restored.to_flat_vector(), flat)
    assert restored.tool_trigger == 1
    assert restored.left_hand_q[3] == pytest.approx(0.2)


def test_wrong_dim_rejected() -> None:
    spec = load_hand_spec()
    with pytest.raises(ValueError):
        CommandVector.from_flat_vector(np.zeros(3), spec)


def test_duplicate_joint_order_rejected(tmp_path) -> None:
    raw = load_yaml(HAND_SPEC_PATH)
    raw = copy.deepcopy(raw)
    raw["joint_order"][1] = raw["joint_order"][0]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_hand_spec(path)


def test_collect_repo_required_inputs_nonempty() -> None:
    missing = collect_repo_required_inputs()
    assert any("mount_transform.yaml" in item for item in missing)
    assert any("dexhand2_spec.yaml" in item for item in missing)
