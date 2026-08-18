"""Table S4 root-push extrema are the PPO domain-rand source, not pad–cardboard."""

from __future__ import annotations

import ast

import pytest
import yaml

from interface.schema import REPO_ROOT
from wbc.ppo.recipe import load_ppo_recipe
from wbc.ppo.table_s4 import (
    ANG_VEL_JITTER_AXES,
    ANGVEL_AXES,
    COM_AXES,
    LIN_VEL_JITTER_AXES,
    ORI_JITTER_AXES,
    POS_JITTER_AXES,
    RECORDED_ONLY_PHYSICAL_FIELDS,
    SWEEP_AXES,
    VERTICAL_AXES,
    RecordedOnlyPhysicalMapError,
    axis_aligned_angvel_extrema_rad_s,
    axis_aligned_linvel_extrema_mps,
    base_com_offset_extrema,
    default_joint_pos_offset_extrema,
    duration_extrema_s,
    force_n_from_impulse,
    load_table_s4,
    physical_base_com_offset_ranges_m,
    physical_default_joint_pos_offset_range_rad,
    physical_dynamic_friction_range,
    physical_restitution_range,
    physical_static_friction_range,
    recorded_only_physical,
    refuse_dynamic_friction_map,
    refuse_recorded_only_physical_map,
    refuse_restitution_solref_map,
    root_push_angvel_ranges_rad_s,
    root_push_duration_s,
    root_push_ranges_mps,
    static_friction_extrema,
    sustained_force_cases,
    sustained_torque_cases,
    target_motion_ang_vel_jitter_extrema,
    target_motion_ang_vel_jitter_ranges_rad_s,
    target_motion_joint_jitter_extrema,
    target_motion_joint_jitter_range_rad,
    target_motion_lin_vel_jitter_extrema,
    target_motion_lin_vel_jitter_ranges_mps,
    target_motion_ori_jitter_extrema,
    target_motion_ori_jitter_ranges_rad,
    target_motion_pos_jitter_extrema,
    target_motion_pos_jitter_ranges_m,
    torque_nm_from_impulse,
    vertical_force_cases,
    vertical_linvel_extrema_mps,
)

_FORBIDDEN_SOLREF_ATTRS = frozenset({"geom_solref", "geom_solimp"})
_SCAN_ROOTS = (REPO_ROOT / "sim", REPO_ROOT / "eval", REPO_ROOT / "wbc")


def _is_full_array_slice(slice_node: ast.AST) -> bool:
    return (
        isinstance(slice_node, ast.Slice)
        and slice_node.lower is None
        and slice_node.upper is None
        and slice_node.step is None
    )


def _last_index_is_slide_zero(slice_node: ast.AST) -> bool:
    if isinstance(slice_node, ast.Tuple) and slice_node.elts:
        last = slice_node.elts[-1]
        return isinstance(last, ast.Constant) and last.value == 0
    return False


def geom_contact_write_hits(source: str, *, filename: str = "<mem>") -> list[str]:
    """Assignment targets that would map restitution or μd onto a MuJoCo geom."""
    tree = ast.parse(source, filename=filename)
    hits: list[str] = []
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets.append(node.target)
        for target in targets:
            hits.extend(_contact_target_hits(target, filename, node.lineno))
    return hits


def _contact_target_hits(target: ast.AST, filename: str, lineno: int) -> list[str]:
    hits: list[str] = []
    if isinstance(target, ast.Tuple):
        for elt in target.elts:
            hits.extend(_contact_target_hits(elt, filename, lineno))
        return hits
    if isinstance(target, ast.Attribute) and target.attr in _FORBIDDEN_SOLREF_ATTRS:
        hits.append(f"{filename}:{lineno}:{target.attr}")
        return hits
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
        attr = target.value.attr
        if attr in _FORBIDDEN_SOLREF_ATTRS:
            hits.append(f"{filename}:{lineno}:{attr}")
            return hits
        if attr == "geom_friction":
            if _is_full_array_slice(target.slice) or _last_index_is_slide_zero(target.slice):
                return hits
            hits.append(f"{filename}:{lineno}:geom_friction_non_slide")
    return hits


def test_table_s4_is_not_dexhand2_contact() -> None:
    raw = load_table_s4()
    assert raw["not_dexhand2_contact"] is True
    assert raw["applies_to"] == "t800_wbc_ppo"
    ranges = root_push_ranges_mps()
    assert ranges["x"] == (-0.5, 0.5)
    assert ranges["y"] == (-0.5, 0.5)
    assert ranges["z"] == (-0.2, 0.2)
    assert root_push_duration_s() == (1.0, 3.0)
    assert duration_extrema_s() == (1.0, 3.0)
    ang = root_push_angvel_ranges_rad_s()
    assert ang["roll"] == (-0.52, 0.52)
    assert ang["pitch"] == (-0.52, 0.52)
    assert ang["yaw"] == (-0.78, 0.78)


def test_planar_extrema_match_domain_rand_and_exclude_z() -> None:
    cases = axis_aligned_linvel_extrema_mps()
    assert SWEEP_AXES == ("x", "y")
    assert [c["name"] for c in cases] == ["-x", "+x", "-y", "+y"]
    by_name = {c["name"]: c["lin_vel_mps"] for c in cases}
    assert by_name["-x"] == [-0.5, 0.0, 0.0]
    assert by_name["+x"] == [0.5, 0.0, 0.0]
    assert by_name["-y"] == [0.0, -0.5, 0.0]
    assert by_name["+y"] == [0.0, 0.5, 0.0]
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    assert all(c["kind"] == "one_shot_qvel" for c in cases)
    recipe = load_ppo_recipe()
    push = recipe["domain_rand"]["root_push"]["lin_vel_mps"]
    assert by_name["+x"][0] == float(push["x"][1])
    assert by_name["-y"][1] == float(push["y"][0])
    assert all(c["lin_vel_mps"][2] == 0.0 for c in cases)


def test_z_extrema_are_a_separate_vertical_axis() -> None:
    assert VERTICAL_AXES == ("z",)
    cases = vertical_linvel_extrema_mps()
    assert [c["name"] for c in cases] == ["-z", "+z"]
    assert cases[0]["lin_vel_mps"] == [0.0, 0.0, -0.2]
    assert cases[1]["lin_vel_mps"] == [0.0, 0.0, 0.2]
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    via_axes = axis_aligned_linvel_extrema_mps(axes=("z",))
    assert [c["lin_vel_mps"] for c in via_axes] == [c["lin_vel_mps"] for c in cases]
    force = vertical_force_cases()
    assert [c["name"] for c in force] == ["-z_T1.0s", "-z_T3.0s", "+z_T1.0s", "+z_T3.0s"]
    by_name = {c["name"]: c for c in force}
    assert by_name["+z_T1.0s"]["lin_vel_mps"] == [0.0, 0.0, 0.2]
    assert by_name["+z_T1.0s"]["duration_s"] == 1.0
    assert by_name["-z_T3.0s"]["lin_vel_mps"] == [0.0, 0.0, -0.2]
    assert by_name["-z_T3.0s"]["duration_s"] == 3.0
    fz = force_n_from_impulse(84.917, [0.0, 0.0, 0.2], 1.0)
    assert fz == pytest.approx([0.0, 0.0, 16.9834])
    fz_t3 = force_n_from_impulse(84.917, [0.0, 0.0, 0.2], 3.0)
    assert fz_t3 == pytest.approx([0.0, 0.0, 16.9834 / 3.0])


def test_unknown_axis_raises() -> None:
    with pytest.raises(ValueError, match="unknown Table S4 axis"):
        axis_aligned_linvel_extrema_mps(axes=("w",))
    with pytest.raises(ValueError, match="unknown Table S4 angvel axis"):
        axis_aligned_angvel_extrema_rad_s(axes=("spin",))


def test_impulse_force_matches_mass_times_delta_v() -> None:
    force = force_n_from_impulse(80.0, [0.0, 0.5, 0.0], 1.0)
    assert force == [0.0, 40.0, 0.0]
    force_t3 = force_n_from_impulse(80.0, [0.0, 0.5, 0.0], 3.0)
    assert force_t3 == [0.0, 40.0 / 3.0, 0.0]
    impulse = [f * 3.0 for f in force_t3]
    assert impulse == [0.0, 40.0, 0.0]
    with pytest.raises(ValueError, match="duration_s"):
        force_n_from_impulse(80.0, [0.0, 0.5, 0.0], 0.0)
    with pytest.raises(ValueError, match="mass_kg"):
        force_n_from_impulse(0.0, [0.0, 0.5, 0.0], 1.0)


def test_sustained_force_cases_are_planar_times_duration() -> None:
    cases = sustained_force_cases()
    assert [c["name"] for c in cases] == [
        "-x_T1.0s",
        "-x_T3.0s",
        "+x_T1.0s",
        "+x_T3.0s",
        "-y_T1.0s",
        "-y_T3.0s",
        "+y_T1.0s",
        "+y_T3.0s",
    ]
    assert all(c["kind"] == "sustained_force" for c in cases)
    assert all(c["not_one_shot_qvel"] is True for c in cases)
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    by_name = {c["name"]: c for c in cases}
    assert by_name["+y_T1.0s"]["lin_vel_mps"] == [0.0, 0.5, 0.0]
    assert by_name["+y_T1.0s"]["duration_s"] == 1.0
    assert by_name["-x_T3.0s"]["lin_vel_mps"] == [-0.5, 0.0, 0.0]
    assert by_name["-x_T3.0s"]["duration_s"] == 3.0


def test_angvel_extrema_match_domain_rand() -> None:
    cases = axis_aligned_angvel_extrema_rad_s()
    assert ANGVEL_AXES == ("roll", "pitch", "yaw")
    assert [c["name"] for c in cases] == ["-roll", "+roll", "-pitch", "+pitch", "-yaw", "+yaw"]
    by_name = {c["name"]: c["ang_vel_rad_s"] for c in cases}
    assert by_name["-roll"] == [-0.52, 0.0, 0.0]
    assert by_name["+roll"] == [0.52, 0.0, 0.0]
    assert by_name["-pitch"] == [0.0, -0.52, 0.0]
    assert by_name["+pitch"] == [0.0, 0.52, 0.0]
    assert by_name["-yaw"] == [0.0, 0.0, -0.78]
    assert by_name["+yaw"] == [0.0, 0.0, 0.78]
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    assert all(c["kind"] == "one_shot_qvel" for c in cases)
    assert all(c["not_sustained_torque"] is True for c in cases)
    recipe = load_ppo_recipe()
    ang = recipe["domain_rand"]["root_push"]["ang_vel_rad_s"]
    assert by_name["+roll"][0] == float(ang["roll"][1])
    assert by_name["-yaw"][2] == float(ang["yaw"][0])


def test_torque_matches_inertia_times_omega() -> None:
    torque = torque_nm_from_impulse([2.0, 2.0, 4.0], [0.0, 0.0, 0.78], 1.0)
    assert torque == pytest.approx([0.0, 0.0, 3.12])
    torque_t3 = torque_nm_from_impulse([2.0, 2.0, 4.0], [0.0, 0.0, 0.78], 3.0)
    assert torque_t3 == pytest.approx([0.0, 0.0, 1.04])
    coupled = torque_nm_from_impulse(
        [[2.0, 0.1, 0.0], [0.1, 3.0, 0.0], [0.0, 0.0, 4.0]],
        [0.52, 0.0, 0.0],
        1.0,
    )
    assert coupled == pytest.approx([1.04, 0.052, 0.0])
    with pytest.raises(ValueError, match="duration_s"):
        torque_nm_from_impulse([1.0, 1.0, 1.0], [0.0, 0.0, 0.78], 0.0)
    with pytest.raises(ValueError, match="principal inertia"):
        torque_nm_from_impulse([1.0, 0.0, 1.0], [0.0, 0.0, 0.78], 1.0)


def test_sustained_torque_cases_are_angvel_times_duration() -> None:
    cases = sustained_torque_cases()
    assert [c["name"] for c in cases] == [
        "-roll_T1.0s",
        "-roll_T3.0s",
        "+roll_T1.0s",
        "+roll_T3.0s",
        "-pitch_T1.0s",
        "-pitch_T3.0s",
        "+pitch_T1.0s",
        "+pitch_T3.0s",
        "-yaw_T1.0s",
        "-yaw_T3.0s",
        "+yaw_T1.0s",
        "+yaw_T3.0s",
    ]
    assert all(c["kind"] == "sustained_torque" for c in cases)
    assert all(c["not_one_shot_qvel"] is True for c in cases)
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    by_name = {c["name"]: c for c in cases}
    assert by_name["+yaw_T1.0s"]["ang_vel_rad_s"] == [0.0, 0.0, 0.78]
    assert by_name["+yaw_T1.0s"]["duration_s"] == 1.0
    assert by_name["-roll_T3.0s"]["ang_vel_rad_s"] == [-0.52, 0.0, 0.0]
    assert by_name["-roll_T3.0s"]["duration_s"] == 3.0


def test_static_friction_extrema_match_domain_rand_and_are_not_pad_cardboard() -> None:
    assert physical_static_friction_range() == (0.3, 1.6)
    assert physical_dynamic_friction_range() == (0.3, 1.2)
    assert physical_restitution_range() == (0.0, 0.5)
    cases = static_friction_extrema()
    assert [c["name"] for c in cases] == ["mu_s_0.3", "mu_s_1.6"]
    by_name = {c["name"]: c for c in cases}
    assert by_name["mu_s_0.3"]["mu_slide"] == 0.3
    assert by_name["mu_s_1.6"]["mu_slide"] == 1.6
    assert all(c["kind"] == "wbc_floor_slide_friction" for c in cases)
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    assert all(c["not_pad_cardboard"] is True for c in cases)
    assert all(c["dynamic_friction_not_mapped"] is True for c in cases)
    assert all(c["restitution_not_mapped"] is True for c in cases)
    recipe = load_ppo_recipe()
    static = recipe["domain_rand"]["physical"]["static_friction"]
    assert by_name["mu_s_0.3"]["mu_slide"] == float(static[0])
    assert by_name["mu_s_1.6"]["mu_slide"] == float(static[1])
    spec = (REPO_ROOT / "assets" / "dexhand2" / "meta" / "dexhand2_spec.yaml").read_text(
        encoding="utf-8"
    )
    assert "friction_vs_cardboard_static: REQUIRED_INPUT" in spec
    assert "friction_vs_cardboard_dynamic: REQUIRED_INPUT" in spec
    # Table S4 WBC floor numbers must not leak into the Hand 2 spec.
    assert "static_friction: [0.3, 1.6]" not in spec


def test_static_friction_is_not_a_root_push_axis() -> None:
    push_names = [c["name"] for c in axis_aligned_linvel_extrema_mps()]
    assert "mu_s_0.3" not in push_names
    assert "mu_s_1.6" not in push_names
    force_names = [c["name"] for c in sustained_force_cases()]
    assert "mu_s_0.3" not in force_names
    assert "mu_s_1.6" not in force_names


def test_base_com_offset_extrema_match_domain_rand_and_are_not_wrist_com() -> None:
    ranges = physical_base_com_offset_ranges_m()
    assert COM_AXES == ("x", "y", "z")
    assert ranges["x"] == (-0.075, 0.075)
    assert ranges["y"] == (-0.1, 0.1)
    assert ranges["z"] == (-0.1, 0.1)
    cases = base_com_offset_extrema()
    assert [c["name"] for c in cases] == [
        "com_-x",
        "com_+x",
        "com_-y",
        "com_+y",
        "com_-z",
        "com_+z",
    ]
    by_name = {c["name"]: c for c in cases}
    assert by_name["com_-x"]["offset_m"] == [-0.075, 0.0, 0.0]
    assert by_name["com_+x"]["offset_m"] == [0.075, 0.0, 0.0]
    assert by_name["com_-y"]["offset_m"] == [0.0, -0.1, 0.0]
    assert by_name["com_+y"]["offset_m"] == [0.0, 0.1, 0.0]
    assert by_name["com_-z"]["offset_m"] == [0.0, 0.0, -0.1]
    assert by_name["com_+z"]["offset_m"] == [0.0, 0.0, 0.1]
    assert all(c["kind"] == "wbc_base_com_ipos_offset" for c in cases)
    assert all(c["additive_to_compiled_ipos"] is True for c in cases)
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    assert all(c["not_pad_cardboard"] is True for c in cases)
    assert all(c["not_wrist_com"] is True for c in cases)
    recipe = load_ppo_recipe()
    com = recipe["domain_rand"]["physical"]["base_com_offset_m"]
    assert by_name["com_+x"]["offset_m"][0] == float(com["x"][1])
    assert by_name["com_-z"]["offset_m"][2] == float(com["z"][0])
    spec = (REPO_ROOT / "assets" / "dexhand2" / "meta" / "dexhand2_spec.yaml").read_text(
        encoding="utf-8"
    )
    assert "com_in_wrist_frame_m: REQUIRED_INPUT" in spec
    assert "base_com_offset_m" not in spec


def test_base_com_offset_is_not_a_root_push_axis() -> None:
    push_names = [c["name"] for c in axis_aligned_linvel_extrema_mps()]
    com_names = [c["name"] for c in base_com_offset_extrema()]
    assert set(com_names).isdisjoint(push_names)
    force_names = [c["name"] for c in sustained_force_cases()]
    assert set(com_names).isdisjoint(force_names)
    friction_names = [c["name"] for c in static_friction_extrema()]
    assert set(com_names).isdisjoint(friction_names)


def test_default_joint_pos_offset_extrema_match_domain_rand() -> None:
    assert physical_default_joint_pos_offset_range_rad() == (-0.01, 0.01)
    cases = default_joint_pos_offset_extrema()
    assert [c["name"] for c in cases] == ["qpos_-0.01", "qpos_+0.01"]
    by_name = {c["name"]: c for c in cases}
    assert by_name["qpos_-0.01"]["offset_rad"] == -0.01
    assert by_name["qpos_+0.01"]["offset_rad"] == 0.01
    assert all(c["kind"] == "wbc_default_joint_pos_offset" for c in cases)
    assert all(c["additive_to_default_q_des"] is True for c in cases)
    assert all(c["freejoint_unchanged"] is True for c in cases)
    assert all(c["not_per_joint_corner_grid"] is True for c in cases)
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    assert all(c["not_pad_cardboard"] is True for c in cases)
    assert all(c["restitution_not_mapped"] is True for c in cases)
    recipe = load_ppo_recipe()
    raw = recipe["domain_rand"]["physical"]["default_joint_pos_offset_rad"]
    assert by_name["qpos_-0.01"]["offset_rad"] == float(raw[0])
    assert by_name["qpos_+0.01"]["offset_rad"] == float(raw[1])
    spec = (REPO_ROOT / "assets" / "dexhand2" / "meta" / "dexhand2_spec.yaml").read_text(
        encoding="utf-8"
    )
    assert "default_joint_pos_offset_rad" not in spec
    assert "command_latency_ms: REQUIRED_INPUT" in spec


def test_default_joint_pos_offset_is_not_a_root_push_axis() -> None:
    push_names = [c["name"] for c in axis_aligned_linvel_extrema_mps()]
    qpos_names = [c["name"] for c in default_joint_pos_offset_extrema()]
    assert set(qpos_names).isdisjoint(push_names)
    force_names = [c["name"] for c in sustained_force_cases()]
    assert set(qpos_names).isdisjoint(force_names)
    friction_names = [c["name"] for c in static_friction_extrema()]
    assert set(qpos_names).isdisjoint(friction_names)
    com_names = [c["name"] for c in base_com_offset_extrema()]
    assert set(qpos_names).isdisjoint(com_names)


def test_target_motion_joint_jitter_extrema_match_domain_rand() -> None:
    assert target_motion_joint_jitter_range_rad() == (-0.1, 0.1)
    cases = target_motion_joint_jitter_extrema()
    assert [c["name"] for c in cases] == ["q_jit_-0.1", "q_jit_+0.1"]
    by_name = {c["name"]: c for c in cases}
    assert by_name["q_jit_-0.1"]["offset_rad"] == -0.1
    assert by_name["q_jit_+0.1"]["offset_rad"] == 0.1
    assert all(c["kind"] == "target_motion_joint_jitter" for c in cases)
    assert all(c["additive_to_clip_dof_pos"] is True for c in cases)
    assert all(c["not_default_joint_pos_offset"] is True for c in cases)
    assert all(c["not_per_joint_corner_grid"] is True for c in cases)
    assert all(c["not_root_push"] is True for c in cases)
    assert all(c["not_dexhand2_contact"] is True for c in cases)
    assert all(c["not_pad_cardboard"] is True for c in cases)
    assert all(c["restitution_not_mapped"] is True for c in cases)
    recipe = load_ppo_recipe()
    raw = recipe["domain_rand"]["target_motion"]["joint_jitter_rad"]
    assert by_name["q_jit_-0.1"]["offset_rad"] == float(raw[0])
    assert by_name["q_jit_+0.1"]["offset_rad"] == float(raw[1])
    spec = (REPO_ROOT / "assets" / "dexhand2" / "meta" / "dexhand2_spec.yaml").read_text(
        encoding="utf-8"
    )
    assert "joint_jitter_rad" not in spec
    assert "command_latency_ms: REQUIRED_INPUT" in spec
    qpos_names = [c["name"] for c in default_joint_pos_offset_extrema()]
    assert set(by_name).isdisjoint(qpos_names)


def test_target_motion_pos_and_ori_jitter_extrema_match_domain_rand() -> None:
    assert POS_JITTER_AXES == ("x", "y", "z")
    assert ORI_JITTER_AXES == ("roll", "pitch", "yaw")
    pos_ranges = target_motion_pos_jitter_ranges_m()
    assert pos_ranges["x"] == (-0.05, 0.05)
    assert pos_ranges["y"] == (-0.05, 0.05)
    assert pos_ranges["z"] == (-0.01, 0.01)
    pos = target_motion_pos_jitter_extrema()
    assert [c["name"] for c in pos] == [
        "pos_-x",
        "pos_+x",
        "pos_-y",
        "pos_+y",
        "pos_-z",
        "pos_+z",
    ]
    by_pos = {c["name"]: c["offset_m"] for c in pos}
    assert by_pos["pos_+x"] == [0.05, 0.0, 0.0]
    assert by_pos["pos_-z"] == [0.0, 0.0, -0.01]
    assert all(c["kind"] == "target_motion_pos_jitter" for c in pos)
    assert all(c["additive_to_clip_root_pos"] is True for c in pos)
    assert all(c["not_joint_jitter"] is True for c in pos)
    assert all(c["not_height_ori_gate"] is True for c in pos)
    assert all(c["mujoco_channel"] == "clip_root_pos" for c in pos)
    assert all(c["not_root_push"] is True for c in pos)
    ori_ranges = target_motion_ori_jitter_ranges_rad()
    assert ori_ranges["roll"] == (-0.1, 0.1)
    assert ori_ranges["pitch"] == (-0.1, 0.1)
    assert ori_ranges["yaw"] == (-0.2, 0.2)
    ori = target_motion_ori_jitter_extrema()
    assert [c["name"] for c in ori] == [
        "ori_-roll",
        "ori_+roll",
        "ori_-pitch",
        "ori_+pitch",
        "ori_-yaw",
        "ori_+yaw",
    ]
    by_ori = {c["name"]: c["offset_rad"] for c in ori}
    assert by_ori["ori_+roll"] == [0.1, 0.0, 0.0]
    assert by_ori["ori_-yaw"] == [0.0, 0.0, -0.2]
    assert all(c["kind"] == "target_motion_ori_jitter" for c in ori)
    assert all(c["additive_to_clip_root_rot"] is True for c in ori)
    assert all(c["not_joint_jitter"] is True for c in ori)
    assert all(c["not_height_ori_gate"] is True for c in ori)
    assert all(c["mujoco_channel"] == "clip_root_rot" for c in ori)
    recipe = load_ppo_recipe()
    tm = recipe["domain_rand"]["target_motion"]
    assert by_pos["pos_+y"][1] == float(tm["pos_jitter_m"]["y"][1])
    assert by_ori["ori_+pitch"][1] == float(tm["ori_jitter_rad"]["pitch"][1])


def test_synthetic_walk_clip_has_velocity_content() -> None:
    from wbc.gmr.synthetic_clip import (
        SYNTHETIC_WALK_ANGVEL_RAD_S,
        SYNTHETIC_WALK_LINVEL_MPS,
        synthetic_stand_clip,
        synthetic_walk_clip,
    )

    stand = synthetic_stand_clip(n_frames=4, fps=50.0, amplitude_rad=0.02)
    walk = synthetic_walk_clip(n_frames=4, fps=50.0, amplitude_rad=0.02)
    assert "root_linvel" not in stand
    assert "root_angvel" not in stand
    assert walk["source"] == "synthetic_walk_not_bones_seed"
    assert walk["pose_is_stand"] is True
    assert walk["root_linvel"][0].tolist() == list(SYNTHETIC_WALK_LINVEL_MPS)
    assert walk["root_angvel"][0].tolist() == list(SYNTHETIC_WALK_ANGVEL_RAD_S)
    assert walk["dof_pos"].shape == stand["dof_pos"].shape
    assert walk["root_pos"][0].tolist() == stand["root_pos"][0].tolist()


def test_target_motion_lin_vel_and_ang_vel_jitter_extrema_match_domain_rand() -> None:
    assert LIN_VEL_JITTER_AXES == ("x", "y", "z")
    assert ANG_VEL_JITTER_AXES == ("roll", "pitch", "yaw")
    lin_ranges = target_motion_lin_vel_jitter_ranges_mps()
    assert lin_ranges["x"] == (-0.5, 0.5)
    assert lin_ranges["y"] == (-0.5, 0.5)
    assert lin_ranges["z"] == (-0.2, 0.2)
    lin = target_motion_lin_vel_jitter_extrema()
    assert [c["name"] for c in lin] == [
        "lin_vel_-x",
        "lin_vel_+x",
        "lin_vel_-y",
        "lin_vel_+y",
        "lin_vel_-z",
        "lin_vel_+z",
    ]
    by_lin = {c["name"]: c["offset_mps"] for c in lin}
    assert by_lin["lin_vel_+x"] == [0.5, 0.0, 0.0]
    assert by_lin["lin_vel_-z"] == [0.0, 0.0, -0.2]
    assert all(c["kind"] == "target_motion_lin_vel_jitter" for c in lin)
    assert all(c["additive_to_clip_root_linvel"] is True for c in lin)
    assert all(c["not_root_push"] is True for c in lin)
    assert all(c["not_height_ori_gate"] is True for c in lin)
    assert all(c["mujoco_channel"] == "clip_root_linvel" for c in lin)
    ang_ranges = target_motion_ang_vel_jitter_ranges_rad_s()
    assert ang_ranges["roll"] == (-0.52, 0.52)
    assert ang_ranges["pitch"] == (-0.52, 0.52)
    assert ang_ranges["yaw"] == (-0.78, 0.78)
    ang = target_motion_ang_vel_jitter_extrema()
    assert [c["name"] for c in ang] == [
        "ang_vel_-roll",
        "ang_vel_+roll",
        "ang_vel_-pitch",
        "ang_vel_+pitch",
        "ang_vel_-yaw",
        "ang_vel_+yaw",
    ]
    by_ang = {c["name"]: c["offset_rad_s"] for c in ang}
    assert by_ang["ang_vel_+roll"] == [0.52, 0.0, 0.0]
    assert by_ang["ang_vel_-yaw"] == [0.0, 0.0, -0.78]
    assert all(c["kind"] == "target_motion_ang_vel_jitter" for c in ang)
    assert all(c["additive_to_clip_root_angvel"] is True for c in ang)
    assert all(c["not_root_push"] is True for c in ang)
    assert all(c["mujoco_channel"] == "clip_root_angvel" for c in ang)
    recipe = load_ppo_recipe()
    tm = recipe["domain_rand"]["target_motion"]
    assert by_lin["lin_vel_+y"][1] == float(tm["lin_vel_jitter_mps"]["y"][1])
    assert by_ang["ang_vel_+pitch"][1] == float(tm["ang_vel_jitter_rad_s"]["pitch"][1])
    spec = (REPO_ROOT / "assets" / "dexhand2" / "meta" / "dexhand2_spec.yaml").read_text(
        encoding="utf-8"
    )
    assert "lin_vel_jitter_mps" not in spec
    assert "command_latency_ms: REQUIRED_INPUT" in spec


def test_target_motion_jitter_is_not_a_root_push_or_physical_hold() -> None:
    push_names = [c["name"] for c in axis_aligned_linvel_extrema_mps()]
    ang_names = [c["name"] for c in axis_aligned_angvel_extrema_rad_s()]
    qpos_names = [c["name"] for c in default_joint_pos_offset_extrema()]
    com_names = [c["name"] for c in base_com_offset_extrema()]
    friction_names = [c["name"] for c in static_friction_extrema()]
    jit_names = [c["name"] for c in target_motion_joint_jitter_extrema()]
    pos_names = [c["name"] for c in target_motion_pos_jitter_extrema()]
    ori_names = [c["name"] for c in target_motion_ori_jitter_extrema()]
    lin_names = [c["name"] for c in target_motion_lin_vel_jitter_extrema()]
    angvel_jit_names = [c["name"] for c in target_motion_ang_vel_jitter_extrema()]
    assert set(jit_names).isdisjoint(push_names)
    assert set(jit_names).isdisjoint(qpos_names)
    assert set(jit_names).isdisjoint(com_names)
    assert set(jit_names).isdisjoint(friction_names)
    assert set(pos_names).isdisjoint(push_names)
    assert set(ori_names).isdisjoint(ang_names)
    assert set(pos_names).isdisjoint(com_names)
    assert set(lin_names).isdisjoint(push_names)
    assert set(lin_names).isdisjoint(pos_names)
    assert set(angvel_jit_names).isdisjoint(ang_names)
    assert set(angvel_jit_names).isdisjoint(ori_names)


def test_recorded_only_physical_loads_from_domain_rand_and_is_not_a_mujoco_channel() -> None:
    raw = load_table_s4()
    assert tuple(raw["recorded_only_physical"]) == RECORDED_ONLY_PHYSICAL_FIELDS
    assert physical_dynamic_friction_range() == (0.3, 1.2)
    assert physical_restitution_range() == (0.0, 0.5)
    info = recorded_only_physical()
    assert set(info) == {"dynamic_friction", "restitution"}
    assert info["dynamic_friction"]["range"] == [0.3, 1.2]
    assert info["restitution"]["range"] == [0.0, 0.5]
    assert info["dynamic_friction"]["mujoco_channel"] is None
    assert info["restitution"]["mujoco_channel"] is None
    assert info["dynamic_friction"]["not_pad_cardboard"] is True
    assert info["restitution"]["not_dexhand2_contact"] is True
    recipe = load_ppo_recipe()
    physical = recipe["domain_rand"]["physical"]
    assert physical["dynamic_friction"] == [0.3, 1.2]
    assert physical["restitution"] == [0.0, 0.5]
    cfg = yaml.safe_load(
        (REPO_ROOT / "eval" / "configs" / "l2_freebase_push.yaml").read_text(encoding="utf-8")
    )
    recorded = cfg["recorded_only_physical"]
    assert recorded["fields"] == list(RECORDED_ONLY_PHYSICAL_FIELDS)
    assert recorded["dynamic_friction"] == [0.3, 1.2]
    assert recorded["restitution"] == [0.0, 0.5]
    assert recorded["mujoco_channel"] is None
    spec = (REPO_ROOT / "assets" / "dexhand2" / "meta" / "dexhand2_spec.yaml").read_text(
        encoding="utf-8"
    )
    assert "friction_vs_cardboard_dynamic: REQUIRED_INPUT" in spec
    assert "dynamic_friction: [0.3, 1.2]" not in spec
    assert "restitution: [0.0, 0.5]" not in spec


def test_refuse_recorded_only_physical_map_always_raises() -> None:
    with pytest.raises(RecordedOnlyPhysicalMapError, match="dynamic_friction"):
        refuse_dynamic_friction_map()
    with pytest.raises(RecordedOnlyPhysicalMapError, match="restitution"):
        refuse_restitution_solref_map()
    with pytest.raises(RecordedOnlyPhysicalMapError, match="solref"):
        refuse_recorded_only_physical_map("restitution")
    with pytest.raises(ValueError, match="not a recorded-only"):
        refuse_recorded_only_physical_map("static_friction")


def test_geom_contact_scanner_allows_slide_and_restore_forbids_solref() -> None:
    allowed = """
model.geom_friction[gid, 0] = mu
model.geom_friction[:] = backup
"""
    assert geom_contact_write_hits(allowed) == []
    solref = "model.geom_solref[gid] = [0.02, 1.0]\n"
    assert any("geom_solref" in hit for hit in geom_contact_write_hits(solref))
    solimp = "self.model.geom_solimp = other\n"
    assert any("geom_solimp" in hit for hit in geom_contact_write_hits(solimp))
    spin = "model.geom_friction[gid, 1] = 0.3\n"
    assert any("geom_friction_non_slide" in hit for hit in geom_contact_write_hits(spin))
    mud = "model.geom_friction[:, 1:] = 1.2\n"
    assert any("geom_friction_non_slide" in hit for hit in geom_contact_write_hits(mud))


def test_sim_eval_wbc_do_not_write_solref_or_dynamic_friction() -> None:
    hits: list[str] = []
    for root in _SCAN_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            rel = str(path.relative_to(REPO_ROOT))
            hits.extend(geom_contact_write_hits(text, filename=rel))
    assert hits == []
