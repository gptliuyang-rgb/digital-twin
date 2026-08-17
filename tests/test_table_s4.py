"""Table S4 root-push extrema are the PPO domain-rand source, not pad–cardboard."""

from __future__ import annotations

import pytest

from wbc.ppo.recipe import load_ppo_recipe
from wbc.ppo.table_s4 import (
    ANGVEL_AXES,
    SWEEP_AXES,
    VERTICAL_AXES,
    axis_aligned_angvel_extrema_rad_s,
    axis_aligned_linvel_extrema_mps,
    duration_extrema_s,
    force_n_from_impulse,
    load_table_s4,
    root_push_angvel_ranges_rad_s,
    root_push_duration_s,
    root_push_ranges_mps,
    sustained_force_cases,
    sustained_torque_cases,
    torque_nm_from_impulse,
    vertical_force_cases,
    vertical_linvel_extrema_mps,
)


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
