"""Table S4 root-push extrema are the PPO domain-rand source, not pad–cardboard."""

from __future__ import annotations

import pytest

from wbc.ppo.recipe import load_ppo_recipe
from wbc.ppo.table_s4 import (
    SWEEP_AXES,
    axis_aligned_linvel_extrema_mps,
    load_table_s4,
    root_push_ranges_mps,
)


def test_table_s4_is_not_dexhand2_contact() -> None:
    raw = load_table_s4()
    assert raw["not_dexhand2_contact"] is True
    assert raw["applies_to"] == "t800_wbc_ppo"
    ranges = root_push_ranges_mps()
    assert ranges["x"] == (-0.5, 0.5)
    assert ranges["y"] == (-0.5, 0.5)
    assert ranges["z"] == (-0.2, 0.2)


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


def test_z_extrema_are_opt_in() -> None:
    cases = axis_aligned_linvel_extrema_mps(axes=("z",))
    assert [c["name"] for c in cases] == ["-z", "+z"]
    assert cases[0]["lin_vel_mps"] == [0.0, 0.0, -0.2]
    assert cases[1]["lin_vel_mps"] == [0.0, 0.0, 0.2]


def test_unknown_axis_raises() -> None:
    with pytest.raises(ValueError, match="unknown Table S4 axis"):
        axis_aligned_linvel_extrema_mps(axes=("w",))
