from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from assets.objects.boxes import sample_box
from sim.mujoco_env.privileged_l2 import (
    combined_robot_blocked,
    pallet_box_mjcf,
    refuse_grasp_success_key,
    uncalibrated_contact,
)


def test_uncalibrated_contact_is_true() -> None:
    assert uncalibrated_contact() is True


def test_refuse_numeric_grasp_success() -> None:
    refuse_grasp_success_key({"grasp_success_rate": None})
    with pytest.raises(AssertionError, match="grasp_success_rate"):
        refuse_grasp_success_key({"grasp_success_rate": 0.9})


def test_combined_robot_blocked() -> None:
    with pytest.raises(PolicyEvalBlocked):
        combined_robot_blocked()


def test_pallet_mjcf_has_no_hand_and_labelled_fixture() -> None:
    box = sample_box(np.random.default_rng(0))
    xml = pallet_box_mjcf(box)
    assert "wuji" not in xml.lower()
    assert "pad" not in xml.lower()
    assert "box_free" in xml
    assert "pallet" in xml


def test_privileged_drop_skips_or_runs() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.privileged_l2 import PrivilegedL2Env

    env = PrivilegedL2Env(rng=np.random.default_rng(1))
    report = env.drop_and_settle(settle_s=0.8)
    assert report["grasp_success_rate"] is None
    assert report["fixture_note"]
    assert "max_drift_m" in report
    refuse_grasp_success_key(report)
