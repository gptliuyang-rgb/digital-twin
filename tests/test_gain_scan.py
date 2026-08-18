"""Gain-scan grid. Grasp-success must stay JSON null."""

from __future__ import annotations

import pytest

from eval.gain_scan import gain_cells, run_hold_scan
from interface.schema import load_hand_spec
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key


def test_gain_cells_are_3x3() -> None:
    cells = gain_cells()
    assert len(cells) == 9
    scales = {(c["kp_scale"], c["kv_scale"]) for c in cells}
    assert (1.0, 1.0) in scales
    assert (0.5, 2.0) in scales
    spec = load_hand_spec()
    assert len(cells[0]["kp"]) == spec.n_active_dof
    assert cells[4]["status"] == "uncalibrated_carried_from_wuji_hand_gen1"


def test_hold_scan_never_reports_grasp_success() -> None:
    report = run_hold_scan(seconds=0.2)
    assert report["grasp_success_rate"] is None
    assert report["n_cells"] == 9
    refuse_grasp_success_key(report)
    assert report["status"] in {
        "skipped_no_mujoco",
        "skipped_no_official_mjcf",
        "uncalibrated_gain_scan",
    }
    if report["status"] == "uncalibrated_gain_scan":
        assert len(report["cells"]) == 9
        assert report["qvel_rms_rad_s_max"] >= report["qvel_rms_rad_s_min"]
        pytest.importorskip("mujoco")
