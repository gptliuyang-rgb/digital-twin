from __future__ import annotations

from assets.objects.pallet import EURO_PALLET_M, pallet_mjcf, support_xy_m
from assets.objects.scanner import scanner_mjcf
from eval.report.generate import render


def test_pallet_and_scanner_xml() -> None:
    xml = pallet_mjcf()
    assert "1.20" not in xml  # half-extents
    assert "0.6000" in xml
    assert EURO_PALLET_M == (1.20, 0.80, 0.144)
    box = support_xy_m()
    assert box[0, 0] == -0.6
    gun = scanner_mjcf()
    assert "scanner_tcp" in gun
    assert 'contype="0"' in gun


def test_report_render_mentions_no_success_rate() -> None:
    md = render(
        {"mode": "synthetic", "mse_mean": 0.01, "left_hand_outliers": [3], "right_hand_outliers": []},
        {"limits": {"all_inside_margin": True}, "coupling": {"ok": True}, "ik": "skipped"},
        {
            "status": "blocked_uncalibrated",
            "uncalibrated": True,
            "combined_eval_allowed": False,
            "warning": "no success rate",
            "gain_scan_cells": [{}] * 9,
            "grasp_success_rate": None,
        },
    )
    assert "blocked_uncalibrated" in md
    assert "null" in md
    assert "0.9" not in md
