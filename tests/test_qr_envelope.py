from __future__ import annotations

import numpy as np
import pytest

from eval.qr_envelope import heatmap_markdown, scan_envelope
from sim.qr_scanner import decode_image, make_qr_png


def test_qr_roundtrip_ideal() -> None:
    img = np.asarray(make_qr_png("BOX-DT-001", box_size=12))
    decoded = decode_image(img)
    if decoded is None:
        pytest.skip("no OpenCV/pyzbar decoder in this environment")
    assert decoded == "BOX-DT-001"


def test_envelope_ideal_cell_decodes() -> None:
    decoded = decode_image(np.asarray(make_qr_png("BOX-DT-001", box_size=12)))
    if decoded is None:
        pytest.skip("no OpenCV/pyzbar decoder in this environment")
    report = scan_envelope(
        payload="BOX-DT-001",
        distances_m=np.array([0.12]),
        angles_deg=np.array([0.0]),
    )
    assert report["cells"][0][0]["success"]
    md = heatmap_markdown(report)
    assert "✓" in md


def test_envelope_far_or_oblique_fails_geometry() -> None:
    report = scan_envelope(
        payload="BOX-DT-001",
        distances_m=np.array([0.50]),
        angles_deg=np.array([80.0]),
    )
    cell = report["cells"][0][0]
    assert not cell["geometry_ok"]
    assert not cell["success"]
