"""Derived pad-sphere MJCF for official Hand 2.2."""

from __future__ import annotations

from assets.dexhand2.build.gen_derived import SITE_TO_SURFACE_GATE_M, generate
from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM


def test_pad_spheres_replace_official_hulls() -> None:
    src = DEFAULT_UPSTREAM / "hand2/hand2_2/body/mjcf/right.xml"
    if not src.is_file():
        import pytest

        pytest.skip("wuji-description not cloned")
    path, metrics = generate("right", simplified=False)
    text = path.read_text(encoding="utf-8")
    assert metrics["n_spheres"] == 15
    assert metrics["gate_ok"], metrics["site_to_surface_m"]
    assert metrics["worst_site_to_surface_m"] <= SITE_TO_SURFACE_GATE_M
    assert metrics["site_to_surface_m"]["r_thumb_tip"] <= SITE_TO_SURFACE_GATE_M
    assert text.count('condim="4"') == 15
    assert text.count("_tip_pad_") == 15
    # Official pad hulls are still present as visuals but do not collide.
    assert text.count('mesh="r_thumb_tip_sensor_frame" group="2" contype="0" conaffinity="0"') == 1
    assert 'mesh="r_thumb_tip_sensor_frame" group="2" />' not in text


def test_simplified_drops_distal_hulls() -> None:
    src = DEFAULT_UPSTREAM / "hand2/hand2_2/body/mjcf/right.xml"
    if not src.is_file():
        import pytest

        pytest.skip("wuji-description not cloned")
    path, metrics = generate("right", simplified=True)
    text = path.read_text(encoding="utf-8")
    assert metrics["n_spheres"] == 15
    assert 'mesh="r_index_finger_distal" group="2" contype="0" conaffinity="0"' in text
    assert path.name == "right_simplified.xml"
