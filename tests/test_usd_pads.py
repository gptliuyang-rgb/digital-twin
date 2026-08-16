from __future__ import annotations

import pytest

from assets.dexhand2.build.gen_derived import FINGERS, FITTED_YAML
from assets.dexhand2.build.gen_usd_pads import generate, usda_for_hand


def test_usda_has_fifteen_spheres_for_right() -> None:
    text = usda_for_hand("right")
    assert text.count("def Sphere") == 15
    assert 'over "r_index_finger_distal"' in text
    assert "PhysicsCollisionAPI" in text
    for finger in FINGERS:
        assert f'over "r_{finger}_distal"' in text
    assert "REQUIRED_INPUT" not in text


def test_usda_refuses_to_mirror_missing_left() -> None:
    with pytest.raises(KeyError, match="left"):
        usda_for_hand("left")


def test_generate_writes_derived(tmp_path, monkeypatch) -> None:
    from assets.dexhand2.build import gen_usd_pads as mod

    monkeypatch.setattr(mod, "DERIVED", tmp_path)
    out = generate("right")
    assert out.is_file()
    assert out.read_text(encoding="utf-8").count("def Sphere") == 15
    assert FITTED_YAML.is_file()
