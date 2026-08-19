from __future__ import annotations

import re

import numpy as np
import pytest

from assets.dexhand2.build.gen_derived import (
    DERIVED,
    FITTED_YAML,
    generate,
    parse_inertial_mass_kg,
    to_meshfree_mit,
    write_fitted_yaml,
)
from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, ingest, parse_collision_audit
from assets.objects.boxes import sample_box
from assets.objects.flex import split_flex_mjcf


def _has_official() -> bool:
    return (DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/mjcf/right.xml").is_file()


@pytest.mark.skipif(not _has_official(), reason="wuji-description not cloned")
def test_ingest_official_collision_gaps() -> None:
    report = ingest()
    assert report["ok"], report["mismatches"]
    assert report["n_actuators"] == 20
    assert report["n_joints"] == 20
    assert len(report["sites"]) == 5
    coll = report["collision"]
    assert coll["n_contact_excludes"] == 10
    assert coll["n_tip_stl"] >= 5
    assert coll["n_pad_spheres"] == 0
    assert coll["tip_stl_used_as_collision"] is False


@pytest.mark.skipif(not _has_official(), reason="wuji-description not cloned")
def test_derived_pad_spheres_fit() -> None:
    report = generate("right", mit_motors=True, simplified=False)
    assert report["n_pad_spheres"] == 15
    assert report["gates"]["n_pad_spheres_ok"]
    xml = (DERIVED / "right_with_pad_spheres_mit.xml").read_text(encoding="utf-8")
    assert xml.count('type="sphere"') == 15
    assert xml.count("<motor ") == 20
    assert "<position " not in xml
    assert re.search(r'mesh="r_index_finger_distal" group="2"\s+contype="0"', xml)
    palmar = [v["palmar_vertex_to_sphere_surface_m"] for v in report["fingers"].values()]
    assert max(palmar) < 0.002 + 1e-6
    official = [v["official_site_to_distal_vertex_m"] for v in report["fingers"].values()]
    # Official site already sits on the distal hull (~0.1 mm). Pad spheres are palmar.
    assert max(official) < 0.002
    write_fitted_yaml([report])
    assert FITTED_YAML.is_file()


@pytest.mark.skipif(not _has_official(), reason="wuji-description not cloned")
def test_derived_left_pad_spheres_not_a_mirror() -> None:
    left = generate("left", mit_motors=True, simplified=False)
    right = generate("right", mit_motors=True, simplified=False)
    assert left["n_pad_spheres"] == 15
    assert left["gates"]["palmar_fit_ok"]
    write_fitted_yaml([left, right])
    ly = left["fingers"]["index_finger"]["spheres"][0]["pos_m"][1]
    ry = right["fingers"]["index_finger"]["spheres"][0]["pos_m"][1]
    # Palmar pulp is on opposite Y for left vs right distal frames.
    assert ly * ry < 0 or abs(ly - ry) > 1e-4


@pytest.mark.skipif(not _has_official(), reason="wuji-description not cloned")
def test_simplified_capsule_budget() -> None:
    report = generate("right", mit_motors=True, simplified=True)
    audit = parse_collision_audit(DERIVED / "right_simplified_mit.xml")
    assert report["n_pad_spheres"] == 15
    assert audit["n_colliding_geoms"] <= 60
    assert audit["n_pad_spheres"] == 15


@pytest.mark.skipif(not _has_official(), reason="wuji-description not cloned")
def test_meshfree_mit_keeps_cad_inertias_drops_position_and_meshes() -> None:
    src = DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/mjcf/right.xml"
    raw = src.read_text(encoding="utf-8")
    assert 'timestep="0.002"' in raw
    xml = to_meshfree_mit(raw)
    assert xml.count("<motor ") == 20
    assert "<position " not in xml
    assert 'timestep="0.001"' in xml
    assert 'timestep="0.002"' not in xml
    assert 'type="mesh"' not in xml
    from interface.schema import load_hand_spec

    spec = load_hand_spec()
    assert abs(parse_inertial_mass_kg(xml) - spec.skeleton_mass_kg) < 1e-3


def test_flex_box_splits_when_large() -> None:
    rng = np.random.default_rng(0)
    box = sample_box(rng)
    xml = split_flex_mjcf(box)
    if box.split_flex:
        assert xml.count("<body ") == 5
        assert "type=\"slide\"" in xml
    else:
        assert xml.count("<body ") == 1
