"""Sphere-fit pad collision that replaces official Beta 2 convex hulls."""

from __future__ import annotations

import numpy as np
import pytest

from assets.dexhand2.build.gen_derived import (
    MAX_COLLISION_GEOMS,
    MAX_SITE_TO_SURFACE_M,
    N_SPHERES,
    PAD_FINGERS,
    count_collision_geoms,
    fit_pad_spheres,
    generate,
    site_to_nearest_sphere_surface_m,
)
from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM


def test_fit_pad_spheres_synthetic_distal_bias() -> None:
    rng = np.random.default_rng(0)
    n = 800
    z = rng.uniform(-0.030, -0.006, n)
    xy = rng.normal(size=(n, 2)) * 0.004
    pts = np.column_stack([xy, z])
    site = np.array([0.0, 0.0, -0.0295])
    spheres = fit_pad_spheres(pts, site_m=site)
    assert len(spheres) == N_SPHERES
    assert all(r > 0 and np.isfinite(r) for _, r in spheres)
    zs = [c[2] for c, _ in spheres]
    assert min(zs) < max(zs)
    # Last station is the distal-biased one (most negative z for this cloud).
    assert spheres[-1][0][2] == pytest.approx(min(zs))
    signed = site_to_nearest_sphere_surface_m(site, spheres)
    assert np.isfinite(signed)


def test_beta2_pad_spheres_replace_hull_when_cloned() -> None:
    src = DEFAULT_UPSTREAM / "hand2/hand2_beta2/body/mjcf/right.xml"
    if not src.is_file():
        pytest.skip("wuji-description not cloned")
    report = generate("right", variant="pad_spheres")
    assert report["revision"] == "hand2_beta2"
    assert report["site_gate_ok"], report
    assert report["max_site_to_surface_abs_m"] <= MAX_SITE_TO_SURFACE_M
    assert report["collision"]["n_sphere_collision"] == N_SPHERES * len(PAD_FINGERS)
    assert report["collision"]["n_collision_geoms"] <= MAX_COLLISION_GEOMS
    text = open(report["output"], encoding="utf-8").read()
    # Official pad hulls must no longer collide.
    assert 'mesh="r_thumb_tip_sensor_frame" group="2" contype="0" conaffinity="0"' in text
    assert "r_thumb_tip_sensor_frame_pad_sphere_0" in text
    assert 'condim="4"' in text
    # Distal bone hull still collides on the pad_spheres variant.
    assert 'mesh="r_thumb_distal" group="2"' in text
    assert 'mesh="r_thumb_distal" group="2" contype="0"' not in text
    official = src.read_text(encoding="utf-8")
    official_counts = count_collision_geoms(official)
    # Five pad hulls gone, 15 spheres added.
    assert report["collision"]["n_mesh_collision"] == official_counts["n_mesh_collision"] - 5
    assert report["collision"]["n_sphere_collision"] == 15


def test_beta2_simplified_disables_distal_hulls_when_cloned() -> None:
    src = DEFAULT_UPSTREAM / "hand2/hand2_beta2/body/mjcf/right.xml"
    if not src.is_file():
        pytest.skip("wuji-description not cloned")
    report = generate("right", variant="simplified")
    text = open(report["output"], encoding="utf-8").read()
    assert 'mesh="r_index_finger_distal" group="2" contype="0" conaffinity="0"' in text
    assert "r_index_finger_tip_sensor_frame_pad_sphere_0" in text
    assert report["collision"]["n_mesh_collision"] < 21
    assert report["collision"]["n_collision_geoms"] <= MAX_COLLISION_GEOMS
    assert report["site_gate_ok"]
