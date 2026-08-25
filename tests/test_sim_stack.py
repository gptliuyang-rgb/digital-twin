"""Sim-only stack: pad site alignment, recorded L0, MuJoCo L1, QR envelope."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from assets.dexhand2.build.pad_inject import (
    align_spheres_to_site,
    tip_finger_stem,
)
from eval.l0_offline_replay import evaluate_recorded_commands
from interface.schema import command_dim, load_hand_spec
from sim.command_from_state import commands_to_chunks
from sim.qr_envelope import geometry_envelope


def test_tip_finger_stem_does_not_eat_r_in_finger() -> None:
    assert tip_finger_stem("r_index_finger_tip", "right") == "index_finger"
    assert tip_finger_stem("r_middle_finger_tip", "right") == "middle_finger"
    assert tip_finger_stem("r_ring_finger_tip", "right") == "ring_finger"
    assert tip_finger_stem("r_thumb_tip", "right") == "thumb"
    assert tip_finger_stem("r_pinky_tip", "right") == "pinky"
    assert tip_finger_stem("l_index_finger_tip", "left") == "index_finger"
    assert tip_finger_stem("r_palm", "right") is None


def test_align_spheres_puts_site_on_surface() -> None:
    site = np.array([0.0, 0.0, -0.025])
    # Pretend STL-frame cluster far from the site (the Wuji tip.STL situation).
    raw = [([0.04, 0.0, 0.01], 0.007), ([0.05, 0.0, 0.01], 0.007), ([0.06, 0.0, 0.01], 0.007)]
    aligned = align_spheres_to_site(raw, site, surface=True)
    centers = np.asarray([c for c, _ in aligned])
    radii = np.asarray([r for _, r in aligned])
    # Palmar offset: centroid should sit one radius along −Y from the site.
    centroid = centers.mean(axis=0)
    np.testing.assert_allclose(centroid, site + np.array([0.0, -1.0, 0.0]) * radii.mean(), atol=1e-9)
    nearest = min(abs(float(np.linalg.norm(c - site)) - r) for c, r in zip(centers, radii, strict=True))
    assert nearest < 0.002


def test_l0_identity_on_recorded_commands() -> None:
    spec = load_hand_spec()
    dim = command_dim(spec)
    rng = np.random.default_rng(0)
    # Stay inside a tiny envelope around zeros so CommandVector isn't needed.
    commands = rng.normal(scale=0.01, size=(48, dim))
    report = evaluate_recorded_commands(commands, spec, horizon=16)
    assert report["mode"] == "recorded_sim_identity"
    assert report["mse_mean"] == pytest.approx(0.0)
    assert report["left_hand_outliers"] == []
    assert report["n_chunks"] == 3
    chunks = commands_to_chunks(commands, 16)
    assert chunks.shape == (3, 16, dim)


def test_qr_geometry_envelope_has_in_and_out_of_range_cells() -> None:
    report = geometry_envelope()
    assert report["n_cells"] == 9 * 8
    assert 0 < report["n_geometry_ok"] < report["n_cells"]
    assert report["policy_eval_forbidden"] is True
    assert all(c.get("decode_ok") is None for c in report["cells"])


def test_assemble_injects_five_pads_per_hand_when_cloned() -> None:
    from assets.combined.assemble import assemble_mjcf
    from assets.dexhand2.build.ingest_official import official_mjcf
    from assets.engineai.paths import t800_mjcf

    if not official_mjcf("left", with_mount=True).is_file():
        pytest.skip("wuji-description not cloned")
    try:
        have_t800 = t800_mjcf().is_file()
    except FileNotFoundError:
        have_t800 = False
    if not have_t800:
        pytest.skip("T800 tree not cloned")
    xml, _manifest = assemble_mjcf(pad_spheres=True)
    for side, fingers in (
        ("l", ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"]),
        ("r", ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"]),
    ):
        for finger in fingers:
            token = f"{side}_{finger}_tip_pad_0"
            assert token in xml, token


def test_sim_stack_quick_when_cloned(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from assets.dexhand2.build.ingest_official import official_mjcf
    from assets.engineai.paths import t800_mjcf
    from eval.run_sim_stack import run_sim_stack

    try:
        have_t800 = t800_mjcf().is_file()
    except FileNotFoundError:
        have_t800 = False
    if not official_mjcf("left", with_mount=True).is_file() or not have_t800:
        pytest.skip("official trees not cloned")

    report = run_sim_stack(
        quick=True,
        with_l2=False,
        steps_per_phase=6,
        out_json=tmp_path / "sim_stack.json",
        out_md=tmp_path / "SIM_STACK.md",
    )
    assert report["ok"], report.get("stages")
    assert report["l0"]["mse_mean"] == pytest.approx(0.0)
    assert report["record"]["n_steps"] > 0
    assert report["replay"]["finite"] is True
    assert "grasp_success_rate" not in str(report)
    assert (tmp_path / "SIM_STACK.md").is_file()
