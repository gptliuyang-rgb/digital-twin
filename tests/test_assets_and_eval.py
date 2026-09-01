from __future__ import annotations

import numpy as np
import pytest

from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM, ingest, ingest_all
from assets.objects.boxes import sample_box
from data.build_modality import build_modality
from interface.schema import load_hand_spec
from sim.payload import Payload, sample_payload
from sim.qr_scanner import decode_image, make_qr_png, simulate_scan


def test_ingest_official_when_cloned() -> None:
    if not (DEFAULT_UPSTREAM / "hand2/hand2_beta2/body/mjcf/right.xml").is_file():
        pytest.skip("wuji-description not cloned")
    reports = ingest_all()
    assert reports["hand2_beta1"]["ok"], reports["hand2_beta1"]["mismatches"]
    assert reports["hand2_beta2"]["ok"], reports["hand2_beta2"]["mismatches"]
    b1 = reports["hand2_beta1"]
    b2 = reports["hand2_beta2"]
    assert b1["n_actuators"] == 20
    assert b1["n_pad_bodies"] == 0
    assert b1["sim_mass_kg"] == pytest.approx(0.6207)
    assert b1["pad_collision_in_official_model"] is False
    assert b2["n_actuators"] == 20
    assert b2["n_pad_bodies"] == 5
    assert b2["n_bodies"] == 26
    assert b2["sim_mass_kg"] == pytest.approx(0.6228)
    assert b2["pad_collision_in_official_model"] is True
    default = ingest()
    assert default["revision"] == "hand2_beta2"
    assert default["ok"]


def test_modality_tracks_joint_order() -> None:
    spec = load_hand_spec()
    mod = build_modality(spec)
    assert mod["joint_order"] == spec.joint_order
    assert mod["n_active_dof"] == 20
    assert mod["action"]["left_hand_q"]["end"] - mod["action"]["left_hand_q"]["start"] == 20
    assert mod["action_dim"] == 75
    assert mod["sim_model_revision"] == "hand2_beta2"
    assert mod["state"]["tactile"]["thumb_points"] == 40
    assert mod["state"]["tactile"]["other_finger_points"] == 34


def test_payload_bounds() -> None:
    p = sample_payload(np.random.default_rng(0), "left")
    assert 0 <= p.mass_kg <= 20
    with pytest.raises(ValueError):
        Payload(mass_kg=21, com_offset_m=[0, 0, 0], side="left")


def test_box_mass_gate() -> None:
    rng = np.random.default_rng(3)
    box = sample_box(rng)
    assert 2 <= box.mass_kg <= 20


def test_qr_encode_roundtrip() -> None:
    img = make_qr_png("BOX-TEST-001", box_size=12)
    arr = np.asarray(img)
    decoded = decode_image(arr)
    if decoded is None:
        pytest.skip("no OpenCV/pyzbar decoder in this environment")
    assert decoded == "BOX-TEST-001"


def test_scan_geometry_gate() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    far = simulate_scan(
        rgb,
        gun_tcp_pos_m=np.zeros(3),
        gun_tcp_z=np.array([0, 0, 1.0]),
        qr_pos_m=np.array([0, 0, 2.0]),
        qr_normal=np.array([0, 0, -1.0]),
        rel_speed_m_s=0.0,
    )
    assert not far.geometry_ok
    assert far.payload is None
