from __future__ import annotations

from pathlib import Path

import pytest

from assets.engineai.build.ingest_t800 import T800_URDF, T800PRO_URDF, build


@pytest.mark.skipif(not T800_URDF.is_file(), reason="engineai native SDK not cloned")
def test_t800_revolute_count() -> None:
    doc = build()
    assert doc["t800"]["revolute_count"] == 25
    assert "J18_WRIST_PITCH_L" not in doc["t800"]["groups"].get("wrist", [])
    assert doc["end_effector_frames"]["left_wrist"] == "LINK_WRIST_END_L"


@pytest.mark.skipif(not T800PRO_URDF.is_file(), reason="T800 Pro URDF missing")
def test_t800pro_body_dof_excluding_hands() -> None:
    doc = build()
    pro = doc["t800pro"]
    assert pro is not None
    # 43 revolute total; 14 are the built-in 7-DoF hands we replace.
    assert pro["revolute_count"] == 43
    assert pro["revolute_excluding_builtin_hands"] == 29
    wrists = pro["groups"]["wrist"]
    assert "J18_WRIST_PITCH_L" in wrists
    assert "J33_WRIST_ROLL_R" in wrists
    assert Path(pro["urdf"]).as_posix().endswith("serial_t800pro.urdf")
