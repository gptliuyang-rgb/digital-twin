"""Hand-only MuJoCo env. Combined T800+Hand remains PolicyEvalBlocked."""

from __future__ import annotations

import pytest

from assets.combined.assemble import PolicyEvalBlocked
from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM
from sim.mujoco_env.hand_env import refuse_combined_robot, resolve_hand_xml

OFFICIAL = DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/mjcf/right.xml"


def test_combined_robot_refused() -> None:
    with pytest.raises(PolicyEvalBlocked):
        refuse_combined_robot()


@pytest.mark.skipif(not OFFICIAL.is_file(), reason="wuji-description not cloned")
def test_resolve_derived_xml() -> None:
    pytest.importorskip("mujoco")
    path = resolve_hand_xml("right", derived=True, simplified=False)
    assert path.is_file()
    xml = path.read_text(encoding="utf-8")
    assert "<motor " in xml
    assert xml.count('type="sphere"') == 15


@pytest.mark.skipif(not OFFICIAL.is_file(), reason="wuji-description not cloned")
def test_hold_zero_no_blowup() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco_env.hand_env import HandOnlyMujocoEnv

    env = HandOnlyMujocoEnv(side="right", derived=True, simplified=False)
    env.reset()
    stats = env.hold(0.5)
    assert stats["qvel_rms_rad_s"] < 50.0
    assert env.model.nu == 20
