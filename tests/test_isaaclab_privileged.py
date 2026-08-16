from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from assets.objects.boxes import sample_box
from sim.isaaclab_env.privileged import (
    IsaacLabUnavailable,
    PrivilegedIsaacCfg,
    PrivilegedIsaacEnv,
    privileged_report,
    refuse_combined_robot,
)
from sim.isaaclab_env.scene_spec import PalletBoxSceneSpec
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key


def test_cfg_has_no_hands_and_null_grasp() -> None:
    cfg = PrivilegedIsaacCfg()
    payload = cfg.to_dict()
    assert payload["include_hands"] is False
    assert payload["include_t800"] is False
    assert payload["sensors"] == "privileged_state_only"
    assert payload["grasp_success_rate"] is None
    refuse_combined_robot(cfg)


def test_refuse_combined_flag() -> None:
    with pytest.raises(PolicyEvalBlocked):
        refuse_combined_robot(PrivilegedIsaacCfg(include_t800=True))


def test_privileged_report_never_numeric_grasp() -> None:
    report = privileged_report()
    assert report["grasp_success_rate"] is None
    assert report["uncalibrated"] is True
    assert report["isaaclab_loaded"] is False
    refuse_grasp_success_key(report)


def test_env_constructor_without_isaaclab() -> None:
    with pytest.raises(IsaacLabUnavailable):
        PrivilegedIsaacEnv()


def test_scene_spec_matches_privileged_l2_and_null_grasp() -> None:
    spec = PalletBoxSceneSpec()
    payload = spec.to_dict()
    assert payload["include_hands"] is False
    assert payload["include_t800"] is False
    assert payload["grasp_success_rate"] is None
    xml = spec.mujoco_xml(sample_box(np.random.default_rng(0)))
    assert "pallet" in xml
    assert "box" in xml
    assert "DexHand" not in xml
    refuse_grasp_success_key(payload)


def test_reset_step_are_wired_through_runtime() -> None:
    assert PrivilegedIsaacEnv.reset is not object.__dict__.get("reset")
    assert callable(PrivilegedIsaacEnv.reset)
    assert callable(PrivilegedIsaacEnv.step)
    # Without Isaac Lab the constructor still fails before reset; that is required.
