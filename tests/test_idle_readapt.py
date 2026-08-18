"""SONIC idle-mode ADAPTING/RECOVERING. No ONNX, no simulator."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    G1_N_DOF,
    IDLE_READAPT_ADAPT_STOP_RAD,
    IDLE_READAPT_ADAPT_TRIGGER_RAD,
    IDLE_READAPT_BLEND_KEEP,
    IDLE_READAPT_RECOVER_TRIGGER_RAD,
    PLANNER_IDLE_MODE,
    T800_N_LOWER_BODY_DOF,
)
from wbc.idle_readapt import (
    IDLE_READAPT_YAML,
    IdleReadapt,
    IdleReadaptError,
    IdleReadaptState,
    blend_last_frame,
    load_idle_readapt_cfg,
    mean_abs_lower_body_error,
    refuse_planner_blend_mix,
    refuse_recover_stop,
    refuse_run_idle_readapt_onnx,
    transition,
)
from wbc.planner_onnx import PlannerOnnxBlocked


def _q(*, lower: float = 0.0, upper: float = 0.0) -> np.ndarray:
    q = np.zeros(25)
    q[:T800_N_LOWER_BODY_DOF] = lower
    q[T800_N_LOWER_BODY_DOF:] = upper
    return q


def test_yaml_locks_official_table() -> None:
    cfg = load_idle_readapt_cfg()
    assert cfg["adr"] == "ADR-048"
    assert cfg["not_onnx_run"] is True
    assert cfg["not_planner_blend"] is True
    assert cfg["not_recover_stop"] is True
    assert cfg["k_adapt_trigger_rad"] == IDLE_READAPT_ADAPT_TRIGGER_RAD == 0.10
    assert cfg["k_adapt_stop_rad"] == IDLE_READAPT_ADAPT_STOP_RAD == 0.05
    assert cfg["k_recover_trigger_rad"] == IDLE_READAPT_RECOVER_TRIGGER_RAD == 0.045
    assert cfg["blend_keep"] == IDLE_READAPT_BLEND_KEEP
    assert cfg["n_dof"] == 25
    assert cfg["n_lower_body_dof"] == 12
    assert cfg["idle_mode"] == PLANNER_IDLE_MODE
    assert "k_recover_stop_rad" not in cfg


def test_strict_thresholds() -> None:
    assert transition(IdleReadaptState.IDLE, 0.10) is IdleReadaptState.IDLE
    assert transition(IdleReadaptState.IDLE, 0.10 + 1e-12) is IdleReadaptState.ADAPTING
    assert transition(IdleReadaptState.IDLE, 0.045) is IdleReadaptState.IDLE
    assert transition(IdleReadaptState.IDLE, 0.045 - 1e-12) is IdleReadaptState.RECOVERING
    assert transition(IdleReadaptState.ADAPTING, 0.05) is IdleReadaptState.ADAPTING
    assert transition(IdleReadaptState.ADAPTING, 0.05 - 1e-12) is IdleReadaptState.IDLE
    assert transition(IdleReadaptState.RECOVERING, 0.0) is IdleReadaptState.RECOVERING
    assert transition(IdleReadaptState.RECOVERING, 0.10) is IdleReadaptState.RECOVERING
    assert transition(IdleReadaptState.RECOVERING, 0.10 + 1e-12) is IdleReadaptState.ADAPTING


def test_no_recover_stop() -> None:
    with pytest.raises(IdleReadaptError, match="kRecoverStop"):
        refuse_recover_stop()
    machine = IdleReadapt()
    planner = _q(lower=0.0, upper=0.7)
    motor = _q(lower=0.0, upper=0.7)
    tick = machine.tick(planner, motor, locomotion_mode=0, at_last_frame=True)
    assert tick.state is IdleReadaptState.RECOVERING
    tick2 = machine.tick(tick.q_planner, motor, locomotion_mode=0, at_last_frame=True)
    assert tick2.state is IdleReadaptState.RECOVERING


def test_adapting_blend_and_upper_untouched() -> None:
    planner = _q(lower=0.0, upper=0.4)
    motor = _q(lower=0.20, upper=9.0)
    out = blend_last_frame(
        planner,
        state=IdleReadaptState.ADAPTING,
        q_motor=motor,
        q_original=_q(lower=-1.0),
    )
    expected = 0.98 * 0.0 + 0.02 * 0.20
    assert out[0] == pytest.approx(expected)
    assert out[11] == pytest.approx(expected)
    np.testing.assert_allclose(out[12:], planner[12:])
    idle = blend_last_frame(
        planner, state=IdleReadaptState.IDLE, q_motor=motor, q_original=planner
    )
    np.testing.assert_allclose(idle, planner)


def test_recovering_blend_toward_original() -> None:
    planner = _q(lower=0.10, upper=0.3)
    original = _q(lower=0.0, upper=0.3)
    out = blend_last_frame(
        planner,
        state=IdleReadaptState.RECOVERING,
        q_motor=_q(lower=9.0),
        q_original=original,
    )
    assert out[0] == pytest.approx(0.98 * 0.10 + 0.02 * 0.0)
    np.testing.assert_allclose(out[12:], planner[12:])


def test_first_store_then_adapt_same_tick() -> None:
    machine = IdleReadapt()
    planner = _q(lower=0.0, upper=0.5)
    motor = _q(lower=0.20, upper=0.5)
    assert mean_abs_lower_body_error(planner, motor) == pytest.approx(0.20)
    tick = machine.tick(planner, motor, locomotion_mode=0, at_last_frame=True)
    assert tick.first_store
    assert tick.stored
    assert tick.state is IdleReadaptState.ADAPTING
    assert tick.applied
    assert tick.q_planner[0] == pytest.approx(0.02 * 0.20)
    assert tick.q_planner[12] == pytest.approx(0.5)


def test_skips_walk_mid_clip_and_paused() -> None:
    machine = IdleReadapt()
    q = _q(lower=0.2)
    mid = machine.tick(q, q, locomotion_mode=0, at_last_frame=False)
    assert mid.skipped_reason == "not_last_frame"
    assert not machine.stored
    walk = machine.tick(q, q, locomotion_mode=2, at_last_frame=True)
    assert walk.skipped_reason == "not_idle"
    squat = machine.tick(q, q, locomotion_mode=4, at_last_frame=True)
    assert squat.skipped_reason == "not_idle"
    paused = machine.tick(q, q, locomotion_mode=0, at_last_frame=True, play=False)
    assert paused.skipped_reason == "play_false"


def test_new_clip_clears_store() -> None:
    machine = IdleReadapt()
    q = _q()
    machine.tick(q, q, locomotion_mode=0, at_last_frame=True)
    assert machine.stored
    machine.notify_new_clip()
    assert not machine.stored
    assert machine.original_lower is None


def test_g1_and_hands_refused() -> None:
    machine = IdleReadapt()
    with pytest.raises(G1CheckpointIncompatible):
        machine.tick(np.zeros(G1_N_DOF), np.zeros(G1_N_DOF), locomotion_mode=0, at_last_frame=True)
    with pytest.raises(IdleReadaptError, match="DexHand2"):
        machine.tick(np.zeros(45), np.zeros(45), locomotion_mode=0, at_last_frame=True)
    with pytest.raises(IdleReadaptError, match="32-D qpos"):
        machine.tick(np.zeros(32), np.zeros(32), locomotion_mode=0, at_last_frame=True)
    with pytest.raises(G1CheckpointIncompatible):
        refuse_run_idle_readapt_onnx("planner_sonic.onnx")
    with pytest.raises(PlannerOnnxBlocked, match="25"):
        refuse_run_idle_readapt_onnx()
    with pytest.raises(IdleReadaptError, match="8-frame"):
        refuse_planner_blend_mix()


def test_yaml_flag_required(tmp_path) -> None:
    raw = yaml.safe_load(IDLE_READAPT_YAML.read_text(encoding="utf-8"))
    raw["not_recover_stop"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(IdleReadaptError, match="not_recover_stop"):
        load_idle_readapt_cfg(path)
    raw = yaml.safe_load(IDLE_READAPT_YAML.read_text(encoding="utf-8"))
    raw["k_recover_stop_rad"] = 0.02
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(IdleReadaptError, match="k_recover_stop"):
        load_idle_readapt_cfg(path)
