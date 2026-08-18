"""SONIC 8-frame planner cross-fade + replan timer. No ONNX, no simulator."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from interface.schema import CommandVector
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import (
    G1_PLANNER_QPOS_DIM,
    PLANNER_BLEND_FRAMES,
    PLANNER_BOXING_MODES,
    PLANNER_CRAWLING_MODE,
    PLANNER_DT_S,
    PLANNER_REPLAN_INTERVAL_CRAWLING_S,
    PLANNER_REPLAN_INTERVAL_DEFAULT_S,
    PLANNER_REPLAN_INTERVAL_RUNNING_S,
    PLANNER_RUNNING_MODE,
    PLANNER_STATIC_MODES,
    t800_planner_qpos_dim,
)
from wbc.planner_blend import (
    PLANNER_BLEND_YAML,
    MovementState,
    PlannerBlendError,
    PlannerReplanClock,
    blend_weight,
    cross_fade_qpos,
    default_last_movement_state,
    evaluate_replan,
    finite_diff_dq,
    is_static_motion_mode,
    load_planner_blend_cfg,
    movement_from_command,
    refuse_command_schema_as_qpos,
    refuse_idle_readapt,
    refuse_run_blend_onnx,
    replan_interval_s,
)
from wbc.planner_onnx import HEIGHT_DISABLED, PlannerOnnxBlocked, PlannerOnnxError, PlannerQpos, pack_qpos


def _row(*, q0: float = 0.0, x_m: float = 0.0, yaw_rad: float = 0.0) -> np.ndarray:
    q = np.zeros(25)
    q[0] = q0
    half = yaw_rad / 2.0
    return pack_qpos(
        PlannerQpos(
            root_pos_m=np.array([x_m, 0.0, 1.03]),
            root_rot_wxyz=np.array([np.cos(half), 0.0, 0.0, np.sin(half)]),
            q_rad=q,
        )
    )


def _clip(*, n: int, q0: float = 0.0, x0: float = 0.0, dx: float = 0.01) -> np.ndarray:
    return np.stack([_row(q0=q0 + 0.01 * i, x_m=x0 + dx * i) for i in range(n)], axis=0)


def _state(
    *,
    mode: int = 2,
    speed: float = 0.35,
    move: tuple[float, float, float] = (1.0, 0.0, 0.0),
    face: tuple[float, float, float] = (1.0, 0.0, 0.0),
    height: float = HEIGHT_DISABLED,
) -> MovementState:
    return MovementState(
        locomotion_mode=mode,
        movement_direction=np.array(move),
        facing_direction=np.array(face),
        movement_speed=speed,
        height=height,
    )


def test_yaml_locks_official_table() -> None:
    cfg = load_planner_blend_cfg()
    assert cfg["adr"] == "ADR-047"
    assert cfg["blend_num_frames"] == PLANNER_BLEND_FRAMES == 8
    assert cfg["not_onnx_run"] is True
    assert cfg["not_g1_planner_onnx"] is True
    assert cfg["not_interpolator_substitute"] is True
    assert cfg["not_idle_readapt"] is True
    assert cfg["replan_interval_s"]["running"] == PLANNER_REPLAN_INTERVAL_RUNNING_S
    assert cfg["replan_interval_s"]["crawling"] == PLANNER_REPLAN_INTERVAL_CRAWLING_S
    assert set(cfg["static_modes"]) == set(PLANNER_STATIC_MODES)
    assert cfg["crawling_modes"] == [PLANNER_CRAWLING_MODE]
    assert 14 not in cfg["crawling_modes"]
    assert 9 not in cfg["boxing_modes"] and 10 not in cfg["boxing_modes"]
    assert list(cfg["loco_mode_map"].values()) == [0, 1, 2]


def test_intervals_and_static_set() -> None:
    assert replan_interval_s(PLANNER_RUNNING_MODE) == 0.1
    assert replan_interval_s(PLANNER_CRAWLING_MODE) == 0.2
    assert replan_interval_s(11) == 1.0
    assert replan_interval_s(14) == PLANNER_REPLAN_INTERVAL_DEFAULT_S  # elbow crawl is default
    assert replan_interval_s(2) == 1.0  # walk
    assert is_static_motion_mode(0)
    assert is_static_motion_mode(9)
    assert not is_static_motion_mode(2)
    assert not is_static_motion_mode(3)
    for m in PLANNER_BOXING_MODES:
        assert replan_interval_s(m) == 1.0


def test_command_schema_fast_walk_is_not_run() -> None:
    cmd = CommandVector.zeros()
    cmd.loco_mode = 2
    cmd.nav_cmd[:] = [0.35, 0.0, 0.0]
    st = movement_from_command(cmd, yaw_world_rad=0.0)
    assert st.locomotion_mode == 2
    assert st.movement_speed == pytest.approx(0.35)
    assert st.height == pytest.approx(HEIGHT_DISABLED)
    assert replan_interval_s(st.locomotion_mode) == 1.0
    assert replan_interval_s(st.locomotion_mode) != 0.1


def test_category1_always_replans() -> None:
    last = default_last_movement_state()
    walk = _state()
    d = evaluate_replan(last, walk, 0.0)
    assert d.need_replan and d.mode_changed
    idle = _state(mode=0, speed=0.0)
    squat = _state(mode=4, speed=0.0, height=0.5)
    d2 = evaluate_replan(idle, squat, 0.0)
    assert d2.need_replan and d2.height_changed and d2.mode_changed
    face = _state(face=(0.0, 1.0, 0.0))
    d3 = evaluate_replan(walk, face, 0.0)
    assert d3.need_replan and d3.facing_changed


def test_static_ignores_timer_and_speed() -> None:
    idle = _state(mode=0, speed=0.0)
    idle_speed = _state(mode=0, speed=0.4)
    d = evaluate_replan(idle, idle_speed, 0.9)
    assert d.under_static
    assert d.time_to_replan  # counter 0.9+0.1 >= 1.0, still reset
    assert not d.need_replan
    assert d.counter_s == pytest.approx(0.0)


def test_walk_timer_requires_nonzero_speed() -> None:
    still = _state(mode=2, speed=0.0)
    d0 = evaluate_replan(still, still, 0.9)
    assert d0.time_to_replan
    assert not d0.need_replan
    moving = _state(mode=2, speed=0.35)
    d1 = evaluate_replan(moving, moving, 0.9)
    assert d1.need_replan
    assert "timer" in d1.reasons


def test_run_replans_every_tick_when_moving() -> None:
    run = _state(mode=PLANNER_RUNNING_MODE, speed=1.0)
    d = evaluate_replan(run, run, 0.0)
    assert d.interval_s == 0.1
    assert d.time_to_replan
    assert d.need_replan


def test_crawling_interval_two_ticks() -> None:
    crawl = _state(mode=PLANNER_CRAWLING_MODE, speed=0.2)
    d0 = evaluate_replan(crawl, crawl, 0.0)
    assert d0.interval_s == 0.2
    assert not d0.time_to_replan
    assert not d0.need_replan
    d1 = evaluate_replan(crawl, crawl, d0.counter_s)
    assert d1.time_to_replan
    assert d1.need_replan


def test_clock_updates_last_only_on_replan() -> None:
    clock = PlannerReplanClock(last=_state(mode=2, speed=0.0), counter_s=0.0)
    still = _state(mode=2, speed=0.0)
    d = clock.tick(still)
    assert not d.need_replan
    assert clock.last.movement_speed == pytest.approx(0.0)
    moving = _state(mode=2, speed=0.35)
    d2 = clock.tick(moving)
    assert d2.need_replan
    assert clock.last.movement_speed == pytest.approx(0.35)


def test_blend_weights_match_cpp() -> None:
    assert blend_weight(0, 0) == pytest.approx(0.0)
    assert blend_weight(7, 0) == pytest.approx(7 / 8)
    assert blend_weight(8, 0) == pytest.approx(1.0)
    assert blend_weight(2, 2) == pytest.approx(0.0)
    assert blend_weight(1, 2) == pytest.approx(0.0)
    with pytest.raises(PlannerBlendError, match="8"):
        blend_weight(0, 0, n_frames=4)


def test_first_copy_and_late_skip() -> None:
    new = _clip(n=10)
    first = cross_fade_qpos(None, new, current_frame=0)
    assert first.first_copy and not first.skipped
    assert first.current_frame == 0
    np.testing.assert_allclose(first.qpos, new)
    old = _clip(n=4, x0=0.0)
    late = cross_fade_qpos(old, _clip(n=2), current_frame=10, gen_frame=0)
    assert late.skipped
    assert late.new_anim_length <= 0
    np.testing.assert_allclose(late.qpos, old)


def test_cross_fade_alignment_and_slerp() -> None:
    old = np.stack([_row(q0=0.0, x_m=0.0) for _ in range(20)], axis=0)
    new = np.stack([_row(q0=1.0, x_m=1.0) for _ in range(20)], axis=0)
    old_dq = np.zeros((20, 25))
    new_dq = np.ones((20, 25))
    cur, fgen = 5, 7
    got = cross_fade_qpos(
        old, new, current_frame=cur, gen_frame=fgen, old_dq=old_dq, new_dq=new_dq
    )
    assert not got.skipped
    assert got.current_frame == 0
    assert got.blend_start_frame == 2
    assert got.new_anim_length == fgen - cur + 20
    assert got.weights_new[0] == pytest.approx(0.0)
    assert got.weights_new[2] == pytest.approx(0.0)
    assert got.weights_new[3] == pytest.approx(1 / 8)
    assert got.weights_new[10] == pytest.approx(1.0)
    # f=2 is blend start: still fully old (q0=0, x=0)
    assert got.qpos[2, 7] == pytest.approx(0.0)
    assert got.qpos[2, 0] == pytest.approx(0.0)
    # f=3: 7/8 old + 1/8 new
    assert got.qpos[3, 7] == pytest.approx(0.125)
    assert got.dq[3, 0] == pytest.approx(0.125)
    # fully new
    assert got.qpos[10, 7] == pytest.approx(1.0)
    assert got.qpos[10, 0] == pytest.approx(1.0)


def test_g1_qpos_and_command_schema_refused() -> None:
    with pytest.raises(G1CheckpointIncompatible):
        cross_fade_qpos(None, np.zeros((8, G1_PLANNER_QPOS_DIM)), current_frame=0)
    cmd = CommandVector.zeros()
    with pytest.raises(PlannerOnnxError, match="interpolators"):
        refuse_command_schema_as_qpos(cmd.to_flat_vector())
    with pytest.raises(PlannerBlendError, match="ADAPTING"):
        refuse_idle_readapt()
    with pytest.raises(G1CheckpointIncompatible):
        refuse_run_blend_onnx("planner_sonic.onnx")
    with pytest.raises(PlannerOnnxBlocked, match="32"):
        refuse_run_blend_onnx()


def test_finite_diff_stays_50hz() -> None:
    rows = np.stack([_row(q0=0.0), _row(q0=0.5)], axis=0)
    dq = finite_diff_dq(rows)
    assert dq.shape == (2, 25)
    assert dq[0, 0] == pytest.approx(0.5 * 50)
    with pytest.raises(PlannerBlendError, match="50"):
        finite_diff_dq(rows, hz=30)


def test_yaml_flag_required(tmp_path) -> None:
    raw = yaml.safe_load(PLANNER_BLEND_YAML.read_text(encoding="utf-8"))
    raw["not_onnx_run"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PlannerBlendError, match="not_onnx_run"):
        load_planner_blend_cfg(path)


def test_t800_qpos_dim() -> None:
    assert t800_planner_qpos_dim() == 32
    assert PLANNER_DT_S == 0.1
