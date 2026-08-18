"""Official SONIC 8-frame planner cross-fade and replan-timer table.

Numpy only — no simulator, no onnxruntime. Does **not** run
``planner_sonic.onnx``. ADR-018 interpolators stay the runtime L1a.

C++ source (NVlabs/GR00T-WholeBodyControl ``g1_deploy_onnx_ref.cpp``):

* ``CurrentFrameAdvancement``: rebase old so ``current_frame`` is 0, align
  new at ``gen_frame - current_frame``, blend 8 frames with
  ``w_new = clamp((f - blend_start) / 8, 0, 1)``. Linear mix on joints /
  root xyz; SLERP on root quat. ``current_frame`` resets to 0.
* ``Planner()`` at 10 Hz: always replan on mode/facing/height change;
  non-static modes also replan on speed/direction change or periodic
  timer with nonzero speed. Intervals: run 0.1 s, crawl 0.2 s, boxing
  1.0 s, else 1.0 s. Crawling is ``LocomotionMode::CRAWLING`` (8) only.

Hands still bypass WBC. ``command_schema`` loco_mode 2 is walk, not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import CommandVector
from vla.adapters.rotation import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    slerp_matrices,
)
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import (
    G1_N_DOF,
    G1_PLANNER_QPOS_DIM,
    PLANNER_BLEND_FRAMES,
    PLANNER_BOXING_MODES,
    PLANNER_CONTROL_HZ,
    PLANNER_CRAWLING_MODE,
    PLANNER_DT_S,
    PLANNER_LOOKAHEAD_STEPS_50HZ,
    PLANNER_N_MODES_V2,
    PLANNER_REPLAN_INTERVAL_BOXING_S,
    PLANNER_REPLAN_INTERVAL_CRAWLING_S,
    PLANNER_REPLAN_INTERVAL_DEFAULT_S,
    PLANNER_REPLAN_INTERVAL_RUNNING_S,
    PLANNER_RUNNING_MODE,
    PLANNER_STATIC_MODES,
    load_t800_sonic,
    planner_qpos_dim,
    t800_planner_qpos_dim,
)
from wbc.planner_onnx import (
    HEIGHT_DISABLED,
    PlannerOnnxBlocked,
    PlannerOnnxError,
    PlannerQpos,
    loco_mode_to_planner_mode,
    nav_to_planner_dirs,
    pack_qpos,
    refuse_g1_planner_onnx,
    refuse_run_planner_onnx,
    unpack_qpos,
)

PLANNER_BLEND_YAML = Path(__file__).with_name("planner_blend.yaml")
SPEED_ZERO = 0.0


class PlannerBlendError(ValueError):
    """Wrong qpos width, G1 36-D, empty new clip, or invented blend width."""


@dataclass(frozen=True)
class MovementState:
    """C++ ``MovementState`` analogue. Directions are world-frame xyz."""

    locomotion_mode: int
    movement_direction: np.ndarray
    facing_direction: np.ndarray
    movement_speed: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "movement_direction", np.asarray(self.movement_direction, dtype=np.float64).reshape(3)
        )
        object.__setattr__(
            self, "facing_direction", np.asarray(self.facing_direction, dtype=np.float64).reshape(3)
        )
        mode = int(self.locomotion_mode)
        if mode < 0 or mode >= PLANNER_N_MODES_V2:
            raise PlannerBlendError(f"planner mode {mode} outside V2 0..{PLANNER_N_MODES_V2 - 1}")
        if not np.isfinite(self.movement_direction).all() or not np.isfinite(self.facing_direction).all():
            raise PlannerBlendError("movement/facing contains NaN/Inf")
        if not np.isfinite(self.movement_speed) or not np.isfinite(self.height):
            raise PlannerBlendError("speed/height contains NaN/Inf")


def default_last_movement_state() -> MovementState:
    """C++ default: IDLE, zero move, facing +X, speed -1, height -1."""
    return MovementState(
        locomotion_mode=0,
        movement_direction=np.zeros(3),
        facing_direction=np.array([1.0, 0.0, 0.0]),
        movement_speed=-1.0,
        height=HEIGHT_DISABLED,
    )


@dataclass
class ReplanDecision:
    need_replan: bool
    time_to_replan: bool
    mode_changed: bool
    facing_changed: bool
    height_changed: bool
    speed_changed: bool
    direction_changed: bool
    under_static: bool
    interval_s: float
    counter_s: float
    reasons: tuple[str, ...] = ()


@dataclass
class BlendResult:
    qpos: np.ndarray
    dq: np.ndarray | None
    current_frame: int
    skipped: bool
    first_copy: bool
    blend_start_frame: int
    new_anim_length: int
    weights_new: np.ndarray


def load_planner_blend_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or PLANNER_BLEND_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("planner_blend.yaml must be a mapping")
    flags = (
        "not_dexhand2_contact",
        "not_table_s4",
        "not_hand_mit_ring",
        "not_invented_latency",
        "not_pico_sdk",
        "not_invented_clip",
        "not_g1_planner_onnx",
        "not_interpolator_substitute",
        "not_onnx_run",
        "not_idle_readapt",
        "not_wrist_pose_augmentation_sampler",
    )
    for key in flags:
        if raw.get(key) is not True:
            raise PlannerBlendError(f"planner_blend.yaml must keep {key}: true")
    if int(raw["blend_num_frames"]) != PLANNER_BLEND_FRAMES:
        raise PlannerBlendError(f"blend_num_frames must stay {PLANNER_BLEND_FRAMES}")
    if float(raw["planner_dt_s"]) != PLANNER_DT_S:
        raise PlannerBlendError("planner_dt_s must stay 0.1")
    n = int(raw["n_dof"])
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=n)
    if int(raw["expected_qpos_dim"]) != planner_qpos_dim(n):
        raise PlannerBlendError("expected_qpos_dim must be 32")
    if int(raw["g1_qpos_dim_forbidden"]) != G1_PLANNER_QPOS_DIM:
        raise PlannerBlendError("g1_qpos_dim_forbidden must stay 36")
    intervals = raw["replan_interval_s"]
    if float(intervals["running"]) != PLANNER_REPLAN_INTERVAL_RUNNING_S:
        raise PlannerBlendError("running interval must stay 0.1 s")
    if float(intervals["crawling"]) != PLANNER_REPLAN_INTERVAL_CRAWLING_S:
        raise PlannerBlendError("crawling interval must stay 0.2 s")
    if float(intervals["boxing"]) != PLANNER_REPLAN_INTERVAL_BOXING_S:
        raise PlannerBlendError("boxing interval must stay 1.0 s")
    if float(intervals["default"]) != PLANNER_REPLAN_INTERVAL_DEFAULT_S:
        raise PlannerBlendError("default interval must stay 1.0 s")
    if set(raw["static_modes"]) != set(PLANNER_STATIC_MODES):
        raise PlannerBlendError("static_modes must match official Idle/Squat/Kneel/Lying/IdleBoxing")
    if list(raw["crawling_modes"]) != [PLANNER_CRAWLING_MODE]:
        raise PlannerBlendError("crawling_modes must be [8] (C++ CRAWLING only, not elbow 14)")
    if set(raw["boxing_modes"]) != set(PLANNER_BOXING_MODES):
        raise PlannerBlendError("boxing_modes must be punches/hooks, not idle/walk boxing")
    if list(raw["loco_mode_map"].values()) != [0, 1, 2]:
        raise PlannerBlendError("loco_mode_map must be stand→idle, slow_walk→slowWalk, fast_walk→walk")
    if 3 in {int(v) for v in raw["loco_mode_map"].values()}:
        raise PlannerBlendError("fast_walk must not map to official run (3)")
    return raw


def is_static_motion_mode(mode: int) -> bool:
    """Official static set: Idle, Squat, Kneel, Lying, Idle Boxing."""
    return int(mode) in PLANNER_STATIC_MODES


def replan_interval_s(mode: int) -> float:
    """C++ interval selected from the *current* locomotion mode."""
    m = int(mode)
    if m == PLANNER_RUNNING_MODE:
        return PLANNER_REPLAN_INTERVAL_RUNNING_S
    if m == PLANNER_CRAWLING_MODE:
        return PLANNER_REPLAN_INTERVAL_CRAWLING_S
    if m in PLANNER_BOXING_MODES:
        return PLANNER_REPLAN_INTERVAL_BOXING_S
    return PLANNER_REPLAN_INTERVAL_DEFAULT_S


def _changed_vec3(a: np.ndarray, b: np.ndarray) -> bool:
    """C++ uses exact ``!=`` on each component."""
    return bool(a[0] != b[0] or a[1] != b[1] or a[2] != b[2])


def evaluate_replan(
    last: MovementState,
    current: MovementState,
    counter_s: float,
    *,
    planner_dt_s: float = PLANNER_DT_S,
) -> ReplanDecision:
    """One 10 Hz planner tick. Counter increments even when replan is skipped."""
    if float(planner_dt_s) != PLANNER_DT_S:
        raise PlannerBlendError("planner_dt_s must stay 0.1 (official 10 Hz)")
    interval = replan_interval_s(current.locomotion_mode)
    new_counter = float(counter_s) + float(planner_dt_s)
    time_to_replan = new_counter >= interval
    if time_to_replan:
        new_counter = 0.0
    mode_changed = int(current.locomotion_mode) != int(last.locomotion_mode)
    facing_changed = _changed_vec3(current.facing_direction, last.facing_direction)
    height_changed = float(current.height) != float(last.height)
    speed_changed = float(current.movement_speed) != float(last.movement_speed)
    direction_changed = _changed_vec3(current.movement_direction, last.movement_direction)
    under_static = is_static_motion_mode(current.locomotion_mode)
    need = False
    reasons: list[str] = []
    if mode_changed or facing_changed or height_changed:
        need = True
        if mode_changed:
            reasons.append("mode")
        if facing_changed:
            reasons.append("facing")
        if height_changed:
            reasons.append("height")
    elif not under_static and (
        speed_changed or direction_changed or (time_to_replan and current.movement_speed != SPEED_ZERO)
    ):
        need = True
        if speed_changed:
            reasons.append("speed")
        if direction_changed:
            reasons.append("direction")
        if time_to_replan and current.movement_speed != SPEED_ZERO:
            reasons.append("timer")
    return ReplanDecision(
        need_replan=need,
        time_to_replan=time_to_replan,
        mode_changed=mode_changed,
        facing_changed=facing_changed,
        height_changed=height_changed,
        speed_changed=speed_changed,
        direction_changed=direction_changed,
        under_static=under_static,
        interval_s=interval,
        counter_s=new_counter,
        reasons=tuple(reasons),
    )


@dataclass
class PlannerReplanClock:
    """Stateful 10 Hz replan gate. ``last`` updates only when a replan fires."""

    last: MovementState = field(default_factory=default_last_movement_state)
    counter_s: float = 0.0

    def tick(self, current: MovementState) -> ReplanDecision:
        decision = evaluate_replan(self.last, current, self.counter_s)
        self.counter_s = decision.counter_s
        if decision.need_replan:
            self.last = current
        return decision


def movement_from_command(
    cmd: CommandVector,
    *,
    yaw_world_rad: float,
    facing: str = "travel",
) -> MovementState:
    """Pack ``command_schema_v1`` into a MovementState. Height stays disabled."""
    mode = loco_mode_to_planner_mode(int(cmd.loco_mode))
    move, face, vel = nav_to_planner_dirs(cmd.nav_cmd, yaw_world_rad=yaw_world_rad, facing=facing)
    speed = float(np.hypot(cmd.nav_cmd[0], cmd.nav_cmd[1]))
    if vel < 0.0:
        speed = 0.0
    return MovementState(
        locomotion_mode=mode,
        movement_direction=move,
        facing_direction=face,
        movement_speed=speed,
        height=HEIGHT_DISABLED,
    )


def blend_weight(f: int, blend_start_frame: int, *, n_frames: int = PLANNER_BLEND_FRAMES) -> float:
    """C++ ``w_new = clamp((f - blend_start) / 8, 0, 1)``."""
    if int(n_frames) != PLANNER_BLEND_FRAMES:
        raise PlannerBlendError(f"blend width must stay {PLANNER_BLEND_FRAMES}")
    w = float(int(f) - int(blend_start_frame)) / float(n_frames)
    return float(np.clip(w, 0.0, 1.0))


def _as_t800_qpos(qpos: np.ndarray, *, n_dof: int) -> np.ndarray:
    arr = np.asarray(qpos, dtype=np.float64)
    if arr.ndim != 2:
        raise PlannerBlendError(f"qpos must be (T, D), got {arr.shape}")
    if arr.shape[-1] == G1_PLANNER_QPOS_DIM:
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    expected = planner_qpos_dim(n_dof)
    if arr.shape[-1] != expected:
        raise PlannerBlendError(
            f"qpos last dim {arr.shape[-1]} != T800 {expected} "
            "(hands bypass WBC; do not concatenate DexHand2 q)"
        )
    if arr.shape[0] < 1:
        raise PlannerBlendError("qpos has zero frames")
    if not np.isfinite(arr).all():
        raise PlannerBlendError("qpos contains NaN/Inf")
    return arr


def _as_t800_dq(dq: np.ndarray | None, n_frames: int, *, n_dof: int) -> np.ndarray | None:
    if dq is None:
        return None
    arr = np.asarray(dq, dtype=np.float64)
    if arr.shape == (n_frames, G1_N_DOF):
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if arr.shape != (n_frames, n_dof):
        raise PlannerBlendError(f"dq shape {arr.shape} != ({n_frames}, {n_dof})")
    if not np.isfinite(arr).all():
        raise PlannerBlendError("dq contains NaN/Inf")
    return arr


def cross_fade_qpos(
    old_qpos: np.ndarray | None,
    new_qpos: np.ndarray,
    *,
    current_frame: int,
    gen_frame: int | None = None,
    old_dq: np.ndarray | None = None,
    new_dq: np.ndarray | None = None,
    n_dof: int | None = None,
    look_ahead: int = PLANNER_LOOKAHEAD_STEPS_50HZ,
) -> BlendResult:
    """C++ ``CurrentFrameAdvancement`` blend. Empty ``old`` copies ``new``.

    ``gen_frame`` defaults to ``current_frame + look_ahead`` (official context
    start). Caller must supply already-resampled **50 Hz** T800 qpos. This does
    not run ONNX and does not invent a stand clip.
    """
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    if n == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=n)
    new = _as_t800_qpos(new_qpos, n_dof=n)
    new_d = _as_t800_dq(new_dq, new.shape[0], n_dof=n)
    if old_qpos is None or np.asarray(old_qpos).size == 0 or np.asarray(old_qpos).shape[0] == 0:
        dq_out = None if new_d is None else new_d.copy()
        return BlendResult(
            qpos=new.copy(),
            dq=dq_out,
            current_frame=0,
            skipped=False,
            first_copy=True,
            blend_start_frame=0,
            new_anim_length=int(new.shape[0]),
            weights_new=np.ones(new.shape[0], dtype=np.float64),
        )
    old = _as_t800_qpos(old_qpos, n_dof=n)
    old_d = _as_t800_dq(old_dq, old.shape[0], n_dof=n)
    if (old_d is None) ^ (new_d is None):
        raise PlannerBlendError("old_dq and new_dq must both be set or both omitted")
    fgen = int(current_frame) + int(look_ahead) if gen_frame is None else int(gen_frame)
    cur = int(current_frame)
    if cur < 0:
        raise PlannerBlendError("current_frame must be >= 0")
    new_anim_length = fgen - cur + int(new.shape[0])
    if new_anim_length <= 0:
        return BlendResult(
            qpos=old.copy(),
            dq=None if old_d is None else old_d.copy(),
            current_frame=cur,
            skipped=True,
            first_copy=False,
            blend_start_frame=0,
            new_anim_length=new_anim_length,
            weights_new=np.zeros(0, dtype=np.float64),
        )
    blend_start = max(0, fgen - cur)
    expected = planner_qpos_dim(n)
    out = np.zeros((new_anim_length, expected), dtype=np.float64)
    out_dq = None if new_d is None else np.zeros((new_anim_length, n), dtype=np.float64)
    weights = np.zeros(new_anim_length, dtype=np.float64)
    old_t = int(old.shape[0])
    new_t = int(new.shape[0])
    for f in range(new_anim_length):
        f_old = int(np.clip(f + cur, 0, old_t - 1))
        f_new = int(np.clip(f + cur - fgen, 0, new_t - 1))
        w_new = blend_weight(f, blend_start)
        w_old = 1.0 - w_new
        weights[f] = w_new
        a = unpack_qpos(old[f_old], n_dof=n)
        b = unpack_qpos(new[f_new], n_dof=n)
        pos = w_old * a.root_pos_m + w_new * b.root_pos_m
        q = w_old * a.q_rad + w_new * b.q_rad
        r = slerp_matrices(
            quaternion_wxyz_to_matrix(a.root_rot_wxyz),
            quaternion_wxyz_to_matrix(b.root_rot_wxyz),
            w_new,
        )
        out[f] = pack_qpos(
            PlannerQpos(root_pos_m=pos, root_rot_wxyz=matrix_to_quaternion_wxyz(r), q_rad=q),
            n_dof=n,
        )
        if out_dq is not None:
            out_dq[f] = w_old * old_d[f_old] + w_new * new_d[f_new]
    return BlendResult(
        qpos=out,
        dq=out_dq,
        current_frame=0,
        skipped=False,
        first_copy=False,
        blend_start_frame=blend_start,
        new_anim_length=new_anim_length,
        weights_new=weights,
    )


def finite_diff_dq(qpos: np.ndarray, *, hz: int = PLANNER_CONTROL_HZ, n_dof: int | None = None) -> np.ndarray:
    """Official resample velocity: ``(q[t+1] - q[t]) * 50``. Last row repeats."""
    n = int(n_dof if n_dof is not None else load_t800_sonic()["n_revolute"])
    arr = _as_t800_qpos(qpos, n_dof=n)
    if int(hz) != PLANNER_CONTROL_HZ:
        raise PlannerBlendError("finite-diff hz must stay 50 (control ring)")
    q = arr[:, 7:]
    dq = np.zeros_like(q)
    if q.shape[0] >= 2:
        dq[:-1] = (q[1:] - q[:-1]) * float(hz)
        dq[-1] = dq[-2]
    return dq


def refuse_idle_readapt() -> None:
    """Idle ADAPTING/RECOVERING is a separate C++ path. Not this module."""
    raise PlannerBlendError(
        "Idle-mode ADAPTING/RECOVERING readapt is not the 8-frame planner blend. "
        "Do not mix kAdaptTrigger into this table."
    )


def refuse_run_blend_onnx(path: str | Path | None = None) -> None:
    """Never execute planner weights from the blend module."""
    if path is not None:
        refuse_g1_planner_onnx(path)
    try:
        refuse_run_planner_onnx()
    except PlannerOnnxBlocked as exc:
        raise PlannerOnnxBlocked(
            f"{exc} 8-frame cross-fade still does not run ONNX. "
            f"T800 qpos is {t800_planner_qpos_dim()}-D; blend width is {PLANNER_BLEND_FRAMES}."
        ) from exc


def refuse_command_schema_as_qpos(flat: np.ndarray) -> None:
    """75-D command_schema is not a planner animation. Use ADR-018 for that."""
    arr = np.asarray(flat)
    last = int(arr.shape[-1]) if arr.ndim >= 1 else 0
    if last == G1_PLANNER_QPOS_DIM:
        refuse_g1_checkpoint(planner_qpos_dim=G1_PLANNER_QPOS_DIM)
    raise PlannerOnnxError(
        f"command last-dim {last} is not T800 planner qpos {t800_planner_qpos_dim()}. "
        "ADR-018 interpolators stay the runtime L1a; this module only blends 50 Hz qpos."
    )
