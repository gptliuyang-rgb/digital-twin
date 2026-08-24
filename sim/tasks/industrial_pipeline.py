"""Kinematic demo playback for the industrial twin (no contact physics).

While friction/stiffness are REQUIRED_INPUT, the visual story is driven by
setting arm/hand qpos + prop poses and calling mj_forward only (ADR-004/006).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hand.grasp_primitives import GraspLibrary
from interface.schema import load_hand_spec
from sim.mujoco_env.env import CombinedMujocoEnv
from sim.mujoco_env.ik import dls_ik_pos
from sim.mujoco_env.kinematic_assist import (
    attach_gun_to_right_wrist,
    place_box_on_pallet,
    set_free_body_pose,
    set_mocap_pose,
    snap_gun_tcp_for_geometry,
    stack_center_on_pallet,
)
from sim.mujoco_env.scene import SceneSpec
from sim.qr_scanner import ScanSpec, simulate_scan

PHASES = (
    "approach_box",
    "grasp",
    "lift",
    "carry",
    "stack",
    "release",
    "approach_gun",
    "grip_gun",
    "scan",
    "done",
)


@dataclass
class PipelineResult:
    phases: list[str] = field(default_factory=list)
    ik_err_m: list[float] = field(default_factory=list)
    box0_z: list[float] = field(default_factory=list)
    scan_geometry_ok: bool = False
    scan_distance_m: float | None = None
    finite: bool = True
    n_steps: int = 0
    policy_eval_forbidden: bool = True
    kinematic_assist: bool = True
    note: str = (
        "Kinematic demo playback (mj_forward only). Uncalibrated contact; "
        "identity flange. Not a sim2real pick rate."
    )


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64)


def _set_hand_q(env: CombinedMujocoEnv, left_q: np.ndarray, right_q: np.ndarray) -> None:
    env.data.qpos[env.handles.left_hand.qadr] = np.asarray(left_q, dtype=np.float64)
    env.data.qpos[env.handles.right_hand.qadr] = np.asarray(right_q, dtype=np.float64)


def _ik_wrists(
    env: CombinedMujocoEnv,
    left_tgt: np.ndarray | None,
    right_tgt: np.ndarray | None,
    left_arm: list[str],
    right_arm: list[str],
) -> list[float]:
    errs: list[float] = []
    if left_tgt is not None:
        r = dls_ik_pos(
            env.model,
            env.data,
            "l_wrist",
            left_tgt,
            left_arm,
            damping=0.05,
            iters=24,
            step=0.55,
            max_delta_rad=0.12,
        )
        errs.append(r["err_m"])
    if right_tgt is not None:
        r = dls_ik_pos(
            env.model,
            env.data,
            "r_wrist",
            right_tgt,
            right_arm,
            damping=0.05,
            iters=24,
            step=0.55,
            max_delta_rad=0.12,
        )
        errs.append(r["err_m"])
    return errs


def _zero_vel(env: CombinedMujocoEnv) -> None:
    env.data.qvel[:] = 0.0
    env.data.qacc[:] = 0.0


def _site_z(env: CombinedMujocoEnv, name: str) -> np.ndarray:
    sid = int(env.model.site(name).id)
    return env.data.site_xmat[sid].reshape(3, 3)[:, 2].copy()


def run_industrial_pipeline(
    env: CombinedMujocoEnv | None = None,
    *,
    steps_per_phase: int = 80,
    kp_scale: float = 1.0,
    on_step: Callable | None = None,
    on_phase: Callable | None = None,
) -> PipelineResult:
    """Run the dual-arm industrial story as kinematic demo playback.

    ``kp_scale`` is unused (kept for API compatibility with older callers).
    """
    del kp_scale
    import mujoco

    env = env or CombinedMujocoEnv(scene="industrial")
    env.reset()
    _zero_vel(env)
    mujoco.mj_forward(env.model, env.data)

    spec = load_hand_spec()
    lib = GraspLibrary(spec)
    q_open = lib.q_active("open", 0.0)
    q_power = lib.q_active("power_grasp", 0.85)
    q_gun = lib.q_active("gun_grip", 0.75)
    left_arm = env.handles.arm_joints["left"]
    right_arm = env.handles.arm_joints["right"]
    result = PipelineResult()

    scene = SceneSpec()
    box_hz = scene.box_size_m[2] / 2.0
    pallet_h = scene.pad_size_m[2]

    box0 = env.xpos("box_0").copy()
    pallet = env.xpos("pallet").copy()
    gun0 = env.xpos("scan_gun").copy()
    stack = stack_center_on_pallet(pallet, pallet_h, box_hz)

    # Reachable dual-arm waypoints (T800 serial arms have ~0.3 m forward reach).
    approach_l = box0 + np.array([0.0, 0.15, 0.05])
    approach_r = box0 + np.array([0.0, -0.15, 0.05])
    grasp_l = box0 + np.array([0.0, 0.11, 0.01])
    grasp_r = box0 + np.array([0.0, -0.11, 0.01])
    lift_l = box0 + np.array([0.0, 0.11, 0.26])
    lift_r = box0 + np.array([0.0, -0.11, 0.26])
    # Mid-air waypoint between pick (y≈-0.22) and stack (y≈+0.22).
    mid = 0.5 * (box0 + stack)
    carry_l = mid + np.array([0.0, 0.11, 0.28])
    carry_r = mid + np.array([0.0, -0.11, 0.28])
    stack_l = stack + np.array([0.0, 0.11, 0.12])
    stack_r = stack + np.array([0.0, -0.11, 0.12])
    gun_reach = gun0 + np.array([0.02, 0.06, 0.03])
    scan_r = stack + np.array([-0.05, -0.15, 0.05])

    def _phase(name: str) -> None:
        result.phases.append(name)
        if on_phase is not None:
            on_phase(name, env, result)

    def _box_between_hands(z_off: float = -0.08) -> None:
        lp = env.xpos("l_wrist")
        rp = env.xpos("r_wrist")
        center = 0.5 * (lp + rp)
        center[2] += z_off
        set_free_body_pose(env.model, env.data, "box_0", center, np.array([1.0, 0.0, 0.0, 0.0]))

    def _tick(
        n: int,
        *,
        left_tgt: np.ndarray | None,
        right_tgt: np.ndarray | None,
        left_q: np.ndarray,
        right_q: np.ndarray,
        box_mode: str = "hold",
        box_pos: np.ndarray | None = None,
        gun_mode: str = "table",
    ) -> None:
        for _ in range(n):
            errs = _ik_wrists(env, left_tgt, right_tgt, left_arm, right_arm)
            result.ik_err_m.extend(errs)
            _set_hand_q(env, left_q, right_q)

            if box_mode == "follow":
                _box_between_hands()
            elif box_mode == "place":
                place_box_on_pallet(
                    env.model,
                    env.data,
                    "box_0",
                    pallet,
                    pallet_height_m=pallet_h,
                    box_half_z_m=box_hz,
                )
            elif box_mode == "hold" and box_pos is not None:
                set_free_body_pose(
                    env.model, env.data, "box_0", box_pos, np.array([1.0, 0.0, 0.0, 0.0])
                )
            elif box_mode == "lerp" and box_pos is not None:
                cur = env.xpos("box_0")
                set_free_body_pose(
                    env.model,
                    env.data,
                    "box_0",
                    _lerp(cur, box_pos, 0.18),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )

            if gun_mode == "hand":
                attach_gun_to_right_wrist(env.model, env.data, "scan_gun")
            elif gun_mode == "table":
                set_mocap_pose(
                    env.model,
                    env.data,
                    "scan_gun",
                    gun0,
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )

            _zero_vel(env)
            mujoco.mj_forward(env.model, env.data)
            result.n_steps += 1
            result.finite = result.finite and bool(np.isfinite(env.data.qpos).all())
            result.box0_z.append(float(env.xpos("box_0")[2]))
            if on_step is not None:
                on_step(env, result)

    # Park left arm after release so it does not block the scan view.
    park_l = np.array([0.12, 0.28, 0.95])

    # --- approach (box stays on pick spot) ---
    _tick(
        steps_per_phase,
        left_tgt=approach_l,
        right_tgt=approach_r,
        left_q=q_open,
        right_q=q_open,
        box_mode="hold",
        box_pos=box0,
        gun_mode="table",
    )
    _phase("approach_box")

    # --- grasp: close fingers while wrists move in ---
    for i in range(steps_per_phase):
        t = (i + 1) / steps_per_phase
        _tick(
            1,
            left_tgt=grasp_l,
            right_tgt=grasp_r,
            left_q=_lerp(q_open, q_power, t),
            right_q=_lerp(q_open, q_power, t),
            box_mode="follow",
            gun_mode="table",
        )
    _phase("grasp")

    _tick(
        steps_per_phase,
        left_tgt=lift_l,
        right_tgt=lift_r,
        left_q=q_power,
        right_q=q_power,
        box_mode="follow",
        gun_mode="table",
    )
    _phase("lift")

    _tick(
        steps_per_phase,
        left_tgt=carry_l,
        right_tgt=carry_r,
        left_q=q_power,
        right_q=q_power,
        box_mode="follow",
        gun_mode="table",
    )
    _phase("carry")

    _tick(
        steps_per_phase,
        left_tgt=stack_l,
        right_tgt=stack_r,
        left_q=q_power,
        right_q=q_power,
        box_mode="lerp",
        box_pos=stack,
        gun_mode="table",
    )
    place_box_on_pallet(
        env.model, env.data, "box_0", pallet, pallet_height_m=pallet_h, box_half_z_m=box_hz
    )
    mujoco.mj_forward(env.model, env.data)
    _phase("stack")

    n_rel = max(30, steps_per_phase // 2)
    for i in range(n_rel):
        t = (i + 1) / n_rel
        _tick(
            1,
            left_tgt=stack_l,
            right_tgt=stack_r,
            left_q=_lerp(q_power, q_open, t),
            right_q=_lerp(q_power, q_open, t),
            box_mode="place",
            gun_mode="table",
        )
    _phase("release")

    _tick(
        steps_per_phase,
        left_tgt=park_l,
        right_tgt=gun_reach,
        left_q=q_open,
        right_q=q_open,
        box_mode="place",
        gun_mode="table",
    )
    _phase("approach_gun")

    n_grip = max(40, steps_per_phase // 2)
    for i in range(n_grip):
        t = (i + 1) / n_grip
        _tick(
            1,
            left_tgt=park_l,
            right_tgt=gun_reach,
            left_q=q_open,
            right_q=_lerp(q_open, q_gun, t),
            box_mode="place",
            gun_mode="hand" if t > 0.2 else "table",
        )
    _phase("grip_gun")

    _tick(
        steps_per_phase,
        left_tgt=park_l,
        right_tgt=scan_r,
        left_q=q_open,
        right_q=q_gun,
        box_mode="place",
        gun_mode="hand",
    )
    _phase("scan")

    # Geometry gate after last visual frame.
    if _has_site(env, "gun_tcp") and _has_site(env, "box_0_qr"):
        attach_gun_to_right_wrist(env.model, env.data, "scan_gun")
        mujoco.mj_forward(env.model, env.data)
        gun_pos = env.site_xpos("gun_tcp").copy()
        qr_pos = env.site_xpos("box_0_qr").copy()
        gun_z = _site_z(env, "gun_tcp")
        dist = float(np.linalg.norm(qr_pos - gun_pos))
        if 0.05 <= dist <= 0.35:
            scan = simulate_scan(
                np.zeros((8, 8, 3), dtype=np.uint8),
                gun_tcp_pos_m=gun_pos,
                gun_tcp_z=gun_z,
                qr_pos_m=qr_pos,
                qr_normal=np.array([1.0, 0.0, 0.0]),
                rel_speed_m_s=0.0,
                spec=ScanSpec(d_min_m=0.05, d_max_m=0.35, theta_max_rad=np.deg2rad(70.0)),
            )
            result.scan_geometry_ok = scan.geometry_ok
            result.scan_distance_m = scan.distance_m
        else:
            snap_gun_tcp_for_geometry(env.model, env.data, "scan_gun", "box_0_qr")
            mujoco.mj_forward(env.model, env.data)
            gun_pos = env.site_xpos("gun_tcp")
            gun_z = _site_z(env, "gun_tcp")
            scan = simulate_scan(
                np.zeros((8, 8, 3), dtype=np.uint8),
                gun_tcp_pos_m=gun_pos,
                gun_tcp_z=gun_z,
                qr_pos_m=qr_pos,
                qr_normal=np.array([1.0, 0.0, 0.0]),
                rel_speed_m_s=0.0,
                spec=ScanSpec(d_min_m=0.05, d_max_m=0.35, theta_max_rad=np.deg2rad(55.0)),
            )
            result.scan_geometry_ok = scan.geometry_ok
            result.scan_distance_m = scan.distance_m
            attach_gun_to_right_wrist(env.model, env.data, "scan_gun")
            mujoco.mj_forward(env.model, env.data)

    _phase("done")
    return result


def _has_site(env: CombinedMujocoEnv, name: str) -> bool:
    try:
        env.model.site(name)
        return True
    except Exception:
        return False
