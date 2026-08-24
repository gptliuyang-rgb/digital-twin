"""Scripted industrial twin: dual-arm approach / grasp / lift / stack / QR scan.

Physics runs with **kinematic demo assists** while contact is uncalibrated:
box follow + gun mocap attach are not validated grasps (ADR-004/006).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from assets.objects.boxes import EURO_PALLET_M
from hand.grasp_primitives import GraspLibrary
from interface.schema import load_hand_spec
from sim.mujoco_env.env import CombinedMujocoEnv
from sim.mujoco_env.ik import smooth_move_wrist
from sim.mujoco_env.kinematic_assist import (
    aim_gun_at_site,
    attach_gun_to_right_wrist,
    follow_box_between_wrists,
    hold_free_body,
    lerp_free_body_toward,
    place_box_on_pallet,
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
        "Uncalibrated contact; kinematic_bringup flange; box/gun use demo kinematic "
        "assist (not E1/E2 grasp). Not a sim2real pick rate."
    )


def _site_z(env: CombinedMujocoEnv, name: str) -> np.ndarray:
    sid = int(env.model.site(name).id)
    mat = env.data.site_xmat[sid].reshape(3, 3)
    return mat[:, 2].copy()


def _lerp_q(open_q: np.ndarray, closed_q: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * open_q + t * closed_q


def _scan_wrist_target(env: CombinedMujocoEnv) -> np.ndarray | None:
    """Right wrist IK goal so gun_tcp sits in front of box_0 QR sticker."""
    if not _has_site(env, "box_0_qr"):
        return None
    qr = env.site_xpos("box_0_qr")
    # Stand off to the robot's right-front so the right wrist can reach with gun in hand.
    return qr + np.array([-0.08, -0.12, 0.0], dtype=np.float64)


def run_industrial_pipeline(
    env: CombinedMujocoEnv | None = None,
    *,
    steps_per_phase: int = 80,
    kp_scale: float = 1.0,
    on_step: Callable | None = None,
    on_phase: Callable | None = None,
) -> PipelineResult:
    import mujoco

    env = env or CombinedMujocoEnv(scene="industrial")
    env.reset()
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
    pallet_h = EURO_PALLET_M[2]

    box = env.xpos("box_0") if _has_body(env, "box_0") else np.array([0.30, -0.12, 0.94])
    pallet = env.xpos("pallet") if _has_body(env, "pallet") else np.array([0.38, 0.0, 0.07])
    gun_table = env.xpos("scan_gun") if _has_body(env, "scan_gun") else np.array([0.34, -0.28, 0.87])
    stack_pos = stack_center_on_pallet(pallet, pallet_h, box_hz)
    box_hold_pos = box.copy()
    box_hold_quat = np.array([1.0, 0.0, 0.0, 0.0])

    targets: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}

    def _phase(name: str) -> None:
        result.phases.append(name)
        if on_phase is not None:
            on_phase(name, env, result)

    def _tick(
        phase: str,
        n: int,
        *,
        left_q: np.ndarray,
        right_q: np.ndarray,
        box_follow: bool = False,
        box_hold: bool = False,
        box_lerp_to: np.ndarray | None = None,
        box_on_pallet: bool = False,
        gun_follow: bool = False,
        gun_aim_qr: bool = False,
        dynamic_scan: bool = False,
        tick_kp_scale: float | None = None,
    ) -> None:
        lt, rt = targets.get(phase, (None, None))
        step_kp = kp_scale if tick_kp_scale is None else tick_kp_scale
        for _ in range(n):
            if dynamic_scan:
                rt = _scan_wrist_target(env)
            if lt is not None:
                err = smooth_move_wrist(env.model, env.data, "l_wrist", lt, left_arm, alpha=0.15)
                result.ik_err_m.append(err["err_m"])
            if rt is not None:
                err = smooth_move_wrist(env.model, env.data, "r_wrist", rt, right_arm, alpha=0.12)
                result.ik_err_m.append(err["err_m"])
            if box_on_pallet and _has_body(env, "box_0"):
                place_box_on_pallet(
                    env.model,
                    env.data,
                    "box_0",
                    pallet,
                    pallet_height_m=pallet_h,
                    box_half_z_m=box_hz,
                )
            elif box_hold and _has_body(env, "box_0"):
                hold_free_body(env.model, env.data, "box_0", box_hold_pos, box_hold_quat)
            elif box_lerp_to is not None and _has_body(env, "box_0"):
                lerp_free_body_toward(env.model, env.data, "box_0", box_lerp_to, alpha=0.10)
            elif box_follow and _has_body(env, "box_0"):
                follow_box_between_wrists(env.model, env.data, "box_0", z_offset_m=-0.05)
            if gun_aim_qr and _has_body(env, "scan_gun") and _has_site(env, "box_0_qr"):
                aim_gun_at_site(env.model, env.data, "scan_gun", "box_0_qr")
            elif gun_follow and _has_body(env, "scan_gun"):
                attach_gun_to_right_wrist(env.model, env.data, "scan_gun")
            if (
                box_follow
                or box_hold
                or box_lerp_to is not None
                or box_on_pallet
                or gun_follow
                or gun_aim_qr
            ):
                mujoco.mj_forward(env.model, env.data)
            body_q = env.data.qpos.copy()
            env.step_mit(left_q, right_q, body_q_des=body_q, kp_scale=step_kp)
            result.n_steps += 1
            result.finite = result.finite and bool(np.isfinite(env.data.qpos).all())
            try:
                result.box0_z.append(float(env.xpos("box_0")[2]))
            except Exception:
                pass
            if on_step is not None:
                on_step(env, result)

    # --- phase targets (world frame, within T800 elbow reach) ---
    targets["approach_box"] = (
        box + np.array([0.0, 0.14, 0.04]),
        box + np.array([0.0, -0.14, 0.04]),
    )
    targets["grasp"] = targets["approach_box"]
    lift_z = 0.12
    targets["lift"] = (
        box + np.array([0.0, 0.14, lift_z]),
        box + np.array([0.0, -0.14, lift_z]),
    )
    carry_z = stack_pos[2] + 0.14
    targets["carry"] = (
        np.array([stack_pos[0], stack_pos[1] + 0.14, carry_z]),
        np.array([stack_pos[0], stack_pos[1] - 0.14, carry_z]),
    )
    targets["stack"] = (
        np.array([stack_pos[0], stack_pos[1] + 0.14, stack_pos[2] + 0.10]),
        np.array([stack_pos[0], stack_pos[1] - 0.14, stack_pos[2] + 0.10]),
    )
    targets["release"] = targets["stack"]
    targets["approach_gun"] = (
        None,
        gun_table + np.array([0.0, 0.0, 0.08]),
    )
    targets["grip_gun"] = targets["approach_gun"]
    scan0 = _scan_wrist_target(env)
    targets["scan"] = (None, scan0 if scan0 is not None else stack_pos + np.array([-0.12, 0.0, 0.08]))

    _phase("approach_box")
    _tick("approach_box", steps_per_phase, left_q=q_open, right_q=q_open, box_hold=True)

    _phase("grasp")
    for i in range(steps_per_phase):
        t = (i + 1) / steps_per_phase
        lq = _lerp_q(q_open, q_power, t)
        rq = _lerp_q(q_open, q_power, t)
        _tick("grasp", 1, left_q=lq, right_q=rq, box_follow=True)

    _phase("lift")
    _tick("lift", steps_per_phase, left_q=q_power, right_q=q_power, box_follow=True)

    _phase("carry")
    _tick(
        "carry",
        steps_per_phase,
        left_q=q_power,
        right_q=q_power,
        box_follow=True,
    )

    _phase("stack")
    _tick(
        "stack",
        steps_per_phase,
        left_q=q_power,
        right_q=q_power,
        box_lerp_to=stack_pos,
    )
    place_box_on_pallet(env.model, env.data, "box_0", pallet, pallet_height_m=pallet_h, box_half_z_m=box_hz)
    mujoco.mj_forward(env.model, env.data)

    _phase("release")
    for i in range(max(30, steps_per_phase // 2)):
        t = (i + 1) / max(30, steps_per_phase // 2)
        lq = _lerp_q(q_power, q_open, t)
        rq = _lerp_q(q_power, q_open, t)
        _tick("release", 1, left_q=lq, right_q=rq, box_on_pallet=True)

    _phase("approach_gun")
    _tick(
        "approach_gun",
        steps_per_phase,
        left_q=q_open,
        right_q=q_open,
        box_on_pallet=True,
        tick_kp_scale=0.6,
    )

    _phase("grip_gun")
    for i in range(max(40, steps_per_phase // 2)):
        t = (i + 1) / max(40, steps_per_phase // 2)
        rq = _lerp_q(q_open, q_gun, t)
        _tick(
            "grip_gun",
            1,
            left_q=q_open,
            right_q=rq,
            box_on_pallet=True,
            gun_follow=t > 0.25,
        )

    _phase("scan")
    _tick(
        "scan",
        steps_per_phase,
        left_q=q_open,
        right_q=q_gun,
        box_on_pallet=True,
        gun_follow=True,
        dynamic_scan=True,
    )

    if _has_site(env, "gun_tcp") and _has_site(env, "box_0_qr"):
        from sim.mujoco_env.kinematic_assist import snap_gun_tcp_for_geometry

        snap_gun_tcp_for_geometry(env.model, env.data, "scan_gun", "box_0_qr")
        mujoco.mj_forward(env.model, env.data)
        gun_pos = env.site_xpos("gun_tcp")
        qr_pos = env.site_xpos("box_0_qr")
        gun_z = _site_z(env, "gun_tcp")
        dummy = np.zeros((8, 8, 3), dtype=np.uint8)
        scan = simulate_scan(
            dummy,
            gun_tcp_pos_m=gun_pos,
            gun_tcp_z=gun_z,
            qr_pos_m=qr_pos,
            qr_normal=np.array([1.0, 0.0, 0.0]),
            rel_speed_m_s=0.0,
            spec=ScanSpec(d_min_m=0.05, d_max_m=0.35, theta_max_rad=np.deg2rad(55.0)),
        )
        result.scan_geometry_ok = scan.geometry_ok
        result.scan_distance_m = scan.distance_m

    _phase("done")
    return result


def _has_body(env: CombinedMujocoEnv, name: str) -> bool:
    try:
        env.model.body(name)
        return True
    except Exception:
        return False


def _has_site(env: CombinedMujocoEnv, name: str) -> bool:
    try:
        env.model.site(name)
        return True
    except Exception:
        return False
