"""Scripted industrial twin: dual-arm approach / grasp / lift / stack / QR scan.

Physics runs, but contact is uncalibrated. The FSM does not claim pick success.
Scan success is geometry (+ optional decode), not a learned IBVS policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hand.grasp_primitives import GraspLibrary
from interface.schema import load_hand_spec
from sim.mujoco_env.env import CombinedMujocoEnv
from sim.mujoco_env.ik import dls_ik_pos
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
    note: str = (
        "Uncalibrated contact; kinematic_bringup flange. Not a sim2real pick rate."
    )


def _site_z(env: CombinedMujocoEnv, name: str) -> np.ndarray:
    sid = int(env.model.site(name).id)
    mat = env.data.site_xmat[sid].reshape(3, 3)
    return mat[:, 2].copy()


def run_industrial_pipeline(
    env: CombinedMujocoEnv | None = None,
    *,
    steps_per_phase: int = 80,
    kp_scale: float = 1.0,
    on_step: Callable | None = None,
    on_phase: Callable | None = None,
) -> PipelineResult:
    env = env or CombinedMujocoEnv(scene="industrial")
    env.reset()
    spec = load_hand_spec()
    lib = GraspLibrary(spec)
    q_open = lib.q_active("open", 0.0)
    q_power = lib.q_active("power_grasp", 0.9)
    q_gun = lib.q_active("gun_grip", 0.8)
    left_q = q_open.copy()
    right_q = q_open.copy()
    body_q = env.data.qpos.copy()
    result = PipelineResult()

    def _tick(l_q: np.ndarray, r_q: np.ndarray, n: int) -> None:
        for _ in range(n):
            env.step_mit(l_q, r_q, body_q_des=body_q, kp_scale=kp_scale)
            result.n_steps += 1
            result.finite = result.finite and bool(np.isfinite(env.data.qpos).all())
            try:
                result.box0_z.append(float(env.xpos("box_0")[2]))
            except Exception:
                pass
            if on_step is not None:
                on_step(env, result)

    def _phase(name: str) -> None:
        result.phases.append(name)
        if on_phase is not None:
            on_phase(name, env, result)

    # Approach: IK wrists toward box sides.
    try:
        box = env.xpos("box_0")
    except Exception:
        box = np.array([0.48, -0.12, 0.95])
    left_tgt = box + np.array([0.0, 0.16, 0.02])
    right_tgt = box + np.array([0.0, -0.16, 0.02])
    ik_l = dls_ik_pos(env.model, env.data, "l_wrist", left_tgt, env.handles.arm_joints["left"])
    ik_r = dls_ik_pos(env.model, env.data, "r_wrist", right_tgt, env.handles.arm_joints["right"])
    body_q = env.data.qpos.copy()
    result.ik_err_m.extend([ik_l["err_m"], ik_r["err_m"]])
    _phase("approach_box")
    _tick(q_open, q_open, steps_per_phase)

    _phase("grasp")
    _tick(q_power, q_power, steps_per_phase)
    left_q, right_q = q_power, q_power

    _phase("lift")
    lift = dls_ik_pos(
        env.model,
        env.data,
        "l_wrist",
        env.xpos("l_wrist") + np.array([0.0, 0.0, 0.08]),
        env.handles.arm_joints["left"],
    )
    dls_ik_pos(
        env.model,
        env.data,
        "r_wrist",
        env.xpos("r_wrist") + np.array([0.0, 0.0, 0.08]),
        env.handles.arm_joints["right"],
    )
    body_q = env.data.qpos.copy()
    result.ik_err_m.append(lift["err_m"])
    _tick(left_q, right_q, steps_per_phase)

    _phase("carry")
    pallet = env.xpos("pallet") if _has_body(env, "pallet") else np.array([0.70, 0.0, 0.2])
    dls_ik_pos(env.model, env.data, "l_wrist", pallet + np.array([0.0, 0.16, 0.25]), env.handles.arm_joints["left"])
    dls_ik_pos(env.model, env.data, "r_wrist", pallet + np.array([0.0, -0.16, 0.25]), env.handles.arm_joints["right"])
    body_q = env.data.qpos.copy()
    _tick(left_q, right_q, steps_per_phase)

    _phase("stack")
    dls_ik_pos(env.model, env.data, "l_wrist", pallet + np.array([0.0, 0.16, 0.18]), env.handles.arm_joints["left"])
    dls_ik_pos(env.model, env.data, "r_wrist", pallet + np.array([0.0, -0.16, 0.18]), env.handles.arm_joints["right"])
    body_q = env.data.qpos.copy()
    _tick(left_q, right_q, steps_per_phase)

    _phase("release")
    left_q, right_q = q_open, q_open
    _tick(left_q, right_q, max(20, steps_per_phase // 2))

    _phase("approach_gun")
    if _has_body(env, "scan_gun"):
        gun = env.xpos("scan_gun")
        dls_ik_pos(env.model, env.data, "r_wrist", gun + np.array([0.0, 0.0, 0.05]), env.handles.arm_joints["right"])
        body_q = env.data.qpos.copy()
    _tick(q_open, q_open, steps_per_phase)

    _phase("grip_gun")
    right_q = q_gun
    _tick(q_open, right_q, max(20, steps_per_phase // 2))

    _phase("scan")
    if _has_body(env, "box_1"):
        qr = env.site_xpos("box_1_qr")
        dls_ik_pos(env.model, env.data, "r_wrist", qr + np.array([-0.18, 0.0, 0.0]), env.handles.arm_joints["right"])
        body_q = env.data.qpos.copy()
    _tick(q_open, q_gun, steps_per_phase)
    if _has_site(env, "gun_tcp") and _has_site(env, "box_1_qr"):
        gun_pos = env.site_xpos("gun_tcp")
        qr_pos = env.site_xpos("box_1_qr")
        gun_z = _site_z(env, "gun_tcp")
        dummy = np.zeros((8, 8, 3), dtype=np.uint8)
        scan = simulate_scan(
            dummy,
            gun_tcp_pos_m=gun_pos,
            gun_tcp_z=gun_z,
            qr_pos_m=qr_pos,
            qr_normal=np.array([-1.0, 0.0, 0.0]),
            rel_speed_m_s=0.0,
            spec=ScanSpec(),
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
