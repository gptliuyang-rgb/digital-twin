"""Free-base T800 Table S4 root-push sweep and air-drop. Not a SONIC pass/fail.

The unperturbed 3 s PD hold can lean ~41° (0.72 rad) and still sit under the
0.80 rad fall tilt band. That is ``leaned``, not a stand, and not a tracker
success (ADR-026). ADR-027 sweeps Table S4 planar extrema (±X / ±Y at 0.5 m/s)
as one-shot root velocities after that hold. ADR-028 spreads the same impulse
as a constant world force F = m v / T for Table S4 duration extrema 1 s and
3 s. ADR-029 adds Table S4 angular-velocity extrema (roll/pitch ±0.52 rad/s,
yaw ±0.78 rad/s) as one-shot ``qvel[3:6]`` and as τ = I ω / T. ADR-030 sweeps
Table S4 linear-z (±0.2 m/s) as its own one-shot and F = m v / T — not mixed
into the planar 0.5 m/s cases. Air-drop (no floor) proves the freejoint.
``grasp_success_rate`` stays JSON null. Combined T800+Hand stays PolicyEvalBlocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from assets.combined.assemble import PolicyEvalBlocked
from eval.l2_freebase_stand import evaluate_freebase_stand
from eval.lean_classify import fallen_from_cfg, posture_from_cfg
from interface.schema import REPO_ROOT
from sim.mujoco_env.privileged_l2 import refuse_grasp_success_key
from sim.mujoco_env.t800_env import T800MujocoEnv, pelvis_tilt_rad, refuse_combined_robot
from wbc.foot_frame import FOOT_FRAME_DECISION, assert_foot_frame
from wbc.pd_stand import pd_stand_q_des_rad
from wbc.ppo.table_s4 import (
    axis_aligned_angvel_extrema_rad_s,
    axis_aligned_linvel_extrema_mps,
    force_n_from_impulse,
    sustained_force_cases,
    sustained_torque_cases,
    torque_nm_from_impulse,
    vertical_force_cases,
    vertical_linvel_extrema_mps,
)


def _q_des(env: T800MujocoEnv) -> np.ndarray:
    return pd_stand_q_des_rad() if env.source == "official" else np.zeros(len(env.joint_order))


def evaluate_lateral_push(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
    lin_vel_mps: list[float] | np.ndarray | None = None,
    case_name: str | None = None,
    meta_key: str = "push",
) -> dict[str, Any]:
    if env.pinned_base:
        raise ValueError("lateral push requires pinned_base=False")
    if env.n_plane < 1:
        raise ValueError("lateral push requires a floor plane")
    q_des = _q_des(env)
    if lin_vel_mps is None:
        lin_vel = np.asarray(cfg[meta_key]["lin_vel_mps"], dtype=np.float64).reshape(3)
        name = str(cfg[meta_key].get("name", "+y"))
    else:
        lin_vel = np.asarray(lin_vel_mps, dtype=np.float64).reshape(3)
        name = case_name or "custom"
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    dt = float(env.model.opt.timestep)
    n_settle = int(round(settle_s / dt))
    n_post = int(round(post_push_s / dt))
    z_hist: list[float] = []
    tilt_hist: list[float] = []
    time_to_fall_s: float | None = None
    injected = False
    t = 0.0
    for i in range(max(n_settle, 1)):
        env.step(q_des)
        t = float(i + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    env.apply_root_linvel(lin_vel)
    injected = True
    t_inject_s = t
    pre_z = float(z_hist[-1]) if z_hist else float("nan")
    pre_tilt = float(tilt_hist[-1]) if tilt_hist else float("nan")
    pre_posture = posture_from_cfg(pre_z, pre_tilt, cfg)
    for j in range(max(n_post, 1)):
        env.step(q_des)
        t = t_inject_s + float(j + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    pelvis, rot = env.body_pose("pelvis")
    end_tilt = pelvis_tilt_rad(rot)
    posture = posture_from_cfg(float(pelvis[2]), end_tilt, cfg)
    fallen = posture == "fallen"
    return {
        "settle_s": settle_s,
        "post_push_s": post_push_s,
        "inject_t_s": t_inject_s,
        "pre_push_pelvis_z_m": pre_z,
        "pre_push_tilt_rad": pre_tilt,
        "pre_push_posture": pre_posture,
        "name": name,
        "lin_vel_mps": [float(x) for x in lin_vel],
        "kind": cfg[meta_key]["kind"],
        "source": cfg[meta_key]["source"],
        "injected": injected,
        "end_pelvis_z_m": float(pelvis[2]),
        "min_pelvis_z_m": float(np.min(z_hist)),
        "end_tilt_rad": float(end_tilt),
        "max_tilt_rad": float(np.max(tilt_hist)),
        "posture": posture,
        "fallen": fallen,
        "fall_rate": 1.0 if fallen else 0.0,
        "time_to_fall_s": time_to_fall_s,
        "end_foot": env.foot_diagnostics(),
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "grasp_success_rate": None,
    }


def evaluate_airdrop(env: T800MujocoEnv, *, hold_s: float, min_drop_m: float) -> dict[str, Any]:
    if env.pinned_base:
        raise ValueError("air-drop requires pinned_base=False")
    if env.n_plane != 0:
        raise ValueError("air-drop requires add_floor=False (no plane)")
    q_des = _q_des(env)
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    pelvis0, _ = env.body_pose("pelvis")
    n = int(round(hold_s / float(env.model.opt.timestep)))
    for _ in range(max(n, 1)):
        env.step(q_des)
    pelvis, rot = env.body_pose("pelvis")
    drop_m = float(pelvis0[2] - pelvis[2])
    return {
        "hold_s": hold_s,
        "add_floor": False,
        "n_plane": env.n_plane,
        "nq": int(env.model.nq),
        "start_pelvis_z_m": float(pelvis0[2]),
        "end_pelvis_z_m": float(pelvis[2]),
        "drop_m": drop_m,
        "min_drop_m": min_drop_m,
        "end_tilt_rad": float(pelvis_tilt_rad(rot)),
        "freejoint_moved": drop_m > min_drop_m,
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "not_a_stand_rating": True,
        "grasp_success_rate": None,
        "note": "no floor: gravity must move the freejoint; not a stand rating",
    }


def evaluate_push_sweep(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
) -> list[dict[str, Any]]:
    """Run one-shot qvel at each Table S4 planar extremum. Not a SONIC gate."""
    axes = tuple(cfg.get("sweep", {}).get("axes", ["x", "y"]))
    cases = axis_aligned_linvel_extrema_mps(axes=axes)
    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(
            evaluate_lateral_push(
                env,
                cfg=cfg,
                settle_s=settle_s,
                post_push_s=post_push_s,
                lin_vel_mps=case["lin_vel_mps"],
                case_name=str(case["name"]),
            )
        )
    return out


def evaluate_vertical_sweep(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
) -> list[dict[str, Any]]:
    """Run one-shot qvel at Table S4 ±Z 0.2 m/s. Not a SONIC gate. Not planar."""
    axes = tuple(cfg.get("vertical_sweep", {}).get("axes", ["z"]))
    cases = (
        vertical_linvel_extrema_mps()
        if axes == ("z",)
        else axis_aligned_linvel_extrema_mps(axes=axes)
    )
    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(
            evaluate_lateral_push(
                env,
                cfg=cfg,
                settle_s=settle_s,
                post_push_s=post_push_s,
                lin_vel_mps=case["lin_vel_mps"],
                case_name=str(case["name"]),
                meta_key="vertical",
            )
        )
    return out


def evaluate_sustained_force(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
    lin_vel_mps: list[float] | np.ndarray,
    duration_s: float,
    case_name: str,
    meta_key: str = "sustained",
) -> dict[str, Any]:
    """Apply F = m v / T on LINK_BASE for Table S4 duration. Not a SONIC gate."""
    if env.pinned_base:
        raise ValueError("sustained force requires pinned_base=False")
    if env.n_plane < 1:
        raise ValueError("sustained force requires a floor plane")
    q_des = _q_des(env)
    lin_vel = np.asarray(lin_vel_mps, dtype=np.float64).reshape(3)
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    mass_kg = env.root_subtree_mass_kg()
    force_n = force_n_from_impulse(mass_kg, lin_vel.tolist(), float(duration_s))
    dt = float(env.model.opt.timestep)
    n_settle = int(round(settle_s / dt))
    n_force = int(round(float(duration_s) / dt))
    n_post = int(round(post_push_s / dt))
    z_hist: list[float] = []
    tilt_hist: list[float] = []
    time_to_fall_s: float | None = None
    t = 0.0
    for i in range(max(n_settle, 1)):
        env.step(q_des)
        t = float(i + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    env.apply_root_force_n(np.asarray(force_n, dtype=np.float64))
    t_inject_s = t
    pre_z = float(z_hist[-1]) if z_hist else float("nan")
    pre_tilt = float(tilt_hist[-1]) if tilt_hist else float("nan")
    pre_posture = posture_from_cfg(pre_z, pre_tilt, cfg)
    for j in range(max(n_force, 1)):
        env.step(q_des)
        t = t_inject_s + float(j + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    env.clear_root_force()
    t_release_s = t
    for k in range(max(n_post, 1)):
        env.step(q_des)
        t = t_release_s + float(k + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    pelvis, rot = env.body_pose("pelvis")
    end_tilt = pelvis_tilt_rad(rot)
    posture = posture_from_cfg(float(pelvis[2]), end_tilt, cfg)
    fallen = posture == "fallen"
    impulse_n_s = [float(f) * float(duration_s) for f in force_n]
    return {
        "settle_s": settle_s,
        "duration_s": float(duration_s),
        "post_push_s": post_push_s,
        "inject_t_s": t_inject_s,
        "release_t_s": t_release_s,
        "pre_push_pelvis_z_m": pre_z,
        "pre_push_tilt_rad": pre_tilt,
        "pre_push_posture": pre_posture,
        "name": case_name,
        "lin_vel_mps": [float(x) for x in lin_vel],
        "mass_kg": mass_kg,
        "force_n": [float(x) for x in force_n],
        "impulse_n_s": impulse_n_s,
        "force_formula": "F = m * v / T",
        "kind": "sustained_force",
        "source": cfg[meta_key]["source"],
        "injected": True,
        "end_pelvis_z_m": float(pelvis[2]),
        "min_pelvis_z_m": float(np.min(z_hist)),
        "end_tilt_rad": float(end_tilt),
        "max_tilt_rad": float(np.max(tilt_hist)),
        "posture": posture,
        "fallen": fallen,
        "fall_rate": 1.0 if fallen else 0.0,
        "time_to_fall_s": time_to_fall_s,
        "end_foot": env.foot_diagnostics(),
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "not_one_shot_qvel": True,
        "grasp_success_rate": None,
    }


def evaluate_force_sweep(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
) -> list[dict[str, Any]]:
    """Planar extrema × duration extrema as F = m v / T. Not a SONIC gate."""
    axes = tuple(cfg.get("sweep", {}).get("axes", ["x", "y"]))
    cases = sustained_force_cases(axes=axes)
    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(
            evaluate_sustained_force(
                env,
                cfg=cfg,
                settle_s=settle_s,
                post_push_s=post_push_s,
                lin_vel_mps=case["lin_vel_mps"],
                duration_s=float(case["duration_s"]),
                case_name=str(case["name"]),
            )
        )
    return out


def evaluate_vertical_force_sweep(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
) -> list[dict[str, Any]]:
    """±Z 0.2 m/s × duration extrema as F = m v / T. Not a SONIC gate."""
    cases = vertical_force_cases()
    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(
            evaluate_sustained_force(
                env,
                cfg=cfg,
                settle_s=settle_s,
                post_push_s=post_push_s,
                lin_vel_mps=case["lin_vel_mps"],
                duration_s=float(case["duration_s"]),
                case_name=str(case["name"]),
                meta_key="vertical_force",
            )
        )
    return out


def evaluate_angvel_push(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
    ang_vel_rad_s: list[float] | np.ndarray,
    case_name: str,
) -> dict[str, Any]:
    """One-shot freejoint body-frame angvel. Not a SONIC gate."""
    if env.pinned_base:
        raise ValueError("angvel push requires pinned_base=False")
    if env.n_plane < 1:
        raise ValueError("angvel push requires a floor plane")
    q_des = _q_des(env)
    omega = np.asarray(ang_vel_rad_s, dtype=np.float64).reshape(3)
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    dt = float(env.model.opt.timestep)
    n_settle = int(round(settle_s / dt))
    n_post = int(round(post_push_s / dt))
    z_hist: list[float] = []
    tilt_hist: list[float] = []
    time_to_fall_s: float | None = None
    t = 0.0
    for i in range(max(n_settle, 1)):
        env.step(q_des)
        t = float(i + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    env.apply_root_angvel(omega)
    t_inject_s = t
    pre_z = float(z_hist[-1]) if z_hist else float("nan")
    pre_tilt = float(tilt_hist[-1]) if tilt_hist else float("nan")
    pre_posture = posture_from_cfg(pre_z, pre_tilt, cfg)
    for j in range(max(n_post, 1)):
        env.step(q_des)
        t = t_inject_s + float(j + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    pelvis, rot = env.body_pose("pelvis")
    end_tilt = pelvis_tilt_rad(rot)
    posture = posture_from_cfg(float(pelvis[2]), end_tilt, cfg)
    fallen = posture == "fallen"
    return {
        "settle_s": settle_s,
        "post_push_s": post_push_s,
        "inject_t_s": t_inject_s,
        "pre_push_pelvis_z_m": pre_z,
        "pre_push_tilt_rad": pre_tilt,
        "pre_push_posture": pre_posture,
        "name": case_name,
        "ang_vel_rad_s": [float(x) for x in omega],
        "kind": "one_shot_qvel",
        "source": cfg["angvel"]["source"],
        "injected": True,
        "end_pelvis_z_m": float(pelvis[2]),
        "min_pelvis_z_m": float(np.min(z_hist)),
        "end_tilt_rad": float(end_tilt),
        "max_tilt_rad": float(np.max(tilt_hist)),
        "posture": posture,
        "fallen": fallen,
        "fall_rate": 1.0 if fallen else 0.0,
        "time_to_fall_s": time_to_fall_s,
        "end_foot": env.foot_diagnostics(),
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "not_sustained_torque": True,
        "grasp_success_rate": None,
    }


def evaluate_angvel_sweep(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
) -> list[dict[str, Any]]:
    """Run one-shot qvel at each Table S4 angvel extremum. Not a SONIC gate."""
    axes = tuple(cfg.get("angvel_sweep", {}).get("axes", ["roll", "pitch", "yaw"]))
    cases = axis_aligned_angvel_extrema_rad_s(axes=axes)
    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(
            evaluate_angvel_push(
                env,
                cfg=cfg,
                settle_s=settle_s,
                post_push_s=post_push_s,
                ang_vel_rad_s=case["ang_vel_rad_s"],
                case_name=str(case["name"]),
            )
        )
    return out


def evaluate_sustained_torque(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
    ang_vel_rad_s: list[float] | np.ndarray,
    duration_s: float,
    case_name: str,
) -> dict[str, Any]:
    """Apply τ = I ω / T on LINK_BASE for Table S4 duration. Not a SONIC gate."""
    if env.pinned_base:
        raise ValueError("sustained torque requires pinned_base=False")
    if env.n_plane < 1:
        raise ValueError("sustained torque requires a floor plane")
    q_des = _q_des(env)
    omega = np.asarray(ang_vel_rad_s, dtype=np.float64).reshape(3)
    env.reset()
    env.set_q(q_des)
    env._mujoco.mj_forward(env.model, env.data)
    dt = float(env.model.opt.timestep)
    n_settle = int(round(settle_s / dt))
    n_torque = int(round(float(duration_s) / dt))
    n_post = int(round(post_push_s / dt))
    z_hist: list[float] = []
    tilt_hist: list[float] = []
    time_to_fall_s: float | None = None
    t = 0.0
    for i in range(max(n_settle, 1)):
        env.step(q_des)
        t = float(i + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    inertia = env.root_ang_inertia_kgm2()
    torque_nm = torque_nm_from_impulse(inertia.tolist(), omega.tolist(), float(duration_s))
    env.apply_root_torque_nm(np.asarray(torque_nm, dtype=np.float64))
    t_inject_s = t
    pre_z = float(z_hist[-1]) if z_hist else float("nan")
    pre_tilt = float(tilt_hist[-1]) if tilt_hist else float("nan")
    pre_posture = posture_from_cfg(pre_z, pre_tilt, cfg)
    for j in range(max(n_torque, 1)):
        env.step(q_des)
        t = t_inject_s + float(j + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    env.clear_root_force()
    t_release_s = t
    for k in range(max(n_post, 1)):
        env.step(q_des)
        t = t_release_s + float(k + 1) * dt
        pelvis, rot = env.body_pose("pelvis")
        z = float(pelvis[2])
        tilt = pelvis_tilt_rad(rot)
        z_hist.append(z)
        tilt_hist.append(tilt)
        if time_to_fall_s is None and fallen_from_cfg(z, tilt, cfg):
            time_to_fall_s = t
    pelvis, rot = env.body_pose("pelvis")
    end_tilt = pelvis_tilt_rad(rot)
    posture = posture_from_cfg(float(pelvis[2]), end_tilt, cfg)
    fallen = posture == "fallen"
    ang_impulse = [float(tau) * float(duration_s) for tau in torque_nm]
    return {
        "settle_s": settle_s,
        "duration_s": float(duration_s),
        "post_push_s": post_push_s,
        "inject_t_s": t_inject_s,
        "release_t_s": t_release_s,
        "pre_push_pelvis_z_m": pre_z,
        "pre_push_tilt_rad": pre_tilt,
        "pre_push_posture": pre_posture,
        "name": case_name,
        "ang_vel_rad_s": [float(x) for x in omega],
        "inertia_kgm2": [[float(x) for x in row] for row in inertia.tolist()],
        "torque_nm": [float(x) for x in torque_nm],
        "ang_impulse_nms": ang_impulse,
        "torque_formula": "tau = I @ omega / T",
        "kind": "sustained_torque",
        "source": cfg["sustained_torque"]["source"],
        "injected": True,
        "end_pelvis_z_m": float(pelvis[2]),
        "min_pelvis_z_m": float(np.min(z_hist)),
        "end_tilt_rad": float(end_tilt),
        "max_tilt_rad": float(np.max(tilt_hist)),
        "posture": posture,
        "fallen": fallen,
        "fall_rate": 1.0 if fallen else 0.0,
        "time_to_fall_s": time_to_fall_s,
        "end_foot": env.foot_diagnostics(),
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "not_one_shot_qvel": True,
        "grasp_success_rate": None,
    }


def evaluate_torque_sweep(
    env: T800MujocoEnv,
    *,
    cfg: dict[str, Any],
    settle_s: float,
    post_push_s: float,
) -> list[dict[str, Any]]:
    """Angvel extrema × duration extrema as τ = I ω / T. Not a SONIC gate."""
    axes = tuple(cfg.get("angvel_sweep", {}).get("axes", ["roll", "pitch", "yaw"]))
    cases = sustained_torque_cases(axes=axes)
    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(
            evaluate_sustained_torque(
                env,
                cfg=cfg,
                settle_s=settle_s,
                post_push_s=post_push_s,
                ang_vel_rad_s=case["ang_vel_rad_s"],
                duration_s=float(case["duration_s"]),
                case_name=str(case["name"]),
            )
        )
    return out


def _sweep_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, Any] = {}
    for c in cases:
        entry: dict[str, Any] = {
            "kind": c["kind"],
            "pre_push_posture": c["pre_push_posture"],
            "posture": c["posture"],
            "fallen": c["fallen"],
            "time_to_fall_s": c["time_to_fall_s"],
            "end_pelvis_z_m": c["end_pelvis_z_m"],
            "end_tilt_rad": c["end_tilt_rad"],
        }
        if "lin_vel_mps" in c:
            entry["lin_vel_mps"] = c["lin_vel_mps"]
        if "duration_s" in c:
            entry["duration_s"] = c["duration_s"]
        if "force_n" in c:
            entry["force_n"] = c["force_n"]
            entry["mass_kg"] = c["mass_kg"]
        if "ang_vel_rad_s" in c:
            entry["ang_vel_rad_s"] = c["ang_vel_rad_s"]
        if "torque_nm" in c:
            entry["torque_nm"] = c["torque_nm"]
        by_name[c["name"]] = entry
    return {
        "n_cases": len(cases),
        "names": [c["name"] for c in cases],
        "n_fallen": int(sum(1 for c in cases if c["fallen"])),
        "any_fallen": any(c["fallen"] for c in cases),
        "all_fallen": all(c["fallen"] for c in cases) if cases else False,
        "not_a_sonic_gate": True,
        "by_name": by_name,
    }


def run(*, source: str = "auto") -> dict[str, Any]:
    cfg_path = REPO_ROOT / "eval" / "configs" / "l2_freebase_push.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert_foot_frame()
    hold_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    hold = evaluate_freebase_stand(hold_env, hold_s=float(cfg["hold_s"]), cfg=cfg)
    push_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    sweep = evaluate_push_sweep(
        push_env,
        cfg=cfg,
        settle_s=float(cfg["settle_s"]),
        post_push_s=float(cfg["post_push_s"]),
    )
    plus_y = next((c for c in sweep if c["name"] == "+y"), None)
    if plus_y is None:
        raise ValueError("Table S4 planar sweep must include +y (ADR-026 compatibility)")
    force_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    force_sweep = evaluate_force_sweep(
        force_env,
        cfg=cfg,
        settle_s=float(cfg["settle_s"]),
        post_push_s=float(cfg["post_push_s"]),
    )
    plus_y_t1 = next((c for c in force_sweep if c["name"] == "+y_T1.0s"), None)
    if plus_y_t1 is None:
        raise ValueError("Table S4 force sweep must include +y_T1.0s (ADR-028 compatibility)")
    angvel_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    angvel_sweep = evaluate_angvel_sweep(
        angvel_env,
        cfg=cfg,
        settle_s=float(cfg["settle_s"]),
        post_push_s=float(cfg["post_push_s"]),
    )
    plus_yaw = next((c for c in angvel_sweep if c["name"] == "+yaw"), None)
    if plus_yaw is None:
        raise ValueError("Table S4 angvel sweep must include +yaw (ADR-029 compatibility)")
    torque_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    torque_sweep = evaluate_torque_sweep(
        torque_env,
        cfg=cfg,
        settle_s=float(cfg["settle_s"]),
        post_push_s=float(cfg["post_push_s"]),
    )
    plus_yaw_t1 = next((c for c in torque_sweep if c["name"] == "+yaw_T1.0s"), None)
    if plus_yaw_t1 is None:
        raise ValueError("Table S4 torque sweep must include +yaw_T1.0s (ADR-029 compatibility)")
    vert_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    vertical_sweep = evaluate_vertical_sweep(
        vert_env,
        cfg=cfg,
        settle_s=float(cfg["settle_s"]),
        post_push_s=float(cfg["post_push_s"]),
    )
    plus_z = next((c for c in vertical_sweep if c["name"] == "+z"), None)
    if plus_z is None:
        raise ValueError("Table S4 vertical sweep must include +z (ADR-030 compatibility)")
    vert_force_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=True)
    vertical_force_sweep = evaluate_vertical_force_sweep(
        vert_force_env,
        cfg=cfg,
        settle_s=float(cfg["settle_s"]),
        post_push_s=float(cfg["post_push_s"]),
    )
    plus_z_t1 = next((c for c in vertical_force_sweep if c["name"] == "+z_T1.0s"), None)
    if plus_z_t1 is None:
        raise ValueError("Table S4 vertical force sweep must include +z_T1.0s (ADR-030 compatibility)")
    air_env = T800MujocoEnv(source=source, pinned_base=False, add_floor=False)
    airdrop = evaluate_airdrop(
        air_env,
        hold_s=float(cfg["airdrop_s"]),
        min_drop_m=float(cfg["airdrop"]["min_drop_m"]),
    )
    report = {
        "status": "freebase_pd_push",
        "physics": "mujoco",
        "source": hold_env.source,
        "xml_note": hold_env.xml_note,
        "gains_source": hold_env.gains_source,
        "pinned_base": False,
        "foot_frame": FOOT_FRAME_DECISION,
        "floor_friction": "official_wbc_collision_default_not_pad_cardboard",
        "lean_tilt_rad": float(cfg["lean"]["tilt_rad"]),
        "fall": cfg["fall"],
        "hold": hold,
        "lateral_push": plus_y,
        "push_sweep": sweep,
        "push_sweep_summary": _sweep_summary(sweep),
        "sustained_force": plus_y_t1,
        "force_sweep": force_sweep,
        "force_sweep_summary": _sweep_summary(force_sweep),
        "angvel_push": plus_yaw,
        "angvel_sweep": angvel_sweep,
        "angvel_sweep_summary": _sweep_summary(angvel_sweep),
        "sustained_torque": plus_yaw_t1,
        "torque_sweep": torque_sweep,
        "torque_sweep_summary": _sweep_summary(torque_sweep),
        "vertical_push": plus_z,
        "vertical_sweep": vertical_sweep,
        "vertical_sweep_summary": _sweep_summary(vertical_sweep),
        "sustained_vertical": plus_z_t1,
        "vertical_force_sweep": vertical_force_sweep,
        "vertical_force_sweep_summary": _sweep_summary(vertical_force_sweep),
        "airdrop": airdrop,
        "local_tracking_success": False,
        "not_a_sonic_gate": True,
        "bringup_pd_is_not_a_balance_controller": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "success_rate_note": "lean vs fall vs planar/vertical/angvel sweep vs air-drop are diagnostics; not SONIC gates",
    }
    refuse_grasp_success_key(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto", choices=("auto", "official", "fixture"))
    parser.add_argument("--out", default="eval/report/generated/l2_freebase_push.json")
    args = parser.parse_args()
    try:
        refuse_combined_robot()
    except PolicyEvalBlocked:
        pass
    report = run(source=args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "status": report["status"],
        "source": report["source"],
        "hold_posture": report["hold"]["posture"],
        "hold_fallen": report["hold"]["fallen"],
        "hold_end_tilt_rad": report["hold"]["end_tilt_rad"],
        "push_posture": report["lateral_push"]["posture"],
        "push_pre_posture": report["lateral_push"]["pre_push_posture"],
        "push_fallen": report["lateral_push"]["fallen"],
        "push_sweep_names": report["push_sweep_summary"]["names"],
        "push_sweep_n_fallen": report["push_sweep_summary"]["n_fallen"],
        "force_sweep_names": report["force_sweep_summary"]["names"],
        "force_sweep_n_fallen": report["force_sweep_summary"]["n_fallen"],
        "sustained_kind": report["sustained_force"]["kind"],
        "sustained_duration_s": report["sustained_force"]["duration_s"],
        "angvel_sweep_names": report["angvel_sweep_summary"]["names"],
        "angvel_sweep_n_fallen": report["angvel_sweep_summary"]["n_fallen"],
        "torque_sweep_names": report["torque_sweep_summary"]["names"],
        "torque_sweep_n_fallen": report["torque_sweep_summary"]["n_fallen"],
        "sustained_torque_kind": report["sustained_torque"]["kind"],
        "sustained_torque_duration_s": report["sustained_torque"]["duration_s"],
        "vertical_sweep_names": report["vertical_sweep_summary"]["names"],
        "vertical_sweep_n_fallen": report["vertical_sweep_summary"]["n_fallen"],
        "vertical_force_sweep_names": report["vertical_force_sweep_summary"]["names"],
        "vertical_force_sweep_n_fallen": report["vertical_force_sweep_summary"]["n_fallen"],
        "sustained_vertical_kind": report["sustained_vertical"]["kind"],
        "sustained_vertical_duration_s": report["sustained_vertical"]["duration_s"],
        "airdrop_freejoint_moved": report["airdrop"]["freejoint_moved"],
        "airdrop_drop_m": report["airdrop"]["drop_m"],
        "not_a_sonic_gate": report["not_a_sonic_gate"],
        "grasp_success_rate": report["grasp_success_rate"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
