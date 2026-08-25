"""Industrial cell with mj_step, no prop teleports (ADR-007 off).

Arm PD tracks IK waypoints; hands use MIT. Boxes/gun move only via physics.
Relative metrics only — never grasp_success_rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hand.grasp_primitives import GraspLibrary
from interface.schema import load_hand_spec
from sim.mujoco_env.env import CombinedMujocoEnv
from sim.mujoco_env.ik import dls_ik_pos
from sim.mujoco_env.scene import SceneSpec
from sim.qr_scanner import ScanSpec, simulate_scan
from sim.tasks.stack_metrics import box_speed_m_s, release_sample


@dataclass
class PhysicsIndustrialResult:
    phases: list[str] = field(default_factory=list)
    ik_err_m: list[float] = field(default_factory=list)
    box0_z: list[float] = field(default_factory=list)
    n_contacts_max: int = 0
    n_steps: int = 0
    finite: bool = True
    kinematic_assist: bool = False
    policy_eval_forbidden: bool = True
    scan_geometry_ok: bool = False
    scan_decode_ok: bool | None = None
    scan_distance_m: float | None = None
    release_speed_m_s: float | None = None
    stack_alignment_m: float | None = None
    gap_z_m: float | None = None
    box_drop_m: float | None = None
    note: str = (
        "Physics industrial attempt (mj_step, no kinematic assists). "
        "Uncalibrated contact and identity flange. Relative metrics only."
    )


def _ik(env: CombinedMujocoEnv, left_tgt, right_tgt, left_arm, right_arm) -> list[float]:
    errs: list[float] = []
    if left_tgt is not None:
        r = dls_ik_pos(env.model, env.data, "l_wrist", left_tgt, left_arm, damping=0.05, iters=12, step=0.4)
        errs.append(r["err_m"])
    if right_tgt is not None:
        r = dls_ik_pos(env.model, env.data, "r_wrist", right_tgt, right_arm, damping=0.05, iters=12, step=0.4)
        errs.append(r["err_m"])
    return errs


def run_physics_industrial(
    env: CombinedMujocoEnv | None = None,
    *,
    steps_per_phase: int = 20,
    substeps: int = 8,
) -> PhysicsIndustrialResult:
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
    result = PhysicsIndustrialResult()
    scene = SceneSpec()
    box0 = env.xpos("box_0").copy()
    z0 = float(box0[2])
    pallet = env.xpos("pallet").copy()
    gun0 = env.xpos("scan_gun").copy()
    box_hz = scene.box_size_m[2] / 2.0
    pallet_h = scene.pad_size_m[2]
    stack = pallet + np.array([0.0, 0.0, pallet_h / 2.0 + box_hz])

    approach_l = box0 + np.array([0.0, 0.15, 0.05])
    approach_r = box0 + np.array([0.0, -0.15, 0.05])
    grasp_l = box0 + np.array([0.0, 0.11, 0.01])
    grasp_r = box0 + np.array([0.0, -0.11, 0.01])
    lift_l = box0 + np.array([0.0, 0.11, 0.20])
    lift_r = box0 + np.array([0.0, -0.11, 0.20])
    mid = 0.5 * (box0 + stack)
    carry_l = mid + np.array([0.0, 0.11, 0.22])
    carry_r = mid + np.array([0.0, -0.11, 0.22])
    stack_l = stack + np.array([0.0, 0.11, 0.10])
    stack_r = stack + np.array([0.0, -0.11, 0.10])
    park_l = np.array([0.12, 0.28, 0.95])
    gun_reach = gun0 + np.array([0.02, 0.06, 0.03])
    scan_r = stack + np.array([-0.14, -0.12, 0.03])

    def _phase(name: str) -> None:
        result.phases.append(name)

    def _tick(n: int, left_tgt, right_tgt, left_q, right_q) -> None:
        for _ in range(n):
            q_before = env.data.qpos.copy()
            errs = _ik(env, left_tgt, right_tgt, left_arm, right_arm)
            result.ik_err_m.extend(errs)
            q_des = env.data.qpos.copy()
            env.data.qpos[:] = q_before
            env.data.qvel[:] *= 0.2
            for _s in range(substeps):
                env.step_mit(left_q, right_q, body_q_des=q_des)
            result.n_steps += 1
            result.n_contacts_max = max(result.n_contacts_max, int(env.data.ncon))
            result.finite = result.finite and bool(np.isfinite(env.data.qpos).all())
            result.box0_z.append(float(env.xpos("box_0")[2]))

    _phase("approach_box")
    _tick(steps_per_phase, approach_l, approach_r, q_open, q_open)
    _phase("grasp")
    for i in range(steps_per_phase):
        t = (i + 1) / steps_per_phase
        q = (1 - t) * q_open + t * q_power
        _tick(1, grasp_l, grasp_r, q, q)
    _phase("lift")
    _tick(steps_per_phase, lift_l, lift_r, q_power, q_power)
    _phase("carry")
    _tick(steps_per_phase, carry_l, carry_r, q_power, q_power)
    _phase("stack")
    _tick(steps_per_phase, stack_l, stack_r, q_power, q_power)

    cube_id = int(env.model.body("box_0").id)
    sample = release_sample(
        speed_m_s=box_speed_m_s(env.data, cube_id),
        box_bottom_z_m=float(env.xpos("box_0")[2]) - box_hz,
        support_top_z_m=float(pallet[2]) + pallet_h / 2.0,
        box_xy=env.xpos("box_0"),
        support_xy=pallet,
        xmat=env.data.xmat[cube_id],
    )
    result.release_speed_m_s = sample.release_speed_m_s
    result.stack_alignment_m = sample.alignment_xy_m
    result.gap_z_m = sample.gap_z_m

    _phase("release")
    n_rel = max(8, steps_per_phase // 2)
    for i in range(n_rel):
        t = (i + 1) / n_rel
        q = (1 - t) * q_power + t * q_open
        _tick(1, stack_l, stack_r, q, q)

    _phase("approach_gun")
    _tick(steps_per_phase, park_l, gun_reach, q_open, q_open)
    _phase("grip_gun")
    n_grip = max(8, steps_per_phase // 2)
    for i in range(n_grip):
        t = (i + 1) / n_grip
        _tick(1, park_l, gun_reach, q_open, (1 - t) * q_open + t * q_gun)
    _phase("scan")
    _tick(steps_per_phase, park_l, scan_r, q_open, q_gun)

    mujoco.mj_forward(env.model, env.data)
    try:
        gun_pos = env.site_xpos("gun_tcp")
        qr_pos = env.site_xpos("box_0_qr")
        sid = int(env.model.site("gun_tcp").id)
        gun_z = env.data.site_xmat[sid].reshape(3, 3)[:, 2].copy()
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        from sim.mujoco_env.render_cam import render_camera

        frame, meta = render_camera(env.model, env.data, "gun_cam")
        if meta.get("ok") and frame is not None:
            rgb = frame
        scan = simulate_scan(
            rgb,
            gun_tcp_pos_m=gun_pos,
            gun_tcp_z=gun_z,
            qr_pos_m=qr_pos,
            qr_normal=np.array([-1.0, 0.0, 0.0]),
            rel_speed_m_s=0.0,
            spec=ScanSpec(d_min_m=0.05, d_max_m=0.40, theta_max_rad=np.deg2rad(70.0)),
        )
        result.scan_geometry_ok = scan.geometry_ok
        result.scan_distance_m = scan.distance_m
        result.scan_decode_ok = scan.decode_ok if meta.get("ok") else None
    except Exception as exc:  # noqa: BLE001
        result.note += f" scan_error={exc}"

    result.box_drop_m = z0 - float(env.xpos("box_0")[2])
    result.phases.append("done")
    return result
