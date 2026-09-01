"""Industrial cell with mj_step, gravity-compensated arm PD, contact-gated welds.

Arm PD tracks IK waypoints (τ = kp e − kd q̇ + qfrc_bias + Jᵀ F_payload).
Hands use MIT. Carton/gun are free bodies.

T800 has no wrist pitch/roll (ADR-001): DexHand pads hang ~15 cm below the
wrist in −Z, so a straight hang→box IK drives the fingers through the bench.
Playback raises the wrists first, then overlays from above, then closes.

A weld snapshots the relative pose only after DexHand pad/hull contact — not
a no-touch lift, not Coulomb E1/E2, not grasp_success_rate (ADR-004).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hand.grasp_primitives import GraspLibrary
from interface.schema import load_hand_spec
from sim.mujoco_env.contacts import hand_object_contacts, pad_object_contacts
from sim.mujoco_env.dynamics import rate_limit_qpos
from sim.mujoco_env.env import CombinedMujocoEnv
from sim.mujoco_env.ik import dls_ik_pos
from sim.mujoco_env.scene import SceneSpec
from sim.mujoco_env.welds import has_equality, set_weld_active
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
    constraint_weld: bool = True
    policy_eval_forbidden: bool = True
    scan_geometry_ok: bool = False
    scan_decode_ok: bool | None = None
    scan_distance_m: float | None = None
    release_speed_m_s: float | None = None
    stack_alignment_m: float | None = None
    gap_z_m: float | None = None
    box_drop_m: float | None = None
    wrist_track_err_m: list[float] = field(default_factory=list)
    max_abs_tau_nm: float = 0.0
    n_pad_box_contacts: int = 0
    n_hand_box_contacts: int = 0
    n_pad_gun_contacts: int = 0
    n_hand_gun_contacts: int = 0
    box_weld_from_contact: bool = False
    gun_weld_from_contact: bool = False
    box_weld_side: str | None = None
    note: str = (
        "Physics industrial (mj_step + gravity-compensated PD). "
        "Raise-then-overlay so DexHand pads meet the carton/gun; welds snapshot "
        "only after that contact. Constraint grasp, not E1/E2 Coulomb. "
        "Identity flange. Relative metrics only."
    )


def _ik(env: CombinedMujocoEnv, left_tgt, right_tgt, left_arm, right_arm) -> list[float]:
    errs: list[float] = []
    if left_tgt is not None:
        r = dls_ik_pos(
            env.model,
            env.data,
            "l_wrist",
            left_tgt,
            left_arm,
            damping=0.05,
            iters=18,
            step=0.52,
            max_delta_rad=0.10,
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
            iters=18,
            step=0.52,
            max_delta_rad=0.10,
        )
        errs.append(r["err_m"])
    return errs


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64)


def run_physics_industrial(
    env: CombinedMujocoEnv | None = None,
    *,
    steps_per_phase: int = 32,
    substeps: int = 24,
    use_welds: bool = True,
    settle_steps: int = 200,
    on_step: Callable | None = None,
    on_phase: Callable | None = None,
) -> PhysicsIndustrialResult:
    import mujoco

    env = env or CombinedMujocoEnv(scene="industrial")
    env.reset()
    spec = load_hand_spec()
    lib = GraspLibrary(spec)
    q_open = lib.q_active("open", 0.0)
    q_power = lib.q_active("power_grasp", 0.85)
    q_gun = lib.q_active("gun_grip", 0.90)
    left_arm = env.handles.arm_joints["left"]
    right_arm = env.handles.arm_joints["right"]
    result = PhysicsIndustrialResult(constraint_weld=use_welds)
    if not use_welds:
        result.note = (
            "Physics industrial with welds off: carton/gun move only via contact. "
            "Uncalibrated μ — the box is expected to drop. Relative metrics only."
        )
        result.constraint_weld = False
    scene = SceneSpec()
    box0 = env.xpos("box_0").copy()
    z0 = float(box0[2])
    pallet = env.xpos("pallet").copy()
    gun0 = env.xpos("scan_gun").copy()
    box_hz = scene.box_size_m[2] / 2.0
    pallet_h = scene.pad_size_m[2]
    stack = pallet + np.array([0.0, 0.0, pallet_h / 2.0 + box_hz])

    payload_body: str | None = None
    payload_mass = 0.0
    act_ids = env.handles.body_actuator_ids

    hold_q = env.data.qpos.copy()
    for _ in range(max(1, settle_steps)):
        env.step_mit(q_open, q_open, body_q_des=hold_q)

    hang_l = env.xpos("l_wrist").copy()
    hang_r = env.xpos("r_wrist").copy()
    box0 = env.xpos("box_0").copy()
    gun0 = env.xpos("scan_gun").copy()
    # Straight up first: pads hang ~0.15 m below the wrist, bench top is 0.88 m.
    raise_l = hang_l + np.array([0.0, 0.0, 0.26])
    raise_r = hang_r + np.array([0.0, 0.0, 0.26])
    # Overhead wrap: IK rotates the dummy wrist, so hang-frame "pads 15 cm
    # below" is false at the carton. Approach stays high (open, no hit);
    # grasp is a tighter, lower overlay that puts pads on the top face.
    approach_l = box0 + np.array([-0.08, 0.08, 0.22])
    approach_r = box0 + np.array([-0.08, -0.08, 0.22])
    grasp_l = box0 + np.array([-0.08, 0.04, 0.08])
    grasp_r = box0 + np.array([-0.08, -0.04, 0.08])
    lift_l = grasp_l + np.array([0.0, 0.0, 0.16])
    lift_r = grasp_r + np.array([0.0, 0.0, 0.16])
    mid = 0.5 * (box0 + stack)
    carry_l = mid + np.array([-0.08, 0.08, 0.22])
    carry_r = mid + np.array([-0.08, -0.08, 0.22])
    hover_l = stack + np.array([-0.08, 0.08, 0.20])
    hover_r = stack + np.array([-0.08, -0.08, 0.20])
    stack_l = stack + np.array([-0.08, 0.08, 0.16])
    stack_r = stack + np.array([-0.08, -0.08, 0.16])
    park_l = raise_l.copy()
    gun_high = gun0 + np.array([-0.04, -0.06, 0.18])
    gun_reach = gun0 + np.array([-0.04, -0.06, 0.08])
    scan_r = stack + np.array([-0.10, -0.10, 0.04])

    def _phase(name: str) -> None:
        result.phases.append(name)
        if on_phase is not None:
            on_phase(name, env, result)

    def _tick(
        n: int,
        left_tgt,
        right_tgt,
        left_q,
        right_q,
        *,
        lerp_cart: bool = False,
        kp_scale: float = 1.0,
        body_kp: float = 280.0,
        body_kd: float = 24.0,
        max_dq: float = 0.10,
        max_err_rad: float = 1.2,
    ) -> None:
        nonlocal payload_body, payload_mass
        left0 = env.xpos("l_wrist").copy()
        right0 = env.xpos("r_wrist").copy()
        for i in range(n):
            t = (i + 1) / max(n, 1)
            l_goal = left_tgt
            r_goal = right_tgt
            if lerp_cart:
                l_goal = _lerp(left0, left_tgt, t) if left_tgt is not None else None
                r_goal = _lerp(right0, right_tgt, t) if right_tgt is not None else None
            q_before = env.data.qpos.copy()
            errs = _ik(env, l_goal, r_goal, left_arm, right_arm)
            result.ik_err_m.extend(errs)
            q_ik = env.data.qpos.copy()
            env.data.qpos[:] = q_before
            mujoco.mj_forward(env.model, env.data)
            if l_goal is not None:
                result.wrist_track_err_m.append(float(np.linalg.norm(env.xpos("l_wrist") - l_goal)))
            if r_goal is not None:
                result.wrist_track_err_m.append(float(np.linalg.norm(env.xpos("r_wrist") - r_goal)))
            q_cmd = q_before
            for _s in range(substeps):
                q_cmd = rate_limit_qpos(q_cmd, q_ik, max_dq=max_dq)
                env.step_mit(
                    left_q,
                    right_q,
                    body_q_des=q_cmd,
                    body_kp=body_kp,
                    body_kd=body_kd,
                    kp_scale=kp_scale,
                    payload_body=payload_body,
                    payload_mass_kg=payload_mass,
                    max_err_rad=max_err_rad,
                )
            if act_ids.size:
                result.max_abs_tau_nm = max(
                    result.max_abs_tau_nm,
                    float(np.max(np.abs(env.data.ctrl[act_ids]))),
                )
            result.n_steps += 1
            result.n_contacts_max = max(result.n_contacts_max, int(env.data.ncon))
            result.finite = result.finite and bool(np.isfinite(env.data.qpos).all())
            result.box0_z.append(float(env.xpos("box_0")[2]))
            _box_contacts()
            _gun_contacts()
            if on_step is not None:
                on_step(env, result)

    can_weld = use_welds and has_equality(env.model, "weld_box_grasp")

    def _box_contacts() -> tuple[int, int, int, int]:
        nl = hand_object_contacts(env.model, env.data, "left", "box_0")
        nr = hand_object_contacts(env.model, env.data, "right", "box_0")
        pl = pad_object_contacts(env.model, env.data, "left", "box_0")
        pr = pad_object_contacts(env.model, env.data, "right", "box_0")
        result.n_hand_box_contacts = max(result.n_hand_box_contacts, nl + nr)
        result.n_pad_box_contacts = max(result.n_pad_box_contacts, pl + pr)
        return nl, nr, pl, pr

    def _gun_contacts() -> tuple[int, int]:
        n = hand_object_contacts(env.model, env.data, "right", "scan_gun")
        p = pad_object_contacts(env.model, env.data, "right", "scan_gun")
        result.n_hand_gun_contacts = max(result.n_hand_gun_contacts, n)
        result.n_pad_gun_contacts = max(result.n_pad_gun_contacts, p)
        return n, p

    def _maybe_weld_box() -> None:
        nonlocal payload_body, payload_mass
        if not can_weld:
            return
        nl, nr, pl, pr = _box_contacts()
        # Dual-arm wrap often has 1 pad per hand; require two DexHand–carton
        # contacts in the same tick. Empty-air lift was the last-trajectory bug.
        if (pl + pr) < 2 and (nl + nr) < 2:
            result.note += " box_weld_skipped_no_pad_contact."
            return
        side = "right" if (pr, nr) >= (pl, nl) else "left"
        name = "weld_box_grasp_r" if side == "right" else "weld_box_grasp"
        if not has_equality(env.model, name):
            name = "weld_box_grasp"
            side = "left"
        mujoco.mj_forward(env.model, env.data)
        set_weld_active(env.model, env.data, name, True)
        payload_body = "box_0"
        payload_mass = float(scene.box_mass_kg)
        result.box_weld_from_contact = True
        result.box_weld_side = side

    def _maybe_weld_gun() -> None:
        nonlocal payload_body, payload_mass
        if not (use_welds and has_equality(env.model, "weld_gun_grasp")):
            return
        n, p = _gun_contacts()
        if (n + p) < 2:
            result.note += " gun_weld_skipped_no_contact."
            return
        mujoco.mj_forward(env.model, env.data)
        set_weld_active(env.model, env.data, "weld_gun_grasp", True)
        payload_body = "scan_gun"
        payload_mass = 0.38
        result.gun_weld_from_contact = True

    _phase("raise")
    _tick(steps_per_phase, raise_l, raise_r, q_open, q_open, max_dq=0.12)
    _phase("approach_box")
    _tick(steps_per_phase, approach_l, approach_r, q_open, q_open)
    _phase("grasp")
    for i in range(steps_per_phase):
        t = (i + 1) / steps_per_phase
        q = (1 - t) * q_open + t * q_power
        _tick(1, grasp_l, grasp_r, q, q, kp_scale=1.4)
    _phase("squeeze")
    n_sq = max(8, steps_per_phase)
    for _ in range(n_sq):
        _tick(1, grasp_l, grasp_r, q_power, q_power, kp_scale=1.8, body_kp=320.0)
        nl, nr, pl, pr = _box_contacts()
        if (pl + pr) >= 2 or (nl + nr) >= 2:
            _maybe_weld_box()
            break
    else:
        _maybe_weld_box()
    _phase("lift")
    _tick(steps_per_phase, lift_l, lift_r, q_power, q_power)
    _phase("carry")
    _tick(steps_per_phase, carry_l, carry_r, q_power, q_power)
    _phase("stack")
    n_hover = max(4, steps_per_phase // 2)
    n_lower = max(4, steps_per_phase - n_hover)
    _tick(n_hover, hover_l, hover_r, q_power, q_power)
    _tick(n_lower, stack_l, stack_r, q_power, q_power)

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

    if result.box_weld_from_contact:
        if has_equality(env.model, "weld_box_grasp"):
            set_weld_active(env.model, env.data, "weld_box_grasp", False)
        if has_equality(env.model, "weld_box_grasp_r"):
            set_weld_active(env.model, env.data, "weld_box_grasp_r", False)
        payload_body = None
        payload_mass = 0.0
    _phase("release")
    n_rel = max(8, steps_per_phase // 2)
    for i in range(n_rel):
        t = (i + 1) / n_rel
        q = (1 - t) * q_power + t * q_open
        _tick(1, stack_l, stack_r, q, q)
    # Let the carton settle on the pallet under gravity + friction.
    hold_after = env.data.qpos.copy()
    for _ in range(max(20, substeps * 4)):
        env.step_mit(q_open, q_open, body_q_des=hold_after)

    _phase("approach_gun")
    _tick(steps_per_phase, park_l, gun_high, q_open, q_open)
    _phase("grip_gun")
    n_grip = max(8, steps_per_phase // 2)
    _tick(max(4, n_grip // 2), park_l, gun_reach, q_open, q_open, kp_scale=1.2)
    for i in range(n_grip):
        t = (i + 1) / n_grip
        _tick(
            1,
            park_l,
            gun_reach,
            q_open,
            (1 - t) * q_open + t * q_gun,
            kp_scale=1.6,
        )
    _tick(max(4, n_grip // 2), park_l, gun_reach, q_open, q_gun, kp_scale=1.8, body_kp=320.0)
    for _ in range(max(6, n_grip)):
        _tick(1, park_l, gun_reach, q_open, q_gun, kp_scale=1.8, body_kp=320.0)
        n, p = _gun_contacts()
        if (n + p) >= 2:
            _maybe_weld_gun()
            break
    else:
        _maybe_weld_gun()
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
    if on_phase is not None:
        on_phase("done", env, result)
    return result
