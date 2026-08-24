"""L2.2 single-hand grasp micro-env.

A small cube (not an industrial carton) sits under a fixed-base right Hand 2.
The carrier slides up after a power grasp. Metrics are uncalibrated relatives
across a μ × solref grid — never published as grasp_success_rate (ADR-004).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from assets.combined.assemble import convert_position_actuators
from assets.combined.mjcf_xml import dump_mjcf, load_mjcf
from assets.dexhand2.build.ingest_official import official_mjcf
from hand.grasp_primitives import GraspLibrary
from interface.schema import load_hand_spec
from sim.mujoco_env.hand_mit import apply_mit, lookup_hand_actuators


@dataclass
class MicroMetrics:
    friction: float
    solref_timeconst_s: float
    slip_m: float
    lift_m: float
    cube_drop_m: float
    max_abs_tau: float
    n_contacts: int
    finite: bool


def _micro_xml(friction: float, solref_s: float) -> tuple[str, list[dict]]:
    src = official_mjcf("right", with_mount=True)
    text, gains = convert_position_actuators(src.read_text(encoding="utf-8"))
    resolved = src.parent / "_micro_right.xml"
    # Write next to official so relative meshdir still resolves, then load_mjcf absolutizes.
    resolved.write_text(text, encoding="utf-8")
    try:
        hand = load_mjcf(resolved)
    finally:
        resolved.unlink(missing_ok=True)

    root = ET.Element("mujoco", model="dexhand2_l22_micro")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast")
    asset = ET.SubElement(root, "asset")
    hand_asset = hand.find("asset")
    if hand_asset is not None:
        for child in list(hand_asset):
            asset.append(child)
    world = ET.SubElement(root, "worldbody")
    world.append(
        ET.fromstring('<geom name="table" type="box" size="0.15 0.15 0.02" pos="0 0 0.02" rgba="0.5 0.4 0.3 1"/>')
    )
    solref = f"{solref_s} {solref_s * 2}"
    fr = f"{friction} {friction * 0.1} 0.001"
    world.append(
        ET.fromstring(
            '<body name="cube" pos="0.0 -0.02 0.07">'
            '<inertial pos="0 0 0" mass="0.15" diaginertia="0.00008 0.00008 0.00008"/>'
            "<freejoint/>"
            f'<geom name="cube_geom" type="box" size="0.025 0.025 0.025" rgba="0.8 0.5 0.2 1" '
            f'friction="{fr}" solref="{solref}" condim="4"/>'
            "</body>"
        )
    )
    carrier = ET.fromstring(
        '<body name="hand_carrier" pos="0 0 0.18">'
        '<joint name="hand_lift" type="slide" axis="0 0 1" range="0 0.15"/>'
        "</body>"
    )
    mount = None
    hworld = hand.find("worldbody")
    for body in list(hworld):
        if body.tag == "body" and body.get("name") == "r_mount":
            mount = body
            break
    if mount is None:
        raise ValueError("r_mount missing")
    mount.set("pos", "0 0 0")
    carrier.append(mount)
    world.append(carrier)
    act = ET.SubElement(root, "actuator")
    hand_act = hand.find("actuator")
    if hand_act is not None:
        for child in list(hand_act):
            act.append(child)
    act.append(
        ET.fromstring(
            '<motor name="hand_lift_motor" joint="hand_lift" gear="1" ctrllimited="true" ctrlrange="-20 20"/>'
        )
    )
    contact = ET.SubElement(root, "contact")
    hand_contact = hand.find("contact")
    if hand_contact is not None:
        for child in list(hand_contact):
            contact.append(child)
    return dump_mjcf(root), gains


def run_micro_episode(
    friction: float,
    solref_s: float,
    *,
    n_close: int = 200,
    n_hold: int = 200,
    lift_m: float = 0.08,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
) -> MicroMetrics:
    import mujoco

    xml, gains = _micro_xml(friction, solref_s)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    plant = lookup_hand_actuators(model, gains, "right")
    spec = load_hand_spec()
    lib = GraspLibrary(spec)
    q_open = lib.q_active("open", 0.0)
    q_closed = lib.q_active("power_grasp", 1.0)
    lift_id = int(model.actuator("hand_lift_motor").id)
    lift_j = int(model.joint("hand_lift").id)
    lift_qadr = int(model.jnt_qposadr[lift_j])
    cube_id = int(model.body("cube").id)
    mujoco.mj_forward(model, data)
    z0 = float(data.xpos[cube_id][2])
    xy0 = data.xpos[cube_id][:2].copy()
    max_tau = 0.0
    n_con = 0

    def _step(q_des: np.ndarray, lift_target: float) -> None:
        nonlocal max_tau, n_con
        tau = apply_mit(model, data, plant, q_des, kp_scale=kp_scale, kd_scale=kd_scale)
        max_tau = max(max_tau, float(np.max(np.abs(tau))))
        # PD on the lift slider
        q = data.qpos[lift_qadr]
        dq = data.qvel[int(model.jnt_dofadr[lift_j])]
        data.ctrl[lift_id] = float(np.clip(80.0 * (lift_target - q) - 4.0 * dq, -20, 20))
        mujoco.mj_step(model, data)
        n_con = max(n_con, int(data.ncon))

    for k in range(n_close):
        a = (k + 1) / n_close
        _step((1 - a) * q_open + a * q_closed, 0.0)
    for _ in range(n_hold):
        _step(q_closed, lift_m)
    mujoco.mj_forward(model, data)
    z1 = float(data.xpos[cube_id][2])
    xy1 = data.xpos[cube_id][:2]
    return MicroMetrics(
        friction=friction,
        solref_timeconst_s=solref_s,
        slip_m=float(np.linalg.norm(xy1 - xy0)),
        lift_m=float(data.qpos[lift_qadr]),
        cube_drop_m=float(z0 - z1),
        max_abs_tau=max_tau,
        n_contacts=n_con,
        finite=bool(np.isfinite(data.qpos).all()),
    )


def run_scan_grid(cfg: dict, *, n_close: int = 80, n_hold: int = 80) -> list[dict]:
    rows = []
    for mu in cfg["friction_static"]:
        for sol in cfg["solref_timeconst_s"]:
            m = run_micro_episode(float(mu), float(sol), n_close=n_close, n_hold=n_hold, lift_m=float(cfg.get("lift_m", 0.08)))
            rows.append(
                {
                    "friction_static": m.friction,
                    "solref_timeconst_s": m.solref_timeconst_s,
                    "uncalibrated_slip_m": m.slip_m,
                    "uncalibrated_cube_drop_m": m.cube_drop_m,
                    "uncalibrated_n_contacts": m.n_contacts,
                    "max_abs_tau_nm": m.max_abs_tau,
                    "finite": m.finite,
                    "label": "SCAN_PLACEHOLDER",
                }
            )
    return rows
