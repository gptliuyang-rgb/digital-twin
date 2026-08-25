"""Two fixed-base Hands squeezing a small box (contact micro, not industrial cell).

Relative slip/drop/contact counts only. Never writes grasp_success_rate (ADR-004).
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
class BimanualMetrics:
    friction: float
    solref_timeconst_s: float
    slip_m: float
    cube_drop_m: float
    n_contacts: int
    max_abs_tau: float
    finite: bool
    policy_eval_forbidden: bool = True


def _mount_body(hand_root: ET.Element, name: str) -> ET.Element:
    world = hand_root.find("worldbody")
    if world is None:
        raise ValueError("no worldbody")
    for body in list(world):
        if body.tag == "body" and body.get("name") == name:
            return body
    raise ValueError(name)


def _hand_xml(side: str) -> tuple[ET.Element, list[dict]]:
    src = official_mjcf(side, with_mount=True)
    text = src.read_text(encoding="utf-8")
    text, gains = convert_position_actuators(text)
    tmp = src.parent / f"_bimanual_{side}.xml"
    tmp.write_text(text, encoding="utf-8")
    try:
        return load_mjcf(tmp), gains
    finally:
        tmp.unlink(missing_ok=True)


def _scene_xml(friction: float, solref_s: float) -> tuple[str, list[dict]]:
    left, g_l = _hand_xml("left")
    right, g_r = _hand_xml("right")
    gains = g_l + g_r
    root = ET.Element("mujoco", model="dexhand2_bimanual_micro")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast")
    asset = ET.SubElement(root, "asset")
    for tree in (left, right):
        a = tree.find("asset")
        if a is not None:
            for child in list(a):
                asset.append(child)
    world = ET.SubElement(root, "worldbody")
    solref = f"{solref_s} {solref_s * 2}"
    fr = f"{friction} {friction * 0.1} 0.001"
    world.append(
        ET.fromstring(
            '<geom name="table" type="box" size="0.18 0.18 0.02" pos="0 0 0.02" '
            f'rgba="0.5 0.4 0.3 1" friction="{fr}" solref="{solref}"/>'
        )
    )
    world.append(
        ET.fromstring(
            '<body name="cube" pos="0 0 0.075">'
            '<inertial pos="0 0 0" mass="0.20" diaginertia="0.00012 0.00012 0.00012"/>'
            "<freejoint/>"
            f'<geom name="cube_geom" type="box" size="0.03 0.04 0.03" rgba="0.8 0.5 0.2 1" '
            f'friction="{fr}" solref="{solref}" condim="4"/>'
            "</body>"
        )
    )
    left_carrier = ET.fromstring(
        '<body name="left_carrier" pos="0 0.11 0.16">'
        '<joint name="left_lift" type="slide" axis="0 0 1" range="0 0.12"/>'
        "</body>"
    )
    right_carrier = ET.fromstring(
        '<body name="right_carrier" pos="0 -0.11 0.16">'
        '<joint name="right_lift" type="slide" axis="0 0 1" range="0 0.12"/>'
        "</body>"
    )
    l_mount = _mount_body(left, "l_mount")
    r_mount = _mount_body(right, "r_mount")
    l_mount.set("pos", "0 0 0")
    r_mount.set("pos", "0 0 0")
    left_carrier.append(l_mount)
    right_carrier.append(r_mount)
    world.append(left_carrier)
    world.append(right_carrier)
    act = ET.SubElement(root, "actuator")
    for tree in (left, right):
        ha = tree.find("actuator")
        if ha is not None:
            for child in list(ha):
                act.append(child)
    act.append(
        ET.fromstring(
            '<motor name="left_lift_motor" joint="left_lift" gear="1" ctrllimited="true" ctrlrange="-20 20"/>'
        )
    )
    act.append(
        ET.fromstring(
            '<motor name="right_lift_motor" joint="right_lift" gear="1" ctrllimited="true" ctrlrange="-20 20"/>'
        )
    )
    contact = ET.SubElement(root, "contact")
    for tree in (left, right):
        hc = tree.find("contact")
        if hc is not None:
            for child in list(hc):
                contact.append(child)
    return dump_mjcf(root), gains


def run_bimanual_episode(
    friction: float,
    solref_s: float,
    *,
    n_close: int = 80,
    n_hold: int = 80,
    lift_m: float = 0.06,
) -> BimanualMetrics:
    import mujoco

    xml, gains = _scene_xml(friction, solref_s)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    left = lookup_hand_actuators(model, gains, "left")
    right = lookup_hand_actuators(model, gains, "right")
    spec = load_hand_spec()
    lib = GraspLibrary(spec)
    q_open = lib.q_active("open", 0.0)
    q_closed = lib.q_active("power_grasp", 1.0)
    lift_ids = (
        int(model.actuator("left_lift_motor").id),
        int(model.actuator("right_lift_motor").id),
    )
    lift_qadr = (
        int(model.jnt_qposadr[int(model.joint("left_lift").id)]),
        int(model.jnt_qposadr[int(model.joint("right_lift").id)]),
    )
    lift_dadr = (
        int(model.jnt_dofadr[int(model.joint("left_lift").id)]),
        int(model.jnt_dofadr[int(model.joint("right_lift").id)]),
    )
    cube_id = int(model.body("cube").id)
    mujoco.mj_forward(model, data)
    z0 = float(data.xpos[cube_id][2])
    xy0 = data.xpos[cube_id][:2].copy()
    max_tau = 0.0
    n_con = 0

    def _step(q_des: np.ndarray, lift_target: float) -> None:
        nonlocal max_tau, n_con
        tau_l = apply_mit(model, data, left, q_des)
        tau_r = apply_mit(model, data, right, q_des)
        max_tau = max(max_tau, float(np.max(np.abs(tau_l))), float(np.max(np.abs(tau_r))))
        for act_id, qadr, dadr in zip(lift_ids, lift_qadr, lift_dadr, strict=True):
            q = data.qpos[qadr]
            dq = data.qvel[dadr]
            data.ctrl[act_id] = float(np.clip(80.0 * (lift_target - q) - 4.0 * dq, -20, 20))
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
    return BimanualMetrics(
        friction=friction,
        solref_timeconst_s=solref_s,
        slip_m=float(np.linalg.norm(xy1 - xy0)),
        cube_drop_m=float(z0 - z1),
        n_contacts=n_con,
        max_abs_tau=max_tau,
        finite=bool(np.isfinite(data.qpos).all()),
    )
