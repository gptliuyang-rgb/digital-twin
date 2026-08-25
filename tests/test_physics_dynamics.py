"""Unit tests for gravity-comp PD and weld snapshots (no T800 required)."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")


_PENDULUM = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <inertial pos="0 0 -0.25" mass="1" diaginertia="0.02 0.02 0.002"/>
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-40 40"/>
  </actuator>
</mujoco>
"""

_WELD_PAIR = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast"/>
  <worldbody>
    <body name="a" pos="0 0 1.2">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.4" diaginertia="0.001 0.001 0.001"/>
      <geom type="sphere" size="0.04" mass="0.4"/>
    </body>
    <body name="b" pos="0.15 0 1.2">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.4" diaginertia="0.001 0.001 0.001"/>
      <geom type="sphere" size="0.04" mass="0.4"/>
    </body>
  </worldbody>
  <equality>
    <weld name="weld_ab" body1="a" body2="b" active="false" solref="0.004 1"/>
  </equality>
</mujoco>
"""


def test_gravity_comp_holds_hinge_away_from_hang() -> None:
    from sim.mujoco_env.dynamics import apply_body_pd

    model = mujoco.MjModel.from_xml_string(_PENDULUM)
    data = mujoco.MjData(model)
    q_des = np.array([0.45])
    data.qpos[0] = 0.45
    data.qvel[0] = 0.0
    act_ids = np.array([0], dtype=np.int32)
    for _ in range(2500):
        mujoco.mj_forward(model, data)
        apply_body_pd(model, data, act_ids, q_des, kp=60.0, kd=8.0, gravity_comp=True)
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert abs(float(data.qpos[0]) - 0.45) < 0.08


def test_hand_object_contact_counter() -> None:
    from sim.mujoco_env.contacts import hand_object_contacts, pad_object_contacts

    xml = """
    <mujoco>
      <worldbody>
        <body name="hand">
          <geom name="r_index_finger_tip_pad_0" type="sphere" size="0.02" pos="0 0 0.1"/>
          <geom name="r_palm" type="sphere" size="0.02" pos="0 0 0.12"/>
        </body>
        <body name="box_0" pos="0 0 0.1">
          <freejoint/>
          <geom name="box_0_geom" type="sphere" size="0.02"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    assert hand_object_contacts(model, data, "right", "box_0") >= 1
    assert pad_object_contacts(model, data, "right", "box_0") >= 1
    assert hand_object_contacts(model, data, "left", "box_0") == 0


def test_weld_snapshot_keeps_relative_pose_in_free_fall() -> None:
    from sim.mujoco_env.welds import set_weld_active, weld_relpose

    model = mujoco.MjModel.from_xml_string(_WELD_PAIR)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rel0, _ = weld_relpose(model, data, "weld_ab")
    set_weld_active(model, data, "weld_ab", True)
    for _ in range(1800):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    rel1, _ = weld_relpose(model, data, "weld_ab")
    assert float(np.linalg.norm(rel1 - rel0)) < 0.02
    # Pair fell under gravity; z dropped, offset held.
    assert float(data.xpos[int(model.body("a").id)][2]) < 1.0
