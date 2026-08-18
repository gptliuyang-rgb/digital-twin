"""T800 MuJoCo 500 Hz PD plant. Combined T800+Hand remains PolicyEvalBlocked."""

from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked
from sim.mujoco_env.t800_env import T800MujocoEnv, refuse_combined_robot
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import TOKEN_DIM
from wbc.gather import HardwareSnapshot, ObsGather
from wbc.pd_physics import PHYSICS_N_STEPS, PHYSICS_TIMESTEP_S
from wbc.planner_onnx import PlannerQpos, pack_qpos
from wbc.shared_cursor import SharedPlaybackCursor, decoder_joint_history


def _hw(*, q0: float = 0.0, t_s: float = 0.0) -> HardwareSnapshot:
    q = np.zeros(25)
    q[0] = q0
    return HardwareSnapshot(
        t_s=t_s,
        q_hw=q,
        dq_hw=np.zeros(25),
        omega_imu=np.zeros(3),
        imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(25),
    )


def test_combined_robot_refused() -> None:
    with pytest.raises(PolicyEvalBlocked):
        refuse_combined_robot()


def test_apply_tau_g1_refused() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=True)
    env.reset()
    with pytest.raises(G1CheckpointIncompatible):
        env.apply_tau_nm(np.zeros(29))


def test_fixture_timestep_is_500hz() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=True)
    env.reset()
    assert env.n_dof == 25
    assert env.timestep_s == pytest.approx(PHYSICS_TIMESTEP_S)


def test_apply_tau_and_step_moves_hinge() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=True)
    env.reset()
    tau = np.zeros(25)
    tau[0] = 2.0
    q0 = env.get_q().copy()
    q, dq = env.apply_tau_and_step(tau)
    assert q.shape == (25,)
    assert dq.shape == (25,)
    assert np.isfinite(q).all()
    assert not np.allclose(q, q0)


def test_cursor_physics_period_on_fixture() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=True)
    env.reset()
    gather = ObsGather()
    gather.push_hw(_hw())
    cur = SharedPlaybackCursor(gather=gather, physics=env)
    q = np.zeros(25)
    clip = np.stack(
        [
            pack_qpos(
                PlannerQpos(
                    root_pos_m=np.array([0.01 * i, 0.0, 1.03]),
                    root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    q_rad=q,
                )
            )
            for i in range(16)
        ]
    )
    token = np.zeros(TOKEN_DIM)
    q_des = np.zeros(25)
    step = cur.control_tick(
        np.zeros(25),
        locomotion_mode=0,
        token=token,
        new_qpos=clip,
        t_s=0.0,
        policy_action=q_des,
    )
    assert step.pd_physics_n_steps == PHYSICS_N_STEPS
    assert np.isfinite(step.pd_physics_q).all()
    np.testing.assert_allclose(decoder_joint_history(step.decoder_obs)[-1], 0.0)
    np.testing.assert_allclose(step.pd_q_des, 0.0)


def test_cursor_omit_push_hw_follows_fixture_q() -> None:
    pytest.importorskip("mujoco")
    env = T800MujocoEnv(source="fixture", pinned_base=True)
    env.reset()
    q0 = env.get_q().copy()
    gather = ObsGather()
    gather.push_hw(
        HardwareSnapshot(
            t_s=0.0,
            q_hw=q0,
            dq_hw=np.zeros(25),
            omega_imu=np.array([0.1, 0.2, 0.3]),
            imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            last_action=np.zeros(25),
        )
    )
    cur = SharedPlaybackCursor(gather=gather, physics=env)
    token = np.zeros(TOKEN_DIM)
    q = np.zeros(25)
    clip = np.stack(
        [
            pack_qpos(
                PlannerQpos(
                    root_pos_m=np.array([0.01 * i, 0.0, 1.03]),
                    root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    q_rad=q,
                )
            )
            for i in range(16)
        ]
    )
    q_des = np.zeros(25)
    q_des[0] = 0.05
    first = cur.control_tick(
        np.zeros(25),
        locomotion_mode=0,
        token=token,
        new_qpos=clip,
        t_s=0.0,
        policy_action=q_des,
    )
    np.testing.assert_allclose(decoder_joint_history(first.decoder_obs)[-1], q0)
    plant_q = first.pd_physics_q.copy()
    second = cur.control_tick(
        np.zeros(25),
        locomotion_mode=0,
        token=token,
        t_s=0.02,
        policy_action=q_des,
    )
    np.testing.assert_allclose(decoder_joint_history(second.decoder_obs)[-1], plant_q)
    np.testing.assert_allclose(gather.hw.read().omega_imu, [0.1, 0.2, 0.3])
    assert not np.allclose(second.pd_physics_q, plant_q)
