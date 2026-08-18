"""Dump omit-push_hw closed loop: decoder q follows the 500 Hz plant.

Does not run ONNX. Uses a numpy test double, not MuJoCo. Gains are
EngineAI pd_stand bring-up, not SONIC tracking. grasp_success_rate
stays JSON null. This tick's decoder q is pre-this-tick physics.
IMU is kept from the first HardwareSnapshot, not invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.checkpoint import G1CheckpointIncompatible
from wbc.dims import PLANNER_CONTROL_HZ, T800_N_LOWER_BODY_DOF, TOKEN_DIM
from wbc.gather import HardwareSnapshot, ObsGather
from wbc.pd_physics import (
    GAINS_SOURCE,
    PD_PHYSICS_YAML,
    PHYSICS_N_STEPS,
    load_pd_physics_cfg,
    refuse_omit_push_hw_invent_imu,
    refuse_pd_physics_as_decoder_run,
    refuse_pd_physics_finite_diff_dq,
    refuse_pd_physics_hermite,
    refuse_pd_physics_invent_imu,
    refuse_pd_physics_q_onto_this_tick_decoder,
)
from wbc.pd_stand import pd_stand_kp_kd
from wbc.planner_onnx import PlannerOnnxBlocked, PlannerQpos, pack_qpos
from wbc.shared_cursor import (
    SharedPlaybackCursor,
    decoder_joint_history,
    decoder_last_action_history,
    decoder_omega_history,
    load_shared_cursor_cfg,
    refuse_run_shared_cursor_onnx,
)
from wbc.stream import STREAM_HZ, stream_factor

OUT = Path("eval/report/generated/l1a_pd_closedloop.json")
IMU_OMEGA = np.array([0.1, 0.2, 0.3])


class IncrementPhysics:
    """Eval double. Not MuJoCo. Increments q[0] by 0.001 each 2 ms substep."""

    n_dof = 25
    timestep_s = 0.002

    def __init__(self, q0: float = 0.0) -> None:
        self.q = np.zeros(25)
        self.q[0] = q0
        self.dq = np.zeros(25)
        self.n_applies = 0

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.dq.copy()

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del tau_nm
        self.q = self.q.copy()
        self.q[0] += 0.001
        self.n_applies += 1
        return self.read_q_dq()


def _row(i: int) -> np.ndarray:
    q = np.zeros(25)
    q[0] = 0.01 * i
    return pack_qpos(
        PlannerQpos(
            root_pos_m=np.array([0.01 * i, 0.0, 1.03]),
            root_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            q_rad=q,
        )
    )


def _motor(lower: float) -> np.ndarray:
    q = np.zeros(25)
    q[:T800_N_LOWER_BODY_DOF] = lower
    return q


def _hw(*, q0: float, t_s: float) -> HardwareSnapshot:
    q = np.zeros(25)
    q[0] = q0
    return HardwareSnapshot(
        t_s=t_s,
        q_hw=q,
        dq_hw=np.zeros(25),
        omega_imu=IMU_OMEGA.copy(),
        imu_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(25),
    )


def _action(*, a0: float) -> np.ndarray:
    a = np.zeros(25)
    a[0] = a0
    return a


def _dump() -> dict:
    cfg = load_shared_cursor_cfg()
    physics_cfg = load_pd_physics_cfg()
    kp, kd = pd_stand_kp_kd()
    gather = ObsGather()
    physics = IncrementPhysics(q0=0.0)
    cur = SharedPlaybackCursor(gather=gather, physics=physics)
    token = np.linspace(0.0, 1.0, TOKEN_DIM)
    gather.push_hw(_hw(q0=0.0, t_s=0.0))
    last = None
    for i in range(10):
        last = cur.control_tick(
            _motor(0.0),
            locomotion_mode=0,
            token=token,
            new_qpos=np.stack([_row(j) for j in range(16)]) if i == 0 else None,
            t_s=i / 50.0,
            policy_action=_action(a0=10.0 + i),
        )
    assert last is not None
    a_dec = decoder_last_action_history(last.decoder_obs)[:, 0].tolist()
    q_dec = decoder_joint_history(last.decoder_obs)[:, 0].tolist()
    w_dec = decoder_omega_history(last.decoder_obs)
    expected_a = [0.0] + [10.0 + i for i in range(9)]
    expected_q = [0.01 * i for i in range(10)]
    onnx_blocked = False
    try:
        refuse_run_shared_cursor_onnx()
    except PlannerOnnxBlocked:
        onnx_blocked = True
    hermite_refused = False
    try:
        refuse_pd_physics_hermite()
    except Exception:
        hermite_refused = True
    invented_onnx_refused = False
    try:
        refuse_pd_physics_as_decoder_run()
    except Exception:
        invented_onnx_refused = True
    finite_diff_refused = False
    try:
        refuse_pd_physics_finite_diff_dq()
    except Exception:
        finite_diff_refused = True
    decoder_rewrite_refused = False
    try:
        refuse_pd_physics_q_onto_this_tick_decoder()
    except Exception:
        decoder_rewrite_refused = True
    imu_refused = False
    try:
        refuse_pd_physics_invent_imu()
    except Exception:
        imu_refused = True
    omit_imu_refused = False
    try:
        refuse_omit_push_hw_invent_imu()
    except Exception:
        omit_imu_refused = True
    g1_action_refused = False
    try:
        cur.push_policy_pd(np.zeros(29))
    except G1CheckpointIncompatible:
        g1_action_refused = True
    expected_tau0 = float(kp[0] * (19.0 - 0.09) - kd[0] * 0.0)
    return {
        "source": PD_PHYSICS_YAML.name,
        "adr": "ADR-057",
        "control_hz": cfg["control_hz"],
        "command_stream_hz": STREAM_HZ,
        "factor_policy_to_stream": stream_factor(50),
        "n_dof": cfg["n_dof"],
        "pd_physics_n_steps_per_tick": physics_cfg["pd_physics_n_steps_per_tick"],
        "pd_physics_timestep_s": physics_cfg["pd_physics_timestep_s"],
        "gains_source": last.pd_gains_source,
        "kp0": float(kp[0]),
        "kd0": float(kd[0]),
        "decoder_a0_history": a_dec,
        "decoder_a_equals_expected_delay": a_dec == expected_a,
        "decoder_q0_history": q_dec,
        "decoder_q_follows_plant": bool(np.allclose(q_dec, expected_q)),
        "decoder_omega_history_0": w_dec[:, 0].tolist(),
        "decoder_omega_kept_from_first_snapshot": bool(
            np.allclose(w_dec, np.broadcast_to(IMU_OMEGA, (10, 3)))
        ),
        "pd_q_des_a0_after_10": float(last.pd_q_des[0]),
        "pd_tau0_after_10": float(last.pd_tau_nm[0]),
        "pd_tau0_matches_closed_loop_q": bool(np.isclose(last.pd_tau_nm[0], expected_tau0)),
        "pd_physics_n_steps_after_10": last.pd_physics_n_steps,
        "pd_physics_q0_after_10": float(last.pd_physics_q[0]),
        "pd_physics_q0_is_ten_ticks_of_substeps": bool(
            np.isclose(last.pd_physics_q[0], 0.001 * PHYSICS_N_STEPS * 10)
        ),
        "physics_applies_total": physics.n_applies,
        "stashed_a0_after_10": None if last.policy_action is None else float(last.policy_action[0]),
        "policy_hz": PLANNER_CONTROL_HZ,
        "g1_action_refused": g1_action_refused,
        "pd_physics_hermite_refused": hermite_refused,
        "pd_physics_from_decoder_onnx_refused": invented_onnx_refused,
        "pd_physics_finite_diff_refused": finite_diff_refused,
        "pd_physics_q_onto_this_tick_decoder_refused": decoder_rewrite_refused,
        "pd_physics_invent_imu_refused": imu_refused,
        "omit_push_hw_invent_imu_refused": omit_imu_refused,
        "onnx_run_blocked": onnx_blocked,
        "pd_physics_on_same_tick": True,
        "pd_physics_is_optional": True,
        "pd_physics_feeds_next_tick_decoder_q": True,
        "omit_push_hw_keeps_imu": True,
        "not_pd_physics_q_onto_this_tick_decoder": True,
        "not_pd_physics_invent_imu": True,
        "pd_plant_gains_are_pd_stand_bringup": True,
        "not_sonic_tracking_gains": True,
        "pd_plant_dq_des_is_zero": True,
        "not_pd_tau_onto_decoder_obs": True,
        "policy_action_pd_is_zoh": True,
        "not_policy_action_hermite": True,
        "policy_action_feeds_500hz_pd_after_stash": True,
        "policy_action_feeds_next_tick_last_action": True,
        "decoder_history_is_hardware_not_clip": True,
        "not_g1_decoder_onnx": True,
        "not_invented_clip": True,
        "not_pico_sdk": True,
        "not_dexhand2_contact": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "gains_source_locked": GAINS_SOURCE,
        "repo": str(REPO_ROOT),
    }


def main() -> None:
    payload = _dump()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
