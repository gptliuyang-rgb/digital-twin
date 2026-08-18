"""Dump DexHand2 1 kHz MIT ring: ZOH hand_q, bypass WBC.

Does not run ONNX. Uses a numpy test double, not MuJoCo. Gains are
MJCF gen-1 carry-over, not Hand 2 sys-id. grasp_success_rate stays
JSON null. Combined T800+Hand stays PolicyEvalBlocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand.mit_ring import (
    GAINS_SOURCE,
    HAND_N_STEPS,
    HAND_STREAM_HZ,
    HAND_TIMESTEP_S,
    MIT_RING_YAML,
    HandMitPlant,
    load_mit_ring_cfg,
    refuse_force_mode_without_hardware_tau,
    refuse_hand_q_hermite,
    refuse_hand_q_onto_wbc_last_action,
    refuse_hardware_kp,
    refuse_invented_latency,
    refuse_mit_ring_as_decoder_run,
    refuse_mit_ring_finite_diff_dq,
    refuse_sim_forcerange_as_payload_rating,
    require_hand_q,
    require_hand_timestep_1khz,
    sim_kp_kd,
)
from interface.schema import REPO_ROOT, CommandVector
from runtime.hand_bypass import run_bimanual_period

OUT = Path("eval/report/generated/l1c_mit_ring.json")


class IncrementHandPhysics:
    """Eval double. Not MuJoCo. Increments q[0] by 0.001 each 1 ms substep."""

    n_dof = 20
    timestep_s = 0.001

    def __init__(self, q0: float = 0.0) -> None:
        self.q = np.zeros(20)
        self.q[0] = q0
        self.dq = np.zeros(20)
        self.n_applies = 0

    def read_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.dq.copy()

    def apply_tau_and_step(self, tau_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del tau_nm
        self.q = self.q.copy()
        self.q[0] += 0.001
        self.n_applies += 1
        return self.read_q_dq()


def _cmd(*, q0: float) -> CommandVector:
    cmd = CommandVector.zeros()
    cmd.left_hand_q = np.zeros(20)
    cmd.left_hand_q[0] = q0
    cmd.right_hand_q = np.zeros(20)
    cmd.right_hand_q[0] = q0
    return cmd


def _dump() -> dict:
    cfg = load_mit_ring_cfg()
    plant = HandMitPlant.from_spec()
    kp, _kd = sim_kp_kd()
    left = IncrementHandPhysics(q0=0.0)
    right = IncrementHandPhysics(q0=0.0)
    last = None
    for i in range(10):
        last = run_bimanual_period(_cmd(q0=0.5), left, right, t0_s=i / 50.0)
    assert last is not None
    expected_q = 0.001 * HAND_N_STEPS * 10
    q_before_last_substep = expected_q - 0.001
    expected_tau0 = float(kp[0] * (0.5 - q_before_last_substep))
    last_tau0 = float(last.right.tau_nm[-1, 0])
    hermite_refused = False
    try:
        refuse_hand_q_hermite()
    except Exception:
        hermite_refused = True
    wbc_refused = False
    try:
        refuse_hand_q_onto_wbc_last_action()
    except Exception:
        wbc_refused = True
    onnx_refused = False
    try:
        refuse_mit_ring_as_decoder_run()
    except Exception:
        onnx_refused = True
    finite_diff_refused = False
    try:
        refuse_mit_ring_finite_diff_dq()
    except Exception:
        finite_diff_refused = True
    hw_kp_refused = False
    try:
        refuse_hardware_kp()
    except Exception:
        hw_kp_refused = True
    latency_refused = False
    try:
        refuse_invented_latency()
    except Exception:
        latency_refused = True
    force_refused = False
    try:
        refuse_force_mode_without_hardware_tau()
    except Exception:
        force_refused = True
    forcerange_refused = False
    try:
        refuse_sim_forcerange_as_payload_rating()
    except Exception:
        forcerange_refused = True
    t800_refused = False
    try:
        require_hand_q(np.zeros(25))
    except Exception:
        t800_refused = True
    concat_refused = False
    try:
        require_hand_q(np.zeros(45))
    except Exception:
        concat_refused = True
    schema_refused = False
    try:
        require_hand_q(np.zeros(75))
    except Exception:
        schema_refused = True
    dt_refused = False
    try:
        require_hand_timestep_1khz(0.002)
    except Exception:
        dt_refused = True
    return {
        "source": MIT_RING_YAML.name,
        "adr": "ADR-058",
        "control_hz": cfg["control_hz"],
        "hand_stream_hz": HAND_STREAM_HZ,
        "factor_policy_to_hand_stream": HAND_N_STEPS,
        "n_active_dof": cfg["n_active_dof"],
        "mit_n_steps_per_tick": cfg["mit_n_steps_per_tick"],
        "mit_timestep_s": HAND_TIMESTEP_S,
        "gains_source": last.right.gains_source,
        "kp0": float(kp[0]),
        "right_q0_after_10": float(last.right.q_rad[-1, 0]),
        "right_q0_is_ten_ticks_of_substeps": bool(np.isclose(last.right.q_rad[-1, 0], expected_q)),
        "left_q0_matches_right": bool(np.isclose(last.left.q_rad[-1, 0], last.right.q_rad[-1, 0])),
        "right_tau0_last_substep": last_tau0,
        "right_tau0_matches_closed_loop_q": bool(np.isclose(last_tau0, expected_tau0, rtol=0, atol=1e-12)),
        "physics_applies_total_right": right.n_applies,
        "physics_applies_is_200": right.n_applies == 200,
        "q_des0": 0.5,
        "hold": last.right.hold,
        "dq_des_is_zero": last.right.dq_des_rad_s == 0.0,
        "tool_trigger": last.tool_trigger,
        "t800_action_refused": t800_refused,
        "t800_plus_hand_concat_refused": concat_refused,
        "command_schema_flat_refused": schema_refused,
        "hand_q_hermite_refused": hermite_refused,
        "hand_q_onto_wbc_last_action_refused": wbc_refused,
        "mit_ring_from_decoder_onnx_refused": onnx_refused,
        "mit_ring_finite_diff_dq_refused": finite_diff_refused,
        "hardware_kp_refused": hw_kp_refused,
        "invented_latency_refused": latency_refused,
        "force_mode_without_hardware_tau_refused": force_refused,
        "sim_forcerange_as_payload_rating_refused": forcerange_refused,
        "body_500hz_dt_on_hand_ring_refused": dt_refused,
        "gains_are_mjcf_gen1_carryover": True,
        "not_hardware_kp": True,
        "not_wbc_last_action": True,
        "not_decoder_obs": True,
        "not_concat_t800_plus_hand": True,
        "hand_q_is_zoh": True,
        "not_dexhand2_contact": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "gains_source_locked": GAINS_SOURCE,
        "plant_gains_source": plant.gains_source,
        "repo": str(REPO_ROOT),
    }


def main() -> None:
    payload = _dump()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
