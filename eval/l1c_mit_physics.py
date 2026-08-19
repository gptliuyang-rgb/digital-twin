"""Dump DexHand2 1 kHz MIT ring on a MuJoCo HandMitPhysics backend.

Does not run ONNX. Gains are MJCF gen-1 carry-over, not Hand 2 sys-id.
Fixture inertias are placeholders (not CAD). Official skeleton inertias
are used when wuji-description is cloned (ADR-060 mesh-free MIT rewrite).
grasp_success_rate stays JSON null. Combined T800+Hand stays
PolicyEvalBlocked. CI without MuJoCo reports runtime=unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hand.mit_ring import (
    GAINS_SOURCE,
    HAND_N_STEPS,
    HAND_STREAM_HZ,
    HAND_TIMESTEP_S,
    HandMitPlant,
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
    run_mit_period,
    sim_kp_kd,
)
from interface.schema import REPO_ROOT
from sim.mujoco_env.hand_mit_physics import (
    HAND_MIT_PHYSICS_YAML,
    load_hand_mit_physics_cfg,
    official_mjcf_present,
    plant_mass_kg,
    refuse_body_500hz_dt,
    refuse_combined_robot,
    refuse_official_002s_dt,
    refuse_official_position_actuators,
    refuse_treat_fixture_as_cad,
    resolve_source,
)

OUT = Path("eval/report/generated/l1c_mit_physics.json")


def _raised(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _period(env: Any, q_des: np.ndarray, n_ticks: int = 10) -> Any:
    plant = HandMitPlant.from_spec()
    last = None
    for i in range(n_ticks):
        last = run_mit_period(plant, env, q_des, t0_s=i / 50.0, side=env.side)
    return last


def _dump() -> dict[str, Any]:
    cfg = load_hand_mit_physics_cfg()
    report: dict[str, Any] = {
        "source": HAND_MIT_PHYSICS_YAML.name,
        "adr": "ADR-060",
        "control_hz": cfg["control_hz"],
        "hand_stream_hz": HAND_STREAM_HZ,
        "factor_policy_to_hand_stream": HAND_N_STEPS,
        "n_active_dof": cfg["n_active_dof"],
        "mit_n_steps_per_tick": cfg["mit_n_steps_per_tick"],
        "mit_timestep_s": HAND_TIMESTEP_S,
        "gains_source": GAINS_SOURCE,
        "gains_are_mjcf_gen1_carryover": True,
        "not_hardware_kp": True,
        "not_wbc_last_action": True,
        "not_decoder_obs": True,
        "not_concat_t800_plus_hand": True,
        "not_t800_weld": True,
        "not_official_position_actuators": True,
        "not_body_500hz_dt": True,
        "not_official_002s_dt": True,
        "hand_q_is_zoh": True,
        "fixture_inertias_are_placeholders": True,
        "not_treat_fixture_as_cad": True,
        "official_inertias_from_skeleton_mjcf": True,
        "not_soft_body_in_inertia_plant": True,
        "not_dexhand2_contact": True,
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "t800_action_refused": _raised(lambda: require_hand_q(np.zeros(25))),
        "t800_plus_hand_concat_refused": _raised(lambda: require_hand_q(np.zeros(45))),
        "command_schema_flat_refused": _raised(lambda: require_hand_q(np.zeros(75))),
        "hand_q_hermite_refused": _raised(refuse_hand_q_hermite),
        "hand_q_onto_wbc_last_action_refused": _raised(refuse_hand_q_onto_wbc_last_action),
        "mit_ring_from_decoder_onnx_refused": _raised(refuse_mit_ring_as_decoder_run),
        "mit_ring_finite_diff_dq_refused": _raised(refuse_mit_ring_finite_diff_dq),
        "hardware_kp_refused": _raised(refuse_hardware_kp),
        "invented_latency_refused": _raised(refuse_invented_latency),
        "force_mode_without_hardware_tau_refused": _raised(refuse_force_mode_without_hardware_tau),
        "sim_forcerange_as_payload_rating_refused": _raised(refuse_sim_forcerange_as_payload_rating),
        "body_500hz_dt_on_hand_ring_refused": _raised(lambda: require_hand_timestep_1khz(0.002)),
        "official_002s_dt_refused": _raised(refuse_official_002s_dt),
        "official_position_actuators_refused": _raised(refuse_official_position_actuators),
        "combined_weld_refused": _raised(refuse_combined_robot),
        "refuse_body_500hz_alias": _raised(refuse_body_500hz_dt),
        "treat_fixture_as_cad_refused": _raised(refuse_treat_fixture_as_cad),
        "auto_source": resolve_source("auto"),
        "official_mjcf_present": official_mjcf_present(),
        "runtime": "unavailable",
        "status": "mujoco_unavailable",
        "physics_applies_is_200": False,
        "q0_moved_toward_q_des": False,
        "repo": str(REPO_ROOT),
    }
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return report
    from sim.mujoco_env.hand_mit_physics import HandMitMujocoEnv

    kp, _kd = sim_kp_kd()
    q_des = np.zeros(20)
    q_des[0] = 0.5

    fixture = HandMitMujocoEnv(source="fixture")
    fixture.reset()
    last_fix = _period(fixture, q_des)
    fix_q0 = float(last_fix.q_rad[-1, 0])
    fix_mass = plant_mass_kg(fixture.model)
    report["fixture"] = {
        "xml_note": fixture.xml_note,
        "plant_mass_kg": fix_mass,
        "right_q0_after_10": fix_q0,
        "right_tau0_last_substep": float(last_fix.tau_nm[-1, 0]),
        "inertias": "placeholders",
    }

    env_source = resolve_source("auto")
    env = HandMitMujocoEnv(source=env_source)
    env.reset()
    last = _period(env, q_des)
    q0 = float(last.q_rad[-1, 0])
    mass = plant_mass_kg(env.model)
    status = "mit_physics_ok"
    runtime = f"mujoco_{env.source}"
    if env.source == "official_inertia":
        status = "mit_physics_official_inertia_ok"
    report.update(
        {
            "runtime": runtime,
            "status": status,
            "xml_note": env.xml_note,
            "env_source": env.source,
            "plant_mass_kg": mass,
            "kp0": float(kp[0]),
            "q_des0": 0.5,
            "right_q0_after_10": q0,
            "right_tau0_last_substep": float(last.tau_nm[-1, 0]),
            "hold": last.hold,
            "dq_des_is_zero": last.dq_des_rad_s == 0.0,
            "physics_applies_is_200": True,
            "n_steps_last_period": last.n_steps,
            "q0_moved_toward_q_des": bool(q0 > 0.05 and q0 < 0.6),
            "plant_gains_source": env.plant.gains_source,
            "timestep_s": env.timestep_s,
            "fixture_q0_differs_from_official": bool(
                env.source == "official_inertia" and abs(q0 - fix_q0) > 1e-6
            ),
        }
    )
    return report


def main() -> None:
    payload = _dump()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    keys = (
        "status",
        "runtime",
        "adr",
        "env_source",
        "plant_mass_kg",
        "q0_moved_toward_q_des",
        "right_q0_after_10",
        "fixture_q0_differs_from_official",
        "grasp_success_rate",
        "combined_robot",
        "not_t800_weld",
        "not_treat_fixture_as_cad",
    )
    print(json.dumps({k: payload[k] for k in keys if k in payload}, indent=2))


if __name__ == "__main__":
    main()
