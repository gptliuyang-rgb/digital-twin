"""Replay E1 Coulomb pull and E2 pad indent in MuJoCo once contact params exist.

Refuses if friction/stiffness are still REQUIRED_INPUT. Does not write
grasp_success_rate. The solref mapping is a proposal (human_must_accept_solref).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hand.calibration.contact_mujoco import (
    contact_is_calibrated,
    friction_attr,
    solref_timeconst_s,
)
from interface.schema import HAND_SPEC_PATH, REQUIRED_INPUT_TOKEN, load_hand_spec


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"validate_sim: mujoco is required ({exc})") from exc
    return mujoco


def replay_e1_coulomb(mu_s: float, *, n_steps: int = 800, dt: float = 0.001) -> dict[str, Any]:
    """Horizontal pull on a free puck: F < μN should hold, F > μN should slide."""
    mujoco = _require_mujoco()
    fr = friction_attr(mu_s)
    xml = f"""
    <mujoco model="e1_coulomb">
      <option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast"
              cone="elliptic" impratio="10"/>
      <worldbody>
        <geom name="table" type="plane" size="2 2 0.1" friction="{fr}"
              solref="0.005 1" condim="4"/>
        <body name="puck" pos="0 0 0.021">
          <inertial pos="0 0 0" mass="1" diaginertia="0.0004 0.0004 0.0004"/>
          <freejoint/>
          <geom name="puck_g" type="box" size="0.05 0.05 0.02"
                friction="{fr}" solref="0.005 1" condim="4"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    bid = int(model.body("puck").id)
    n = 1.0 * 9.81
    f_hold = 0.55 * mu_s * n
    f_slip = 1.50 * mu_s * n

    def _run(force: float) -> tuple[float, int]:
        data = mujoco.MjData(model)
        for _ in range(250):
            data.xfrc_applied[bid] = 0
            mujoco.mj_step(model, data)
        x0 = float(data.xpos[bid][0])
        for _ in range(n_steps):
            data.xfrc_applied[bid] = [force, 0, 0, 0, 0, 0]
            mujoco.mj_step(model, data)
        return abs(float(data.xpos[bid][0]) - x0), int(data.ncon)

    hold_disp, hold_ncon = _run(f_hold)
    slip_disp, slip_ncon = _run(f_slip)
    hold_ok = hold_disp < 0.02 and hold_ncon > 0
    slip_ok = slip_disp > 0.05
    return {
        "mu_s": mu_s,
        "f_hold_n": f_hold,
        "f_slip_n": f_slip,
        "hold_disp_m": hold_disp,
        "slip_disp_m": slip_disp,
        "hold_ncon": hold_ncon,
        "slip_ncon": slip_ncon,
        "hold_ok": hold_ok,
        "slip_ok": slip_ok,
        "ok": bool(hold_ok and slip_ok),
        "method": "horizontal_coulomb_pull",
        "note": "Equivalent to incline tanθ=μ; used because ramp rest poses are noisy in MuJoCo.",
    }


def replay_e2_indent(
    k_n_per_m: float,
    solref_s: float,
    *,
    m_eff_kg: float = 0.03,
    force_n: float = 2.0,
    dt: float = 0.0005,
) -> dict[str, Any]:
    """Press a pad sphere into a plane with a known force; k_sim = F/δ.

    solref is an impedance time-const, so k_sim will not equal E2 k exactly.
    The ratio gate is intentionally wide; a human must accept the mapping.
    """
    mujoco = _require_mujoco()
    radius = 0.01
    sr = f"{solref_s} 1"
    xml = f"""
    <mujoco model="e2_indent">
      <option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <geom name="plate" type="plane" size="0.2 0.2 0.01" pos="0 0 0" solref="{sr}" condim="1"/>
        <body name="pad" pos="0 0 {radius}">
          <inertial pos="0 0 0" mass="{m_eff_kg}" diaginertia="0.00001 0.00001 0.00001"/>
          <joint name="z" type="slide" axis="0 0 1" damping="0.5"/>
          <geom name="pad_g" type="sphere" size="{radius}" solref="{sr}" condim="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    bid = int(model.body("pad").id)
    for _ in range(4000):
        data.xfrc_applied[bid] = [0, 0, -force_n, 0, 0, 0]
        mujoco.mj_step(model, data)
    finite = bool(np.isfinite(data.qpos).all())
    z = float(data.xpos[bid][2])
    pen = radius - z
    k_sim = force_n / pen if pen > 1e-7 else float("nan")
    ratio = k_sim / k_n_per_m if k_n_per_m else float("nan")
    ok = bool(math.isfinite(ratio) and 0.1 <= ratio <= 10.0 and finite and int(data.ncon) > 0)
    return {
        "k_e2_n_per_m": k_n_per_m,
        "solref_timeconst_s": solref_s,
        "k_sim_n_per_m": k_sim,
        "penetration_m": pen,
        "force_n": force_n,
        "ratio": ratio,
        "ok": ok,
        "finite": finite,
        "ncon": int(data.ncon),
        "gate": [0.1, 10.0],
        "human_must_accept_solref": True,
        "note": "solref[0]=2π/sqrt(k/m_eff) is a proposal; ratio gate is intentionally loose.",
    }


def run_validate(raw: dict[str, Any]) -> dict[str, Any]:
    mu_s = float(raw["friction_vs_cardboard_static"])
    k = float(raw["normal_stiffness_n_per_m"])
    sol = solref_timeconst_s(raw)
    m_eff = float(raw["m_eff_kg"]) if raw.get("m_eff_kg") not in (None, REQUIRED_INPUT_TOKEN) else 0.03
    e1 = replay_e1_coulomb(mu_s)
    e2 = replay_e2_indent(k, sol, m_eff_kg=m_eff)
    finite = bool(e2.get("finite", True))
    return {
        "status": "ran",
        "policy_eval_forbidden": True,
        "grasp_success_rate": None,
        "calibration_kind": raw.get("calibration_kind", "unknown"),
        "do_not_treat_as_committed_hardware": bool(raw.get("do_not_treat_as_committed_hardware", False)),
        "e1": e1,
        "e2": e2,
        "ok": bool(e1["ok"] and e2["ok"] and finite),
        "finite": finite,
        "human_must_accept_solref": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="", help="Overlay spec. Default: live dexhand2_spec.yaml")
    parser.add_argument("--out", default="hand/calibration/results/validate_sim.json")
    args = parser.parse_args()
    spec_path = Path(args.spec) if args.spec else HAND_SPEC_PATH
    spec = load_hand_spec(spec_path)
    if not contact_is_calibrated(spec.raw):
        raise SystemExit(
            "validate_sim: friction/stiffness are REQUIRED_INPUT. "
            "Run E1/E2, fit_params.py, and apply_calibration_fragment.py to an overlay spec."
        )
    report = run_validate(spec.raw)
    report["spec"] = str(spec_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "ok": report["ok"],
        "e1_ok": report["e1"]["ok"],
        "e2_ok": report["e2"]["ok"],
        "human_must_accept_solref": True,
        "out": str(out),
    }
    print(json.dumps(summary, indent=2))
    if not report["finite"]:
        raise SystemExit("validate_sim: non-finite qpos")
    if not report["e1"]["ok"]:
        raise SystemExit("validate_sim: E1 Coulomb gate failed")


if __name__ == "__main__":
    main()
