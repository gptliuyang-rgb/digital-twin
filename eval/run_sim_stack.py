"""Run the simulation-only digital-twin stack end-to-end (no real robot).

Order: assemble → pad metrics → record industrial episode → L0 on that
dataset → MuJoCo L1 → L2 (optional) → QR envelope → PolicyClient replay.

Does not fill live spec friction/stiffness. Does not claim grasp_success_rate
(ADR-004). Identity flange (ADR-006). Kinematic industrial (ADR-007).
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from eval.gates import assert_no_grasp_success_rate
from interface.schema import REPO_ROOT

GEN = REPO_ROOT / "eval" / "report" / "generated"
DOCS = REPO_ROOT / "docs" / "reports" / "SIM_STACK.md"


def _stage(name: str, fn, report: dict[str, Any]) -> Any:
    try:
        value = fn()
        report["stages"][name] = {"ok": True}
        return value
    except Exception as exc:  # noqa: BLE001
        report["stages"][name] = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
        report["ok"] = False
        return None


def run_sim_stack(
    *,
    quick: bool = False,
    with_l2: bool = True,
    physics_industrial: bool = False,
    steps_per_phase: int | None = None,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> dict[str, Any]:
    GEN.mkdir(parents=True, exist_ok=True)
    steps = int(steps_per_phase if steps_per_phase is not None else (8 if quick else 24))
    do_l2 = bool(with_l2)
    npz_path = GEN / "sim_episode.npz"
    report: dict[str, Any] = {
        "kind": "sim_stack",
        "quick": quick,
        "steps_per_phase": steps,
        "with_l2": do_l2,
        "physics_industrial": physics_industrial,
        "policy_eval_forbidden": True,
        "ok": True,
        "stages": {},
        "note": (
            "Simulation-only bring-up. Uncalibrated contact; identity flange; "
            "kinematic industrial playback. Not sim2real pick rates."
        ),
    }

    def assemble() -> dict:
        from assets.combined.assemble import write_generated

        info = write_generated(mjcf=True, urdf=True)
        try:
            from assets.combined.assemble import compile_mjcf

            model = compile_mjcf(Path(info["mjcf"]).read_text(encoding="utf-8"))
            info["nq"] = int(model.nq)
            info["nu"] = int(model.nu)
        except Exception as exc:  # noqa: BLE001
            info["compile_error"] = str(exc)
        report["assemble"] = {k: info[k] for k in info if k != "urdf"}
        return info

    _stage("assemble", assemble, report)

    def pads() -> dict:
        from assets.dexhand2.build.pad_metrics import write_report

        return write_report(GEN / "phase1_baseline.json", REPO_ROOT / "docs" / "reports" / "PHASE_1_baseline.md")

    pad_report = _stage("pad_metrics", pads, report)
    if pad_report:
        sides = pad_report.get("sides", {})
        report["pad"] = {
            side: {
                "derived_max_pad_sphere_m": block.get("derived_max_pad_sphere_m"),
                "derived_pad_under_2mm": block.get("derived_pad_under_2mm"),
                "n_tip_sites": block.get("n_tip_sites"),
            }
            for side, block in sides.items()
            if isinstance(block, dict)
        }

    def record() -> dict:
        from eval.record_sim_episode import record_industrial_episode

        return record_industrial_episode(steps_per_phase=steps, out_npz=npz_path)

    rec = _stage("record", record, report)
    if rec:
        report["record"] = rec

    def l0() -> dict:
        from eval.l0_offline_replay import evaluate_recorded_commands

        commands = np.load(npz_path)["commands"]
        out = evaluate_recorded_commands(commands)
        (GEN / "l0_sim.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    l0_rep = _stage("l0", l0, report)
    if l0_rep:
        report["l0"] = {
            "mode": l0_rep.get("mode"),
            "mse_mean": l0_rep.get("mse_mean"),
            "left_hand_outliers": l0_rep.get("left_hand_outliers"),
            "right_hand_outliers": l0_rep.get("right_hand_outliers"),
            "n_chunks": l0_rep.get("n_chunks"),
        }

    def l1() -> dict:
        from eval.l1_mujoco import evaluate_l1_mujoco

        qpos = np.load(npz_path)["qpos"]
        out = evaluate_l1_mujoco(qpos_dataset=qpos)
        (GEN / "l1_mujoco.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    l1_rep = _stage("l1_mujoco", l1, report)
    if l1_rep:
        report["l1"] = {
            "ik_err_m": l1_rep.get("ik", {}).get("err_m"),
            "ik_ok": l1_rep.get("ik", {}).get("ok"),
            "coupling_ok": l1_rep.get("coupling", {}).get("ok"),
            "ncon_at_primitive": l1_rep.get("ncon_at_primitive"),
        }

    if do_l2:

        def l2() -> dict:
            import sys

            from eval.l2_mujoco_closedloop import main as l2_main

            out = GEN / "l2_sim.json"
            old = sys.argv
            argv = ["l2", "--out", str(out)]
            if physics_industrial:
                argv.append("--physics-industrial")
            try:
                sys.argv = argv
                l2_main()
            finally:
                sys.argv = old
            return json.loads(out.read_text(encoding="utf-8"))

        l2_rep = _stage("l2", l2, report)
        if l2_rep:
            report["l2"] = {
                "status": l2_rep.get("status"),
                "physics": l2_rep.get("physics"),
                "l2_3_scan_geometry_ok": (l2_rep.get("l2_3_industrial") or {}).get("scan_geometry_ok"),
            }
    else:
        report["stages"]["l2"] = {"ok": True, "skipped": True, "reason": "quick_or_disabled"}

    def envelope() -> dict:
        from sim.qr_envelope import geometry_envelope

        out = geometry_envelope()
        (GEN / "qr_envelope.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    env_rep = _stage("qr_envelope", envelope, report)
    if env_rep:
        report["qr_envelope"] = {
            "n_cells": env_rep.get("n_cells"),
            "n_geometry_ok": env_rep.get("n_geometry_ok"),
        }

    def replay() -> dict:
        from eval.replay_sim_policy import replay_recorded_commands

        return replay_recorded_commands(
            npz_path,
            stride=2 if quick else 1,
            max_steps=40 if quick else None,
        )

    replay_rep = _stage("policy_replay", replay, report)
    if replay_rep:
        report["replay"] = replay_rep
        (GEN / "sim_replay.json").write_text(json.dumps(replay_rep, indent=2), encoding="utf-8")

    assert_no_grasp_success_rate(report)
    out_json = out_json or (GEN / "sim_stack.json")
    out_md = out_md or DOCS
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(report), encoding="utf-8")
    report["json"] = str(out_json)
    report["md"] = str(out_md)
    return report


def _fmt_m(x: Any) -> str:
    try:
        return f"{float(x) * 1000:.2f} mm"
    except (TypeError, ValueError):
        return "n/a"


def _markdown(report: dict[str, Any]) -> str:
    stages = report.get("stages", {})
    pad = report.get("pad") or {}
    rec = report.get("record") or {}
    l0 = report.get("l0") or {}
    l1 = report.get("l1") or {}
    l2 = report.get("l2") or {}
    qr = report.get("qr_envelope") or {}
    replay = report.get("replay") or {}
    lines = [
        "# Sim stack (simulation only)",
        "",
        "Generated by `python -m eval.run_sim_stack` / `make sim`.",
        "No real robot. Uncalibrated contact (ADR-004). Identity flange (ADR-006).",
        "Industrial carton/gun motion is kinematic playback (ADR-007).",
        "",
        f"- **ok:** `{report.get('ok')}`",
        f"- **quick:** `{report.get('quick')}`",
        f"- **steps_per_phase:** `{report.get('steps_per_phase')}`",
        "",
        "## Stages",
        "",
        "| stage | ok |",
        "|---|---|",
    ]
    for name, block in stages.items():
        flag = "yes" if block.get("ok") else "NO"
        extra = ""
        if block.get("skipped"):
            extra = " (skipped)"
        if block.get("error"):
            extra = f" — {block['error'][:80]}"
        lines.append(f"| {name} | {flag}{extra} |")
    lines.extend(
        [
            "",
            "## Numbers (relative / stack integrity, not pick rates)",
            "",
            f"- pad max site→sphere (right): {_fmt_m((pad.get('right') or {}).get('derived_max_pad_sphere_m'))}; "
            f"< 2 mm: `{(pad.get('right') or {}).get('derived_pad_under_2mm')}`",
            f"- pad max site→sphere (left): {_fmt_m((pad.get('left') or {}).get('derived_max_pad_sphere_m'))}; "
            f"< 2 mm: `{(pad.get('left') or {}).get('derived_pad_under_2mm')}`",
            f"- recorded steps: `{rec.get('n_steps')}`; scan_geometry_ok: `{rec.get('scan_geometry_ok')}`; "
            f"scan_decode_ok: `{rec.get('scan_decode_ok')}`",
            f"- L0 identity mse_mean: `{l0.get('mse_mean')}`; outliers L/R: "
            f"`{l0.get('left_hand_outliers')}` / `{l0.get('right_hand_outliers')}`",
            f"- L1 IK err: {_fmt_m(l1.get('ik_err_m'))}; coupling_ok: `{l1.get('coupling_ok')}`; "
            f"ncon: `{l1.get('ncon_at_primitive')}`",
            f"- L2 status: `{l2.get('status')}`; industrial scan_geometry_ok: `{l2.get('l2_3_scan_geometry_ok')}`",
            f"- QR envelope geometry_ok cells: `{qr.get('n_geometry_ok')}` / `{qr.get('n_cells')}`",
            f"- PolicyClient replay mean wrist err: {_fmt_m(replay.get('mean_wrist_err_m'))}; "
            f"hand q err: `{replay.get('mean_hand_q_err_rad')}` rad; finite: `{replay.get('finite')}`",
            "",
            "## What this does not mean",
            "",
            "- Not `grasp_success_rate`. Live μ/k are still `REQUIRED_INPUT`.",
            "- Not a CAD flange. `kinematic_bringup_identity` is sim-only.",
            "- Not GR00T / π0.5 / SONIC. Replay uses a recorded scripted policy.",
            "- Physics industrial (`make eval-l2-physics`) still drops the box; it is honesty, not a task pass.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip L2 micro scan; shorter episode")
    parser.add_argument("--steps-per-phase", type=int, default=0)
    parser.add_argument("--l2", dest="with_l2", action="store_true", default=None)
    parser.add_argument("--no-l2", dest="with_l2", action="store_false")
    parser.add_argument("--physics-industrial", action="store_true")
    args = parser.parse_args()
    with_l2 = (not args.quick) if args.with_l2 is None else bool(args.with_l2)
    report = run_sim_stack(
        quick=args.quick,
        with_l2=with_l2,
        physics_industrial=args.physics_industrial,
        steps_per_phase=args.steps_per_phase or None,
    )
    print(json.dumps({"ok": report["ok"], "stages": {k: v.get("ok") for k, v in report["stages"].items()}}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
