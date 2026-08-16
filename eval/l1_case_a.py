"""L1 Case A: explicit FK then kinematic checks. No physics, no grasp-success.

Requires apply_fk=True. Head/nav come from the labelled fixture in
eval/configs/l1_case_a.yaml, not from the 50-D vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from eval.l1_kinematic import check_coupling, check_joint_limits
from interface.schema import REPO_ROOT, CommandVector, command_dim, load_hand_spec
from vla.adapters.case_a import CaseAConversionError, CaseAToCommandSchema, HeadNavCommand, case_a_dim
from vla.adapters.joint_to_wrist_adapter import T800_KINEMATICS, T800_URDF


def _head_nav(cfg: dict) -> HeadNavCommand:
    fx = cfg["head_nav_fixture"]
    return HeadNavCommand(
        pelvis_height_m=float(fx["pelvis_height_m"]),
        nav_cmd_mps=np.asarray(fx["nav_cmd_mps"], dtype=np.float64),
        loco_mode=int(fx["loco_mode"]),
        tool_trigger=int(fx["tool_trigger"]),
        source=str(fx["label"]),
        head_pos_m=np.asarray(fx["head_pos_m"], dtype=np.float64),
        head_rot6d=np.asarray(fx["head_rot6d"], dtype=np.float64),
    )


def _synthetic_case_a(n: int, spec) -> np.ndarray:
    rng = np.random.default_rng(2)
    dim = case_a_dim(spec)
    vec = np.zeros((n, dim), dtype=np.float64)
    # Small arm motion; fingers stay near zero (inside official limits).
    vec[:, :10] = rng.uniform(-0.2, 0.2, size=(n, 10))
    return vec


def evaluate_case_a(actions: np.ndarray, *, apply_fk: bool, cfg: dict) -> dict:
    spec = load_hand_spec()
    if not apply_fk:
        raise CaseAConversionError("L1 Case A refuses to run without apply_fk=True (ADR-020)")
    if not T800_URDF.is_file() and not T800_KINEMATICS.is_file():
        return {
            "status": "skipped_no_t800_kinematics",
            "note": "Clone engineai-native-sdk or keep assets/engineai/meta/t800_kinematics.yaml",
        }
    adapter = CaseAToCommandSchema.from_t800_urdf()
    head_nav = _head_nav(cfg)
    rows = adapter.convert_chunk(actions, apply_fk=True, head_nav=head_nav)
    assert rows.shape[1] == command_dim(spec)
    parsed = [CommandVector.from_flat_vector(r, spec) for r in rows]
    hands = np.stack([c.left_hand_q for c in parsed])
    wrists = np.stack([np.concatenate([c.left_wrist_pos, c.right_wrist_pos]) for c in parsed])
    finite = bool(np.isfinite(wrists).all())
    limits = check_joint_limits(hands, spec)
    coupling = check_coupling(hands[0], spec)
    return {
        "status": "ok",
        "n": int(actions.shape[0]),
        "case_a_dim": int(actions.shape[-1]),
        "command_schema_dim": int(rows.shape[1]),
        "apply_fk": True,
        "head_nav_source": head_nav.source,
        "wrist_finite": finite,
        "wrist_pos_mean_m": wrists.mean(axis=0).tolist(),
        "limits": limits,
        "coupling": coupling,
        "note": "Geometry only. Not a grasp-success number. Combined T800+Hand still PolicyEvalBlocked.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval/configs/l1_case_a.yaml")
    parser.add_argument("--actions", default="", help="optional .npy Case A dump (N, 50)")
    parser.add_argument("--apply-fk", action="store_true", help="required; refuses to convert without this flag")
    parser.add_argument("--out", default="eval/report/generated/l1_case_a.json")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    spec = load_hand_spec()
    if args.actions:
        actions = np.load(args.actions)
    else:
        actions = _synthetic_case_a(32, spec)
        cfg = {**cfg, "mode": "synthetic_no_ckpt"}
    try:
        report = evaluate_case_a(actions, apply_fk=args.apply_fk, cfg=cfg)
    except CaseAConversionError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.actions:
        report["mode"] = "synthetic_no_ckpt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "wrist_pos_mean_m"}, indent=2))


if __name__ == "__main__":
    main()
