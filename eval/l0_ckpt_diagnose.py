"""L0 ckpt diagnose: classify action space before replay or SONIC load.

Does not need a trained weight file. Pass `--dim`, a numpy dump, or modality.json.
G1 29-DoF vectors are refused. Unknown dims raise (no silent pad/slice).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from interface.schema import command_dim, load_hand_spec
from vla.adapters.action_space import ActionSpaceMismatch, diagnose_action_vector
from wbc.checkpoint import G1CheckpointIncompatible


def _load_vector(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        blob = np.load(p)
        key = "action" if "action" in blob.files else blob.files[0]
        return np.asarray(blob[key])
    raise ValueError(f"unsupported action dump {p}; use .npy or .npz")


def _load_modality(path: str) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def diagnose(
    *,
    dim: int | None = None,
    vec: np.ndarray | None = None,
    modality: dict | None = None,
) -> dict:
    spec = load_hand_spec()
    report = diagnose_action_vector(vec, dim=dim, modality=modality, spec=spec)
    out = report.to_dict()
    out["command_schema_v1_dim"] = command_dim(spec)
    out["l0_replay_allowed"] = bool(report.aligned_with_command_schema)
    out["sonic_g1_load"] = "forbidden"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval/configs/l0_diagnose.yaml")
    parser.add_argument("--dim", type=int, default=0, help="action last-dim if no dump")
    parser.add_argument("--actions", default="", help=".npy/.npz action dump")
    parser.add_argument("--modality", default="")
    parser.add_argument("--out", default="eval/report/generated/l0_diagnose.json")
    args = parser.parse_args()
    vec = _load_vector(args.actions) if args.actions else None
    modality = _load_modality(args.modality) if args.modality else None
    dim = args.dim if args.dim else None
    if vec is None and dim is None and modality is None:
        spec = load_hand_spec()
        from vla.adapters.action_space import known_layouts

        report = {
            "status": "catalog",
            "note": "No ckpt/dump provided. Known layouts only; not a model evaluation.",
            "command_schema_v1_dim": command_dim(spec),
            "known_layouts": {k: v["dim"] for k, v in known_layouts(spec).items()},
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    try:
        report = diagnose(dim=dim, vec=vec, modality=modality)
        report["status"] = "ok"
    except (ActionSpaceMismatch, G1CheckpointIncompatible) as exc:
        report = {"status": "refused", "error": str(exc)}
        if vec is not None:
            report["dim"] = int(np.asarray(vec).shape[-1])
        elif dim:
            report["dim"] = int(dim)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("status") == "refused":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
