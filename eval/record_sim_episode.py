"""Record a kinematic industrial episode to npz (qpos + 75-D commands).

This is sim-only playback (ADR-007). Not a VLA dataset and not a sim2real claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.gates import assert_no_grasp_success_rate
from interface.schema import REPO_ROOT, command_dim, load_hand_spec
from sim.command_from_state import command_from_env


def record_industrial_episode(
    *,
    steps_per_phase: int = 24,
    out_npz: Path | None = None,
) -> dict:
    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.tasks.industrial_pipeline import run_industrial_pipeline

    spec = load_hand_spec()
    env = CombinedMujocoEnv(scene="industrial")
    rows: dict[str, list] = {
        "commands": [],
        "qpos": [],
        "qvel": [],
        "time": [],
        "phase": [],
        "box_xyz": [],
        "wrist_left": [],
        "wrist_right": [],
        "ik_err_last": [],
    }

    def on_step(step_env, result) -> None:
        phase = result.phases[-1] if result.phases else ""
        cmd = command_from_env(step_env, spec, phase=phase)
        rows["commands"].append(cmd.to_flat_vector())
        rows["qpos"].append(step_env.data.qpos.copy())
        rows["qvel"].append(step_env.data.qvel.copy())
        rows["time"].append(float(step_env.data.time))
        rows["phase"].append(phase)
        rows["box_xyz"].append(step_env.xpos("box_0").copy())
        rows["wrist_left"].append(cmd.left_wrist_pos.copy())
        rows["wrist_right"].append(cmd.right_wrist_pos.copy())
        rows["ik_err_last"].append(float(result.ik_err_m[-1]) if result.ik_err_m else float("nan"))

    result = run_industrial_pipeline(env, steps_per_phase=steps_per_phase, on_step=on_step)
    commands = np.stack(rows["commands"], axis=0)
    payload = {
        "commands": commands,
        "qpos": np.stack(rows["qpos"], axis=0),
        "qvel": np.stack(rows["qvel"], axis=0),
        "time": np.asarray(rows["time"], dtype=np.float64),
        "phase": np.asarray(rows["phase"]),
        "box_xyz": np.stack(rows["box_xyz"], axis=0),
        "wrist_left": np.stack(rows["wrist_left"], axis=0),
        "wrist_right": np.stack(rows["wrist_right"], axis=0),
        "ik_err_last": np.asarray(rows["ik_err_last"], dtype=np.float64),
        "command_dim": np.asarray([command_dim(spec)], dtype=np.int32),
        "policy_eval_forbidden": np.asarray([True]),
        "kinematic_assist": np.asarray([True]),
        "n_steps": np.asarray([len(commands)], dtype=np.int32),
        "scan_geometry_ok": np.asarray([bool(result.scan_geometry_ok)]),
        "scan_distance_m": np.asarray(
            [result.scan_distance_m if result.scan_distance_m is not None else np.nan]
        ),
    }
    meta = {
        "kind": "sim_industrial_episode",
        "n_steps": int(len(commands)),
        "command_dim": int(command_dim(spec)),
        "phases": result.phases,
        "scan_geometry_ok": result.scan_geometry_ok,
        "scan_decode_ok": result.scan_decode_ok,
        "scan_distance_m": result.scan_distance_m,
        "finite": result.finite,
        "policy_eval_forbidden": True,
        "kinematic_assist": True,
        "command_wrist_frame": "hand_l_r_wrist_site",
        "note": (
            "Kinematic industrial playback recorded as 75-D commands + qpos. "
            "Not a VLA dataset. Identity flange. Uncalibrated contact."
        ),
    }
    assert_no_grasp_success_rate(meta)
    if out_npz is not None:
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **payload)
        out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta["npz"] = str(out_npz)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-phase", type=int, default=24)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "eval" / "report" / "generated" / "sim_episode.npz"),
    )
    args = parser.parse_args()
    meta = record_industrial_episode(steps_per_phase=args.steps_per_phase, out_npz=Path(args.out))
    print(json.dumps({k: meta[k] for k in ("n_steps", "command_dim", "scan_geometry_ok", "npz") if k in meta}, indent=2))


if __name__ == "__main__":
    main()
