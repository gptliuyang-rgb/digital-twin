"""Replay a recorded 75-D command stream through PolicyClient in the twin.

Default path is kinematic: apply hand q from the command and DLS-IK the wrists
to the packed wrist positions, then ``mj_forward``. Uncalibrated contact (ADR-004)
is not used. Tracking error is command vs realized site pose after IK — a
stack-integrity metric, not a policy score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.gates import assert_no_grasp_success_rate
from interface.schema import REPO_ROOT, CommandVector, load_hand_spec
from vla.client.policy_client import PolicyClient


def _set_hands(env, left_q: np.ndarray, right_q: np.ndarray) -> None:
    env.data.qpos[env.handles.left_hand.qadr] = np.asarray(left_q, dtype=np.float64)
    env.data.qpos[env.handles.right_hand.qadr] = np.asarray(right_q, dtype=np.float64)


def replay_recorded_commands(
    npz_path: Path,
    *,
    stride: int = 1,
    max_steps: int | None = None,
) -> dict[str, Any]:
    import mujoco

    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.mujoco_env.ik import dls_ik_pos

    spec = load_hand_spec()
    blob = np.load(npz_path, allow_pickle=True)
    commands = np.asarray(blob["commands"], dtype=np.float64)
    if max_steps is not None:
        commands = commands[: int(max_steps)]
    commands = commands[:: max(1, int(stride))]

    cursor = {"i": 0}

    def infer_fn(_obs: dict) -> np.ndarray:
        i = min(cursor["i"], len(commands) - 1)
        cursor["i"] += 1
        return commands[i]

    client = PolicyClient(infer_fn, spec=spec)
    env = CombinedMujocoEnv(scene="industrial")
    env.reset()
    if "qpos" in blob.files:
        env.data.qpos[:] = np.asarray(blob["qpos"][0], dtype=np.float64)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
    left_arm = env.handles.arm_joints["left"]
    right_arm = env.handles.arm_joints["right"]

    wrist_err = []
    hand_err = []
    finite = True
    n = 0
    for _ in range(len(commands)):
        cmd: CommandVector = client.step({"t": n})
        _set_hands(env, cmd.left_hand_q, cmd.right_hand_q)
        ik_l = dls_ik_pos(
            env.model,
            env.data,
            "l_wrist",
            cmd.left_wrist_pos,
            left_arm,
            damping=0.05,
            iters=24,
            step=0.55,
            max_delta_rad=0.12,
        )
        ik_r = dls_ik_pos(
            env.model,
            env.data,
            "r_wrist",
            cmd.right_wrist_pos,
            right_arm,
            damping=0.05,
            iters=24,
            step=0.55,
            max_delta_rad=0.12,
        )
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        finite = finite and bool(np.isfinite(env.data.qpos).all())
        lw = env.xpos("l_wrist")
        rw = env.xpos("r_wrist")
        wrist_err.append(0.5 * (float(np.linalg.norm(lw - cmd.left_wrist_pos)) + float(ik_l["err_m"])))
        wrist_err.append(0.5 * (float(np.linalg.norm(rw - cmd.right_wrist_pos)) + float(ik_r["err_m"])))
        hand_err.append(float(np.max(np.abs(env.data.qpos[env.handles.left_hand.qadr] - cmd.left_hand_q))))
        hand_err.append(float(np.max(np.abs(env.data.qpos[env.handles.right_hand.qadr] - cmd.right_hand_q))))
        n += 1

    report = {
        "kind": "sim_policy_replay",
        "n_steps": n,
        "stride": int(stride),
        "mean_wrist_err_m": float(np.mean(wrist_err)) if wrist_err else None,
        "max_wrist_err_m": float(np.max(wrist_err)) if wrist_err else None,
        "mean_hand_q_err_rad": float(np.mean(hand_err)) if hand_err else None,
        "finite": finite,
        "policy_client": "vla.client.policy_client.PolicyClient",
        "infer_fn": "recorded_npz_playback",
        "kinematic_assist": True,
        "policy_eval_forbidden": True,
        "note": (
            "Scripted recorded commands through PolicyClient, kinematic apply. "
            "Not GR00T/π0.5. Identity flange. Uncalibrated contact."
        ),
    }
    assert_no_grasp_success_rate(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="eval/report/generated/sim_replay.json")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    report = replay_recorded_commands(
        Path(args.dataset),
        stride=args.stride,
        max_steps=args.max_steps or None,
    )
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_steps", "mean_wrist_err_m", "finite")}, indent=2))


if __name__ == "__main__":
    main()
