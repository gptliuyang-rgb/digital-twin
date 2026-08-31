"""Report DexHand frames in the T800 robot base.

YAML composition always runs. Live MuJoCo FK runs when official trees + mujoco
are present. Does not invent CAD flange numbers.

  python3 -m eval.frame_alignment
  make frame-report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assets.combined.flange import chain_yaml
from interface.schema import REPO_ROOT

OUT_DEFAULT = REPO_ROOT / "eval" / "report" / "generated" / "frame_alignment.json"


def _live_snapshot() -> dict:
    try:
        from assets.dexhand2.build.ingest_official import official_mjcf
        from assets.engineai.paths import t800_mjcf
        from sim.mujoco_env.env import CombinedMujocoEnv
        from sim.mujoco_env.frames import alignment_snapshot
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": str(exc)}
    try:
        have_t800 = t800_mjcf().is_file()
    except FileNotFoundError as exc:
        return {"status": "skipped", "reason": str(exc)}
    if not official_mjcf("left", with_mount=True).is_file() or not have_t800:
        return {"status": "skipped", "reason": "official T800/Hand 2 trees not cloned"}
    env = CombinedMujocoEnv(scene="empty")
    env.reset()
    snap = alignment_snapshot(env.model, env.data)
    snap["status"] = "ok"
    return snap


def build_report() -> dict:
    chain = chain_yaml()
    return {
        "kind": "dexhand_in_robot_base",
        "policy_eval_forbidden": chain["policy_eval_forbidden"],
        "yaml_chain": chain,
        "live_fk": _live_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DexHand frames in T800 LINK_BASE")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    report = build_report()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    chain = report["yaml_chain"]
    flange = chain["T_flange"]
    palm = chain["T_wrist_end_palm"]
    print(chain["formula"])
    print(
        f"active flange: {flange['kind']}  "
        f"policy_eval_forbidden={flange['policy_eval_forbidden']}  "
        f"cad_ready={flange['cad_ready']}"
    )
    print(f"T_flange pos_m={flange['pos_m']} quat_wxyz={flange['quat_wxyz']}")
    print(f"T_mount_wrist (official) pos_m={chain['T_mount_wrist']['pos_m']}")
    print(f"T_wrist_end_palm (constant) pos_m={palm['pos_m']} quat_wxyz={palm['quat_wxyz']}")
    live = report["live_fk"]
    if live.get("status") == "ok":
        chk = live["yaml_T_wrist_end_palm_matches_left_mjcf"]
        print(
            f"live FK vs YAML: pos_err={chk['pos_err_m']:.4e} m  "
            f"rot_fro={chk['rot_fro_err']:.4e}  ok={chk['ok']}"
        )
        lw = live["bodies_in_base"]["l_wrist"]
        print(f"l_wrist in LINK_BASE pos_m={lw['pos_m']}")
        print(f"  x={lw['x_axis']}  z={lw['z_axis']}")
    else:
        print(f"live FK skipped: {live.get('reason')}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
