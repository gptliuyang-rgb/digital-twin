#!/usr/bin/env python3
"""Open the industrial digital-twin scene in the MuJoCo interactive viewer.

Default path is physics playback (mj_step + gravity-compensated PD + welds).
Pass --kinematic for the L2.3 geometry FSM (mj_forward only).

Requires a display (local desktop or X11 forwarding). Headless servers will fail
unless you use a virtual framebuffer (xvfb-run).

Example:
  make assemble
  python3 scripts/view_industrial_twin.py --steps-per-phase 80
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="MuJoCo viewer for T800 + DexHand2 industrial pipeline")
    parser.add_argument("--steps-per-phase", type=int, default=80, help="Control ticks per FSM phase")
    parser.add_argument("--substeps", type=int, default=24, help="mj_step calls per control tick (physics path)")
    parser.add_argument("--real-time", action="store_true", help="Sleep to match sim timestep")
    parser.add_argument("--sync-every", type=int, default=4, help="Update viewer every N sim steps")
    parser.add_argument("--loop", action="store_true", help="Restart pipeline when it finishes")
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Use mj_forward kinematic playback (L2.3 geometry). Default is physics.",
    )
    parser.add_argument(
        "--no-welds",
        action="store_true",
        help="Physics path: disable constraint welds (carton/gun only move via contact).",
    )
    args = parser.parse_args()

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        print("mujoco is required: pip install -e '.[sim]'", file=sys.stderr)
        raise SystemExit(1) from exc

    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.tasks.industrial_pipeline import run_industrial_pipeline
    from sim.tasks.physics_industrial import run_physics_industrial

    env = CombinedMujocoEnv(scene="industrial")
    dt = float(env.model.opt.timestep)
    use_physics = not args.kinematic

    def run_once(viewer: mujoco.viewer.Handle) -> None:
        phase = {"name": ""}

        def on_phase(name: str, _env, _result) -> None:
            phase["name"] = name
            print(f"[phase] {name}", flush=True)

        step_i = {"n": 0}

        def on_step(_env, _result) -> None:
            step_i["n"] += 1
            if step_i["n"] % max(1, args.sync_every) == 0:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
            if args.real_time:
                time.sleep(dt * (args.substeps if use_physics else 1))

        try:
            if use_physics:
                result = run_physics_industrial(
                    env,
                    steps_per_phase=args.steps_per_phase,
                    substeps=args.substeps,
                    use_welds=not args.no_welds,
                    on_phase=on_phase,
                    on_step=on_step,
                )
            else:
                result = run_industrial_pipeline(
                    env,
                    steps_per_phase=args.steps_per_phase,
                    on_phase=on_phase,
                    on_step=on_step,
                )
        except KeyboardInterrupt:
            return
        extra = ""
        if use_physics:
            extra = (
                f" weld={result.constraint_weld} drop={result.box_drop_m} "
                f"tau_max={result.max_abs_tau_nm:.1f}"
            )
        print(
            f"[done] steps={result.n_steps} finite={result.finite} "
            f"scan_geom={result.scan_geometry_ok} scan_d={result.scan_distance_m}{extra}",
            flush=True,
        )
        print(f"[note] {result.note}", flush=True)

    mode = "physics (mj_step + PD + welds)" if use_physics else "kinematic (mj_forward)"
    print(
        f"Industrial twin viewer — {mode}. Identity flange, uncalibrated contact.\n"
        "Mouse: orbit/zoom/pan. Space: pause. Esc or close window to quit.\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.30, 0.0, 0.55]
        viewer.cam.distance = 2.9
        viewer.cam.elevation = -8
        viewer.cam.azimuth = 135

        while viewer.is_running():
            env.reset()
            run_once(viewer)
            if not args.loop:
                break
            print("[loop] restarting pipeline…", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
