#!/usr/bin/env python3
"""Open the industrial digital-twin scene in the MuJoCo interactive viewer.

Requires a display (local desktop or X11 forwarding). Headless servers will fail
unless you use a virtual framebuffer (xvfb-run).

Example:
  make assemble
  python3 scripts/view_industrial_twin.py --steps-per-phase 120
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="MuJoCo viewer for T800 + DexHand2 industrial pipeline")
    parser.add_argument("--steps-per-phase", type=int, default=100, help="Physics steps per FSM phase")
    parser.add_argument("--real-time", action="store_true", help="Sleep to match 1 kHz sim timestep")
    parser.add_argument("--loop", action="store_true", help="Restart pipeline when it finishes")
    args = parser.parse_args()

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        print("mujoco is required: pip install -e '.[sim]'", file=sys.stderr)
        raise SystemExit(1) from exc

    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.tasks.industrial_pipeline import run_industrial_pipeline

    env = CombinedMujocoEnv(scene="industrial")
    dt = float(env.model.opt.timestep)

    def run_once(viewer: mujoco.viewer.Handle) -> None:
        phase = {"name": ""}

        def on_phase(name: str, _env, _result) -> None:
            phase["name"] = name
            print(f"[phase] {name}", flush=True)

        def on_step(_env, _result) -> None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            if args.real_time:
                time.sleep(dt)

        try:
            result = run_industrial_pipeline(
                env,
                steps_per_phase=args.steps_per_phase,
                on_phase=on_phase,
                on_step=on_step,
            )
        except KeyboardInterrupt:
            return
        print(
            f"[done] steps={result.n_steps} finite={result.finite} "
            f"scan_geom={result.scan_geometry_ok} scan_d={result.scan_distance_m}",
            flush=True,
        )
        print(f"[note] {result.note}", flush=True)

    print(
        "Industrial twin viewer — identity flange, uncalibrated contact.\n"
        "Mouse: orbit/zoom/pan. Space: pause. Esc or close window to quit.\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Pull camera back so pallet + robot are visible.
        viewer.cam.lookat[:] = [0.55, 0.0, 0.95]
        viewer.cam.distance = 3.2
        viewer.cam.elevation = -18
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
