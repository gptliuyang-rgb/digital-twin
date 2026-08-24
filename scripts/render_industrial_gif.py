#!/usr/bin/env python3
"""Render the industrial pipeline to a GIF (headless, no GUI).

Example:
  make assemble
  python3 scripts/render_industrial_gif.py --out industrial_demo.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _configure_camera(model, data, renderer) -> None:
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.38, 0.0, 0.95]
    cam.distance = 3.2
    cam.elevation = -18.0
    cam.azimuth = 135.0
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    mujoco.mjv_updateScene(
        model,
        data,
        opt,
        pert,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    renderer.update_scene(data, camera=cam)


def render_gif(
    *,
    out_path: Path,
    steps_per_phase: int = 80,
    frame_stride: int = 40,
    width: int = 960,
    height: int = 540,
    fps: float = 25.0,
    duration_ms: int | None = None,
) -> dict:
    import mujoco
    from PIL import Image

    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.tasks.industrial_pipeline import run_industrial_pipeline

    env = CombinedMujocoEnv(scene="industrial")
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), width)
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(env.model, width=width, height=height)
    frames: list[Image.Image] = []
    phase_labels: list[str] = []
    step_i = 0

    def _capture() -> None:
        _configure_camera(env.model, env.data, renderer)
        frames.append(Image.fromarray(renderer.render()))

    def on_phase(name: str, _env, _result) -> None:
        phase_labels.append(name)
        print(f"[phase] {name}", flush=True)
        if name != "done":  # skip metric-only snap frame
            _capture()

    def on_step(_env, _result) -> None:
        nonlocal step_i
        step_i += 1
        if step_i % frame_stride != 0:
            return
        _capture()

    result = run_industrial_pipeline(
        env,
        steps_per_phase=steps_per_phase,
        on_phase=on_phase,
        on_step=on_step,
    )

    if not frames:
        _configure_camera(env.model, env.data, renderer)
        frames.append(Image.fromarray(renderer.render()))

    ms = duration_ms if duration_ms is not None else int(round(1000.0 / max(fps, 1.0)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=ms,
        loop=0,
        optimize=False,
    )
    renderer.close()
    return {
        "out": str(out_path.resolve()),
        "n_frames": len(frames),
        "fps": fps,
        "duration_ms": ms,
        "steps": result.n_steps,
        "scan_geometry_ok": result.scan_geometry_ok,
        "scan_distance_m": result.scan_distance_m,
        "phases": phase_labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render industrial twin pipeline to GIF")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/industrial_demo.gif"),
        help="Output GIF path",
    )
    parser.add_argument("--steps-per-phase", type=int, default=80)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=25,
        help="Capture one frame every N sim steps (~40 fps at 1 kHz when stride=25)",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=25.0, help="Playback rate of the GIF")
    args = parser.parse_args()

    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        print("mujoco is required: pip install -e '.[sim]'", file=sys.stderr)
        raise SystemExit(1) from exc

    info = render_gif(
        out_path=args.out,
        steps_per_phase=args.steps_per_phase,
        frame_stride=args.frame_stride,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(
        f"[saved] {info['out']} ({info['n_frames']} frames @ {info['fps']} fps, "
        f"scan_geom={info['scan_geometry_ok']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
