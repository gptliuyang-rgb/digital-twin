#!/usr/bin/env python3
"""Render the industrial pipeline to a GIF (headless, no GUI).

Example:
  make assemble
  python3 scripts/render_industrial_gif.py --out artifacts/industrial_demo.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _configure_camera(model, data, renderer) -> None:
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.30, 0.0, 0.55]
    cam.distance = 2.9
    cam.elevation = -8.0
    cam.azimuth = 135.0
    renderer.update_scene(data, camera=cam)


def _label_frame(img, text: str):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([12, 12, 12 + tw + 2 * pad, 12 + th + 2 * pad], fill=(0, 0, 0, 180))
    draw.text((12 + pad, 12 + pad), text, fill=(255, 220, 80), font=font)
    return img


def render_gif(
    *,
    out_path: Path,
    steps_per_phase: int = 60,
    frame_stride: int = 20,
    width: int = 960,
    height: int = 540,
    fps: float = 12.0,
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
    current_phase = {"name": "approach_box"}

    def _capture(label: str | None = None) -> None:
        _configure_camera(env.model, env.data, renderer)
        img = Image.fromarray(renderer.render()).convert("RGBA")
        tag = label or current_phase["name"]
        frames.append(_label_frame(img, tag).convert("RGB"))

    def on_phase(name: str, _env, _result) -> None:
        current_phase["name"] = name
        phase_labels.append(name)
        print(f"[phase] {name}", flush=True)
        if name != "done":
            _capture(name)

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
        _capture("done")

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
    parser.add_argument("--steps-per-phase", type=int, default=60)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=20,
        help="Capture one frame every N kinematic ticks",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=12.0, help="Playback rate of the GIF")
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
        f"scan_geom={info['scan_geometry_ok']} scan_d={info['scan_distance_m']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
