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
    steps_per_phase: int = 48,
    frame_stride: int = 8,
    width: int = 960,
    height: int = 540,
    fps: float = 12.0,
    duration_ms: int | None = None,
    physics: bool = True,
    substeps: int = 12,
    use_welds: bool = True,
) -> dict:
    import mujoco
    from PIL import Image

    from sim.mujoco_env.env import CombinedMujocoEnv
    from sim.tasks.industrial_pipeline import run_industrial_pipeline
    from sim.tasks.physics_industrial import run_physics_industrial

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

    if physics:
        result = run_physics_industrial(
            env,
            steps_per_phase=steps_per_phase,
            substeps=substeps,
            use_welds=use_welds,
            on_phase=on_phase,
            on_step=on_step,
        )
    else:
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
        "physics": physics,
        "constraint_weld": bool(getattr(result, "constraint_weld", False)),
        "kinematic_assist": bool(getattr(result, "kinematic_assist", not physics)),
    }


def render_gun_closeup(
    *,
    out_path: Path,
    width: int = 960,
    height: int = 540,
    in_hand_path: Path | None = None,
) -> dict:
    """Still of the barcode scanner on the pick bench; optional in-hand still."""
    import mujoco
    from PIL import Image

    from sim.mujoco_env.env import CombinedMujocoEnv

    env = CombinedMujocoEnv(scene="industrial")
    env.reset()
    # Let the freejoint scanner settle on the bench before the still.
    q_hold = env.data.qpos.copy()
    from hand.grasp_primitives import GraspLibrary
    from interface.schema import load_hand_spec

    q_open = GraspLibrary(load_hand_spec()).q_active("open", 0.0)
    for _ in range(180):
        env.step_mit(q_open, q_open, body_q_des=q_hold)
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), width)
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(env.model, width=width, height=height)

    def _shot(lookat, distance: float, elevation: float, azimuth: float, dest: Path) -> None:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
        renderer.update_scene(env.data, camera=cam)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(renderer.render()).convert("RGB").save(dest)

    gun = env.xpos("scan_gun")
    _shot([gun[0] + 0.03, gun[1] - 0.05, gun[2]], 0.40, -52.0, 155.0, out_path)
    saved = {"out": str(out_path.resolve()), "lookat": gun.tolist()}
    if in_hand_path is not None:
        from sim.tasks.physics_industrial import run_physics_industrial

        def on_phase(name: str, _env, _result) -> None:
            if name not in {"grip_gun", "scan"}:
                return
            g = env.xpos("scan_gun")
            _shot([g[0] + 0.02, g[1], g[2] + 0.02], 0.32, -14.0, 142.0, in_hand_path)

        run_physics_industrial(env, steps_per_phase=16, substeps=8, on_phase=on_phase)
        saved["in_hand"] = str(in_hand_path.resolve())
    renderer.close()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Render industrial twin pipeline to GIF")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/industrial_demo.gif"),
        help="Output GIF path",
    )
    parser.add_argument("--steps-per-phase", type=int, default=48)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=8,
        help="Capture one frame every N control ticks",
    )
    parser.add_argument("--substeps", type=int, default=12, help="mj_step per tick (physics path)")
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Use mj_forward kinematic playback instead of physics",
    )
    parser.add_argument(
        "--no-welds",
        action="store_true",
        help="Physics path: disable constraint welds",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=12.0, help="Playback rate of the GIF")
    parser.add_argument(
        "--gun-closeup",
        type=Path,
        default=None,
        help="Write a still of the barcode scanner (identity pose on the bench)",
    )
    parser.add_argument(
        "--gun-in-hand",
        type=Path,
        default=None,
        help="Write a still after the pipeline seats the scanner in the right hand",
    )
    parser.add_argument("--skip-gif", action="store_true", help="Skip the full pipeline GIF")
    args = parser.parse_args()

    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        print("mujoco is required: pip install -e '.[sim]'", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.gun_closeup or args.gun_in_hand:
        still = render_gun_closeup(
            out_path=args.gun_closeup or Path("artifacts/scan_gun_closeup.png"),
            width=args.width,
            height=args.height,
            in_hand_path=args.gun_in_hand,
        )
        print(f"[saved] {still}", flush=True)
    if args.skip_gif:
        if args.gun_closeup is None and args.gun_in_hand is None:
            raise SystemExit("--skip-gif requires --gun-closeup or --gun-in-hand")
        return 0

    info = render_gif(
        out_path=args.out,
        steps_per_phase=args.steps_per_phase,
        frame_stride=args.frame_stride,
        width=args.width,
        height=args.height,
        fps=args.fps,
        physics=not args.kinematic,
        substeps=args.substeps,
        use_welds=not args.no_welds,
    )
    print(
        f"[saved] {info['out']} ({info['n_frames']} frames @ {info['fps']} fps, "
        f"scan_geom={info['scan_geometry_ok']} scan_d={info['scan_distance_m']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
