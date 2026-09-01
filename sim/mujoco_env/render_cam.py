"""Offscreen RGB from a named MuJoCo camera. Optional — EGL/OSMesa may be missing."""

from __future__ import annotations

from typing import Any

import numpy as np


def render_camera(
    model,
    data,
    camera_name: str,
    *,
    width: int = 320,
    height: int = 240,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    import mujoco

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=camera_name)
        rgb = np.asarray(renderer.render())
        renderer.close()
        return rgb, {"ok": True, "camera": camera_name, "shape": list(rgb.shape)}
    except Exception as exc:  # noqa: BLE001 — GL backend missing is expected in CI
        return None, {"ok": False, "camera": camera_name, "error": str(exc)}
