"""QR approach: VLA coarse pose + IBVS fine loop + real decode.

The 6 cm mean SONIC wrist error cannot open-loop a 2–8 cm sticker. This module
is the closed-loop that has to exist in sim *and* on the robot (IBVS itself lives
in runtime/ibvs.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.ibvs import CameraIntrinsics, ibvs_camera_twist, integrate_twist
from sim.qr_scanner import ScanResult, ScanSpec, simulate_scan


@dataclass
class ServoState:
    pos_m: np.ndarray
    rot_wxyz: np.ndarray
    steps: int
    decoded: str | None
    last_scan: ScanResult | None
    pixel_error_px: np.ndarray


def qr_center_px(bbox_uv: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(bbox_uv, dtype=np.float64).reshape(-1, 2)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def servo_until_decode(
    *,
    rgb_fn,
    bbox_fn,
    gun_z_fn,
    qr_pos_m: np.ndarray,
    qr_normal: np.ndarray,
    pos0_m: np.ndarray,
    rot0_wxyz: np.ndarray,
    z_m: float,
    cam: CameraIntrinsics,
    spec: ScanSpec | None = None,
    dt_s: float = 1.0 / 25.0,
    max_steps: int = 80,
    gain: float = 0.8,
    rel_speed_m_s: float = 0.05,
) -> ServoState:
    """Iterate IBVS until simulate_scan decodes or timeout.

    rgb_fn(pos, rot) -> HxWx3, bbox_fn(rgb) -> (N,2) or None.
    """
    spec = spec or ScanSpec()
    pos = np.asarray(pos0_m, dtype=np.float64).copy()
    rot = np.asarray(rot0_wxyz, dtype=np.float64).copy()
    decoded = None
    last = None
    pix = np.array([np.inf, np.inf])
    n_done = 0
    for i in range(1, max_steps + 1):
        n_done = i
        rgb = rgb_fn(pos, rot)
        last = simulate_scan(
            rgb,
            gun_tcp_pos_m=pos,
            gun_tcp_z=gun_z_fn(rot),
            qr_pos_m=qr_pos_m,
            qr_normal=qr_normal,
            rel_speed_m_s=rel_speed_m_s,
            spec=spec,
        )
        if last.decode_ok:
            decoded = last.payload
            pix = np.zeros(2)
            break
        bbox = bbox_fn(rgb)
        if bbox is None:
            continue
        u, v = qr_center_px(bbox)
        result = ibvs_camera_twist(u, v, z_m, cam, gain=gain)
        pix = result.pixel_error_px
        pos, rot = integrate_twist(pos, rot, result.twist_cam, dt_s)
    return ServoState(
        pos_m=pos,
        rot_wxyz=rot,
        steps=n_done,
        decoded=decoded,
        last_scan=last,
        pixel_error_px=pix,
    )
