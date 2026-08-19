"""Image-based visual servo (IBVS) for the QR-scan wrist loop.

Shared sim/real code: no simulator imports. Camera numbers must be passed in;
this module never reads a typical-webcam default out of calib_real.yaml.

Point-feature interaction matrix: Chaumette & Hutchinson, IEEE RAM 2006.
Pixel error is converted to normalized image coordinates before L is built.
Rotation is not averaged here — the output is a camera-frame twist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int
    source: str

    @classmethod
    def synthetic_pinhole(cls) -> CameraIntrinsics:
        """Matches eval/qr_envelope.py (fx=600, 640×480). NOT a robot calibration."""
        return cls(
            fx_px=600.0,
            fy_px=600.0,
            cx_px=320.0,
            cy_px=240.0,
            width_px=640,
            height_px=480,
            source="synthetic_pinhole_not_real",
        )


@dataclass(frozen=True)
class IbvsResult:
    twist_cam: np.ndarray  # [vx, vy, vz, wx, wy, wz]  m/s, rad/s
    error_norm: np.ndarray  # [x−x*, y−y*] in normalized image coords
    pixel_error_px: np.ndarray


def pixel_to_normalized(u_px: float, v_px: float, cam: CameraIntrinsics) -> tuple[float, float]:
    return (float(u_px) - cam.cx_px) / cam.fx_px, (float(v_px) - cam.cy_px) / cam.fy_px


def interaction_matrix_point(x: float, y: float, z_m: float) -> np.ndarray:
    """2×6 interaction matrix of a point feature at depth z_m (m)."""
    if z_m <= 1e-4:
        raise ValueError("IBVS depth z_m must be > 0.1 mm")
    inv_z = 1.0 / z_m
    return np.array(
        [
            [-inv_z, 0.0, x * inv_z, x * y, -(1.0 + x * x), y],
            [0.0, -inv_z, y * inv_z, 1.0 + y * y, -x * y, -x],
        ],
        dtype=np.float64,
    )


def ibvs_camera_twist(
    u_px: float,
    v_px: float,
    z_m: float,
    cam: CameraIntrinsics,
    *,
    u_star_px: float | None = None,
    v_star_px: float | None = None,
    gain: float = 0.5,
) -> IbvsResult:
    """v_cam = −λ L⁺ e. Default target is the principal point (QR centered)."""
    u_star = cam.cx_px if u_star_px is None else u_star_px
    v_star = cam.cy_px if v_star_px is None else v_star_px
    x, y = pixel_to_normalized(u_px, v_px, cam)
    xs, ys = pixel_to_normalized(u_star, v_star, cam)
    err = np.array([x - xs, y - ys], dtype=np.float64)
    inter = interaction_matrix_point(x, y, z_m)
    twist = -float(gain) * np.linalg.pinv(inter) @ err
    return IbvsResult(
        twist_cam=twist,
        error_norm=err,
        pixel_error_px=np.array([u_px - u_star, v_px - v_star], dtype=np.float64),
    )


def integrate_twist(
    pos_m: np.ndarray,
    rot_wxyz: np.ndarray,
    twist_cam: np.ndarray,
    dt_s: float,
    *,
    cam_in_wrist_rot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler-integrate a camera twist into a wrist pose. Identity cam→wrist if unset."""
    from vla.adapters.rotation import (
        matrix_to_quaternion_wxyz,
        quaternion_wxyz_to_matrix,
    )

    r_cam = np.eye(3) if cam_in_wrist_rot is None else np.asarray(cam_in_wrist_rot, dtype=np.float64)
    v_w = r_cam @ twist_cam[:3]
    w_w = r_cam @ twist_cam[3:]
    new_pos = np.asarray(pos_m, dtype=np.float64) + v_w * dt_s
    r = quaternion_wxyz_to_matrix(rot_wxyz)
    # Small-angle so(3) increment.
    wx, wy, wz = w_w * dt_s
    skew = np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])
    r_new = r @ (np.eye(3) + skew)
    u, _, vt = np.linalg.svd(r_new)
    r_new = u @ vt
    if np.linalg.det(r_new) < 0:
        u[:, -1] *= -1
        r_new = u @ vt
    return new_pos, matrix_to_quaternion_wxyz(r_new)
