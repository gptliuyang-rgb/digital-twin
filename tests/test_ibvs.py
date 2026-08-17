from __future__ import annotations

import pytest

from runtime.ibvs import CameraIntrinsics, ibvs_camera_twist, interaction_matrix_point, pixel_to_normalized


def test_pixel_at_principal_point_is_zero() -> None:
    cam = CameraIntrinsics.synthetic_pinhole()
    x, y = pixel_to_normalized(cam.cx_px, cam.cy_px, cam)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)


def test_positive_u_error_produces_positive_vx() -> None:
    """Feature to the right of the principal point: camera translates +X (feature slides left)."""
    cam = CameraIntrinsics.synthetic_pinhole()
    result = ibvs_camera_twist(cam.cx_px + 40.0, cam.cy_px, z_m=0.15, cam=cam, gain=1.0)
    assert result.twist_cam[0] > 0
    assert abs(result.twist_cam[1]) < abs(result.twist_cam[0])


def test_interaction_matrix_shape() -> None:
    mat = interaction_matrix_point(0.1, -0.2, 0.2)
    assert mat.shape == (2, 6)
    # Depth column: x/Z, y/Z
    assert mat[0, 2] == pytest.approx(0.1 / 0.2)
    assert mat[1, 2] == pytest.approx(-0.2 / 0.2)


def test_zero_depth_rejected() -> None:
    with pytest.raises(ValueError):
        interaction_matrix_point(0.0, 0.0, 0.0)
