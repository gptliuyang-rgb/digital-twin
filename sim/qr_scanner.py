"""Decode a QR from a rendered image. Geometric gate + real decoder.

Success is decode, not 'close enough'. Decoder backends: OpenCV, then pyzbar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanSpec:
    d_min_m: float = 0.05
    d_max_m: float = 0.30
    theta_max_rad: float = np.deg2rad(45.0)
    v_max_m_s: float = 0.30


@dataclass
class ScanResult:
    payload: str | None
    distance_m: float
    incidence_rad: float
    speed_m_s: float
    geometry_ok: bool
    decode_ok: bool


def _incidence(gun_z: np.ndarray, qr_normal: np.ndarray) -> float:
    c = float(np.clip(np.dot(_unit(gun_z), _unit(-qr_normal)), -1.0, 1.0))
    return float(np.arccos(c))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def decode_image(rgb: np.ndarray) -> str | None:
    if rgb.ndim != 3:
        raise ValueError("rgb must be HxWxC")
    try:
        import cv2

        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(rgb)
        if data:
            return str(data)
    except Exception:
        pass
    try:
        from pyzbar import pyzbar

        gray = rgb if rgb.ndim == 2 else np.mean(rgb, axis=2).astype(np.uint8)
        codes = pyzbar.decode(gray)
        if codes:
            return codes[0].data.decode("utf-8")
    except Exception:
        pass
    return None


def simulate_scan(
    wrist_cam_rgb: np.ndarray,
    gun_tcp_pos_m: np.ndarray,
    gun_tcp_z: np.ndarray,
    qr_pos_m: np.ndarray,
    qr_normal: np.ndarray,
    rel_speed_m_s: float,
    spec: ScanSpec | None = None,
) -> ScanResult:
    spec = spec or ScanSpec()
    delta = np.asarray(qr_pos_m, dtype=np.float64) - np.asarray(gun_tcp_pos_m, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    theta = _incidence(np.asarray(gun_tcp_z, dtype=np.float64), np.asarray(qr_normal, dtype=np.float64))
    geom = spec.d_min_m <= dist <= spec.d_max_m and theta <= spec.theta_max_rad and rel_speed_m_s <= spec.v_max_m_s
    payload = decode_image(wrist_cam_rgb) if geom else None
    return ScanResult(
        payload=payload,
        distance_m=dist,
        incidence_rad=theta,
        speed_m_s=rel_speed_m_s,
        geometry_ok=geom,
        decode_ok=payload is not None,
    )


def make_qr_png(payload: str, box_size: int = 8, border: int = 4):
    import qrcode

    img = qrcode.make(payload, box_size=box_size, border=border)
    return img.convert("RGB")
