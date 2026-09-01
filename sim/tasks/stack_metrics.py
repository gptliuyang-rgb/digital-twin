"""Stacking process metrics. No success-rate field (ADR-004)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StackSample:
    release_speed_m_s: float
    gap_z_m: float
    alignment_xy_m: float
    tilt_rad: float
    settled: bool | None = None


def box_speed_m_s(data, body_id: int) -> float:
    cvel = np.asarray(data.cvel[body_id], dtype=np.float64)
    # cvel: [rot; lin] in MuJoCo
    return float(np.linalg.norm(cvel[3:6]))


def tilt_from_xmat(xmat: np.ndarray) -> float:
    z = np.asarray(xmat, dtype=np.float64).reshape(3, 3)[:, 2]
    return float(np.arccos(np.clip(z[2], -1.0, 1.0)))


def stack_alignment_xy_m(box_xy: np.ndarray, support_xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(box_xy)[:2] - np.asarray(support_xy)[:2]))


def release_sample(
    *,
    speed_m_s: float,
    box_bottom_z_m: float,
    support_top_z_m: float,
    box_xy: np.ndarray,
    support_xy: np.ndarray,
    xmat: np.ndarray,
    speed_gate_m_s: float = 0.05,
    gap_gate_m: float = 0.005,
    tilt_gate_rad: float = np.deg2rad(3.0),
    align_gate_m: float = 0.03,
) -> StackSample:
    gap = float(box_bottom_z_m - support_top_z_m)
    align = stack_alignment_xy_m(box_xy, support_xy)
    tilt = tilt_from_xmat(xmat)
    settled = (
        speed_m_s < speed_gate_m_s
        and abs(gap) < gap_gate_m + 0.02
        and tilt < tilt_gate_rad
        and align < align_gate_m
    )
    return StackSample(
        release_speed_m_s=speed_m_s,
        gap_z_m=gap,
        alignment_xy_m=align,
        tilt_rad=tilt,
        settled=settled,
    )
