"""SONIC Table S4 root-push extrema (arXiv:2511.07820v3).

These numbers randomize the *humanoid* during motion-tracking PPO. They are
not DexHand2 pad–cardboard coefficients and must not enter dexhand2_spec.yaml
(ADR-004 / ADR-014 / ADR-022 / ADR-027 / ADR-028).

This module does not import MuJoCo or Isaac.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import yaml

from wbc.ppo.recipe import PPO_DIR

# Planar extrema used by the free-base diagnostic sweep. Table S4 z is ±0.2 m/s
# (vertical) and is a different impulse; it is recorded but not swept.
SWEEP_AXES = ("x", "y")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
# Table S4 "Push duration Δt ∼ [1, 3] s". Extrema only; do not invent a third T.
DURATION_EXTREMA_S = (1.0, 3.0)


def load_table_s4() -> dict[str, Any]:
    raw = yaml.safe_load((PPO_DIR / "domain_rand.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("domain_rand.yaml must be a mapping")
    if raw.get("not_dexhand2_contact") is not True:
        raise ValueError("domain_rand.yaml must keep not_dexhand2_contact: true")
    return raw


def root_push_ranges_mps() -> dict[str, tuple[float, float]]:
    """Return Table S4 root_push lin_vel ranges per axis, m/s."""
    push = load_table_s4()["root_push"]["lin_vel_mps"]
    out: dict[str, tuple[float, float]] = {}
    for axis in ("x", "y", "z"):
        lo, hi = (float(v) for v in push[axis])
        if lo >= 0.0 or hi <= 0.0:
            raise ValueError(f"Table S4 root_push {axis} must straddle zero, got {[lo, hi]}")
        out[axis] = (lo, hi)
    return out


def axis_aligned_linvel_extrema_mps(
    *,
    axes: tuple[str, ...] = SWEEP_AXES,
) -> list[dict[str, Any]]:
    """One-shot world linvel at each signed planar extremum.

    Default is ±X and ±Y at 0.5 m/s. Z (±0.2 m/s in Table S4) is excluded
    unless requested: a vertical impulse is not the same diagnostic as a
    planar root push.
    """
    ranges = root_push_ranges_mps()
    cases: list[dict[str, Any]] = []
    for axis in axes:
        if axis not in AXIS_INDEX:
            raise ValueError(f"unknown Table S4 axis {axis!r}")
        lo, hi = ranges[axis]
        idx = AXIS_INDEX[axis]
        for sign, value in (("-", lo), ("+", hi)):
            vec = [0.0, 0.0, 0.0]
            vec[idx] = float(value)
            cases.append(
                {
                    "name": f"{sign}{axis}",
                    "axis": axis,
                    "sign": sign,
                    "lin_vel_mps": vec,
                    "source": "He et al., SONIC, arXiv:2511.07820v3 Table S4 root_push",
                    "kind": "one_shot_qvel",
                    "not_sustained_force": True,
                    "not_dexhand2_contact": True,
                }
            )
    return cases


def root_push_duration_s() -> tuple[float, float]:
    """Return Table S4 root_push duration range, seconds."""
    raw = load_table_s4()["root_push"]["duration_s"]
    lo, hi = float(raw[0]), float(raw[1])
    if lo <= 0.0 or hi < lo:
        raise ValueError(f"Table S4 root_push duration_s must be positive lo<=hi, got {[lo, hi]}")
    return (lo, hi)


def duration_extrema_s() -> tuple[float, float]:
    """Inclusive duration extrema from domain_rand.yaml (1 s and 3 s)."""
    lo, hi = root_push_duration_s()
    if (lo, hi) != DURATION_EXTREMA_S:
        raise ValueError(
            f"domain_rand.yaml duration_s {(lo, hi)} drifted from Table S4 extrema {DURATION_EXTREMA_S}"
        )
    return (lo, hi)


def force_n_from_impulse(
    mass_kg: float,
    lin_vel_mps: Sequence[float],
    duration_s: float,
) -> list[float]:
    """Constant world force whose impulse equals ``mass * Δv`` over ``duration_s``.

    Table S4 publishes velocity and duration, not Newtons. One-shot ``qvel``
    applies Δv instantly. This spreads the same linear impulse over T:

        F = m * v / T

    Floor contact, gravity, and joint PD still act; this is not a free-space
    identity. ``mass_kg`` must come from the loaded MJCF subtree, not a guessed
    T800 datasheet number.
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if mass_kg <= 0.0:
        raise ValueError(f"mass_kg must be > 0, got {mass_kg}")
    if len(lin_vel_mps) != 3:
        raise ValueError(f"lin_vel_mps must be length 3, got {len(lin_vel_mps)}")
    return [float(mass_kg) * float(v) / float(duration_s) for v in lin_vel_mps]


def sustained_force_cases(
    *,
    axes: tuple[str, ...] = SWEEP_AXES,
    durations_s: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    """Planar linvel extrema × duration extrema. Force Newtons need mass at apply time."""
    if durations_s is None:
        durations_s = duration_extrema_s()
    vel_cases = axis_aligned_linvel_extrema_mps(axes=axes)
    out: list[dict[str, Any]] = []
    for vel in vel_cases:
        for duration_s in durations_s:
            t = float(duration_s)
            if t <= 0.0:
                raise ValueError(f"duration_s must be > 0, got {t}")
            out.append(
                {
                    "name": f"{vel['name']}_T{t}s",
                    "axis": vel["axis"],
                    "sign": vel["sign"],
                    "lin_vel_mps": list(vel["lin_vel_mps"]),
                    "duration_s": t,
                    "source": "He et al., SONIC, arXiv:2511.07820v3 Table S4 root_push",
                    "kind": "sustained_force",
                    "force_formula": "F = m * v / T",
                    "not_one_shot_qvel": True,
                    "not_dexhand2_contact": True,
                }
            )
    return out
