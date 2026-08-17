"""SONIC Table S4 root-push extrema (arXiv:2511.07820v3).

These numbers randomize the *humanoid* during motion-tracking PPO. They are
not DexHand2 pad–cardboard coefficients and must not enter dexhand2_spec.yaml
(ADR-004 / ADR-014 / ADR-022 / ADR-027).

This module does not import MuJoCo or Isaac.
"""

from __future__ import annotations

from typing import Any

import yaml

from wbc.ppo.recipe import PPO_DIR

# Planar extrema used by the free-base diagnostic sweep. Table S4 z is ±0.2 m/s
# (vertical) and is a different impulse; it is recorded but not swept.
SWEEP_AXES = ("x", "y")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


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
