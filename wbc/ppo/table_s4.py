"""SONIC Table S4 domain-rand extrema (arXiv:2511.07820v3).

These numbers randomize the *humanoid* / floor during motion-tracking PPO. They
are not DexHand2 pad–cardboard coefficients and must not enter
dexhand2_spec.yaml (ADR-004 / ADR-014 / ADR-022 / ADR-027 / ADR-028 / ADR-029 /
ADR-030 / ADR-031).

This module does not import MuJoCo or Isaac.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import yaml

from wbc.ppo.recipe import PPO_DIR

# Planar extrema used by the free-base diagnostic sweep (ADR-027). Table S4 z
# is ±0.2 m/s — a different impulse from planar ±0.5 m/s — and is swept on
# its own axis list (ADR-030), never mixed into SWEEP_AXES.
SWEEP_AXES = ("x", "y")
VERTICAL_AXES = ("z",)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
# Table S4 root_push ang_vel: roll/pitch ±0.52 rad/s, yaw ±0.78 rad/s.
# Indices match MuJoCo freejoint qvel[3:6] (LINK_BASE body frame).
ANGVEL_AXES = ("roll", "pitch", "yaw")
ANGVEL_AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}
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
    """One-shot world linvel at each signed extremum.

    Default is ±X and ±Y at 0.5 m/s (ADR-027). Z (±0.2 m/s) is a different
    impulse and lives on ``VERTICAL_AXES`` (ADR-030) — pass ``axes=("z",)``
    or call ``vertical_linvel_extrema_mps()``. Do not mix z into SWEEP_AXES.
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


def vertical_linvel_extrema_mps() -> list[dict[str, Any]]:
    """One-shot world linvel at Table S4 ±Z 0.2 m/s. Not the planar 0.5 m/s."""
    return axis_aligned_linvel_extrema_mps(axes=VERTICAL_AXES)


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


def root_push_angvel_ranges_rad_s() -> dict[str, tuple[float, float]]:
    """Return Table S4 root_push ang_vel ranges per axis, rad/s."""
    push = load_table_s4()["root_push"]["ang_vel_rad_s"]
    out: dict[str, tuple[float, float]] = {}
    for axis in ANGVEL_AXES:
        lo, hi = (float(v) for v in push[axis])
        if lo >= 0.0 or hi <= 0.0:
            raise ValueError(f"Table S4 root_push ang_vel {axis} must straddle zero, got {[lo, hi]}")
        out[axis] = (lo, hi)
    return out


def axis_aligned_angvel_extrema_rad_s(
    *,
    axes: tuple[str, ...] = ANGVEL_AXES,
) -> list[dict[str, Any]]:
    """One-shot freejoint body-frame angvel at each signed Table S4 extremum.

    Default is ±roll / ±pitch at 0.52 rad/s and ±yaw at 0.78 rad/s. These are
    the paper's root_push ang_vel bounds, not pad–cardboard coefficients.
    """
    ranges = root_push_angvel_ranges_rad_s()
    cases: list[dict[str, Any]] = []
    for axis in axes:
        if axis not in ANGVEL_AXIS_INDEX:
            raise ValueError(f"unknown Table S4 angvel axis {axis!r}")
        lo, hi = ranges[axis]
        idx = ANGVEL_AXIS_INDEX[axis]
        for sign, value in (("-", lo), ("+", hi)):
            vec = [0.0, 0.0, 0.0]
            vec[idx] = float(value)
            cases.append(
                {
                    "name": f"{sign}{axis}",
                    "axis": axis,
                    "sign": sign,
                    "ang_vel_rad_s": vec,
                    "source": "He et al., SONIC, arXiv:2511.07820v3 Table S4 root_push",
                    "kind": "one_shot_qvel",
                    "not_sustained_torque": True,
                    "not_dexhand2_contact": True,
                }
            )
    return cases


def _matvec3(matrix: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"inertia matrix must be 3x3, got {[len(row) for row in matrix]}")
    if len(vec) != 3:
        raise ValueError(f"ang_vel_rad_s must be length 3, got {len(vec)}")
    return [
        float(matrix[i][0]) * float(vec[0])
        + float(matrix[i][1]) * float(vec[1])
        + float(matrix[i][2]) * float(vec[2])
        for i in range(3)
    ]


def _as_inertia_matrix(inertia_kgm2: Sequence[float] | Sequence[Sequence[float]]) -> list[list[float]]:
    """Accept a 3-vector (principal moments) or a 3×3 matrix."""
    rows = list(inertia_kgm2)
    if len(rows) != 3:
        raise ValueError(f"inertia_kgm2 must be length 3 or 3x3, got {len(rows)}")
    first = rows[0]
    try:
        iter(first)  # type: ignore[arg-type]
        matrix = True
    except TypeError:
        matrix = False
    if not matrix:
        diag = [float(x) for x in rows]  # type: ignore[arg-type]
        if any(v <= 0.0 for v in diag):
            raise ValueError(f"principal inertia must be > 0, got {diag}")
        return [
            [diag[0], 0.0, 0.0],
            [0.0, diag[1], 0.0],
            [0.0, 0.0, diag[2]],
        ]
    return [[float(x) for x in row] for row in rows]  # type: ignore[union-attr]


def torque_nm_from_impulse(
    inertia_kgm2: Sequence[float] | Sequence[Sequence[float]],
    ang_vel_rad_s: Sequence[float],
    duration_s: float,
) -> list[float]:
    """Constant body-frame torque whose angular impulse equals ``I @ ω`` over T.

    Table S4 publishes angular velocity and duration, not Newton-metres.
    One-shot ``qvel[3:6]`` applies Δω instantly. This spreads the same
    angular impulse over T:

        τ = I ω / T

    ``inertia_kgm2`` must come from the loaded MJCF freejoint angular mass
    block (``mj_fullM`` at the inject pose), not a guessed T800 datasheet
    number. Floor contact, gravity, and joint PD still act.
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if len(ang_vel_rad_s) != 3:
        raise ValueError(f"ang_vel_rad_s must be length 3, got {len(ang_vel_rad_s)}")
    inertia = _as_inertia_matrix(inertia_kgm2)
    ang_mom = _matvec3(inertia, ang_vel_rad_s)
    return [float(h) / float(duration_s) for h in ang_mom]


def sustained_torque_cases(
    *,
    axes: tuple[str, ...] = ANGVEL_AXES,
    durations_s: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    """Angvel extrema × duration extrema. Torque N·m needs inertia at apply time."""
    if durations_s is None:
        durations_s = duration_extrema_s()
    vel_cases = axis_aligned_angvel_extrema_rad_s(axes=axes)
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
                    "ang_vel_rad_s": list(vel["ang_vel_rad_s"]),
                    "duration_s": t,
                    "source": "He et al., SONIC, arXiv:2511.07820v3 Table S4 root_push",
                    "kind": "sustained_torque",
                    "torque_formula": "tau = I @ omega / T",
                    "not_one_shot_qvel": True,
                    "not_dexhand2_contact": True,
                }
            )
    return out


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
    """Linvel extrema × duration extrema. Default axes are planar (ADR-028).

    Pass ``axes=VERTICAL_AXES`` (or call ``vertical_force_cases``) for the
    Table S4 ±Z 0.2 m/s duration sweep (ADR-030). Force Newtons need mass
    at apply time.
    """
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


def vertical_force_cases(
    *,
    durations_s: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    """±Z 0.2 m/s × duration extrema as F = m v / T. Not planar 0.5 m/s."""
    return sustained_force_cases(axes=VERTICAL_AXES, durations_s=durations_s)


def _range2(raw: Any, *, field: str) -> tuple[float, float]:
    lo, hi = float(raw[0]), float(raw[1])
    if lo < 0.0 or hi < lo:
        raise ValueError(f"Table S4 {field} must be lo>=0 and lo<=hi, got {[lo, hi]}")
    return (lo, hi)


def physical_static_friction_range() -> tuple[float, float]:
    """Return Table S4 physical.static_friction, dimensionless.

    Isaac Lab / PhysX has a separate dynamic coefficient. MuJoCo does not.
    This range is the paper number, not a pad–cardboard measurement, and must
    not be copied into ``dexhand2_spec.yaml``.
    """
    return _range2(load_table_s4()["physical"]["static_friction"], field="physical.static_friction")


def physical_dynamic_friction_range() -> tuple[float, float]:
    """Recorded Table S4 physical.dynamic_friction. Not mapped onto MuJoCo."""
    return _range2(load_table_s4()["physical"]["dynamic_friction"], field="physical.dynamic_friction")


def physical_restitution_range() -> tuple[float, float]:
    """Recorded Table S4 physical.restitution. Not mapped onto MuJoCo solref."""
    return _range2(load_table_s4()["physical"]["restitution"], field="physical.restitution")


def _mu_case_name(mu_slide: float) -> str:
    return f"mu_s_{mu_slide:g}"


def static_friction_extrema() -> list[dict[str, Any]]:
    """Inclusive extrema of Table S4 physical.static_friction (0.3 and 1.6).

    Applied in the free-base diagnostic as MuJoCo ``geom_friction[0]`` (sliding)
    on the eval floor *and* foot collision geoms (ADR-031). MuJoCo contacts use
    the element-wise max of the two geoms, so both sides must be set. Spin and
    roll channels stay the official collision default. Dynamic friction and
    restitution stay recorded-only — mapping them would invent a PhysX→MuJoCo
    conversion. Do not mix these cases into ``push_sweep``.
    """
    lo, hi = physical_static_friction_range()
    if lo <= 0.0:
        raise ValueError(f"Table S4 static_friction lo must be > 0, got {lo}")
    out: list[dict[str, Any]] = []
    for mu_slide in (lo, hi):
        value = float(mu_slide)
        out.append(
            {
                "name": _mu_case_name(value),
                "mu_slide": value,
                "table_s4_field": "physical.static_friction",
                "kind": "wbc_floor_slide_friction",
                "mujoco_channel": "geom_friction[0]",
                "source": "He et al., SONIC, arXiv:2511.07820v3 Table S4 physical.static_friction",
                "dynamic_friction_not_mapped": True,
                "restitution_not_mapped": True,
                "not_dexhand2_contact": True,
                "not_pad_cardboard": True,
            }
        )
    return out
