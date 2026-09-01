"""Fit μ and stiffness from E1/E2 CSV. Writes a spec fragment, does not silently patch the live spec.

Also exposes ``fit_fingertip_radius`` to read the geometry radius from official *_tip.STL files
so that ``fingertip_geometry_radius_m`` is filled from real mesh data, not guessed.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import yaml

from hand.calibration.contact_mujoco import (
    DEFAULT_PAD_M_EFF_KG,
    solref_timeconst_from_stiffness,
)


def fit_fingertip_radius(mesh_dir: Path | None = None) -> float | None:
    """Return mean pad-sphere radius (m) across all *_tip.STL in mesh_dir, or None if missing."""
    try:
        from assets.dexhand2.build.gen_derived import fit_pad_spheres, read_stl_vertices
        from assets.dexhand2.build.ingest_official import DEFAULT_UPSTREAM
    except ImportError:
        return None

    root = mesh_dir or DEFAULT_UPSTREAM / "hand2/hand2_beta1/body/meshes/right"
    stls = sorted(root.glob("*_tip.STL"))
    if not stls:
        return None
    radii = []
    for stl in stls:
        try:
            spheres = fit_pad_spheres(read_stl_vertices(stl), n_spheres=1)
            radii.append(spheres[0][1])
        except Exception:  # noqa: BLE001
            continue
    return float(sum(radii) / len(radii)) if radii else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def fit_e1(path: Path) -> dict:
    mu_s, mu_d = [], []
    batch = fw = skin = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("mu_s"):
                mu_s.append(float(row["mu_s"]))
            elif row.get("theta_s_deg"):
                mu_s.append(math.tan(math.radians(float(row["theta_s_deg"]))))
            if row.get("mu_d"):
                mu_d.append(float(row["mu_d"]))
            elif row.get("theta_d_deg"):
                mu_d.append(math.tan(math.radians(float(row["theta_d_deg"]))))
            batch = batch or row.get("batch")
            fw = fw or row.get("fw")
            skin = skin or row.get("skin")
    out = {
        "friction_vs_cardboard_static": _mean(mu_s),
        "friction_vs_cardboard_dynamic": _mean(mu_d),
        "n_static": len(mu_s),
        "n_dynamic": len(mu_d),
    }
    if batch:
        out["soft_body_batch_id"] = batch
        out["skin_batch_id"] = batch
    if fw:
        out["firmware_version"] = fw
    if skin:
        out["skin_present"] = skin.lower() in {"1", "true", "yes", "on", "skin_on"}
    return out


def _fit_k_from_trial(rows: list[dict], *, disp_lo_m: float = 0.0002, disp_hi_m: float = 0.001) -> float | None:
    xs, ys = [], []
    for row in rows:
        disp = float(row["disp_m"])
        force = float(row["force_n"])
        if disp_lo_m <= disp <= disp_hi_m:
            xs.append(disp)
            ys.append(force)
    if len(xs) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den <= 0:
        return None
    k = num / den
    return k if k > 0 else None


def fit_e3(path: Path, *, hold_s: float = 5.0) -> dict:
    """Largest mass held without slip for ``hold_s`` seconds.

    Does **not** invent ``motor_max_torque_nm``. Writes an observed mass only.
    """
    held: list[float] = []
    slipped: list[float] = []
    batch = fw = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mass = float(row["mass_kg"])
            held_t = float(row.get("held_s") or 0.0)
            flag = str(row.get("slipped", "")).strip().lower()
            did_slip = flag in {"1", "true", "yes", "slip"}
            batch = batch or row.get("batch")
            fw = fw or row.get("fw")
            if (not did_slip) and held_t + 1e-9 >= hold_s:
                held.append(mass)
            if did_slip:
                slipped.append(mass)
    out = {
        "e3_max_held_mass_kg": max(held) if held else float("nan"),
        "e3_first_slip_mass_kg": min(slipped) if slipped else float("nan"),
        "n_hold": len(held),
        "n_slip": len(slipped),
        "hold_s": hold_s,
        "do_not_treat_as_payload_rating": True,
    }
    if batch:
        out["soft_body_batch_id"] = batch
    if fw:
        out["firmware_version"] = fw
    return out


def fit_e2(path: Path, *, m_eff_kg: float = DEFAULT_PAD_M_EFF_KG) -> dict:
    ks: list[float] = []
    by_trial: dict[str, list[dict]] = defaultdict(list)
    batch = fw = skin = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("k_n_per_m"):
                ks.append(float(row["k_n_per_m"]))
            elif row.get("disp_m") and row.get("force_n"):
                trial = row.get("trial") or str(len(by_trial))
                by_trial[trial].append(row)
            batch = batch or row.get("batch")
            fw = fw or row.get("fw")
            skin = skin or row.get("skin")
    for trial_rows in by_trial.values():
        k = _fit_k_from_trial(trial_rows)
        if k is not None:
            ks.append(k)
    k_mean = _mean(ks)
    solref = solref_timeconst_from_stiffness(k_mean, m_eff_kg=m_eff_kg) if ks else float("nan")
    out = {
        "normal_stiffness_n_per_m": k_mean,
        "solref_timeconst_s": solref,
        "m_eff_kg": m_eff_kg,
        "n": len(ks),
        "n_trials_fitted": len(by_trial),
    }
    if batch:
        out["soft_body_batch_id"] = batch
        out["skin_batch_id"] = batch
    if fw:
        out["firmware_version"] = fw
    if skin:
        out["skin_present"] = skin.lower() in {"1", "true", "yes", "on", "skin_on"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", default="")
    parser.add_argument("--e2", default="")
    parser.add_argument("--e3", default="")
    parser.add_argument("--out", default="hand/calibration/results/fragment.yaml")
    parser.add_argument("--m-eff-kg", type=float, default=DEFAULT_PAD_M_EFF_KG)
    parser.add_argument(
        "--fit-tip-radius",
        action="store_true",
        help="Fit fingertip_geometry_radius_m from official *_tip.STL files",
    )
    args = parser.parse_args()
    fragment: dict = {
        "source": "fit_params.py",
        "do_not_treat_as_committed_hardware": True,
        "human_must_accept_solref": True,
    }
    if args.e1:
        fragment.update(fit_e1(Path(args.e1)))
    if args.e2:
        fragment.update(fit_e2(Path(args.e2), m_eff_kg=args.m_eff_kg))
        fragment["human_must_accept_solref"] = True
    if args.e3:
        fragment.update(fit_e3(Path(args.e3)))
    if args.fit_tip_radius:
        r = fit_fingertip_radius()
        if r is not None:
            fragment["fingertip_geometry_radius_m"] = round(r, 6)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
