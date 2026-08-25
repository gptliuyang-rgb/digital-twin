"""Deterministic synthetic E1/E2 sheets for pipeline dry-run. Not hardware."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from interface.schema import REPO_ROOT

BATCH_ID = "synthetic_batch_v1.0"
FIRMWARE = "synthetic-fw"
SKIN = "on"

# Ground-truth values the fitter should recover (plus small noise).
MU_S_TRUE = 0.75
MU_D_TRUE = 0.55
K_TRUE_N_PER_M = 2500.0

SYNTHETIC_ROOT = REPO_ROOT / "hand" / "calibration" / "results" / BATCH_ID
SKIN_ON_DIR = SYNTHETIC_ROOT / "skin_on"

FRAGMENT_EXTRAS = {
    "fingertip_material": "silicone_synthetic_placeholder",
    "fingertip_geometry_radius_m": 0.008,
    "calibration_kind": "synthetic_dry_run",
    "do_not_treat_as_committed_hardware": True,
}

E1_FIELDS = [
    "trial",
    "finger",
    "skin",
    "normal_n",
    "theta_s_deg",
    "theta_d_deg",
    "mu_s",
    "mu_d",
    "batch",
    "fw",
    "notes",
]
E2_FIELDS = ["trial", "finger", "skin", "disp_m", "force_n", "k_n_per_m", "batch", "fw"]
E3_FIELDS = [
    "trial",
    "closure",
    "mass_kg",
    "held_s",
    "slipped",
    "max_current_a",
    "finger",
    "batch",
    "fw",
]
E3_MAX_HELD_KG = 10.0
E3_FIRST_SLIP_KG = 12.0


def write_synthetic_csvs(
    root: Path | None = None,
    *,
    seed: int = 0,
    skin: str = SKIN,
    mu_s: float = MU_S_TRUE,
    mu_d: float = MU_D_TRUE,
    k_n_per_m: float = K_TRUE_N_PER_M,
) -> dict[str, Path]:
    out_dir = (root or SKIN_ON_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    e1 = out_dir / "e1.csv"
    e2 = out_dir / "e2.csv"
    fingers = ["thumb", "index", "middle"]

    with e1.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=E1_FIELDS)
        writer.writeheader()
        for finger in fingers:
            for trial in range(1, 11):
                theta_s = math.degrees(math.atan(mu_s)) + rng.gauss(0.0, 0.35)
                theta_d = math.degrees(math.atan(mu_d)) + rng.gauss(0.0, 0.40)
                writer.writerow(
                    {
                        "trial": trial,
                        "finger": finger,
                        "skin": skin,
                        "normal_n": 5.0 if trial <= 5 else 10.0,
                        "theta_s_deg": f"{theta_s:.4f}",
                        "theta_d_deg": f"{theta_d:.4f}",
                        "mu_s": f"{math.tan(math.radians(theta_s)):.6f}",
                        "mu_d": f"{math.tan(math.radians(theta_d)):.6f}",
                        "batch": BATCH_ID,
                        "fw": FIRMWARE,
                        "notes": "synthetic_dry_run",
                    }
                )

    with e2.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=E2_FIELDS)
        writer.writeheader()
        for finger in fingers:
            for trial in range(1, 6):
                k_trial = k_n_per_m * (1.0 + rng.gauss(0.0, 0.04))
                for raw in (0.00010, 0.00020, 0.00035, 0.00050, 0.00070, 0.00090, 0.00100, 0.00120):
                    # Nonlinear toe below 0.2 mm; fitter uses the 0.2–1.0 mm window.
                    scale = 0.35 if raw < 0.0002 else 1.0
                    force = scale * k_trial * raw + rng.gauss(0.0, 0.04)
                    writer.writerow(
                        {
                            "trial": f"{finger}_{trial}",
                            "finger": finger,
                            "skin": skin,
                            "disp_m": f"{raw:.5f}",
                            "force_n": f"{force:.5f}",
                            "k_n_per_m": "",
                            "batch": BATCH_ID,
                            "fw": FIRMWARE,
                        }
                    )

    e3 = out_dir / "e3.csv"
    with e3.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=E3_FIELDS)
        writer.writeheader()
        trial = 0
        for mass in (2, 4, 6, 8, 10, 12, 14):
            for closure in (0.70, 0.85, 1.0):
                trial += 1
                slipped = mass >= E3_FIRST_SLIP_KG
                writer.writerow(
                    {
                        "trial": trial,
                        "closure": f"{closure:.2f}",
                        "mass_kg": mass,
                        "held_s": 5.0 if not slipped else 1.2,
                        "slipped": "1" if slipped else "0",
                        "max_current_a": f"{0.4 + 0.05 * mass:.3f}",
                        "finger": "power_grasp",
                        "batch": BATCH_ID,
                        "fw": FIRMWARE,
                    }
                )

    meta = (out_dir.parent / "meta.yaml")
    meta.write_text(
        "\n".join(
            [
                "kind: synthetic_dry_run",
                "do_not_treat_as_committed_hardware: true",
                f"batch: {BATCH_ID}",
                f"firmware_version: {FIRMWARE}",
                f"mu_s_true: {MU_S_TRUE}",
                f"mu_d_true: {MU_D_TRUE}",
                f"k_n_per_m_true: {K_TRUE_N_PER_M}",
                f"e3_max_held_kg_true: {E3_MAX_HELD_KG}",
                f"e3_first_slip_kg_true: {E3_FIRST_SLIP_KG}",
                "note: Pipeline fixture. Never copy into live dexhand2_spec.yaml as hardware.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"e1": e1, "e2": e2, "e3": e3, "meta": meta}


def main() -> None:
    on = write_synthetic_csvs(SKIN_ON_DIR, seed=0, skin="on")
    off = write_synthetic_csvs(
        SYNTHETIC_ROOT / "skin_off",
        seed=1,
        skin="off",
        mu_s=0.58,
        mu_d=0.42,
        k_n_per_m=3800.0,
    )
    for label, paths in (("skin_on", on), ("skin_off", off)):
        print(label)
        for key, path in paths.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
