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


def write_synthetic_csvs(root: Path | None = None, *, seed: int = 0) -> dict[str, Path]:
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
                theta_s = math.degrees(math.atan(MU_S_TRUE)) + rng.gauss(0.0, 0.35)
                theta_d = math.degrees(math.atan(MU_D_TRUE)) + rng.gauss(0.0, 0.40)
                writer.writerow(
                    {
                        "trial": trial,
                        "finger": finger,
                        "skin": SKIN,
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
                k_trial = K_TRUE_N_PER_M * (1.0 + rng.gauss(0.0, 0.04))
                for raw in (0.00010, 0.00020, 0.00035, 0.00050, 0.00070, 0.00090, 0.00100, 0.00120):
                    # Nonlinear toe below 0.2 mm; fitter uses the 0.2–1.0 mm window.
                    scale = 0.35 if raw < 0.0002 else 1.0
                    force = scale * k_trial * raw + rng.gauss(0.0, 0.04)
                    writer.writerow(
                        {
                            "trial": f"{finger}_{trial}",
                            "finger": finger,
                            "skin": SKIN,
                            "disp_m": f"{raw:.5f}",
                            "force_n": f"{force:.5f}",
                            "k_n_per_m": "",
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
                "note: Pipeline fixture. Never copy into live dexhand2_spec.yaml as hardware.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"e1": e1, "e2": e2, "meta": meta}


def main() -> None:
    paths = write_synthetic_csvs()
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
