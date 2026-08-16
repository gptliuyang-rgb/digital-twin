"""Fit μ and stiffness from E1/E2 CSV. Writes a spec fragment, does not silently patch the live spec."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import yaml


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def fit_e1(path: Path) -> dict:
    mu_s, mu_d = [], []
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
    return {
        "friction_vs_cardboard_static": _mean(mu_s),
        "friction_vs_cardboard_dynamic": _mean(mu_d),
        "n_static": len(mu_s),
        "n_dynamic": len(mu_d),
    }


def fit_e2(path: Path) -> dict:
    ks: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("k_n_per_m"):
                ks.append(float(row["k_n_per_m"]))
    return {"normal_stiffness_n_per_m": _mean(ks), "n": len(ks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", default="")
    parser.add_argument("--e2", default="")
    parser.add_argument("--out", default="hand/calibration/results/fragment.yaml")
    args = parser.parse_args()
    fragment: dict = {"source": "fit_params.py", "do_not_treat_as_committed_hardware": True}
    if args.e1:
        fragment.update(fit_e1(Path(args.e1)))
    if args.e2:
        fragment.update(fit_e2(Path(args.e2)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
