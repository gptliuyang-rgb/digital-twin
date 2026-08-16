"""Replay E1/E2 in MuJoCo once contact params exist. Refuses if spec is uncalibrated."""

from __future__ import annotations

from interface.schema import REQUIRED_INPUT_TOKEN, load_hand_spec


def main() -> None:
    spec = load_hand_spec()
    if spec.raw.get("friction_vs_cardboard_static") == REQUIRED_INPUT_TOKEN:
        raise SystemExit(
            "validate_sim: friction_vs_cardboard_static is REQUIRED_INPUT. "
            "Run E1 and fit_params.py before claiming a sim match."
        )
    raise SystemExit("validate_sim: MuJoCo replay not executed in this environment (no calibrated unit).")


if __name__ == "__main__":
    main()
