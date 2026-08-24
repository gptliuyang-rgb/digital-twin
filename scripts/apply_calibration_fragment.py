#!/usr/bin/env python3
"""Merge an E1/E2 fit fragment into a hand spec *copy* (contact block only).

Refuses to overwrite the live dexhand2_spec.yaml unless --commit-live is set.
Synthetic fragments additionally require --i-accept-synthetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hand.calibration.apply_fragment import guard_live_spec_write, merge_fragment
from interface.schema import HAND_SPEC_PATH, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=HAND_SPEC_PATH)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--commit-live",
        action="store_true",
        help="Allow overwriting assets/dexhand2/meta/dexhand2_spec.yaml",
    )
    parser.add_argument(
        "--i-accept-synthetic",
        action="store_true",
        help="Required together with --commit-live when the fragment is a synthetic dry-run",
    )
    args = parser.parse_args()
    base = load_yaml(args.base)
    fragment = load_yaml(args.fragment)
    if not isinstance(base, dict) or not isinstance(fragment, dict):
        raise SystemExit("base spec and fragment must be YAML mappings")
    guard_live_spec_write(
        args.out,
        fragment,
        commit_live=args.commit_live,
        accept_synthetic=args.i_accept_synthetic,
    )
    out_spec = merge_fragment(base, fragment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(out_spec, sort_keys=False), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
