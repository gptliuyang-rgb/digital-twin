"""Turn L0–L2 JSON into a Markdown report. No marketing language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interface.schema import REPO_ROOT, load_hand_spec

DEFAULT_DIR = REPO_ROOT / "eval" / "report" / "generated"


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render(l0: dict | None, l1: dict | None, l2: dict | None) -> str:
    spec = load_hand_spec()
    lines = [
        "# Eval report",
        "",
        f"joint_order: `{spec.joint_order}`",
        "",
        "## L0 offline replay",
        "",
    ]
    if l0 is None:
        lines.append("_not run_")
    else:
        lines.append(f"- mode: `{l0.get('mode')}`")
        lines.append(f"- mse_mean: `{l0.get('mse_mean')}`")
        lines.append(f"- left_hand_outliers: `{l0.get('left_hand_outliers')}`")
        lines.append(f"- right_hand_outliers: `{l0.get('right_hand_outliers')}`")
        named = l0.get("left_hand_outlier_names") or []
        if named:
            lines.append("- named left-hand outliers:")
            for item in named:
                lines.append(f"  - {item}")
        if l0.get("note"):
            lines.append(f"- note: {l0['note']}")
    lines += ["", "## L1 kinematic", ""]
    if l1 is None:
        lines.append("_not run_")
    else:
        lines.append(f"- limits: `{l1.get('limits')}`")
        lines.append(f"- coupling: `{l1.get('coupling')}`")
        lines.append(f"- ik: `{l1.get('ik')}`")
        lines.append(f"- self_collision: `{l1.get('self_collision')}`")
    lines += ["", "## L2 MuJoCo closed loop", ""]
    if l2 is None:
        lines.append("_not run_")
    else:
        lines.append(f"- status: **{l2.get('status')}**")
        lines.append(f"- uncalibrated: `{l2.get('uncalibrated')}`")
        lines.append(f"- combined_eval_allowed: `{l2.get('combined_eval_allowed')}`")
        if l2.get("warning"):
            lines.append(f"- warning: {l2['warning']}")
        if l2.get("grasp_success_rate") is not None:
            lines.append("- ERROR: numeric grasp_success_rate is forbidden until E1/E2")
        else:
            lines.append("- grasp_success_rate: null (blocked until contact calibration)")
        cells = l2.get("gain_scan_cells") or []
        if cells:
            lines.append(f"- gain scan cells: {len(cells)} (do not collapse to one success number)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()
    d = Path(args.dir)
    md = render(_load(d / "l0.json"), _load(d / "l1.json"), _load(d / "l2.json"))
    out = d / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
