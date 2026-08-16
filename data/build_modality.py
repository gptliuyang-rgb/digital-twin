"""LeRobot-style modality.json generated from dexhand2_spec.joint_order."""

from __future__ import annotations

import json
from pathlib import Path

from interface.schema import REPO_ROOT, command_dim, command_layout, load_hand_spec


def build_modality(spec=None) -> dict:
    spec = spec or load_hand_spec()
    layout = command_layout(spec)
    action = {name: {"start": a, "end": b} for name, (a, b) in layout.items() if name != "_total"}
    return {
        "schema_version": "1.0",
        "embodiment": "engineai_t800_wuji_hand2",
        "n_active_dof": spec.n_active_dof,
        "joint_order": spec.joint_order,
        "action_dim": command_dim(spec),
        "action": action,
        "state": {
            "left_hand_q": {"dtype": "float32", "shape": [spec.n_active_dof], "unit": "rad"},
            "right_hand_q": {"dtype": "float32", "shape": [spec.n_active_dof], "unit": "rad"},
        },
        "video": ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
        "annotation": "generated from dexhand2_spec.yaml — do not edit by hand",
    }


def main() -> None:
    out = REPO_ROOT / "data" / "modality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_modality(), indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
