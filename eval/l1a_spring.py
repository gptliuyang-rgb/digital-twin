"""Dump SONIC Eq. 8 keyframes. grasp_success_rate stays JSON null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT
from wbc.planner import KinematicPlanner
from wbc.spring import RootSpringState, load_spring_cfg, spring_root_keyframe

OUT = Path("eval/report/generated/l1a_spring.json")


def _state(*, vx: float = 0.0) -> RootSpringState:
    return RootSpringState(
        pos_xy_m=np.zeros(2),
        heading_rad=0.0,
        vel_xy_mps=np.array([vx, 0.0], dtype=np.float64),
        yaw_rate_rad_s=0.0,
    )


def main() -> None:
    load_spring_cfg()
    rest = spring_root_keyframe(_state(), np.zeros(3), t_s=1.0)
    reverse = spring_root_keyframe(_state(vx=6.0), np.array([-6.0, 0.0, 0.0]), t_s=1.0)
    planner = KinematicPlanner(horizon_s=1.6)
    traj = planner.plan_nav_root(_state(vx=6.0), np.array([-6.0, 0.0, 0.0]))
    payload = {
        "source": "He et al., SONIC, arXiv:2511.07820v3 §3.3 Equation 8",
        "adr": "ADR-038",
        "rest_keyframe_xy_m": rest.pos_xy_m.tolist(),
        "reverse_6_to_minus6_xy_m": reverse.pos_xy_m.tolist(),
        "reverse_smoothed": bool(reverse.pos_xy_m[0] > -6.0 + 1.0),
        "planner_rate_hz": traj.rate_hz,
        "planner_n_steps": traj.n_steps,
        "grasp_success_rate": None,
        "not_dexhand2_contact": True,
        "not_table_s4": True,
        "combined_robot": "PolicyEvalBlocked",
        "repo": str(REPO_ROOT),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
