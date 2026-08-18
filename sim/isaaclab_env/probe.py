"""Isaac Lab / omni.usd availability. Safe without Isaac Sim.

PrivilegedIsaacEnv.reset/step still require Isaac Lab. This probe only reports
whether that path can bind. grasp_success_rate stays JSON null.
"""

from __future__ import annotations

from typing import Any


def isaac_runtime_status() -> dict[str, Any]:
    isaaclab_ok = False
    omni_usd_ok = False
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        isaaclab_ok = False
    else:
        isaaclab_ok = True
    try:
        import omni.usd  # noqa: F401
    except ImportError:
        omni_usd_ok = False
    else:
        omni_usd_ok = True
    bound = isaaclab_ok and omni_usd_ok
    return {
        "isaaclab": isaaclab_ok,
        "omni_usd": omni_usd_ok,
        "runtime": "bound" if bound else "unavailable",
        "grasp_success_rate": None,
        "combined_robot": "PolicyEvalBlocked",
        "note": (
            "PrivilegedIsaacEnv.reset/step bind IsaacLabSceneRuntime only when "
            "both isaaclab and omni.usd import. CI without Isaac Sim reports unavailable."
        ),
    }
