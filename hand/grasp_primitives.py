"""Named grasp primitives as closure∈[0,1] → q_active trajectories.

Open is official q=0. Closed poses are **fractions of joint_limits**, not measured
grasp postures. Replace with teleop-recorded q once data exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interface.schema import HandSpec, load_hand_spec


@dataclass(frozen=True)
class Primitive:
    name: str
    q_open: np.ndarray
    q_closed: np.ndarray

    def at(self, closure: float) -> np.ndarray:
        a = float(np.clip(closure, 0.0, 1.0))
        return (1.0 - a) * self.q_open + a * self.q_closed


class GraspLibrary:
    def __init__(self, spec: HandSpec | None = None) -> None:
        self.spec = spec or load_hand_spec()
        n = self.spec.n_active_dof
        zero = np.zeros(n)
        limits = self.spec.limits_vector()
        hi = limits[:, 1]
        names = {name: i for i, name in enumerate(self.spec.joint_order)}

        def flex_closed(scale: float) -> np.ndarray:
            q = zero.copy()
            for name, idx in names.items():
                if "abd" in name:
                    continue
                q[idx] = scale * hi[idx]
            return q

        power = flex_closed(0.85)
        pinch = zero.copy()
        for key in ("thumb_cmc_flex", "thumb_mcp", "thumb_ip", "index_finger_mcp_flex", "index_finger_pip", "index_finger_dip"):
            pinch[names[key]] = 0.55 * hi[names[key]]
        gun = flex_closed(0.4)
        # Index stays independent for the trigger (tool_trigger is a separate command bit).
        gun[names["index_finger_mcp_flex"]] = 0.15 * hi[names["index_finger_mcp_flex"]]
        gun[names["index_finger_pip"]] = 0.1 * hi[names["index_finger_pip"]]
        gun[names["index_finger_dip"]] = 0.1 * hi[names["index_finger_dip"]]
        flat = zero.copy()
        self._prims = {
            "open": Primitive("open", zero, zero),
            "power_grasp": Primitive("power_grasp", zero, power),
            "pinch": Primitive("pinch", zero, pinch),
            "gun_grip": Primitive("gun_grip", zero, gun),
            "flat_support": Primitive("flat_support", zero, flat),
        }

    def names(self) -> list[str]:
        return list(self._prims)

    def q_active(self, name: str, closure: float) -> np.ndarray:
        if name not in self._prims:
            raise KeyError(f"unknown primitive {name}; known={self.names()}")
        return self._prims[name].at(closure)
