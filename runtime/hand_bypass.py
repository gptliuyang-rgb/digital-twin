"""Split command_schema hand_slice onto the DexHand2 1 kHz MIT ring.

Simulation-package-free. Same object on robot and in sim. Does **not**
import ``wbc/``. T800 last_action / policy_action / decoder 874-D stay
on the WBC path. Hands are L1c.

``left_hand_mode`` / ``right_hand_mode`` 0 (position) uses the 20-D q
vector. Mode 2 (force) is refused until ``motor_max_torque_nm`` is
filled. ``tool_trigger`` is metadata, not a finger joint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand.mit_ring import (
    HandMitPhysics,
    HandMitPlant,
    MitRingError,
    MitRingPeriod,
    refuse_force_mode_without_hardware_tau,
    refuse_hand_q_onto_wbc_last_action,
    run_mit_period,
)
from interface.schema import CommandVector, HandSpec, load_hand_spec


def hand_targets_from_command(cmd: CommandVector) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of left_hand_q, right_hand_q. Does not touch WBC fields."""
    if int(cmd.left_hand_mode) == 2 or int(cmd.right_hand_mode) == 2:
        refuse_force_mode_without_hardware_tau()
    return cmd.left_hand_q.copy(), cmd.right_hand_q.copy()


@dataclass(frozen=True)
class BimanualMitPeriod:
    left: MitRingPeriod
    right: MitRingPeriod
    tool_trigger: int

    @property
    def n_steps(self) -> int:
        return int(self.left.n_steps)


def run_bimanual_period(
    cmd: CommandVector,
    left_physics: HandMitPhysics,
    right_physics: HandMitPhysics,
    *,
    left_plant: HandMitPlant | None = None,
    right_plant: HandMitPlant | None = None,
    spec: HandSpec | None = None,
    t0_s: float = 0.0,
) -> BimanualMitPeriod:
    """ZOH both hands for one 50 Hz tick. Never writes WBC last_action."""
    spec = spec or cmd.spec or load_hand_spec()
    left_q, right_q = hand_targets_from_command(cmd)
    left_plant = left_plant or HandMitPlant.from_spec(spec)
    right_plant = right_plant or HandMitPlant.from_spec(spec)
    left = run_mit_period(left_plant, left_physics, left_q, t0_s=t0_s, side="left")
    right = run_mit_period(right_plant, right_physics, right_q, t0_s=t0_s, side="right")
    return BimanualMitPeriod(left=left, right=right, tool_trigger=int(cmd.tool_trigger))


def refuse_bypass_into_wbc() -> None:
    """Re-export the WBC-concat refusal next to the command splitter."""
    refuse_hand_q_onto_wbc_last_action()


def require_wbc_action_untouched(last_action: np.ndarray) -> np.ndarray:
    """T800 last_action must stay 25-D after a hand-ring tick."""
    arr = np.asarray(last_action, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 25:
        raise MitRingError(
            f"WBC last_action dim {arr.shape[0]} != T800 25 after a hand-ring tick. "
            "DexHand2 q must not be concatenated."
        )
    return arr
