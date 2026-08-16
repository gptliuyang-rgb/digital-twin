"""L3 task tokens. Simulator-free. Same object drives sim eval and the real stack.

Subtask tokens match the digital-twin plan: pick_box / stack_to / scan_qr.
Transitions consume predicates the bridges supply (decode_ok, grasp_closed, …).
They never read MuJoCo or Isaac.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Subtask(str, Enum):
    PICK_BOX = "pick_box"
    STACK_TO = "stack_to"
    SCAN_QR = "scan_qr"


class Phase(str, Enum):
    IDLE = "idle"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    CARRY = "carry"
    PLACE = "place"
    SETTLE = "settle"
    APPROACH_QR = "approach_qr"
    IBVS_SCAN = "ibvs_scan"
    DONE = "done"
    FAIL = "fail"


_PICK = (Phase.APPROACH, Phase.GRASP, Phase.LIFT, Phase.DONE)
_STACK = (Phase.CARRY, Phase.PLACE, Phase.SETTLE, Phase.DONE)
_SCAN = (Phase.APPROACH_QR, Phase.IBVS_SCAN, Phase.DONE)


@dataclass
class TaskPredicates:
    at_pregrasp: bool = False
    grasp_closed: bool = False
    lifted: bool = False
    at_place: bool = False
    released: bool = False
    stack_stable: bool = False
    qr_in_envelope: bool = False
    decode_ok: bool = False
    fallen: bool = False
    timed_out: bool = False


@dataclass
class TaskFsm:
    subtask: Subtask
    phase: Phase = Phase.IDLE
    box_id: str | None = None
    stack_slot: str | None = None
    history: list[str] = field(default_factory=list)

    def start(self) -> Phase:
        first = {Subtask.PICK_BOX: Phase.APPROACH, Subtask.STACK_TO: Phase.CARRY, Subtask.SCAN_QR: Phase.APPROACH_QR}[
            self.subtask
        ]
        self.phase = first
        self.history.append(self.phase.value)
        return self.phase

    def step(self, pred: TaskPredicates) -> Phase:
        if self.phase in (Phase.DONE, Phase.FAIL):
            return self.phase
        if pred.fallen or pred.timed_out:
            return self._set(Phase.FAIL)
        if self.subtask == Subtask.PICK_BOX:
            return self._step_pick(pred)
        if self.subtask == Subtask.STACK_TO:
            return self._step_stack(pred)
        return self._step_scan(pred)

    def _set(self, phase: Phase) -> Phase:
        if phase != self.phase:
            self.phase = phase
            self.history.append(phase.value)
        return self.phase

    def _step_pick(self, pred: TaskPredicates) -> Phase:
        if self.phase == Phase.IDLE:
            return self.start()
        if self.phase == Phase.APPROACH and pred.at_pregrasp:
            return self._set(Phase.GRASP)
        if self.phase == Phase.GRASP and pred.grasp_closed:
            return self._set(Phase.LIFT)
        if self.phase == Phase.LIFT and pred.lifted:
            return self._set(Phase.DONE)
        return self.phase

    def _step_stack(self, pred: TaskPredicates) -> Phase:
        if self.phase == Phase.IDLE:
            return self.start()
        if self.phase == Phase.CARRY and pred.at_place:
            return self._set(Phase.PLACE)
        if self.phase == Phase.PLACE and pred.released:
            return self._set(Phase.SETTLE)
        if self.phase == Phase.SETTLE and pred.stack_stable:
            return self._set(Phase.DONE)
        return self.phase

    def _step_scan(self, pred: TaskPredicates) -> Phase:
        if self.phase == Phase.IDLE:
            return self.start()
        if self.phase == Phase.APPROACH_QR and pred.qr_in_envelope:
            return self._set(Phase.IBVS_SCAN)
        if self.phase == Phase.IBVS_SCAN and pred.decode_ok:
            return self._set(Phase.DONE)
        return self.phase
