"""Task FSM and in-process policy server."""

from __future__ import annotations

import numpy as np
import pytest

from interface.schema import CommandVector
from runtime.task_fsm import Phase, Subtask, TaskFsm, TaskPredicates
from vla.server.policy_server import PolicyServer
from wbc.checkpoint import G1CheckpointIncompatible


def test_pick_box_happy_path() -> None:
    fsm = TaskFsm(Subtask.PICK_BOX)
    assert fsm.step(TaskPredicates()) == Phase.APPROACH
    assert fsm.step(TaskPredicates(at_pregrasp=True)) == Phase.GRASP
    assert fsm.step(TaskPredicates(grasp_closed=True)) == Phase.LIFT
    assert fsm.step(TaskPredicates(lifted=True)) == Phase.DONE
    assert fsm.history == ["approach", "grasp", "lift", "done"]


def test_scan_requires_decode_not_distance() -> None:
    fsm = TaskFsm(Subtask.SCAN_QR)
    fsm.start()
    assert fsm.step(TaskPredicates(qr_in_envelope=True)) == Phase.IBVS_SCAN
    assert fsm.step(TaskPredicates(qr_in_envelope=True, decode_ok=False)) == Phase.IBVS_SCAN
    assert fsm.step(TaskPredicates(decode_ok=True)) == Phase.DONE


def test_fall_fails() -> None:
    fsm = TaskFsm(Subtask.STACK_TO)
    fsm.start()
    assert fsm.step(TaskPredicates(fallen=True)) == Phase.FAIL


def test_replay_server_roundtrip() -> None:
    cmd = CommandVector.zeros()

    def infer(_obs: dict) -> np.ndarray:
        return cmd.to_flat_vector()

    server = PolicyServer("replay", infer_fn=infer)
    out = server.infer({})
    np.testing.assert_allclose(out.to_flat_vector(), cmd.to_flat_vector())


def test_server_refuses_g1_meta() -> None:
    server = PolicyServer("replay", infer_fn=lambda _o: CommandVector.zeros().to_flat_vector())
    with pytest.raises(G1CheckpointIncompatible):
        server.infer({"wbc_checkpoint": {"robot": "g1", "n_dof": 29}})


def test_groot_without_infer_fn_is_explicit() -> None:
    with pytest.raises(NotImplementedError, match="P1"):
        PolicyServer("groot")
