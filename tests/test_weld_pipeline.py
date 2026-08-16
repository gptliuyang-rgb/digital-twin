from __future__ import annotations

import numpy as np
import pytest

from assets.combined.assemble import PolicyEvalBlocked, assert_policy_eval_allowed, mount_ready, weld_recipe
from eval.gain_scan import gain_cells
from eval.l0_offline_replay import evaluate_episode, named_hand_outliers
from interface.schema import CommandVector, SpecIncompleteError, command_dim, command_layout, load_hand_spec
from sim.hand_mass import mass_budget, require_sonic_mass
from vla.client.pipeline import DeployPipeline


def test_weld_recipe_blocks_eval() -> None:
    recipe = weld_recipe()
    assert recipe["bringup_identity_forbidden_for_eval"] is True
    assert recipe["eval_allowed"] is False
    assert mount_ready() is False
    with pytest.raises(PolicyEvalBlocked, match="CAD-measured"):
        assert_policy_eval_allowed("l2_combined")


def test_mass_budget_refuses_sonic() -> None:
    spec = load_hand_spec()
    budget = mass_budget(spec)
    assert budget.product_kg == pytest.approx(0.745)
    assert budget.two_hands_product_kg == pytest.approx(1.49)
    assert budget.sonic_ready is False
    with pytest.raises(SpecIncompleteError):
        require_sonic_mass(spec)


def test_gain_scan_has_nine_cells() -> None:
    cells = gain_cells()
    assert len(cells) == 9
    scales = {(c["kp_scale"], c["kv_scale"]) for c in cells}
    assert (1.0, 1.0) in scales
    assert (0.5, 2.0) in scales


def test_pipeline_roundtrip() -> None:
    spec = load_hand_spec()
    horizon = 16
    base = CommandVector.zeros(spec).to_flat_vector()

    def infer(_obs):
        chunk = np.tile(base, (horizon, 1))
        chunk[:, command_layout(spec)["left_hand_q"][0]] = 0.05
        return chunk

    pipe = DeployPipeline(infer, spec, chunk_horizon=horizon, dt_chunk_s=0.1, latency_s=0.2)
    cmd = pipe.step_chunk({})
    assert cmd.left_hand_q[0] == pytest.approx(0.05)


def test_l0_names_swapped_joint() -> None:
    spec = load_hand_spec()
    dim = command_dim(spec)
    n, h = 8, 4
    gt = np.zeros((n, h, dim))
    pred = gt.copy()
    lh0 = command_layout(spec)["left_hand_q"][0]
    pred[..., lh0 + 3] = 1.0
    report = evaluate_episode(pred, gt, spec)
    assert 3 in report["left_hand_outliers"]
    names = named_hand_outliers(np.asarray(report["mse"]), spec, "left")
    assert any(item["joint"] == spec.joint_order[3] for item in names)
