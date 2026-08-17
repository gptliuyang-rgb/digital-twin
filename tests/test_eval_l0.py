from __future__ import annotations

import numpy as np

from eval.l0_offline_replay import evaluate_episode
from interface.schema import command_dim, command_layout, load_hand_spec


def test_l0_flags_swapped_finger_channel() -> None:
    spec = load_hand_spec()
    dim = command_dim(spec)
    n, h = 40, 8
    rng = np.random.default_rng(4)
    gt = rng.normal(scale=0.05, size=(n, h, dim))
    pred = gt + rng.normal(scale=0.002, size=gt.shape)
    lh0 = command_layout(spec)["left_hand_q"][0]
    pred[..., lh0 + 2] = gt[..., lh0 + 9] * 3.0
    report = evaluate_episode(pred, gt, spec)
    assert 2 in report["left_hand_outliers"]
    names = {item["joint"] for item in report["left_hand_outlier_names"]}
    assert spec.joint_order[2] in names
