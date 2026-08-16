"""L0: open-loop replay of a ckpt / dataset. No physics.

Hand dimensions are reported separately — a 10× error on one finger channel is
the usual signature of a joint-order swap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interface.schema import command_dim, command_layout, load_hand_spec


def per_dim_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean((pred - gt) ** 2, axis=0)


def flag_order_swap(mse: np.ndarray, hand_slice: slice) -> list[int]:
    hand = mse[hand_slice]
    med = np.median(hand) + 1e-12
    return [int(i) for i, v in enumerate(hand) if v > 10.0 * med]


def named_hand_outliers(mse: np.ndarray, spec, side: str) -> list[dict]:
    layout = command_layout(spec)
    sl = slice(*layout[f"{side}_hand_q"])
    idx = flag_order_swap(mse, sl)
    hand_mse = mse[sl]
    return [
        {"index": i, "joint": spec.joint_order[i], "mse": float(hand_mse[i])}
        for i in idx
    ]


def evaluate_episode(pred_chunks: np.ndarray, gt_chunks: np.ndarray, spec=None) -> dict:
    spec = spec or load_hand_spec()
    layout = command_layout(spec)
    pred = pred_chunks.reshape(-1, command_dim(spec))
    gt = gt_chunks.reshape(-1, command_dim(spec))
    mse = per_dim_mse(pred, gt)
    lh = slice(*layout["left_hand_q"])
    rh = slice(*layout["right_hand_q"])
    return {
        "mse": mse.tolist(),
        "mse_mean": float(mse.mean()),
        "left_hand_outliers": flag_order_swap(mse, lh),
        "right_hand_outliers": flag_order_swap(mse, rh),
        "left_hand_outlier_names": named_hand_outliers(mse, spec, "left"),
        "right_hand_outlier_names": named_hand_outliers(mse, spec, "right"),
        "chunk_step_mse": np.mean((pred_chunks - gt_chunks) ** 2, axis=(0, 2)).tolist()
        if pred_chunks.ndim == 3
        else None,
        "joint_order": spec.joint_order,
    }


def _synthetic_demo(spec) -> dict:
    dim = command_dim(spec)
    n = 32
    h = 16
    gt = np.zeros((n, h, dim))
    rng = np.random.default_rng(0)
    gt[..., :] = rng.normal(scale=0.05, size=gt.shape)
    pred = gt + rng.normal(scale=0.01, size=gt.shape)
    # Inject a swapped finger channel so the report can name it.
    lh0 = command_layout(spec)["left_hand_q"][0]
    pred[..., lh0 + 3] = gt[..., lh0 + 7]
    return evaluate_episode(pred, gt, spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval/configs/l0_offline.yaml")
    parser.add_argument("--pred", default="")
    parser.add_argument("--gt", default="")
    parser.add_argument("--out", default="eval/report/generated/l0.json")
    args = parser.parse_args()
    spec = load_hand_spec()
    if args.pred and args.gt:
        pred = np.load(args.pred)
        gt = np.load(args.gt)
        report = evaluate_episode(pred, gt, spec)
        report["mode"] = "dataset"
    else:
        report = _synthetic_demo(spec)
        report["mode"] = "synthetic_demo_no_ckpt"
        report["note"] = "No ckpt/dataset provided. Demo only; not a model evaluation."
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("mode", "mse_mean", "left_hand_outliers", "right_hand_outliers")}, indent=2))


if __name__ == "__main__":
    main()
