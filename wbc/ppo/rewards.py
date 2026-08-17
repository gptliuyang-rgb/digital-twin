"""SONIC Table S3 tracking kernels. Numpy only — no simulator.

r = exp(-mean_b ||e_b||^2 / scale^2) for body-averaged terms.
Identity (pred == goal) yields 1.0 on every tracking term.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml

from wbc.ppo.recipe import PPO_DIR


def load_reward_cfg() -> dict[str, Any]:
    raw = yaml.safe_load((PPO_DIR / "rewards.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("rewards.yaml must be a mapping")
    return raw


def exp_kernel(mean_sq_err: float, scale: float) -> float:
    """exp(-||e||^2 / scale^2) with a pre-averaged squared error."""
    s = float(scale)
    if s <= 0:
        raise ValueError("reward scale must be positive")
    return float(np.exp(-float(mean_sq_err) / (s * s)))


def _mean_sq_err(pred: np.ndarray, goal: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"pred/goal shape {p.shape} vs {g.shape}")
    err = p - g
    if err.ndim == 1:
        return float(np.dot(err, err))
    # (B, D) — paper averages over bodies then uses ||.||_2^2
    return float(np.mean(np.sum(err * err, axis=-1)))


def tracking_reward_terms(
    *,
    root_pos_p: np.ndarray,
    root_pos_g: np.ndarray,
    root_ori_p: np.ndarray,
    root_ori_g: np.ndarray,
    body_pos_rel_p: np.ndarray,
    body_pos_rel_g: np.ndarray,
    body_ori_rel_p: np.ndarray,
    body_ori_rel_g: np.ndarray,
    body_lin_p: np.ndarray,
    body_lin_g: np.ndarray,
    body_ang_p: np.ndarray,
    body_ang_g: np.ndarray,
    ee_pos_p: np.ndarray,
    ee_pos_g: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return unweighted tracking terms in [0, 1]. Weights live in YAML."""
    cfg = cfg or load_reward_cfg()
    tr = cfg["tracking"]
    terms = {
        "root_pos": exp_kernel(_mean_sq_err(root_pos_p, root_pos_g), tr["root_pos"]["scale_m"]),
        "root_ori": exp_kernel(_mean_sq_err(root_ori_p, root_ori_g), tr["root_ori"]["scale"]),
        "body_pos_rel": exp_kernel(
            _mean_sq_err(body_pos_rel_p, body_pos_rel_g), tr["body_pos_rel"]["scale_m"]
        ),
        "body_ori_rel": exp_kernel(
            _mean_sq_err(body_ori_rel_p, body_ori_rel_g), tr["body_ori_rel"]["scale"]
        ),
        "body_lin_vel": exp_kernel(
            _mean_sq_err(body_lin_p, body_lin_g), tr["body_lin_vel"]["scale_m_s"]
        ),
        "body_ang_vel": exp_kernel(
            _mean_sq_err(body_ang_p, body_ang_g), tr["body_ang_vel"]["scale_rad_s"]
        ),
        "ee_pos": exp_kernel(_mean_sq_err(ee_pos_p, ee_pos_g), tr["ee_pos"]["scale_m"]),
    }
    return terms


def weighted_tracking_return(terms: dict[str, float], cfg: dict[str, Any] | None = None) -> float:
    cfg = cfg or load_reward_cfg()
    tr = cfg["tracking"]
    key_map = {
        "root_pos": "root_pos",
        "root_ori": "root_ori",
        "body_pos_rel": "body_pos_rel",
        "body_ori_rel": "body_ori_rel",
        "body_lin_vel": "body_lin_vel",
        "body_ang_vel": "body_ang_vel",
        "ee_pos": "ee_pos",
    }
    total = 0.0
    for term_key, yaml_key in key_map.items():
        total += float(tr[yaml_key]["weight"]) * float(terms[term_key])
    return total


def action_rate_penalty(a_t: np.ndarray, a_tm1: np.ndarray, cfg: dict[str, Any] | None = None) -> float:
    cfg = cfg or load_reward_cfg()
    w = float(cfg["penalty"]["action_rate"]["weight"])
    d = np.asarray(a_t, dtype=np.float64) - np.asarray(a_tm1, dtype=np.float64)
    return w * float(np.dot(d, d))


def joint_limit_penalty(q_rad: np.ndarray, limits_rad: np.ndarray, cfg: dict[str, Any] | None = None) -> float:
    cfg = cfg or load_reward_cfg()
    w = float(cfg["penalty"]["joint_limit"]["weight"])
    q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
    lim = np.asarray(limits_rad, dtype=np.float64).reshape(-1, 2)
    n = int(np.sum((q < lim[:, 0]) | (q > lim[:, 1])))
    return w * n
