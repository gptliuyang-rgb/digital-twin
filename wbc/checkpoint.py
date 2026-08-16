"""Refuse Unitree-G1 SONIC weights on T800. Same check in sim and on the robot."""

from __future__ import annotations

from typing import Any

from wbc.dims import G1_DECODER_INPUT_DIM, G1_N_DOF, decoder_history_dim, load_t800_sonic


class G1CheckpointIncompatible(RuntimeError):
    """Raised when a G1 SONIC checkpoint is offered to a T800 policy client."""


def refuse_g1_checkpoint(
    *,
    robot: str | None = None,
    n_dof: int | None = None,
    decoder_input_dim: int | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
) -> None:
    cfg = load_t800_sonic()
    host = (robot or cfg["robot"]).lower()
    if not host.startswith("t800"):
        return
    meta = checkpoint_meta or {}
    ckpt_robot = str(meta.get("robot", "")).lower()
    ckpt_dof = meta.get("n_dof", n_dof)
    ckpt_dim = meta.get("decoder_input_dim", decoder_input_dim)
    g1_like = (
        ckpt_robot.startswith("g1")
        or ckpt_dof == G1_N_DOF
        or ckpt_dim == G1_DECODER_INPUT_DIM
    )
    if not g1_like:
        if ckpt_dof is not None and int(ckpt_dof) != int(cfg["n_revolute"]):
            raise G1CheckpointIncompatible(
                f"checkpoint n_dof={ckpt_dof} does not match T800 {cfg['n_revolute']}"
            )
        return
    t800_dim = decoder_history_dim(int(cfg["n_revolute"]))
    raise G1CheckpointIncompatible(
        "Official GEAR-SONIC checkpoints are Unitree G1 29-DoF only "
        f"(decoder_input_dim={G1_DECODER_INPUT_DIM}). T800 is {cfg['n_revolute']}-DoF "
        f"(decoder_input_dim={t800_dim}). Retrain from BONES-SEED via GMR; "
        "do not load G1 ONNX."
    )
