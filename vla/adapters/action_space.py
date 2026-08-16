"""Classify a VLA checkpoint's action space before any physics or SONIC load.

The digital-twin contract is command_schema_v1 (75-D: heading-frame wrist SE(3)
+ hands). Existing ckpts are one of:

  A. dual-arm joint angles + finger angles  → needs FK adapter
  B. wrist SE(3) + finger angles            → pass-through after frame/6D convert
  C. joint velocity or increment            → do not integrate ad-hoc; retrain

Do not guess. Unknown dims raise ActionSpaceMismatch listing every known layout.
G1 29-DoF / decoder-994 vectors are refused (ADR-011). 5-point (81-D) is not a
concat onto a 3-point ckpt (ADR-017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from interface.schema import REPO_ROOT, command_dim, load_hand_spec
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import G1_N_DOF, hybrid_encoder_cmd_dim, load_t800_sonic

T800_JOINTS = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_joints.yaml"


class ActionSpaceMismatch(ValueError):
    """Raised when a vector cannot be loaded as command_schema_v1."""


def _arm_dof() -> int:
    raw = yaml.safe_load(T800_JOINTS.read_text(encoding="utf-8"))
    left = len(raw["arm_joints"]["left"])
    right = len(raw["arm_joints"]["right"])
    return left + right


def known_layouts(spec=None) -> dict[str, dict[str, Any]]:
    """Named action last-dims this repo will accept or explicitly refuse."""
    spec = spec or load_hand_spec()
    n_h = spec.n_active_dof
    n_arm = _arm_dof()
    n_body = int(load_t800_sonic()["n_revolute"])
    cmd = command_dim(spec)
    five = cmd + 6
    return {
        "command_schema_v1": {
            "dim": cmd,
            "case": "B",
            "aligned": True,
            "adapter": None,
            "note": "head+wrists rot6d + pelvis/nav + DexHand2 q. Load as CommandVector.",
        },
        "command_schema_v1_5point": {
            "dim": five,
            "case": "B",
            "aligned": False,
            "adapter": "refuse_3point_ckpt",
            "note": "Opt-in elbows. Requires a 5-point T800 SONIC retrain (ADR-017).",
        },
        "sonic_vr_3point": {
            "dim": hybrid_encoder_cmd_dim("vr_3point"),
            "case": "B",
            "aligned": False,
            "adapter": "teleop.vr_3point_to_wbc_fields + hands from elsewhere",
            "note": "WBC token only. Hands bypass SONIC and are missing from this vector.",
        },
        "sonic_vr_5point": {
            "dim": hybrid_encoder_cmd_dim("vr_5point"),
            "case": "B",
            "aligned": False,
            "adapter": "refuse_3point_ckpt",
            "note": "5-point WBC token (27). Not a deploy-time concat onto 21-D.",
        },
        "t800_dual_arm_q_plus_hands": {
            "dim": n_arm + 2 * n_h,
            "case": "A",
            "aligned": False,
            "adapter": "vla.adapters.JointToWristAdapter",
            "note": f"T800 2×{n_arm // 2} arm joints + 2×{n_h} fingers. FK to wrist SE(3).",
        },
        "t800_body_q": {
            "dim": n_body,
            "case": "A",
            "aligned": False,
            "adapter": "not a VLA action; this is the SONIC joint target a_t",
            "note": "25-D T800 body q. Do not treat as command_schema_v1.",
        },
        "g1_body_q": {
            "dim": G1_N_DOF,
            "case": "refuse",
            "aligned": False,
            "adapter": None,
            "note": "Unitree G1 29-DoF. Unloadable on T800 (ADR-011).",
        },
        "wrist_se3_quat_plus_hands": {
            "dim": 2 * (3 + 4) + 2 * n_h,
            "case": "B",
            "aligned": False,
            "adapter": "quaternion_wxyz → rot6d + heading-frame convert",
            "note": "Two wrists as pos+quat plus fingers. Missing head/nav/modes.",
        },
        "wrist_rot6d_plus_hands": {
            "dim": 2 * (3 + 6) + 2 * n_h,
            "case": "B",
            "aligned": False,
            "adapter": "frame_transform.world_pose_to_heading; pad head/nav",
            "note": "Two wrists as pos+rot6d plus fingers. Missing head/nav/modes.",
        },
    }


def _action_keys(modality: dict[str, Any] | None) -> list[str]:
    if not modality:
        return []
    action = modality.get("action", modality)
    if isinstance(action, dict):
        return [str(k).lower() for k in action]
    return []


def _case_from_modality(keys: list[str]) -> str | None:
    blob = " ".join(keys)
    vel = any(tok in blob for tok in ("dpos", "delta", "velocity", "dq", "d_q", "increment"))
    if vel:
        return "C"
    pose = any(tok in blob for tok in ("wrist", "eef", "ee_pose", "rot6d", "quat", "pose"))
    joints = any(tok in blob for tok in ("joint", "_q", "arm_q", "qpos"))
    if pose and not joints:
        return "B"
    if joints and not pose:
        return "A"
    if pose and joints:
        return "B"
    return None


@dataclass
class ActionSpaceReport:
    dim: int
    case: str
    layout_name: str | None
    aligned_with_command_schema: bool
    adapter: str | None
    notes: list[str] = field(default_factory=list)
    known: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "case": self.case,
            "layout_name": self.layout_name,
            "aligned_with_command_schema": self.aligned_with_command_schema,
            "adapter": self.adapter,
            "notes": self.notes,
            "known_layouts": self.known,
        }


def _last_dim(vec: np.ndarray) -> int:
    arr = np.asarray(vec)
    if arr.ndim == 0:
        raise ActionSpaceMismatch("action vector is scalar")
    return int(arr.shape[-1])


def diagnose_action_vector(
    vec: np.ndarray | None = None,
    *,
    dim: int | None = None,
    modality: dict[str, Any] | None = None,
    spec=None,
) -> ActionSpaceReport:
    """Return the A/B/C diagnosis. Never invent a layout for an unknown dim."""
    spec = spec or load_hand_spec()
    layouts = known_layouts(spec)
    known_dims = {name: int(meta["dim"]) for name, meta in layouts.items()}
    if dim is None:
        if vec is not None:
            dim = _last_dim(np.asarray(vec))
        elif modality is not None and modality.get("action_dim") is not None:
            dim = int(modality["action_dim"])
        else:
            raise ActionSpaceMismatch("pass a vector, an explicit dim, or modality.action_dim")
    dim = int(dim)

    if dim == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=dim)

    match_name = next((name for name, d in known_dims.items() if d == dim), None)
    keys = _action_keys(modality)
    modal_case = _case_from_modality(keys)

    if match_name is None:
        notes = [
            f"action last-dim {dim} matches none of the frozen layouts {known_dims}.",
            "Do not pad, slice, or interpret this ckpt. Dump modality.json and joint_order.",
        ]
        if modal_case == "C":
            notes.append("modality looks like velocity/delta (case C) — retrain, do not integrate.")
        raise ActionSpaceMismatch(" ".join(notes))

    meta = layouts[match_name]
    case = str(meta["case"])
    if modal_case == "C":
        case = "C"
    elif modal_case is not None and case != "refuse" and modal_case != case:
        # Dim matched a pose layout but modality says joints (or vice versa).
        case = modal_case

    notes = [str(meta["note"])]
    if case == "C":
        notes.append("Case C: chunk-internal drift will accumulate. Do not write an integrator.")
    if match_name == "command_schema_v1_5point":
        notes.append("A 3-point SONIC checkpoint cannot consume this vector (ADR-017).")
    if match_name == "command_schema_v1" and case == "B":
        notes.append("L0 open-loop replay may run. L2/L3 still blocked on contact/flange P0.")

    return ActionSpaceReport(
        dim=dim,
        case=case,
        layout_name=match_name,
        aligned_with_command_schema=bool(meta["aligned"]) and case == "B",
        adapter=None if case == "C" else meta["adapter"],
        notes=notes,
        known=known_dims,
    )


def require_command_schema_vector(vec: np.ndarray, spec=None) -> np.ndarray:
    """Flatten to (D,) and demand D == command_schema_v1. Used by PolicyClient."""
    spec = spec or load_hand_spec()
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)
    expected = command_dim(spec)
    if arr.shape[0] != expected:
        report = diagnose_action_vector(arr, spec=spec)
        raise ActionSpaceMismatch(
            f"expected command_schema_v1 dim {expected}, got {arr.shape[0]} "
            f"(case {report.case}, layout {report.layout_name}). "
            f"adapter={report.adapter}. " + " ".join(report.notes)
        )
    return arr
