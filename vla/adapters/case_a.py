"""Explicit Case A → command_schema_v1 conversion (arm q + fingers → wrist SE(3)).

PolicyClient, DeployPipeline, and π0.5 glue never call this. A 50-D vector stays
refused at those boundaries until a caller invokes
``CaseAToCommandSchema.convert(..., apply_fk=True, head_nav=...)``.

Head pose, pelvis height, nav, loco_mode, and tool_trigger are **not** in the
50-D action. They come from L3 / HeadNavCommand. This module does not invent them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from interface.schema import (
    REPO_ROOT,
    CommandVector,
    HandSpec,
    command_dim,
    load_frames,
    load_hand_spec,
)
from vla.adapters.action_space import ActionSpaceMismatch, diagnose_action_vector
from vla.adapters.frame_transform import world_pose_to_heading
from vla.adapters.joint_to_wrist_adapter import JointToWristAdapter, WristPose

LAYOUT_PATH = Path(__file__).with_name("case_a_layout.yaml")
T800_JOINTS = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_joints.yaml"


class CaseAConversionError(ActionSpaceMismatch):
    """Raised when Case A conversion is missing apply_fk, head_nav, or layout."""


def load_case_a_layout(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or LAYOUT_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("case_a_layout.yaml must be a mapping")
    if raw.get("case") != "A":
        raise ValueError("case_a_layout.yaml case must be A")
    return raw


def _nested(raw: dict[str, Any], dotted: str) -> Any:
    node: Any = raw
    for part in dotted.split("."):
        node = node[part]
    return node


def case_a_slices(spec: HandSpec | None = None) -> dict[str, tuple[int, int]]:
    """Inclusive-exclusive slices of the 50-D Case A vector. Dims from YAML refs."""
    spec = spec or load_hand_spec()
    layout = load_case_a_layout()
    t800 = yaml.safe_load(T800_JOINTS.read_text(encoding="utf-8"))
    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for item in layout["packing_order"]:
        name = str(item["name"])
        key = str(item["key"])
        if key.startswith("arm_joints"):
            dim = len(_nested(t800, key))
        elif key == "joint_order":
            dim = spec.n_active_dof
        else:
            raise ValueError(f"unknown packing key {key}")
        slices[name] = (cursor, cursor + dim)
        cursor += dim
    slices["_total"] = (0, cursor)
    return slices


def case_a_dim(spec: HandSpec | None = None) -> int:
    return case_a_slices(spec)["_total"][1]


@dataclass(frozen=True)
class HeadNavCommand:
    """Fields a Case A ckpt does not output. Required; no silent zeros.

    Case A is 10 arm joints + 40 finger joints. Neck/torso q are not in that
    vector, so head pose is supplied here (L3 FSM or a labelled eval fixture).
    ``source`` names the supplier (e.g. ``l3_fsm`` or ``synthetic_l1_fixture``).
    """

    pelvis_height_m: float
    nav_cmd_mps: np.ndarray
    loco_mode: int
    tool_trigger: int
    source: str
    head_pos_m: np.ndarray
    head_rot6d: np.ndarray
    left_hand_mode: int = 0
    right_hand_mode: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "nav_cmd_mps", np.asarray(self.nav_cmd_mps, dtype=np.float64).reshape(3))
        object.__setattr__(self, "head_pos_m", np.asarray(self.head_pos_m, dtype=np.float64).reshape(3))
        object.__setattr__(self, "head_rot6d", np.asarray(self.head_rot6d, dtype=np.float64).reshape(6))
        if not self.source:
            raise CaseAConversionError("HeadNavCommand.source must name who supplied these fields")
        if self.loco_mode not in (0, 1, 2):
            raise ValueError(f"loco_mode {self.loco_mode} not in {{0,1,2}}")
        if self.tool_trigger not in (0, 1):
            raise ValueError("tool_trigger must be 0 or 1")
        if not (0.30 <= float(self.pelvis_height_m) <= 0.80):
            raise ValueError(f"pelvis_height_m {self.pelvis_height_m} outside schema range [0.30, 0.80] m")


class CaseAToCommandSchema:
    """FK adapter. Construct explicitly; never registered as a PolicyClient hook."""

    def __init__(
        self,
        *,
        left: JointToWristAdapter,
        right: JointToWristAdapter,
        spec: HandSpec | None = None,
    ) -> None:
        self.left = left
        self.right = right
        self.spec = spec or load_hand_spec()
        self.slices = case_a_slices(self.spec)
        expected = diagnose_action_vector(dim=case_a_dim(self.spec), spec=self.spec)
        if expected.layout_name != "t800_dual_arm_q_plus_hands":
            raise CaseAConversionError(f"layout mismatch: {expected.layout_name}")

    @classmethod
    def from_injected_fk(
        cls,
        left_fk,
        right_fk,
        spec: HandSpec | None = None,
    ) -> CaseAToCommandSchema:
        """Unit-test / no-URDF constructor. ``fk`` is (q_arm,) -> (pos_m, rot_3x3)."""
        return cls(
            left=JointToWristAdapter(fk_fn=left_fk, side="left"),
            right=JointToWristAdapter(fk_fn=right_fk, side="right"),
            spec=spec,
        )

    @classmethod
    def from_t800_urdf(cls, urdf_path: Path | None = None, spec: HandSpec | None = None) -> CaseAToCommandSchema:
        return cls(
            left=JointToWristAdapter(urdf_path=urdf_path, side="left"),
            right=JointToWristAdapter(urdf_path=urdf_path, side="right"),
            spec=spec,
        )

    def convert(
        self,
        vec: np.ndarray,
        *,
        apply_fk: bool,
        head_nav: HeadNavCommand,
        fk_frame: Literal["pelvis", "world"] = "pelvis",
        pelvis_pos_world_m: np.ndarray | None = None,
        pelvis_yaw_rad: float = 0.0,
        emit_five_point: bool = False,
    ) -> CommandVector:
        """Map one Case A vector to command_schema_v1.

        ``apply_fk`` must be the literal True. Omitting it is a TypeError.
        Passing False is a hard refuse (no identity / no pad).
        """
        if apply_fk is not True:
            raise CaseAConversionError(
                "refusing silent Case A conversion. Call convert(..., apply_fk=True, head_nav=...)."
            )
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        report = diagnose_action_vector(arr, spec=self.spec)
        if report.case != "A" or report.layout_name != "t800_dual_arm_q_plus_hands":
            raise CaseAConversionError(
                f"convert() only accepts Case A t800_dual_arm_q_plus_hands, got "
                f"case={report.case} layout={report.layout_name} dim={report.dim}"
            )
        sl = self.slices
        q_left = arr[slice(*sl["left_arm_q"])]
        q_right = arr[slice(*sl["right_arm_q"])]
        left_hand = arr[slice(*sl["left_hand_q"])]
        right_hand = arr[slice(*sl["right_hand_q"])]
        left = self.left(q_left)
        right = self.right(q_right)
        left_pos, left_rot = self._maybe_heading(left, fk_frame, pelvis_pos_world_m, pelvis_yaw_rad)
        right_pos, right_rot = self._maybe_heading(right, fk_frame, pelvis_pos_world_m, pelvis_yaw_rad)
        head_pose = WristPose(
            pos_m=np.asarray(head_nav.head_pos_m, dtype=np.float64),
            rot6d=np.asarray(head_nav.head_rot6d, dtype=np.float64),
            link=load_frames()["frames"]["pelvis"]["t800_link"],
        )
        head_pos, head_rot = self._maybe_heading(
            head_pose, fk_frame, pelvis_pos_world_m, pelvis_yaw_rad
        )
        cmd = CommandVector(
            head_pos=head_pos,
            head_rot6d=head_rot,
            left_wrist_pos=left_pos,
            left_wrist_rot6d=left_rot,
            right_wrist_pos=right_pos,
            right_wrist_rot6d=right_rot,
            pelvis_height=float(head_nav.pelvis_height_m),
            nav_cmd=head_nav.nav_cmd_mps,
            loco_mode=int(head_nav.loco_mode),
            left_hand_q=left_hand,
            right_hand_q=right_hand,
            left_hand_mode=int(head_nav.left_hand_mode),
            right_hand_mode=int(head_nav.right_hand_mode),
            tool_trigger=int(head_nav.tool_trigger),
            spec=self.spec,
        )
        cmd.validate()
        if emit_five_point:
            # Elbows stay off the 75-D vector (ADR-017). Requires URDF trees, not wrist-only fk_fn.
            _ = (self.left.elbow_pose(q_left), self.right.elbow_pose(q_right))
        assert cmd.to_flat_vector().shape == (command_dim(self.spec),)
        return cmd

    def convert_chunk(
        self,
        chunk: np.ndarray,
        *,
        apply_fk: bool,
        head_nav: HeadNavCommand,
        **kwargs: Any,
    ) -> np.ndarray:
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows = [
            self.convert(row, apply_fk=apply_fk, head_nav=head_nav, **kwargs).to_flat_vector()
            for row in arr
        ]
        return np.stack(rows, axis=0)

    @staticmethod
    def _maybe_heading(
        pose: WristPose,
        fk_frame: str,
        pelvis_pos_world_m: np.ndarray | None,
        pelvis_yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if fk_frame == "pelvis":
            return pose.pos_m.copy(), pose.rot6d.copy()
        if fk_frame != "world":
            raise CaseAConversionError(f"fk_frame must be pelvis or world, got {fk_frame!r}")
        if pelvis_pos_world_m is None:
            raise CaseAConversionError("fk_frame='world' requires pelvis_pos_world_m")
        return world_pose_to_heading(pose.pos_m, pose.rot6d, pelvis_pos_world_m, pelvis_yaw_rad)


def split_case_a(vec: np.ndarray, spec: HandSpec | None = None) -> dict[str, np.ndarray]:
    sl = case_a_slices(spec)
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != sl["_total"][1]:
        raise CaseAConversionError(f"expected Case A dim {sl['_total'][1]}, got {arr.shape[0]}")
    return {name: arr[a:b].copy() for name, (a, b) in sl.items() if name != "_total"}
