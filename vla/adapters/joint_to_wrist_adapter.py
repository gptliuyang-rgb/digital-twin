"""FK adapter: arm joint angles → wrist SE(3).

Uses `sim/urdf_fk.py` (pure numpy URDF). Joint names are read from
`assets/engineai/meta/t800_joints.yaml` and `interface/frames.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import REPO_ROOT, load_frames
from vla.adapters.rotation import matrix_to_rot6d

T800_JOINTS = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_joints.yaml"
T800_KINEMATICS = REPO_ROOT / "assets" / "engineai" / "meta" / "t800_kinematics.yaml"
T800_URDF = (
    REPO_ROOT
    / "third_party"
    / "engineai-native-sdk"
    / "assets"
    / "resource"
    / "robot"
    / "t800"
    / "urdf"
    / "serial_t800.urdf"
)


@dataclass
class WristPose:
    pos_m: np.ndarray
    rot6d: np.ndarray
    link: str


def _t800_meta() -> dict:
    return yaml.safe_load(T800_JOINTS.read_text(encoding="utf-8"))


class JointToWristAdapter:
    def __init__(
        self,
        fk_fn=None,
        *,
        tree: Any = None,
        urdf_path: Path | None = None,
        side: str = "right",
    ) -> None:
        self.fk_fn = fk_fn
        self.side = side
        frames = load_frames()["frames"]
        key = "left_wrist" if side == "left" else "right_wrist"
        self.wrist_link = frames[key]["t800_link"]
        elbow_key = "left_elbow" if side == "left" else "right_elbow"
        self.elbow_link = frames[elbow_key]["t800_link"]
        self.root_link = frames["pelvis"]["t800_link"]
        meta = _t800_meta()
        self.arm_joints = list(meta["arm_joints"][side])
        self.tree = tree
        if self.tree is None and fk_fn is None:
            from sim.urdf_fk import UrdfTree

            path = urdf_path or T800_URDF
            if path.is_file():
                self.tree = UrdfTree.from_path(path, root_link=self.root_link)
            elif T800_KINEMATICS.is_file():
                # Committed kinematics extract — CI without Native SDK clone.
                self.tree = UrdfTree.from_kinematics_yaml(T800_KINEMATICS, root_link=self.root_link)
            else:
                raise FileNotFoundError(
                    f"T800 URDF missing at {path} and no kinematics YAML at {T800_KINEMATICS}. "
                    "Run scripts/bootstrap_resources.sh"
                )

    def __call__(self, q_arm: np.ndarray, extra_q: dict[str, float] | None = None) -> WristPose:
        if self.fk_fn is not None:
            pos, rot = self.fk_fn(np.asarray(q_arm, dtype=np.float64))
            return WristPose(
                pos_m=np.asarray(pos, dtype=np.float64).reshape(3),
                rot6d=matrix_to_rot6d(rot),
                link=self.wrist_link,
            )
        q_map = {name: float(v) for name, v in zip(self.arm_joints, np.asarray(q_arm, dtype=np.float64), strict=True)}
        if extra_q:
            q_map.update(extra_q)
        pos, rot = self.tree.fk_link(self.wrist_link, q_map)
        return WristPose(pos_m=pos, rot6d=matrix_to_rot6d(rot), link=self.wrist_link)

    def elbow_pose(self, q_arm: np.ndarray, extra_q: dict[str, float] | None = None) -> WristPose:
        """FK of the 5-point elbow body (LINK_ELBOW_PITCH_*), heading/pelvis frame of the URDF."""
        if self.fk_fn is not None:
            raise NotImplementedError("elbow_pose needs the URDF tree, not a wrist-only fk_fn")
        q_map = {name: float(v) for name, v in zip(self.arm_joints, np.asarray(q_arm, dtype=np.float64), strict=True)}
        if extra_q:
            q_map.update(extra_q)
        pos, rot = self.tree.fk_link(self.elbow_link, q_map)
        return WristPose(pos_m=pos, rot6d=matrix_to_rot6d(rot), link=self.elbow_link)
