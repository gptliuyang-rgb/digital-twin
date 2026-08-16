"""Lightweight URDF forward kinematics. No Pinocchio / MuJoCo.

Used by the VLA joint→wrist adapter and L1. Joint names come from the URDF,
never from string literals in callers — pass them in from frames.yaml / t800_joints.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll). Kept local to avoid vla↔sim import cycles."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz_m: np.ndarray
    origin_rpy_rad: np.ndarray
    axis: np.ndarray
    lower_rad: float | None = None
    upper_rad: float | None = None


def _origin_xyz_rpy(block: str) -> tuple[np.ndarray, np.ndarray]:
    match = re.search(r'<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)"', block)
    if not match:
        match = re.search(r'<origin\s+rpy="([^"]+)"\s+xyz="([^"]+)"', block)
        if match:
            rpy = np.array([float(x) for x in match.group(1).split()], dtype=np.float64)
            xyz = np.array([float(x) for x in match.group(2).split()], dtype=np.float64)
            return xyz, rpy
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(x) for x in match.group(1).split()], dtype=np.float64)
    rpy = np.array([float(x) for x in match.group(2).split()], dtype=np.float64)
    return xyz, rpy


def _axis(block: str) -> np.ndarray:
    match = re.search(r'<axis\s+xyz="([^"]+)"', block)
    if not match:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = np.array([float(x) for x in match.group(1).split()], dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        raise ValueError("zero joint axis")
    return axis / n


def _limit(block: str) -> tuple[float | None, float | None]:
    match = re.search(r'<limit\s+[^>]*lower="([^"]+)"\s+upper="([^"]+)"', block)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_urdf_joints(text: str) -> list[UrdfJoint]:
    joints: list[UrdfJoint] = []
    for match in re.finditer(r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', text, re.S):
        block = match.group(3)
        parent = re.search(r'<parent\s+link="([^"]+)"', block)
        child = re.search(r'<child\s+link="([^"]+)"', block)
        if not parent or not child:
            continue
        xyz, rpy = _origin_xyz_rpy(block)
        lo, hi = _limit(block)
        joints.append(
            UrdfJoint(
                name=match.group(1),
                joint_type=match.group(2),
                parent=parent.group(1),
                child=child.group(1),
                origin_xyz_m=xyz,
                origin_rpy_rad=rpy,
                axis=_axis(block),
                lower_rad=lo,
                upper_rad=hi,
            )
        )
    return joints


def se3(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


class UrdfTree:
    def __init__(self, joints: list[UrdfJoint], *, root_link: str) -> None:
        self.joints = joints
        self.root_link = root_link
        self.by_child = {j.child: j for j in joints}
        self.by_name = {j.name: j for j in joints}

    @classmethod
    def from_urdf_text(cls, text: str, *, root_link: str) -> UrdfTree:
        return cls(parse_urdf_joints(text), root_link=root_link)

    @classmethod
    def from_path(cls, path: Path, *, root_link: str) -> UrdfTree:
        return cls.from_urdf_text(path.read_text(encoding="utf-8"), root_link=root_link)

    def chain_to(self, target_link: str) -> list[UrdfJoint]:
        chain: list[UrdfJoint] = []
        link = target_link
        seen: set[str] = set()
        while link != self.root_link:
            if link in seen:
                raise ValueError(f"cycle while walking to {target_link}")
            seen.add(link)
            joint = self.by_child.get(link)
            if joint is None:
                raise KeyError(f"no joint with child={link}; cannot reach {self.root_link}")
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    def fk_link(
        self,
        target_link: str,
        q_rad_by_joint: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_m, rot_3x3) of `target_link` in the root-link frame."""
        t = np.eye(4)
        for joint in self.chain_to(target_link):
            t_origin = se3(rpy_to_matrix(*joint.origin_rpy_rad.tolist()), joint.origin_xyz_m)
            if joint.joint_type == "fixed":
                t_motion = np.eye(4)
            elif joint.joint_type in ("revolute", "continuous"):
                q = float(q_rad_by_joint.get(joint.name, 0.0))
                t_motion = se3(rodrigues(joint.axis, q), np.zeros(3))
            elif joint.joint_type == "prismatic":
                q = float(q_rad_by_joint.get(joint.name, 0.0))
                t_motion = se3(np.eye(3), joint.axis * q)
            else:
                raise ValueError(f"unsupported joint type {joint.joint_type} ({joint.name})")
            t = t @ t_origin @ t_motion
        return t[:3, 3].copy(), t[:3, :3].copy()
