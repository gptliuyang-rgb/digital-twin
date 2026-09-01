"""T800 wrist → DexHand mount flange, composed into the robot base chain.

The live SE(3) the palm sees in ``LINK_BASE`` is

    T_base_palm = T_base_wrist_end · T_flange · T_mount_wrist

``T_base_wrist_end`` is T800 FK (dummy elbow-yaw frame, ADR-001).
``T_mount_wrist`` is official with-mount MJCF (filled).
``T_flange`` is ``t800_wrist_to_hand_mount`` from CAD, or the identity
kinematic-bringup weld (ADR-006) while CAD is ``REQUIRED_INPUT``.

Do not invent CAD numbers. Identity is sim-only and ``policy_eval_forbidden``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import yaml

from interface.schema import MOUNT_PATH, REQUIRED_INPUT_TOKEN, find_required_inputs

FlangeKind = Literal["cad", "kinematic_bringup_identity"]


def quat_wxyz_to_R(quat: np.ndarray | list[float]) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 rotation. MuJoCo / Hamilton convention."""
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4))
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """3×3 rotation → unit quaternion (w, x, y, z), w ≥ 0."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-18
    if q[0] < 0.0:
        q = -q
    return q


def R_to_rpy_xyz(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix → URDF rpy (roll about X, then pitch Y, then yaw Z)."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    pitch = float(np.arcsin(np.clip(-m[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-m[0, 1], m[1, 1]))
    else:
        roll = float(np.arctan2(m[2, 1], m[2, 2]))
        yaw = float(np.arctan2(m[1, 0], m[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def se3(pos: np.ndarray | list[float], quat_wxyz: np.ndarray | list[float]) -> np.ndarray:
    """4×4 homogeneous transform from translation (m) + wxyz quaternion."""
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return t


def se3_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def se3_inv(t: np.ndarray) -> np.ndarray:
    r = np.asarray(t, dtype=np.float64).reshape(4, 4)
    rot = r[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ r[:3, 3]
    return out


def se3_to_pos_quat(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(t, dtype=np.float64).reshape(4, 4)
    return r[:3, 3].copy(), R_to_quat_wxyz(r[:3, :3])


@dataclass(frozen=True)
class FlangeSpec:
    """Active T800 ``LINK_WRIST_END_*`` → ``{l,r}_mount`` weld."""

    kind: FlangeKind
    pos_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    policy_eval_forbidden: bool
    cad_ready: bool
    note: str = ""

    def se3(self) -> np.ndarray:
        return se3(self.pos_m, self.quat_wxyz)

    def pos_attr(self) -> str:
        return " ".join(f"{x:.8g}" for x in self.pos_m)

    def quat_attr(self) -> str:
        return " ".join(f"{x:.8g}" for x in self.quat_wxyz)

    def rpy_attr(self) -> str:
        rpy = R_to_rpy_xyz(quat_wxyz_to_R(self.quat_wxyz))
        return " ".join(f"{x:.8g}" for x in rpy)


def load_mount_raw() -> dict[str, Any]:
    return yaml.safe_load(MOUNT_PATH.read_text(encoding="utf-8"))


def mount_to_wrist_se3(raw: dict[str, Any] | None = None) -> np.ndarray:
    """Official ``{side}_mount`` → ``{side}_wrist`` (filled from with-mount MJCF)."""
    block = (raw or load_mount_raw())["hand_mount_to_wrist"]
    return se3(block["pos_m"], block["quat_wxyz"])


def cad_flange_complete(raw: dict[str, Any] | None = None) -> bool:
    cad = (raw or load_mount_raw()).get("t800_wrist_to_hand_mount") or {}
    return not find_required_inputs(cad)


def active_flange(raw: dict[str, Any] | None = None) -> FlangeSpec:
    """CAD SE(3) when both pos/quat are filled; else identity kinematic bring-up."""
    raw = raw or load_mount_raw()
    cad = raw.get("t800_wrist_to_hand_mount") or {}
    bringup = raw.get("kinematic_bringup_identity") or {}
    if cad_flange_complete(raw):
        return FlangeSpec(
            kind="cad",
            pos_m=(float(cad["pos_m"][0]), float(cad["pos_m"][1]), float(cad["pos_m"][2])),
            quat_wxyz=(
                float(cad["quat_wxyz"][0]),
                float(cad["quat_wxyz"][1]),
                float(cad["quat_wxyz"][2]),
                float(cad["quat_wxyz"][3]),
            ),
            policy_eval_forbidden=False,
            cad_ready=True,
            note=(
                "CAD flange from t800_wrist_to_hand_mount. Contact μ/k may still "
                "be REQUIRED_INPUT (ADR-004)."
            ),
        )
    if bringup.get("enabled"):
        return FlangeSpec(
            kind="kinematic_bringup_identity",
            pos_m=(
                float(bringup["pos_m"][0]),
                float(bringup["pos_m"][1]),
                float(bringup["pos_m"][2]),
            ),
            quat_wxyz=(
                float(bringup["quat_wxyz"][0]),
                float(bringup["quat_wxyz"][1]),
                float(bringup["quat_wxyz"][2]),
                float(bringup["quat_wxyz"][3]),
            ),
            policy_eval_forbidden=True,
            cad_ready=False,
            note=str(bringup.get("note") or "Identity weld; not a CAD flange."),
        )
    raise SystemExit(
        "t800_wrist_to_hand_mount is still "
        f"{REQUIRED_INPUT_TOKEN} and kinematic_bringup_identity.enabled is false."
    )


def wrist_end_to_palm_se3(raw: dict[str, Any] | None = None) -> np.ndarray:
    """``LINK_WRIST_END_*`` → ``{l,r}_wrist`` (flange · official mount offset)."""
    raw = raw or load_mount_raw()
    return se3_mul(active_flange(raw).se3(), mount_to_wrist_se3(raw))


def chain_yaml(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serializable composition for reports. No MuJoCo, no invented CAD."""
    raw = raw or load_mount_raw()
    flange = active_flange(raw)
    mount = raw["hand_mount_to_wrist"]
    t_palm = wrist_end_to_palm_se3(raw)
    pos, quat = se3_to_pos_quat(t_palm)
    return {
        "formula": "T_base_palm = T_base_wrist_end * T_flange * T_mount_wrist",
        "base_body": "LINK_BASE",
        "heading_frame": "robot_heading_frame",
        "t800_wrist_body": {
            "left": raw["humanoid"]["left_wrist_link"],
            "right": raw["humanoid"]["right_wrist_link"],
        },
        "hand_root": {"left": "l_wrist", "right": "r_wrist"},
        "T_flange": {
            "kind": flange.kind,
            "parent": "LINK_WRIST_END_*",
            "child": "{l,r}_mount",
            "pos_m": list(flange.pos_m),
            "quat_wxyz": list(flange.quat_wxyz),
            "policy_eval_forbidden": flange.policy_eval_forbidden,
            "cad_ready": flange.cad_ready,
            "note": flange.note,
        },
        "T_mount_wrist": {
            "source": mount.get("source"),
            "pos_m": list(mount["pos_m"]),
            "quat_wxyz": list(mount["quat_wxyz"]),
        },
        "T_wrist_end_palm": {
            "pos_m": pos.tolist(),
            "quat_wxyz": quat.tolist(),
            "note": "Constant. T_base_wrist_end is T800 FK and changes with q.",
        },
        "policy_eval_forbidden": flange.policy_eval_forbidden,
    }


def rgb_axis_site_xml(prefix: str, *, length_m: float = 0.04, radius_m: float = 0.0025) -> list[str]:
    """RGB triad sites in a body frame (X red, Y green, Z blue).

    Sites never collide in MuJoCo — do not put ``contype`` / ``conaffinity`` here
    (those attributes are geom-only and fail XML schema validation).
    """
    axes = (
        ("x", f"{length_m} 0 0", "0.85 0.12 0.12 1"),
        ("y", f"0 {length_m} 0", "0.12 0.75 0.18 1"),
        ("z", f"0 0 {length_m}", "0.15 0.35 0.90 1"),
    )
    out = []
    for name, tip, rgba in axes:
        out.append(
            f'<site name="frame_{prefix}_{name}" type="capsule" size="{radius_m}" '
            f'fromto="0 0 0 {tip}" rgba="{rgba}" group="4"/>'
        )
    return out
