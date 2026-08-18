"""SONIC §S7 YAML-driven observation gathering. Numpy only — no simulator.

He et al., arXiv:2511.07820v3 §S7: at each 50 Hz control tick the stack
reads IMU orientation / angular velocity and joint q/dq, remaps to the
policy joint order, and pushes a timestamped snapshot into a dual-purpose
state logger (ring buffer + optional CSV). History is strided lookback
with zero-padding at startup. Observation order and offsets come from
``wbc/obs_gather.yaml``, compiled into (function, offset, dim) triples.

Official GEAR-SONIC ``observation_config.yaml`` concatenates grouped
history blocks (token | ω_hist | q_hist | dq_hist | a_hist | g_hist).
T800 dims: 64+30+250+250+250+30 = 874. G1 994 is refused (ADR-011).

Hands still bypass WBC. No invented sensor latency (delay ticks stay 0
until SPEC_INTAKE measures one). Encoder ``motion_*_10frame_step5`` is a
future look-ahead assembled from a caller-supplied 50 Hz reference
(ADR-042). ADR-043 adds official ``encoder_mode_4`` plus the teleop
lower-body + VR 3-point list; unused superset slots are zero-filled.
ADR-044 adds the official ``low_latency/`` layout: g1/teleop
``*_10frame_step1`` (no root_z); SMPL/wrist ``*_4frame_step1`` stay refused.
ADR-045 adds official ``sonic_v1_1/``: g1/teleop ``*_10frame_step5`` with
``motion_anchor_orientation_heading_*`` (no root_z); SMPL/wrist
``*_10frame_step1`` stay refused. G1 encoder ONNX / 650-D / 1751-D /
1247-D / wrist / SMPL channels are refused. The T800 v1.1 vector is
831-D — the same integer as low-latency, different names. Do not pair them.
VR 3-point is packed from ``command_schema_v1``, not a PICO SDK.
Not Table S4. Not pad–cardboard. Not an invented clip.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from interface.schema import CommandVector
from vla.adapters.frame_transform import yaw_from_quat_wxyz
from vla.adapters.rotation import quaternion_wxyz_to_matrix
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import (
    ANCHOR_ORI_DIM,
    ENCODER_VARIANT_DEFAULT,
    ENCODER_VARIANT_LOW_LATENCY,
    ENCODER_VARIANT_SONIC_V1_1,
    G1_DECODER_INPUT_DIM,
    G1_ENCODER_MOTION_DIM,
    G1_ENCODER_MOTION_DIM_LOW_LATENCY,
    G1_ENCODER_ONNX_DIM,
    G1_ENCODER_ONNX_DIM_LOW_LATENCY,
    G1_ENCODER_ONNX_DIM_UNPATCHED,
    G1_ENCODER_ONNX_DIM_V1_1,
    G1_N_DOF,
    HISTORY_FRAMES,
    ROOT_Z_DIM,
    SMPL_LOW_LATENCY_FRAMES,
    T800_N_LOWER_BODY_DOF,
    TOKEN_DIM,
    decoder_history_dim,
    encoder_motion_dim,
    load_t800_sonic,
    t800_encoder_onnx_dim,
    t800_encoder_onnx_dim_low_latency,
    t800_encoder_onnx_dim_v1_1,
)
from wbc.motion_ref import (
    MotionHold,
    MotionRefError,
    heading_corrected_rel_rot6d,
    lower_body_slice,
    refuse_smpl_observation,
    refuse_wrist_observation,
)
from wbc.observation import heading_angular_velocity, heading_gravity
from wbc.stream import OPERATOR_INPUT_HZ, PLANNER_HZ, POLICY_HZ, STREAM_HZ
from wbc.teleop import FivePointCommand, command_to_vr_3point

GATHER_YAML = Path(__file__).with_name("obs_gather.yaml")
GATHER_LOW_LATENCY_YAML = Path(__file__).with_name("obs_gather_low_latency.yaml")
GATHER_SONIC_V1_1_YAML = Path(__file__).with_name("obs_gather_sonic_v1_1.yaml")
_G1_ENCODER_DIMS_FORBIDDEN = frozenset(
    {
        G1_ENCODER_MOTION_DIM,
        G1_ENCODER_MOTION_DIM_LOW_LATENCY,
        G1_ENCODER_ONNX_DIM,
        G1_ENCODER_ONNX_DIM_UNPATCHED,
        G1_ENCODER_ONNX_DIM_LOW_LATENCY,
        G1_ENCODER_ONNX_DIM_V1_1,
    }
)
_HISTORY_NAME = re.compile(r"^(?P<base>.+)_(?P<n>\d+)frame_step(?P<s>\d+)$")
_FORBIDDEN_NAME_BITS = ("finger", "hand_q", "dexhand", "tactile")

GatherFn = Callable[["ObsGather", np.ndarray, int], None]


class ObsGatherError(ValueError):
    """Wrong YAML, dimension, joint order, delay, or history request."""


@dataclass
class HardwareSnapshot:
    """One robot-state sample. Joints are hardware order until remapped."""

    t_s: float
    q_hw: np.ndarray
    dq_hw: np.ndarray
    omega_imu: np.ndarray
    imu_quat_wxyz: np.ndarray
    last_action: np.ndarray


@dataclass
class PolicySnapshot:
    """One 50 Hz control-tick sample in policy joint order, heading frame."""

    t_s: float
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    omega_heading: np.ndarray
    gravity_heading: np.ndarray
    last_action: np.ndarray


@dataclass(frozen=True)
class ObsSlot:
    name: str
    offset: int
    dim: int
    gather: GatherFn


def load_gather_cfg(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load((path or GATHER_YAML).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("obs_gather.yaml must be a mapping")
    if raw.get("not_dexhand2_contact") is not True:
        raise ValueError("obs_gather.yaml must keep not_dexhand2_contact: true")
    if raw.get("not_table_s4") is not True:
        raise ValueError("obs_gather.yaml must keep not_table_s4: true")
    if raw.get("not_hand_mit_ring") is not True:
        raise ValueError("obs_gather.yaml must keep not_hand_mit_ring: true")
    if raw.get("not_invented_latency") is not True:
        raise ValueError("obs_gather.yaml must keep not_invented_latency: true")
    if raw.get("not_pico_sdk") is not True:
        raise ValueError("obs_gather.yaml must keep not_pico_sdk: true")
    if raw.get("not_invented_clip") is not True:
        raise ValueError("obs_gather.yaml must keep not_invented_clip: true")
    if raw.get("not_g1_encoder_onnx") is not True:
        raise ValueError("obs_gather.yaml must keep not_g1_encoder_onnx: true")
    if raw.get("not_smpl_encoder") is not True:
        raise ValueError("obs_gather.yaml must keep not_smpl_encoder: true")
    if int(raw["policy_hz"]) != POLICY_HZ:
        raise ValueError("policy_hz must stay 50 (SONIC §3.5 / S7 control loop)")
    if int(raw["command_stream_hz"]) != STREAM_HZ:
        raise ValueError("command_stream_hz must stay 500 (SONIC §3.5)")
    if int(raw["operator_input_hz"]) != OPERATOR_INPUT_HZ:
        raise ValueError("operator_input_hz must stay 100 (SONIC §3.5)")
    if int(raw["planner_hz"]) != PLANNER_HZ:
        raise ValueError("planner_hz must stay 10 (SONIC §3.5)")
    if int(raw["n_dof"]) == G1_N_DOF:
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    sonic = load_t800_sonic()
    if int(raw["n_dof"]) != int(sonic["n_revolute"]):
        raise ObsGatherError(f"n_dof {raw['n_dof']} != T800 {sonic['n_revolute']}")
    if int(raw["token_dim"]) != TOKEN_DIM:
        raise ObsGatherError(f"token_dim must stay {TOKEN_DIM}")
    if int(raw["history_frames"]) != HISTORY_FRAMES:
        raise ObsGatherError(f"history_frames must stay {HISTORY_FRAMES}")
    if int(raw["sensor_delay_ticks"]) != 0:
        raise ObsGatherError(
            "sensor_delay_ticks is not 0. Paper S7 is latest-data-wins. "
            "A non-zero delay needs a measured IMU/joint latency from "
            "SPEC_INTAKE — do not invent 40–150 ms."
        )
    if "sensor_delay_ms" in raw and raw["sensor_delay_ms"] not in (0, "REQUIRED_INPUT", None):
        raise ObsGatherError(
            "sensor_delay_ms is set. Do not invent camera/IMU latency; "
            "leave it REQUIRED_INPUT in sim/sensors/calib_real.yaml."
        )
    if str(raw.get("joint_remap")) != "identity":
        raise ObsGatherError("only joint_remap: identity is wired (T800 Native SDK order)")
    if int(raw["n_wrist_dof"]) != 0:
        raise ObsGatherError("n_wrist_dof must stay 0 on T800 (ADR-001 dummy wrists)")
    if int(raw["n_lower_body_dof"]) != T800_N_LOWER_BODY_DOF:
        raise ObsGatherError(f"n_lower_body_dof must stay {T800_N_LOWER_BODY_DOF} (J00–J11)")
    encoder = raw.get("encoder")
    if not isinstance(encoder, dict):
        raise ObsGatherError("encoder: section is required (ADR-042)")
    if int(encoder.get("dimension", -1)) != TOKEN_DIM:
        raise ObsGatherError(f"encoder.dimension must stay {TOKEN_DIM}")
    enabled_names = [
        str(entry["name"])
        for entry in encoder.get("encoder_observations", [])
        if entry.get("enabled", True)
    ]
    for mode in encoder.get("encoder_modes", []):
        name = str(mode.get("name", "")).lower()
        if name == "g1":
            raise ObsGatherError("encoder_modes must not use the G1 mode name on T800")
        if name == "smpl":
            raise ObsGatherError(
                "encoder_modes must not use the SMPL mode on T800 "
                "(needs smpl_joint.csv and G1 wrist DoF)"
            )
        if name not in ("t800", "teleop"):
            raise ObsGatherError(
                f"encoder mode {name!r} is not wired. Allowed: t800, teleop. "
                "Do not invent a PICO/SMPL mode."
            )
        required = [str(n) for n in mode.get("required_observations", [])]
        missing = [n for n in required if n not in enabled_names]
        if missing:
            raise ObsGatherError(f"encoder mode {name!r} required unknown observations {missing}")
    default_mode = str(raw.get("default_encoder_mode", "t800")).lower()
    mode_names = {str(m.get("name", "")).lower() for m in encoder.get("encoder_modes", [])}
    if default_mode not in mode_names:
        raise ObsGatherError(f"default_encoder_mode {default_mode!r} is not in encoder_modes")
    variant = encoder_variant_of(raw)
    if variant == ENCODER_VARIANT_SONIC_V1_1:
        if raw.get("not_low_latency") is not True:
            raise ValueError("sonic_v1_1 YAML must keep not_low_latency: true")
        if raw.get("not_wrist_pose_augmentation_sampler") is not True:
            raise ValueError(
                "sonic_v1_1 YAML must keep not_wrist_pose_augmentation_sampler: true. "
                "Wrist-pose augmentation is training-time; do not invent a sampler."
            )
    steps, _frames, four = motion_history_horizons(enabled_names)
    heading_ori = [n for n in enabled_names if is_heading_ori_name(n)]
    full_ori = [n for n in enabled_names if is_full_ori_name(n)]
    if four:
        raise ObsGatherError(
            f"{four} is the official SMPL/wrist 4frame_step1 horizon. "
            "T800 t800/teleop modes use 10frame_step1. SMPL and wrists stay refused."
        )
    if len(steps) > 1:
        raise ObsGatherError(
            f"encoder_observations mix look-ahead steps {sorted(steps)}. "
            "Official default and sonic_v1_1 g1/teleop are 10frame_step5; "
            "official low_latency g1/teleop is 10frame_step1. "
            "Official v1.1 SMPL/wrist 10frame_step1 is refused on T800. "
            "Do not concatenate the YAMLs."
        )
    if variant == ENCODER_VARIANT_DEFAULT and 1 in steps:
        raise ObsGatherError(
            "default encoder_variant cannot list *_step1 motion names. "
            "Use wbc/obs_gather_low_latency.yaml (ADR-044)."
        )
    if variant == ENCODER_VARIANT_LOW_LATENCY and 5 in steps:
        raise ObsGatherError(
            "low_latency encoder_variant cannot list *_step5 motion names. "
            "Default 10frame_step5 stays in wbc/obs_gather.yaml (ADR-043). "
            "SONIC v1.1 heading step5 stays in wbc/obs_gather_sonic_v1_1.yaml (ADR-045)."
        )
    if variant == ENCODER_VARIANT_SONIC_V1_1 and 1 in steps:
        raise ObsGatherError(
            "sonic_v1_1 g1/teleop names are *_10frame_step5. "
            "Official SMPL/wrist 10frame_step1 stays refused on T800."
        )
    if variant == ENCODER_VARIANT_DEFAULT and heading_ori:
        raise ObsGatherError(
            f"{heading_ori} is the SONIC v1.1 heading-normalized ori. "
            "Use wbc/obs_gather_sonic_v1_1.yaml (ADR-045)."
        )
    if variant == ENCODER_VARIANT_LOW_LATENCY and heading_ori:
        raise ObsGatherError(
            f"{heading_ori} is SONIC v1.1 heading ori. "
            "low_latency uses full motion_anchor_orientation_*_step1 (ADR-044)."
        )
    if variant == ENCODER_VARIANT_SONIC_V1_1 and full_ori:
        raise ObsGatherError(
            f"{full_ori} is the default/full relative ori. "
            "sonic_v1_1 must use motion_anchor_orientation_heading_*."
        )
    if variant == ENCODER_VARIANT_SONIC_V1_1 and not heading_ori:
        raise ObsGatherError(
            "sonic_v1_1 must list motion_anchor_orientation_heading "
            "(robot-heading-normalized targets, ADR-045)."
        )
    z_names = [n for n in enabled_names if "root_z" in n]
    if variant in (ENCODER_VARIANT_LOW_LATENCY, ENCODER_VARIANT_SONIC_V1_1) and z_names:
        raise ObsGatherError(
            f"{z_names} are omitted from official {variant}/observation_config.yaml. "
            "Do not carry motion_root_z_* into this layout."
        )
    expected_enc = int(raw["expected_encoder_dim"])
    g1_enc = int(raw["g1_encoder_dim_forbidden"])
    if g1_enc != G1_ENCODER_MOTION_DIM:
        raise ObsGatherError(f"g1_encoder_dim_forbidden drifted from {G1_ENCODER_MOTION_DIM}")
    g1_onnx = int(raw["g1_encoder_onnx_dim_forbidden"])
    include_z = variant == ENCODER_VARIANT_DEFAULT
    window = encoder_motion_dim(int(raw["n_dof"]), include_root_z=include_z)
    if int(raw.get("encoder_motion_window_dim", window)) != window:
        raise ObsGatherError(
            f"encoder_motion_window_dim {raw.get('encoder_motion_window_dim')} != {window}"
        )
    if variant == ENCODER_VARIANT_DEFAULT:
        t800_enc = t800_encoder_onnx_dim(n_dof=int(raw["n_dof"]))
        if g1_onnx != G1_ENCODER_ONNX_DIM:
            raise ObsGatherError(f"g1_encoder_onnx_dim_forbidden drifted from {G1_ENCODER_ONNX_DIM}")
    elif variant == ENCODER_VARIANT_LOW_LATENCY:
        t800_enc = t800_encoder_onnx_dim_low_latency(n_dof=int(raw["n_dof"]))
        if g1_onnx != G1_ENCODER_ONNX_DIM_LOW_LATENCY:
            raise ObsGatherError(
                f"g1_encoder_onnx_dim_forbidden drifted from {G1_ENCODER_ONNX_DIM_LOW_LATENCY}"
            )
    else:
        t800_enc = t800_encoder_onnx_dim_v1_1(n_dof=int(raw["n_dof"]))
        if g1_onnx != G1_ENCODER_ONNX_DIM_V1_1:
            raise ObsGatherError(
                f"g1_encoder_onnx_dim_forbidden drifted from {G1_ENCODER_ONNX_DIM_V1_1}"
            )
    if expected_enc != t800_enc:
        raise ObsGatherError(f"expected_encoder_dim {expected_enc} != T800 {t800_enc}")
    _refuse_g1_encoder_dims(expected_enc)
    return raw


def parse_history_name(name: str) -> tuple[str, int, int] | None:
    match = _HISTORY_NAME.match(name)
    if match is None:
        return None
    return match.group("base"), int(match.group("n")), int(match.group("s"))


def encoder_variant_of(cfg: dict[str, Any]) -> str:
    raw = (
        str(cfg.get("encoder_variant", ENCODER_VARIANT_DEFAULT))
        .lower()
        .replace("-", "_")
        .replace(".", "_")
    )
    if raw in ("default", "step5", "release"):
        return ENCODER_VARIANT_DEFAULT
    if raw in ("low_latency", "lowlatency", "step1"):
        return ENCODER_VARIANT_LOW_LATENCY
    if raw in ("sonic_v1_1", "v1_1"):
        return ENCODER_VARIANT_SONIC_V1_1
    raise ObsGatherError(
        f"encoder_variant {raw!r} is not wired. "
        "Allowed: default, low_latency, sonic_v1_1. "
        "Do not mix heading ori into the default YAML or step1 into v1.1."
    )


def is_heading_ori_name(name: str) -> bool:
    """Official sonic_v1_1 heading-normalized anchor orientation."""
    return "anchor_orientation_heading" in name or "anchor_orientation_refheading" in name


def is_full_ori_name(name: str) -> bool:
    """Default / low-latency full relative orientation (R_robot.T @ R_ref)."""
    return "anchor_orientation" in name and not is_heading_ori_name(name)


def motion_history_horizons(names: list[str]) -> tuple[set[int], set[int], list[str]]:
    """Return (steps, n_frames, four_frame_names) for motion_* history observations."""
    steps: set[int] = set()
    frames: set[int] = set()
    four: list[str] = []
    for name in names:
        parsed = parse_history_name(name)
        if parsed is None:
            continue
        base, n_frames, step = parsed
        if not base.startswith("motion_"):
            continue
        if n_frames == SMPL_LOW_LATENCY_FRAMES:
            four.append(name)
        steps.add(step)
        frames.add(n_frames)
    return steps, frames, four


def _refuse_g1_encoder_dims(dim: int) -> None:
    if dim in _G1_ENCODER_DIMS_FORBIDDEN:
        refuse_g1_checkpoint(n_dof=G1_N_DOF)


def _refuse_hand_name(name: str) -> None:
    lowered = name.lower()
    if any(bit in lowered for bit in _FORBIDDEN_NAME_BITS):
        raise ObsGatherError(
            f"observation {name!r} looks like a DexHand2 channel; hands bypass WBC"
        )


def history_block_dim(base: str, n_frames: int, n_dof: int) -> int:
    if base in ("his_base_angular_velocity", "his_gravity_dir"):
        return 3 * n_frames
    if base in (
        "his_body_joint_positions",
        "his_body_joint_velocities",
        "his_last_actions",
        "motion_joint_positions",
        "motion_joint_velocities",
    ):
        return int(n_dof) * n_frames
    if base in (
        "motion_joint_positions_lowerbody",
        "motion_joint_velocities_lowerbody",
    ):
        return T800_N_LOWER_BODY_DOF * n_frames
    if base in (
        "motion_anchor_orientation",
        "motion_anchor_orientation_heading",
        "motion_anchor_orientation_refheading",
    ):
        return ANCHOR_ORI_DIM * n_frames
    if base in ("motion_root_z_position",):
        return ROOT_Z_DIM * n_frames
    raise ObsGatherError(f"unknown history observation {base!r}")


def single_frame_dim(name: str, n_dof: int, token_dim: int) -> int:
    if name == "token_state":
        return int(token_dim)
    if name in ("base_angular_velocity", "gravity_dir"):
        return 3
    if name in ("body_joint_positions", "body_joint_velocities", "last_actions"):
        return int(n_dof)
    if name in ("motion_joint_positions", "motion_joint_velocities"):
        return int(n_dof)
    if name in ("motion_joint_positions_lowerbody", "motion_joint_velocities_lowerbody"):
        return T800_N_LOWER_BODY_DOF
    if name in (
        "motion_anchor_orientation",
        "motion_anchor_orientation_heading",
        "motion_anchor_orientation_refheading",
    ):
        return ANCHOR_ORI_DIM
    if name == "motion_root_z_position":
        return ROOT_Z_DIM
    if name == "encoder_mode":
        return 3
    if name == "encoder_mode_4":
        return 4
    if name == "vr_3point_local_target":
        return 9
    if name == "vr_3point_local_orn_target":
        return 12
    raise ObsGatherError(f"unknown observation {name!r}")


class Vr3PointHold:
    """Latest VR 3-point pose from command_schema_v1. Not a PICO SDK.

    Official teleop encoder required_observations include
    ``vr_3point_local_target`` (9) and ``vr_3point_local_orn_target`` (12).
    Empty hold raises in teleop mode — do not invent a headset pose.
    """

    def __init__(self) -> None:
        self._pos: np.ndarray | None = None
        self._orn: np.ndarray | None = None

    def push(self, pos9: np.ndarray, orn12: np.ndarray) -> None:
        pos = np.asarray(pos9, dtype=np.float64).reshape(9)
        orn = np.asarray(orn12, dtype=np.float64).reshape(12)
        if not np.isfinite(pos).all() or not np.isfinite(orn).all():
            raise ObsGatherError("vr_3point contains NaN/Inf")
        self._pos = pos.copy()
        self._orn = orn.copy()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pos is None or self._orn is None:
            raise MotionRefError(
                "vr_3point hold is empty. Push command_schema_v1 via "
                "push_vr_3point. Do not invent a PICO / CloudXR pose."
            )
        return self._pos, self._orn


class HardwareHold:
    """500 Hz (or any rate) latest-data-wins hardware snapshot. No delay model."""

    def __init__(self) -> None:
        self._snap: HardwareSnapshot | None = None

    def push(self, snap: HardwareSnapshot) -> None:
        if not np.isfinite(snap.t_s):
            raise ObsGatherError("t_s must be finite")
        self._snap = snap

    def push_last_action(self, last_action: np.ndarray, *, n_dof: int = 25) -> None:
        """Overwrite last_action on the current snapshot. Does not invent q/dq/IMU.

        ``last_action`` is the previous 25-D policy output (PPO / decoder).
        Do not copy planner clip qpos or a fake decoder ONNX vector.
        """
        if self._snap is None:
            raise ObsGatherError("hardware hold is empty; push a snapshot first")
        a = remap_identity(last_action, n_dof)
        if not np.isfinite(a).all():
            raise ObsGatherError("last_action contains NaN/Inf")
        prev = self._snap
        self._snap = HardwareSnapshot(
            t_s=prev.t_s,
            q_hw=np.asarray(prev.q_hw, dtype=np.float64).copy(),
            dq_hw=np.asarray(prev.dq_hw, dtype=np.float64).copy(),
            omega_imu=np.asarray(prev.omega_imu, dtype=np.float64).copy(),
            imu_quat_wxyz=np.asarray(prev.imu_quat_wxyz, dtype=np.float64).copy(),
            last_action=a,
        )

    def push_joints(
        self,
        q_hw: np.ndarray,
        dq_hw: np.ndarray,
        *,
        n_dof: int = 25,
        t_s: float | None = None,
    ) -> None:
        """Overwrite q/dq after a 500 Hz physics period. Does not invent IMU.

        Call *after* this tick's decoder assemble (ADR-056). Keeps omega/quat
        and last_action from the previous snapshot. Measured IMU latency
        stays REQUIRED_INPUT.
        """
        if self._snap is None:
            raise ObsGatherError("hardware hold is empty; push a snapshot first")
        q = remap_identity(q_hw, n_dof)
        dq = remap_identity(dq_hw, n_dof)
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ObsGatherError("q/dq contain NaN/Inf")
        prev = self._snap
        stamp = float(prev.t_s if t_s is None else t_s)
        if not np.isfinite(stamp):
            raise ObsGatherError("t_s must be finite")
        self._snap = HardwareSnapshot(
            t_s=stamp,
            q_hw=q,
            dq_hw=dq,
            omega_imu=np.asarray(prev.omega_imu, dtype=np.float64).copy(),
            imu_quat_wxyz=np.asarray(prev.imu_quat_wxyz, dtype=np.float64).copy(),
            last_action=np.asarray(prev.last_action, dtype=np.float64).copy(),
        )

    def read(self) -> HardwareSnapshot:
        if self._snap is None:
            raise ObsGatherError("hardware hold is empty; push a snapshot first")
        return self._snap


class StateLogger:
    """Fixed-capacity 50 Hz ring + optional per-signal CSV (paper S7)."""

    def __init__(self, *, capacity: int, n_dof: int, csv_dir: Path | None = None) -> None:
        if capacity < HISTORY_FRAMES:
            raise ObsGatherError(f"ring capacity {capacity} < history_frames {HISTORY_FRAMES}")
        self.capacity = int(capacity)
        self.n_dof = int(n_dof)
        self._buf: list[PolicySnapshot] = []
        self._csv_dir = csv_dir
        self._csv_files: dict[str, Any] | None = None
        if csv_dir is not None:
            csv_dir.mkdir(parents=True, exist_ok=True)
            widths = {"q": n_dof, "dq": n_dof, "omega": 3}
            self._csv_files = {}
            for key, width in widths.items():
                handle = open(csv_dir / f"{key}.csv", "w", newline="", encoding="utf-8")
                csv.writer(handle).writerow(["t_s"] + [f"i{i}" for i in range(width)])
                self._csv_files[key] = handle

    def close(self) -> None:
        if self._csv_files is None:
            return
        for handle in self._csv_files.values():
            handle.close()
        self._csv_files = None

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, snap: PolicySnapshot) -> None:
        if snap.q_rad.shape != (self.n_dof,):
            raise ObsGatherError(f"logger q dim {snap.q_rad.shape} != {self.n_dof}")
        self._buf.append(snap)
        if len(self._buf) > self.capacity:
            self._buf = self._buf[-self.capacity :]
        if self._csv_files is not None:
            csv.writer(self._csv_files["q"]).writerow([snap.t_s, *snap.q_rad.tolist()])
            csv.writer(self._csv_files["dq"]).writerow([snap.t_s, *snap.dq_rad_s.tolist()])
            csv.writer(self._csv_files["omega"]).writerow([snap.t_s, *snap.omega_heading.tolist()])

    def history(self, n_frames: int, step: int) -> list[PolicySnapshot | None]:
        """Oldest-first strided lookback. Missing ticks are None (caller zeros)."""
        if n_frames < 1 or step < 1:
            raise ObsGatherError("n_frames and step must be >= 1")
        out: list[PolicySnapshot | None] = []
        n = len(self._buf)
        for i in range(n_frames):
            idx = n - 1 - (n_frames - 1 - i) * step
            if idx < 0 or idx >= n:
                out.append(None)
            else:
                out.append(self._buf[idx])
        return out


def remap_identity(vec: np.ndarray, n_dof: int) -> np.ndarray:
    q = np.asarray(vec, dtype=np.float64).reshape(-1)
    if q.shape == (G1_N_DOF,):
        refuse_g1_checkpoint(n_dof=G1_N_DOF)
    if q.shape != (n_dof,):
        raise ObsGatherError(
            f"joint vector dim {q.shape} != T800 {n_dof} "
            "(hands bypass WBC; do not concatenate DexHand2 q)"
        )
    return q.copy()


def to_policy_snapshot(hw: HardwareSnapshot, n_dof: int) -> PolicySnapshot:
    q = remap_identity(hw.q_hw, n_dof)
    dq = remap_identity(hw.dq_hw, n_dof)
    a = remap_identity(hw.last_action, n_dof)
    quat = np.asarray(hw.imu_quat_wxyz, dtype=np.float64).reshape(4)
    yaw = yaw_from_quat_wxyz(quat)
    omega = heading_angular_velocity(hw.omega_imu, yaw)
    # Heading-frame gravity from IMU yaw (SONIC v1.1). Full-quat body gravity
    # is recorded via the same yaw-only contract — do not invent IMU bias.
    grav = heading_gravity(yaw)
    _ = quaternion_wxyz_to_matrix(quat)  # validate finite unit quat
    return PolicySnapshot(
        t_s=float(hw.t_s),
        q_rad=q,
        dq_rad_s=dq,
        omega_heading=omega,
        gravity_heading=grav,
        last_action=a,
    )


def grouped_history_to_interleaved(obs: np.ndarray, n_dof: int, *, token_dim: int = TOKEN_DIM) -> np.ndarray:
    """YAML grouped 874-D → ProprioHistory interleaved 874-D. Token unchanged."""
    n = int(n_dof)
    vec = np.asarray(obs, dtype=np.float64).reshape(-1)
    expected = decoder_history_dim(n, token_dim=token_dim)
    if vec.shape != (expected,):
        raise ObsGatherError(f"grouped dim {vec.shape} != {expected}")
    token = vec[:token_dim]
    rest = vec[token_dim:]
    n_hist = n * HISTORY_FRAMES
    cuts = [3 * HISTORY_FRAMES, 3 * HISTORY_FRAMES + n_hist, 3 * HISTORY_FRAMES + 2 * n_hist, 3 * HISTORY_FRAMES + 3 * n_hist]
    w, q, dq, a, g = np.split(rest, cuts)
    omega = w.reshape(HISTORY_FRAMES, 3)
    qj = q.reshape(HISTORY_FRAMES, n)
    dqj = dq.reshape(HISTORY_FRAMES, n)
    aj = a.reshape(HISTORY_FRAMES, n)
    grav = g.reshape(HISTORY_FRAMES, 3)
    frames = [np.concatenate([omega[i], qj[i], dqj[i], aj[i], grav[i]]) for i in range(HISTORY_FRAMES)]
    return np.concatenate([token, *frames])


def interleaved_history_to_grouped(obs: np.ndarray, n_dof: int, *, token_dim: int = TOKEN_DIM) -> np.ndarray:
    """ProprioHistory interleaved 874-D → YAML grouped 874-D."""
    n = int(n_dof)
    vec = np.asarray(obs, dtype=np.float64).reshape(-1)
    expected = decoder_history_dim(n, token_dim=token_dim)
    if vec.shape != (expected,):
        raise ObsGatherError(f"interleaved dim {vec.shape} != {expected}")
    token = vec[:token_dim]
    step = 3 + 3 * n + 3
    frames = vec[token_dim:].reshape(HISTORY_FRAMES, step)
    omega = frames[:, :3].reshape(-1)
    qj = frames[:, 3 : 3 + n].reshape(-1)
    dqj = frames[:, 3 + n : 3 + 2 * n].reshape(-1)
    aj = frames[:, 3 + 2 * n : 3 + 3 * n].reshape(-1)
    grav = frames[:, 3 + 3 * n :].reshape(-1)
    return np.concatenate([token, omega, qj, dqj, aj, grav])


@dataclass
class ObsGather:
    """50 Hz control-loop assembler. HardwareHold may run at 500 Hz."""

    cfg: dict[str, Any] = field(default_factory=load_gather_cfg)
    hw: HardwareHold = field(default_factory=HardwareHold)
    vr: Vr3PointHold = field(default_factory=Vr3PointHold)
    motion: MotionHold = field(init=False)
    logger: StateLogger = field(init=False)
    slots: list[ObsSlot] = field(init=False)
    encoder_slots: list[ObsSlot] = field(init=False)
    total_dim: int = field(init=False)
    encoder_dim: int = field(init=False)
    encoder_mode_name: str = field(init=False)
    token: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(self.cfg["n_dof"])
        csv_dir = None
        self.motion = MotionHold(n_dof=n)
        self.logger = StateLogger(
            capacity=int(self.cfg["ring_capacity"]),
            n_dof=n,
            csv_dir=csv_dir,
        )
        self.slots, self.total_dim = compile_observations(self.cfg)
        self.encoder_slots, self.encoder_dim = compile_encoder_observations(self.cfg)
        self.encoder_mode_name = str(self.cfg.get("default_encoder_mode", "t800")).lower()
        if self.total_dim == G1_DECODER_INPUT_DIM:
            refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
        _refuse_g1_encoder_dims(self.encoder_dim)
        expected = int(self.cfg["expected_total_dim"])
        if self.total_dim != expected:
            raise ObsGatherError(f"compiled dim {self.total_dim} != expected {expected}")
        contract = decoder_history_dim(n)
        if self.total_dim != contract:
            raise ObsGatherError(f"compiled dim {self.total_dim} != decoder_history_dim {contract}")
        if self.encoder_dim != int(self.cfg["expected_encoder_dim"]):
            raise ObsGatherError(
                f"compiled encoder dim {self.encoder_dim} != {self.cfg['expected_encoder_dim']}"
            )

    def push_hw(self, snap: HardwareSnapshot) -> None:
        self.hw.push(snap)

    def push_joints(
        self,
        q_hw: np.ndarray,
        dq_hw: np.ndarray,
        *,
        n_dof: int = 25,
        t_s: float | None = None,
    ) -> None:
        """Post-physics q/dq into HardwareHold. Does not invent IMU."""
        self.hw.push_joints(q_hw, dq_hw, n_dof=n_dof, t_s=t_s)

    def push_motion(self, frames, *, cursor: int = 0) -> None:
        self.motion.push_sequence(frames, cursor=cursor)

    def push_vr_3point(self, cmd: CommandVector | FivePointCommand) -> None:
        """Pack VR 3-point from command_schema. Five-point is refused here."""
        if isinstance(cmd, FivePointCommand):
            raise ObsGatherError(
                "encoder teleop mode is official 3-point "
                "(vr_3point_local_target + vr_3point_local_orn_target). "
                "5-point elbows do not enter this encoder list."
            )
        pos, orn = command_to_vr_3point(cmd)
        self.vr.push(pos, orn)

    def set_encoder_mode(self, name: str) -> None:
        key = str(name).lower().replace("-", "_")
        if key in ("3point", "vr_3point"):
            key = "teleop"
        if key in ("g1", "smpl"):
            raise ObsGatherError(f"encoder mode {name!r} is refused on T800")
        names = {str(m.get("name", "")).lower() for m in self.cfg.get("encoder", {}).get("encoder_modes", [])}
        if key not in names:
            raise ObsGatherError(f"encoder mode {name!r} is not in encoder_modes")
        self.encoder_mode_name = key

    def encoder_mode_id(self) -> int:
        for mode in self.cfg.get("encoder", {}).get("encoder_modes", []):
            if str(mode.get("name", "")).lower() == self.encoder_mode_name:
                return int(mode.get("mode_id", 0))
        raise ObsGatherError(f"encoder mode {self.encoder_mode_name!r} has no mode_id")

    def required_encoder_observations(self) -> list[str]:
        for mode in self.cfg.get("encoder", {}).get("encoder_modes", []):
            if str(mode.get("name", "")).lower() == self.encoder_mode_name:
                return [str(n) for n in mode.get("required_observations", [])]
        raise ObsGatherError(f"encoder mode {self.encoder_mode_name!r} is missing required_observations")

    def control_tick(self, token: np.ndarray, *, t_s: float | None = None) -> np.ndarray:
        """Copy latest hardware into the 50 Hz ring and assemble the decoder vector."""
        hw = self.hw.read()
        snap = to_policy_snapshot(hw, int(self.cfg["n_dof"]))
        if t_s is not None:
            snap.t_s = float(t_s)
        self.logger.push(snap)
        return self.assemble(token)

    def assemble(self, token: np.ndarray) -> np.ndarray:
        tok = np.asarray(token, dtype=np.float64).reshape(-1)
        if tok.shape != (int(self.cfg["token_dim"]),):
            raise ObsGatherError(f"token dim {tok.shape} != {self.cfg['token_dim']}")
        self.token = tok
        out = np.zeros(self.total_dim, dtype=np.float64)
        for slot in self.slots:
            slot.gather(self, out, slot.offset)
        return out

    def assemble_encoder(self, mode: str | None = None) -> np.ndarray:
        """Encoder INPUT (842-D default / 831-D low-latency / 831-D v1.1 on T800).

        Does not run G1 ONNX. Official multi-mode layout: concatenate the YAML
        superset, zero-fill observations that are not in the active mode's
        required_observations. Default YAML is 10frame_step5 full ori (ADR-043).
        low_latency YAML is 10frame_step1 with no root_z (ADR-044). sonic_v1_1
        YAML is 10frame_step5 heading ori with no root_z (ADR-045). SMPL
        4frame_step1 and v1.1 SMPL/wrist 10frame_step1 stay refused. The two
        831-D layouts are not interchangeable.
        """
        if mode is not None:
            self.set_encoder_mode(mode)
        required = set(self.required_encoder_observations())
        out = np.zeros(self.encoder_dim, dtype=np.float64)
        try:
            for slot in self.encoder_slots:
                if slot.name in required:
                    slot.gather(self, out, slot.offset)
        except MotionRefError:
            raise
        return out


def _fill_token(gather: ObsGather, buf: np.ndarray, offset: int) -> None:
    if gather.token is None:
        raise ObsGatherError("token_state requested before assemble()")
    dim = int(gather.cfg["token_dim"])
    buf[offset : offset + dim] = gather.token


def _fill_history(
    gather: ObsGather,
    buf: np.ndarray,
    offset: int,
    *,
    field: str,
    n_frames: int,
    step: int,
) -> None:
    n_dof = int(gather.cfg["n_dof"])
    hist = gather.logger.history(n_frames, step)
    cursor = offset
    for snap in hist:
        if field in ("omega_heading", "gravity_heading"):
            width = 3
            piece = np.zeros(3, dtype=np.float64) if snap is None else getattr(snap, field)
        else:
            width = n_dof
            piece = np.zeros(n_dof, dtype=np.float64) if snap is None else getattr(snap, field)
        buf[cursor : cursor + width] = piece
        cursor += width


def _make_history_fn(base: str, n_frames: int, step: int) -> GatherFn:
    field = {
        "his_base_angular_velocity": "omega_heading",
        "his_body_joint_positions": "q_rad",
        "his_body_joint_velocities": "dq_rad_s",
        "his_last_actions": "last_action",
        "his_gravity_dir": "gravity_heading",
    }.get(base)
    if field is None:
        raise ObsGatherError(f"history base {base!r} is not a robot-state observation")

    def _fn(gather: ObsGather, buf: np.ndarray, offset: int) -> None:
        _fill_history(gather, buf, offset, field=field, n_frames=n_frames, step=step)

    return _fn


def _make_current_fn(name: str) -> GatherFn:
    field = {
        "base_angular_velocity": "omega_heading",
        "body_joint_positions": "q_rad",
        "body_joint_velocities": "dq_rad_s",
        "last_actions": "last_action",
        "gravity_dir": "gravity_heading",
    }.get(name)
    if field is None:
        raise ObsGatherError(f"current-frame observation {name!r} is not wired")

    def _fn(gather: ObsGather, buf: np.ndarray, offset: int) -> None:
        hist = gather.logger.history(1, 1)
        snap = hist[0]
        n_dof = int(gather.cfg["n_dof"])
        width = 3 if field in ("omega_heading", "gravity_heading") else n_dof
        piece = np.zeros(width, dtype=np.float64) if snap is None else getattr(snap, field)
        buf[offset : offset + width] = piece

    return _fn


def _anchor_mode(base: str) -> str:
    if base.endswith("_refheading"):
        return "refheading"
    if base.endswith("_heading"):
        return "heading"
    return "full"


def _robot_quat_wxyz(gather: ObsGather) -> np.ndarray:
    try:
        hw = gather.hw.read()
    except ObsGatherError as exc:
        raise MotionRefError(
            "encoder anchor orientation needs the latest IMU quaternion; push hardware first"
        ) from exc
    return np.asarray(hw.imu_quat_wxyz, dtype=np.float64).reshape(4)


def _fill_motion(
    gather: ObsGather,
    buf: np.ndarray,
    offset: int,
    *,
    base: str,
    n_frames: int,
    step: int,
) -> None:
    window = gather.motion.look_ahead(n_frames, step)
    n_dof = int(gather.cfg["n_dof"])
    cursor = offset
    robot_quat = None
    if "anchor_orientation" in base:
        robot_quat = _robot_quat_wxyz(gather)
    refheading = window[0].root_rot_wxyz if window else None
    for frame in window:
        if base == "motion_joint_positions":
            piece = frame.q_ref_rad
        elif base == "motion_joint_velocities":
            piece = frame.dq_ref_rad_s
        elif base == "motion_joint_positions_lowerbody":
            piece = lower_body_slice(frame.q_ref_rad)
        elif base == "motion_joint_velocities_lowerbody":
            piece = lower_body_slice(frame.dq_ref_rad_s)
        elif base == "motion_root_z_position":
            piece = np.array([frame.root_pos_m[2]], dtype=np.float64)
        elif "anchor_orientation" in base:
            piece = heading_corrected_rel_rot6d(
                robot_quat,
                frame.root_rot_wxyz,
                mode=_anchor_mode(base),
                refheading_quat_wxyz=refheading,
            )
        else:
            raise ObsGatherError(f"motion base {base!r} is not wired")
        width = int(piece.shape[0])
        buf[cursor : cursor + width] = piece
        cursor += width
    expected = history_block_dim(base, n_frames, n_dof)
    if cursor - offset != expected:
        raise ObsGatherError(f"motion fill wrote {cursor - offset} != {expected} for {base}")


def _make_motion_fn(base: str, n_frames: int, step: int) -> GatherFn:
    def _fn(gather: ObsGather, buf: np.ndarray, offset: int) -> None:
        _fill_motion(gather, buf, offset, base=base, n_frames=n_frames, step=step)

    return _fn


def _make_motion_current_fn(name: str) -> GatherFn:
    def _fn(gather: ObsGather, buf: np.ndarray, offset: int) -> None:
        _fill_motion(gather, buf, offset, base=name, n_frames=1, step=1)

    return _fn


def _make_encoder_mode_fn(dim: int) -> GatherFn:
    def _fn(gather: ObsGather, buf: np.ndarray, offset: int) -> None:
        buf[offset] = float(gather.encoder_mode_id())
        buf[offset + 1 : offset + dim] = 0.0

    return _fn


def _make_vr_fn(name: str) -> GatherFn:
    def _fn(gather: ObsGather, buf: np.ndarray, offset: int) -> None:
        pos, orn = gather.vr.read()
        if name == "vr_3point_local_target":
            buf[offset : offset + 9] = pos
        elif name == "vr_3point_local_orn_target":
            buf[offset : offset + 12] = orn
        else:
            raise ObsGatherError(f"vr observation {name!r} is not wired")

    return _fn


def _slot_for_name(name: str, cfg: dict[str, Any], offset: int) -> ObsSlot:
    n_dof = int(cfg["n_dof"])
    token_dim = int(cfg["token_dim"])
    _refuse_hand_name(name)
    try:
        refuse_wrist_observation(name)
        refuse_smpl_observation(name)
    except MotionRefError as exc:
        raise ObsGatherError(str(exc)) from exc
    if "vr_5point" in name.lower() or name.lower().startswith("vr_5"):
        raise ObsGatherError(
            f"observation {name!r} is 5-point. Official teleop encoder is 3-point "
            "(vr_3point_local_target + vr_3point_local_orn_target)."
        )
    parsed = parse_history_name(name)
    if parsed is not None and parsed[0].startswith("motion_") and parsed[1] == SMPL_LOW_LATENCY_FRAMES:
        raise ObsGatherError(
            f"observation {name!r} is the official SMPL/wrist 4frame_step1 horizon. "
            "T800 t800/teleop modes use 10frame. Do not pack a 4-frame body window."
        )
    if name == "token_state":
        dim = single_frame_dim(name, n_dof, token_dim)
        fn: GatherFn = _fill_token
    elif name in ("encoder_mode", "encoder_mode_4"):
        dim = single_frame_dim(name, n_dof, token_dim)
        fn = _make_encoder_mode_fn(dim)
    elif name in ("vr_3point_local_target", "vr_3point_local_orn_target"):
        dim = single_frame_dim(name, n_dof, token_dim)
        fn = _make_vr_fn(name)
    elif parsed is not None:
        base, n_frames, step = parsed
        dim = history_block_dim(base, n_frames, n_dof)
        if base.startswith("motion_"):
            fn = _make_motion_fn(base, n_frames, step)
        else:
            fn = _make_history_fn(base, n_frames, step)
    elif name.startswith("motion_"):
        dim = single_frame_dim(name, n_dof, token_dim)
        fn = _make_motion_current_fn(name)
    else:
        dim = single_frame_dim(name, n_dof, token_dim)
        fn = _make_current_fn(name)
    return ObsSlot(name=name, offset=offset, dim=dim, gather=fn)


def compile_observations(cfg: dict[str, Any] | None = None) -> tuple[list[ObsSlot], int]:
    """Paper S7: match YAML names to the registry and precompute (fn, offset, dim)."""
    cfg = cfg if cfg is not None else load_gather_cfg()
    slots: list[ObsSlot] = []
    offset = 0
    for entry in cfg["observations"]:
        if not entry.get("enabled", True):
            continue
        slot = _slot_for_name(str(entry["name"]), cfg, offset)
        slots.append(slot)
        offset += slot.dim
    if offset == G1_DECODER_INPUT_DIM:
        refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
    return slots, offset


def compile_encoder_observations(cfg: dict[str, Any] | None = None) -> tuple[list[ObsSlot], int]:
    """Official encoder_observations SUPERSET. T800 842-D default / 831-D
    low-latency / 831-D sonic_v1_1 (heading). G1 1751 / 1247 / 650 / 640 refused.
    """
    cfg = cfg if cfg is not None else load_gather_cfg()
    encoder = cfg["encoder"]
    slots: list[ObsSlot] = []
    offset = 0
    for entry in encoder["encoder_observations"]:
        if not entry.get("enabled", True):
            continue
        slot = _slot_for_name(str(entry["name"]), cfg, offset)
        slots.append(slot)
        offset += slot.dim
    _refuse_g1_encoder_dims(offset)
    expected = int(cfg["expected_encoder_dim"])
    if offset != expected:
        raise ObsGatherError(f"compiled encoder dim {offset} != expected {expected}")
    return slots, offset
