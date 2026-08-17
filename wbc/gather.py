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
until SPEC_INTAKE measures one). Not Table S4. Not pad–cardboard.
Not a PICO SDK.
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

from vla.adapters.frame_transform import yaw_from_quat_wxyz
from vla.adapters.rotation import quaternion_wxyz_to_matrix
from wbc.checkpoint import refuse_g1_checkpoint
from wbc.dims import (
    G1_DECODER_INPUT_DIM,
    G1_N_DOF,
    HISTORY_FRAMES,
    TOKEN_DIM,
    decoder_history_dim,
    load_t800_sonic,
)
from wbc.observation import heading_angular_velocity, heading_gravity
from wbc.stream import OPERATOR_INPUT_HZ, PLANNER_HZ, POLICY_HZ, STREAM_HZ

GATHER_YAML = Path(__file__).with_name("obs_gather.yaml")
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
    return raw


def parse_history_name(name: str) -> tuple[str, int, int] | None:
    match = _HISTORY_NAME.match(name)
    if match is None:
        return None
    return match.group("base"), int(match.group("n")), int(match.group("s"))


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
    raise ObsGatherError(f"unknown history observation {base!r}")


def single_frame_dim(name: str, n_dof: int, token_dim: int) -> int:
    if name == "token_state":
        return int(token_dim)
    if name in ("base_angular_velocity", "gravity_dir"):
        return 3
    if name in ("body_joint_positions", "body_joint_velocities", "last_actions"):
        return int(n_dof)
    if name == "encoder_mode_4":
        return 4
    if name == "vr_3point_local_target":
        return 9
    if name == "vr_3point_local_orn_target":
        return 12
    raise ObsGatherError(f"unknown observation {name!r}")


class HardwareHold:
    """500 Hz (or any rate) latest-data-wins hardware snapshot. No delay model."""

    def __init__(self) -> None:
        self._snap: HardwareSnapshot | None = None

    def push(self, snap: HardwareSnapshot) -> None:
        if not np.isfinite(snap.t_s):
            raise ObsGatherError("t_s must be finite")
        self._snap = snap

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
    logger: StateLogger = field(init=False)
    slots: list[ObsSlot] = field(init=False)
    total_dim: int = field(init=False)
    token: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(self.cfg["n_dof"])
        csv_dir = None
        self.logger = StateLogger(
            capacity=int(self.cfg["ring_capacity"]),
            n_dof=n,
            csv_dir=csv_dir,
        )
        self.slots, self.total_dim = compile_observations(self.cfg)
        if self.total_dim == G1_DECODER_INPUT_DIM:
            refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
        expected = int(self.cfg["expected_total_dim"])
        if self.total_dim != expected:
            raise ObsGatherError(f"compiled dim {self.total_dim} != expected {expected}")
        contract = decoder_history_dim(n)
        if self.total_dim != contract:
            raise ObsGatherError(f"compiled dim {self.total_dim} != decoder_history_dim {contract}")

    def push_hw(self, snap: HardwareSnapshot) -> None:
        self.hw.push(snap)

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


def compile_observations(cfg: dict[str, Any] | None = None) -> tuple[list[ObsSlot], int]:
    """Paper S7: match YAML names to the registry and precompute (fn, offset, dim)."""
    cfg = cfg if cfg is not None else load_gather_cfg()
    n_dof = int(cfg["n_dof"])
    token_dim = int(cfg["token_dim"])
    slots: list[ObsSlot] = []
    offset = 0
    for entry in cfg["observations"]:
        if not entry.get("enabled", True):
            continue
        name = str(entry["name"])
        _refuse_hand_name(name)
        parsed = parse_history_name(name)
        if name == "token_state":
            dim = single_frame_dim(name, n_dof, token_dim)
            fn: GatherFn = _fill_token
        elif parsed is not None:
            base, n_frames, step = parsed
            dim = history_block_dim(base, n_frames, n_dof)
            fn = _make_history_fn(base, n_frames, step)
        else:
            dim = single_frame_dim(name, n_dof, token_dim)
            fn = _make_current_fn(name)
        slots.append(ObsSlot(name=name, offset=offset, dim=dim, gather=fn))
        offset += dim
    if offset == G1_DECODER_INPUT_DIM:
        refuse_g1_checkpoint(decoder_input_dim=G1_DECODER_INPUT_DIM)
    return slots, offset
