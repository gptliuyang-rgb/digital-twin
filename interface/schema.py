"""Load and validate versioned YAML contracts.

Joint names, dimensions, and frames are read from YAML. Runtime code must not
hard-code them. Incomplete DexHand2 specs raise SpecIncompleteError listing every
REQUIRED_INPUT field — never a silent default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_INPUT_TOKEN = "REQUIRED_INPUT"
HAND_SPEC_PATH = REPO_ROOT / "assets" / "dexhand2" / "meta" / "dexhand2_spec.yaml"
JOINT_MAP_PATH = REPO_ROOT / "assets" / "dexhand2" / "meta" / "joint_name_map.yaml"
COMMAND_SCHEMA_PATH = Path(__file__).with_name("command_schema_v1.yaml")
FIVE_POINT_SCHEMA_PATH = Path(__file__).with_name("command_schema_v1_5point.yaml")
FRAMES_PATH = Path(__file__).with_name("frames.yaml")
COUPLING_PATH = REPO_ROOT / "assets" / "dexhand2" / "meta" / "coupling.yaml"
MOUNT_PATH = REPO_ROOT / "assets" / "dexhand2" / "meta" / "mount_transform.yaml"

# WBC pose block is 32 scalars (see command_schema_v1.yaml).
WBC_DIM = 32
HAND_MODE_DIM = 2
TOOL_DIM = 1


class SpecIncompleteError(RuntimeError):
    """Raised when a spec still contains REQUIRED_INPUT leaves."""

    def __init__(self, missing: Sequence[str], spec_path: Path | None = None) -> None:
        self.missing = list(missing)
        self.spec_path = spec_path
        loc = f" in {spec_path}" if spec_path else ""
        lines = "\n".join(f"  - {item}" for item in self.missing)
        super().__init__(
            f"DexHand2 spec is incomplete{loc}. Missing REQUIRED_INPUT fields:\n{lines}\n"
            "Fill these in assets/dexhand2/meta/*.yaml (see docs/SPEC_INTAKE.md). "
            "Do not invent numbers."
        )


def load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing contract file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_required_inputs(node: Any, prefix: str = "") -> list[str]:
    """Return dotted paths whose value is the REQUIRED_INPUT sentinel."""
    missing: list[str] = []
    if isinstance(node, str) and node.strip() == REQUIRED_INPUT_TOKEN:
        missing.append(prefix or "<root>")
        return missing
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            missing.extend(find_required_inputs(value, path))
        return missing
    if isinstance(node, list):
        for index, value in enumerate(node):
            path = f"{prefix}[{index}]"
            missing.extend(find_required_inputs(value, path))
    return missing


def require_complete(node: Any, spec_path: Path) -> None:
    missing = find_required_inputs(node)
    if missing:
        raise SpecIncompleteError(missing, spec_path=spec_path)


@dataclass(frozen=True)
class JointMapEntry:
    canonical: str
    doc_name: str
    mjcf_joint_template: str
    mjcf_actuator_template: str
    sdk_label: str
    sdk_index: int
    nid: int
    finger: str
    dof_role: str

    def mjcf_joint(self, side: str) -> str:
        return self.mjcf_joint_template.format(side=side)

    def mjcf_actuator(self, side: str) -> str:
        return self.mjcf_actuator_template.format(side=side)


@dataclass
class HandSpec:
    raw: dict[str, Any]
    path: Path
    n_fingers: int
    n_active_dof: int
    n_total_dof: int
    coupling_type: str
    joint_order: list[str]
    joint_limits_rad: dict[str, tuple[float, float]]
    skeleton_mass_kg: float
    product_mass_kg: float
    missing_fields: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields

    def require_complete(self) -> None:
        if self.missing_fields:
            raise SpecIncompleteError(self.missing_fields, spec_path=self.path)

    def require_p0_topology(self) -> None:
        """Topology fields needed to size command vectors. Official values are filled."""
        if self.n_active_dof != len(self.joint_order):
            raise ValueError(
                f"joint_order length {len(self.joint_order)} != n_active_dof {self.n_active_dof}"
            )
        if len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("joint_order contains duplicates")
        if self.n_active_dof != self.n_total_dof and self.coupling_type == "none":
            raise ValueError("coupling_type=none requires n_active_dof == n_total_dof")

    def limits_vector(self) -> np.ndarray:
        lo = np.array([self.joint_limits_rad[name][0] for name in self.joint_order], dtype=np.float64)
        hi = np.array([self.joint_limits_rad[name][1] for name in self.joint_order], dtype=np.float64)
        return np.stack([lo, hi], axis=1)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def _as_limit_map(raw: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for name, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"joint_limits_rad[{name}] must be [lower, upper] rad")
        out[name] = (float(pair[0]), float(pair[1]))
    return out


def load_hand_spec(path: Path | None = None, *, require_complete: bool = False) -> HandSpec:
    spec_path = path or HAND_SPEC_PATH
    raw = load_yaml(spec_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Hand spec must be a mapping: {spec_path}")
    missing = find_required_inputs(raw)
    joint_order = list(raw["joint_order"])
    spec = HandSpec(
        raw=raw,
        path=spec_path,
        n_fingers=int(raw["n_fingers"]),
        n_active_dof=int(raw["n_active_dof"]),
        n_total_dof=int(raw["n_total_dof"]),
        coupling_type=str(raw["coupling_type"]),
        joint_order=joint_order,
        joint_limits_rad=_as_limit_map(raw["joint_limits_rad"]),
        skeleton_mass_kg=float(raw["skeleton_mass_kg"]["value"]),
        product_mass_kg=float(raw["total_mass_kg"]["value"]),
        missing_fields=missing,
    )
    spec.require_p0_topology()
    for name in spec.joint_order:
        if name not in spec.joint_limits_rad:
            raise ValueError(f"joint_limits_rad missing {name}")
    if require_complete:
        spec.require_complete()
    return spec


def load_joint_map(path: Path | None = None) -> list[JointMapEntry]:
    raw = load_yaml(path or JOINT_MAP_PATH)
    entries: list[JointMapEntry] = []
    for item in raw["joints"]:
        entries.append(
            JointMapEntry(
                canonical=item["canonical"],
                doc_name=item["doc_name"],
                mjcf_joint_template=item["mjcf_joint"],
                mjcf_actuator_template=item["mjcf_actuator"],
                sdk_label=item["sdk_label"],
                sdk_index=int(item["sdk_index"]),
                nid=int(item["nid"]),
                finger=item["finger"],
                dof_role=item["dof_role"],
            )
        )
    spec = load_hand_spec()
    canonical = [e.canonical for e in entries]
    if canonical != spec.joint_order:
        raise ValueError(
            "joint_name_map.yaml canonical order does not match dexhand2_spec.joint_order:\n"
            f"  map : {canonical}\n"
            f"  spec: {spec.joint_order}"
        )
    indices = [e.sdk_index for e in entries]
    if indices != list(range(spec.n_active_dof)):
        raise ValueError(f"sdk_index must be 0..{spec.n_active_dof - 1} in order, got {indices}")
    if len({e.canonical for e in entries}) != len(entries):
        raise ValueError("duplicate canonical names in joint map")
    return entries


def load_command_schema(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or COMMAND_SCHEMA_PATH)


def load_five_point_schema(path: Path | None = None) -> dict[str, Any]:
    """Optional 5-point elbow extension. Does not change command_schema_v1 dim."""
    raw = load_yaml(path or FIVE_POINT_SCHEMA_PATH)
    extra = int(raw["extra_dim"])
    if extra != 6:
        raise ValueError(f"5-point extra_dim must be 6 (two elbow xyz), got {extra}")
    if list(raw["teleop_5point_order"]) != [
        "left_wrist",
        "right_wrist",
        "head",
        "left_elbow",
        "right_elbow",
    ]:
        raise ValueError("teleop_5point_order must be left_wrist, right_wrist, head, left_elbow, right_elbow")
    return raw


def five_point_command_dim(spec: HandSpec | None = None) -> int:
    return command_dim(spec) + int(load_five_point_schema()["extra_dim"])


def load_frames(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or FRAMES_PATH)


def command_layout(spec: HandSpec | None = None) -> dict[str, tuple[int, int]]:
    """Inclusive-exclusive slices of the flat command vector."""
    n_h = (spec or load_hand_spec()).n_active_dof
    cursor = 0
    layout: dict[str, tuple[int, int]] = {}
    wbc_fields = [
        ("head_pos", 3),
        ("head_rot6d", 6),
        ("left_wrist_pos", 3),
        ("left_wrist_rot6d", 6),
        ("right_wrist_pos", 3),
        ("right_wrist_rot6d", 6),
        ("pelvis_height", 1),
        ("nav_cmd", 3),
        ("loco_mode", 1),
    ]
    for name, dim in wbc_fields:
        layout[name] = (cursor, cursor + dim)
        cursor += dim
    assert cursor == WBC_DIM, cursor
    layout["left_hand_q"] = (cursor, cursor + n_h)
    cursor += n_h
    layout["right_hand_q"] = (cursor, cursor + n_h)
    cursor += n_h
    layout["left_hand_mode"] = (cursor, cursor + 1)
    cursor += 1
    layout["right_hand_mode"] = (cursor, cursor + 1)
    cursor += 1
    layout["tool_trigger"] = (cursor, cursor + 1)
    cursor += 1
    layout["_total"] = (0, cursor)
    return layout


def command_dim(spec: HandSpec | None = None) -> int:
    n_h = (spec or load_hand_spec()).n_active_dof
    return WBC_DIM + 2 * n_h + HAND_MODE_DIM + TOOL_DIM


@dataclass
class CommandVector:
    """Versioned VLA→WBC/hand command. Flattened layout is defined by command_schema_v1.yaml."""

    head_pos: np.ndarray
    head_rot6d: np.ndarray
    left_wrist_pos: np.ndarray
    left_wrist_rot6d: np.ndarray
    right_wrist_pos: np.ndarray
    right_wrist_rot6d: np.ndarray
    pelvis_height: float
    nav_cmd: np.ndarray
    loco_mode: int
    left_hand_q: np.ndarray
    right_hand_q: np.ndarray
    left_hand_mode: int = 0
    right_hand_mode: int = 0
    tool_trigger: int = 0
    spec: HandSpec | None = None

    def __post_init__(self) -> None:
        spec = self.spec or load_hand_spec()
        n_h = spec.n_active_dof
        self.head_pos = np.asarray(self.head_pos, dtype=np.float64).reshape(3)
        self.head_rot6d = np.asarray(self.head_rot6d, dtype=np.float64).reshape(6)
        self.left_wrist_pos = np.asarray(self.left_wrist_pos, dtype=np.float64).reshape(3)
        self.left_wrist_rot6d = np.asarray(self.left_wrist_rot6d, dtype=np.float64).reshape(6)
        self.right_wrist_pos = np.asarray(self.right_wrist_pos, dtype=np.float64).reshape(3)
        self.right_wrist_rot6d = np.asarray(self.right_wrist_rot6d, dtype=np.float64).reshape(6)
        self.nav_cmd = np.asarray(self.nav_cmd, dtype=np.float64).reshape(3)
        self.left_hand_q = np.asarray(self.left_hand_q, dtype=np.float64).reshape(n_h)
        self.right_hand_q = np.asarray(self.right_hand_q, dtype=np.float64).reshape(n_h)
        object.__setattr__(self, "spec", spec)

    def validate(self) -> None:
        spec = self.spec or load_hand_spec()
        if self.loco_mode not in (0, 1, 2):
            raise ValueError(f"loco_mode {self.loco_mode} not in {{0,1,2}}")
        if self.left_hand_mode not in (0, 1, 2) or self.right_hand_mode not in (0, 1, 2):
            raise ValueError("hand_mode must be 0=position, 1=grasp_primitive, 2=force")
        if self.tool_trigger not in (0, 1):
            raise ValueError("tool_trigger must be 0 or 1")
        if not np.isfinite(self.to_flat_vector()).all():
            raise ValueError("command contains NaN/Inf")
        limits = spec.limits_vector()
        for side, q in (("left", self.left_hand_q), ("right", self.right_hand_q)):
            if np.any(q < limits[:, 0] - 1e-6) or np.any(q > limits[:, 1] + 1e-6):
                bad = np.where((q < limits[:, 0]) | (q > limits[:, 1]))[0]
                names = [spec.joint_order[i] for i in bad]
                raise ValueError(f"{side} hand joints out of limits: {names}")

    def to_flat_vector(self) -> np.ndarray:
        spec = self.spec or load_hand_spec()
        parts = [
            self.head_pos,
            self.head_rot6d,
            self.left_wrist_pos,
            self.left_wrist_rot6d,
            self.right_wrist_pos,
            self.right_wrist_rot6d,
            np.array([self.pelvis_height], dtype=np.float64),
            self.nav_cmd,
            np.array([self.loco_mode], dtype=np.float64),
            self.left_hand_q,
            self.right_hand_q,
            np.array([self.left_hand_mode], dtype=np.float64),
            np.array([self.right_hand_mode], dtype=np.float64),
            np.array([self.tool_trigger], dtype=np.float64),
        ]
        vec = np.concatenate(parts)
        expected = command_dim(spec)
        if vec.shape != (expected,):
            raise ValueError(f"flat command dim {vec.shape} != {expected}")
        return vec

    @classmethod
    def from_flat_vector(
        cls, vec: np.ndarray | Sequence[float], spec: HandSpec | None = None
    ) -> CommandVector:
        spec = spec or load_hand_spec()
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        layout = command_layout(spec)
        total = layout["_total"][1]
        if arr.shape[0] != total:
            raise ValueError(f"expected dim {total}, got {arr.shape[0]}")

        def sl(name: str) -> np.ndarray:
            a, b = layout[name]
            return arr[a:b]

        return cls(
            head_pos=sl("head_pos"),
            head_rot6d=sl("head_rot6d"),
            left_wrist_pos=sl("left_wrist_pos"),
            left_wrist_rot6d=sl("left_wrist_rot6d"),
            right_wrist_pos=sl("right_wrist_pos"),
            right_wrist_rot6d=sl("right_wrist_rot6d"),
            pelvis_height=float(sl("pelvis_height")[0]),
            nav_cmd=sl("nav_cmd"),
            loco_mode=int(round(float(sl("loco_mode")[0]))),
            left_hand_q=sl("left_hand_q"),
            right_hand_q=sl("right_hand_q"),
            left_hand_mode=int(round(float(sl("left_hand_mode")[0]))),
            right_hand_mode=int(round(float(sl("right_hand_mode")[0]))),
            tool_trigger=int(round(float(sl("tool_trigger")[0]))),
            spec=spec,
        )

    @classmethod
    def zeros(cls, spec: HandSpec | None = None) -> CommandVector:
        spec = spec or load_hand_spec()
        n_h = spec.n_active_dof
        identity6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        return cls(
            head_pos=np.zeros(3),
            head_rot6d=identity6.copy(),
            left_wrist_pos=np.zeros(3),
            left_wrist_rot6d=identity6.copy(),
            right_wrist_pos=np.zeros(3),
            right_wrist_rot6d=identity6.copy(),
            pelvis_height=0.55,
            nav_cmd=np.zeros(3),
            loco_mode=0,
            left_hand_q=np.zeros(n_h),
            right_hand_q=np.zeros(n_h),
            spec=spec,
        )


def collect_repo_required_inputs() -> list[str]:
    """All REQUIRED_INPUT paths across frozen contract files."""
    missing: list[str] = []
    for path in (HAND_SPEC_PATH, MOUNT_PATH, JOINT_MAP_PATH, COUPLING_PATH, COMMAND_SCHEMA_PATH, FRAMES_PATH):
        raw = load_yaml(path)
        for item in find_required_inputs(raw):
            missing.append(f"{path.relative_to(REPO_ROOT)}:{item}")
    return missing


def validate_repo_spec() -> None:
    """Fail loud if the production spec still has REQUIRED_INPUT leaves.

    Topology is always checked. Calibration / hardware fields stay REQUIRED_INPUT
    until humans fill them; this function raises SpecIncompleteError listing them.
    """
    spec = load_hand_spec(require_complete=False)
    spec.require_p0_topology()
    load_joint_map()
    load_command_schema()
    load_five_point_schema()
    load_frames()
    missing = collect_repo_required_inputs()
    if missing:
        raise SpecIncompleteError(missing, spec_path=HAND_SPEC_PATH)


def main_check_spec() -> None:
    try:
        validate_repo_spec()
    except SpecIncompleteError as exc:
        raise SystemExit(str(exc)) from exc
    print("spec complete")


if __name__ == "__main__":
    main_check_spec()
