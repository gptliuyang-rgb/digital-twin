"""Beta 2 fingertip tactile contract.

Official SDK: decode from FingertipSensorInfo.format. Never hard-code scale or unit.
Firmware < v2.4.0 reports per-point force in newtons; v2.4.0+ is normalized full-scale.
Resultant force stays in newtons either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from interface.schema import load_hand_spec

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
ANATOMICAL_TO_LAYOUT = {
    "thumb": "thumb",
    "index": "index",
    "index_finger": "index",
    "middle": "middle",
    "middle_finger": "middle",
    "ring": "ring",
    "ring_finger": "ring",
    "pinky": "pinky",
}


@dataclass(frozen=True)
class TactileLayout:
    native_rate_hz: float
    thumb_points: int
    other_finger_points: int
    finger_order: tuple[str, ...]
    decode_from: str
    firmware_min: str
    sdk_min: str

    def point_count(self, finger: str) -> int:
        key = ANATOMICAL_TO_LAYOUT.get(finger, finger)
        if key not in self.finger_order:
            raise ValueError(f"unknown tactile finger {finger!r}")
        return self.thumb_points if key == "thumb" else self.other_finger_points

    def total_points(self) -> int:
        return self.thumb_points + self.other_finger_points * (len(self.finger_order) - 1)


def load_tactile_layout(spec=None) -> TactileLayout:
    spec = spec or load_hand_spec()
    raw = spec.raw.get("tactile_layout")
    if not isinstance(raw, dict):
        raise ValueError("dexhand2_spec.tactile_layout must be a mapping (Beta 2 contract)")
    return TactileLayout(
        native_rate_hz=float(raw["native_rate_hz"]),
        thumb_points=int(raw["thumb_points"]),
        other_finger_points=int(raw["other_finger_points"]),
        finger_order=tuple(raw["finger_order"]),
        decode_from=str(raw["decode_from"]),
        firmware_min=str(raw["firmware_min"]),
        sdk_min=str(raw["sdk_min"]),
    )


@dataclass(frozen=True)
class TactileFrame:
    finger: str
    n_points: int
    scale: tuple[float, ...]
    unit: str
    values: tuple[float, ...]


def parse_sensor_info_format(format_json: str | dict[str, Any]) -> dict[str, Any]:
    """Return the official format mapping. Refuse empty / unknown layouts."""
    payload = json.loads(format_json) if isinstance(format_json, str) else dict(format_json)
    if "scale" not in payload or "unit" not in payload:
        raise ValueError(
            "FingertipSensorInfo.format must include scale and unit. "
            "Do not assume newtons vs normalized full-scale."
        )
    return payload


def expected_point_vector_len(finger: str, n_axes: int = 3, *, spec=None) -> int:
    layout = load_tactile_layout(spec)
    return layout.point_count(finger) * n_axes
