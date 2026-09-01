from __future__ import annotations

import json

import pytest

from hand.tactile import (
    expected_point_vector_len,
    load_tactile_layout,
    parse_sensor_info_format,
)


def test_official_tactile_point_counts() -> None:
    layout = load_tactile_layout()
    assert layout.native_rate_hz == 100
    assert layout.point_count("thumb") == 40
    assert layout.point_count("index") == 34
    assert layout.point_count("pinky") == 34
    assert layout.total_points() == 40 + 34 * 4
    assert expected_point_vector_len("thumb") == 120
    assert expected_point_vector_len("index_finger") == 102


def test_refuse_format_without_scale_unit() -> None:
    with pytest.raises(ValueError, match="scale and unit"):
        parse_sensor_info_format({"axes": ["n", "u", "v"]})


def test_parse_format_keeps_official_scale() -> None:
    payload = parse_sensor_info_format(
        json.dumps({"scale": [1.0, 1.0, 1.0], "unit": "normalized", "n_points": 40})
    )
    assert payload["unit"] == "normalized"
    assert payload["n_points"] == 40
