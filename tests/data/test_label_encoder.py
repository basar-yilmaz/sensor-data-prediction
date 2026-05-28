"""Tests for the gesture label encoder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bfrb_sensors.data.label_encoder import (
    LabelEncoder,
    build_label_encoder,
)


def test_build_encoder_assigns_sorted_indices():
    encoder = build_label_encoder(["b_gesture", "a_gesture", "c_gesture"])
    assert encoder.encode("a_gesture") == 0
    assert encoder.encode("b_gesture") == 1
    assert encoder.encode("c_gesture") == 2


def test_encoder_round_trip(tmp_path: Path):
    encoder = build_label_encoder(["g1", "g2", "g3"])
    path = tmp_path / "label_encoder.json"
    encoder.save(path)

    loaded = LabelEncoder.load(path)
    for label in ["g1", "g2", "g3"]:
        assert loaded.decode(loaded.encode(label)) == label


def test_encoder_rejects_unknown_label():
    encoder = build_label_encoder(["a", "b"])
    with pytest.raises(ValueError, match="unknown gesture"):
        encoder.encode("c")


def test_encoder_save_is_human_readable(tmp_path: Path):
    encoder = build_label_encoder(["a", "b"])
    path = tmp_path / "label_encoder.json"
    encoder.save(path)

    payload = json.loads(path.read_text())
    assert payload == {"a": 0, "b": 1}
