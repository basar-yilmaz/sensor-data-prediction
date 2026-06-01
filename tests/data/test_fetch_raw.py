from __future__ import annotations

import subprocess

import pytest

from bfrb_sensors.data.fetch_raw import fetch_raw_dataset


def test_fetch_raw_dataset_streams_csv_to_partial_then_renames(tmp_path, monkeypatch):
    calls = []

    def _fake_run(cmd, check):
        calls.append({"cmd": cmd, "check": check})
        output = tmp_path / "train.csv.part"
        output.write_text("csv")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    path = fetch_raw_dataset("http://example.test/train.csv", tmp_path)

    assert path == tmp_path / "train.csv"
    assert path.read_text() == "csv"
    assert not (tmp_path / "train.csv.part").exists()
    assert calls == [
        {
            "cmd": [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "-o",
                str(tmp_path / "train.csv.part"),
                "http://example.test/train.csv",
            ],
            "check": True,
        }
    ]


def test_fetch_raw_dataset_keeps_partial_file_on_failure(tmp_path, monkeypatch):
    def _fake_run(cmd, check):
        (tmp_path / "train.csv.part").write_text("partial")
        raise subprocess.CalledProcessError(22, cmd)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        fetch_raw_dataset("http://example.test/train.csv", tmp_path)

    assert not (tmp_path / "train.csv").exists()
    assert (tmp_path / "train.csv.part").read_text() == "partial"
