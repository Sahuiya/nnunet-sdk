"""Tests for config merge and path resolution."""

from __future__ import annotations

from pathlib import Path

import yaml

from segkit.config import deep_merge, load_config


def test_deep_merge_nested():
    base = {"train": {"fold": 0, "trainer": "nnUNetTrainer"}, "a": 1}
    override = {"train": {"fold": 2}, "b": 2}
    merged = deep_merge(base, override)
    assert merged["train"]["fold"] == 2
    assert merged["train"]["trainer"] == "nnUNetTrainer"
    assert merged["a"] == 1
    assert merged["b"] == 2


def test_load_config_defaults(tmp_path: Path):
    cfg = load_config()
    assert cfg["train"]["configuration"] == "3d_fullres"
    assert Path(cfg["paths"]["raw"]).name == "nnUNet_raw"


def test_load_config_relative_paths(tmp_path: Path):
    conf = {
        "paths": {"root": ".", "raw": "my_raw"},
        "dataset": {"id": 7, "name": "Toy"},
        "predict": {"input": "in_dir", "output": "out_dir", "model_folder": "weights"},
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(conf), encoding="utf-8")
    cfg = load_config(path)
    assert cfg["dataset"]["id"] == 7
    assert Path(cfg["paths"]["raw"]) == (tmp_path / "my_raw").resolve()
    assert Path(cfg["predict"]["input"]) == (tmp_path / "in_dir").resolve()
    assert Path(cfg["predict"]["model_folder"]) == (tmp_path / "weights").resolve()


def test_cli_overrides_win(tmp_path: Path):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump({"train": {"fold": 0}}), encoding="utf-8")
    cfg = load_config(path, overrides={"train": {"fold": 3, "trainer": "CustomTrainer"}})
    assert cfg["train"]["fold"] == 3
    assert cfg["train"]["trainer"] == "CustomTrainer"
