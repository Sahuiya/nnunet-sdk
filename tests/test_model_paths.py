"""Auto-resolve trained model folder from YAML."""

from __future__ import annotations

from pathlib import Path

from segkit.config import load_config
from segkit.model_paths import default_model_folder, resolve_model_folder


def test_default_model_folder_layout(tmp_path: Path):
    conf = tmp_path / "task.yaml"
    conf.write_text(
        """
paths:
  results: /data/results
dataset:
  id: 101
  name: SPINE
train:
  trainer: nnUNetTrainer_5epochs
  plans: nnUNetPlans
  configuration: 3d_fullres
predict:
  model_folder: null
""",
        encoding="utf-8",
    )
    cfg = load_config(conf)
    path = default_model_folder(cfg)
    assert path == Path(
        "/data/results/Dataset101_SPINE/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres"
    )


def test_explicit_model_folder_wins(tmp_path: Path):
    conf = tmp_path / "task.yaml"
    conf.write_text(
        """
paths:
  results: /data/results
dataset:
  id: 101
  name: SPINE
train:
  trainer: nnUNetTrainer_5epochs
  plans: nnUNetPlans
  configuration: 3d_fullres
predict:
  model_folder: /custom/weights
""",
        encoding="utf-8",
    )
    cfg = load_config(conf)
    assert resolve_model_folder(cfg) == Path("/custom/weights").resolve()
