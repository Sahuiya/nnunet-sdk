"""Adapter prepare smoke tests (no nibabel required for nifti_folder copy)."""

from __future__ import annotations

from pathlib import Path

from segkit.adapters import list_adapters
from segkit.config import load_config
from segkit.prepare import prepare_dataset


def test_adapters_registered():
    names = list_adapters()
    assert "nifti_folder" in names
    assert "nifti_multilabel_merge" in names


def test_nifti_folder_prepare(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "caseA_0000.nii.gz").write_bytes(b"img")
    (labels / "caseA.nii.gz").write_bytes(b"lbl")

    conf = tmp_path / "task.yaml"
    conf.write_text(
        f"""
paths:
  root: {tmp_path.as_posix()}
dataset:
  id: 9
  name: Tiny
  file_ending: .nii.gz
  labels:
    background: 0
    organ_a: 1
  source:
    adapter: nifti_folder
    images: {(images).as_posix()}
    labels: {(labels).as_posix()}
""",
        encoding="utf-8",
    )
    cfg = load_config(conf)
    report = prepare_dataset(cfg)
    assert report["num_training"] == 1
    out = Path(report["dataset_dir"])
    assert (out / "imagesTr" / "caseA_0000.nii.gz").is_file()
    assert (out / "labelsTr" / "caseA.nii.gz").is_file()
    assert (out / "dataset.json").is_file()
