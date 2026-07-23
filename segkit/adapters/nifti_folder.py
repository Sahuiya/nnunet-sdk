"""Copy paired NIfTI image/label folders into nnU-Net raw layout."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from segkit.adapters import (
    ConvertReport,
    dataset_folder_name,
    register_adapter,
    write_dataset_json,
)


def _strip_ending(name: str, ending: str) -> str:
    if name.endswith(ending):
        return name[: -len(ending)]
    return Path(name).stem


def _case_id_from_image(stem: str) -> str:
    if "_" in stem:
        prefix, maybe_mod = stem.rsplit("_", 1)
        if maybe_mod.isdigit():
            return prefix
    return stem


def _out_image_name(stem: str, case_id: str, ending: str) -> str:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"{stem}{ending}"
    return f"{case_id}_0000{ending}"


@register_adapter("nifti_folder")
class NiftiFolderAdapter:
    name = "nifti_folder"

    def validate(self, cfg: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        dataset = cfg.get("dataset") or {}
        source = dataset.get("source") or {}
        if dataset.get("id") is None:
            errors.append("dataset.id is required")
        if not dataset.get("name"):
            errors.append("dataset.name is required")
        if not source.get("images"):
            errors.append("dataset.source.images is required")
        if not source.get("labels"):
            errors.append("dataset.source.labels is required")
        else:
            images = Path(source["images"])
            labels = Path(source["labels"])
            if not images.is_dir():
                errors.append(f"images dir not found: {images}")
            if not labels.is_dir():
                errors.append(f"labels dir not found: {labels}")
        return errors

    def convert(self, cfg: dict[str, Any], out_raw_dataset_dir: Path) -> ConvertReport:
        dataset = cfg["dataset"]
        source = dataset["source"]
        ending = dataset.get("file_ending") or ".nii.gz"
        images_dir = Path(source["images"])
        labels_dir = Path(source["labels"])

        out_raw_dataset_dir.mkdir(parents=True, exist_ok=True)
        images_tr = out_raw_dataset_dir / "imagesTr"
        labels_tr = out_raw_dataset_dir / "labelsTr"
        images_tr.mkdir(exist_ok=True)
        labels_tr.mkdir(exist_ok=True)

        report = ConvertReport(dataset_dir=str(out_raw_dataset_dir), dataset_json="")
        label_values: set[int] = {0}

        for img_path in sorted(images_dir.glob(f"*{ending}")):
            stem = _strip_ending(img_path.name, ending)
            case_id = _case_id_from_image(stem)
            candidates = [
                labels_dir / f"{case_id}{ending}",
                labels_dir / f"{stem}{ending}",
            ]
            lbl_path = next((p for p in candidates if p.is_file()), None)
            if lbl_path is None:
                report.skipped.append(case_id)
                continue

            out_img_name = _out_image_name(stem, case_id, ending)
            out_lbl_name = f"{case_id}{ending}"
            try:
                shutil.copy2(img_path, images_tr / out_img_name)
                shutil.copy2(lbl_path, labels_tr / out_lbl_name)
                report.succeeded.append(case_id)
                try:
                    import nibabel as nib
                    import numpy as np

                    data = np.asanyarray(nib.load(str(lbl_path)).dataobj)
                    label_values.update(int(x) for x in np.unique(data))
                except Exception:
                    pass
            except OSError as exc:
                report.failed.append({"case": case_id, "error": str(exc)})

        dj = write_dataset_json(
            out_raw_dataset_dir,
            channel_names=dataset.get("channel_names") or {"0": "CT"},
            labels=dataset.get("labels") or {"background": 0},
            num_training=len(report.succeeded),
            file_ending=ending,
        )
        report.dataset_json = str(dj)
        report.num_training = len(report.succeeded)
        report.label_values = sorted(label_values)
        return report


def default_out_dir(cfg: dict[str, Any]) -> Path:
    dataset = cfg["dataset"]
    raw = Path(cfg["paths"]["raw"])
    return raw / dataset_folder_name(int(dataset["id"]), str(dataset["name"]))
