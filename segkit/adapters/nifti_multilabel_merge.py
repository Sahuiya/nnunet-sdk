"""Merge multiple binary class NIfTI masks into one multi-label volume, then export nnU-Net raw."""

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


@register_adapter("nifti_multilabel_merge")
class NiftiMultilabelMergeAdapter:
    name = "nifti_multilabel_merge"

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
        class_dirs = source.get("class_dirs")
        if not class_dirs:
            errors.append("dataset.source.class_dirs (list) is required")
        else:
            for d in class_dirs:
                if not Path(d).is_dir():
                    errors.append(f"class dir not found: {d}")
        if not Path(source.get("images", "")).is_dir():
            errors.append(f"images dir not found: {source.get('images')}")
        try:
            import nibabel  # noqa: F401
        except ImportError:
            errors.append("nibabel is required for nifti_multilabel_merge (pip install nibabel)")
        return errors

    def convert(self, cfg: dict[str, Any], out_raw_dataset_dir: Path) -> ConvertReport:
        import nibabel as nib
        import numpy as np

        dataset = cfg["dataset"]
        source = dataset["source"]
        ending = dataset.get("file_ending") or ".nii.gz"
        images_dir = Path(source["images"])
        class_dirs = [Path(p) for p in source["class_dirs"]]

        out_raw_dataset_dir.mkdir(parents=True, exist_ok=True)
        images_tr = out_raw_dataset_dir / "imagesTr"
        labels_tr = out_raw_dataset_dir / "labelsTr"
        images_tr.mkdir(exist_ok=True)
        labels_tr.mkdir(exist_ok=True)

        report = ConvertReport(dataset_dir=str(out_raw_dataset_dir), dataset_json="")
        label_values: set[int] = {0}

        for img_path in sorted(images_dir.glob(f"*{ending}")):
            stem = _strip_ending(img_path.name, ending)
            case_id = stem.rsplit("_", 1)[0] if "_" in stem and stem.rsplit("_", 1)[1].isdigit() else stem

            class_paths: list[Path] = []
            for class_dir in class_dirs:
                matches = list(class_dir.glob(f"{case_id}*{ending}")) + list(
                    class_dir.glob(f"{stem}*{ending}")
                )
                # unique preserve order
                seen: set[str] = set()
                for m in matches:
                    if str(m) not in seen:
                        class_paths.append(m)
                        seen.add(str(m))
                        break

            if len(class_paths) != len(class_dirs):
                report.skipped.append(case_id)
                continue

            try:
                first = nib.load(str(class_paths[0]))
                combined = np.zeros(first.shape, dtype=np.int16)
                affine = first.affine
                for idx, class_path in enumerate(class_paths, start=1):
                    data = nib.load(str(class_path)).get_fdata()
                    combined[data > 0] = idx
                    label_values.add(idx)

                out_img = images_tr / f"{case_id}_0000{ending}"
                out_lbl = labels_tr / f"{case_id}{ending}"
                shutil.copy2(img_path, out_img)
                nib.save(nib.Nifti1Image(combined, affine), str(out_lbl))
                report.succeeded.append(case_id)
            except Exception as exc:  # noqa: BLE001
                report.failed.append({"case": case_id, "error": str(exc)})

        # Auto-fill labels if user only provided background
        labels = dict(dataset.get("labels") or {"background": 0})
        if set(labels.values()) == {0} and len(label_values) > 1:
            for v in sorted(label_values):
                if v == 0:
                    continue
                labels[f"class_{v}"] = v

        dj = write_dataset_json(
            out_raw_dataset_dir,
            channel_names=dataset.get("channel_names") or {"0": "CT"},
            labels=labels,
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
