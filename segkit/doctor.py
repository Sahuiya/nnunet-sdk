"""Environment and dataset integrity checks."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class DoctorReport:
    items: list[CheckItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(i.ok for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [{"name": i.name, "ok": i.ok, "detail": i.detail} for i in self.items],
        }


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_cli_on_path(exe: str) -> CheckItem:
    path = shutil.which(exe)
    if path:
        return CheckItem(exe, True, path)
    return CheckItem(exe, False, f"{exe} not found on PATH")


def run_doctor(cfg: Optional[Mapping[str, Any]] = None) -> DoctorReport:
    report = DoctorReport()

    report.items.append(
        CheckItem("python_package:yaml", _has_module("yaml"), "PyYAML")
    )
    report.items.append(
        CheckItem("python_package:typer", _has_module("typer"), "typer")
    )
    report.items.append(
        CheckItem("python_package:nnunetv2", _has_module("nnunetv2"), "nnU-Net v2 importable")
    )

    torch_ok = _has_module("torch")
    cuda_detail = "torch not installed"
    if torch_ok:
        import torch

        cuda_detail = f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda_detail += f", device0={torch.cuda.get_device_name(0)}"
    report.items.append(CheckItem("torch", torch_ok, cuda_detail))

    for exe in (
        "nnUNetv2_plan_and_preprocess",
        "nnUNetv2_train",
        "nnUNetv2_predict",
        "nnUNetv2_predict_from_modelfolder",
        "nnUNetv2_evaluate_simple",
    ):
        report.items.append(check_cli_on_path(exe))

    if cfg is not None:
        paths = cfg.get("paths") or {}
        for key in ("raw", "preprocessed", "results", "runs"):
            p = paths.get(key)
            if not p:
                report.items.append(CheckItem(f"path:{key}", False, "missing"))
                continue
            path = Path(p)
            try:
                path.mkdir(parents=True, exist_ok=True)
                writable = os_access_write(path)
                report.items.append(
                    CheckItem(f"path:{key}", writable, str(path.resolve()))
                )
            except OSError as exc:
                report.items.append(CheckItem(f"path:{key}", False, str(exc)))

        dataset = cfg.get("dataset") or {}
        dataset_id = dataset.get("id")
        dataset_name = dataset.get("name")
        raw = paths.get("raw")
        if dataset_id is not None and dataset_name and raw:
            folder = Path(raw) / f"Dataset{int(dataset_id):03d}_{dataset_name}"
            dj = folder / "dataset.json"
            images = folder / "imagesTr"
            labels = folder / "labelsTr"
            if dj.is_file():
                report.items.append(CheckItem("dataset.json", True, str(dj)))
                try:
                    meta = json.loads(dj.read_text(encoding="utf-8"))
                    ending = meta.get("file_ending", ".nii.gz")
                except json.JSONDecodeError as exc:
                    report.items.append(CheckItem("dataset.json.parse", False, str(exc)))
                    ending = ".nii.gz"
                if images.is_dir() and labels.is_dir():
                    img_stems = {
                        p.name.replace(ending, "").rsplit("_", 1)[0]
                        for p in images.glob(f"*{ending}")
                    }
                    lbl_stems = {p.name.replace(ending, "") for p in labels.glob(f"*{ending}")}
                    missing = sorted(img_stems - lbl_stems)
                    extra = sorted(lbl_stems - img_stems)
                    ok = not missing and not extra and len(img_stems) > 0
                    detail = f"pairs={len(img_stems & lbl_stems)}"
                    if missing:
                        detail += f", images_without_label={len(missing)}"
                    if extra:
                        detail += f", labels_without_image={len(extra)}"
                    report.items.append(CheckItem("dataset.pairing", ok, detail))
                else:
                    report.items.append(
                        CheckItem(
                            "dataset.folders",
                            False,
                            f"missing imagesTr/labelsTr under {folder}",
                        )
                    )
            else:
                report.items.append(
                    CheckItem(
                        "dataset.json",
                        True,
                        f"not found yet (optional until prepare): {dj}",
                    )
                )

    return report


def os_access_write(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".segkit_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False
