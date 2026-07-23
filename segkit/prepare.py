"""Prepare / convert raw datasets via adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Ensure adapters register on import
from segkit.adapters import get_adapter, list_adapters  # noqa: F401
from segkit.adapters import nifti_folder as _nifti_folder  # noqa: F401
from segkit.adapters import nifti_multilabel_merge as _nifti_merge  # noqa: F401
from segkit.adapters.nifti_folder import default_out_dir


def prepare_dataset(cfg: dict[str, Any], *, out_dir: Path | None = None) -> dict[str, Any]:
    dataset = cfg.get("dataset") or {}
    source = dataset.get("source") or {}
    adapter_name = source.get("adapter") or "nifti_folder"
    adapter = get_adapter(adapter_name)
    errors = adapter.validate(cfg)
    if errors:
        raise ValueError("prepare validation failed:\n- " + "\n- ".join(errors))

    target = Path(out_dir) if out_dir else default_out_dir(cfg)
    report = adapter.convert(cfg, target)
    return report.to_dict()
