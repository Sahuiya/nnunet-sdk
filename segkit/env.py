"""Inject nnU-Net environment variables from resolved config paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def apply(cfg: Mapping[str, Any], *, create_dirs: bool = True) -> dict[str, str]:
    """Set nnUNet_raw / nnUNet_preprocessed / nnUNet_results from cfg['paths']."""
    paths = cfg.get("paths") or {}
    mapping = {
        "nnUNet_raw": str(paths["raw"]),
        "nnUNet_preprocessed": str(paths["preprocessed"]),
        "nnUNet_results": str(paths["results"]),
    }
    for key, value in mapping.items():
        os.environ[key] = value
        if create_dirs:
            Path(value).mkdir(parents=True, exist_ok=True)

    runs = paths.get("runs")
    if runs and create_dirs:
        Path(runs).mkdir(parents=True, exist_ok=True)

    return mapping


def current() -> dict[str, str | None]:
    return {
        "nnUNet_raw": os.environ.get("nnUNet_raw"),
        "nnUNet_preprocessed": os.environ.get("nnUNet_preprocessed"),
        "nnUNet_results": os.environ.get("nnUNet_results"),
    }
