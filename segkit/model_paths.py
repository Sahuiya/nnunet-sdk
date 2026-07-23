"""Resolve nnU-Net trained model folder from segkit config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def dataset_dir_name(cfg: Mapping[str, Any]) -> str:
    dataset = cfg.get("dataset") or {}
    dataset_id = dataset.get("id")
    name = dataset.get("name")
    if dataset_id is None or not name:
        raise ValueError("dataset.id and dataset.name are required to resolve model folder")
    return f"Dataset{int(dataset_id):03d}_{name}"


def trained_model_folder_name(cfg: Mapping[str, Any]) -> str:
    """nnU-Net results subfolder: {trainer}__{plans}__{configuration}."""
    train = cfg.get("train") or {}
    predict = cfg.get("predict") or {}
    trainer = train.get("trainer") or "nnUNetTrainer"
    plans = train.get("plans") or "nnUNetPlans"
    configuration = (
        predict.get("configuration")
        or train.get("configuration")
        or "3d_fullres"
    )
    return f"{trainer}__{plans}__{configuration}"


def default_model_folder(cfg: Mapping[str, Any]) -> Path:
    """
    {paths.results}/DatasetXXX_Name/{trainer}__{plans}__{configuration}

    Matches nnU-Net's default results layout after train.
    """
    results = (cfg.get("paths") or {}).get("results")
    if not results:
        raise ValueError("paths.results is required to resolve model folder")
    return Path(results) / dataset_dir_name(cfg) / trained_model_folder_name(cfg)


def resolve_model_folder(
    cfg: Mapping[str, Any],
    *,
    explicit: Optional[str | Path] = None,
    require_exists: bool = False,
) -> Path:
    """Prefer explicit path / predict.model_folder; else derive from train + dataset + results."""
    if explicit is not None:
        path = Path(explicit)
    else:
        configured = (cfg.get("predict") or {}).get("model_folder")
        if configured:
            path = Path(str(configured))
        else:
            path = default_model_folder(cfg)

    path = path.expanduser()
    if not path.is_absolute():
        base = Path(cfg.get("_base_dir") or Path.cwd())
        path = (base / path).resolve()
    else:
        path = path.resolve()

    if require_exists and not path.is_dir():
        raise FileNotFoundError(
            f"Model folder not found: {path}\n"
            f"Train first, or set predict.model_folder / pass --weights."
        )
    return path


def apply_resolved_model_folder(
    cfg: dict[str, Any],
    *,
    explicit: Optional[str | Path] = None,
    require_exists: bool = False,
) -> dict[str, Any]:
    """Write resolved path into cfg['predict']['model_folder'] and return cfg."""
    path = resolve_model_folder(cfg, explicit=explicit, require_exists=require_exists)
    predict = dict(cfg.get("predict") or {})
    predict["model_folder"] = str(path)
    cfg["predict"] = predict
    return cfg
