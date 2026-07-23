"""YAML config load, defaults, CLI overrides, and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml

DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "segkit_task",
        "seed": 42,
    },
    "paths": {
        "root": ".",
        "raw": None,
        "preprocessed": None,
        "results": None,
        "runs": None,
    },
    "dataset": {
        "id": None,
        "name": None,
        "file_ending": ".nii.gz",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0},
        "source": None,
    },
    "train": {
        "configuration": "3d_fullres",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
        "fold": 0,
        "num_gpus": 1,
        "device": "cuda",
        "continue_training": False,
        "npz": False,
    },
    "predict": {
        "input": None,
        "output": None,
        "model_folder": None,
        "checkpoint": "checkpoint_best.pth",
        "configuration": None,
        "fold": 0,
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
        "tta": True,
        "save_probabilities": False,
        "device": "cuda",
        "backend": "pytorch",
        # null | plugin name (e.g. generic_largest_cc); applied in-place on predict.output
        "postprocess": None,
    },
    "eval": {
        "gt_folder": None,
        "pred_folder": None,
        "labels": None,
        "metrics": ["dice"],
        "output": None,
        "chill": True,
    },
    "export": {
        "formats": ["torchscript", "onnx"],
        "opset_version": 18,
    },
    "postprocess": {
        "enabled": False,
        "name": "identity",
        "params": {},
    },
    "bundle": {
        "include_onnx": False,
    },
    "plan": {
        "verify_dataset_integrity": False,
        "no_pp": False,
        "configurations": None,
    },
}


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
            and not isinstance(value, (str, bytes))
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_path(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def _fill_path_defaults(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    paths = cfg.setdefault("paths", {})
    root = _resolve_path(paths.get("root") or ".", base_dir)
    paths["root"] = root
    root_path = Path(root)

    if not paths.get("raw"):
        paths["raw"] = str(root_path / "nnUNet_raw")
    else:
        paths["raw"] = _resolve_path(paths["raw"], base_dir)

    if not paths.get("preprocessed"):
        paths["preprocessed"] = str(root_path / "nnUNet_preprocessed")
    else:
        paths["preprocessed"] = _resolve_path(paths["preprocessed"], base_dir)

    if not paths.get("results"):
        paths["results"] = str(root_path / "nnUNet_results")
    else:
        paths["results"] = _resolve_path(paths["results"], base_dir)

    if not paths.get("runs"):
        paths["runs"] = str(root_path / "runs")
    else:
        paths["runs"] = _resolve_path(paths["runs"], base_dir)

    predict = cfg.setdefault("predict", {})
    for key in ("input", "output", "model_folder"):
        if predict.get(key):
            predict[key] = _resolve_path(predict[key], base_dir)

    eval_cfg = cfg.setdefault("eval", {})
    for key in ("gt_folder", "pred_folder", "output"):
        if eval_cfg.get(key):
            eval_cfg[key] = _resolve_path(eval_cfg[key], base_dir)

    source = cfg.get("dataset", {}).get("source")
    if isinstance(source, dict):
        for key in ("images", "labels", "output"):
            if source.get(key):
                source[key] = _resolve_path(source[key], base_dir)
        class_dirs = source.get("class_dirs")
        if isinstance(class_dirs, list):
            source["class_dirs"] = [_resolve_path(p, base_dir) for p in class_dirs]

    return cfg


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config(
    config_path: Optional[Path | str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load defaults, optional YAML, then apply overrides. Resolve paths relative to config dir (or cwd)."""
    cfg = deepcopy(DEFAULTS)
    base_dir = Path.cwd()

    if config_path is not None:
        config_path = Path(config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        base_dir = config_path.parent
        cfg = _deep_merge(cfg, load_yaml(config_path))
        cfg["_config_path"] = str(config_path)

    if overrides:
        # Allow dotted flat overrides via nested dict only; CLI builds nested dicts.
        cfg = _deep_merge(cfg, dict(overrides))

    cfg = _fill_path_defaults(cfg, base_dir)
    cfg["_base_dir"] = str(base_dir)
    return cfg


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    return _deep_merge(dict(base), override)


def dump_config(cfg: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    if cfg.get("_config_path"):
        clean = dict(clean)
        clean["_config_path"] = cfg["_config_path"]
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
