"""Model bundle pack / unpack."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pack_model_folder(
    model_folder: Path | str,
    out_path: Path | str,
    *,
    dataset_json: Optional[Path | str] = None,
    configuration: Optional[str] = None,
    folds: Optional[list[int]] = None,
    default_checkpoint: str = "checkpoint_best.pth",
    include_onnx: bool = False,
) -> Path:
    """Pack a trained model folder (with fold_X) into a .bundle directory or .zip."""
    model_folder = Path(model_folder)
    out_path = Path(out_path)
    if not model_folder.is_dir():
        raise FileNotFoundError(f"model folder not found: {model_folder}")

    fold_dirs = sorted(model_folder.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"no fold_* under {model_folder}")

    plans = model_folder / "plans.json"
    dsjson = Path(dataset_json) if dataset_json else (model_folder / "dataset.json")

    # Materialize as directory first
    if out_path.suffix == ".zip":
        bundle_dir = out_path.with_suffix("")
    else:
        bundle_dir = out_path
        if bundle_dir.suffix == ".bundle":
            pass
        elif not bundle_dir.exists():
            # treat as directory path
            pass

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    if plans.is_file():
        shutil.copy2(plans, bundle_dir / "plans.json")
    if dsjson.is_file():
        shutil.copy2(dsjson, bundle_dir / "dataset.json")

    fold_names: list[str] = []
    for fold_dir in fold_dirs:
        fold_names.append(fold_dir.name)
        dest = bundle_dir / fold_dir.name
        dest.mkdir()
        for item in fold_dir.iterdir():
            if item.is_file() and (
                item.name.startswith("checkpoint")
                or item.suffix in {".json", ".pkl"}
                or item.name in {"debug.json"}
            ):
                shutil.copy2(item, dest / item.name)

    if include_onnx:
        for onnx_file in model_folder.glob("*.onnx"):
            shutil.copy2(onnx_file, bundle_dir / onnx_file.name)

    detected_folds = []
    for name in fold_names:
        try:
            detected_folds.append(int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass

    manifest = {
        "segkit_version": "0.1.0",
        "created_at": _utc_now(),
        "configuration": configuration,
        "folds": folds if folds is not None else detected_folds,
        "default_checkpoint": default_checkpoint,
        "source_model_folder": str(model_folder.resolve()),
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if out_path.suffix == ".zip":
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in bundle_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(bundle_dir).as_posix())
        shutil.rmtree(bundle_dir)
        return out_path

    return bundle_dir


def unpack_bundle(bundle: Path | str, dest: Path | str) -> Path:
    bundle = Path(bundle)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if bundle.is_dir():
        # copy tree
        target = dest / bundle.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bundle, target)
        return target

    if zipfile.is_zipfile(bundle):
        target = dest / bundle.stem
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        with zipfile.ZipFile(bundle, "r") as zf:
            zf.extractall(target)
        return target

    raise ValueError(f"Unsupported bundle path: {bundle}")


def resolve_weights(weights: Path | str) -> Path:
    """Return a model folder usable by nnUNetv2_predict_from_modelfolder.

    Non-existent paths are returned as-is (useful for --dry-run). Existing
    directories with fold_* or manifest.json are accepted; zip bundles are unpacked.
    """
    path = Path(weights)
    if not path.exists():
        return path
    if path.is_dir() and (path / "manifest.json").is_file():
        return path
    if path.is_dir() and any(path.glob("fold_*")):
        return path
    if path.is_dir():
        # Allow plain model folders even before folds exist (dry-run / early pack).
        return path
    if path.is_file() and zipfile.is_zipfile(path):
        return unpack_bundle(path, path.parent / f".unpacked_{path.stem}")
    raise FileNotFoundError(f"Cannot resolve weights: {weights}")
