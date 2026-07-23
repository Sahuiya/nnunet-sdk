"""Generic postprocess plugins (task-agnostic)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass
class PostprocessReport:
    name: str
    input: str
    output: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "details": self.details,
        }


class PostprocessPlugin(Protocol):
    name: str

    def run(self, seg_path: Path, out_dir: Path, params: dict[str, Any]) -> PostprocessReport:
        ...


_REGISTRY: dict[str, Callable[[], PostprocessPlugin]] = {}


def register_post(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory

    return deco


def get_post(name: str) -> PostprocessPlugin:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown postprocess '{name}'. Registered: {known}")
    return _REGISTRY[name]()


def list_post() -> list[str]:
    return sorted(_REGISTRY)


@register_post("identity")
class IdentityPost:
    name = "identity"

    def run(self, seg_path: Path, out_dir: Path, params: dict[str, Any]) -> PostprocessReport:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / seg_path.name
        shutil.copy2(seg_path, out_path)
        report = PostprocessReport(self.name, str(seg_path), str(out_path), {"action": "copy"})
        (out_dir / "postprocess_report.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return report


def _largest_cc_label_map(seg, connectivity: int = 1):
    import numpy as np
    from scipy import ndimage

    out = np.zeros_like(seg)
    for label in sorted(int(x) for x in np.unique(seg) if int(x) != 0):
        mask = seg == label
        labeled, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(seg.ndim, connectivity))
        if n == 0:
            continue
        counts = ndimage.sum(mask, labeled, index=list(range(1, n + 1)))
        best = int(np.argmax(counts)) + 1
        out[labeled == best] = label
    return out


@register_post("generic_largest_cc")
class LargestCCPost:
    name = "generic_largest_cc"

    def run(self, seg_path: Path, out_dir: Path, params: dict[str, Any]) -> PostprocessReport:
        import nibabel as nib
        import numpy as np

        out_dir.mkdir(parents=True, exist_ok=True)
        img = nib.load(str(seg_path))
        data = np.asanyarray(img.dataobj)
        connectivity = int(params.get("connectivity", 1))
        cleaned = _largest_cc_label_map(data, connectivity=connectivity)
        out_path = out_dir / seg_path.name
        nib.save(nib.Nifti1Image(cleaned.astype(data.dtype), img.affine, img.header), str(out_path))
        kept = sorted(int(x) for x in np.unique(cleaned) if int(x) != 0)
        report = PostprocessReport(
            self.name,
            str(seg_path),
            str(out_path),
            {"connectivity": connectivity, "labels_kept": kept},
        )
        (out_dir / "postprocess_report.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return report


@register_post("generic_min_size")
class MinSizePost:
    name = "generic_min_size"

    def run(self, seg_path: Path, out_dir: Path, params: dict[str, Any]) -> PostprocessReport:
        import nibabel as nib
        import numpy as np
        from scipy import ndimage

        out_dir.mkdir(parents=True, exist_ok=True)
        min_voxels = int(params.get("min_voxels", 100))
        connectivity = int(params.get("connectivity", 1))
        img = nib.load(str(seg_path))
        seg = np.asanyarray(img.dataobj).copy()
        removed = 0
        for label in sorted(int(x) for x in np.unique(seg) if int(x) != 0):
            mask = seg == label
            labeled, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(seg.ndim, connectivity))
            for comp in range(1, n + 1):
                comp_mask = labeled == comp
                if int(comp_mask.sum()) < min_voxels:
                    seg[comp_mask] = 0
                    removed += 1
        out_path = out_dir / seg_path.name
        nib.save(nib.Nifti1Image(seg.astype(np.asanyarray(img.dataobj).dtype), img.affine, img.header), str(out_path))
        report = PostprocessReport(
            self.name,
            str(seg_path),
            str(out_path),
            {"min_voxels": min_voxels, "components_removed": removed},
        )
        (out_dir / "postprocess_report.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return report


def run_postprocess(cfg: dict[str, Any], seg_path: Path, out_dir: Path) -> PostprocessReport:
    pp = cfg.get("postprocess") or {}
    name = pp.get("name") or "identity"
    params = dict(pp.get("params") or {})
    plugin = get_post(name)
    return plugin.run(Path(seg_path), Path(out_dir), params)


def resolve_predict_postprocess_name(cfg: dict[str, Any]) -> str | None:
    """Return plugin name if predict should apply postprocess, else None.

    Priority: ``predict.postprocess`` (string plugin name). Empty / null / false / \"none\" disables.
    """
    predict = cfg.get("predict") or {}
    raw = predict.get("postprocess")
    if raw is None or raw is False:
        return None
    if isinstance(raw, str):
        name = raw.strip()
        if not name or name.lower() in {"none", "null", "false", "off"}:
            return None
        return name
    raise TypeError(
        f"predict.postprocess must be a plugin name string or null, got {type(raw).__name__}"
    )


def _iter_seg_files(folder: Path, file_ending: str) -> list[Path]:
    ending = file_ending if file_ending.startswith(".") else f".{file_ending}"
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith(ending))
    # Prefer plain segs: skip nnU-Net probability dumps if present
    return [p for p in files if not p.name.endswith(f".npz{ending}") and not p.name.endswith(".npz.npz")]


def run_postprocess_folder(
    folder: Path,
    *,
    plugin_name: str,
    params: dict[str, Any] | None = None,
    file_ending: str = ".nii.gz",
    out_dir: Path | None = None,
) -> list[PostprocessReport]:
    """Apply a postprocess plugin to every segmentation in ``folder``.

    Default ``out_dir`` is ``folder`` (in-place overwrite of predict outputs).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"postprocess folder not found: {folder}")
    dest = Path(out_dir) if out_dir is not None else folder
    dest.mkdir(parents=True, exist_ok=True)
    plugin = get_post(plugin_name)
    params = dict(params or {})
    reports: list[PostprocessReport] = []
    for seg_path in _iter_seg_files(folder, file_ending):
        reports.append(plugin.run(seg_path, dest, params))
    return reports


def maybe_postprocess_predict_outputs(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """If ``predict.postprocess`` is set, postprocess ``predict.output`` in place."""
    name = resolve_predict_postprocess_name(cfg)
    if name is None:
        return None
    predict = cfg.get("predict") or {}
    output = predict.get("output")
    if not output:
        raise ValueError("predict.output is required when predict.postprocess is set")
    dataset = cfg.get("dataset") or {}
    pp = cfg.get("postprocess") or {}
    params = dict(pp.get("params") or {})
    file_ending = str(dataset.get("file_ending") or ".nii.gz")
    reports = run_postprocess_folder(
        Path(output),
        plugin_name=name,
        params=params,
        file_ending=file_ending,
    )
    return {
        "plugin": name,
        "folder": str(Path(output).resolve()),
        "n_files": len(reports),
        "outputs": [r.output for r in reports],
    }
