"""Dataset adapter registry and base protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass
class ConvertReport:
    dataset_dir: str
    dataset_json: str
    num_training: int = 0
    succeeded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    label_values: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "dataset_json": self.dataset_json,
            "num_training": self.num_training,
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "failed": self.failed,
            "label_values": self.label_values,
        }


class DatasetAdapter(Protocol):
    name: str

    def validate(self, cfg: dict[str, Any]) -> list[str]:
        ...

    def convert(self, cfg: dict[str, Any], out_raw_dataset_dir: Path) -> ConvertReport:
        ...


_REGISTRY: dict[str, Callable[[], DatasetAdapter]] = {}


def register_adapter(name: str) -> Callable[[Callable[[], DatasetAdapter]], Callable[[], DatasetAdapter]]:
    def deco(factory: Callable[[], DatasetAdapter]) -> Callable[[], DatasetAdapter]:
        _REGISTRY[name] = factory
        return factory

    return deco


def get_adapter(name: str) -> DatasetAdapter:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown adapter '{name}'. Registered: {known}")
    return _REGISTRY[name]()


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


def dataset_folder_name(dataset_id: int, name: str) -> str:
    return f"Dataset{int(dataset_id):03d}_{name}"


def write_dataset_json(
    out_dir: Path,
    *,
    channel_names: dict[str, Any],
    labels: dict[str, Any],
    num_training: int,
    file_ending: str = ".nii.gz",
) -> Path:
    import json

    payload = {
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": file_ending,
    }
    path = out_dir / "dataset.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
