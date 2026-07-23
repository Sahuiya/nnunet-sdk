"""Lightweight golden-case bench interface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class BenchCase:
    name: str
    pred: Path
    gt: Path
    max_dice_drop: float = 0.05
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchReport:
    ok: bool
    cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "cases": self.cases}


def _dice(pred, gt, label: int) -> float:
    import numpy as np

    p = pred == label
    g = gt == label
    inter = float(np.logical_and(p, g).sum())
    denom = float(p.sum() + g.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def run_bench(
    cases: list[BenchCase],
    *,
    labels: Optional[list[int]] = None,
    baseline: Optional[dict[str, float]] = None,
) -> BenchReport:
    """Compare pred vs gt Dice; optionally enforce max drop vs baseline means."""
    import nibabel as nib
    import numpy as np

    baseline = baseline or {}
    results: list[dict[str, Any]] = []
    all_ok = True

    for case in cases:
        pred = np.asanyarray(nib.load(str(case.pred)).dataobj)
        gt = np.asanyarray(nib.load(str(case.gt)).dataobj)
        if labels is None:
            labels_use = sorted(int(x) for x in np.unique(gt) if int(x) != 0)
        else:
            labels_use = labels
        per_label = {str(l): _dice(pred, gt, l) for l in labels_use}
        mean_dice = float(np.mean(list(per_label.values()))) if per_label else 1.0
        base = baseline.get(case.name)
        ok = True
        if base is not None and mean_dice < base - case.max_dice_drop:
            ok = False
            all_ok = False
        entry = {
            "name": case.name,
            "mean_dice": mean_dice,
            "per_label": per_label,
            "baseline": base,
            "ok": ok,
        }
        results.append(entry)

    return BenchReport(ok=all_ok, cases=results)


def load_bench_spec(path: Path | str) -> list[BenchCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for item in data.get("cases", []):
        cases.append(
            BenchCase(
                name=item["name"],
                pred=Path(item["pred"]),
                gt=Path(item["gt"]),
                max_dice_drop=float(item.get("max_dice_drop", 0.05)),
            )
        )
    return cases


def write_bench_report(report: BenchReport, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
