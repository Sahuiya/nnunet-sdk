"""Experiment run directories: args.yaml + summary.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from segkit.config import dump_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_exp_dir(runs_root: Path | str, command: str, name: Optional[str] = None) -> Path:
    base = Path(runs_root) / command
    base.mkdir(parents=True, exist_ok=True)
    if name:
        out = base / name
        out.mkdir(parents=True, exist_ok=True)
        return out

    idx = 0
    while True:
        candidate = base / f"exp{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def write_args(run_dir: Path, cfg: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / "args.yaml"
    dump_config(cfg, path)
    return path


def write_summary(run_dir: Path, summary: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / "summary.json"
    payload = dict(summary)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def start_run(
    cfg: Mapping[str, Any],
    command: str,
    *,
    name: Optional[str] = None,
) -> Path:
    runs_root = cfg["paths"]["runs"]
    run_dir = next_exp_dir(runs_root, command, name=name)
    write_args(run_dir, cfg)
    write_summary(
        run_dir,
        {
            "status": "started",
            "command": command,
            "started_at": _utc_now(),
            "finished_at": None,
            "paths": {},
            "metrics": None,
            "artifacts": [],
            "argv": None,
            "returncode": None,
            "stderr_tail": None,
        },
    )
    return run_dir


def finish_run(
    run_dir: Path,
    *,
    status: str,
    argv: Optional[list[str]] = None,
    returncode: Optional[int] = None,
    paths: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[list[Any]] = None,
    metrics: Any = None,
    stderr_tail: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    summary_path = Path(run_dir) / "summary.json"
    existing: dict[str, Any] = {}
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))

    existing.update(
        {
            "status": status,
            "finished_at": _utc_now(),
            "argv": argv,
            "returncode": returncode,
            "paths": dict(paths or existing.get("paths") or {}),
            "artifacts": list(artifacts if artifacts is not None else existing.get("artifacts") or []),
            "metrics": metrics if metrics is not None else existing.get("metrics"),
            "stderr_tail": stderr_tail,
        }
    )
    if extra:
        existing.update(dict(extra))
    return write_summary(run_dir, existing)
