"""YAML-driven nnU-Net trainer options for segkit.

When ``train`` contains any of ``num_epochs`` / ``loss`` / ``oversample_fg`` /
``mirroring`` / ``initial_lr``, training uses ``nnUNetTrainerSegkit`` and
injects options via environment variables so the nnU-Net subprocess can apply
them.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, MutableMapping, Optional

OPTION_KEYS = ("num_epochs", "loss", "oversample_fg", "mirroring", "initial_lr")
ENV_CONFIG = "SEGKIT_TRAINER_CONFIG"
ENV_FOLDER_NAME = "SEGKIT_TRAINER_FOLDER_NAME"
SEGKIT_TRAINER_CLASS = "nnUNetTrainerSegkit"

_LOSS_ALIASES = {
    "dice_ce": "dice_ce",
    "dicece": "dice_ce",
    "default": "dice_ce",
    "dice": "dice",
    "dice_only": "dice",
    "diceheavy": "dice_heavy_ce",
    "dice_heavy": "dice_heavy_ce",
    "dice_heavy_ce": "dice_heavy_ce",
}


def has_trainer_options(train: Mapping[str, Any] | None) -> bool:
    train = train or {}
    return any(k in train and train[k] is not None for k in OPTION_KEYS)


def _parse_mirroring(value: Any) -> str:
    if value is True or value is False:
        return "default" if value else "off"
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on", "default"):
        return "default"
    if s in ("0", "false", "no", "off", "none", "nomirror", "no_mirroring"):
        return "off"
    if s in ("only_01", "only01", "01", "no_lr", "nolr"):
        return "only_01"
    raise ValueError(
        f"Unsupported train.mirroring={value!r}. Use true|false|only_01."
    )


def _parse_loss(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in _LOSS_ALIASES:
        raise ValueError(
            f"Unsupported train.loss={value!r}. "
            f"Use dice_ce | dice | dice_heavy_ce."
        )
    return _LOSS_ALIASES[key]


def normalize_trainer_options(train: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return normalized options; empty dict if no knobs set."""
    train = train or {}
    if not has_trainer_options(train):
        return {}

    opts: dict[str, Any] = {}
    if train.get("num_epochs") is not None:
        opts["num_epochs"] = int(train["num_epochs"])
        if opts["num_epochs"] <= 0:
            raise ValueError("train.num_epochs must be positive")
    if train.get("loss") is not None:
        opts["loss"] = _parse_loss(train["loss"])
    if train.get("oversample_fg") is not None:
        fg = float(train["oversample_fg"])
        if not 0.0 <= fg <= 1.0:
            raise ValueError("train.oversample_fg must be in [0, 1]")
        opts["oversample_fg"] = fg
    if train.get("mirroring") is not None:
        opts["mirroring"] = _parse_mirroring(train["mirroring"])
    if train.get("initial_lr") is not None:
        lr = float(train["initial_lr"])
        if not (lr > 0.0):
            raise ValueError("train.initial_lr must be > 0")
        opts["initial_lr"] = lr
    return opts


def _lr_folder_tag(lr: float) -> str:
    """Filesystem-safe tag, e.g. 0.001 -> lr0p001, 1e-4 -> lr0p0001."""
    return "lr" + f"{lr:g}".replace("-", "m").replace(".", "p")


def trainer_folder_tag(opts: Mapping[str, Any]) -> str:
    """Short deterministic tag for results folder naming."""
    parts: list[str] = []
    if "num_epochs" in opts:
        parts.append(f"ep{opts['num_epochs']}")
    if "loss" in opts:
        loss = opts["loss"]
        parts.append({"dice_ce": "ldicece", "dice": "ldice", "dice_heavy_ce": "ldiceh"}.get(loss, f"l{loss}"))
    if "oversample_fg" in opts:
        parts.append(f"fg{int(round(float(opts['oversample_fg']) * 100))}")
    if "mirroring" in opts:
        parts.append({"default": "m1", "off": "m0", "only_01": "m01"}[opts["mirroring"]])
    if "initial_lr" in opts:
        parts.append(_lr_folder_tag(float(opts["initial_lr"])))
    return "_".join(parts) if parts else "default"


def trainer_folder_name(opts: Mapping[str, Any]) -> str:
    tag = trainer_folder_tag(opts)
    return f"{SEGKIT_TRAINER_CLASS}__{tag}"


def resolve_train_trainer(cfg: Mapping[str, Any]) -> str:
    """Trainer class name passed to ``nnUNetv2_train -tr``."""
    train = cfg.get("train") or {}
    if has_trainer_options(train):
        return SEGKIT_TRAINER_CLASS
    return str(train.get("trainer") or "nnUNetTrainer")


def resolve_trainer_folder_component(cfg: Mapping[str, Any]) -> str:
    """First ``__``-separated component of the nnU-Net results folder name."""
    train = cfg.get("train") or {}
    opts = normalize_trainer_options(train)
    if opts:
        return trainer_folder_name(opts)
    return str(train.get("trainer") or "nnUNetTrainer")


def inject_trainer_env(
    cfg: Mapping[str, Any],
    env: Optional[MutableMapping[str, str]] = None,
) -> dict[str, str]:
    """Write SEGKIT_* into ``os.environ`` (and optional ``env`` mapping for snapshots)."""
    train = cfg.get("train") or {}
    opts = normalize_trainer_options(train)
    if not opts:
        for key in (ENV_CONFIG, ENV_FOLDER_NAME):
            os.environ.pop(key, None)
            if env is not None:
                env.pop(key, None)
        return {}

    folder = trainer_folder_name(opts)
    payload = json.dumps(opts, ensure_ascii=False, sort_keys=True)
    os.environ[ENV_CONFIG] = payload
    os.environ[ENV_FOLDER_NAME] = folder
    if env is not None:
        env[ENV_CONFIG] = payload
        env[ENV_FOLDER_NAME] = folder
    return {ENV_CONFIG: payload, ENV_FOLDER_NAME: folder}


def load_options_from_env() -> dict[str, Any]:
    raw = os.environ.get(ENV_CONFIG)
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"{ENV_CONFIG} must be a JSON object")
    return data
