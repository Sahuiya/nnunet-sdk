"""Tests for YAML-driven trainer options."""

from __future__ import annotations

import pytest

from segkit.engine import build_train_argv
from segkit.model_paths import trained_model_folder_name
from segkit.trainer_options import (
    has_trainer_options,
    normalize_trainer_options,
    resolve_train_trainer,
    resolve_trainer_folder_component,
    trainer_folder_name,
)


def test_no_options_keeps_legacy_trainer():
    cfg = {"train": {"trainer": "nnUNetTrainer_250epochs", "configuration": "3d_fullres"}}
    assert not has_trainer_options(cfg["train"])
    assert resolve_train_trainer(cfg) == "nnUNetTrainer_250epochs"
    assert resolve_trainer_folder_component(cfg) == "nnUNetTrainer_250epochs"
    assert trained_model_folder_name(cfg) == "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"


def test_options_switch_to_segkit_trainer_and_tag_folder():
    cfg = {
        "dataset": {"id": 102, "name": "ORGANS7"},
        "train": {
            "configuration": "3d_fullres",
            "trainer": "nnUNetTrainer_250epochs",  # ignored when knobs present
            "num_epochs": 250,
            "loss": "dice",
            "oversample_fg": 1.0,
            "mirroring": False,
        },
    }
    opts = normalize_trainer_options(cfg["train"])
    assert opts == {
        "num_epochs": 250,
        "loss": "dice",
        "oversample_fg": 1.0,
        "mirroring": "off",
    }
    assert resolve_train_trainer(cfg) == "nnUNetTrainerSegkit"
    assert trainer_folder_name(opts) == "nnUNetTrainerSegkit__ep250_ldice_fg100_m0"
    assert trained_model_folder_name(cfg).startswith("nnUNetTrainerSegkit__ep250_ldice_fg100_m0__")
    argv = build_train_argv(cfg)
    assert argv[argv.index("-tr") + 1] == "nnUNetTrainerSegkit"


def test_mirroring_only_01_and_loss_aliases():
    train = {"loss": "dice_only", "mirroring": "only_01"}
    opts = normalize_trainer_options(train)
    assert opts["loss"] == "dice"
    assert opts["mirroring"] == "only_01"
    assert "m01" in trainer_folder_name(opts)


def test_invalid_loss_raises():
    with pytest.raises(ValueError, match="Unsupported train.loss"):
        normalize_trainer_options({"loss": "focal"})


def test_initial_lr_option_and_folder_tag():
    train = {
        "num_epochs": 50,
        "loss": "dice",
        "oversample_fg": 0.5,
        "mirroring": False,
        "initial_lr": 0.001,
    }
    opts = normalize_trainer_options(train)
    assert opts["initial_lr"] == 0.001
    assert "lr0p001" in trainer_folder_name(opts)


def test_invalid_initial_lr_raises():
    with pytest.raises(ValueError, match="train.initial_lr"):
        normalize_trainer_options({"initial_lr": 0})
    with pytest.raises(ValueError, match="train.initial_lr"):
        normalize_trainer_options({"initial_lr": -1e-3})
