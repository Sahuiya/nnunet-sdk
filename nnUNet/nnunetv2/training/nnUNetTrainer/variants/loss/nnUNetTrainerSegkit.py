"""Configurable nnU-Net trainer driven by segkit YAML (via env)."""

from __future__ import annotations

import json
import os
from datetime import datetime
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.paths import nnUNet_results
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


def _load_segkit_options() -> dict:
    raw = os.environ.get("SEGKIT_TRAINER_CONFIG")
    if not raw:
        return {}
    return json.loads(raw)


class nnUNetTrainerSegkit(nnUNetTrainer):
    """Apply ``train.num_epochs|loss|oversample_fg|mirroring|initial_lr`` from segkit YAML."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        self.segkit_options = _load_segkit_options()
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._apply_segkit_options()
        self._relocate_output_folder_if_needed()
        self._write_segkit_options_sidecar()

    def _apply_segkit_options(self) -> None:
        opts = self.segkit_options
        if not opts:
            self.print_to_log_file(
                "nnUNetTrainerSegkit: no SEGKIT_TRAINER_CONFIG; using nnU-Net defaults."
            )
            return
        if "num_epochs" in opts:
            self.num_epochs = int(opts["num_epochs"])
        if "oversample_fg" in opts:
            self.oversample_foreground_percent = float(opts["oversample_fg"])
        if "initial_lr" in opts:
            self.initial_lr = float(opts["initial_lr"])
        self.print_to_log_file(f"nnUNetTrainerSegkit options: {opts}")

    def _relocate_output_folder_if_needed(self) -> None:
        folder_name = os.environ.get("SEGKIT_TRAINER_FOLDER_NAME")
        if not folder_name or nnUNet_results is None:
            return
        new_base = join(
            nnUNet_results,
            self.plans_manager.dataset_name,
            f"{folder_name}__{self.plans_manager.plans_name}__{self.configuration_name}",
        )
        if new_base == self.output_folder_base:
            return

        old_fold = self.output_folder
        self.output_folder_base = new_base
        self.output_folder = join(self.output_folder_base, f"fold_{self.fold}")
        maybe_mkdir_p(self.output_folder)

        timestamp = datetime.now()
        self.log_file = join(
            self.output_folder,
            "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt"
            % (
                timestamp.year,
                timestamp.month,
                timestamp.day,
                timestamp.hour,
                timestamp.minute,
                timestamp.second,
            ),
        )
        self.print_to_log_file(
            f"nnUNetTrainerSegkit: output folder -> {self.output_folder}",
            also_print_to_console=True,
        )
        # Best-effort cleanup of the empty default-class folder created by super().
        try:
            if old_fold and os.path.isdir(old_fold):
                leftover = os.listdir(old_fold)
                if all(name.startswith("training_log_") for name in leftover):
                    for name in leftover:
                        os.remove(join(old_fold, name))
                    os.rmdir(old_fold)
                    parent = os.path.dirname(old_fold)
                    if os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
        except OSError:
            pass

    def _write_segkit_options_sidecar(self) -> None:
        if not self.segkit_options or not self.output_folder:
            return
        path = join(self.output_folder_base, "segkit_trainer_options.json")
        maybe_mkdir_p(self.output_folder_base)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.segkit_options, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        mode = self.segkit_options.get("mirroring")
        if mode == "off":
            mirror_axes = None
            self.inference_allowed_mirroring_axes = None
        elif mode == "only_01":
            dim = len(self.configuration_manager.patch_size)
            mirror_axes = (0,) if dim == 2 else (0, 1)
            self.inference_allowed_mirroring_axes = mirror_axes
        # mode == default / missing → keep parent
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def _build_loss(self):
        loss_name = self.segkit_options.get("loss", "dice_ce")
        if loss_name == "dice":
            loss = MemoryEfficientSoftDiceLoss(
                **{
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": self.label_manager.has_regions,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1,
            )
        elif loss_name == "dice_heavy_ce":
            if self.label_manager.has_regions:
                raise RuntimeError("loss=dice_heavy_ce does not support region-based labels")
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=0.2,
                weight_dice=1.0,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            # dice_ce (nnU-Net default)
            if self.label_manager.has_regions:
                loss = DC_and_BCE_loss(
                    {},
                    {
                        "batch_dice": self.configuration_manager.batch_dice,
                        "do_bg": True,
                        "smooth": 1e-5,
                        "ddp": self.is_ddp,
                    },
                    use_ignore_label=self.label_manager.ignore_label is not None,
                    dice_class=MemoryEfficientSoftDiceLoss,
                )
            else:
                loss = DC_and_CE_loss(
                    {
                        "batch_dice": self.configuration_manager.batch_dice,
                        "smooth": 1e-5,
                        "do_bg": False,
                        "ddp": self.is_ddp,
                    },
                    {},
                    weight_ce=1,
                    weight_dice=1,
                    ignore_label=self.label_manager.ignore_label,
                    dice_class=MemoryEfficientSoftDiceLoss,
                )

        if self._do_i_compile() and hasattr(loss, "dc"):
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
