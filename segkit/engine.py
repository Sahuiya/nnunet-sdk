"""Build nnU-Net CLI argv and optionally execute via subprocess."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass
class EngineResult:
    argv: list[str]
    returncode: int = 0
    dry_run: bool = False
    stdout: str = ""
    stderr: str = ""
    env_snapshot: dict[str, str] = field(default_factory=dict)


def _which_or_name(exe: str, *, required: bool = False) -> str:
    found = shutil.which(exe)
    if found:
        return found
    if required:
        raise FileNotFoundError(
            f"Command not found on PATH: {exe}\n"
            f"Install the local nnU-Net in this env, e.g.\n"
            f"  pip install -e ./nnUNet\n"
            f"Then verify: which {exe}"
        )
    return exe


def _fold_to_str_list(fold: Any) -> list[str]:
    if fold is None:
        return ["0"]
    if isinstance(fold, (list, tuple)):
        return [str(x) for x in fold]
    return [str(fold)]


def build_plan_argv(cfg: Mapping[str, Any]) -> list[str]:
    dataset_id = cfg.get("dataset", {}).get("id")
    if dataset_id is None:
        raise ValueError("dataset.id is required for plan")
    argv = [_which_or_name("nnUNetv2_plan_and_preprocess"), "-d", str(dataset_id)]
    plan = cfg.get("plan") or {}
    train = cfg.get("train") or {}
    configurations = plan.get("configurations")
    if not configurations:
        train_c = train.get("configuration")
        configurations = [train_c] if train_c else None
    if configurations:
        if isinstance(configurations, str):
            configurations = [configurations]
        argv.extend(["-c", *[str(c) for c in configurations]])
    if plan.get("verify_dataset_integrity"):
        argv.append("--verify_dataset_integrity")
    if plan.get("no_pp"):
        argv.append("--no_pp")

    planner = plan.get("planner") or plan.get("experiment_planner")
    if planner:
        argv.extend(["-pl", str(planner)])

    gpu_memory_target = plan.get("gpu_memory_target")
    if gpu_memory_target is not None:
        gb = float(gpu_memory_target)
        if gb <= 0:
            raise ValueError("plan.gpu_memory_target must be a positive number (GB)")
        gb_arg = str(int(gb)) if gb.is_integer() else str(gb)
        argv.extend(["-gpu_memory_target", gb_arg])
        # Avoid silently overwriting default nnUNetPlans when customizing VRAM target.
        overwrite = (
            plan.get("overwrite_plans_name")
            or plan.get("plans_name")
            or (train.get("plans") if train.get("plans") not in (None, "nnUNetPlans") else None)
        )
        if not overwrite:
            tag = str(int(gb)) if gb.is_integer() else str(gb).replace(".", "p")
            overwrite = f"nnUNetPlans_{tag}G"
        argv.extend(["-overwrite_plans_name", str(overwrite)])

    preprocessor = plan.get("preprocessor_name")
    if preprocessor:
        argv.extend(["-preprocessor_name", str(preprocessor)])

    return argv


def build_train_argv(cfg: Mapping[str, Any]) -> list[str]:
    from segkit.trainer_options import resolve_train_trainer

    dataset = cfg.get("dataset") or {}
    train = cfg.get("train") or {}
    dataset_id = dataset.get("id")
    if dataset_id is None:
        raise ValueError("dataset.id is required for train")
    configuration = train.get("configuration") or "3d_fullres"
    fold = train.get("fold", 0)
    trainer = resolve_train_trainer(cfg)
    plans = train.get("plans") or "nnUNetPlans"

    argv = [
        _which_or_name("nnUNetv2_train"),
        str(dataset_id),
        str(configuration),
        str(fold),
        "-tr",
        str(trainer),
        "-p",
        str(plans),
    ]
    num_gpus = train.get("num_gpus")
    if num_gpus is not None and int(num_gpus) != 1:
        argv.extend(["-num_gpus", str(num_gpus)])
    device = train.get("device")
    if device and device != "cuda":
        argv.extend(["-device", str(device)])
    if train.get("continue_training"):
        argv.append("--c")
    if train.get("npz"):
        argv.append("--npz")
    pretrained = train.get("pretrained_weights")
    if pretrained:
        argv.extend(["-pretrained_weights", str(pretrained)])
    return argv


def build_predict_argv(cfg: Mapping[str, Any]) -> list[str]:
    predict = cfg.get("predict") or {}
    dataset = cfg.get("dataset") or {}
    train = cfg.get("train") or {}

    input_path = predict.get("input")
    output_path = predict.get("output")
    if not input_path or not output_path:
        raise ValueError("predict.input and predict.output are required for predict")

    model_folder = predict.get("model_folder")
    checkpoint = predict.get("checkpoint") or "checkpoint_best.pth"
    folds = _fold_to_str_list(predict.get("fold", 0))
    tta = predict.get("tta", True)
    save_probabilities = predict.get("save_probabilities", False)
    device = predict.get("device") or "cuda"
    backend = str(predict.get("backend") or "pytorch").strip().lower()

    if backend in {"onnx", "onnxruntime"}:
        if not model_folder:
            raise ValueError(
                "predict.backend=onnx requires predict.model_folder "
                "(trained results dir with plans.json / dataset.json)"
            )
        onnx_folder = predict.get("onnx_folder")
        if not onnx_folder:
            raise ValueError(
                "predict.backend=onnx requires predict.onnx_folder "
                "(directory from seg export containing model.onnx)"
            )
        configuration = (
            predict.get("configuration")
            or train.get("configuration")
            or None
        )
        argv = [
            _which_or_name("nnUNetv2_predict_from_onnx_modelfolder"),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-m",
            str(model_folder),
            "--onnx-folder",
            str(onnx_folder),
            "-f",
            *folds,
        ]
        if configuration:
            argv.extend(["-configuration", str(configuration)])
    elif model_folder:
        argv = [
            _which_or_name("nnUNetv2_predict_from_modelfolder"),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-m",
            str(model_folder),
            "-f",
            *folds,
            "-chk",
            str(checkpoint),
        ]
    else:
        dataset_id = dataset.get("id")
        if dataset_id is None:
            raise ValueError("dataset.id is required for predict without model_folder")
        configuration = (
            predict.get("configuration")
            or train.get("configuration")
            or "3d_fullres"
        )
        trainer = predict.get("trainer") or train.get("trainer") or "nnUNetTrainer"
        plans = predict.get("plans") or train.get("plans") or "nnUNetPlans"
        argv = [
            _which_or_name("nnUNetv2_predict"),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-d",
            str(dataset_id),
            "-c",
            str(configuration),
            "-f",
            *folds,
            "-chk",
            str(checkpoint),
            "-tr",
            str(trainer),
            "-p",
            str(plans),
        ]

    if not tta:
        argv.append("--disable_tta")
    if save_probabilities:
        argv.append("--save_probabilities")
    if device and device != "cuda":
        argv.extend(["-device", str(device)])
    return argv


def build_eval_argv(cfg: Mapping[str, Any]) -> list[str]:
    """Prefer nnUNetv2_evaluate_simple with explicit label ids."""
    eval_cfg = cfg.get("eval") or {}
    dataset = cfg.get("dataset") or {}
    gt = eval_cfg.get("gt_folder")
    pred = eval_cfg.get("pred_folder")
    if not gt or not pred:
        raise ValueError("eval.gt_folder and eval.pred_folder are required")

    labels = eval_cfg.get("labels")
    if labels is None:
        label_map = dataset.get("labels") or {}
        labels = sorted(
            int(v) for k, v in label_map.items() if str(k).lower() != "background" and int(v) != 0
        )
    elif isinstance(labels, dict):
        labels = sorted(
            int(v) for k, v in labels.items() if str(k).lower() != "background" and int(v) != 0
        )
    else:
        labels = [int(x) for x in labels]

    if not labels:
        raise ValueError("No foreground labels for eval; set eval.labels or dataset.labels")

    argv = [
        _which_or_name("nnUNetv2_evaluate_simple"),
        str(gt),
        str(pred),
        "-l",
        *[str(x) for x in labels],
    ]
    out = eval_cfg.get("output")
    if out:
        argv.extend(["-o", str(out)])
    if eval_cfg.get("chill", True):
        argv.append("--chill")
    return argv


def _stream_pipe(pipe, sink, collector: list[str]) -> None:
    """Read lines from pipe, print live, and keep a copy for summary.json."""
    try:
        for line in iter(pipe.readline, ""):
            collector.append(line)
            print(line, end="", file=sink, flush=True)
    finally:
        pipe.close()


def run_argv(
    argv: Sequence[str],
    *,
    dry_run: bool = False,
    env_snapshot: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path | str] = None,
    stream: bool = True,
    during=None,
) -> EngineResult:
    """Run a command.

    When ``stream=True`` (default), stdout/stderr are printed live while also
    collected for ``summary.json``. Set ``stream=False`` to capture silently
    (useful in unit tests).

    ``during`` is an optional callable invoked after the child process starts and
    before waiting for it to exit (e.g. start a postprocess watcher). It may
    return a cleanup callable that is invoked in a ``finally`` block after the
    process ends (success or failure).
    """
    import os
    import sys
    import threading

    argv_list = [str(x) for x in argv]
    snap = dict(env_snapshot or {})
    if dry_run:
        return EngineResult(argv=argv_list, returncode=0, dry_run=True, env_snapshot=snap)

    child_env = os.environ.copy()
    child_env.update({str(k): str(v) for k, v in snap.items() if v is not None})
    # Prefer line-buffered / unbuffered Python logs from nnU-Net child processes.
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    # Resolve first token to absolute path when possible; fail early if missing.
    if argv_list:
        argv_list = [_which_or_name(argv_list[0], required=True), *argv_list[1:]]

    if not stream:
        completed = subprocess.run(
            argv_list,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=child_env,
        )
        return EngineResult(
            argv=argv_list,
            returncode=completed.returncode,
            dry_run=False,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            env_snapshot=snap,
        )

    proc = subprocess.Popen(
        argv_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(cwd) if cwd else None,
        env=child_env,
    )
    assert proc.stdout is not None and proc.stderr is not None
    out_lines: list[str] = []
    err_lines: list[str] = []
    t_out = threading.Thread(target=_stream_pipe, args=(proc.stdout, sys.stdout, out_lines), daemon=True)
    t_err = threading.Thread(target=_stream_pipe, args=(proc.stderr, sys.stderr, err_lines), daemon=True)
    t_out.start()
    t_err.start()
    cleanup = None
    if during is not None:
        cleanup = during()
    try:
        returncode = proc.wait()
    finally:
        if callable(cleanup):
            cleanup()
    t_out.join()
    t_err.join()
    return EngineResult(
        argv=argv_list,
        returncode=returncode,
        dry_run=False,
        stdout="".join(out_lines),
        stderr="".join(err_lines),
        env_snapshot=snap,
    )


def stderr_tail(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
