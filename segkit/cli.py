"""seg CLI — YOLO-style entrypoint for nnU-Net orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer

from segkit import __version__
from segkit import env as envmod
from segkit import engine
from segkit import runs as runsmod
from segkit.adapters import list_adapters
from segkit.bench import load_bench_spec, run_bench, write_bench_report
from segkit.bundle import pack_model_folder, resolve_weights, unpack_bundle
from segkit.config import load_config
from segkit.doctor import run_doctor
from segkit.model_paths import apply_resolved_model_folder
from segkit.plugins import list_post, maybe_postprocess_predict_outputs, run_postprocess
from segkit.prepare import prepare_dataset
from segkit.sdk import SegModel

app = typer.Typer(
    name="seg",
    help="segkit: YOLO-style orchestration for nnU-Net segmentation.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


def _load(
    config: Optional[Path],
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return load_config(config, overrides=overrides)


def _execute_engine(
    command: str,
    cfg: dict[str, Any],
    builder,
    *,
    dry_run: bool,
    run_name: Optional[str] = None,
) -> int:
    env_snap = envmod.apply(cfg)
    argv = builder(cfg)
    run_dir = runsmod.start_run(cfg, command, name=run_name)
    typer.echo(f"[segkit] run_dir={run_dir}")
    typer.echo(f"[segkit] env={env_snap}")
    typer.echo(f"[segkit] argv={' '.join(argv)}")

    result = engine.run_argv(argv, dry_run=dry_run, env_snapshot=env_snap)
    status = "dry_run" if dry_run else ("success" if result.returncode == 0 else "error")
    paths: dict[str, Any] = dict(env_snap)
    if command == "predict":
        paths.update(
            {
                "input": (cfg.get("predict") or {}).get("input"),
                "output": (cfg.get("predict") or {}).get("output"),
                "weights": (cfg.get("predict") or {}).get("model_folder"),
            }
        )
    if command == "eval":
        paths.update(
            {
                "gt_folder": (cfg.get("eval") or {}).get("gt_folder"),
                "pred_folder": (cfg.get("eval") or {}).get("pred_folder"),
            }
        )
    runsmod.finish_run(
        run_dir,
        status=status,
        argv=result.argv,
        returncode=result.returncode,
        paths=paths,
        stderr_tail=engine.stderr_tail(result.stderr),
        extra={"stdout_tail": engine.stderr_tail(result.stdout)},
    )
    # Live logs are already streamed by engine.run_argv; do not reprint.
    if not dry_run and result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    return 0


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("doctor")
def doctor_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Task YAML config"),
) -> None:
    """Check dependencies, CUDA, paths, and optional dataset pairing."""
    cfg = _load(config) if config else None
    if config:
        envmod.apply(cfg)  # type: ignore[arg-type]
    report = run_doctor(cfg)
    _print_json(report.to_dict())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("init")
def init_cmd(
    out: Path = typer.Option(Path("configs/task.yaml"), "--out", help="Where to write template config"),
    name: str = typer.Option("ExampleTask", "--name"),
    dataset_id: int = typer.Option(101, "--id"),
) -> None:
    """Write a starter task YAML."""
    out.parent.mkdir(parents=True, exist_ok=True)
    template = f"""project:
  name: {name}
  seed: 42

paths:
  root: .
  raw: nnUNet_raw
  preprocessed: nnUNet_preprocessed
  results: nnUNet_results
  runs: runs

dataset:
  id: {dataset_id}
  name: {name}
  file_ending: .nii.gz
  channel_names:
    0: CT
  labels:
    background: 0
    organ_a: 1
  source:
    adapter: nifti_folder
    images: data/images
    labels: data/labels

train:
  configuration: 3d_fullres
  trainer: nnUNetTrainer
  plans: nnUNetPlans
  fold: 0

predict:
  input: null
  output: null
  model_folder: null
  checkpoint: checkpoint_best.pth
  fold: 0
  tta: true
  # null | identity | generic_largest_cc | generic_min_size
  postprocess: null

eval:
  gt_folder: null
  pred_folder: null
  labels: null
  chill: true

postprocess:
  enabled: false
  name: identity
  params: {{}}

bundle:
  include_onnx: false
"""
    out.write_text(template, encoding="utf-8")
    typer.echo(f"Wrote {out.resolve()}")


@app.command("prepare")
def prepare_cmd(
    config: Path = typer.Option(..., "--config", "-c"),
    run_name: Optional[str] = typer.Option(None, "--name"),
) -> None:
    """Convert source data into nnU-Net raw DatasetXXX layout."""
    cfg = _load(config)
    envmod.apply(cfg)
    run_dir = runsmod.start_run(cfg, "prepare", name=run_name)
    typer.echo(f"[segkit] adapters={list_adapters()}")
    try:
        report = prepare_dataset(cfg)
        runsmod.finish_run(
            run_dir,
            status="success",
            paths={"dataset_dir": report.get("dataset_dir"), "dataset_json": report.get("dataset_json")},
            artifacts=[report],
        )
        _print_json({"run_dir": str(run_dir), **report})
    except Exception as exc:  # noqa: BLE001
        runsmod.finish_run(run_dir, status="error", stderr_tail=str(exc))
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@app.command("plan")
def plan_cmd(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_name: Optional[str] = typer.Option(None, "--name"),
    verify: bool = typer.Option(False, "--verify-dataset-integrity"),
) -> None:
    """Fingerprint + plan + preprocess via nnUNetv2_plan_and_preprocess."""
    overrides: dict[str, Any] = {}
    if verify:
        overrides["plan"] = {"verify_dataset_integrity": True}
    cfg = _load(config, overrides=overrides or None)
    _execute_engine("plan", cfg, engine.build_plan_argv, dry_run=dry_run, run_name=run_name)


@app.command("train")
def train_cmd(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_name: Optional[str] = typer.Option(None, "--name"),
    fold: Optional[str] = typer.Option(None, "--fold"),
    configuration: Optional[str] = typer.Option(None, "--configuration"),
    trainer: Optional[str] = typer.Option(None, "--trainer"),
) -> None:
    """Train via nnUNetv2_train."""
    train_over: dict[str, Any] = {}
    if fold is not None:
        train_over["fold"] = fold
    if configuration is not None:
        train_over["configuration"] = configuration
    if trainer is not None:
        train_over["trainer"] = trainer
    overrides = {"train": train_over} if train_over else None
    cfg = _load(config, overrides=overrides)
    _execute_engine("train", cfg, engine.build_train_argv, dry_run=dry_run, run_name=run_name)


@app.command("predict")
def predict_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    source: Optional[Path] = typer.Option(None, "--source", "-i", help="Input folder"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    weights: Optional[Path] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Model folder or bundle; omit to auto-resolve from YAML train/dataset/results",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_name: Optional[str] = typer.Option(None, "--name"),
    fold: Optional[str] = typer.Option(None, "--fold"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint"),
    disable_tta: bool = typer.Option(False, "--disable-tta"),
    postprocess: Optional[str] = typer.Option(
        None,
        "--postprocess",
        help=(
            "Optional postprocess plugin applied in-place on --output after predict. "
            f"Choices: {', '.join(list_post()) or 'identity'}. "
            "Omit to use YAML predict.postprocess (default: null)."
        ),
    ),
) -> None:
    """Predict via modelfolder. Weights default to YAML-derived results path."""
    if config is None and weights is None:
        raise typer.BadParameter("provide --config and/or --weights")
    predict_over: dict[str, Any] = {}
    if source is not None:
        predict_over["input"] = str(source.resolve())
    if output is not None:
        predict_over["output"] = str(output.resolve())
    if fold is not None:
        predict_over["fold"] = fold
    if checkpoint is not None:
        predict_over["checkpoint"] = checkpoint
    if disable_tta:
        predict_over["tta"] = False
    if postprocess is not None:
        predict_over["postprocess"] = postprocess
    overrides = {"predict": predict_over} if predict_over else None
    cfg = _load(config, overrides=overrides)
    explicit = str(resolve_weights(weights)) if weights is not None else None
    cfg = apply_resolved_model_folder(cfg, explicit=explicit, require_exists=not dry_run)
    typer.echo(f"[segkit] model_folder={cfg['predict']['model_folder']}")
    if (cfg.get("predict") or {}).get("postprocess"):
        typer.echo(f"[segkit] predict.postprocess={cfg['predict']['postprocess']}")
    _execute_engine("predict", cfg, engine.build_predict_argv, dry_run=dry_run, run_name=run_name)
    if dry_run:
        return
    pp_info = maybe_postprocess_predict_outputs(cfg)
    if pp_info is not None:
        typer.echo(
            f"[segkit] postprocess={pp_info['plugin']} files={pp_info['n_files']} "
            f"dir={pp_info['folder']}"
        )


@app.command("eval")
def eval_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    gt: Optional[Path] = typer.Option(None, "--gt"),
    pred: Optional[Path] = typer.Option(None, "--pred"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_name: Optional[str] = typer.Option(None, "--name"),
) -> None:
    """Evaluate predictions with nnUNetv2_evaluate_simple."""
    eval_over: dict[str, Any] = {}
    if gt is not None:
        eval_over["gt_folder"] = str(gt.resolve())
    if pred is not None:
        eval_over["pred_folder"] = str(pred.resolve())
    overrides = {"eval": eval_over} if eval_over else None
    cfg = _load(config, overrides=overrides)
    _execute_engine("eval", cfg, engine.build_eval_argv, dry_run=dry_run, run_name=run_name)


@app.command("postprocess")
def postprocess_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    seg: Path = typer.Option(..., "--seg"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    name: Optional[str] = typer.Option(None, "--plugin", help=f"One of: {', '.join(list_post()) or 'identity'}"),
) -> None:
    """Run a generic postprocess plugin on one segmentation."""
    overrides: dict[str, Any] = {}
    if name:
        overrides["postprocess"] = {"enabled": True, "name": name}
    cfg = _load(config, overrides=overrides or None)
    report = run_postprocess(cfg, seg, out_dir)
    _print_json(report.to_dict())


@app.command("pack")
def pack_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    model_folder: Optional[Path] = typer.Option(
        None,
        "--model-folder",
        "-m",
        help="Model folder; omit to auto-resolve from --config",
    ),
    out: Path = typer.Option(..., "--out", "-o"),
    include_onnx: bool = typer.Option(False, "--include-onnx"),
) -> None:
    """Pack fold_* model folder into a bundle dir or .zip."""
    if model_folder is None and config is None:
        raise typer.BadParameter("provide --model-folder and/or --config")
    if model_folder is not None:
        path = pack_model_folder(model_folder, out, include_onnx=include_onnx)
    else:
        cfg = apply_resolved_model_folder(_load(config), require_exists=True)
        path = pack_model_folder(cfg["predict"]["model_folder"], out, include_onnx=include_onnx)
    typer.echo(str(path))


@app.command("unpack")
def unpack_cmd(
    bundle: Path = typer.Option(..., "--bundle", "-b"),
    dest: Path = typer.Option(..., "--dest", "-d"),
) -> None:
    """Unpack a bundle zip/dir to dest."""
    path = unpack_bundle(bundle, dest)
    typer.echo(str(path))


@app.command("export")
def export_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    weights: Optional[Path] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Model folder or bundle; omit to auto-resolve from YAML",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Export deployment artifacts via nnUNetv2_export_from_modelfolder."""
    if config is None and weights is None:
        raise typer.BadParameter("provide --config and/or --weights")
    predict_over: dict[str, Any] = {}
    if output is not None:
        predict_over["output"] = str(output.resolve())
    overrides = {"predict": predict_over} if predict_over else None
    cfg = _load(config, overrides=overrides)
    explicit = str(resolve_weights(weights)) if weights is not None else None
    cfg = apply_resolved_model_folder(cfg, explicit=explicit, require_exists=not dry_run)
    model = SegModel(cfg)
    result = model.export(dry_run=dry_run)
    if dry_run:
        typer.echo("export dry-run ok")
    elif result.get("returncode") == 0:
        out_dir = result.get("output_dir")
        typer.echo(f"导出成功: {out_dir}")
    else:
        raise typer.Exit(code=int(result.get("returncode") or 1))


@app.command("bench")
def bench_cmd(
    spec: Path = typer.Option(..., "--spec", help="JSON bench spec with cases[]"),
    out: Path = typer.Option(Path("runs/bench/report.json"), "--out"),
) -> None:
    """Run golden-case Dice bench from a JSON spec."""
    cases = load_bench_spec(spec)
    report = run_bench(cases)
    write_bench_report(report, out)
    _print_json(report.to_dict())
    if not report.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
