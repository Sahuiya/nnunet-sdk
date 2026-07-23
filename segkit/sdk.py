"""Python SDK mirroring CLI pipelines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

from segkit import env as envmod
from segkit import engine
from segkit import runs as runsmod
from segkit.bundle import pack_model_folder, resolve_weights, unpack_bundle
from segkit.config import load_config
from segkit.doctor import run_doctor
from segkit.model_paths import apply_resolved_model_folder
from segkit.prepare import prepare_dataset


class SegModel:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    @classmethod
    def from_config(cls, path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> "SegModel":
        return cls(load_config(path, overrides=overrides))

    @classmethod
    def from_bundle(cls, path: str | Path, config_path: Optional[str | Path] = None) -> "SegModel":
        model_folder = resolve_weights(path)
        overrides: dict[str, Any] = {"predict": {"model_folder": str(model_folder)}}
        cfg = load_config(config_path, overrides=overrides) if config_path else load_config(overrides=overrides)
        return cls(cfg)

    @classmethod
    def from_result_folder(cls, path: str | Path, config_path: Optional[str | Path] = None) -> "SegModel":
        overrides: dict[str, Any] = {"predict": {"model_folder": str(Path(path).resolve())}}
        cfg = load_config(config_path, overrides=overrides) if config_path else load_config(overrides=overrides)
        return cls(cfg)

    def _merge(self, **overrides: Any) -> dict[str, Any]:
        cfg = deepcopy(self.cfg)
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(cfg.get(key), dict):
                from segkit.config import deep_merge

                cfg[key] = deep_merge(cfg[key], value)
            elif value is not None:
                cfg[key] = value
        return cfg

    def doctor(self) -> dict[str, Any]:
        return run_doctor(self.cfg).to_dict()

    def prepare(self, **overrides: Any) -> dict[str, Any]:
        cfg = self._merge(**overrides)
        envmod.apply(cfg)
        run_dir = runsmod.start_run(cfg, "prepare")
        try:
            report = prepare_dataset(cfg)
            runsmod.finish_run(
                run_dir,
                status="success",
                paths={"dataset_dir": report.get("dataset_dir"), "dataset_json": report.get("dataset_json")},
                artifacts=[report],
            )
            return {"run_dir": str(run_dir), **report}
        except Exception as exc:  # noqa: BLE001
            runsmod.finish_run(run_dir, status="error", stderr_tail=str(exc))
            raise

    def plan(self, dry_run: bool = False, **overrides: Any) -> dict[str, Any]:
        return self._run_engine("plan", engine.build_plan_argv, dry_run=dry_run, **overrides)

    def train(self, dry_run: bool = False, **overrides: Any) -> dict[str, Any]:
        return self._run_engine("train", engine.build_train_argv, dry_run=dry_run, **overrides)

    def predict(
        self,
        source: Optional[str] = None,
        dry_run: bool = False,
        postprocess: Optional[str] = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        predict_over = dict(overrides.pop("predict", {}) if "predict" in overrides else {})
        if source is not None:
            predict_over["input"] = source
        if postprocess is not None:
            predict_over["postprocess"] = postprocess
        if predict_over:
            overrides["predict"] = predict_over
        cfg = self._merge(**overrides)
        cfg = apply_resolved_model_folder(cfg, require_exists=not dry_run)
        result = self._run_engine("predict", engine.build_predict_argv, dry_run=dry_run, cfg=cfg)
        if dry_run:
            return result
        from segkit.plugins import maybe_postprocess_predict_outputs

        pp_info = maybe_postprocess_predict_outputs(cfg)
        if pp_info is not None:
            result["postprocess"] = pp_info
        return result

    def eval(self, pred_dir: Optional[str] = None, gt_dir: Optional[str] = None, dry_run: bool = False, **overrides: Any) -> dict[str, Any]:
        eval_over = dict(overrides.pop("eval", {}) if "eval" in overrides else {})
        if pred_dir:
            eval_over["pred_folder"] = pred_dir
        if gt_dir:
            eval_over["gt_folder"] = gt_dir
        if eval_over:
            overrides["eval"] = eval_over
        return self._run_engine("eval", engine.build_eval_argv, dry_run=dry_run, **overrides)

    def export(self, fmt: str = "torchscript", dry_run: bool = False, **overrides: Any) -> dict[str, Any]:
        """Export via nnUNetv2_export_from_modelfolder; model folder auto-resolved from YAML if unset."""
        cfg = self._merge(**overrides)
        env_snap = envmod.apply(cfg)
        cfg = apply_resolved_model_folder(cfg, require_exists=not dry_run)
        predict = cfg.get("predict") or {}
        train = cfg.get("train") or {}
        export_cfg = cfg.get("export") or {}
        model_folder = predict["model_folder"]
        out_dir = Path(predict.get("output") or str(Path(cfg["paths"]["runs"]) / "export"))
        fold = predict.get("fold", train.get("fold", 0))
        checkpoint = predict.get("checkpoint") or "checkpoint_best.pth"
        opset = int(export_cfg.get("opset_version") or 18)

        argv = [
            engine._which_or_name("nnUNetv2_export_from_modelfolder"),
            "-m",
            str(model_folder),
            "-o",
            str(out_dir),
            "-f",
            str(fold),
            "-chk",
            str(checkpoint),
            "--opset-version",
            str(opset),
        ]
        formats = export_cfg.get("formats") or ["torchscript", "onnx"]
        if "torchscript" not in formats and "jit" not in formats:
            argv.append("--no-jit")
        if "onnx" not in formats:
            argv.append("--no-onnx")
        _ = fmt

        run_dir = runsmod.start_run(cfg, "export")
        result = engine.run_argv(argv, dry_run=dry_run, env_snapshot=env_snap)
        status = "dry_run" if dry_run else ("success" if result.returncode == 0 else "error")

        artifacts: dict[str, str] = {}
        if not dry_run and result.returncode == 0:
            for name, key in (
                ("model.pt", "torchscript"),
                ("model.onnx", "onnx"),
                ("export_metadata.json", "metadata"),
            ):
                p = out_dir / name
                if p.is_file():
                    artifacts[key] = str(p.resolve())

        runsmod.finish_run(
            run_dir,
            status=status,
            argv=result.argv,
            returncode=result.returncode,
            paths={
                "model_folder": model_folder,
                "output": str(out_dir),
                **artifacts,
            },
            artifacts=list(artifacts.values()),
            stderr_tail=engine.stderr_tail(result.stderr),
        )
        if not dry_run and result.returncode != 0:
            raise RuntimeError(f"export failed ({result.returncode}): {engine.stderr_tail(result.stderr)}")
        return {
            "status": status,
            "run_dir": str(run_dir),
            "model_folder": str(model_folder),
            "output_dir": str(out_dir),
            "fold": fold,
            "checkpoint": checkpoint,
            "opset_version": opset,
            "artifacts": artifacts,
            "dry_run": dry_run,
            "returncode": result.returncode,
            "argv": result.argv,
        }

    def pack(self, out_path: str, **overrides: Any) -> str:
        cfg = self._merge(**overrides)
        cfg = apply_resolved_model_folder(cfg, require_exists=True)
        model_folder = cfg["predict"]["model_folder"]
        include_onnx = bool((cfg.get("bundle") or {}).get("include_onnx"))
        path = pack_model_folder(
            model_folder,
            out_path,
            configuration=(cfg.get("train") or {}).get("configuration"),
            default_checkpoint=(cfg.get("predict") or {}).get("checkpoint") or "checkpoint_best.pth",
            include_onnx=include_onnx,
        )
        return str(path)

    def unpack(self, bundle: str, dest: str) -> str:
        return str(unpack_bundle(bundle, dest))

    def _run_engine(
        self,
        command: str,
        builder,
        dry_run: bool = False,
        cfg: Optional[dict[str, Any]] = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        if cfg is None:
            cfg = self._merge(**overrides)
        env_snap = envmod.apply(cfg)
        argv = builder(cfg)
        run_dir = runsmod.start_run(cfg, command)
        result = engine.run_argv(argv, dry_run=dry_run, env_snapshot=env_snap)
        status = "dry_run" if dry_run else ("success" if result.returncode == 0 else "error")
        paths = {
            "nnUNet_raw": env_snap.get("nnUNet_raw"),
            "nnUNet_preprocessed": env_snap.get("nnUNet_preprocessed"),
            "nnUNet_results": env_snap.get("nnUNet_results"),
        }
        if command == "predict":
            paths["input"] = (cfg.get("predict") or {}).get("input")
            paths["output"] = (cfg.get("predict") or {}).get("output")
            paths["weights"] = (cfg.get("predict") or {}).get("model_folder")
        runsmod.finish_run(
            run_dir,
            status=status,
            argv=result.argv,
            returncode=result.returncode,
            paths=paths,
            stderr_tail=engine.stderr_tail(result.stderr),
            extra={"env": env_snap, "stdout_tail": engine.stderr_tail(result.stdout)},
        )
        if not dry_run and result.returncode != 0:
            raise RuntimeError(
                f"{command} failed ({result.returncode}): {engine.stderr_tail(result.stderr)}"
            )
        return {
            "run_dir": str(run_dir),
            "argv": result.argv,
            "dry_run": dry_run,
            "returncode": result.returncode,
            "env": env_snap,
        }
