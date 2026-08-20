"""Argv construction must match official nnU-Net CLI shapes."""

from __future__ import annotations

from segkit.engine import build_eval_argv, build_plan_argv, build_predict_argv, build_train_argv


def _base_cfg():
    return {
        "dataset": {
            "id": 101,
            "name": "ExampleOrgans",
            "labels": {"background": 0, "organ_a": 1, "organ_b": 2},
        },
        "train": {
            "configuration": "3d_fullres",
            "trainer": "nnUNetTrainer_250epochs",
            "plans": "nnUNetPlans",
            "fold": 0,
            "device": "cuda",
        },
        "plan": {"configurations": ["3d_fullres"], "verify_dataset_integrity": True},
        "predict": {
            "input": "/data/in",
            "output": "/data/out",
            "model_folder": "/data/weights",
            "checkpoint": "checkpoint_best.pth",
            "fold": 0,
            "tta": True,
            "save_probabilities": False,
            "device": "cuda",
        },
        "eval": {
            "gt_folder": "/data/gt",
            "pred_folder": "/data/pred",
            "chill": True,
        },
    }


def test_plan_argv():
    argv = build_plan_argv(_base_cfg())
    assert argv[0].endswith("nnUNetv2_plan_and_preprocess") or argv[0] == "nnUNetv2_plan_and_preprocess"
    assert argv[1:4] == ["-d", "101", "-c"]
    assert "3d_fullres" in argv
    assert "--verify_dataset_integrity" in argv


def test_plan_argv_gpu_memory_target_auto_plans_name():
    cfg = _base_cfg()
    cfg["plan"]["gpu_memory_target"] = 24
    argv = build_plan_argv(cfg)
    assert argv[argv.index("-gpu_memory_target") + 1] == "24"
    assert argv[argv.index("-overwrite_plans_name") + 1] == "nnUNetPlans_24G"


def test_plan_argv_gpu_memory_target_uses_train_plans():
    cfg = _base_cfg()
    cfg["plan"]["gpu_memory_target"] = 40
    cfg["train"]["plans"] = "nnUNetPlans_40G"
    argv = build_plan_argv(cfg)
    assert argv[argv.index("-overwrite_plans_name") + 1] == "nnUNetPlans_40G"


def test_train_argv():
    argv = build_train_argv(_base_cfg())
    assert argv[0].endswith("nnUNetv2_train") or argv[0] == "nnUNetv2_train"
    assert argv[1:4] == ["101", "3d_fullres", "0"]
    assert argv[argv.index("-tr") + 1] == "nnUNetTrainer_250epochs"
    assert argv[argv.index("-p") + 1] == "nnUNetPlans"


def test_train_argv_pretrained_weights():
    cfg = _base_cfg()
    cfg["train"]["pretrained_weights"] = "/data/ckpt/checkpoint_best.pth"
    argv = build_train_argv(cfg)
    assert argv[argv.index("-pretrained_weights") + 1] == "/data/ckpt/checkpoint_best.pth"


def test_predict_from_modelfolder_argv_matches_eval_sh():
    # Aligns with eval.sh:
    # nnUNetv2_predict_from_modelfolder -i ... -o ... -m ... -f 0 -chk checkpoint_best.pth
    argv = build_predict_argv(_base_cfg())
    assert "nnUNetv2_predict_from_modelfolder" in argv[0]
    assert argv[argv.index("-i") + 1] == "/data/in"
    assert argv[argv.index("-o") + 1] == "/data/out"
    assert argv[argv.index("-m") + 1] == "/data/weights"
    assert argv[argv.index("-f") + 1] == "0"
    assert argv[argv.index("-chk") + 1] == "checkpoint_best.pth"
    assert "--disable_tta" not in argv


def test_predict_onnx_argv():
    cfg = _base_cfg()
    cfg["predict"]["backend"] = "onnx"
    cfg["predict"]["onnx_folder"] = "/data/export_onnx"
    cfg["predict"]["tta"] = False
    argv = build_predict_argv(cfg)
    assert "nnUNetv2_predict_from_onnx_modelfolder" in argv[0]
    assert argv[argv.index("-i") + 1] == "/data/in"
    assert argv[argv.index("-o") + 1] == "/data/out"
    assert argv[argv.index("-m") + 1] == "/data/weights"
    assert argv[argv.index("--onnx-folder") + 1] == "/data/export_onnx"
    assert argv[argv.index("-f") + 1] == "0"
    assert argv[argv.index("-configuration") + 1] == "3d_fullres"
    assert "-chk" not in argv
    assert "--disable_tta" in argv


def test_predict_onnx_requires_onnx_folder():
    cfg = _base_cfg()
    cfg["predict"]["backend"] = "onnx"
    try:
        build_predict_argv(cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "onnx_folder" in str(exc)


def test_predict_dataset_id_argv():
    cfg = _base_cfg()
    cfg["predict"]["model_folder"] = None
    cfg["predict"]["tta"] = False
    argv = build_predict_argv(cfg)
    assert "nnUNetv2_predict" in argv[0]
    assert argv[argv.index("-d") + 1] == "101"
    assert argv[argv.index("-c") + 1] == "3d_fullres"
    assert "--disable_tta" in argv


def test_eval_argv_uses_foreground_labels():
    argv = build_eval_argv(_base_cfg())
    assert "nnUNetv2_evaluate_simple" in argv[0]
    assert argv[1:3] == ["/data/gt", "/data/pred"]
    li = argv.index("-l")
    assert argv[li + 1 : li + 3] == ["1", "2"]
    assert "--chill" in argv
