"""Predict postprocess switch and folder batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from segkit.plugins import (
    resolve_predict_postprocess_name,
    run_postprocess_folder,
)


def test_resolve_predict_postprocess_name():
    assert resolve_predict_postprocess_name({"predict": {}}) is None
    assert resolve_predict_postprocess_name({"predict": {"postprocess": None}}) is None
    assert resolve_predict_postprocess_name({"predict": {"postprocess": "none"}}) is None
    assert resolve_predict_postprocess_name({"predict": {"postprocess": "generic_largest_cc"}}) == (
        "generic_largest_cc"
    )


def test_run_postprocess_folder_largest_cc(tmp_path: Path):
    nibabel = pytest.importorskip("nibabel")
    pytest.importorskip("scipy")

    # two components of label 1: small at [0,0,0], large 2x2x2 block
    data = np.zeros((8, 8, 8), dtype=np.uint8)
    data[0, 0, 0] = 1
    data[3:5, 3:5, 3:5] = 1
    affine = np.eye(4)
    path = tmp_path / "case.nii.gz"
    nibabel.save(nibabel.Nifti1Image(data, affine), str(path))

    reports = run_postprocess_folder(tmp_path, plugin_name="generic_largest_cc")
    assert len(reports) == 1
    out = np.asanyarray(nibabel.load(str(path)).dataobj)
    assert int(out[0, 0, 0]) == 0
    assert int(out[3, 3, 3]) == 1
    assert int(out.sum()) == 8
