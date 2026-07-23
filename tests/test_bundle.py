"""Bundle pack/unpack smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from segkit.bundle import pack_model_folder, resolve_weights, unpack_bundle


def test_pack_unpack_roundtrip(tmp_path: Path):
    model = tmp_path / "model"
    fold = model / "fold_0"
    fold.mkdir(parents=True)
    (fold / "checkpoint_best.pth").write_bytes(b"ckpt")
    (model / "plans.json").write_text("{}", encoding="utf-8")
    (model / "dataset.json").write_text("{}", encoding="utf-8")

    bundle_zip = tmp_path / "m.bundle.zip"
    packed = pack_model_folder(model, bundle_zip)
    assert packed.is_file()

    dest = tmp_path / "out"
    unpacked = unpack_bundle(packed, dest)
    assert (unpacked / "manifest.json").is_file()
    assert (unpacked / "fold_0" / "checkpoint_best.pth").is_file()
    manifest = json.loads((unpacked / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["default_checkpoint"] == "checkpoint_best.pth"

    resolved = resolve_weights(unpacked)
    assert resolved == unpacked
