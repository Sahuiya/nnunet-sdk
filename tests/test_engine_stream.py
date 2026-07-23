"""Streaming subprocess behavior."""

from __future__ import annotations

import sys

from segkit.engine import run_argv


def test_run_argv_streams_and_captures(capsys):
    result = run_argv(
        [sys.executable, "-u", "-c", "import sys; print('hello-out'); print('hello-err', file=sys.stderr)"],
        stream=True,
    )
    assert result.returncode == 0
    assert "hello-out" in result.stdout
    assert "hello-err" in result.stderr
    captured = capsys.readouterr()
    assert "hello-out" in captured.out
    assert "hello-err" in captured.err
