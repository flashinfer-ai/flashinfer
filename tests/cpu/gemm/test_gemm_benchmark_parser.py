# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

import importlib
import shlex
from pathlib import Path

import pytest

_BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"


def _entry(monkeypatch):
    monkeypatch.syspath_prepend(str(_BENCHMARKS))
    return importlib.import_module("flashinfer_benchmark")


def _gemm_routines():
    import sys

    sys.path.insert(0, str(_BENCHMARKS))
    try:
        from routines.flashinfer_benchmark_utils import benchmark_apis
    finally:
        sys.path.remove(str(_BENCHMARKS))
    return benchmark_apis["gemm"]


def _sample_gemm_lines():
    routines = set(_gemm_routines())
    lines = []
    for line in (
        (_BENCHMARKS / "samples" / "sample_testlist.txt").read_text().split("\n")
    ):
        tokens = shlex.split(line, comments=True)
        if "--routine" in tokens and tokens[tokens.index("--routine") + 1] in routines:
            lines.append(tokens)
    return lines


def test_sample_gemm_lines_parse(monkeypatch):
    entry = _entry(monkeypatch)
    lines = _sample_gemm_lines()
    assert lines
    for tokens in lines:
        assert entry.parse_args(list(tokens)).routine in _gemm_routines()


@pytest.mark.parametrize("routine", _gemm_routines())
def test_gemm_help_formats(routine, monkeypatch, capsys):
    # argparse %-formats help strings: a bare "%" raises here (and, on
    # Python 3.14+, already in add_argument).
    entry = _entry(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        entry.parse_args(["--routine", routine, "--help"])
    assert exc.value.code == 0
    assert "--routine" in capsys.readouterr().out
