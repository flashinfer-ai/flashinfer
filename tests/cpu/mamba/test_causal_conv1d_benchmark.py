# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

import importlib
import shlex
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "routine,options,expected",
    [
        (
            "causal_conv1d_fn",
            "--dim 129 --seq_len 33 --varlen --has_initial_state --input_row_stride 256",
            {
                "dim": 129,
                "varlen": True,
                "has_initial_state": True,
                "input_row_stride": 256,
            },
        ),
        (
            "causal_conv1d_update",
            "--dim 129 --seq_len 5 --speculative",
            {"dim": 129, "seq_len": 5, "speculative": True, "input_row_stride": 129},
        ),
        (
            "causal_conv1d_fn",
            "--dim 4096 --seq_lens 0 512 256 128",
            {"seq_lens": [0, 512, 256, 128], "seq_len": 512},
        ),
        (
            "selective_state_update",
            "--dim 64 --nheads 16 --dstate 128 --ngroups 2",
            {"dim": 64, "nheads": 16, "dstate": 128},
        ),
    ],
)
def test_unified_mamba_benchmark_parser(routine, options, expected, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks"))
    entry = importlib.import_module("flashinfer_benchmark")
    args = entry.parse_args(
        ["--routine", routine, "--batch_size", "4", "--refcheck"] + options.split()
    )
    assert args.routine == routine and args.batch_size == 4 and args.refcheck
    for name, value in expected.items():
        assert getattr(args, name) == value
    if routine.startswith("causal_conv1d"):
        assert args.backends == ["flashinfer", "triton"]


def test_causal_conv1d_benchmark_samples(monkeypatch):
    benchmarks = Path(__file__).resolve().parents[3] / "benchmarks"
    monkeypatch.syspath_prepend(str(benchmarks))
    entry = importlib.import_module("flashinfer_benchmark")
    for name in ("causal_conv1d.txt", "causal_conv1d_prefill.txt"):
        for line in (benchmarks / "samples" / name).read_text().splitlines():
            args = entry.parse_args(shlex.split(line))
            assert args.refcheck and args.use_cupti and not args.no_cuda_graph
            assert set(args.backends) == {"flashinfer", "triton"}
