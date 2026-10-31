# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Elementwise numerical acceptance, including real cross-rank reductions."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_report():
    path = Path(__file__).resolve().parents[2] / "examples" / "mok_bf16_toy.py"
    spec = importlib.util.spec_from_file_location("mok_tolerance_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.error_report


@pytest.fixture
def report(tmp_path):
    if dist.is_initialized():
        pytest.skip("Run tolerance unit checks outside an existing process group")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        yield _load_report()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "actual,expected,passes",
    [
        ([1.015], [1.0], True),
        ([0.01], [0.0], True),
        ([0.01001], [0.0], False),
        ([1.0201], [1.0], False),
        ([0.02, 10000.0], [0.0, 10000.0], False),
        ([], [], True),
        ([float("inf")], [float("inf")], False),
        ([float("nan")], [float("nan")], False),
    ],
)
def test_elementwise_acceptance(report, actual, expected, passes):
    a, b = torch.tensor(actual), torch.tensor(expected)
    results = report((a,) * 9, (b,) * 9, {"atol": 1e-2, "rtol": 1e-2})
    for result in results.values():
        assert result["pass"] is passes
        assert result["global_elements"] == a.numel()
        if passes:
            assert result["global_mismatched"] == 0
        else:
            assert result["global_mismatched"] > 0
    if passes and a.numel():
        torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)


def test_chunk_boundary_mismatch(report):
    actual = torch.zeros(1048577)
    expected = torch.zeros_like(actual)
    actual[-1] = 0.02
    results = report((actual,) * 9, (expected,) * 9, {"atol": 1e-2, "rtol": 1e-2})
    for result in results.values():
        assert not result["pass"]
        assert result["global_mismatched"] == 1
        assert result["rank_worst_error"]["flat_index"] == actual.numel() - 1
        assert result["global_max_error_ratio"] == pytest.approx(2.0)


def _distributed_worker(rank, path):
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2
    )
    try:
        actual = torch.empty(0) if rank == 0 else torch.tensor([0.02])
        expected = torch.zeros_like(actual)
        results = _load_report()(
            (actual,) * 9, (expected,) * 9, {"atol": 1e-2, "rtol": 1e-2}
        )
        for result in results.values():
            assert not result["pass"]
            assert result["global_mismatched"] == 1
            assert result["global_elements"] == 1
            assert result["rank_mismatched"] == rank
    finally:
        dist.destroy_process_group()


def test_remote_failure_reaches_empty_rank(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "distributed"),), nprocs=2)
