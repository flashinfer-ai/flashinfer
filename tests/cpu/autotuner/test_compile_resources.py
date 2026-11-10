# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing
from types import SimpleNamespace

import pytest

from flashinfer import autotune
from flashinfer.gemm import gemm_mm_fp4_cute_dsl as fp4

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def pools(monkeypatch):
    created = []

    class Pool:
        def __init__(self, workers):
            self.workers = workers
            self.batches = []
            self.closed = False
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

        def map(self, function, payloads):
            assert not self.closed
            self.batches.append(payloads)
            return [(p["kernel_name"], None) for p in payloads]

    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: SimpleNamespace(Pool=Pool)
    )
    monkeypatch.setattr(fp4, "_get_mm_fp4_cute_dsl_compile_workers", lambda: 4)
    return created


def payloads(device=0, count=4):
    return [
        {"device_index": device, "kernel_name": f"kernel_{i}"} for i in range(count)
    ]


def test_workers_reused_across_batches_and_nested_contexts(pools):
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        with autotune(True, tuning_buckets=(8, 16)):
            fp4._run_mm_fp4_precompile_pool(payloads())
        assert len(pools) == 1
        assert len(pools[0].batches) == 2
        assert not pools[0].closed
    assert pools[0].closed


def test_workers_released_on_exception_and_not_reused_by_next_context(pools):
    with pytest.raises(RuntimeError, match="failed tuning"), autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        raise RuntimeError("failed tuning")
    assert pools[0].closed
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        assert len(pools) == 2
        assert not pools[1].closed
    assert pools[1].closed


def test_devices_and_worker_counts_have_separate_pools(pools):
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads(device=0))
        fp4._run_mm_fp4_precompile_pool(payloads(device=1))
        fp4._run_mm_fp4_precompile_pool(payloads(device=0, count=2))
        assert [p.workers for p in pools] == [4, 4, 2]
        assert not any(p.closed for p in pools)
    assert all(p.closed for p in pools)


def test_direct_preparation_outside_context_keeps_temporary_pool_lifetime(pools):
    fp4._run_mm_fp4_precompile_pool(payloads())
    fp4._run_mm_fp4_precompile_pool(payloads())
    assert len(pools) == 2
    assert all(p.closed for p in pools)


def test_unused_context_does_not_create_workers(pools):
    with autotune(True):
        pass
    assert not pools


def test_replay_context_does_not_extend_tuning_worker_lifetime(pools):
    with autotune(False):
        with autotune(True):
            fp4._run_mm_fp4_precompile_pool(payloads())
            assert not pools[0].closed
        assert pools[0].closed
