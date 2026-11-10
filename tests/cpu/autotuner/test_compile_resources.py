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
    """Track pool creation, submitted batches, and cleanup without CUDA."""
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
    """Build minimal compile requests for the fake pool."""
    return [
        {"device_index": device, "kernel_name": f"kernel_{i}"} for i in range(count)
    ]


def test_workers_reused_across_batches_and_nested_contexts(pools):
    """Nested bucket overrides share the enclosing context's workers."""
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        with autotune(True, tuning_buckets=(8, 16)):
            fp4._run_mm_fp4_precompile_pool(payloads())
        assert len(pools) == 1
        assert len(pools[0].batches) == 2
        assert not pools[0].closed
    assert pools[0].closed


def test_workers_released_on_exception_and_not_reused_by_next_context(pools):
    """Failed sessions release workers before another session starts."""
    with pytest.raises(RuntimeError, match="failed tuning"), autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        raise RuntimeError("failed tuning")
    assert pools[0].closed
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        assert len(pools) == 2
        assert not pools[1].closed
    assert pools[1].closed


def test_devices_have_separate_pools(pools):
    """CUDA device contexts are never shared across devices."""
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads(device=0))
        fp4._run_mm_fp4_precompile_pool(payloads(device=1))
        assert [p.workers for p in pools] == [4, 4]
        assert not any(p.closed for p in pools)
    assert all(p.closed for p in pools)


def test_mixed_batch_sizes_reuse_first_pool(pools):
    """Small batches reuse workers instead of retaining extra smaller pools."""
    with autotune(True):
        for count in (4, 3, 2, 8):
            fp4._run_mm_fp4_precompile_pool(payloads(count=count))
        assert len(pools) == 1
        assert pools[0].workers == 4
        assert [len(batch) for batch in pools[0].batches] == [4, 3, 2, 8]
    assert pools[0].closed


def test_pool_size_is_fixed_until_context_exit(pools, monkeypatch):
    """Memory used by live workers cannot trigger another pool allocation."""
    limits = iter((4, 2))
    monkeypatch.setattr(
        fp4, "_get_mm_fp4_cute_dsl_compile_workers", lambda: next(limits)
    )
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        fp4._run_mm_fp4_precompile_pool(payloads())
        assert [p.workers for p in pools] == [4]
    with autotune(True):
        fp4._run_mm_fp4_precompile_pool(payloads())
        assert [p.workers for p in pools] == [4, 2]
    assert all(p.closed for p in pools)


def test_direct_preparation_outside_context_keeps_temporary_pool_lifetime(pools):
    """Direct preparation releases each temporary pool immediately."""
    fp4._run_mm_fp4_precompile_pool(payloads())
    fp4._run_mm_fp4_precompile_pool(payloads())
    assert len(pools) == 2
    assert all(p.closed for p in pools)


def test_unused_context_does_not_create_workers(pools):
    """Entering autotuning alone does not spawn workers."""
    with autotune(True):
        pass
    assert not pools


def test_replay_context_does_not_extend_tuning_worker_lifetime(pools):
    """An enclosing replay-only context does not retain tuning workers."""
    with autotune(False):
        with autotune(True):
            fp4._run_mm_fp4_precompile_pool(payloads())
            assert not pools[0].closed
        assert pools[0].closed
