"""MXFP8 precompilation preserves the kernel contract and shares cache work."""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import json
import time

import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.gemm import gemm_mm_mxfp8_precompile as precompile
from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _mm_mxfp8_cache_key
from flashinfer.gemm.kernels.utils import _rank_mm_mxfp8_autotune_tactics
from flashinfer.jit import cute_dsl_core

PERSISTENT = ((128, 128), (1, 1), False, False, 1)
SPLITK = ((128, 8), (1, 1), True, False, 2)


@pytest.fixture
def cpu_cache(monkeypatch, tmp_path):
    from flashinfer.cute_dsl import utils

    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_103a")
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(cute_dsl_core.jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.setattr(precompile, "_get_mm_mxfp8_compile_workers", lambda: 4)
    monkeypatch.setattr(
        precompile.torch.cuda, "device", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(utils, "get_max_active_clusters", lambda size: 148 // size)
    return tmp_path


def prepare(tactics, memory_cache=None):
    precompile.precompile_mm_mxfp8_tactics(
        tactics,
        8,
        512,
        4096,
        False,
        torch.bfloat16,
        {} if memory_cache is None else memory_cache,
        torch.device("cuda:0"),
    )


def test_concurrent_preparation_rechecks_disk_and_deduplicates(cpu_cache, monkeypatch):
    pools = []

    def fake_pool(payloads, workers):
        pools.append((payloads, workers))
        time.sleep(0.05)  # let the peer encounter the outstanding compile
        for payload in payloads:
            spec = cute_dsl_core.JitSpecCuteDsl(
                "mm_mxfp8", payload["kernel_name"], None, payload["source_sha256"]
            )
            spec.module_dir.mkdir(parents=True, exist_ok=True)
            spec.object_path.write_bytes(b"compiled")
            spec.meta_path.write_text(json.dumps(spec.expected_meta))

    monkeypatch.setattr(precompile, "_run_mm_mxfp8_precompile_pool", fake_pool)
    tactics = [PERSISTENT, PERSISTENT, SPLITK]
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(prepare, tactics) for _ in range(2)]
        for future in futures:
            future.result()
    assert len(pools) == 1
    payloads, workers = pools[0]
    assert len(payloads) == workers == 2
    split = next(p for p in payloads if p["cache_key"][-1] == 2)
    assert (split["sf_m"], split["sf_n"], split["sf_k"]) == (4, 1, 32)
    assert split["max_active_clusters"] == 74
    prepare(tactics)
    assert len(pools) == 1


@pytest.mark.parametrize(
    "mode", ["disabled", "jit_disabled", "memory", "single", "one_worker"]
)
def test_no_unnecessary_compile_pool(cpu_cache, monkeypatch, mode):
    def unexpected(*args):
        pytest.fail("a compile pool should not be needed")

    monkeypatch.setattr(precompile, "_run_mm_mxfp8_precompile_pool", unexpected)
    cache = {}
    tactics = [PERSISTENT, SPLITK]
    if mode == "disabled":
        monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    elif mode == "jit_disabled":
        monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    elif mode == "memory":
        for tile, cluster, swap, prefetch, slices in tactics:
            cache[
                (
                    0,
                    _mm_mxfp8_cache_key(
                        32, tile, cluster, swap, prefetch, False, torch.bfloat16, slices
                    ),
                )
            ] = object()
    elif mode == "single":
        tactics = [PERSISTENT]
    else:
        monkeypatch.setattr(precompile, "_get_mm_mxfp8_compile_workers", lambda: 1)
    prepare(tactics, cache)


@pytest.mark.parametrize("tactic", [PERSISTENT, SPLITK])
def test_worker_builds_fp8_persistent_and_splitk_contract(monkeypatch, tactic):
    import cutlass
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm100 as persistent
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm100_splitk as splitk

    created = []
    compiled = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        persistent,
        "Sm100BlockScaledPersistentDenseGemmKernel",
        lambda *a: created.append(("persistent", a)),
    )
    monkeypatch.setattr(
        splitk,
        "Sm100BlockScaledSplitKGemmKernel",
        lambda *a: created.append(("splitk", a)),
    )
    monkeypatch.setattr(
        precompile,
        "_make_blockscaled_gemm_compile_fn",
        lambda gemm, **kw: compiled.append(kw),
    )
    monkeypatch.setattr(
        cute_dsl_core.JitSpecCuteDsl, "is_compiled", property(lambda self: False)
    )
    monkeypatch.setattr(
        cute_dsl_core.JitSpecCuteDsl, "compile_and_persist", lambda self: None
    )
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_103a")
    tile, cluster, swap, prefetch, slices = tactic
    payload = dict(
        kernel_name="test",
        cache_key=_mm_mxfp8_cache_key(
            32, tile, cluster, swap, prefetch, True, torch.bfloat16, slices
        ),
        device_index=0,
        sf_m=1,
        sf_n=1,
        sf_k=32,
        max_active_clusters=74,
        source_sha256="test",
    )
    assert precompile._mm_mxfp8_precompile_worker(payload) == ("test", None)
    expected = (
        ("splitk", (32, tile, slices, True))
        if slices > 1
        else ("persistent", (32, tile, cluster, prefetch, True))
    )
    assert created == [expected]
    assert compiled[0]["ab_cutlass_dtype"] is cutlass.Float8E4M3FN
    assert compiled[0]["sf_dtype"] is cutlass.Float8E8M0FNU
    assert compiled[0]["ab_assumed_align"] == 16
    assert compiled[0]["swap_ab"] == swap


def test_worker_budget_respects_cpu_and_cgroup(monkeypatch):
    monkeypatch.setenv("FLASHINFER_MM_MXFP8_CUTE_DSL_COMPILE_WORKERS", "8")
    monkeypatch.setattr(precompile.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    monkeypatch.setattr(precompile, "_available_host_memory_bytes", lambda: 2 << 30)
    assert precompile._get_mm_mxfp8_compile_workers() == 2
    monkeypatch.setattr(precompile, "_available_host_memory_bytes", lambda: 0)
    assert precompile._get_mm_mxfp8_compile_workers() == 0


def test_runner_precompiles_only_tactics_selected_by_tuner(monkeypatch):
    from flashinfer.gemm import gemm_base

    calls = []
    monkeypatch.setattr(
        gemm_base, "precompile_mm_mxfp8_tactics", lambda *args: calls.append(args)
    )
    runner = gemm_base._cute_dsl_gemm_mxfp8_runner(10, 3, False, torch.bfloat16)
    a, b = torch.empty(8, 4096), torch.empty(4096, 512)
    assert runner.precompile_tactics([a, b], [-1, SPLITK], None)
    assert calls[0][0] == [SPLITK]
    assert calls[0][1:4] == (8, 512, 4096)


def test_top_n_preserves_splitk_and_full_search_order():
    candidates = [((128, n), (1, 1), False, False, 1) for n in (64, 128, 192, 256)] + [
        SPLITK
    ]
    assert (
        _rank_mm_mxfp8_autotune_tactics(candidates, 8, 512, 4096, 148, 0) is candidates
    )
    bounded = _rank_mm_mxfp8_autotune_tactics(candidates, 8, 512, 4096, 148, 3)
    assert len(bounded) == 3
    assert SPLITK in bounded
    assert set(bounded) <= set(candidates)
    with pytest.raises(ValueError, match="budget"):
        _rank_mm_mxfp8_autotune_tactics(candidates, 8, 512, 4096, 148, 1)
