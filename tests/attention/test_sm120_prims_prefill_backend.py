import math
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

import flashinfer


def _has_required_cutlass_dsl() -> bool:
    try:
        installed_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return False
    return Version(installed_version) >= Version("4.7.0")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not _has_required_cutlass_dsl(),
    reason="requires SM120 and nvidia-cutlass-dsl>=4.7.0",
)


def _fp8(shape):
    return torch.randint(-2, 3, shape, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )


def _reference(q, k, v, causal, scale):
    hq, hkv = q.shape[1], k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    vf = v.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * scale
    if causal:
        q_pos = torch.arange(q.shape[0], device=q.device) + k.shape[0] - q.shape[0]
        k_pos = torch.arange(k.shape[0], device=q.device)
        scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -torch.inf)
    lse = torch.logsumexp(scores, -1).transpose(0, 1) * math.log2(math.e)
    out = torch.einsum("hqk,hkd->hqd", scores.softmax(-1), vf).transpose(0, 1)
    return out, lse


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_ragged_public_wrapper(causal, out_dtype):
    hq, hkv, d = 4, 2, 32
    q_lens, kv_lens = [3, 2], [5, 4]
    qo = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 5, 9], dtype=torch.int32, device="cuda")
    q, k, v = _fp8((5, hq, d)), _fp8((9, hkv, d)), _fp8((9, hkv, d))
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo,
        kv,
        hq,
        hkv,
        d,
        causal=causal,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=out_dtype,
    )
    out = torch.empty_like(q, dtype=out_dtype)
    q_scale, k_scale, v_scale = 0.75, 1.25, 0.5
    actual, actual_lse = wrapper.run_return_lse(
        q,
        k,
        v,
        out=out,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=True,
    )
    assert actual is out
    refs, lses = [], []
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens, strict=False)):
        ref, ref_lse = _reference(
            q[qo[i] : qo[i] + ql],
            k[kv[i] : kv[i] + kl],
            v[kv[i] : kv[i] + kl],
            causal,
            q_scale * k_scale / math.sqrt(d),
        )
        refs.append(ref * v_scale)
        lses.append(ref_lse)
    torch.testing.assert_close(actual.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(actual_lse, torch.cat(lses), atol=0.2, rtol=0.2)


@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_paged_public_wrapper(page_size):
    hq, hkv, d = 4, 2, 32
    qo = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    page_indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    page_indices = torch.tensor([3, 1, 4], dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor([5, 4], dtype=torch.int32, device="cuda")
    q = _fp8((5, hq, d))
    combined_cache = _fp8((5, 2, hkv, page_size, d))
    k_pool, v_pool = combined_cache.unbind(dim=1)
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cute-dsl-prims"
    )
    assert wrapper.workspace_size(
        qo,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        q_data_type=q.dtype,
        kv_data_type=k_pool.dtype,
        o_data_type=torch.bfloat16,
    ) == (0, 0)
    wrapper.plan(
        qo,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        causal=True,
        q_data_type=q.dtype,
        kv_data_type=k_pool.dtype,
        o_data_type=torch.bfloat16,
    )
    actual, actual_lse = wrapper.run_return_lse(q, combined_cache, enable_pdl=True)
    refs, lses = [], []
    for batch, (q_start, q_end) in enumerate(zip(qo[:-1], qo[1:], strict=False)):
        pages = page_indices[page_indptr[batch] : page_indptr[batch + 1]]
        k = torch.cat([k_pool[p].transpose(0, 1) for p in pages])[
            : ((len(pages) - 1) * page_size + last_page_len[batch])
        ]
        v = torch.cat([v_pool[p].transpose(0, 1) for p in pages])[: k.shape[0]]
        ref, ref_lse = _reference(q[q_start:q_end], k, v, True, 1 / math.sqrt(d))
        refs.append(ref)
        lses.append(ref_lse)
    torch.testing.assert_close(actual.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(actual_lse, torch.cat(lses), atol=0.2, rtol=0.2)


def test_cuda_graph_reads_updated_caller_block_table():
    device, i32 = "cuda", torch.int32
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "HND",
        backend="cute-dsl-prims",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(2, dtype=i32, device=device),
        paged_kv_indptr_buf=torch.empty(2, dtype=i32, device=device),
        paged_kv_indices_buf=torch.empty(1, dtype=i32, device=device),
        paged_kv_last_page_len_buf=torch.empty(1, dtype=i32, device=device),
    )

    hq, hkv, head_dim, page_size = 2, 1, 32, 16
    q = torch.ones((1, hq, head_dim), device=device).to(torch.float8_e4m3fn)
    cache = torch.zeros(
        (2, 2, hkv, page_size, head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    cache[0, 1].fill_(1)
    cache[1, 1].fill_(-1)
    indptr = torch.tensor([0, 1], dtype=i32, device=device)
    page_indices = torch.tensor([0], dtype=i32, device=device)
    last_page_len = torch.tensor([page_size], dtype=i32, device=device)
    block_tables = torch.tensor([[0]], dtype=i32, device=device)
    wrapper.plan(
        indptr,
        indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        head_dim,
        page_size,
        block_tables=block_tables,
        q_data_type=q.dtype,
        kv_data_type=cache.dtype,
        o_data_type=torch.float16,
    )

    captured = torch.empty((1, hq, head_dim), dtype=torch.float16, device=device)
    for _ in range(3):
        wrapper.run(q, cache, out=captured, enable_pdl=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, cache, out=captured, enable_pdl=False)

    block_tables.fill_(1)
    eager = torch.empty_like(captured)
    wrapper.run(q, cache, out=eager, enable_pdl=False)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager)
    torch.testing.assert_close(eager, torch.full_like(eager, -1))


def test_cuda_graph_rejects_uncompiled_specialization():
    device, i32 = "cuda", torch.int32
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "HND",
        backend="cute-dsl-prims",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(2, dtype=i32, device=device),
        paged_kv_indptr_buf=torch.empty(2, dtype=i32, device=device),
        paged_kv_indices_buf=torch.empty(1, dtype=i32, device=device),
        paged_kv_last_page_len_buf=torch.empty(1, dtype=i32, device=device),
    )

    hq, hkv, head_dim, page_size = 2, 1, 32, 16
    q = _fp8((1, hq, head_dim))
    cache = _fp8((1, 2, hkv, page_size, head_dim))
    indptr = torch.tensor([0, 1], dtype=i32, device=device)
    page_indices = torch.tensor([0], dtype=i32, device=device)
    last_page_len = torch.tensor([1], dtype=i32, device=device)
    wrapper.plan(
        indptr,
        indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        head_dim,
        page_size,
        q_data_type=q.dtype,
        kv_data_type=cache.dtype,
        o_data_type=torch.float16,
    )

    out = torch.empty((1, hq, head_dim), dtype=torch.float16, device=device)
    lse = torch.empty((1, hq), dtype=torch.float32, device=device)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(
            RuntimeError,
            match=r"\(return_lse=True\) was not compiled before CUDA Graph capture",
        ),
        torch.cuda.graph(graph),
    ):
        out.zero_()
        # PDL is runtime-dynamic, so the plan-compiled non-LSE artifact accepts
        # the device-default PDL=True launch during capture without warm-up.
        wrapper.run(q, cache, out=out)
        wrapper.run_return_lse(
            q,
            cache,
            out=out,
            lse=lse,
            enable_pdl=False,
        )


def test_prims_fail_fast_for_unsupported_options():
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    hq, hkv, d, page_size = 2, 1, 32, 16
    q = _fp8((1, hq, d))
    k = _fp8((1, hkv, d))
    v = _fp8((1, hkv, d))
    cache = _fp8((1, 2, hkv, page_size, d))

    paged_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cute-dsl-prims"
    )
    page_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    page_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor([1], dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="max_sequence_kv"):
        paged_wrapper.plan(
            indptr,
            page_indptr,
            page_indices,
            last_page_len,
            hq,
            hkv,
            d,
            page_size,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.float16,
            max_sequence_kv=1,
        )
    paged_wrapper.plan(
        indptr,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        q_data_type=q.dtype,
        kv_data_type=cache.dtype,
        o_data_type=torch.float16,
    )
    sinks = torch.zeros(hq, dtype=torch.float32, device="cuda")
    with pytest.raises(NotImplementedError, match="does not support attention sinks"):
        paged_wrapper.run(q, cache, sinks=sinks)

    ragged_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    with pytest.raises(NotImplementedError, match="max_sequence_kv"):
        ragged_wrapper.plan(
            indptr,
            indptr,
            hq,
            hkv,
            d,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.float16,
            max_sequence_kv=1,
        )
    ragged_wrapper.plan(
        indptr,
        indptr,
        hq,
        hkv,
        d,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=torch.float16,
    )
    ragged_wrapper._sinks = sinks
    with pytest.raises(NotImplementedError, match="does not support attention sinks"):
        ragged_wrapper.run(q, k, v)


def test_prims_accepts_combined_hnd_cache_and_pdl():
    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        compile_sm120_fmha_fp8_paged_kernel,
    )

    compile_sm120_fmha_fp8_paged_kernel.cache_clear()
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    qo = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    last = torch.tensor([1], dtype=torch.int32, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo,
        indptr,
        indices,
        last,
        2,
        1,
        32,
        16,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
    )
    q = _fp8((1, 2, 32))
    combined = _fp8((1, 2, 1, 16, 32))
    cache_after_plan = compile_sm120_fmha_fp8_paged_kernel.cache_info()
    assert wrapper.run(q, combined, enable_pdl=False).shape == q.shape
    assert wrapper.run(q, combined, enable_pdl=True).shape == q.shape
    cache_after_runs = compile_sm120_fmha_fp8_paged_kernel.cache_info()
    assert cache_after_runs.misses == cache_after_plan.misses
    assert cache_after_runs.currsize == cache_after_plan.currsize
    assert cache_after_runs.hits == cache_after_plan.hits + 2
