import math
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

import flashinfer
from flashinfer.utils import ceil_div


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


def _reference(q, k, v, causal, scale, window_left=-1, sinks=None):
    """FP32 attention with FlashInfer's bottom-right causal/window masks and sinks."""
    hq, hkv = q.shape[1], k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    vf = v.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * scale
    q_pos = torch.arange(q.shape[0], device=q.device) + k.shape[0] - q.shape[0]
    k_pos = torch.arange(k.shape[0], device=q.device)
    if causal:
        scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -torch.inf)
    if window_left >= 0:
        scores.masked_fill_(
            k_pos[None, None, :] < q_pos[None, :, None] - window_left, -torch.inf
        )
    if sinks is not None:
        sink_col = sinks.float()[:, None, None].expand(-1, q.shape[0], 1)
        scores = torch.cat([scores, sink_col], dim=-1)
    lse = torch.logsumexp(scores, -1).transpose(0, 1) * math.log2(math.e)
    probs = scores.softmax(-1)
    if sinks is not None:
        probs = probs[..., :-1]
    out = torch.einsum("hqk,hkd->hqd", probs, vf).transpose(0, 1)
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
    with pytest.raises(
        NotImplementedError, match="skip_softmax_threshold_scale_factor"
    ):
        paged_wrapper.run(q, cache, skip_softmax_threshold_scale_factor=2.0)

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
    with pytest.raises(NotImplementedError, match="logits_soft_cap"):
        ragged_wrapper.plan(
            indptr,
            indptr,
            hq,
            hkv,
            d,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            o_data_type=torch.float16,
            window_left=127,
            logits_soft_cap=30.0,
        )


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


def _check_paged(
    q_lens,
    kv_lens,
    hq,
    hkv,
    d,
    page_size,
    causal,
    window_left=-1,
    sinks=None,
    scales=(1.0, 1.0, 1.0),
    **plan_kwargs,
):
    """Run random paged inputs through prims, compare with the reference, return the backend."""
    q_scale, k_scale, v_scale = scales
    pages = [ceil_div(kv, page_size) for kv in kv_lens]
    qo = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    page_indptr = torch.tensor([0, *pages]).cumsum(0, dtype=torch.int32)
    page_indices = torch.randperm(sum(pages), device="cuda").to(torch.int32)
    last_page_len = torch.tensor(
        [kv - (p - 1) * page_size for kv, p in zip(kv_lens, pages, strict=True)],
        dtype=torch.int32,
        device="cuda",
    )
    q = _fp8((sum(q_lens), hq, d))
    cache = _fp8((sum(pages), 2, hkv, page_size, d))
    k_pool, v_pool = cache.unbind(dim=1)
    refs, ref_lses = [], []
    for b in range(len(q_lens)):
        idx = page_indices[page_indptr[b] : page_indptr[b + 1]]
        k = torch.cat([k_pool[p].transpose(0, 1) for p in idx])[: kv_lens[b]]
        v = torch.cat([v_pool[p].transpose(0, 1) for p in idx])[: kv_lens[b]]
        ref, ref_lse = _reference(
            q[qo[b] : qo[b + 1]],
            k,
            v,
            causal,
            q_scale * k_scale / math.sqrt(d),
            window_left,
            sinks,
        )
        refs.append(ref * v_scale)
        ref_lses.append(ref_lse)

    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo.cuda(),
        page_indptr.cuda(),
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        causal=causal,
        q_data_type=q.dtype,
        kv_data_type=cache.dtype,
        o_data_type=torch.bfloat16,
        window_left=window_left,
        **plan_kwargs,
    )
    kwargs = {}
    if scales != (1.0, 1.0, 1.0):
        kwargs = dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)
    out, lse = wrapper.run_return_lse(q, cache, sinks=sinks, **kwargs)
    torch.testing.assert_close(out.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, torch.cat(ref_lses), atol=0.2, rtol=0.2)
    return wrapper._prims_backend


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("split", ["auto", "fixed", "off"])
def test_paged_pack_and_split_match_reference(causal, split):
    backend = _check_paged(
        [1, 9, 130],
        [3000, 2000, 1500],
        16,
        2,
        128,
        16,
        causal,
        fixed_split_size=8 if split == "fixed" else None,
        disable_split_kv=split == "off",
    )
    assert backend._gqa_pack_size == 8
    assert (backend._kv_split is None) == (split == "off")


def test_paged_head_dim_32_stays_unsplit():
    assert _check_paged([8], [3000], 4, 2, 32, 32, True)._kv_split is None


def test_paged_kv_split_bounds():
    from flashinfer.attention.cute_dsl.sm120_fmha import _paged_kv_split_bounds

    # Long prefix: split, with every inner boundary inside the prefix all rows see.
    bounds = _paged_kv_split_bounds([2048], [34816], 16, True, 128, 188, 0, 128, 8)
    assert len(bounds[0]) > 2 and bounds[0][0] == 0 and bounds[0][-1] == 272
    assert max(bounds[0][1:-1]) <= (34816 - 2048) // 128
    # Fresh causal prompt: no prefix all rows see, so no split.
    assert _paged_kv_split_bounds([4096], [4096], 16, True, 128, 188, 0, 128, 8) is None
    # Fixed size: a request's bounds do not depend on the rest of the batch.
    alone = _paged_kv_split_bounds([64], [2112], 16, True, 128, 188, 512, 128, 8)
    batch = _paged_kv_split_bounds(
        [64, 8], [2112, 9000], 16, True, 128, 188, 512, 128, 8
    )
    assert alone == [[0, 4, 8, 12, 16, 17]]
    assert batch[0][: len(alone[0])] == alone[0] and set(batch[0][len(alone[0]) :]) == {
        17
    }
    # Fixed size needing more chunks than the float workspace holds.
    with pytest.raises(ValueError, match="float workspace"):
        _paged_kv_split_bounds([64], [2112], 16, True, 128, 188, 128, 4, 8)


def test_paged_workspace_size_fits_the_split():
    hq, hkv, d, page_size, q_lens, kv_lens = 16, 2, 128, 16, [8, 64], [3000, 2112]
    pages = [ceil_div(kv, page_size) for kv in kv_lens]
    args = (
        torch.tensor([0, *q_lens], device="cuda").cumsum(0, dtype=torch.int32),
        torch.tensor([0, *pages], device="cuda").cumsum(0, dtype=torch.int32),
        torch.arange(sum(pages), dtype=torch.int32, device="cuda"),
        torch.tensor(
            [kv - (p - 1) * page_size for kv, p in zip(kv_lens, pages, strict=True)],
            dtype=torch.int32,
            device="cuda",
        ),
        hq,
        hkv,
        d,
        page_size,
    )
    kwargs = dict(
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    chunks = []
    for size in (64 << 20, None):
        workspace = torch.empty(size or 1, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "HND", backend="cute-dsl-prims"
        )
        if size is None:
            float_bytes, int_bytes = wrapper.workspace_size(*args, **kwargs)
            assert float_bytes > 0 and int_bytes == 0
            assert wrapper.workspace_size(*args, **kwargs, disable_split_kv=True) == (
                0,
                0,
            )
            wrapper.reset_workspace_buffer(
                torch.empty(float_bytes, dtype=torch.uint8, device="cuda"),
                torch.empty(1, dtype=torch.uint8, device="cuda"),
            )
        wrapper.plan(*args, **kwargs)
        chunks.append(wrapper._prims_backend._kv_split.num_chunks)
    assert chunks[0] == chunks[1] > 1


def test_paged_fixed_split_is_batch_invariant():
    hq, hkv, d, page_size = 16, 2, 128, 16
    pages = [ceil_div(2112, page_size), ceil_div(9000, page_size)]
    q = _fp8((64 + 8, hq, d))
    cache = _fp8((sum(pages), 2, hkv, page_size, d))
    outs = []
    for kv_lens, q_lens in (([2112], [64]), ([2112, 9000], [64, 8])):
        n = len(kv_lens)
        workspace = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "HND", backend="cute-dsl-prims"
        )
        wrapper.plan(
            torch.tensor([0, *q_lens], device="cuda").cumsum(0, dtype=torch.int32),
            torch.tensor([0, *pages[:n]], device="cuda").cumsum(0, dtype=torch.int32),
            torch.arange(sum(pages[:n]), dtype=torch.int32, device="cuda"),
            torch.tensor(
                [
                    kv - (p - 1) * page_size
                    for kv, p in zip(kv_lens, pages, strict=False)
                ],
                dtype=torch.int32,
                device="cuda",
            ),
            hq,
            hkv,
            d,
            page_size,
            causal=True,
            q_data_type=q.dtype,
            kv_data_type=cache.dtype,
            o_data_type=torch.bfloat16,
            fixed_split_size=32,
        )
        assert wrapper._prims_backend._kv_split is not None
        outs.append(wrapper.run(q[: sum(q_lens)], cache)[:64])
    torch.testing.assert_close(outs[0], outs[1], atol=0, rtol=0)


@pytest.mark.parametrize("all_ones_v", [True, False])
def test_p_prescale_keeps_diffuse_tail(all_ones_v):
    """4 hot keys over 8188 cold ones: without the P pre-scale the cold P flush to 0 in FP8."""
    hq, hkv, d, q_len, n_kv, hot = 16, 2, 128, 128, 8192, 4
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.zeros((q_len, hq, d), device="cuda")
    q[..., 0] = 64.0
    k = torch.zeros((n_kv, hkv, d), device="cuda")
    k[..., 0] = 0.125
    k[-hot:, :, 0] = 1.375
    if all_ones_v:
        v = torch.ones((n_kv, hkv, d), device="cuda")
    else:
        v = torch.randn((1, hkv, d), generator=g, device="cuda")
        v = v + 0.25 * torch.randn((n_kv, hkv, d), generator=g, device="cuda")
        v[-hot:] = torch.randn((hot, hkv, d), generator=g, device="cuda")
    q, k, v = (x.to(torch.float8_e4m3fn) for x in (q, k, v))
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    wrapper.plan(
        torch.tensor([0, q_len], dtype=torch.int32, device="cuda"),
        torch.tensor([0, n_kv], dtype=torch.int32, device="cuda"),
        hq,
        hkv,
        d,
        causal=False,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=torch.float16,
    )
    out, lse = wrapper.run_return_lse(q, k, v)
    ref, ref_lse = _reference(q, k, v, False, 1 / math.sqrt(d))
    rel_rms = (out.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()
    assert rel_rms.item() < 0.03
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("split", ["auto", "off"])
def test_paged_page_size_1_matches_page_size_16_and_reference(kv_layout, causal, split):
    """page_size=1 matches page_size=16 bit-for-bit across HND/NHD and split-KV."""
    hq, hkv, d = 24, 4, 256
    q_lens = [1, 17, 130]
    kv_lens = [63, 257, 1025]
    total_q = sum(q_lens)
    qo = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    q = _fp8((total_q, hq, d))
    k_seqs = [_fp8((kv, hkv, d)) for kv in kv_lens]
    v_seqs = [_fp8((kv, hkv, d)) for kv in kv_lens]

    # Reference page_size=16 HND run
    pages16 = [ceil_div(kv, 16) for kv in kv_lens]
    p16_indptr = torch.tensor([0, *pages16], dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    p16_indices = torch.randperm(sum(pages16), device="cuda", dtype=torch.int32)
    p16_last = torch.tensor(
        [kv - (p - 1) * 16 for kv, p in zip(kv_lens, pages16, strict=True)],
        dtype=torch.int32,
        device="cuda",
    )
    cache16 = torch.zeros((sum(pages16), 2, hkv, 16, d), dtype=q.dtype, device="cuda")
    for b, (k_b, v_b) in enumerate(zip(k_seqs, v_seqs, strict=True)):
        idx = p16_indices[p16_indptr[b] : p16_indptr[b + 1]]
        for p_i, phys in enumerate(idx.tolist()):
            start = p_i * 16
            end = min(start + 16, k_b.shape[0])
            cache16[phys, 0, :, : end - start, :] = k_b[start:end].transpose(0, 1)
            cache16[phys, 1, :, : end - start, :] = v_b[start:end].transpose(0, 1)

    ws16 = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
    w16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws16, "HND", backend="cute-dsl-prims"
    )
    w16.plan(
        qo.cuda(),
        p16_indptr.cuda(),
        p16_indices,
        p16_last,
        hq,
        hkv,
        d,
        16,
        causal=causal,
        q_data_type=q.dtype,
        kv_data_type=cache16.dtype,
        o_data_type=torch.bfloat16,
        disable_split_kv=(split == "off"),
    )
    out16, lse16 = w16.run_return_lse(q, cache16)

    # page_size=1 run with 3D token-granular (k_pool, v_pool) tuple
    p1_indptr = torch.tensor([0, *kv_lens], dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    p1_indices = torch.randperm(sum(kv_lens), device="cuda", dtype=torch.int32)
    p1_last = torch.ones(len(kv_lens), dtype=torch.int32, device="cuda")
    k_pool1 = torch.zeros((sum(kv_lens), hkv, d), dtype=q.dtype, device="cuda")
    v_pool1 = torch.zeros((sum(kv_lens), hkv, d), dtype=q.dtype, device="cuda")
    for b, (k_b, v_b) in enumerate(zip(k_seqs, v_seqs, strict=True)):
        idx = p1_indices[p1_indptr[b] : p1_indptr[b + 1]]
        k_pool1[idx] = k_b
        v_pool1[idx] = v_b

    ws1 = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
    w1 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws1, kv_layout, backend="cute-dsl-prims"
    )
    w1.plan(
        qo.cuda(),
        p1_indptr.cuda(),
        p1_indices,
        p1_last,
        hq,
        hkv,
        d,
        1,
        causal=causal,
        q_data_type=q.dtype,
        kv_data_type=k_pool1.dtype,
        o_data_type=torch.bfloat16,
        disable_split_kv=(split == "off"),
    )
    out1, lse1 = w1.run_return_lse(q, (k_pool1, v_pool1))

    # Bit-for-bit match against page_size=16 (same swizzled SMEM + MMA pipeline).
    torch.testing.assert_close(out1, out16, atol=0, rtol=0)
    torch.testing.assert_close(lse1, lse16, atol=0, rtol=0)

    refs, ref_lses = [], []
    for b, (k_b, v_b) in enumerate(zip(k_seqs, v_seqs, strict=True)):
        ref, ref_lse = _reference(
            q[qo[b] : qo[b + 1]], k_b, v_b, causal, 1 / math.sqrt(d)
        )
        refs.append(ref)
        ref_lses.append(ref_lse)
    torch.testing.assert_close(out1.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse1, torch.cat(ref_lses), atol=0.2, rtol=0.2)


def test_cuda_graph_page_size_1_reads_updated_block_table():
    device, i32 = "cuda", torch.int32
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="cute-dsl-prims",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(2, dtype=i32, device=device),
        paged_kv_indptr_buf=torch.empty(2, dtype=i32, device=device),
        paged_kv_indices_buf=torch.empty(4, dtype=i32, device=device),
        paged_kv_last_page_len_buf=torch.empty(1, dtype=i32, device=device),
    )

    hq, hkv, head_dim = 4, 2, 128
    q = torch.ones((2, hq, head_dim), device=device).to(torch.float8_e4m3fn)
    k_pool = torch.zeros((8, hkv, head_dim), dtype=torch.float8_e4m3fn, device=device)
    v_pool = torch.zeros((8, hkv, head_dim), dtype=torch.float8_e4m3fn, device=device)
    v_pool[:4].fill_(1)
    v_pool[4:].fill_(-1)

    qo_indptr = torch.tensor([0, 2], dtype=i32, device=device)
    kv_indptr = torch.tensor([0, 4], dtype=i32, device=device)
    page_indices = torch.tensor([0, 1, 2, 3], dtype=i32, device=device)
    last_page_len = torch.tensor([1], dtype=i32, device=device)
    block_tables = torch.tensor([[0, 1, 2, 3]], dtype=i32, device=device)

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        head_dim,
        1,
        block_tables=block_tables,
        q_data_type=q.dtype,
        kv_data_type=k_pool.dtype,
        o_data_type=torch.bfloat16,
    )

    captured = torch.empty((2, hq, head_dim), dtype=torch.bfloat16, device=device)
    for _ in range(3):
        wrapper.run(q, (k_pool, v_pool), out=captured, enable_pdl=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, (k_pool, v_pool), out=captured, enable_pdl=False)

    block_tables.copy_(torch.tensor([[4, 5, 6, 7]], dtype=i32, device=device))
    eager = torch.empty_like(captured)
    wrapper.run(q, (k_pool, v_pool), out=eager, enable_pdl=False)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
    torch.testing.assert_close(eager, torch.full_like(eager, -1))


# ---------------------------------------------------------------------------
# Sliding window and attention sinks
# ---------------------------------------------------------------------------

_LOG2E = math.log2(math.e)


def _sinks(hq, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(hq, generator=g, device="cuda") * 2.0 + 1.0


def _run_ragged(q, k, v, q_lens, kv_lens, causal, window_left=-1, sinks=None):
    hq, d = q.shape[1], q.shape[2]
    qo = torch.tensor([0, *q_lens], device="cuda").cumsum(0, dtype=torch.int32)
    kv = torch.tensor([0, *kv_lens], device="cuda").cumsum(0, dtype=torch.int32)
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo,
        kv,
        hq,
        k.shape[1],
        d,
        causal=causal,
        window_left=window_left,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=torch.bfloat16,
    )
    # vLLM's metadata builder stashes attention sinks on the wrapper.
    wrapper._sinks = sinks
    return wrapper.run_return_lse(q, k, v)


def _ragged_reference(q, k, v, q_lens, kv_lens, causal, window_left, sinks):
    refs, lses = [], []
    qo, kv = 0, 0
    for ql, kl in zip(q_lens, kv_lens, strict=True):
        ref, ref_lse = _reference(
            q[qo : qo + ql],
            k[kv : kv + kl],
            v[kv : kv + kl],
            causal,
            1 / math.sqrt(q.shape[2]),
            window_left,
            sinks,
        )
        refs.append(ref)
        lses.append(ref_lse)
        qo, kv = qo + ql, kv + kl
    return torch.cat(refs), torch.cat(lses)


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_ragged_sliding_window_and_sinks_match_reference(d, causal, with_sinks):
    """Windows of 0, one tile, across a tile boundary and longer than every request."""
    hq, hkv = 8, 2
    q_lens = [300, 1, 129]
    # Non-causal also covers q_len > kv_len, where the window start is negative.
    kv_lens = [300, 700, 385] if causal else [129, 700, 385]
    q = _fp8((sum(q_lens), hq, d))
    k, v = _fp8((sum(kv_lens), hkv, d)), _fp8((sum(kv_lens), hkv, d))
    sinks = _sinks(hq) if with_sinks else None
    for window_left in (0, 1, 63, 127, 200, 511, 1 << 20):
        out, lse = _run_ragged(q, k, v, q_lens, kv_lens, causal, window_left, sinks)
        ref, ref_lse = _ragged_reference(
            q, k, v, q_lens, kv_lens, causal, window_left, sinks
        )
        torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.2)
        torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.2)


@pytest.mark.parametrize("causal", [True, False])
def test_sinks_are_not_scaled_by_fp8_q_and_k_scales(causal):
    """Sinks stay in sm_scale * q.k units while q_scale * k_scale folds into the scores."""
    hq, hkv, d, window_left = 8, 2, 128, 63
    scales = (2.0, 1.5, 0.5)
    sinks = _sinks(hq, seed=1)
    # Paged, unsplit.
    _check_paged(
        [1, 9, 130],
        [700, 300, 385],
        hq,
        hkv,
        d,
        16,
        causal,
        window_left=window_left,
        sinks=sinks,
        scales=scales,
        disable_split_kv=True,
    )
    # Ragged: also check that a sink scaled by q_scale * k_scale would be caught.
    q_scale, k_scale, v_scale = scales
    q_lens = [300, 1, 129]
    kv_lens = [300, 700, 385] if causal else [129, 700, 385]
    q = _fp8((sum(q_lens), hq, d))
    k, v = _fp8((sum(kv_lens), hkv, d)), _fp8((sum(kv_lens), hkv, d))
    qo = torch.tensor([0, *q_lens], device="cuda").cumsum(0, dtype=torch.int32)
    kv = torch.tensor([0, *kv_lens], device="cuda").cumsum(0, dtype=torch.int32)
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
        window_left=window_left,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=torch.bfloat16,
    )
    wrapper._sinks = sinks
    out, lse = wrapper.run_return_lse(
        q, k, v, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
    )

    def reference(sink_values):
        refs, lses = [], []
        for i, (ql, kl) in enumerate(zip(q_lens, kv_lens, strict=True)):
            ref, ref_lse = _reference(
                q[qo[i] : qo[i] + ql],
                k[kv[i] : kv[i] + kl],
                v[kv[i] : kv[i] + kl],
                causal,
                q_scale * k_scale / math.sqrt(d),
                window_left,
                sink_values,
            )
            refs.append(ref * v_scale)
            lses.append(ref_lse)
        return torch.cat(refs), torch.cat(lses)

    ref, ref_lse = reference(sinks)
    torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.2)
    _, wrong_lse = reference(sinks * q_scale * k_scale)
    assert (lse - wrong_lse).abs().max() > 1.0


@pytest.mark.parametrize("causal", [True, False])
def test_ragged_extreme_sinks_match_reference(causal):
    """Sinks far above and far below every score, mixed across heads, with a window."""
    hq, hkv, d = 8, 2, 128
    q_lens = [300, 1, 129]
    kv_lens = [300, 700, 385] if causal else [129, 700, 385]
    q = _fp8((sum(q_lens), hq, d))
    k, v = _fp8((sum(kv_lens), hkv, d)), _fp8((sum(kv_lens), hkv, d))
    sinks = torch.tensor(
        [40.0, -40.0, 20.0, -20.0, 0.0, 60.0, -60.0, 5.0], device="cuda"
    )
    for window_left in (-1, 63):
        out, lse = _run_ragged(q, k, v, q_lens, kv_lens, causal, window_left, sinks)
        ref, ref_lse = _ragged_reference(
            q, k, v, q_lens, kv_lens, causal, window_left, sinks
        )
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()
        torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.2)
        torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.2)


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_ragged_window_wider_than_sequence_matches_no_window(d, causal):
    hq, hkv, q_lens, kv_lens = 8, 2, [300, 129], [300, 385]
    q = _fp8((sum(q_lens), hq, d))
    k, v = _fp8((sum(kv_lens), hkv, d)), _fp8((sum(kv_lens), hkv, d))
    windowed = _run_ragged(q, k, v, q_lens, kv_lens, causal, window_left=385)
    dense = _run_ragged(q, k, v, q_lens, kv_lens, causal)
    torch.testing.assert_close(windowed[0], dense[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(windowed[1], dense[1], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "page_size,causal,split,with_sinks",
    [
        (16, causal, split, with_sinks)
        for causal in (True, False)
        for split in ("fixed", "off")
        for with_sinks in (False, True)
    ]
    + [(1, True, "fixed", True), (1, True, "off", True), (1, False, "off", False)],
)
def test_paged_sliding_window_and_sinks_match_reference(
    page_size, causal, split, with_sinks
):
    """Paged K/V with GQA packing, split KV inside the window, and sinks in chunk 0."""
    hq, hkv, d = 16, 2, 128
    sinks = _sinks(hq) if with_sinks else None
    for window_left in (0, 127, 300):
        backend = _check_paged(
            [1, 9, 130],
            [3000, 2000, 1500],
            hq,
            hkv,
            d,
            page_size,
            causal,
            window_left=window_left,
            sinks=sinks,
            fixed_split_size=(128 // page_size) if split == "fixed" else None,
            disable_split_kv=split == "off",
        )
        assert backend._gqa_pack_size == 8
        if split == "off" or (causal and window_left == 0):
            # A causal zero window leaves one tile per request: nothing to split.
            assert backend._kv_split is None
        else:
            split_plan = backend._kv_split
            assert split_plan is not None and split_plan.num_chunks > 1
            ranges = split_plan.kv_tile_ranges.cpu()
            begins = [
                max(kv - q - window_left, 0) // 128
                for q, kv in ((1, 3000), (9, 2000), (130, 1500))
            ]
            # Chunk 0 of every request starts at its first windowed tile.
            assert ranges[:3, 0].tolist() == begins


def test_gpt_oss_shape_window_128_with_sinks():
    """GPT-OSS sliding layers: Hq=64, Hkv=8, D=64, sliding_window=128 (window_left=127)."""
    hq, hkv, d, window_left = 64, 8, 64, 127
    sinks = _sinks(hq, seed=1)
    # Decode-like rows on a long prefix: the window covers two tiles, split in two.
    backend = _check_paged(
        [16, 3], [16384, 9000], hq, hkv, d, 16, True, window_left, sinks
    )
    assert backend._kv_split is not None
    # Prefill chunk on a prefix, unsplit by default for many CTAs.
    _check_paged([1024], [4096], hq, hkv, d, 16, True, window_left, sinks)
    q_lens, kv_lens = [1024, 64], [1024, 2048]
    q = _fp8((sum(q_lens), hq, d))
    k, v = _fp8((sum(kv_lens), hkv, d)), _fp8((sum(kv_lens), hkv, d))
    out, lse = _run_ragged(q, k, v, q_lens, kv_lens, True, window_left, sinks)
    ref, ref_lse = _ragged_reference(q, k, v, q_lens, kv_lens, True, window_left, sinks)
    torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.2)


def test_gemma_local_layer_shape_window_1024():
    """Gemma 4 31B local layers: Hq=32, Hkv=16, D=256, sliding_window=1024."""
    hq, hkv, d, window_left = 32, 16, 256, 1023
    backend = _check_paged([256], [8192], hq, hkv, d, 16, True, window_left)
    assert backend._kv_split is not None
    q_lens = kv_lens = [2048]
    q = _fp8((2048, hq, d))
    k, v = _fp8((2048, hkv, d)), _fp8((2048, hkv, d))
    out, lse = _run_ragged(q, k, v, q_lens, kv_lens, True, window_left)
    ref, ref_lse = _ragged_reference(q, k, v, q_lens, kv_lens, True, window_left, None)
    torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.2)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("kv_mode", ["ragged", "paged-split", "paged"])
def test_window_never_reads_tiles_left_of_the_window(causal, kv_mode):
    """K/V tiles wholly left of every row's window hold NaN: they must be skipped,
    not loaded and masked (0 * NaN would poison the PV product)."""
    hq, hkv, d, window_left, q_len, kv_len = 8, 2, 128, 127, 128, 4096
    first_tile = (kv_len - q_len - window_left) // 128
    q = _fp8((q_len, hq, d))
    k, v = _fp8((kv_len, hkv, d)), _fp8((kv_len, hkv, d))
    ref, ref_lse = _reference(q, k, v, causal, 1 / math.sqrt(d), window_left)
    nan = torch.tensor(float("nan")).to(torch.float8_e4m3fn)
    k[: first_tile * 128] = nan
    v[: first_tile * 128] = nan
    if kv_mode == "ragged":
        out, lse = _run_ragged(q, k, v, [q_len], [kv_len], causal, window_left)
    else:
        page_size = 16
        cache = torch.stack(
            [
                x.view(kv_len // page_size, page_size, hkv, d).transpose(1, 2)
                for x in (k, v)
            ],
            dim=1,
        ).contiguous()
        num_pages = kv_len // page_size
        workspace = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "HND", backend="cute-dsl-prims"
        )
        wrapper.plan(
            torch.tensor([0, q_len], dtype=torch.int32, device="cuda"),
            torch.tensor([0, num_pages], dtype=torch.int32, device="cuda"),
            torch.arange(num_pages, dtype=torch.int32, device="cuda"),
            torch.tensor([page_size], dtype=torch.int32, device="cuda"),
            hq,
            hkv,
            d,
            page_size,
            causal=causal,
            window_left=window_left,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            o_data_type=torch.bfloat16,
            fixed_split_size=8 if kv_mode == "paged-split" else None,
            disable_split_kv=kv_mode == "paged",
        )
        assert (wrapper._prims_backend._kv_split is not None) == (
            kv_mode == "paged-split"
        )
        out, lse = wrapper.run_return_lse(q, cache)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.2)


@pytest.mark.parametrize("window_left", [-1, 3])
def test_ragged_sinks_with_an_empty_kv_request(window_left):
    """A request with no keys softmaxes over its sink alone: O = 0, LSE = sink."""
    hq, hkv, d, q_lens, kv_lens = 4, 2, 64, [3, 2], [0, 5]
    q = _fp8((5, hq, d))
    k, v = _fp8((5, hkv, d)), _fp8((5, hkv, d))
    sinks = _sinks(hq)
    out, lse = _run_ragged(q, k, v, q_lens, kv_lens, False, window_left, sinks)
    ref, ref_lse = _ragged_reference(
        q, k, v, q_lens, kv_lens, False, window_left, sinks
    )
    torch.testing.assert_close(out[:3].float(), torch.zeros_like(ref[:3]))
    torch.testing.assert_close(lse[:3], (sinks * _LOG2E).expand(3, hq))
    torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.2)


def test_cuda_graph_replays_window_and_sinks():
    hq, hkv, d, page_size, window_left = 16, 2, 128, 16, 127
    q_lens, kv_lens = [64, 1], [1000, 300]
    pages = [ceil_div(kv, page_size) for kv in kv_lens]
    q = _fp8((sum(q_lens), hq, d))
    cache = _fp8((sum(pages), 2, hkv, page_size, d))
    sinks = _sinks(hq)
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "HND",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(3, dtype=torch.int32, device="cuda"),
        paged_kv_indptr_buf=torch.empty(3, dtype=torch.int32, device="cuda"),
        paged_kv_indices_buf=torch.empty(sum(pages), dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len_buf=torch.empty(2, dtype=torch.int32, device="cuda"),
        backend="cute-dsl-prims",
    )
    wrapper.plan(
        torch.tensor([0, *q_lens], device="cuda").cumsum(0, dtype=torch.int32),
        torch.tensor([0, *pages], device="cuda").cumsum(0, dtype=torch.int32),
        torch.arange(sum(pages), dtype=torch.int32, device="cuda"),
        torch.tensor(
            [kv - (p - 1) * page_size for kv, p in zip(kv_lens, pages, strict=True)],
            dtype=torch.int32,
            device="cuda",
        ),
        hq,
        hkv,
        d,
        page_size,
        causal=True,
        window_left=window_left,
        q_data_type=q.dtype,
        kv_data_type=cache.dtype,
        o_data_type=torch.bfloat16,
    )
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):  # warm-up compiles the sinks variant before capture
        wrapper.run(q, cache, out=out, sinks=sinks)
    torch.cuda.synchronize()
    expected = out.clone()
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, cache, out=out, sinks=sinks)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    # Sinks are read on device at replay time.
    sinks.add_(3.0)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(out, expected)
