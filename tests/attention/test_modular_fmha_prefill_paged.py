# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Paged-KV tests for the modular cute-dsl prefill kernel.

The anchor property: paged output must be BITWISE identical to ragged
output on the same logical problem — physical KV layout (page pool +
page table) is unobservable to the compute pipeline (same smem contents,
same schedule).  Each test scatters the ragged K/V into a page pool
under identity or scrambled page tables and compares against the ragged
kernel, so no separate numerical reference is needed.

NaN-canary cases pin the two clamp contracts (both are -1 page ids →
TMA out-of-bounds → zero-fill; P=+0 does NOT protect against NaN
because 0 x NaN = NaN through the PV MMA):
- tail clamp: logical pages past an item's last page;
- window clamp: with prefix caching, serving frameworks reclaim pages
  wholly below the attention window and point their table slots at a
  shared null block that accumulates garbage/NaN (the trtllm-gen
  pageIdxLb contract) — the kernel must not read them.
Live pages referenced by the table are finite everywhere by framework
contract (in-page tails past last_page_len are over-read and masked,
like every TMA-based paged kernel).

Each (mask, page_size) combination JIT-compiles one paged kernel (~30s);
runtime configurations reuse compiled kernels via the functools cache.
"""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import is_sm100a_supported

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL not available", allow_module_level=True)

import cutlass

from flashinfer.cute_dsl.attention.fusion.mask import MaskSpec
from flashinfer.cute_dsl.attention.wrappers.batch_prefill import (
    _get_compiled_prefill_kernel,
)

LOG2_E = math.log2(math.exp(1.0))


def _skip_unless_sm100():
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a (Blackwell)")


def build_paged(
    k_ragged,
    v_ragged,
    kv_indptr,
    page_size,
    scramble,
    pool_fill,
    seed=0,
    null_window_left=None,
    qo_indptr=None,
):
    """Scatter ragged K/V into a page pool; return pool tensors + tables.

    ``null_window_left`` simulates framework page reclamation under SWA:
    table slots for pages wholly below every Q row's window band point at
    a shared null page that keeps ``pool_fill`` (e.g. NaN) and never
    receives valid data.
    """
    device = k_ragged.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = kv_indptr.numel() - 1
    h_k, d = k_ragged.shape[1], k_ragged.shape[2]
    seq_lens = (kv_indptr[1:] - kv_indptr[:-1]).tolist()
    page_counts = [(s + page_size - 1) // page_size for s in seq_lens]
    total_pages = sum(page_counts)
    num_pool = total_pages + 5  # slack so physical ids differ from logical
    null_page = num_pool - 1  # never receives valid data; stays pool_fill

    k_pool = torch.full(
        (num_pool, page_size, h_k, d), pool_fill, dtype=k_ragged.dtype, device=device
    )
    v_pool = torch.full(
        (num_pool, page_size, h_k, d), pool_fill, dtype=v_ragged.dtype, device=device
    )

    if scramble:
        ids = torch.randperm(num_pool, generator=g)[:total_pages]
    else:
        ids = torch.arange(total_pages)

    page_table = ids.to(torch.int32).to(device)
    page_indptr = torch.zeros(B + 1, dtype=torch.int32)
    page_indptr[1:] = torch.cumsum(torch.tensor(page_counts), 0)
    page_indptr = page_indptr.to(device)

    flat = 0
    for b in range(B):
        base = int(kv_indptr[b])
        dead_cutoff = 0
        if null_window_left is not None:
            q_len = int(qo_indptr[b + 1] - qo_indptr[b])
            lo_min = max(0, (seq_lens[b] - q_len) - null_window_left)
            dead_cutoff = lo_min // page_size
        for p in range(page_counts[b]):
            phys = int(ids[flat])
            if p < dead_cutoff:
                page_table[flat] = null_page  # reclaimed slot
                flat += 1
                continue
            # Live pages are finite everywhere, including past
            # last_page_len (framework contract; matches trtllm-gen).
            k_pool[phys] = 0
            v_pool[phys] = 0
            lo = base + p * page_size
            hi = min(base + (p + 1) * page_size, int(kv_indptr[b + 1]))
            n = hi - lo
            k_pool[phys, :n] = k_ragged[lo:hi]
            v_pool[phys, :n] = v_ragged[lo:hi]
            flat += 1
    return k_pool, v_pool, page_table, page_indptr


def run_paged_vs_ragged(
    seq_lens_q,
    seq_lens_k,
    causal,
    page_size,
    scramble=True,
    pool_fill=777.0,
    window_left=-1,
    null_below_window=False,
    h_q=8,
    h_k=2,
    d=128,
    dt=torch.bfloat16,
):
    device = "cuda"
    B = len(seq_lens_q)
    qo_indptr = torch.zeros(B + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lens_q), 0)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(torch.tensor(seq_lens_k), 0)
    s_q_all, s_k_all = int(qo_indptr[-1]), int(kv_indptr[-1])
    max_s_q, max_s_k = max(seq_lens_q), max(seq_lens_k)

    torch.manual_seed(42)
    q = torch.randn(s_q_all, h_q, d, dtype=dt, device=device)
    k = torch.randn(s_k_all, h_k, d, dtype=dt, device=device)
    v = torch.randn(s_k_all, h_k, d, dtype=dt, device=device)

    k_pool, v_pool, page_table, page_indptr = build_paged(
        k,
        v,
        kv_indptr,
        page_size,
        scramble,
        pool_fill,
        null_window_left=(window_left if null_below_window else None),
        qo_indptr=qo_indptr,
    )

    mask_spec = MaskSpec(
        has_window_left=window_left >= 0,
        has_window_right=causal,
    )
    is_persistent = not (causal or window_left >= 0)
    common = (
        cutlass.BFloat16,
        cutlass.BFloat16,
        h_q,
        h_k,
        d,
        mask_spec,
        is_persistent,
        None,
        None,
    )
    kern_ragged = _get_compiled_prefill_kernel(*common)
    kern_paged = _get_compiled_prefill_kernel(*common, page_size=page_size)

    problem_size = (B, max_s_q, max_s_k, h_q, h_k, d)
    scale = (1.0 / math.sqrt(d)) * LOG2_E
    qo_i = qo_indptr.to(device)
    kv_i = kv_indptr.to(device)

    outs = []
    for kern, kk, vv, pt, pi in (
        (kern_ragged, k, v, None, None),
        (kern_paged, k_pool, v_pool, page_table, page_indptr),
    ):
        o_scratch = torch.full(
            (max_s_q + s_q_all, h_q, d), 7.0, dtype=dt, device=device
        )
        kern(
            q,
            kk,
            vv,
            o_scratch[max_s_q:],
            problem_size,
            qo_i,
            s_q_all,
            kv_i,
            s_k_all,
            scale,
            1.0,
            max(window_left, 0),
            0,
            None,
            None,
            pt,
            pi,
        )
        torch.cuda.synchronize()
        outs.append(o_scratch[max_s_q:].clone())

    ref, paged = outs
    assert not paged.float().isnan().any().item(), "NaN leaked into paged output"
    assert torch.equal(ref.view(torch.int16), paged.view(torch.int16)), (
        f"paged output not bitwise-equal to ragged "
        f"(max abs diff {(ref.float() - paged.float()).abs().max().item()})"
    )


SHAPES = [
    ([512, 512], [512, 512]),  # uniform, page-aligned
    ([7, 300, 1291], [23, 300, 1547]),  # varlen, partial last pages, s_k > s_q
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [8, 16, 64, 128])
@pytest.mark.parametrize("shapes", SHAPES)
def test_paged_bitwise_vs_ragged(causal, page_size, shapes):
    _skip_unless_sm100()
    sq, sk = shapes
    run_paged_vs_ragged(sq, sk, causal, page_size)


def test_paged_identity_table():
    """Identity table (pages in pool order) — isolates pure indirection."""
    _skip_unless_sm100()
    run_paged_vs_ragged([512, 512], [512, 512], True, 16, scramble=False)


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_d64_windowed_null_block(page_size):
    """head_dim 64 (single-slice TMA box geometry) x window clamp x NaN
    null blocks — the paged geometry the d128 cases don't reach."""
    _skip_unless_sm100()
    run_paged_vs_ragged(
        [1291],
        [1547],
        True,
        page_size,
        pool_fill=float("nan"),
        window_left=127,
        null_below_window=True,
        d=64,
    )


@pytest.mark.parametrize(
    "causal,page_size",
    [(True, 8), (True, 16), (False, 64)],
)
def test_paged_nan_pool_tail_clamp(causal, page_size):
    """Unreferenced pool pages NaN: the tail -1 clamp must keep them out."""
    _skip_unless_sm100()
    run_paged_vs_ragged(
        [7, 300, 1291], [23, 300, 1547], causal, page_size, pool_fill=float("nan")
    )


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([1024], [1024], False, 16, 255),
        ([1024], [1024], True, 8, 255),
        ([1291], [1547], True, 64, 511),
    ],
)
def test_paged_windowed(sq, sk, causal, page_size, window_left):
    """Windowed paged vs windowed ragged (all pages live)."""
    _skip_unless_sm100()
    run_paged_vs_ragged(sq, sk, causal, page_size, window_left=window_left)


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([300], [1500], False, 16, 127),
        ([1291], [1547], True, 64, 511),
        ([7, 300, 1291], [23, 300, 1547], True, 16, 127),
        ([7, 300, 1291], [23, 300, 1547], True, 8, 127),
    ],
)
def test_paged_null_block_window_clamp(sq, sk, causal, page_size, window_left):
    """Null-block contract: reclaimed out-of-window table slots point at a
    NaN null page; the window clamp (-1 page id) must never read them."""
    _skip_unless_sm100()
    run_paged_vs_ragged(
        sq,
        sk,
        causal,
        page_size,
        pool_fill=float("nan"),
        window_left=window_left,
        null_below_window=True,
    )


# ---------------------------------------------------------------------------
#  Wrapper-level integration (BatchPrefillWithPagedKVCacheWrapper cute-dsl)
# ---------------------------------------------------------------------------


def _build_wrapper_problem(
    seq_lens_q, seq_lens_k, page_size, h_q=8, h_k=2, d=128, dt=torch.bfloat16, seed=42
):
    """Random paged problem: returns q, ragged K/V, combined cache + tables."""
    device = "cuda"
    B = len(seq_lens_q)
    qo_indptr = torch.zeros(B + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lens_q), 0)
    torch.manual_seed(seed)
    q = torch.randn(int(qo_indptr[-1]), h_q, d, dtype=dt, device=device)

    page_counts = [(s + page_size - 1) // page_size for s in seq_lens_k]
    num_pool = sum(page_counts) + 4
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(num_pool, generator=g)[: sum(page_counts)]

    kv_indptr_tok = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr_tok[1:] = torch.cumsum(torch.tensor(seq_lens_k), 0)
    k_ragged = torch.randn(int(kv_indptr_tok[-1]), h_k, d, dtype=dt, device=device)
    v_ragged = torch.randn(int(kv_indptr_tok[-1]), h_k, d, dtype=dt, device=device)

    cache = torch.zeros(num_pool, 2, page_size, h_k, d, dtype=dt, device=device)
    flat = 0
    for b in range(B):
        base = int(kv_indptr_tok[b])
        for p in range(page_counts[b]):
            lo = base + p * page_size
            hi = min(base + (p + 1) * page_size, int(kv_indptr_tok[b + 1]))
            phys = int(ids[flat])
            cache[phys, 0, : hi - lo] = k_ragged[lo:hi]
            cache[phys, 1, : hi - lo] = v_ragged[lo:hi]
            flat += 1

    paged_kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    paged_kv_indptr[1:] = torch.cumsum(torch.tensor(page_counts), 0)
    last_page_len = torch.tensor(
        [(s - 1) % page_size + 1 for s in seq_lens_k], dtype=torch.int32
    )
    return (
        q,
        k_ragged,
        v_ragged,
        cache,
        qo_indptr.to(device),
        kv_indptr_tok.to(device),
        paged_kv_indptr.to(device),
        ids.to(torch.int32).to(device),
        last_page_len.to(device),
    )


def _paged_wrapper(kv_layout="NHD", backend="cute-dsl"):
    import flashinfer

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend=backend
    )


def test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper():
    """Windowed plans route to the modular kernel on BOTH wrappers, so the
    paged wrapper must be bitwise-equal to the ragged cute-dsl wrapper."""
    _skip_unless_sm100()
    import flashinfer

    (q, k_rag, v_rag, cache, qo, kv_tok, kv_pg, ids, lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], 16
    )
    w = _paged_wrapper()
    w.plan(
        qo,
        kv_pg,
        ids,
        lpl,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        causal=True,
        window_left=127,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out_paged = w.run(q, cache)

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w_rag = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cute-dsl")
    w_rag.plan(
        qo,
        kv_tok,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        causal=True,
        window_left=127,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    # Routing contract: windowed ragged plans must stay on the modular
    # kernel, or the bitwise comparison below compares different kernels.
    assert not w_rag._cute_dsl_use_fmha
    out_rag = w_rag.run(q, k_rag, v_rag)
    assert torch.equal(out_paged.view(torch.int16), out_rag.view(torch.int16))


def test_paged_wrapper_hnd_bitwise_vs_nhd():
    _skip_unless_sm100()
    (q, _, _, cache, qo, _, kv_pg, ids, lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], 16
    )
    plan_kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    w_nhd = _paged_wrapper("NHD")
    w_nhd.plan(qo, kv_pg, ids, lpl, **plan_kw)
    out_nhd = w_nhd.run(q, cache)

    w_hnd = _paged_wrapper("HND")
    w_hnd.plan(qo, kv_pg, ids, lpl, **plan_kw)
    out_hnd = w_hnd.run(q, cache.transpose(2, 3).contiguous())
    assert torch.equal(out_hnd.view(torch.int16), out_nhd.view(torch.int16))


def test_paged_wrapper_lse():
    _skip_unless_sm100()
    (q, _, _, cache, qo, _, kv_pg, ids, lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], 16
    )
    w = _paged_wrapper()
    w.plan(
        qo,
        kv_pg,
        ids,
        lpl,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out = w.run(q, cache)
    out_lse, lse = w.run(q, cache, return_lse=True)
    assert torch.equal(out_lse.view(torch.int16), out.view(torch.int16))
    assert lse.shape == (q.shape[0], 8) and torch.isfinite(lse).all().item()


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper(page_size):
    """Uniform fp8 paged vs fp8 ragged, both on the modular kernel via a
    windowed plan — bitwise, no tolerance calibration needed."""
    _skip_unless_sm100()
    import flashinfer

    dt = torch.float8_e4m3fn
    (q, k_rag, v_rag, cache, qo, kv_tok, kv_pg, ids, lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], page_size, dt=torch.bfloat16
    )
    q8, k8, v8 = q.to(dt), k_rag.to(dt), v_rag.to(dt)
    cache8 = cache.to(dt)
    w = _paged_wrapper()
    w.plan(
        qo,
        kv_pg,
        ids,
        lpl,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=page_size,
        causal=True,
        window_left=511,
        q_data_type=dt,
        kv_data_type=dt,
    )
    out_paged = w.run(q8, cache8)

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w_rag = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cute-dsl")
    w_rag.plan(
        qo,
        kv_tok,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        causal=True,
        window_left=511,
        q_data_type=dt,
        kv_data_type=dt,
    )
    # Routing contract: windowed ragged plans must stay on the modular
    # kernel, or the bitwise comparison below compares different kernels.
    assert not w_rag._cute_dsl_use_fmha
    out_rag = w_rag.run(q8, k8, v8)
    assert torch.equal(out_paged.view(torch.int16), out_rag.view(torch.int16))


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged(page_size):
    """Mixed-dtype paged cache: bf16 Q/K with an fp8 V cache (tuple form).

    The ragged route already serves mixed V as a run()-time property; the
    paged route reuses the same lazily compiled per-V-dtype kernel variant,
    so the outputs must be bitwise-equal to the ragged mixed run.
    """
    _skip_unless_sm100()

    (q, k_rag, v_rag, cache, qo, kv_tok, kv_pg, ids, lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], page_size
    )
    plan_kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        causal=True,
        # windowed on BOTH plans: keeps the ragged wrapper off its FMHA
        # route (a different kernel, not bitwise-comparable) and covers
        # the window clamp x paged x mixed-V combination.
        window_left=127,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    from flashinfer.cute_dsl.attention.wrappers.batch_prefill import (
        _dsl_supports_expected_tx,
    )

    w = _paged_wrapper()
    w.plan(qo, kv_pg, ids, lpl, page_size=page_size, **plan_kw)
    if not _dsl_supports_expected_tx():
        # Pre-4.6 DSL: mixed V must be rejected with an actionable error
        # (this asserts the version gate itself).
        with pytest.raises(NotImplementedError, match="nvidia-cutlass-dsl"):
            w.run(q, (cache[:, 0], cache[:, 1].to(torch.float8_e4m3fn)))
        return
    out_paged = w.run(q, (cache[:, 0], cache[:, 1].to(torch.float8_e4m3fn)))

    # Reference via the INTERNAL wrapper: it always runs the modular
    # kernel, so this comparison is pinned regardless of future changes
    # to the public ragged wrapper's FMHA-vs-modular routing policy.
    from flashinfer.cute_dsl.attention import BatchPrefillCuteDSLWrapper

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w_rag = BatchPrefillCuteDSLWrapper(ws)
    w_rag.plan(
        qo,
        kv_tok,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        causal=True,
        window_left=127,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out_rag = w_rag.run(q, k_rag, v_rag.to(torch.float8_e4m3fn))
    assert torch.equal(out_paged.view(torch.int16), out_rag.view(torch.int16))


@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("variant_name", ["sigmoid", "alibi", "sink"])
def test_paged_wrapper_variants_bitwise_vs_ragged(variant_name, page_size):
    """Attention variants must compose with paged KV.

    The loader is the only stage that differs between the paged and ragged
    plans, so under the same variant the outputs must be bitwise-equal.
    Covers one variant per fusion mechanism: sigmoid (logits transform —
    the softmax->epilogue pipeline topology), ALiBi (score_mod with
    per-head extra_params and position math), sink (statistics update +
    output transform in the correction path).
    """
    _skip_unless_sm100()
    from flashinfer.cute_dsl.attention import (
        ALiBiAttention,
        AttentionWithSink,
        BatchPrefillCuteDSLWrapper,
        SigmoidAttention,
    )

    (q, k_rag, v_rag, cache, qo, kv_tok, kv_pg, ids, lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], page_size
    )
    h_q = 8

    def make_variant():
        if variant_name == "sigmoid":
            return SigmoidAttention(scale=1.0 / math.sqrt(128), bias=-2.0)
        if variant_name == "alibi":
            return ALiBiAttention(
                torch.linspace(0.1, 0.9, h_q, dtype=torch.float32, device="cuda")
            )
        return AttentionWithSink(
            torch.linspace(-1.0, 1.0, h_q, dtype=torch.float32, device="cuda")
        )

    plan_kw = dict(
        num_qo_heads=h_q,
        num_kv_heads=2,
        head_dim_qk=128,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w_paged = BatchPrefillCuteDSLWrapper(ws)
    w_paged.plan(
        qo,
        page_size=page_size,
        paged_kv_indptr=kv_pg,
        paged_kv_indices=ids,
        paged_kv_last_page_len=lpl,
        variant=make_variant(),
        **plan_kw,
    )
    out_paged = w_paged.run_paged(q, cache)

    ws2 = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w_rag = BatchPrefillCuteDSLWrapper(ws2)
    w_rag.plan(qo, kv_tok, variant=make_variant(), **plan_kw)
    out_rag = w_rag.run(q, k_rag, v_rag)
    assert torch.equal(out_paged.view(torch.int16), out_rag.view(torch.int16))


def test_paged_wrapper_rejections():
    _skip_unless_sm100()
    (q, _, _, cache, qo, _, kv_pg, ids, lpl) = _build_wrapper_problem([300], [300], 16)
    plan_kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    # unsupported page size
    w = _paged_wrapper()
    with pytest.raises(ValueError, match="page_size"):
        w.plan(qo, kv_pg, ids, lpl, page_size=48, **plan_kw)
    # mixed plan dtypes
    w = _paged_wrapper()
    with pytest.raises(ValueError, match="kv_data_type"):
        w.plan(
            qo,
            kv_pg,
            ids,
            lpl,
            page_size=16,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
        )
    # mixed K at run (only V may differ from the plan); the top-level
    # wrapper's q/k dtype check fires first, the internal wrapper's
    # k_cache check backstops direct callers.
    w = _paged_wrapper()
    w.plan(qo, kv_pg, ids, lpl, page_size=16, **plan_kw)
    k_c, v_c = cache[:, 0].to(torch.float8_e4m3fn), cache[:, 1]
    with pytest.raises(ValueError, match=r"k_cache|dtype of k"):
        w.run(q, (k_c, v_c))
    # unsupported V dtype at run
    with pytest.raises(ValueError, match="v_cache"):
        w.run(q, (cache[:, 0], cache[:, 1].to(torch.float32)))
    # kv_cache_sf reject
    with pytest.raises(NotImplementedError, match=r"kv_cache_sf|NVFP4"):
        w.run(q, cache, kv_cache_sf=torch.zeros(1, device="cuda"))
    # zero-length KV item (would read page-table index -1 in the loader)
    w = _paged_wrapper()
    qo_e = torch.tensor([0, 300, 316], dtype=qo.dtype, device=qo.device)
    kv_pg_e = torch.cat([kv_pg, kv_pg[-1:]])  # second item: zero pages
    lpl_e = torch.cat([lpl, torch.zeros_like(lpl[:1])])
    with pytest.raises(ValueError, match="zero-length KV"):
        w.plan(qo_e, kv_pg_e, ids, lpl_e, page_size=16, **plan_kw)


def test_paged_wrapper_validate_inputs_nan_scan(monkeypatch):
    """FLASHINFER_VALIDATE_INPUTS=1 must catch NaN in a referenced page."""
    _skip_unless_sm100()
    (q, _, _, cache, qo, _, kv_pg, ids, lpl) = _build_wrapper_problem([300], [300], 16)
    w = _paged_wrapper()
    w.plan(
        qo,
        kv_pg,
        ids,
        lpl,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    # poison the tail of the last referenced page (past last_page_len)
    cache[int(ids[-1]), 1, -1] = float("nan")
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    with pytest.raises(ValueError, match="non-finite"):
        w.run(q, cache)
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "0")
    out = w.run(q, cache)  # scan off: runs (output corrupt by contract)
    assert out.shape == q.shape
