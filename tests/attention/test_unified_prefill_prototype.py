"""Conformance matrix for the unified paged-prefill prototype.

One backend-parametrized test, one reference oracle, one output contract:
- outputs match the fp32 paged-attention oracle
- LSE is ALWAYS base-2, packed (total_q_tokens, num_qo_heads), fp32 —
  including for cuDNN, whose native natural-log padded stats are normalized
  in its adapter.  This is the first test in the repo that pins the cuDNN
  LSE base against an independent reference.

This file doubles as executable documentation of the API (proposal P0).
"""

import zlib

import pytest
import torch

from flashinfer.attention.unified import (
    UnifiedPagedPrefill,
    resolve_paged_prefill,
)

from .unified_prefill_reference import reference_paged_prefill

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen", "auto"]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def make_problem(
    seed,
    *,
    batch_size,
    max_q,
    max_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo=None,
    page_size,
    dtype,
    device="cuda:0",
    uniform_q1=False,
    kv_layout="HND",
    input_form="block_tables",
):
    """Random valid paged-prefill problem with scattered (non-identity) page ids."""
    head_dim_vo = head_dim_vo or head_dim_qk
    g = torch.Generator().manual_seed(seed)
    if uniform_q1:
        q_lens = torch.ones(batch_size, dtype=torch.int32)
    else:
        q_lens = torch.randint(
            1, max_q + 1, (batch_size,), generator=g, dtype=torch.int32
        )
    kv_extra = torch.randint(
        0, max_kv - 1, (batch_size,), generator=g, dtype=torch.int32
    )
    kv_lens = torch.minimum(q_lens + kv_extra, torch.tensor(max_kv, dtype=torch.int32))

    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    pages_per_seq = (kv_lens + page_size - 1) // page_size
    width = int(pages_per_seq.max())
    pool_pages = int(pages_per_seq.sum()) + 8  # slack: unused pool pages
    perm = torch.randperm(pool_pages, generator=g, dtype=torch.int32)
    block_tables_cpu = torch.zeros(batch_size, width, dtype=torch.int32)
    off = 0
    for i in range(batch_size):
        n = int(pages_per_seq[i])
        block_tables_cpu[i, :n] = perm[off : off + n]
        off += n

    total_q = int(qo_indptr_cpu[-1])
    q = torch.randn(total_q, num_qo_heads, head_dim_qk, dtype=dtype, device=device)
    if kv_layout == "HND":
        k_shape = (pool_pages, num_kv_heads, page_size, head_dim_qk)
        v_shape = (pool_pages, num_kv_heads, page_size, head_dim_vo)
    else:  # NHD
        k_shape = (pool_pages, page_size, num_kv_heads, head_dim_qk)
        v_shape = (pool_pages, page_size, num_kv_heads, head_dim_vo)
    k_cache = torch.randn(*k_shape, dtype=dtype, device=device)
    v_cache = torch.randn(*v_shape, dtype=dtype, device=device)

    # flat CSR page-id list: request-ordered concatenation of each row's
    # live prefix (same info as the dense table)
    kv_page_indices_cpu = torch.cat(
        [block_tables_cpu[i, : int(pages_per_seq[i])] for i in range(batch_size)]
    ).to(torch.int32)

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        qo_indptr=qo_indptr_cpu.to(device),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(device),
        kv_seq_lens_cpu=kv_lens,
        block_tables=block_tables_cpu.to(device),
        kv_page_indices=kv_page_indices_cpu.to(device),
        kv_layout=kv_layout,
        input_form=input_form,
        page_size=page_size,
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens.max()),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        dtype=dtype,
        device=device,
    )


def run_unified(
    p,
    backend,
    *,
    causal=True,
    return_lse=True,
    with_mirrors=True,
    sm_scale=None,
    window_left=-1,
):
    attn = UnifiedPagedPrefill(torch.device(p["device"]))
    form = p.get("input_form", "block_tables")
    attn.plan(
        qo_indptr=p["qo_indptr"],
        kv_seq_lens=p["kv_seq_lens"],
        block_tables=(p["block_tables"] if form in ("block_tables", "both") else None),
        kv_page_indices=(
            p["kv_page_indices"] if form in ("page_indices", "both") else None
        ),
        page_size=p["page_size"],
        max_q_len=p["max_q_len"],
        max_kv_len=p["max_kv_len"],
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_layout=p.get("kv_layout", "HND"),
        causal=causal,
        window_left=window_left,
        return_lse=return_lse,
        qo_indptr_cpu=p["qo_indptr_cpu"] if with_mirrors else None,
        kv_seq_lens_cpu=p["kv_seq_lens_cpu"] if with_mirrors else None,
        sm_scale=sm_scale,
        backend=backend,
    )
    out, lse = attn.run(
        p["q"],
        (p["k_cache"], p["v_cache"]),
        out=p.get("_out_override"),
        lse=p.get("_lse_override"),
    )
    return attn, out, lse


def _resolve_or_skip(p, backend, *, causal=True, need_lse=True, window_left=-1):
    try:
        return resolve_paged_prefill(
            device=torch.device(p["device"]),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim_qk"],
            head_dim_vo=p["head_dim_vo"],
            q_dtype=p["dtype"],
            page_size=p["page_size"],
            kv_layout=p.get("kv_layout", "HND"),
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            kv_input_form=(
                "page_indices"
                if p.get("input_form") == "page_indices"
                else "block_tables"
            ),
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"backend {backend} not runnable here: {e}")


def check(p, backend, *, causal=True, window_left=-1):
    _resolve_or_skip(p, backend, causal=causal, window_left=window_left)
    _, out, lse = run_unified(p, backend, causal=causal, window_left=window_left)
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_cache"],
        p["v_cache"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"] if p.get("input_form") != "page_indices" else None,
        p["page_size"],
        causal,
        window_left=window_left,
        kv_layout=p.get("kv_layout", "HND"),
        kv_page_indices=p.get("kv_page_indices"),
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    # Output contract: LSE base-2, packed (tokens, h), fp32 — for everyone.
    assert lse.shape == (p["q"].shape[0], p["num_qo_heads"])
    assert lse.dtype == torch.float32
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "batch_size,max_q,max_kv,heads,page_size",
    [
        (4, 64, 512, (8, 8), 16),  # MHA
        (4, 64, 512, (8, 2), 16),  # GQA
        (3, 48, 300, (8, 1), 32),  # MQA, ragged, page 32
        (1, 128, 128, (4, 4), 64),  # single request
    ],
)
def test_unified_prefill_conformance(
    backend, batch_size, max_q, max_kv, heads, page_size
):
    p = make_problem(
        seed=zlib.crc32(
            repr((backend, batch_size, max_q, max_kv, heads, page_size)).encode()
        ),
        batch_size=batch_size,
        max_q=max_q,
        max_kv=max_kv,
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim_qk=128,
        page_size=page_size,
        dtype=torch.bfloat16,
    )
    check(p, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unified_prefill_decode_shape(backend):
    """Uniform q_len=1 through the same API — decode is a special case of the
    unified contract, not a different world (proposal / PD-统一 argument)."""
    p = make_problem(
        seed=7,
        batch_size=8,
        max_q=1,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        uniform_q1=True,
    )
    check(p, backend)


def test_unified_prefill_noncausal_fa2():
    p = make_problem(
        seed=11,
        batch_size=4,
        max_q=32,
        max_kv=128,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, "fa2", causal=False)


def test_unified_prefill_no_mirrors_documented_sync():
    """Without host mirrors the facade does one documented D2H and results
    are identical — the sync is a perf note, never a semantics change."""
    p = make_problem(
        seed=13,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    _resolve_or_skip(p, "fa2")
    _, out_a, lse_a = run_unified(p, "fa2", with_mirrors=True)
    _, out_b, lse_b = run_unified(p, "fa2", with_mirrors=False)
    assert torch.equal(out_a, out_b)
    assert torch.equal(lse_a, lse_b)


def test_resolve_is_static_and_explains():
    """resolve_paged_prefill needs no tensors — callable at engine init —
    and reports per-backend exclusion reasons (the anti-rot 'explain')."""
    res = resolve_paged_prefill(
        cc_major=9,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        page_size=16,
        causal=True,
        need_lse=True,
    )
    assert res.chosen == res.backends[0]
    assert "trtllm-gen" in res.excluded  # sm_90 excluded, with a reason
    assert "sm_9" in res.excluded["trtllm-gen"]
    # explicit pin of an impossible backend raises with the reason
    with pytest.raises(ValueError, match="compute capability"):
        resolve_paged_prefill(
            cc_major=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            page_size=16,
            backend="trtllm-gen",
        )
    # GQA violation is a contract error, not a backend error
    with pytest.raises(ValueError, match="divisible"):
        resolve_paged_prefill(
            cc_major=9,
            num_qo_heads=7,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            page_size=16,
        )


@pytest.mark.parametrize("backend", ["cudnn", "fa2"])
def test_unified_prefill_headdim_192_128(backend):
    """(192,128) head dims — capability-honesty: declared rows are tested."""
    if backend == "fa2":
        pytest.skip("fa2 (192,128) not declared in the prototype capability set")
    p = make_problem(
        seed=17,
        batch_size=3,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=192,
        head_dim_vo=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, backend)


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn"])
def test_unified_prefill_noncausal(backend):
    p = make_problem(
        seed=11,
        batch_size=4,
        max_q=32,
        max_kv=128,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, backend, causal=False)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unified_prefill_sm_scale_replan(backend):
    """Two plans with identical shapes but different sm_scale must each be
    correct.  Regression for the cuDNN graph-cache stale-scale replay (the
    cache key omitted attn_scale; found by this prototype's fuzzer, fixed in
    flashinfer/cudnn/prefill.py)."""
    p = make_problem(
        seed=19,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    _resolve_or_skip(p, backend)
    for scale_mult in (1.0, 3.0):
        sm_scale = scale_mult / (128**0.5)
        _, out, lse = run_unified(p, backend, sm_scale=sm_scale)
        ref_out, ref_lse = reference_paged_prefill(
            p["q"],
            p["k_cache"],
            p["v_cache"],
            p["qo_indptr_cpu"],
            p["kv_seq_lens_cpu"],
            p["block_tables"],
            p["page_size"],
            True,
            sm_scale=sm_scale,
        )
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_resolution_pinning():
    """plan(backend=Resolution) enforces the init-time pinned config."""
    p = make_problem(
        seed=23,
        batch_size=2,
        max_q=16,
        max_kv=64,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    res = _resolve_or_skip(p, "auto")
    attn = UnifiedPagedPrefill(torch.device(p["device"]))
    attn.plan(
        qo_indptr=p["qo_indptr"],
        kv_seq_lens=p["kv_seq_lens"],
        block_tables=p["block_tables"],
        page_size=p["page_size"],
        max_q_len=p["max_q_len"],
        max_kv_len=p["max_kv_len"],
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        return_lse=True,
        qo_indptr_cpu=p["qo_indptr_cpu"],
        kv_seq_lens_cpu=p["kv_seq_lens_cpu"],
        backend=res,
    )
    assert attn._backend in res.backends
    # drifted config (different heads) must be rejected, not silently re-resolved
    with pytest.raises(ValueError, match="pinned Resolution"):
        attn.plan(
            qo_indptr=p["qo_indptr"],
            kv_seq_lens=p["kv_seq_lens"],
            block_tables=p["block_tables"],
            page_size=p["page_size"],
            max_q_len=p["max_q_len"],
            max_kv_len=p["max_kv_len"],
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_qo_heads"],  # MHA instead of GQA
            head_dim_qk=p["head_dim_qk"],
            q_dtype=p["dtype"],
            causal=True,
            return_lse=True,
            backend=res,
        )


def test_envelope_rejections():
    """Zero-length KV rows and causal q_len>kv_len are outside the v1
    envelope and must be rejected loudly (backends disagree on the LSE of
    fully-masked rows: fa2 finite sentinel vs cudnn -inf)."""
    p = make_problem(
        seed=29,
        batch_size=3,
        max_q=8,
        max_kv=64,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    kv0 = p["kv_seq_lens_cpu"].clone()
    kv0[1] = 0
    with pytest.raises(ValueError, match="outside the v1 envelope"):
        run_unified(
            {**p, "kv_seq_lens_cpu": kv0, "kv_seq_lens": kv0.to(p["device"])}, "fa2"
        )
    # causal q>kv: force q_len 8 > kv_len 4 on request 0
    kvq = p["kv_seq_lens_cpu"].clone()
    q_lens = p["qo_indptr_cpu"].diff()
    kvq[0] = max(1, int(q_lens[0]) - 1)
    with pytest.raises(ValueError, match="q_len_i <= kv_len_i"):
        run_unified(
            {**p, "kv_seq_lens_cpu": kvq, "kv_seq_lens": kvq.to(p["device"])}, "fa2"
        )


def test_derive_is_sync_free():
    """The derivation layer must not synchronize (proposal P1 acceptance:
    with mirrors, plan() is zero-D2H).  Guards against masked-select /
    repeat_interleave style data-dependent-size ops sneaking back in."""
    from flashinfer.attention.unified import _derive

    p = make_problem(
        seed=31,
        batch_size=6,
        max_q=16,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        d = _derive(
            p["qo_indptr"],
            p["kv_seq_lens"],
            p["block_tables"],
            None,
            p["page_size"],
            p["max_kv_len"],
            needs_dense=False,
        )
        # reverse direction: flat indices -> dense, also zero-sync
        d2 = _derive(
            p["qo_indptr"],
            p["kv_seq_lens"],
            None,
            p["kv_page_indices"],
            p["page_size"],
            p["max_kv_len"],
            needs_dense=True,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    # correctness of the scatter-compaction vs a host-side reference
    pages = (p["kv_seq_lens_cpu"] + p["page_size"] - 1) // p["page_size"]
    expected = torch.cat(
        [
            p["block_tables"].cpu()[i, : int(pages[i])]
            for i in range(p["kv_seq_lens_cpu"].shape[0])
        ]
    )
    total_pages = int(pages.sum())
    assert torch.equal(d.kv_page_indices.cpu()[:total_pages], expected)
    # CSR->dense round trip: live prefix of each derived dense row matches
    dense = d2.block_tables.cpu()
    for i in range(p["kv_seq_lens_cpu"].shape[0]):
        n = int(pages[i])
        assert torch.equal(dense[i, :n], p["block_tables"].cpu()[i, :n])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("window_left", [0, 16, 127])
def test_unified_prefill_sliding_window(backend, window_left):
    """window_left plumbed through every windowed backend; cudnn is
    capability-excluded (skip via resolve)."""
    p = make_problem(
        seed=37,
        batch_size=4,
        max_q=48,
        max_kv=384,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, backend, window_left=window_left)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unified_prefill_nhd_layout(backend):
    p = make_problem(
        seed=41,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        kv_layout="NHD",
    )
    check(p, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("page_size", [1, 16])
def test_unified_prefill_csr_page_indices(backend, page_size):
    """The flat kv_page_indices form (sglang-style); page_size=1 token-CSR is
    in-envelope here, with dense-needing backends capability-excluded."""
    p = make_problem(
        seed=43,
        batch_size=4,
        max_q=32,
        max_kv=192,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=page_size,
        dtype=torch.bfloat16,
        input_form="page_indices",
    )
    check(p, backend)


def test_unified_prefill_csr_dense_equivalence():
    """Dense and flat-indices forms of the same problem are bitwise identical
    per backend (the derivation is exact, not approximate)."""
    common = dict(
        seed=47,
        batch_size=4,
        max_q=32,
        max_kv=192,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    p_dense = make_problem(**common)
    # SAME tensors, different input form (make_problem's randn draws from the
    # global CUDA RNG, so a second call would build a different problem)
    p_csr = dict(p_dense, input_form="page_indices")
    for backend in ["fa2", "cudnn", "trtllm-gen"]:
        try:
            _resolve_or_skip(p_dense, backend)
        except Exception:
            continue
        _, out_a, lse_a = run_unified(p_dense, backend)
        _, out_b, lse_b = run_unified(p_csr, backend)
        assert torch.equal(out_a, out_b), backend
        assert torch.equal(lse_a, lse_b), backend


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn"])
def test_unified_prefill_fp16(backend):
    p = make_problem(
        seed=53,
        batch_size=3,
        max_q=32,
        max_kv=192,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.float16,
    )
    check(p, backend)


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("head_dim", [64, 256])
def test_unified_prefill_wide_head_dims(backend, head_dim):
    p = make_problem(
        seed=59,
        batch_size=3,
        max_q=24,
        max_kv=160,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=head_dim,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, backend)


# NOTE: fa2/fa3 paged (192,128) is NOT declared: the paged kernel requires
# k_page_stride == v_page_stride ("K and V must have same page stride for
# sparse attention", batch_prefill_sm90.cu:235), which separately-allocated
# K(D=192)/V(D=128) pools violate.  It is reachable only with a
# stride-matched allocation contract — a capability axis with an allocation
# precondition, out of prototype scope.  cudnn handles (192,128) fine.
