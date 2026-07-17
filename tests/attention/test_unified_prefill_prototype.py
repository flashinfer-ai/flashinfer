"""Conformance matrix for the unified paged-prefill prototype.

One backend-parametrized test, one reference oracle, one output contract:
- outputs match the fp32 paged-attention oracle
- LSE is ALWAYS base-2, packed (total_q_tokens, num_qo_heads), fp32 —
  including for cuDNN, whose native natural-log padded stats are normalized
  in its adapter.  This is the first test in the repo that pins the cuDNN
  LSE base against an independent reference.

This file doubles as executable documentation of the API (proposal P0).
"""

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
    k_cache = torch.randn(
        pool_pages, num_kv_heads, page_size, head_dim_qk, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        pool_pages, num_kv_heads, page_size, head_dim_vo, dtype=dtype, device=device
    )

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        qo_indptr=qo_indptr_cpu.to(device),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(device),
        kv_seq_lens_cpu=kv_lens,
        block_tables=block_tables_cpu.to(device),
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


def run_unified(p, backend, *, causal=True, return_lse=True, with_mirrors=True):
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
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        causal=causal,
        return_lse=return_lse,
        qo_indptr_cpu=p["qo_indptr_cpu"] if with_mirrors else None,
        kv_seq_lens_cpu=p["kv_seq_lens_cpu"] if with_mirrors else None,
        backend=backend,
    )
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    return attn, out, lse


def _resolve_or_skip(p, backend, *, causal=True, need_lse=True):
    try:
        return resolve_paged_prefill(
            device=torch.device(p["device"]),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim_qk"],
            head_dim_vo=p["head_dim_vo"],
            q_dtype=p["dtype"],
            page_size=p["page_size"],
            causal=causal,
            need_lse=need_lse,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"backend {backend} not runnable here: {e}")


def check(p, backend, *, causal=True):
    _resolve_or_skip(p, backend, causal=causal)
    _, out, lse = run_unified(p, backend, causal=causal)
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_cache"],
        p["v_cache"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        causal,
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
        seed=hash((backend, batch_size, max_q, max_kv, heads, page_size)) % (2**31),
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
