"""Tests for batch_offsets_units="tokens" on the cuDNN prefill path.

Token-unit indptrs must produce the same results as the historical
element-unit offsets, both on the direct path (cuDNN consumes the indptrs
via cu_seq_len + ragged-offset multipliers, no conversion pre-pass) and on
the conversion path (FlashInfer scales them to element units internally).
The backend="cudnn" wrappers forward token-unit indptrs to this path.
"""

import pytest
import torch

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill


@pytest.mark.parametrize("explicit", ["q", "kv", "both"])
@pytest.mark.parametrize("backend_only", [False, True])
def test_cudnn_prefill_explicit_lengths_exclude_storage_padding(
    monkeypatch, explicit, backend_only
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("requires direct token-unit seqlens")

    # Isolate the real cache so another test cannot bypass the provider control.
    build = cudnn_prefill.cudnn.graph_cache(key_fn=cudnn_prefill._sdpa_prefill_key_fn)(
        cudnn_prefill._build_prefill_graph.__wrapped__
    )
    graphs = []

    def record_graph(*args, **kwargs):
        result = build(*args, **kwargs)
        graphs.append(result[0])
        return result

    monkeypatch.setattr(cudnn_prefill, "_build_prefill_graph", record_graph)
    if backend_only:
        monkeypatch.setattr(
            cudnn_prefill.cudnn.pygraph, "_candidate_engines", lambda self: []
        )

    torch.manual_seed(71)
    q_lengths, kv_lengths = [33, 17], [129, 83]
    q_slots = [64, 32] if explicit in ("q", "both") else q_lengths
    kv_slots = [160, 96] if explicit in ("kv", "both") else kv_lengths
    q_starts = [0, q_slots[0], sum(q_slots)]
    kv_starts = [0, kv_slots[0], sum(kv_slots)]
    heads, kv_heads, dim = 8, 2, 128

    def inputs():
        tensors = []
        for starts, lengths, h in (
            (q_starts, q_lengths, heads),
            (kv_starts, kv_lengths, kv_heads),
            (kv_starts, kv_lengths, kv_heads),
        ):
            tensor = torch.full(
                (starts[-1], h, dim), torch.nan, device="cuda", dtype=torch.bfloat16
            )
            for start, length in zip(starts[:-1], lengths, strict=True):
                tensor[start : start + length].normal_()
            tensors.append(tensor)
        return tensors

    q, k, v = inputs()
    out = torch.empty_like(q)
    lse = torch.empty((len(q), heads), device=q.device)
    common = dict(
        scale=dim**-0.5,
        workspace_buffer=torch.empty(128 << 20, device=q.device, dtype=torch.uint8),
        max_token_per_sequence=max(q_slots),
        max_sequence_kv=max(kv_slots),
        actual_seq_lens_q=(
            torch.tensor(q_lengths, device=q.device, dtype=torch.int32).view(
                -1, 1, 1, 1
            )
            if explicit in ("q", "both")
            else None
        ),
        actual_seq_lens_kv=(
            torch.tensor(kv_lengths, device=q.device, dtype=torch.int32).view(
                -1, 1, 1, 1
            )
            if explicit in ("kv", "both")
            else None
        ),
        batch_offsets_q=torch.tensor(q_starts, device=q.device, dtype=torch.int32),
        batch_offsets_k=torch.tensor(kv_starts, device=q.device, dtype=torch.int32),
        batch_offsets_units="tokens",
        causal=True,
        return_lse=True,
        lse_base="ln",
        out=out,
        lse=lse,
    )

    def run(query, key, value):
        cudnn_batch_prefill_with_kv_cache(query, key, value, **common)

    def check(query, key, value):
        if backend_only:
            from cudnn.engines import is_backend_engine

            assert graphs and all(
                is_backend_engine(g.get_engine_and_knobs_at_index(g._plan_index)[0])
                for g in graphs
            )
        for b, (nq, nk) in enumerate(zip(q_lengths, kv_lengths, strict=True)):
            qs, ks = q_starts[b], kv_starts[b]
            qb = query[qs : qs + nq].double().transpose(0, 1)
            kb = key[ks : ks + nk].double().transpose(0, 1)
            vb = value[ks : ks + nk].double().transpose(0, 1)
            kb = kb.repeat_interleave(heads // kv_heads, 0)
            vb = vb.repeat_interleave(heads // kv_heads, 0)
            scores = qb @ kb.transpose(1, 2) * dim**-0.5
            rows = torch.arange(nq, device=q.device)
            cols = torch.arange(nk, device=q.device)
            scores.masked_fill_(cols[None, :] > rows[:, None] + nk - nq, -torch.inf)
            expected = (scores.softmax(-1) @ vb).transpose(0, 1)
            stats = scores.logsumexp(-1).transpose(0, 1)
            torch.testing.assert_close(
                out[qs : qs + nq].double(), expected, atol=0.01, rtol=0.01
            )
            torch.testing.assert_close(
                lse[qs : qs + nq].double(), stats, atol=4e-4, rtol=0
            )

    run(q, k, v)
    check(q, k, v)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run(q, k, v)
        out.fill_(torch.nan)
        lse.fill_(torch.nan)
        graph.replay()
        check(q, k, v)
        fresh = inputs()
        run(*fresh)
        check(*fresh)
        q.mul_(0.5)
        k.mul_(0.75)
        v.mul_(0.25)
        out.fill_(torch.nan)
        lse.fill_(torch.nan)
        graph.replay()
        check(q, k, v)
    finally:
        graph.reset()


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [32, 87])
@pytest.mark.parametrize("s_kv", [87, 512])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("direct", [True, False])
def test_cudnn_prefill_token_indptr(
    monkeypatch,
    batch_size,
    s_qo,
    s_kv,
    num_kv_heads,
    num_qo_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    direct,
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if direct and not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("cuDNN backend/frontend too old for direct token-unit seqlens")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    if not direct:
        # Force the conversion path even where the direct path is supported.
        monkeypatch.setattr(
            cudnn_prefill,
            "_cudnn_supports_direct_seqlens",
            lambda dtype, **kwargs: False,
        )

    torch.manual_seed(0)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )

    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_q, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_kv, 0)]).int()

    q = torch.randn(
        int(actual_seq_lens_q.sum()),
        num_qo_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )

    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)

    common = dict(
        scale=float(1.0 / (head_dim_qk**0.5)),
        workspace_buffer=workspace_buffer,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=actual_seq_lens_q.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=actual_seq_lens_kv.view(batch_size, 1, 1, 1),
        causal=causal,
        return_lse=True,
    )

    out_ref, lse_ref = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        **common,
        batch_offsets_q=qo_indptr * (num_qo_heads * head_dim_qk),
        batch_offsets_o=qo_indptr * (num_qo_heads * head_dim_vo),
        batch_offsets_k=kv_indptr * (num_kv_heads * head_dim_qk),
        batch_offsets_v=kv_indptr * (num_kv_heads * head_dim_vo),
    )

    out, lse = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        **common,
        batch_offsets_q=qo_indptr,
        batch_offsets_k=kv_indptr,
        batch_offsets_units="tokens",
    )

    if direct:
        # There is no fallback: reaching here proves the direct
        # (unified-engine) path ran. Engines differ from the reference run,
        # so compare with tolerances.
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-2)
    else:
        # The conversion path scales the indptrs and builds the identical
        # legacy graph, so results are bitwise equal.
        assert torch.equal(out, out_ref)
        assert torch.equal(lse, lse_ref)


@pytest.mark.parametrize("direct", [True, False])
def test_cudnn_prefill_token_indptr_omit_actual_seq_lens(monkeypatch, direct):
    """actual_seq_lens_q/kv are optional with token-unit indptrs.

    Explicit lengths and cumulative lengths can select different legal plans.
    Check both against the same reference without requiring identical rounding.
    """
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if direct and not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("cuDNN backend/frontend too old for direct token-unit seqlens")

    if not direct:
        monkeypatch.setattr(
            cudnn_prefill,
            "_cudnn_supports_direct_seqlens",
            lambda dtype, **kwargs: False,
        )

    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_qo, s_kv = 4, 87, 512
    num_qo_heads, num_kv_heads, head_dim = 8, 4, 128

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_q, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_kv, 0)]).int()

    q = torch.randn(
        int(actual_seq_lens_q.sum()),
        num_qo_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    common = dict(
        scale=float(1.0 / (head_dim**0.5)),
        workspace_buffer=torch.empty(
            512 * 1024 * 1024, dtype=torch.int8, device=device
        ),
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        causal=True,
        return_lse=True,
        lse_base="ln",
        batch_offsets_q=qo_indptr,
        batch_offsets_k=kv_indptr,
        batch_offsets_units="tokens",
    )

    out_with, lse_with = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        **common,
        actual_seq_lens_q=actual_seq_lens_q.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=actual_seq_lens_kv.view(batch_size, 1, 1, 1),
    )
    out_without, lse_without = cudnn_batch_prefill_with_kv_cache(
        q, k_cache, v_cache, **common
    )

    q_starts, kv_starts = qo_indptr.cpu().tolist(), kv_indptr.cpu().tolist()
    for qs, qe, ks, ke in zip(
        q_starts[:-1], q_starts[1:], kv_starts[:-1], kv_starts[1:], strict=True
    ):
        qb = q[qs:qe].double().transpose(0, 1)
        kb = k_cache[ks:ke].double().transpose(0, 1)
        vb = v_cache[ks:ke].double().transpose(0, 1)
        kb = kb.repeat_interleave(num_qo_heads // num_kv_heads, 0)
        vb = vb.repeat_interleave(num_qo_heads // num_kv_heads, 0)
        scores = qb @ kb.transpose(1, 2) * head_dim**-0.5
        nq, nk = qe - qs, ke - ks
        rows = torch.arange(nq, device=device)
        cols = torch.arange(nk, device=device)
        scores.masked_fill_(cols[None, :] > rows[:, None] + nk - nq, -torch.inf)
        expected = (scores.softmax(-1) @ vb).transpose(0, 1)
        stats = scores.logsumexp(-1).transpose(0, 1)
        for out, lse in ((out_with, lse_with), (out_without, lse_without)):
            torch.testing.assert_close(
                out[qs:qe].double(), expected, atol=0.01, rtol=0.01
            )
            torch.testing.assert_close(lse[qs:qe].double(), stats, atol=4e-4, rtol=0)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_cudnn_prefill_lse_is_base2(num_kv_heads):
    """cuDNN returns base-2 LSE (log2 sum exp scores), matching every other
    FlashInfer backend. cuDNN's frontend emits natural-log stats, so the prefill
    path folds them to base-2; verify against a float reference."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    from flashinfer.utils import log2e

    torch.manual_seed(0)
    device = "cuda:0"
    s_q, s_kv, num_qo_heads, head_dim = 37, 64, 8, 128
    scale = float(head_dim**-0.5)

    q = torch.randn(s_q, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(s_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(s_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    # batch_size == 1: packed q starts at token 0, so no ragged offsets needed.
    _, lse = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale=scale,
        workspace_buffer=workspace_buffer,
        max_token_per_sequence=s_q,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=torch.tensor([s_q], dtype=torch.int32, device=device).view(
            1, 1, 1, 1
        ),
        actual_seq_lens_kv=torch.tensor([s_kv], dtype=torch.int32, device=device).view(
            1, 1, 1, 1
        ),
        causal=False,
        return_lse=True,
    )

    # Float reference LSE in base-2. GQA: broadcast each kv head to its q group.
    kf = k.float().repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * scale  # [h_qo, s_q, s_kv]
    lse_ref = torch.logsumexp(scores, dim=-1) * log2e  # base-2, [h_qo, s_q]

    assert lse.shape == (s_q, num_qo_heads), (
        "the LSE is packed, one row per query token"
    )
    torch.testing.assert_close(lse.transpose(0, 1), lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(128, 128), (192, 128)])
def test_cudnn_wrapper_token_indptr(head_dim_qk, head_dim_vo):
    """backend="cudnn" wrappers take token-unit qo_indptr/kv_indptr like every
    other backend, and reject the historical element-unit offsets with a
    migration hint."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    import flashinfer

    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_qo, s_kv = 4, 87, 512
    num_qo_heads, num_kv_heads = 8, 4
    scale = float(head_dim_qk**-0.5)

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_q, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_kv, 0)]).int()

    q = torch.randn(
        int(actual_seq_lens_q.sum()),
        num_qo_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )

    def plan(wrapper, qo, kv):
        wrapper.plan(
            qo_indptr=qo,
            kv_indptr=kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=True,
            sm_scale=scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            seq_lens=actual_seq_lens_kv.view(batch_size, 1, 1, 1),
            seq_lens_q=actual_seq_lens_q.view(batch_size, 1, 1, 1),
            max_token_per_sequence=s_qo,
            max_sequence_kv=s_kv,
        )

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)
    cudnn_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cudnn"
    )

    # Same token-unit indptrs as the fa2 reference wrapper.
    plan(cudnn_wrapper, qo_indptr, kv_indptr)
    out = cudnn_wrapper.run(q, k, v)

    ref_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device), "NHD"
    )
    ref_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=True,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out_ref = ref_wrapper.run(q, k, v)
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)

    # Legacy element-unit offsets are rejected with a migration hint.
    plan(
        cudnn_wrapper,
        qo_indptr * (num_qo_heads * head_dim_qk),
        kv_indptr * (num_kv_heads * head_dim_qk),
    )
    with pytest.raises(ValueError, match="element-unit"):
        cudnn_wrapper.run(q, k, v)
