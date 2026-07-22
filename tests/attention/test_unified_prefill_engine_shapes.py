"""Engine-shaped integration tests: metadata built the way vLLM and sglang
actually build it, fed to the unified API unchanged.

These are the machine-checked half of the consumer-integration verification
(the companion diffs live in ``integrations/``).  Each test replicates the
engine's *real* construction pipeline — persistent buffer reuse,
over-allocation, pinned host mirrors, rebased mixed-batch slices — rather
than an idealized version, so any impedance mismatch between the unified
contract and what an engine can cheaply supply shows up here as a failure
or a documented friction note.

Friction log (kept current — this docstring is part of the deliverable;
the full audited list lives in integrations/README.md):
- vLLM: metadata maps 1:1 (dense table + GPU seq_lens + host maxes + CPU
  mirrors; rebased mixed-batch slice; preallocated out).  Carried friction:
  the all-trtllm async mode skips seq_lens_cpu retrieval today and would
  pay it under unified's unconditional validation; DCP rewrites the CPU
  lens to DCP-local while GPU stays global (mirror contract) — DCP stays
  on the current path in v1.
- sglang: page_size=1 token-CSR maps to the flat ``kv_page_indices`` form
  including the +256 over-allocated tail; preallocated indptr buffer
  slices work as-is.  Carried friction: no host copy of paged_kernel_lens
  exists at the plan site today (the diff adds a copy-forward of a
  scheduler-owned host array).  The radix-extend cascade (causal=False
  over prefix lens, zero-length rows for no-prefix requests) is outside
  the v1 envelope entirely.
"""

import pytest
import torch

from flashinfer.attention.unified import UnifiedPagedPrefill, resolve_paged_prefill

from .unified_prefill_reference import reference_paged_prefill

DEVICE = "cuda:0"


def _resolve_or_skip(**kw):
    try:
        return resolve_paged_prefill(**kw)
    except ValueError as e:
        pytest.skip(str(e))


def test_vllm_shaped_prefill_mixed_batch():
    """vLLM v1 FlashInferBackend pipeline (backends/flashinfer.py):

    - one persistent dense ``block_table_tensor`` (b, max_blocks) int32 GPU
    - ``seq_lens_cpu`` / ``query_start_loc_cpu`` pinned host tensors with
      async-H2D GPU twins (the CpuGpuBuffer pattern)
    - a mixed batch (decodes first, prefills after); the prefill wrapper gets
      the REBASED slice: ``qo_indptr[p:] - qo_indptr[p]``
    - host maxes from the CPU side (``.max()`` on host, never a device sync)
    """
    torch.manual_seed(0)
    num_decodes, num_prefills = 2, 3
    b = num_decodes + num_prefills
    page_size, h_qo, h_kv, d = 16, 8, 2, 128
    max_model_len = 512

    # --- engine-owned persistent state (built once, reused every step) ---
    max_blocks = max_model_len // page_size
    pool_pages = b * max_blocks + 7
    block_table_tensor = (
        torch.randperm(pool_pages, dtype=torch.int32)[: b * max_blocks]
        .reshape(b, max_blocks)
        .to(DEVICE)
    )
    k_pool = torch.randn(
        pool_pages, h_kv, page_size, d, dtype=torch.bfloat16, device=DEVICE
    )
    v_pool = torch.randn_like(k_pool)

    # --- per-step metadata, CPU-first like vLLM's builder ---
    q_lens_cpu = torch.tensor([1, 1, 40, 17, 9], dtype=torch.int32, pin_memory=True)
    seq_lens_cpu = torch.tensor(
        [100, 37, 300, 210, 64], dtype=torch.int32, pin_memory=True
    )
    query_start_loc_cpu = torch.zeros(b + 1, dtype=torch.int32, pin_memory=True)
    torch.cumsum(q_lens_cpu, 0, dtype=torch.int32, out=query_start_loc_cpu[1:])
    seq_lens = seq_lens_cpu.to(DEVICE, non_blocking=True)
    query_start_loc = query_start_loc_cpu.to(DEVICE, non_blocking=True)

    # --- the prefill slice, rebased exactly like vLLM's TRTLLMPrefill build ---
    p0 = num_decodes
    qo_indptr_prefill = query_start_loc[p0:] - query_start_loc[p0]
    qo_indptr_prefill_cpu = query_start_loc_cpu[p0:] - query_start_loc_cpu[p0]
    kv_lens_prefill = seq_lens[p0:]
    kv_lens_prefill_cpu = seq_lens_cpu[p0:]
    block_tables_prefill = block_table_tensor[p0:]
    max_q_len = int(q_lens_cpu[p0:].max())
    max_kv_len = int(seq_lens_cpu[p0:].max())

    # --- engine init: resolve before pool/dtype/graph decisions ---
    res = _resolve_or_skip(
        device=torch.device(DEVICE),
        num_qo_heads=h_qo,
        num_kv_heads=h_kv,
        head_dim_qk=d,
        q_dtype=torch.bfloat16,
        page_size=page_size,
        kv_layout="HND",
        causal=True,
        need_lse=True,  # vLLM DCP consumes LSE
    )

    attn = UnifiedPagedPrefill(torch.device(DEVICE))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        # with the mirrors vLLM already owns, plan() must be zero-sync
        attn.plan(
            qo_indptr=qo_indptr_prefill,
            kv_seq_lens=kv_lens_prefill,
            block_tables=block_tables_prefill,
            page_size=page_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_qo_heads=h_qo,
            num_kv_heads=h_kv,
            head_dim_qk=d,
            q_dtype=torch.bfloat16,
            causal=True,
            return_lse=True,
            qo_indptr_cpu=qo_indptr_prefill_cpu,
            kv_seq_lens_cpu=kv_lens_prefill_cpu,
            backend=res,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    total_prefill_tokens = int(qo_indptr_prefill_cpu[-1])
    q = torch.randn(total_prefill_tokens, h_qo, d, dtype=torch.bfloat16, device=DEVICE)
    out_buf = torch.empty(
        total_prefill_tokens, h_qo, d, dtype=torch.bfloat16, device=DEVICE
    )
    out, lse = attn.run(q, (k_pool, v_pool), out=out_buf)
    assert out.data_ptr() == out_buf.data_ptr()  # vLLM preallocates out

    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_pool,
        v_pool,
        qo_indptr_prefill_cpu,
        kv_lens_prefill_cpu,
        block_tables_prefill,
        page_size,
        True,
    )
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=3e-2, rtol=2e-2)


def test_sglang_shaped_prefill_token_csr():
    """sglang FlashInferAttnBackend pipeline (flashinfer_backend.py):

    - page_size == 1 token-granular paging: ``kv_indices`` holds token slot
      ids gathered from the req_to_token pool, OVER-ALLOCATED by +256
      (garbage tail past the live prefix — must be accepted, len >= required)
    - preallocated ``(max_bs+1,)`` int32 indptr buffers, sliced per batch
    - ``seq_lens_cpu`` carried in the ForwardBatch (host is the origin)
    - NHD kv pool layout
    """
    torch.manual_seed(1)
    max_bs = 64
    b, h_qo, h_kv, d = 4, 8, 2, 128
    pool_tokens = 4096  # req_to_token pool slots

    # engine-owned preallocated buffers
    qo_indptr_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device=DEVICE)
    kv_lens = torch.tensor([170, 93, 256, 41], dtype=torch.int32)
    q_lens = torch.tensor([17, 1, 40, 9], dtype=torch.int32)
    qo_indptr_buf[1 : b + 1] = torch.cumsum(q_lens, 0, dtype=torch.int32).to(DEVICE)
    qo_indptr = qo_indptr_buf[: b + 1]
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )

    # token-slot indices, request-ordered, +256 over-allocation like
    # create_flashinfer_kv_indices_triton's destination buffer
    total_slots = int(kv_lens.sum())
    perm = torch.randperm(pool_tokens, dtype=torch.int32)
    kv_indices_buf = torch.empty(total_slots + 256, dtype=torch.int32, device=DEVICE)
    kv_indices_buf[:total_slots] = perm[:total_slots].to(DEVICE)
    # tail is uninitialized garbage in sglang; leave whatever empty() gave us

    # NHD token-slot pool: (slots, page_size=1, H, D)
    k_pool = torch.randn(pool_tokens, 1, h_kv, d, dtype=torch.bfloat16, device=DEVICE)
    v_pool = torch.randn_like(k_pool)

    kv_lens_dev = kv_lens.to(DEVICE)  # on device early in sglang's batch prep

    res = _resolve_or_skip(
        device=torch.device(DEVICE),
        num_qo_heads=h_qo,
        num_kv_heads=h_kv,
        head_dim_qk=d,
        q_dtype=torch.bfloat16,
        page_size=1,
        kv_layout="NHD",
        causal=True,
        need_lse=True,  # merge_state consumes LSE in the extend cascade
        kv_input_form="page_indices",
    )
    # sglang pins fa2 today; assert the pinned form is in the candidate set
    assert "fa2" in res.backends

    attn = UnifiedPagedPrefill(torch.device(DEVICE))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        attn.plan(
            qo_indptr=qo_indptr,
            kv_seq_lens=kv_lens_dev,
            kv_page_indices=kv_indices_buf,  # over-allocated, as-is
            page_size=1,
            max_q_len=int(q_lens.max()),
            max_kv_len=int(kv_lens.max()),
            num_qo_heads=h_qo,
            num_kv_heads=h_kv,
            head_dim_qk=d,
            q_dtype=torch.bfloat16,
            kv_layout="NHD",
            causal=True,
            return_lse=True,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_lens,
            backend=res,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    total_q = int(qo_indptr_cpu[-1])
    q = torch.randn(total_q, h_qo, d, dtype=torch.bfloat16, device=DEVICE)
    out, lse = attn.run(q, (k_pool, v_pool))

    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_pool,
        v_pool,
        qo_indptr_cpu,
        kv_lens,
        None,
        1,
        True,
        kv_layout="NHD",
        kv_page_indices=kv_indices_buf[:total_slots],
    )
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=3e-2, rtol=2e-2)
    # LSE is base-2 packed — directly consumable by merge_state
    assert lse.shape == (total_q, h_qo) and lse.dtype == torch.float32


def test_vllm_shaped_sliding_window_uniform():
    """vLLM requires uniform per-layer hyperparameters on its FI path and
    plans once per batch; a windowed model maps to plan(window_left=...)."""
    torch.manual_seed(2)
    b, page_size, h_qo, h_kv, d = 3, 16, 8, 2, 128
    kv_lens_cpu = torch.tensor([200, 310, 150], dtype=torch.int32)
    q_lens_cpu = torch.tensor([30, 25, 40], dtype=torch.int32)
    qo_indptr_cpu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(q_lens_cpu, 0, dtype=torch.int32),
        ]
    )
    max_blocks = (int(kv_lens_cpu.max()) + page_size - 1) // page_size
    pool = b * max_blocks + 3
    bt = (
        torch.randperm(pool, dtype=torch.int32)[: b * max_blocks]
        .reshape(b, max_blocks)
        .to(DEVICE)
    )
    k_pool = torch.randn(pool, h_kv, page_size, d, dtype=torch.bfloat16, device=DEVICE)
    v_pool = torch.randn_like(k_pool)

    window = 64
    res = _resolve_or_skip(
        device=torch.device(DEVICE),
        num_qo_heads=h_qo,
        num_kv_heads=h_kv,
        head_dim_qk=d,
        q_dtype=torch.bfloat16,
        page_size=page_size,
        causal=True,
        need_lse=False,
        window_left=window,
    )
    assert "cudnn" in res.excluded  # windowed: cudnn capability-excluded

    attn = UnifiedPagedPrefill(torch.device(DEVICE))
    attn.plan(
        qo_indptr=qo_indptr_cpu.to(DEVICE),
        kv_seq_lens=kv_lens_cpu.to(DEVICE),
        block_tables=bt,
        page_size=page_size,
        max_q_len=int(q_lens_cpu.max()),
        max_kv_len=int(kv_lens_cpu.max()),
        num_qo_heads=h_qo,
        num_kv_heads=h_kv,
        head_dim_qk=d,
        q_dtype=torch.bfloat16,
        causal=True,
        window_left=window,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
        backend=res,
    )
    total_q = int(qo_indptr_cpu[-1])
    q = torch.randn(total_q, h_qo, d, dtype=torch.bfloat16, device=DEVICE)
    out, _ = attn.run(q, (k_pool, v_pool))
    ref_out, _ = reference_paged_prefill(
        q,
        k_pool,
        v_pool,
        qo_indptr_cpu,
        kv_lens_cpu,
        bt,
        page_size,
        True,
        window_left=window,
    )
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
