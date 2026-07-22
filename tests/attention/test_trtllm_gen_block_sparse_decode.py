"""Tests for trtllm-gen block-sparse attention (decode/generation phase).

Block-sparse attention lets every KV head attend to its own subset of KV cache
pages: ``block_tables`` is extended to ``[num_kv_heads, batch_size,
max_num_pages_per_seq]`` and ``seq_lens`` to ``[num_kv_heads, batch_size]``,
where each row holds the *selected* page indices packed densely at the front
(ascending original order) and the per-head count of surviving KV tokens.

The tests build genuinely different sparse page sets per KV head and compare
against a pure-torch reference that gathers the selected pages per
(sequence, kv head) pair and runs float32 attention.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability
from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance

from tests.attention.test_trtllm_gen_attention_decode import (
    DTYPE_MAP,
    GPU_DEVICE,
    create_kv_cache,
    create_page_table,
    create_query_tensor,
    create_workspace_buffers,
    generate_seq_lens_decode,
)

pytestmark = pytest.mark.long_running


def _skip_unless_trtllm_gen_supported() -> None:
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("trtllm-gen backend requires SM100 and SM103 GPUs.")


def _run_or_skip_missing_cubins(func, *args, **kwargs):
    """Run the kernel; skip the test when trtllm-gen cubins are unavailable."""
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "Missing TRTLLM-GEN kernel" in str(e) or "cubin" in str(e).lower():
            pytest.skip(f"TRTLLM-GEN kernels are not available: {e}")
        raise


def _scalar(x) -> float:
    """to_float8 returns 0-dim tensors for scales; the fp16/bf16 path uses floats."""
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)


def make_block_sparse_tables(
    page_table: torch.Tensor,
    page_per_seq: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    num_kv_heads: int,
    density: float,
    keep_tail_page: bool,
    min_pages_per_head: int = 1,
):
    """Select a different subset of pages for each (kv head, sequence) pair.

    Returns the head-major kernel inputs plus the selected page-slot positions
    (per kv head, per sequence, ascending) for the torch reference:

    - ``head_tables``: int32 ``[num_kv_heads, batch_size, max_num_pages_per_seq]``,
      selected page ids from ``page_table`` packed densely at the front of each row.
    - ``sparse_lens``: int32 ``[num_kv_heads, batch_size]``, number of surviving
      KV tokens per row. Only the final selected page may be partial, and only
      when it is the sequence's original tail page (``keep_tail_page=True``).
    """
    batch_size, max_pages = page_table.shape
    page_table_cpu = page_table.cpu()
    head_tables = torch.zeros((num_kv_heads, batch_size, max_pages), dtype=torch.int32)
    sparse_lens = torch.zeros((num_kv_heads, batch_size), dtype=torch.int32)
    selected_positions = [[None] * batch_size for _ in range(num_kv_heads)]

    for b in range(batch_size):
        num_pages = int(page_per_seq[b])
        seq_len = int(seq_lens[b])
        # Number of tokens in the sequence's original tail page.
        tail_page_len = seq_len - (num_pages - 1) * page_size
        for h in range(num_kv_heads):
            num_selected = max(min_pages_per_head, int(round(density * num_pages)))
            num_selected = min(num_selected, num_pages)
            perm = torch.randperm(num_pages)
            if keep_tail_page or num_pages == 1:
                head = perm[perm != (num_pages - 1)][: num_selected - 1]
                sel = torch.cat([head, torch.tensor([num_pages - 1], dtype=head.dtype)])
            else:
                # Drop the tail page so every selected page is full
                # (page-aligned per-head seq len).
                sel = perm[perm != (num_pages - 1)][:num_selected]
            sel, _ = torch.sort(sel)
            num_sel = sel.numel()
            head_tables[h, b, :num_sel] = page_table_cpu[b, sel]
            if int(sel[-1]) == num_pages - 1:
                sparse_len = (num_sel - 1) * page_size + tail_page_len
            else:
                sparse_len = num_sel * page_size
            sparse_lens[h, b] = sparse_len
            selected_positions[h][b] = sel

    return (
        head_tables.to(page_table.device).contiguous(),
        sparse_lens.to(page_table.device).contiguous(),
        selected_positions,
    )


def ref_block_sparse_decode(
    ref_q: torch.Tensor,
    ref_kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    selected_positions,
    sparse_lens: torch.Tensor,
    page_size: int,
    sm_scale: float,
    q_len_per_req: int,
    num_kv_heads: int,
    head_grp_size: int,
) -> torch.Tensor:
    """Pure-torch reference: gather each kv head's selected pages, run fp32 SDPA.

    For speculative decoding (``q_len_per_req > 1``) the causal mask applies over
    the *compacted* sparse sequence: query token ``i`` attends to compacted kv
    positions ``j <= L - q_len_per_req + i`` (matching the trtllm-gen generation
    kernels, which mask via the per-head kv length).
    """
    batch_size = page_table.shape[0]
    head_dim = ref_q.shape[-1]
    sparse_lens_cpu = sparse_lens.cpu()
    output = torch.zeros_like(ref_q, dtype=torch.float32)
    for b in range(batch_size):
        for h in range(num_kv_heads):
            sel = selected_positions[h][b]
            page_ids = page_table[b, sel.to(page_table.device)].long()
            kv_len = int(sparse_lens_cpu[h, b])
            # ref_kv_cache: [num_pages, 2, num_kv_heads, page_size, head_dim] (HND)
            kv_pages = ref_kv_cache[page_ids]
            k = kv_pages[:, 0, h].reshape(-1, head_dim)[:kv_len].float()
            v = kv_pages[:, 1, h].reshape(-1, head_dim)[:kv_len].float()
            q_rows = ref_q[
                b * q_len_per_req : (b + 1) * q_len_per_req,
                h * head_grp_size : (h + 1) * head_grp_size,
            ].float()
            scores = torch.einsum("qgd,ld->qgl", q_rows, k) * sm_scale
            if q_len_per_req > 1:
                for i in range(q_len_per_req):
                    limit = kv_len - q_len_per_req + 1 + i
                    scores[i, :, limit:] = float("-inf")
            attn = torch.softmax(scores, dim=-1)
            output[
                b * q_len_per_req : (b + 1) * q_len_per_req,
                h * head_grp_size : (h + 1) * head_grp_size,
            ] = torch.einsum("qgl,ld->qgd", attn, v)
    return output


def _build_case(
    batch_size: int,
    q_len_per_req: int,
    max_in_kv_len: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    q_dtype: str,
    kv_dtype: str,
    density: float,
    keep_tail_page: bool,
):
    """Common construction: dense cache/page table + per-head sparse selection."""
    num_qo_heads = num_kv_heads * head_grp_size
    head_dim = 128
    q_lens, _, seq_lens = generate_seq_lens_decode(
        batch_size, q_len_per_req, max_in_kv_len, None
    )
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    kv_cache, k_scale, v_scale, ref_kv_cache, _ = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if kv_dtype == "fp8" else kv_dtype,
        "HND",
    )
    page_table, _, page_per_seq = create_page_table(batch_size, seq_lens, page_size)
    # Speculative decode needs at least 2 selected pages (incl. the tail page)
    # so every query token has something to attend to under the compacted
    # causal mask.
    min_pages_per_head = 2 if q_len_per_req > 1 else 1
    head_tables, sparse_lens, selected_positions = make_block_sparse_tables(
        page_table,
        page_per_seq,
        seq_lens,
        page_size,
        num_kv_heads,
        density,
        keep_tail_page or q_len_per_req > 1,
        min_pages_per_head,
    )
    sm_scale = float(1.0 / (head_dim**0.5))
    return (
        q,
        q_scale,
        ref_q,
        kv_cache,
        k_scale,
        v_scale,
        ref_kv_cache,
        page_table,
        head_tables,
        sparse_lens,
        selected_positions,
        seq_lens,
        sm_scale,
    )


def _tolerances(q_dtype: str, kv_dtype: str, q_len_per_req: int):
    if q_dtype == "fp8":
        rtol, atol = 4e-2, 7e-2
    else:
        rtol, atol = 1e-2, 1e-2
    if q_len_per_req > 1:
        rtol, atol = rtol * 2, atol * 2
    return rtol, atol


@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("fp16", "fp16", "fp16"),
        ("bf16", "bf16", "bf16"),
        ("fp8", "fp8", "bf16"),
    ],
)
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize(
    "num_kv_heads,head_grp_size",
    [
        (2, 8),
        (4, 1),
    ],
)
@pytest.mark.parametrize("max_in_kv_len", [110, 8192])
@pytest.mark.parametrize(
    "density,keep_tail_page,q_len_per_req",
    [
        (0.35, True, 1),
        (0.75, False, 1),
        (0.5, True, 3),
    ],
)
def test_block_sparse_decode_correctness(
    q_dtype,
    kv_dtype,
    o_dtype,
    page_size,
    num_kv_heads,
    head_grp_size,
    max_in_kv_len,
    density,
    keep_tail_page,
    q_len_per_req,
):
    """Kernel output vs a torch reference with different sparse pages per KV head.

    ``max_in_kv_len=8192`` (with a small batch) exercises the multi-CTA KV path
    and the per-head seqLen fix in the fmhaReduction kernel.
    """
    _skip_unless_trtllm_gen_supported()
    torch.manual_seed(42)
    batch_size = 4

    (
        q,
        q_scale,
        ref_q,
        kv_cache,
        k_scale,
        v_scale,
        ref_kv_cache,
        page_table,
        head_tables,
        sparse_lens,
        selected_positions,
        seq_lens,
        sm_scale,
    ) = _build_case(
        batch_size,
        q_len_per_req,
        max_in_kv_len,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        kv_dtype,
        density,
        keep_tail_page,
    )

    output_ref = ref_block_sparse_decode(
        ref_q,
        ref_kv_cache,
        page_table,
        selected_positions,
        sparse_lens,
        page_size,
        sm_scale,
        q_len_per_req,
        num_kv_heads,
        head_grp_size,
    )

    workspace_buffer, _ = create_workspace_buffers(GPU_DEVICE)
    output = _run_or_skip_missing_cubins(
        flashinfer.decode.trtllm_batch_decode_with_kv_cache,
        q,
        kv_cache,
        workspace_buffer,
        head_tables,
        sparse_lens,
        int(sparse_lens.max().item()),
        bmm1_scale=_scalar(q_scale) * _scalar(k_scale) * sm_scale,
        bmm2_scale=_scalar(v_scale),
        out_dtype=DTYPE_MAP[o_dtype],
        backend="trtllm-gen",
        q_len_per_req=q_len_per_req,
        enable_block_sparse_attention=True,
    )

    rtol, atol = _tolerances(q_dtype, kv_dtype, q_len_per_req)
    max_mismatched_elements = max(1, int(5e-5 * output.numel()))
    assert_close_with_mismatch_tolerance(
        output.float(),
        output_ref,
        rtol=rtol,
        atol=atol,
        max_mismatched_elements=max_mismatched_elements,
    )


@pytest.mark.parametrize("q_dtype,kv_dtype", [("fp16", "fp16"), ("bf16", "bf16")])
@pytest.mark.parametrize("max_in_kv_len", [512, 4096])
def test_block_sparse_full_density_matches_dense(q_dtype, kv_dtype, max_in_kv_len):
    """Full-density per-head tables must reproduce the dense kernel bit-exactly.

    This doubles as a runtime sanity check that the deployed cubins honor the
    ``mUseBlockSparseAttention`` flag: if the flag were silently ignored, the
    per-head table/seqlen layouts would be misinterpreted and the sparse
    correctness tests would fail, while this test isolates the dense-equivalence
    contract on identical page sets.
    """
    _skip_unless_trtllm_gen_supported()
    torch.manual_seed(7)
    batch_size = 4
    page_size = 32
    num_kv_heads = 2
    head_grp_size = 8
    head_dim = 128
    num_qo_heads = num_kv_heads * head_grp_size

    q_lens, _, seq_lens = generate_seq_lens_decode(batch_size, 1, max_in_kv_len, None)
    q, q_scale, _ = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    kv_cache, k_scale, v_scale, _, _ = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        kv_dtype,
        "HND",
    )
    page_table, _, _ = create_page_table(batch_size, seq_lens, page_size)
    sm_scale = float(1.0 / (head_dim**0.5))

    seq_lens_device = seq_lens.to(dtype=torch.int32, device=GPU_DEVICE)
    max_seq_len = int(seq_lens.max().item())

    # Every kv head sees the full dense page set and full sequence lengths.
    head_tables = page_table.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
    head_seq_lens = seq_lens_device.unsqueeze(0).expand(num_kv_heads, -1).contiguous()

    common_kwargs = dict(
        bmm1_scale=_scalar(q_scale) * _scalar(k_scale) * sm_scale,
        bmm2_scale=_scalar(v_scale),
        backend="trtllm-gen",
        q_len_per_req=1,
    )
    workspace_buffer, _ = create_workspace_buffers(GPU_DEVICE)
    output_dense = _run_or_skip_missing_cubins(
        flashinfer.decode.trtllm_batch_decode_with_kv_cache,
        q,
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens_device,
        max_seq_len,
        **common_kwargs,
    )
    output_sparse = _run_or_skip_missing_cubins(
        flashinfer.decode.trtllm_batch_decode_with_kv_cache,
        q,
        kv_cache,
        workspace_buffer,
        head_tables,
        head_seq_lens,
        max_seq_len,
        enable_block_sparse_attention=True,
        **common_kwargs,
    )

    # Same kernel, same schedule, same pages -> expect bitwise-identical output.
    assert torch.equal(output_sparse, output_dense), (
        "block-sparse attention with full-density per-head page tables must "
        "match the dense kernel output"
    )


def test_block_sparse_per_head_difference():
    """Heads with different page sets must attend to different tokens.

    Head 0 selects the even page slots, head 1 the odd page slots (plus the tail
    page for both, so lengths stay valid). The kernel output must (a) match the
    per-head torch reference and (b) differ from a run where both heads use head
    0's page set — this guards against the kernel silently ignoring the per-head
    dimension of the tables.
    """
    _skip_unless_trtllm_gen_supported()
    torch.manual_seed(3)
    batch_size = 2
    page_size = 16
    num_kv_heads = 2
    head_grp_size = 4
    head_dim = 128
    q_len_per_req = 1
    num_qo_heads = num_kv_heads * head_grp_size

    # Fixed, page-aligned-ish kv lens with several pages per sequence.
    seq_lens = torch.tensor([250, 311], dtype=torch.int32)
    q_lens = torch.full((batch_size,), q_len_per_req, dtype=torch.int32)
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, "bf16")
    kv_cache, k_scale, v_scale, ref_kv_cache, _ = create_kv_cache(
        batch_size, seq_lens, page_size, num_kv_heads, head_dim, "bf16", "bf16", "HND"
    )
    page_table, _, page_per_seq = create_page_table(batch_size, seq_lens, page_size)
    sm_scale = float(1.0 / (head_dim**0.5))

    max_pages = page_table.shape[1]
    head_tables = torch.zeros((num_kv_heads, batch_size, max_pages), dtype=torch.int32)
    sparse_lens = torch.zeros((num_kv_heads, batch_size), dtype=torch.int32)
    selected_positions = [[None] * batch_size for _ in range(num_kv_heads)]
    page_table_cpu = page_table.cpu()
    for b in range(batch_size):
        num_pages = int(page_per_seq[b])
        tail_page_len = int(seq_lens[b]) - (num_pages - 1) * page_size
        for h in range(num_kv_heads):
            body = torch.arange(h, num_pages - 1, 2)  # even/odd page slots
            sel = torch.cat([body, torch.tensor([num_pages - 1])])
            selected_positions[h][b] = sel
            head_tables[h, b, : sel.numel()] = page_table_cpu[b, sel]
            sparse_lens[h, b] = (sel.numel() - 1) * page_size + tail_page_len
    head_tables = head_tables.to(GPU_DEVICE).contiguous()
    sparse_lens = sparse_lens.to(GPU_DEVICE).contiguous()

    output_ref = ref_block_sparse_decode(
        ref_q,
        ref_kv_cache,
        page_table,
        selected_positions,
        sparse_lens,
        page_size,
        sm_scale,
        q_len_per_req,
        num_kv_heads,
        head_grp_size,
    )

    workspace_buffer, _ = create_workspace_buffers(GPU_DEVICE)
    common_kwargs = dict(
        bmm1_scale=_scalar(q_scale) * _scalar(k_scale) * sm_scale,
        bmm2_scale=_scalar(v_scale),
        backend="trtllm-gen",
        q_len_per_req=q_len_per_req,
        enable_block_sparse_attention=True,
    )
    output = _run_or_skip_missing_cubins(
        flashinfer.decode.trtllm_batch_decode_with_kv_cache,
        q,
        kv_cache,
        workspace_buffer,
        head_tables,
        sparse_lens,
        int(sparse_lens.max().item()),
        **common_kwargs,
    )

    assert_close_with_mismatch_tolerance(
        output.float(),
        output_ref,
        rtol=1e-2,
        atol=1e-2,
        max_mismatched_elements=max(1, int(5e-5 * output.numel())),
    )

    # Rerun with head 0's rows duplicated for both heads: head 1's output must
    # change, proving the kernel really uses the per-head page tables.
    shared_tables = (
        head_tables[0].unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
    )
    shared_lens = sparse_lens[0].unsqueeze(0).expand(num_kv_heads, -1).contiguous()
    output_shared = _run_or_skip_missing_cubins(
        flashinfer.decode.trtllm_batch_decode_with_kv_cache,
        q,
        kv_cache,
        workspace_buffer,
        shared_tables,
        shared_lens,
        int(shared_lens.max().item()),
        **common_kwargs,
    )
    head1_slice = slice(head_grp_size, 2 * head_grp_size)
    assert not torch.allclose(
        output[:, head1_slice].float(),
        output_shared[:, head1_slice].float(),
        rtol=1e-3,
        atol=1e-3,
    ), "kv head 1's output did not change when its page set changed"
    # Head 0's pages are identical in both runs, so its output must not change.
    assert torch.equal(output[:, :head_grp_size], output_shared[:, :head_grp_size]), (
        "kv head 0's output changed even though its page set is identical"
    )


def test_block_sparse_validation_errors():
    """Invalid configurations must be rejected before reaching the kernel."""
    _skip_unless_trtllm_gen_supported()
    torch.manual_seed(0)
    batch_size = 2
    page_size = 16
    num_kv_heads = 2
    head_grp_size = 4
    head_dim = 128
    num_qo_heads = num_kv_heads * head_grp_size

    q_lens = torch.full((batch_size,), 1, dtype=torch.int32)
    seq_lens = torch.tensor([100, 120], dtype=torch.int32)
    q, q_scale, _ = create_query_tensor(q_lens, num_qo_heads, head_dim, "bf16")
    kv_cache, k_scale, v_scale, _, _ = create_kv_cache(
        batch_size, seq_lens, page_size, num_kv_heads, head_dim, "bf16", "bf16", "HND"
    )
    page_table, _, page_per_seq = create_page_table(batch_size, seq_lens, page_size)
    sm_scale = float(1.0 / (head_dim**0.5))
    head_tables, sparse_lens, _ = make_block_sparse_tables(
        page_table, page_per_seq, seq_lens, page_size, num_kv_heads, 0.5, True
    )
    workspace_buffer, _ = create_workspace_buffers(GPU_DEVICE)

    def run(**overrides):
        kwargs = dict(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=head_tables,
            seq_lens=sparse_lens,
            max_seq_len=int(sparse_lens.max().item()),
            bmm1_scale=_scalar(q_scale) * _scalar(k_scale) * sm_scale,
            bmm2_scale=_scalar(v_scale),
            backend="trtllm-gen",
            q_len_per_req=1,
            enable_block_sparse_attention=True,
        )
        kwargs.update(overrides)
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(**kwargs)

    with pytest.raises(ValueError, match="only supported by the trtllm-gen"):
        run(backend="xqa")
    with pytest.raises(ValueError, match="sliding window"):
        run(window_left=127)
    with pytest.raises(ValueError, match="skip_softmax"):
        run(skip_softmax_threshold_scale_factor=1e-3)
    with pytest.raises(ValueError, match="shared paged-KV"):
        run(uses_shared_paged_kv_idx=False)
    with pytest.raises(ValueError, match="block_tables must be 3D"):
        run(block_tables=page_table)  # dense 2-D table with the flag set
    with pytest.raises(ValueError, match="num_kv_heads"):
        run(block_tables=head_tables[:1].contiguous())
    with pytest.raises(ValueError, match="seq_lens of shape"):
        run(seq_lens=sparse_lens[0].contiguous())  # 1-D seq_lens with the flag set
