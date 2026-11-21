import inspect
import math
from types import SimpleNamespace
from typing import Optional
import random

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.mla import _core as core
from flashinfer.mla import (
    MLALayerDimensions,
    compact_query_mla_dimensions,
    deepseek_mla_dimensions,
    supported_mla_layer_dimensions,
    smaller_mla_dimensions,
)
from flashinfer.utils import (
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)

global_workspace_buffer = None  # can.be empty initialized
global_trtllm_gen_fmha_workspace_buffer = None
workspace_size = 128 * 1024 * 1024

# Guard region we zero past the softmax slab so we can detect OOB writes.
TRTLLM_GEN_WORKSPACE_CHECK_BYTES = 1 * 1024 * 1024


def test_grouped_mla_selection_is_not_a_public_option():
    parameters = inspect.signature(
        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
    ).parameters

    assert "selects_grouped_mla" not in parameters


def trtllm_gen_workspace_softmax_end_bytes_decode(
    num_qo_heads: int, batch_size: int, max_q_len: int
) -> int:
    """End offset of the softmax slab in the generation-mode workspace [softmax|scratch].

    The C++ launcher allocates ``sizeof(float2) * softmax_slots`` bytes, i.e. 8 bytes per
    slot, where ``softmax_slots = num_qo_heads * batch_size * round_up(max_q_len, 256)``.
    """
    rounded_max_q_len = ((max_q_len + 255) // 256) * 256
    softmax_slab = (
        8 * num_qo_heads * batch_size * rounded_max_q_len
    )  # sizeof(float2) == 8
    return softmax_slab


def generate_sparse_indices(
    batch_size: int,
    q_len_per_request: int,
    seq_lens: torch.Tensor,
    topk: int,
    page_size: int,
    block_tables: torch.Tensor,
    device: str,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sparse attention indices for MLA.

    Returns:
        abs_indices: [batch_size, q_len_per_request, topk] - absolute positions in sequence
        indices_in_kvcache: [batch_size, q_len_per_request, topk] - positions in blocked KV cache
    """
    random.seed(seed)
    torch.manual_seed(seed)

    block_tables_cpu = block_tables.cpu()
    seq_lens_cpu = seq_lens.cpu()

    abs_indices = torch.empty(
        batch_size, q_len_per_request, topk, dtype=torch.int32, device="cpu"
    )
    indices_in_kvcache = torch.empty(
        batch_size, q_len_per_request, topk, dtype=torch.int32, device="cpu"
    )

    for i in range(batch_size):
        cur_seq_len = int(seq_lens_cpu[i].item())
        # Generate indices for each query position
        for j in range(q_len_per_request):
            # Randomly sample topk positions from the sequence
            if cur_seq_len > 0:
                # cur_abs_indices = torch.randperm(cur_seq_len, device="cpu")[:topk]
                cur_abs_indices = torch.arange(0, topk, device="cpu")
                # Convert to blocked indices
                cur_blocked_indices = block_tables_cpu[
                    i, cur_abs_indices // page_size
                ] * page_size + (cur_abs_indices % page_size)
            else:
                cur_abs_indices = torch.empty(0, dtype=torch.int32, device="cpu")
                cur_blocked_indices = torch.empty(0, dtype=torch.int32, device="cpu")

            # Pad with -1 if we don't have enough indices
            if len(cur_abs_indices) < topk:
                pad_len = topk - len(cur_abs_indices)
                cur_abs_indices = torch.cat(
                    [
                        cur_abs_indices,
                        torch.full((pad_len,), -1, device="cpu", dtype=torch.int32),
                    ]
                )
                cur_blocked_indices = torch.cat(
                    [
                        cur_blocked_indices,
                        torch.full((pad_len,), -1, device="cpu", dtype=torch.int32),
                    ]
                )

            # Randomly permute the indices
            # perm = torch.randperm(topk, device="cpu")
            perm = torch.arange(0, topk, device="cpu")
            cur_abs_indices = cur_abs_indices[perm]
            cur_blocked_indices = cur_blocked_indices[perm]

            abs_indices[i, j, :] = cur_abs_indices
            indices_in_kvcache[i, j, :] = cur_blocked_indices

    return abs_indices.to(device), indices_in_kvcache.to(device)


def sparse_mla_reference_torch(
    cache_seqlens: torch.Tensor,  # [batch_size]
    block_table: torch.Tensor,  # [batch_size, ?]
    q: torch.Tensor,  # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,  # [?, block_size, d]
    blocked_v: torch.Tensor,  # [?, block_size, dv]
    page_size: int,
    is_causal: bool,
    sm_scale: float,
    indices: torch.Tensor | None = None,  # [batch_size, s_q, topk]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch for MLA attention.
    Based on FlashMLA's reference implementation.

    Args:
        cache_seqlens: Sequence lengths for each batch [batch_size]
        block_table: Block table mapping [batch_size, max_num_blocks]
        q: Query tensor [batch_size, s_q, h_q, d]
        blocked_k: Blocked key cache [num_blocks, block_size, d]
        blocked_v: Blocked value cache [num_blocks, block_size, dv]
        page_size: Size of each block/page
        is_causal: Whether to apply causal masking
        sm_scale: Softmax scale factor
        indices: Optional sparse indices [batch_size, s_q, topk]

    Returns:
        output: Attention output [batch_size, s_q, h_q, dv]
        lse: Log-sum-exp values [batch_size, h_q, s_q]
    """

    def get_topk_attn_mask(s_q: int, s_k: int, indices: torch.Tensor):
        """Create attention mask for top-k sparse attention."""
        mask = torch.zeros(s_q, s_k, dtype=torch.bool)
        for i in range(s_q):
            cur_indices = indices[i]
            valid_indices = cur_indices[cur_indices != -1]
            mask[i, valid_indices] = True
        return mask

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,  # [h_q, s_q, d]
        key: torch.Tensor,  # [s_k, d]
        value: torch.Tensor,  # [s_k, dv]
        is_causal: bool,
        sm_scale: float,
        indices: torch.Tensor | None,  # [s_q, topk]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        h_q = query.size(0)
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        dv = value.shape[-1]

        query = query.float()
        key = key.float()
        value = value.float()

        # Handle NaN values in KV
        key[key != key] = 0.0
        value[value != value] = 0.0

        # Compute attention weights: [h_q, s_q, s_k]
        attn_weight = query @ key.transpose(-2, -1)

        # Apply masking if needed
        if (is_causal and query.size(1) > 1) or indices is not None:
            mask = torch.ones(s_q, s_k, dtype=torch.bool)
            if is_causal:
                mask = mask.tril(diagonal=s_k - s_q)
            if indices is not None:
                mask &= get_topk_attn_mask(s_q, s_k, indices)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float, device=query.device)
            mask = mask.to(device=query.device)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(query.dtype)

        # Scale and softmax
        attn_weight *= sm_scale
        lse = attn_weight.logsumexp(dim=-1)  # [h_q, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)

        # Compute output
        output = attn_weight @ value  # [h_q, s_q, dv]

        # Correct for query tokens which have no attendable keys
        lonely_q_mask = lse == float("-inf")
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output, lse

    b, s_q, h_q, d = q.size()
    dv = blocked_v.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()

    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32)

    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        cur_num_blocks = (cur_len + page_size - 1) // page_size
        cur_block_indices = block_table[i][0:cur_num_blocks]

        # Gather KV for this sequence
        cur_key = blocked_k[cur_block_indices].view(-1, d)[:cur_len, ...]
        cur_value = blocked_v[cur_block_indices].view(-1, dv)[:cur_len, ...]

        cur_out, cur_lse = scaled_dot_product_attention(
            i,
            q[i].transpose(0, 1),  # [h_q, s_q, d]
            cur_key,  # [s_k, d]
            cur_value,  # [s_k, dv]
            is_causal,
            sm_scale,
            indices[i] if indices is not None else None,
        )
        out_ref[i] = cur_out.transpose(0, 1)
        lse_ref[i] = cur_lse

    out_ref = out_ref.to(torch.bfloat16).to(q.device)
    return out_ref, lse_ref


def torch_reference_mla(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    kv_lora_rank,
    qk_rope_head_dim,
    softmax_scale,
    output_scale,
    page_size,
):
    """PyTorch reference for MLA decode. Returns [B, q_len, H, kv_lora_rank]."""
    B, q_len, H, D_qk = query.shape
    kv_flat = kv_cache.reshape(-1, D_qk)
    c_latent = kv_flat[:, :kv_lora_rank]
    c_rope = kv_flat[:, kv_lora_rank:]
    q_nope = query[..., :kv_lora_rank]
    q_rope = query[..., kv_lora_rank:]

    outputs = []
    for b in range(B):
        seq_len = seq_lens[b].item()
        num_pages = (seq_len + page_size - 1) // page_size
        pages = block_tables[b, :num_pages]
        kv_indices = []
        for p in pages:
            start = p.item() * page_size
            kv_indices.extend(range(start, start + page_size))
        kv_indices = kv_indices[:seq_len]
        kv_idx_t = torch.tensor(kv_indices, device=query.device)

        k_lat = c_latent[kv_idx_t]  # [seq_len, kv_lora_rank]
        k_rope = c_rope[kv_idx_t]  # [seq_len, rope_dim]

        attn_lat = torch.einsum("qhd,kd->qhk", q_nope[b].float(), k_lat.float())
        attn_rope = torch.einsum("qhd,kd->qhk", q_rope[b].float(), k_rope.float())
        attn = (attn_lat + attn_rope) * softmax_scale
        attn = F.softmax(attn, dim=-1)
        out_b = torch.einsum("qhk,kd->qhd", attn, k_lat.float()) * output_scale
        outputs.append(out_b)

    return torch.stack(outputs, dim=0)  # [B, q_len, H, kv_lora_rank]


def trtllm_batch_decode_mla(
    layer_dimensions: MLALayerDimensions,
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    page_size: int,
    q_len_per_request: int,
    dynamic_scale: bool,
    enable_pdl: bool,
    backend: str,
    MAX_SEQ_LEN: int,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool = True,
    use_fp16_softmax: bool = False,
    use_cum_seq_lens_q: bool = False,
    max_q_len_exceeds_total_q: bool = False,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "xqa":
        if compute_capability[0] != 12:
            pytest.skip("XQA MLA only supports SM120/SM121 GPUs")
        if q_len_per_request != 1 or dtype != torch.float8_e4m3fn:
            pytest.skip(
                "XQA MLA only supports q_len_per_request == 1 and dtype == torch.float8_e4m3fn"
            )
        if not uses_shared_paged_kv_idx:
            pytest.skip("xqa backend does not support separate KV page indices")
    if backend == "trtllm-gen":
        if compute_capability[0] != 10:
            pytest.skip("TRTLLM-GEN MLA only supports SM100 and SM103 GPUs")
    if backend == "cute-dsl":
        if compute_capability[0] not in (10, 11):
            pytest.skip("cute-dsl MLA requires SM100-SM110 (tcgen05)")
        from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported

        if not is_cute_dsl_arch_supported(*compute_capability):
            pytest.skip("installed CuTe DSL cannot target this device architecture")
        if dynamic_scale:
            pytest.skip("cute-dsl does not support dynamic_scale")
        if enable_pdl is not None:
            pytest.skip("cute-dsl does not support enable_pdl")
        if skips_softmax:
            pytest.skip("cute-dsl does not support skip_softmax")
        if not uses_shared_paged_kv_idx:
            pytest.skip("cute-dsl does not support separate KV page indices")
    if dynamic_scale and dtype != torch.float8_e4m3fn:
        pytest.skip("Dynamic scale is not supported for non-fp8 dtype")

    if skips_softmax and backend != "trtllm-gen":
        pytest.skip("skips_softmax is only supported for trtllm-gen backend")
    if use_cum_seq_lens_q and backend == "xqa":
        pytest.skip("XQA does not support cum_seq_lens_q")

    if use_fp16_softmax and backend != "trtllm-gen":
        pytest.skip("use_fp16_softmax=True is only supported for trtllm-gen backend")
    if use_fp16_softmax and get_compute_capability(torch.device("cuda:0")) != (10, 7):
        # trtllm-gen only exports the Fp16Softmax cubin variants for sm107a.
        pytest.skip("use_fp16_softmax=True is only supported on SM107 (Rubin)")

    torch.manual_seed(42)
    device = "cuda:0"
    qk_head_dim = (
        layer_dimensions.head_dimensions.qk_rope_head_dim
        + layer_dimensions.head_dimensions.kv_lora_rank
    )
    # Initialize tensors
    query = torch.randn(
        batch_size,
        q_len_per_request,
        layer_dimensions.num_heads,
        qk_head_dim,
        device=device,
    ).to(dtype)
    if use_cum_seq_lens_q:
        q_lens = (
            torch.arange(batch_size, device=device, dtype=torch.int32)
            % q_len_per_request
        ) + 1
        if max_q_len_exceeds_total_q:
            q_lens[0] = 0
        q_lens[-1] = q_len_per_request
        cum_seq_lens_q = torch.empty(batch_size + 1, device=device, dtype=torch.int32)
        cum_seq_lens_q[0] = 0
        cum_seq_lens_q[1:] = torch.cumsum(q_lens, dim=0)
        query_input = torch.cat(
            [query[i, : int(q_lens[i].item())] for i in range(batch_size)],
            dim=0,
        )
        # Overestimate the longest segment for CUDA-graph-friendly capacity.
        # Focused tests can also make this exceed the current compact length.
        max_q_len = (
            q_len_per_request + 1
            if max_q_len_exceeds_total_q
            else min(q_len_per_request + 1, query_input.size(0))
        )
    else:
        query_input = query
        cum_seq_lens_q = None
        q_lens = torch.full(
            (batch_size,), q_len_per_request, device=device, dtype=torch.int32
        )
        max_q_len = None

    num_tokens = MAX_SEQ_LEN * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size

    # Sequence lengths and block tables
    seq_lens = [torch.randint(1, MAX_SEQ_LEN, (1,)).item() for _ in range(batch_size)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()

    # Generate random but unique block IDs for all sequences
    total_blocks_needed = int(blocks_per_seq.sum().item())
    all_block_ids = torch.randperm(
        total_blocks_needed, device=device
    )  # Random permutation

    # Generate unique block IDs for all sequences
    block_id = 0
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int, device=device
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = int(blocks_per_seq[i].item())
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # For separate KV page indices, duplicate the page table rows since
    # MLA K and V share the same compressed representation.
    if not uses_shared_paged_kv_idx:
        block_tables_kernel = torch.stack([block_tables, block_tables], dim=1)
    else:
        block_tables_kernel = block_tables

    # Create interleaved KV cache
    # Allocate more than needed blocks, block_id is just enough, to mimick real-world cases
    kv_cache = torch.randn(
        size=(
            num_blocks,
            page_size,
            qk_head_dim,
        ),
        device=device,
    ).to(dtype)
    # (num_blocks, 1, page_size, kv_lora_rank + qk_rope_head_dim)

    # Allocate workspace buffer
    # todo(Yingyi): calculate the actual size of workspace buffer
    global global_workspace_buffer, global_trtllm_gen_fmha_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    workspace_buffer = global_trtllm_gen_fmha_workspace_buffer
    workspace_buffer_ref = global_workspace_buffer

    # Using a tiny threshold should give the same output as standard attention
    skip_softmax_threshold_scale_factor = 1e-30 if skips_softmax else None

    def maybe_get_lse_guard_end(softmax_end: int) -> int | None:
        # The C++ launcher carves the workspace as
        #     [softmax_slab | guard (1 MB) | scratch],
        # where ``softmax_slab = 8 * num_heads * batch * round_up(max_q_len, 256)``
        # bytes. The LSE guard subcheck is only meaningful when
        # ``softmax_end + 1 MB`` still fits in the test workspace. For the largest parametrized shapes
        # (e.g. batch=1024 * num_heads=128 * q_len=2 -> softmax_slab > 256 MB)
        # the slab alone overruns the 128 MB workspace; in that case we skip
        # the LSE subcheck and only exercise the output path (which does not
        # allocate softmax_slab when ``lse=None``). TODO: remove this branch
        # once trtllm-gen workspace sizing is computed dynamically per-call.
        guard_end = softmax_end + TRTLLM_GEN_WORKSPACE_CHECK_BYTES
        if guard_end > workspace_size:
            return None
        return guard_end

    # Only the trtllm-gen MLA path supports LSE output; other backends raise NotImplementedError.
    check_lse = (
        backend == "trtllm-gen"
        and not skips_softmax
        and not use_cum_seq_lens_q
        and dtype != torch.float8_e4m3fn
    )
    softmax_end = None
    guard_end = None
    if check_lse:
        softmax_end = trtllm_gen_workspace_softmax_end_bytes_decode(
            layer_dimensions.num_heads, batch_size, q_len_per_request
        )
        guard_end = maybe_get_lse_guard_end(softmax_end)
        if guard_end is None:
            # Preserve output coverage on shapes too large for the LSE guard
            # subcheck (see ``maybe_get_lse_guard_end`` for the size analysis).
            check_lse = False

    if check_lse:
        provided_lse = torch.full(
            (batch_size * q_len_per_request, layer_dimensions.num_heads),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        assert softmax_end is not None
        assert guard_end is not None
        workspace_buffer[softmax_end:guard_end].zero_()
    else:
        provided_lse = None

    # Run decode-MLA
    output_and_lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query_input,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=layer_dimensions.head_dimensions.qk_nope_head_dim,
        kv_lora_rank=layer_dimensions.head_dimensions.kv_lora_rank,
        qk_rope_head_dim=layer_dimensions.head_dimensions.qk_rope_head_dim,
        block_tables=block_tables_kernel,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale / ((128 + 64) ** 0.5),
        bmm2_scale=1.0,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        backend=backend,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        use_fp16_softmax=use_fp16_softmax,
        lse=provided_lse,
        return_lse=check_lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
    )
    if check_lse:
        output, lse_out = output_and_lse
        assert lse_out is provided_lse
        assert lse_out.dtype == torch.float32
        assert lse_out.shape == (
            batch_size * q_len_per_request,
            layer_dimensions.num_heads,
        )
        assert torch.isfinite(lse_out).all(), (
            "trtllm-gen MLA decode produced non-finite LSE"
        )
        assert softmax_end is not None
        assert guard_end is not None
        assert (workspace_buffer[softmax_end:guard_end].cpu().numpy() == 0).all(), (
            "trtllm-gen MLA decode wrote past the softmax slab"
        )
    else:
        output = output_and_lse

    # Run reference attention and align output
    sm_scale = scale / (
        (128 + 64) ** 0.5
    )  # use head dimension before matrix absorption
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer_ref,
        backend="fa2",
    )

    if dtype == torch.float8_e4m3fn:
        # convert query and kv_cache to bfloat16
        query = query.to(torch.bfloat16)
        kv_cache = kv_cache.to(torch.bfloat16)

    q_ref = query_input if use_cum_seq_lens_q else query
    if dtype == torch.float8_e4m3fn and use_cum_seq_lens_q:
        q_ref = q_ref.to(torch.bfloat16)

    q_indptr = torch.empty(batch_size + 1, device=device, dtype=torch.int32)
    q_indptr[0] = 0
    q_indptr[1:] = torch.cumsum(q_lens, dim=0)
    kv_indptr = torch.zeros_like(q_indptr)
    kv_indptr[1:] = torch.cumsum(blocks_per_seq, dim=0)
    kv_indices = all_block_ids.int()

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        seq_lens_tensor,
        layer_dimensions.num_heads,
        layer_dimensions.head_dimensions.kv_lora_rank,
        layer_dimensions.head_dimensions.qk_rope_head_dim,
        page_size,
        True,
        sm_scale,
        q_ref.dtype,
        kv_cache.dtype,
    )
    q_nope = q_ref[..., : layer_dimensions.head_dimensions.kv_lora_rank].reshape(
        -1,
        layer_dimensions.num_heads,
        layer_dimensions.head_dimensions.kv_lora_rank,
    )
    q_pe = q_ref[..., layer_dimensions.head_dimensions.kv_lora_rank :].reshape(
        -1,
        layer_dimensions.num_heads,
        layer_dimensions.head_dimensions.qk_rope_head_dim,
    )

    # todo: fix kv_cache
    ckv = kv_cache[..., : layer_dimensions.head_dimensions.kv_lora_rank]
    kpe = kv_cache[..., layer_dimensions.head_dimensions.kv_lora_rank :]

    o_ref = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    # cute-dsl fp8 kernel outputs fp8; cast to bf16 to match trtllm-gen / reference
    if backend == "cute-dsl" and output.dtype == torch.float8_e4m3fn:
        output = output.to(torch.bfloat16)

    if backend in ("trtllm-gen", "cute-dsl", "auto"):
        # check is nan
        assert not torch.isnan(o_ref).any(), "o_ref is nan"
        assert not torch.isnan(output).any(), "output is nan"

        if use_cum_seq_lens_q:
            output_view = output
            o_ref_view = o_ref
        else:
            output_view = output.reshape(
                batch_size, q_len_per_request, layer_dimensions.num_heads, -1
            )
            o_ref_view = o_ref.view(
                batch_size, q_len_per_request, layer_dimensions.num_heads, -1
            )

        if dtype == torch.float8_e4m3fn:
            rtol, atol = 1e-1, 1e-1
        else:
            rtol, atol = 1e-2, 1e-2

        if use_fp16_softmax and dtype != torch.float8_e4m3fn:
            rtol, atol = 3e-2, 3e-2

        try:
            torch.testing.assert_close(output_view, o_ref_view, rtol=rtol, atol=atol)
        except AssertionError as fa2_err:
            if backend == "cute-dsl":
                # fa2 reference may diverge from cute-dsl in some configs;
                # fall back to torch reference as ground truth.
                query_for_ref = (
                    query.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else query
                )
                kv_for_ref = (
                    kv_cache.to(torch.bfloat16)
                    if dtype == torch.float8_e4m3fn
                    else kv_cache
                )
                o_torch_ref = torch_reference_mla(
                    query_for_ref,
                    kv_for_ref,
                    block_tables,
                    seq_lens_tensor,
                    layer_dimensions.head_dimensions.kv_lora_rank,
                    layer_dimensions.head_dimensions.qk_rope_head_dim,
                    softmax_scale=sm_scale,
                    output_scale=1.0,
                    page_size=page_size,
                ).to(output.dtype)
                torch.testing.assert_close(output, o_torch_ref, rtol=rtol, atol=atol)
            else:
                print("output:", output)
                print("o_ref:", o_ref)
                raise fa2_err
    elif backend == "xqa":
        atol = 0.05
        rtol = 0.05

        diff_abs = torch.abs(
            o_ref.view(batch_size, q_len_per_request, layer_dimensions.num_heads, -1)
            - output
        )
        diff_rel = diff_abs / (
            torch.abs(
                o_ref.view(
                    batch_size, q_len_per_request, layer_dimensions.num_heads, -1
                )
            )
            + 1e-8
        )

        within_tolerance = (diff_abs <= atol) | (diff_rel <= rtol)

        pass_ratio = within_tolerance.float().mean().item()

        required_ratio = 0.95
        assert pass_ratio >= required_ratio, (
            f"Total {o_ref.numel()} elements, only {pass_ratio:.1%} meet tolerance criteria, "
            f"require at least {required_ratio:.1%}"
        )


def trtllm_batch_decode_mla_sparse(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    q_len_per_request: int,
    topk: int,
    is_varlen: bool,
    enable_pdl: bool,
    backend: str,
    qk_nope_head_dim: int,
    num_attn_heads: int,
    use_cum_seq_lens_q: bool = False,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "trtllm-gen":
        if compute_capability[0] != 10:
            pytest.skip("TRTLLM-GEN MLA only supports SM100 and SM103 GPUs")

    torch.manual_seed(42)
    device = "cuda:0"

    # Deepseek/GLM-5 attention config (decode-MLA)
    num_q_heads = num_attn_heads
    qk_rope_head_dim = 64
    kv_lora_rank = 512

    # Fixed or variable sequence lengths
    if is_varlen:
        # Variable sequence lengths
        MAX_SEQ_LEN = 4096
        seq_lens = [
            max(
                topk,
                int(
                    torch.distributions.Normal(MAX_SEQ_LEN, MAX_SEQ_LEN / 2)
                    .sample()
                    .item()
                ),
            )
            for _ in range(batch_size)
        ]
        seq_lens[-1] = MAX_SEQ_LEN  # Ensure at least one max length
        seq_lens = [min(s, MAX_SEQ_LEN) for s in seq_lens]
    else:
        # Fixed sequence length
        MAX_SEQ_LEN = 4096
        seq_lens = [MAX_SEQ_LEN] * batch_size

    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    # Initialize query tensors
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
    )
    query.clamp_(min=-1.0, max=1.0)
    query = query.to(dtype)

    # Calculate blocks needed
    page_size = 32
    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()
    total_blocks_needed = int(blocks_per_seq.sum().item())

    # Generate random but unique block IDs
    all_block_ids = torch.randperm(total_blocks_needed, device=device)

    # Create block tables
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int, device=device
    )
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = int(blocks_per_seq[i].item())
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # Create KV cache
    num_blocks = total_blocks_needed
    kv_cache = torch.randn(
        size=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim),
        device=device,
    )
    kv_cache.clamp_(min=-1.0, max=1.0)
    kv_cache = kv_cache.to(dtype)

    # Generate sparse indices
    abs_indices, indices_in_kvcache = generate_sparse_indices(
        batch_size,
        q_len_per_request,
        seq_lens_tensor,
        topk,
        page_size,
        block_tables,
        device,
    )

    # Mask unused KV cache entries with NaN for correctness checking
    kv_cache_ref = kv_cache.clone()
    if dtype == torch.float8_e4m3fn:
        kv_cache_ref = kv_cache_ref.to(torch.bfloat16)

    # Mark all positions as NaN initially
    all_indices = indices_in_kvcache.flatten().tolist()
    all_indices = list(set(all_indices))
    if -1 in all_indices:
        all_indices.remove(-1)

    # Only used indices should be valid
    kv_cache_flat = kv_cache_ref.view(-1, kv_lora_rank + qk_rope_head_dim)
    used_mask = torch.zeros(kv_cache_flat.size(0), dtype=torch.bool, device="cpu")
    used_mask[torch.tensor(all_indices, dtype=torch.int64, device="cpu")] = True
    kv_cache_flat[~used_mask] = float("0")

    # Allocate workspace buffers
    global global_workspace_buffer, global_trtllm_gen_fmha_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    workspace_buffer = global_trtllm_gen_fmha_workspace_buffer
    # workspace_buffer_ref = global_workspace_buffer

    # Run sparse decode-MLA
    cum_seq_lens_q = None
    max_q_len = None
    block_tables_input = indices_in_kvcache
    expected_shape = (batch_size, q_len_per_request, num_q_heads, kv_lora_rank)
    if use_cum_seq_lens_q:
        q_lens = (
            torch.arange(batch_size, device=device, dtype=torch.int32)
            % q_len_per_request
        ) + 1
        q_lens[-1] = q_len_per_request
        cum_seq_lens_q = torch.empty(batch_size + 1, device=device, dtype=torch.int32)
        cum_seq_lens_q[0] = 0
        cum_seq_lens_q[1:] = torch.cumsum(q_lens, dim=0)
        query_input = torch.cat(
            [query[i, : int(q_lens[i].item())] for i in range(batch_size)],
            dim=0,
        )
        block_tables_input = torch.cat(
            [indices_in_kvcache[i, : int(q_lens[i].item())] for i in range(batch_size)],
            dim=0,
        )
        max_q_len = q_len_per_request
        expected_shape = (query_input.size(0), num_q_heads, kv_lora_rank)
    else:
        query_input = query.clone()

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query_input,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables_input,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=topk,
        bmm1_scale=scale / ((qk_nope_head_dim + qk_rope_head_dim) ** 0.5),
        bmm2_scale=1.0,
        enable_pdl=enable_pdl,
        backend=backend,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
    )

    # For now, just check that output has correct shape and no NaNs
    assert output.shape == expected_shape, (
        f"Output shape {output.shape} != {expected_shape}"
    )

    # Check for NaNs
    if dtype != torch.float8_e4m3fn:
        assert not torch.isnan(output).any(), "Output contains NaN values"

    # Generate reference output using PyTorch implementation
    query_ref = query.clone()
    if dtype == torch.float8_e4m3fn:
        query_ref = query_ref.to(torch.bfloat16)

    # Split kv_cache into K and V components
    # K uses full dimension (kv_lora_rank + qk_rope_head_dim)
    # V uses only kv_lora_rank dimension
    blocked_k = kv_cache_ref  # [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]
    blocked_v = kv_cache_ref[
        ..., :kv_lora_rank
    ]  # [num_blocks, page_size, kv_lora_rank]

    sm_scale = scale / ((qk_nope_head_dim + qk_rope_head_dim) ** 0.5)

    out_ref, _ = sparse_mla_reference_torch(
        cache_seqlens=seq_lens_tensor,
        block_table=block_tables,
        q=query_ref,
        blocked_k=blocked_k,
        blocked_v=blocked_v,
        page_size=page_size,
        is_causal=True,  # Cover cases where number of attendable kv values are less than topk
        sm_scale=sm_scale,
        indices=abs_indices,
    )
    if use_cum_seq_lens_q:
        out_ref = torch.cat(
            [out_ref[i, : int(q_lens[i].item())] for i in range(batch_size)],
            dim=0,
        )

    # Compare outputs
    assert not torch.isnan(output).any(), "Kernel output contains NaN values"
    assert not torch.isnan(out_ref).any(), "Reference output contains NaN values"

    if dtype == torch.float8_e4m3fn:
        # FP8 has lower precision, use more relaxed tolerances
        try:
            torch.testing.assert_close(
                output.float(),
                out_ref.float(),
                rtol=1e-1,
                atol=1e-1,
            )
        except AssertionError as e:
            # Calculate element-wise differences for debugging
            diff = torch.abs(output.float() - out_ref.float())
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"Max difference: {max_diff}, Mean difference: {mean_diff}")
            output_sample = (
                output[0, 0, :8] if output.ndim == 3 else output[0, 0, 0, :8]
            )
            ref_sample = (
                out_ref[0, 0, :8] if out_ref.ndim == 3 else out_ref[0, 0, 0, :8]
            )
            print(f"Output sample: {output_sample}")
            print(f"Reference sample: {ref_sample}")
            raise e
    else:
        # BF16 should have better precision
        try:
            torch.testing.assert_close(
                output.float(),
                out_ref.float(),
                rtol=2e-2,
                atol=8e-4,
            )
        except AssertionError as e:
            # Calculate element-wise differences for debugging
            diff = torch.abs(output.float() - out_ref.float())
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"Max difference: {max_diff}, Mean difference: {mean_diff}")
            if output.ndim == 3:
                print(f"Output sample: {output[0, 0, :8]}")
                print(f"Output sample: {output[1, 0, :8]}")
                print(f"Reference sample: {out_ref[0, 0, :8]}")
                print(f"Reference sample: {out_ref[1, 0, :8]}")
            else:
                print(f"Output sample: {output[0, 0, 0, :8]}")
                print(f"Output sample: {output[0, 1, 0, :8]}")
                print(f"Reference sample: {out_ref[0, 0, 0, :8]}")
                print(f"Reference sample: {out_ref[0, 1, 0, :8]}")
            raise e

    print(
        f"Sparse MLA test passed: batch_size={batch_size}, topk={topk}, "
        f"q_len={q_len_per_request}, varlen={is_varlen}, dtype={dtype}"
    )


def _run_trtllm_batch_decode_mla_head_case(
    num_heads: int,
    batch_size: int,
    dtype: torch.dtype,
    max_seq_len: int,
) -> None:
    trtllm_batch_decode_mla(
        MLALayerDimensions(
            head_dimensions=deepseek_mla_dimensions,
            num_heads=num_heads,
        ),
        batch_size=batch_size,
        scale=1.0,
        dtype=dtype,
        page_size=64,
        q_len_per_request=1,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=max_seq_len,
        skips_softmax=False,
    )


@pytest.mark.parametrize("num_heads", [6, 12, 24, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("max_seq_len", [128, 1024])
def test_trtllm_batch_decode_mla_non_power_of_two_heads(
    num_heads: int,
    dtype: torch.dtype,
    max_seq_len: int,
) -> None:
    _run_trtllm_batch_decode_mla_head_case(
        num_heads,
        1,
        dtype,
        max_seq_len,
    )


def trtllm_mla_blackwell_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    page_size: int = 32,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kv_flat = kv_cache.reshape(-1, 576)
    outputs = []
    for batch_index in range(query.shape[0]):
        seq_len = int(seq_lens[batch_index].item())
        pages = block_tables[batch_index, : (seq_len + page_size - 1) // page_size]
        indices = (
            pages[:, None] * page_size
            + torch.arange(page_size, device=query.device, dtype=torch.int32)[None, :]
        ).reshape(-1)[:seq_len]
        batch_outputs = []
        for query_index in range(query.shape[1]):
            right = seq_len - query.shape[1] + query_index
            kv = kv_flat[indices.long()][: right + 1]
            logits = torch.einsum(
                "hd,kd->hk",
                query[batch_index, query_index, :, :512].float(),
                kv[:, :512].float(),
            )
            logits += torch.einsum(
                "hd,kd->hk",
                query[batch_index, query_index, :, 512:].float(),
                kv[:, 512:].float(),
            )
            logits = logits * softmax_scale
            if sinks is not None:
                # The sink is one extra logit per head that takes probability
                # mass but contributes no value.
                logits = torch.cat([logits, sinks.float()[:, None]], dim=-1)
            probabilities = F.softmax(logits, dim=-1)[:, : kv.shape[0]]
            batch_outputs.append(
                torch.einsum("hk,kd->hd", probabilities, kv[:, :512].float())
            )
        outputs.append(torch.stack(batch_outputs))
    return torch.stack(outputs).to(torch.bfloat16)


@pytest.mark.parametrize(
    "num_heads,dtype,max_seq_len",
    [
        (3, torch.bfloat16, 128),
        (9, torch.float8_e4m3fn, 1024),
        (17, torch.bfloat16, 1024),
        (33, torch.float8_e4m3fn, 128),
        (63, torch.bfloat16, 1024),
        (192, torch.bfloat16, 128),
    ],
)
def test_trtllm_batch_decode_mla_generic_non_power_of_two_heads(
    num_heads: int,
    dtype: torch.dtype,
    max_seq_len: int,
) -> None:
    _run_trtllm_batch_decode_mla_head_case(
        num_heads,
        1,
        dtype,
        max_seq_len,
    )


def test_trtllm_batch_decode_mla_non_power_of_two_heads_cum_seq_lens_q() -> None:
    trtllm_batch_decode_mla(
        MLALayerDimensions(
            head_dimensions=deepseek_mla_dimensions,
            num_heads=24,
        ),
        batch_size=2,
        scale=1.0,
        dtype=torch.bfloat16,
        page_size=64,
        q_len_per_request=2,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
        use_cum_seq_lens_q=True,
    )


def _run_trtllm_batch_decode_sparse_mla_head_case(
    num_heads: int,
    batch_size: int,
    dtype: torch.dtype,
    topk: int,
) -> None:
    trtllm_batch_decode_mla_sparse(
        batch_size=batch_size,
        scale=1.0,
        dtype=dtype,
        q_len_per_request=1,
        topk=topk,
        is_varlen=False,
        enable_pdl=False,
        backend="trtllm-gen",
        qk_nope_head_dim=128,
        num_attn_heads=num_heads,
        use_cum_seq_lens_q=False,
    )


@pytest.mark.parametrize(
    "num_heads,dtype",
    [
        (6, torch.bfloat16),
        (12, torch.bfloat16),
        (24, torch.bfloat16),
        (48, torch.bfloat16),
        (24, torch.float8_e4m3fn),
    ],
)
@pytest.mark.parametrize("topk", [128, 2048])
def test_trtllm_batch_decode_sparse_mla_non_power_of_two_heads(
    num_heads: int,
    dtype: torch.dtype,
    topk: int,
) -> None:
    _run_trtllm_batch_decode_sparse_mla_head_case(
        num_heads,
        1,
        dtype,
        topk,
    )


@pytest.mark.parametrize(
    "num_heads,dtype,topk",
    [
        (3, torch.bfloat16, 128),
        (9, torch.float8_e4m3fn, 128),
        (17, torch.bfloat16, 2048),
        (33, torch.float8_e4m3fn, 128),
        (63, torch.bfloat16, 2048),
    ],
)
def test_trtllm_batch_decode_sparse_mla_generic_non_power_of_two_heads(
    num_heads: int,
    dtype: torch.dtype,
    topk: int,
) -> None:
    _run_trtllm_batch_decode_sparse_mla_head_case(
        num_heads,
        1,
        dtype,
        topk,
    )


@pytest.mark.parametrize(
    "batch_size,dtype",
    [(1, torch.bfloat16), (2, torch.float8_e4m3fn)],
)
def test_trtllm_batch_decode_sparse_mla_power_of_two_heads(
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    _run_trtllm_batch_decode_sparse_mla_head_case(
        16,
        batch_size,
        dtype,
        2048,
    )


@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("topk", [128, 2048])
def test_trtllm_batch_decode_sparse_mla_small_power_of_two_heads(
    num_heads: int,
    dtype: torch.dtype,
    topk: int,
) -> None:
    _run_trtllm_batch_decode_sparse_mla_head_case(
        num_heads,
        1,
        dtype,
        topk,
    )


@pytest.mark.parametrize(
    "layer_dimensions",
    supported_mla_layer_dimensions,
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize(
    "q_len_per_request", [1, 2]
)  # todo(Yingyi): verify larger q_len_per_request
@pytest.mark.parametrize("dynamic_scale", [False])
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("backend", ["trtllm-gen", "xqa", "cute-dsl"])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode_mla(
    layer_dimensions: MLALayerDimensions,
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    page_size: int,
    q_len_per_request: int,
    dynamic_scale: bool,
    enable_pdl: bool,
    backend: str,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    if backend == "xqa" and layer_dimensions.head_dimensions == smaller_mla_dimensions:
        pytest.skip("XQA MLA does not support smaller MLA dimensions yet.")
    if backend == "xqa" and layer_dimensions.num_heads != 128:
        pytest.skip("XQA MLA only supports 128 query heads (head_group_ratio=128)")
    if (
        backend == "cute-dsl"
        and layer_dimensions.head_dimensions == smaller_mla_dimensions
    ):
        pytest.skip("cute-dsl MLA requires 512 latent dim and 64 rope dim")
    # Note: the cute-dsl branch runs the default implementation (auto →
    # monolithic, since this test passes no sinks).  We don't parametrize
    # cute_dsl_impl here because this test's purpose is public-API smoke
    # across all three backends; the modular vs monolithic matrix lives
    # in test_cute_dsl_mla_decode.py.
    trtllm_batch_decode_mla(
        layer_dimensions,
        batch_size,
        scale,
        dtype,
        page_size,
        q_len_per_request,
        dynamic_scale,
        enable_pdl,
        backend,
        1024,
        skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        use_cum_seq_lens_q=False,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("q_len_per_request", [2, 3, 4, 5, 6, 7, 8, 9, 16, 32])
def test_trtllm_batch_decode_grouped_mla(
    dtype: torch.dtype,
    num_heads: int,
    q_len_per_request: int,
):
    trtllm_batch_decode_mla(
        MLALayerDimensions(deepseek_mla_dimensions, num_heads),
        batch_size=1,
        scale=1.0,
        dtype=dtype,
        page_size=32,
        q_len_per_request=q_len_per_request,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_trtllm_batch_decode_grouped_mla_batch_16(dtype: torch.dtype):
    trtllm_batch_decode_mla(
        MLALayerDimensions(deepseek_mla_dimensions, 32),
        batch_size=16,
        scale=1.0,
        dtype=dtype,
        page_size=32,
        q_len_per_request=8,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
    )


def test_trtllm_batch_decode_q1_mla():
    trtllm_batch_decode_mla(
        MLALayerDimensions(deepseek_mla_dimensions, 16),
        batch_size=1,
        scale=1.0,
        dtype=torch.bfloat16,
        page_size=32,
        q_len_per_request=1,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
    )


def test_trtllm_batch_decode_q1_mla_uses_cga_kernel():
    if get_compute_capability(torch.device("cuda")) != (10, 0):
        pytest.skip("MLA H512 CGA kernel selection is specific to SM100")

    device = "cuda:0"
    batch_size = 1
    num_heads = 128
    page_size = 32
    max_seq_len = 4096
    head_dim_qk = 576
    head_dim_v = 512
    num_pages = max_seq_len // page_size

    query = torch.randn(
        batch_size,
        1,
        num_heads,
        head_dim_qk,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    kv_cache = torch.randn(
        num_pages,
        1,
        page_size,
        head_dim_qk,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    seq_lens = torch.tensor([max_seq_len], dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(workspace_size, dtype=torch.int8, device=device)

    def run_decode():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=128,
            kv_lora_rank=head_dim_v,
            qk_rope_head_dim=64,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (192**0.5),
            bmm2_scale=1.0,
            enable_pdl=False,
            backend="trtllm-gen",
        )

    # Warm up JIT compilation so the profile contains only the runtime selection and launch.
    run_decode()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as kernel_profile:
        run_decode()
        torch.cuda.synchronize()

    fmha_kernel_names = {
        event.name for event in kernel_profile.events() if event.name.startswith("fmha")
    }
    expected_kernel = "HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvCga"
    assert any(expected_kernel in name for name in fmha_kernel_names), fmha_kernel_names


def test_trtllm_batch_decode_grouped_mla_fixed_q_batch_stride():
    trtllm_batch_decode_mla(
        MLALayerDimensions(deepseek_mla_dimensions, 8),
        batch_size=2,
        scale=1.0,
        dtype=torch.bfloat16,
        page_size=32,
        q_len_per_request=2,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
    )


def test_trtllm_batch_decode_grouped_mla_variable_q_lengths():
    trtllm_batch_decode_mla(
        MLALayerDimensions(deepseek_mla_dimensions, 16),
        batch_size=3,
        scale=1.0,
        dtype=torch.bfloat16,
        page_size=32,
        q_len_per_request=8,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
        use_cum_seq_lens_q=True,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode_mla_cum_seq_lens_q(
    dtype: torch.dtype,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    trtllm_batch_decode_mla(
        supported_mla_layer_dimensions[0],
        batch_size=2,
        scale=1.0,
        dtype=dtype,
        page_size=64,
        q_len_per_request=2,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        use_cum_seq_lens_q=True,
    )


@pytest.mark.parametrize("backend", ["trtllm-gen", "auto"])
def test_trtllm_batch_decode_mla_variable_q_capacity_overestimate(backend: str):
    """TRT accepts graph capacity above the current compact token count."""
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("TRTLLM-GEN MLA requires SM100 or SM103")
    trtllm_batch_decode_mla(
        supported_mla_layer_dimensions[0],
        batch_size=2,
        scale=1.0,
        dtype=torch.bfloat16,
        page_size=64,
        q_len_per_request=2,
        dynamic_scale=False,
        enable_pdl=False,
        backend=backend,
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
        uses_shared_paged_kv_idx=False,
        use_cum_seq_lens_q=True,
        max_q_len_exceeds_total_q=True,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 16, 32, 64, 128],
)
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("q_len_per_request", [1, 2])
@pytest.mark.parametrize("topk", [128, 2048])
@pytest.mark.parametrize("is_varlen", [False, True])
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("backend", ["trtllm-gen"])
@pytest.mark.parametrize("qk_nope_head_dim", [128, 192])
@pytest.mark.parametrize("num_attn_heads", [128, 64])
def test_trtllm_batch_decode_mla_sparse(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    q_len_per_request: int,
    topk: int,
    is_varlen: bool,
    enable_pdl: bool,
    backend: str,
    qk_nope_head_dim: int,
    num_attn_heads: int,
):
    """
    Test sparse MLA decoding with top-k attention.
    Based on FlashMLA test patterns from:
    https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_decoding.py
    """
    trtllm_batch_decode_mla_sparse(
        batch_size,
        scale,
        dtype,
        q_len_per_request,
        topk,
        is_varlen,
        enable_pdl,
        backend,
        qk_nope_head_dim,
        num_attn_heads,
        use_cum_seq_lens_q=False,
    )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_trtllm_batch_decode_mla_sparse_compact_query(
    batch_size: int, dtype: torch.dtype
):
    """Compare BF16/FP8 compact-query sparse MLA against the reference."""
    dimensions = compact_query_mla_dimensions
    assert dimensions == core.MLAHeadDimensions(64, 64, 128, 512)
    assert dimensions in core.supported_mla_head_dimensions
    trtllm_batch_decode_mla_sparse(
        batch_size=batch_size,
        scale=1.0,
        dtype=dtype,
        q_len_per_request=1,
        topk=2048,
        is_varlen=True,
        enable_pdl=False,
        backend="trtllm-gen",
        qk_nope_head_dim=dimensions.qk_nope_head_dim,
        num_attn_heads=8,
    )


def test_trtllm_batch_decode_mla_sparse_cum_seq_lens_q():
    trtllm_batch_decode_mla_sparse(
        batch_size=2,
        scale=1.0,
        dtype=torch.bfloat16,
        q_len_per_request=2,
        topk=128,
        is_varlen=False,
        enable_pdl=False,
        backend="trtllm-gen",
        qk_nope_head_dim=128,
        num_attn_heads=128,
        use_cum_seq_lens_q=True,
    )


@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode_mla_native_block_table_width(
    uses_shared_paged_kv_idx: bool,
):
    page_size = 32
    max_seq_len = 1025
    block_table_width = (max_seq_len + page_size - 1) // page_size
    assert block_table_width == 33
    assert block_table_width % (128 // page_size) != 0

    trtllm_batch_decode_mla(
        supported_mla_layer_dimensions[0],
        batch_size=1,
        scale=1.0,
        dtype=torch.bfloat16,
        page_size=page_size,
        q_len_per_request=1,
        dynamic_scale=False,
        enable_pdl=False,
        backend="trtllm-gen",
        MAX_SEQ_LEN=max_seq_len,
        skips_softmax=False,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("q_len_per_request", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_trtllm_batch_decode_mla_preallocated_out(
    q_len_per_request: int,
    batch_size: int,
):
    """Issue #2856: pre-allocated out tensor rejected when q_len_per_req > 1.
    The shape check hardcoded 3D but query is 4D for multi-token generation."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip("trtllm-gen MLA requires SM100/SM103")

    device = "cuda:0"
    layer_dim = supported_mla_layer_dimensions[0]
    kv_lora_rank = layer_dim.head_dimensions.kv_lora_rank
    qk_nope_head_dim = layer_dim.head_dimensions.qk_nope_head_dim
    qk_rope_head_dim = layer_dim.head_dimensions.qk_rope_head_dim
    num_heads = layer_dim.num_heads
    head_dim_qk = kv_lora_rank + qk_rope_head_dim

    page_size = 64
    max_seq_len = 256
    num_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    head_dim_ckv_kpe = kv_lora_rank + qk_rope_head_dim

    kv_cache = torch.randn(
        num_pages_per_seq * batch_size,
        1,
        page_size,
        head_dim_ckv_kpe,
        dtype=torch.bfloat16,
        device=device,
    )
    block_tables = torch.arange(
        num_pages_per_seq * batch_size,
        device=device,
        dtype=torch.int32,
    ).reshape(batch_size, num_pages_per_seq)
    seq_lens = torch.full((batch_size,), max_seq_len, device=device, dtype=torch.int32)

    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_heads,
        head_dim_qk,
        dtype=torch.bfloat16,
        device=device,
    )

    global global_trtllm_gen_fmha_workspace_buffer
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.empty(
            workspace_size,
            dtype=torch.int8,
            device=device,
        )
    workspace = global_trtllm_gen_fmha_workspace_buffer

    bmm1_scale = 1.0 / (head_dim_qk**0.5)
    counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        batch_size, num_heads, get_device_sm_count(torch.device(device))
    )
    counter_buffer = torch.zeros(counter_bytes, dtype=torch.uint8, device=device)

    # out=None should work
    result_none = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        backend="trtllm-gen",
        multi_ctas_kv_counter_buffer=counter_buffer,
    )
    expected_shape = (batch_size, q_len_per_request, num_heads, kv_lora_rank)
    assert result_none.shape == expected_shape
    assert torch.count_nonzero(counter_buffer).item() == 0

    # out=pre-allocated should also work (this was the bug)
    out = torch.empty(expected_shape, dtype=torch.bfloat16, device=device)
    result_pre = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        backend="trtllm-gen",
        multi_ctas_kv_counter_buffer=counter_buffer,
    )
    assert result_pre.data_ptr() == out.data_ptr(), (
        "Expected kernel to write into provided out tensor"
    )
    assert result_pre.shape == expected_shape
    assert torch.count_nonzero(counter_buffer).item() == 0
    torch.testing.assert_close(result_none, result_pre, rtol=1e-3, atol=1e-3)

    # Buffer validation is shape-independent, so cover it once rather than for
    # every entry in the preallocated-output matrix.
    if batch_size == 1 and q_len_per_request == 1:
        with pytest.raises(
            ValueError, match="multi_ctas_kv_counter_buffer is too small"
        ):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=1.0,
                backend="trtllm-gen",
                multi_ctas_kv_counter_buffer=counter_buffer[:4],
            )

        strided_counter_buffer = torch.zeros(
            counter_bytes * 2, dtype=torch.uint8, device=device
        )[::2]
        assert strided_counter_buffer.numel() == counter_bytes
        assert not strided_counter_buffer.is_contiguous()
        with pytest.raises(
            ValueError, match="multi_ctas_kv_counter_buffer must be contiguous"
        ):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=1.0,
                backend="trtllm-gen",
                multi_ctas_kv_counter_buffer=strided_counter_buffer,
            )

        offset_counter_buffer = torch.zeros(
            counter_bytes + 1, dtype=torch.uint8, device=device
        )[1:]
        assert offset_counter_buffer.is_contiguous()
        assert offset_counter_buffer.data_ptr() % 16 != 0
        with pytest.raises(
            ValueError,
            match="multi_ctas_kv_counter_buffer must be 16-byte aligned",
        ):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=1.0,
                backend="trtllm-gen",
                multi_ctas_kv_counter_buffer=offset_counter_buffer,
            )


@pytest.mark.arch_blackwell
def test_trtllm_mla_prefill_matches_decode_multi_token_bf16():
    """The prefill name preserves explicit TRTLLM-GEN output/LSE semantics."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip("trtllm-gen MLA requires SM100/SM103")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    layer_dim = supported_mla_layer_dimensions[-1]
    head_dim = layer_dim.head_dimensions
    batch_size = 1
    q_len_per_request = 2
    page_size = 32
    max_seq_len = 64
    num_pages = max_seq_len // page_size
    head_dim_qk = head_dim.kv_lora_rank + head_dim.qk_rope_head_dim

    query = torch.randn(
        batch_size,
        q_len_per_request,
        layer_dim.num_heads,
        head_dim_qk,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        num_pages,
        1,
        page_size,
        head_dim_qk,
        dtype=torch.bfloat16,
        device=device,
    )
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        batch_size, num_pages
    )
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)

    expected_out_shape = (
        batch_size,
        q_len_per_request,
        layer_dim.num_heads,
        head_dim.kv_lora_rank,
    )
    expected_lse_shape = (
        batch_size * q_len_per_request,
        layer_dim.num_heads,
    )
    decode_workspace = torch.zeros(workspace_size, dtype=torch.int8, device=device)
    prefill_workspace = torch.zeros(workspace_size, dtype=torch.int8, device=device)
    decode_out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=device)
    prefill_out = torch.empty_like(decode_out)
    decode_lse = torch.empty(expected_lse_shape, dtype=torch.float32, device=device)
    prefill_lse = torch.empty_like(decode_lse)
    tensor_only_workspace = torch.zeros(workspace_size, dtype=torch.int8, device=device)
    tensor_only_out = torch.empty_like(decode_out)

    common = {
        "query": query,
        "kv_cache": kv_cache,
        "qk_nope_head_dim": head_dim.qk_nope_head_dim,
        "kv_lora_rank": head_dim.kv_lora_rank,
        "qk_rope_head_dim": head_dim.qk_rope_head_dim,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": max_seq_len,
        "bmm1_scale": 1.0
        / ((head_dim.qk_nope_head_dim + head_dim.qk_rope_head_dim) ** 0.5),
        "bmm2_scale": 1.0,
        "backend": "trtllm-gen",
        "return_lse": True,
    }
    decode_result = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        **common,
        workspace_buffer=decode_workspace,
        out=decode_out,
        lse=decode_lse,
    )
    prefill_result = flashinfer.prefill.trtllm_prefill_with_kv_cache_mla(
        **common,
        workspace_buffer=prefill_workspace,
        out=prefill_out,
        lse=prefill_lse,
    )

    assert isinstance(decode_result, tuple) and len(decode_result) == 2
    assert isinstance(prefill_result, tuple) and len(prefill_result) == 2
    assert decode_result[0] is decode_out
    assert prefill_result[0] is prefill_out
    assert decode_result[1] is decode_lse
    assert prefill_result[1] is prefill_lse
    assert decode_out.shape == prefill_out.shape == expected_out_shape
    assert decode_out.dtype == prefill_out.dtype == torch.bfloat16
    assert decode_lse.shape == prefill_lse.shape == expected_lse_shape
    assert decode_lse.dtype == prefill_lse.dtype == torch.float32
    assert torch.isfinite(decode_out).all()
    assert torch.isfinite(prefill_out).all()
    assert torch.isfinite(decode_lse).all()
    assert torch.isfinite(prefill_lse).all()
    torch.testing.assert_close(decode_out, prefill_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(decode_lse, prefill_lse, rtol=1e-3, atol=1e-3)

    tensor_only_result = flashinfer.prefill.trtllm_prefill_with_kv_cache_mla(
        **{**common, "return_lse": False},
        workspace_buffer=tensor_only_workspace,
        out=tensor_only_out,
    )
    assert tensor_only_result is tensor_only_out
    assert tensor_only_result.shape == expected_out_shape
    assert tensor_only_result.dtype == torch.bfloat16
    assert torch.isfinite(tensor_only_result).all()
    torch.testing.assert_close(tensor_only_result, decode_out, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "overrides,expected_domain",
    [
        (
            {"batch_size": 1, "q_len": 4, "total_q": 4, "kv_lens": (1024,)},
            "mla_bf16_native_split8_pdl",
        ),
        (
            {"batch_size": 1, "q_len": 4, "total_q": 4, "kv_lens": (512,)},
            "mla_bf16_vquarter",
        ),
        ({"batch_size": 1, "q_len": 8, "total_q": 8}, "mla_bf16_vquarter"),
        ({"batch_size": 4, "q_len": 8, "total_q": 32}, "mla_bf16_vhalf"),
        ({"batch_size": 4, "q_len": 16, "total_q": 64}, "mla_bf16_clc"),
        ({"batch_size": 64, "q_len": 16, "total_q": 1024}, "mla_bf16_clc"),
        (
            {
                "batch_size": 64,
                "q_len": 2,
                "total_q": 128,
                "max_seq_len": 8192,
                "table_width": 256,
                "kv_lens": (4096,) * 63 + (8192,),
            },
            "mla_bf16_clc_exact007790",
        ),
        (
            {
                "batch_size": 64,
                "q_len": 2,
                "total_q": 128,
                "max_seq_len": 8192,
                "table_width": 256,
                "kv_lens": (8192,) + (4096,) * 63,
            },
            "mla_bf16_clc",
        ),
        (
            {
                "batch_size": 512,
                "q_len": 16,
                "total_q": 8192,
                "kv_lens": (1024,) * 512,
            },
            "mla_bf16_clc_exact007826",
        ),
        ({"batch_size": 512, "q_len": 16, "total_q": 8192}, "mla_bf16_clc"),
        # A padded request (KV length below q_len) leaves the exact programs, which
        # read the host row plan, for the generic CLC program.
        (
            {
                "batch_size": 512,
                "q_len": 16,
                "total_q": 8192,
                "kv_lens": (0,) + (1024,) * 511,
            },
            "mla_bf16_clc",
        ),
        (
            {
                "batch_size": 64,
                "q_len": 2,
                "total_q": 128,
                "max_seq_len": 8192,
                "table_width": 256,
                "kv_lens": (1,) + (4096,) * 62 + (8192,),
            },
            "mla_bf16_clc",
        ),
        (
            {
                "page_size": 64,
                "enable_sink": True,
                "q_len": 1,
                "total_q": 1,
                "kv_lens": (1024,),
            },
            "mla_bf16_native_split8_pdl",
        ),
        ({"page_size": 64, "enable_sink": True}, "mla_bf16_vquarter"),
        ({"num_heads": 32, "qk_dim": 320, "value_dim": 256}, "mla_bf16_tail"),
        (
            {
                "dtype": torch.float8_e4m3fn,
                "num_heads": 32,
                "qk_dim": 320,
                "value_dim": 256,
            },
            "mla_fp8_tail",
        ),
        (
            {
                "dtype": torch.float8_e4m3fn,
                "q_len": 2,
                "total_q": 2,
                "kv_lens": (1024,),
            },
            "mla_fp8_p32_qk_l2",
        ),
        ({"dtype": torch.float8_e4m3fn, "q_len": 2, "total_q": 2}, "mla_fp8_tail"),
        (
            {
                "dtype": torch.float8_e4m3fn,
                "q_len": 16,
                "total_q": 16,
                "page_size": 64,
                "max_seq_len": 4096,
            },
            "mla_fp8_page64_pdl",
        ),
        (
            {"dtype": torch.float8_e4m3fn, "page_size": 64, "ragged_query": True},
            "mla_fp8_tail",
        ),
        # FP8 page 64 writes the LSE itself, at any max_seq_len; the other
        # aligned programs leave LSE requests to the generic tails.
        (
            {"dtype": torch.float8_e4m3fn, "page_size": 64, "wants_lse": True},
            "mla_fp8_page64_pdl",
        ),
        (
            {
                "dtype": torch.float8_e4m3fn,
                "page_size": 64,
                "wants_lse": True,
                "q_len": 2,
                "total_q": 2,
                "max_seq_len": 16384,
            },
            "mla_fp8_page64_pdl",
        ),
        (
            {
                "dtype": torch.float8_e4m3fn,
                "page_size": 64,
                "wants_lse": True,
                "enable_sink": True,
            },
            "mla_fp8_tail",
        ),
        (
            {
                "dtype": torch.float8_e4m3fn,
                "wants_lse": True,
                "q_len": 2,
                "total_q": 2,
                "kv_lens": (1024,),
            },
            "mla_fp8_tail",
        ),
        ({"page_size": 64, "wants_lse": True}, "mla_bf16_tail"),
        # The split-KV domain serves the devices _SPLITKV_RANGE lists (GB300: from
        # max_seq_len 2049, the first max_seq_len of tuning bucket 4096, up to 18
        # query rows), sinks, page 64, bmm2_scale != 1 and q_len > 1 included.
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "q_len": 1,
                "total_q": 1,
                "max_seq_len": 2048,
            },
            "mla_bf16_vquarter",
        ),
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "q_len": 1,
                "total_q": 1,
                "max_seq_len": 2049,
            },
            "mla_bf16_splitkv",
        ),
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "batch_size": 9,
                "q_len": 2,
                "total_q": 18,
                "max_seq_len": 4096,
            },
            "mla_bf16_splitkv",
        ),
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "batch_size": 19,
                "q_len": 1,
                "total_q": 19,
                "max_seq_len": 4096,
            },
            "mla_bf16_vquarter",
        ),
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "page_size": 64,
                "enable_sink": True,
                "q_len": 1,
                "total_q": 1,
                "max_seq_len": 16384,
            },
            "mla_bf16_splitkv",
        ),
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "bmm2_scale": 0.5,
                "q_len": 2,
                "total_q": 2,
                "max_seq_len": 4096,
            },
            "mla_bf16_splitkv",
        ),
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "skip_softmax": True,
                "max_seq_len": 4096,
            },
            "mla_bf16_vquarter",
        ),
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "ragged_query": True,
                "max_seq_len": 4096,
            },
            "mla_bf16_vquarter",
        ),
        (
            {"arch": "sm_103a", "num_sms": 152, "wants_lse": True, "max_seq_len": 4096},
            "mla_bf16_tail",
        ),
        # The native split-8 program keeps its exact geometry.
        (
            {
                "arch": "sm_103a",
                "num_sms": 152,
                "q_len": 4,
                "total_q": 4,
                "max_seq_len": 2048,
                "kv_lens": (1024,),
            },
            "mla_bf16_native_split8_pdl",
        ),
        # B200: from max_seq_len 2049, the first max_seq_len of tuning bucket
        # 4096, up to 18 query rows.
        (
            {
                "arch": "sm_100a",
                "num_sms": 148,
                "q_len": 1,
                "total_q": 1,
                "max_seq_len": 2048,
            },
            "mla_bf16_vquarter",
        ),
        (
            {
                "arch": "sm_100a",
                "num_sms": 148,
                "q_len": 1,
                "total_q": 1,
                "max_seq_len": 2049,
            },
            "mla_bf16_splitkv",
        ),
        (
            {"arch": "sm_100a", "num_sms": 148, "max_seq_len": 4096},
            "mla_bf16_splitkv",
        ),
        (
            {
                "arch": "sm_100a",
                "num_sms": 148,
                "batch_size": 19,
                "q_len": 1,
                "total_q": 19,
                "max_seq_len": 4096,
            },
            "mla_bf16_vhalf",
        ),
        # Devices not listed (B300 and other SM counts) and callers without arch
        # never select it.
        (
            {"arch": "sm_103a", "num_sms": 148, "max_seq_len": 4096},
            "mla_bf16_vquarter",
        ),
        ({"num_sms": 152, "max_seq_len": 4096}, "mla_bf16_vquarter"),
    ],
)
def test_trtllm_mla_blackwell_domain_selection(
    overrides: dict, expected_domain: str
) -> None:
    from flashinfer.mla.cake_trtllm_mla_blackwell import (
        ROUTES,
        exact_route_candidate,
        select_domain,
    )

    values = {
        "dtype": torch.bfloat16,
        "num_heads": 128,
        "qk_dim": 576,
        "value_dim": 512,
        "qk_nope_head_dim": 128,
        "page_size": 32,
        "topk": 0,
        "uses_shared_paged_kv_idx": True,
        "ragged_query": False,
        "enable_sink": False,
        "skip_softmax": False,
        "wants_lse": False,
        "bmm2_scale": 1.0,
        "batch_size": 1,
        "q_len": 8,
        "total_q": 8,
        "max_seq_len": 1024,
        "table_width": 32,
        "kv_lens": None,
        "num_sms": 148,
        "arch": None,
    }
    values.update(overrides)
    # The host reads the KV lengths only for the pre-filtered geometries; a row
    # that supplies them must be one, and a row that cannot select an exact,
    # native-split or P32 route must not read them.
    candidate = exact_route_candidate(
        **{
            key: values[key]
            for key in (
                "dtype",
                "num_heads",
                "page_size",
                "topk",
                "ragged_query",
                "enable_sink",
                "skip_softmax",
                "batch_size",
                "q_len",
                "table_width",
            )
        }
    )
    if values["kv_lens"] is not None:
        assert candidate
    domain = select_domain(**values)
    assert domain == expected_domain
    if not candidate:
        assert domain in {
            "mla_bf16_vquarter",
            "mla_bf16_vhalf",
            "mla_bf16_clc",
            "mla_bf16_splitkv",
            "mla_bf16_tail",
            "mla_fp8_tail",
            "mla_fp8_page64_pdl",
        }
    assert f"{domain}__sm_100a" in ROUTES and f"{domain}__sm_103a" in ROUTES


@pytest.mark.parametrize(
    ("domain", "arch", "num_sms", "work_items", "expected_route"),
    [
        ("mla_bf16_clc", "sm_100a", 148, 74, "mla_bf16_clc_one_wave__sm_100a__148"),
        ("mla_bf16_clc", "sm_100a", 148, 75, "mla_bf16_clc__sm_100a"),
        ("mla_bf16_clc", "sm_100a", 152, 76, "mla_bf16_clc__sm_100a"),
        ("mla_bf16_clc", "sm_103a", 152, 76, "mla_bf16_clc_one_wave__sm_103a__152"),
        ("mla_bf16_clc", "sm_103a", 152, 77, "mla_bf16_clc__sm_103a__152"),
        ("mla_bf16_clc", "sm_103a", 148, 74, "mla_bf16_clc__sm_103a"),
        (
            "mla_bf16_clc_exact007790",
            "sm_100a",
            148,
            1,
            "mla_bf16_clc_exact007790__sm_100a",
        ),
        (
            "mla_bf16_clc_exact007826",
            "sm_103a",
            152,
            1,
            "mla_bf16_clc_exact007826__sm_103a__152",
        ),
    ],
)
def test_trtllm_mla_blackwell_one_wave_clc_family(
    domain: str, arch: str, num_sms: int, work_items: int, expected_route: str
) -> None:
    from flashinfer.mla.cake_trtllm_mla_blackwell import ROUTES, launch_family

    # A generic CLC launch whose work items fit one wave of clusters takes the
    # one-wave program where the device has one; every other launch keeps its
    # domain's program.
    assert launch_family(domain, arch, num_sms, work_items) == ROUTES[expected_route]


def test_trtllm_mla_blackwell_rejects_non_scalar_or_nonfinite_scales() -> None:
    from flashinfer.mla.cake_trtllm_mla_blackwell import _normalize_scale

    with pytest.raises(TypeError, match="requires scalar bmm1_scale"):
        _normalize_scale(torch.ones(1), "bmm1_scale")
    with pytest.raises(ValueError, match="bmm2_scale must be finite"):
        _normalize_scale(float("inf"), "bmm2_scale")


def test_trtllm_mla_blackwell_row_metadata_is_computed_without_host_reads() -> None:
    from flashinfer.mla.cake_trtllm_mla_blackwell import (
        longest_first_order,
        row_metadata,
    )

    seq_lens = torch.tensor([1024, 300], dtype=torch.int32)
    row_batches, row_seq_lens = row_metadata(
        seq_lens, None, batch_size=2, q_len=2, total_q=4
    )
    assert row_batches.tolist() == [0, 0, 1, 1]
    assert row_seq_lens.tolist() == [1023, 1024, 299, 300]
    cum = torch.tensor([0, 1, 3], dtype=torch.int32)
    row_batches, row_seq_lens = row_metadata(
        seq_lens, cum, batch_size=2, q_len=2, total_q=3
    )
    assert row_batches.tolist() == [0, 1, 1]
    assert row_seq_lens.tolist() == [1024, 299, 300]
    # Longest rows first; equal tile counts keep row order.
    assert longest_first_order(
        torch.tensor([129, 1024, 128, 1024], dtype=torch.int32)
    ).tolist() == [1, 3, 0, 2]


def test_trtllm_mla_blackwell_page64_split_plan_is_size_based() -> None:
    """The FP8 page-64 split count reads the work items, max_seq_len and SM
    count alone. It is one split when the work items exceed half the clusters
    of a wave (``4 * work_items > num_sms``) or a ``max_seq_len`` row has at
    most two KV tiles; otherwise the largest count within one wave of clusters
    and half the row's KV tiles, rounded up."""
    from flashinfer.mla.cake_trtllm_mla_blackwell import plan_num_split

    assert plan_num_split(16, 4096, 148) == 4
    assert plan_num_split(1, 1024, 148) == 4
    assert plan_num_split(2, 600, 148) == 3
    assert plan_num_split(37, 1024, 148) == 2
    assert plan_num_split(38, 1024, 152) == 2
    # One split: the producer writes the output and the LSE, no reducer runs.
    assert plan_num_split(38, 1024, 148) == 1
    assert plan_num_split(39, 1024, 152) == 1
    assert plan_num_split(64, 8192, 152) == 1
    assert plan_num_split(128, 1024, 148) == 1
    assert plan_num_split(1, 128, 148) == 1
    assert plan_num_split(1, 256, 152) == 1
    assert plan_num_split(1, 257, 152) == 2
    for num_sms in (148, 152):
        for work_items in range(1, 257):
            for max_seq_len in (1, 128, 256, 257, 600, 1024, 4096, 16384):
                split = plan_num_split(work_items, max_seq_len, num_sms)
                tiles = -(-max_seq_len // 128)
                half = -(-tiles // 2)
                case = (work_items, max_seq_len, num_sms)
                assert 1 <= split <= half, case
                assert split == 1 or work_items * split * 2 <= num_sms, case
                if work_items * 4 > num_sms or tiles <= 2:
                    assert split == 1, case
                else:
                    # The largest count within the wave and the tile cap.
                    assert split >= 2, case
                    wave_full = work_items * (split + 1) * 2 > num_sms
                    assert wave_full or split == half, case


def test_trtllm_mla_blackwell_splitkv_plan_matches_rule() -> None:
    """The split-KV count comes from host-known sizes: one wave of clusters at
    most, at most 32 splits, equal tiles per split of a max_seq_len row."""
    from flashinfer.mla.cake_trtllm_mla_blackwell import (
        _SPLITKV_MAX_SPLIT,
        plan_kv_split,
    )

    # GB300 (152 SMs, 76 resident clusters): (rows, max_seq_len) -> splits.
    expected = {
        (1, 2048): 16,
        (1, 4096): 32,
        (1, 16384): 32,
        (2, 4096): 32,
        (4, 16384): 19,
        (8, 2048): 8,
        (8, 16384): 9,
        (9, 4096): 8,
        (16, 4096): 4,
        (18, 16384): 4,
        (1, 1025): 9,
    }
    for (rows, max_seq_len), split in expected.items():
        assert plan_kv_split(rows, max_seq_len, 152) == split, (rows, max_seq_len)
    for num_sms in (148, 152):
        resident = num_sms // 2
        for rows in range(1, 39):
            for max_seq_len in (1025, 2048, 4096, 8192, 16384, 131072):
                split = plan_kv_split(rows, max_seq_len, num_sms)
                tiles = -(-max_seq_len // 128)
                per_split = -(-tiles // split)
                assert 1 <= split <= _SPLITKV_MAX_SPLIT
                assert rows * split <= resident, (num_sms, rows, max_seq_len)
                # Every split of a max_seq_len row is busy and no count below
                # gives the same tiles per split.
                assert (split - 1) * per_split < tiles
                assert split == 1 or -(-tiles // (split - 1)) > per_split


_SMEM_FILL_SOURCE = r"""
#include <torch/extension.h>

__global__ void fill_shared_memory(int bytes) {
  extern __shared__ unsigned char smem[];
  for (int i = threadIdx.x; i < bytes; i += blockDim.x) smem[i] = 0x7F;
  __syncthreads();
}

void fill_shared_memory_of_every_sm() {
  int device = 0, sms = 0, bytes = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaFuncSetAttribute(fill_shared_memory, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  fill_shared_memory<<<4 * sms, 1024, bytes>>>(bytes);
}
"""


@pytest.mark.parametrize("kv_lens", [(600, 2), (2, 600)])
def test_trtllm_mla_blackwell_fp8_page64_empty_splits(kv_lens) -> None:
    """A row with fewer KV tiles than splits has empty splits; they must not touch its output.

    The split count is planned from the longest row, so the 2-key row's single tile leaves two of its three splits
    empty. Every SM's shared memory is filled with 0x7F bytes (an FP8 E4M3 NaN) right before each of eight calls, so a
    split that reads stale shared memory instead of skipping its PV turns the whole row non-finite. The 2-key row's two
    keys are identical, so its output is that key's value row whichever of its keys the route attends.
    """
    if not torch.cuda.is_available():
        pytest.skip("TRT-LLM MLA Blackwell requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in {(10, 0), (10, 3)}:
        pytest.skip("TRT-LLM MLA Blackwell requires SM100a or SM103a")
    from torch.utils.cpp_extension import load_inline

    from flashinfer.mla.cake_trtllm_mla_blackwell import plan_num_split

    smem_fill = load_inline(
        name="test_trtllm_mla_blackwell_smem_fill",
        cpp_sources=["void fill_shared_memory_of_every_sm();"],
        cuda_sources=[_SMEM_FILL_SOURCE],
        functions=["fill_shared_memory_of_every_sm"],
        verbose=False,
    )
    device = torch.device("cuda")
    torch.manual_seed(0)
    page_size, max_seq_len = 64, max(kv_lens)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert plan_num_split(len(kv_lens), max_seq_len, sms) == 3
    pages = -(-max_seq_len // page_size)
    query = (torch.randn(len(kv_lens), 1, 128, 576, device=device) * 0.5).to(
        torch.float8_e4m3fn
    )
    kv_cache = (
        torch.randn(len(kv_lens) * pages, page_size, 576, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    block_tables = torch.arange(
        len(kv_lens) * pages, dtype=torch.int32, device=device
    ).reshape(len(kv_lens), pages)
    short = kv_lens.index(2)
    kv_cache[block_tables[short, 0], 1] = kv_cache[block_tables[short, 0], 0]

    def decode():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=torch.empty(1, dtype=torch.uint8, device=device),
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=block_tables,
            seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
            max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (192**0.5),
            bmm2_scale=1.0,
            backend="cake",
        )

    decode()  # builds the route and its plan before shared memory is filled
    for _ in range(8):
        smem_fill.fill_shared_memory_of_every_sm()
        output = decode()
        assert output.isfinite().all()
    keys = kv_cache.float().reshape(-1, 576)
    for batch, kv_len in enumerate(kv_lens):
        tokens = (
            block_tables[batch, :, None] * page_size
            + torch.arange(page_size, device=device)[None, :]
        ).reshape(-1)
        visible = keys[tokens[:kv_len].long()]
        probabilities = torch.softmax(
            query[batch, 0].float() @ visible.T / (192**0.5), dim=-1
        )
        reference = (probabilities @ visible[:, :512]).to(torch.bfloat16)
        torch.testing.assert_close(output[batch, 0], reference, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "batch_size,q_len_per_request,page_size,enable_sink",
    [
        (1, 4, 32, False),
        (1, 8, 32, False),
        (4, 2, 32, False),
        (4, 4, 32, False),
        (4, 8, 32, False),
        (4, 16, 32, False),
        (1, 1, 64, True),
    ],
)
def test_trtllm_mla_blackwell_bf16_dispatch(
    batch_size: int, q_len_per_request: int, page_size: int, enable_sink: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("TRT-LLM MLA Blackwell requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in {(10, 0), (10, 3)}:
        pytest.skip("TRT-LLM MLA Blackwell requires SM100a or SM103a")

    device = torch.device("cuda")
    torch.manual_seed(42)
    max_seq_len = 1024
    pages_per_sequence = max_seq_len // page_size
    # Inputs of magnitude 0.5 give a peaked softmax, so a score read from the
    # wrong keys moves the output well past the tolerance; near-uniform
    # probabilities from small inputs would hide it.
    query = (
        torch.randn(batch_size, q_len_per_request, 128, 576, device=device) * 0.5
    ).to(torch.bfloat16)
    kv_cache = (
        torch.randn(batch_size * pages_per_sequence, page_size, 576, device=device)
        * 0.5
    ).to(torch.bfloat16)
    block_tables = torch.arange(
        batch_size * pages_per_sequence, dtype=torch.int32, device=device
    ).reshape(batch_size, pages_per_sequence)
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    sinks = (
        torch.randn(128, dtype=torch.float32, device=device) * 2.0
        if enable_sink
        else None
    )
    workspace = torch.empty(1, dtype=torch.uint8, device=device)
    bmm1_scale = 1.0 / (192**0.5)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        sinks=[sinks] if enable_sink else None,
        backend="cake",
    )
    reference = trtllm_mla_blackwell_reference(
        query, kv_cache, block_tables, seq_lens, bmm1_scale, page_size, sinks
    )
    assert output.shape == (batch_size, q_len_per_request, 128, 512)
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("q_len_per_request", [1, 2])
def test_trtllm_mla_blackwell_fp8_page64_attends_the_last_key(
    q_len_per_request: int,
) -> None:
    """Every query row of the FP8 page-64 route attends its causal keys, the last one included.

    The last key of each request is aimed at head 0's query of the request's last row, so its logit dominates: a route
    that drops it moves head 0's output by about the whole value range, far beyond the FP8 tolerance.
    """
    if not torch.cuda.is_available():
        pytest.skip("TRT-LLM MLA Blackwell requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in {(10, 0), (10, 3)}:
        pytest.skip("TRT-LLM MLA Blackwell requires SM100a or SM103a")

    device = torch.device("cuda")
    torch.manual_seed(0)
    page_size, batch_size, max_seq_len = 64, 2, 1024
    pages = max_seq_len // page_size
    query = (
        torch.randn(batch_size, q_len_per_request, 128, 576, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    kv_cache = (
        torch.randn(batch_size * pages, page_size, 576, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    block_tables = torch.arange(
        batch_size * pages, dtype=torch.int32, device=device
    ).reshape(batch_size, pages)
    kv_lens = [max_seq_len, 600]
    for batch, kv_len in enumerate(kv_lens):
        page, slot = divmod(kv_len - 1, page_size)
        kv_cache[block_tables[batch, page], slot] = (
            (query[batch, -1, 0].float() * 6).clamp(-400, 400).to(torch.float8_e4m3fn)
        )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    bmm1_scale = 1.0 / (192**0.5)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=torch.empty(1, dtype=torch.uint8, device=device),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        backend="cake",
    )
    keys = kv_cache.float().reshape(-1, 576)
    for batch, kv_len in enumerate(kv_lens):
        tokens = (
            block_tables[batch, :, None] * page_size
            + torch.arange(page_size, device=device)[None, :]
        ).reshape(-1)
        for row in range(q_len_per_request):
            visible = keys[tokens[: kv_len - q_len_per_request + row + 1].long()]
            q = query[batch, row].float()
            probabilities = torch.softmax(q @ visible.T * bmm1_scale, dim=-1)
            reference = (probabilities @ visible[:, :512]).to(torch.bfloat16)
            torch.testing.assert_close(
                output[batch, row], reference, rtol=0.1, atol=0.1
            )


# --- Device-derived rows of the value-split, generic CLC and FP8 page-64 programs ---

_BLACKWELL_BMM1 = 1.0 / (192**0.5)


def _skip_unless_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("TRT-LLM MLA Blackwell requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in {(10, 0), (10, 3)}:
        pytest.skip("TRT-LLM MLA Blackwell requires SM100a or SM103a")


def _blackwell_resident() -> int:
    return max(1, get_device_sm_count(torch.device("cuda")) // 2)


def _list_splitkv_device(domain: str, monkeypatch) -> None:
    """For a split-KV case, list the current device in ``_SPLITKV_RANGE`` from
    max_seq_len 1025 up to 18 query rows, so the case reaches the split-KV
    program on every Blackwell device, also one that is not listed or whose
    listed range starts at a longer max_seq_len (B200 and GB300: 2049)."""
    from flashinfer.mla import cake_trtllm_mla_blackwell as module

    if domain != "mla_bf16_splitkv":
        return
    facts = module._device_facts(torch.cuda.current_device())
    if module._SPLITKV_RANGE.get(facts) != (1025, 18):
        monkeypatch.setattr(
            module, "_SPLITKV_RANGE", {**module._SPLITKV_RANGE, facts: (1025, 18)}
        )


def _blackwell_decode(query, kv_cache, block_tables, seq_lens, max_seq_len, **kwargs):
    """``backend="cake"`` decode called on the Blackwell module directly."""
    from flashinfer.mla.cake_trtllm_mla_blackwell import trtllm_mla_blackwell_decode

    sinks = kwargs.pop("sinks", None)
    arguments = {
        "qk_nope_head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "sparse_mla_top_k": 0,
        "out": None,
        "bmm1_scale": _BLACKWELL_BMM1,
        "bmm2_scale": 1.0,
        "sinks": None if sinks is None else [sinks],
        "skip_softmax_threshold_scale_factor": None,
        "enable_pdl": None,
        "uses_shared_paged_kv_idx": True,
        "lse": None,
        "return_lse": False,
        "cum_seq_lens_q": None,
        "max_q_len": None,
        "multi_ctas_kv_counter_buffer": None,
        "sparse_mla_top_k_lens": None,
        "enable_dcp": False,
        "backend": "cake",
    }
    arguments.update(kwargs)
    return trtllm_mla_blackwell_decode(
        query, kv_cache, block_tables, seq_lens, max_seq_len, **arguments
    )


def _blackwell_inputs(
    kv_lens: list[int],
    q_len: int,
    page_size: int,
    dtype: torch.dtype,
    *,
    width: Optional[int] = None,
    trailing_poison: int = 0,
    seed: int = 0,
):
    """Query, KV cache, page table and KV lengths of a paged decode request.

    Page ids are a permutation of the first ``batch * width`` pages; one more page holds
    NaN and no table entry names it.  ``trailing_poison`` entries naming that page
    follow the table in its allocation, so a page-id read past the caller's table
    that feeds a V tile turns its row non-finite.  With ``trailing_poison=0`` the
    table is its own allocation of exactly ``batch * width`` entries.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = len(kv_lens)
    if width is None:
        width = max(1, -(-max(kv_lens) // page_size))
    nan_page = batch * width
    query = (
        torch.randn(batch, q_len, 128, 576, device=device, generator=generator) * 0.5
    ).to(dtype)
    kv_cache = (
        torch.randn(nan_page + 1, page_size, 576, device=device, generator=generator)
        * 0.5
    )
    kv_cache[nan_page] = float("nan")
    kv_cache = kv_cache.to(dtype)
    storage = torch.full(
        (batch * width + trailing_poison,), nan_page, dtype=torch.int32, device=device
    )
    storage[: batch * width] = torch.randperm(
        batch * width, device=device, generator=generator
    ).to(torch.int32)
    block_tables = storage[: batch * width].view(batch, width)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return query, kv_cache, block_tables, seq_lens


def _padded_kv_lens(batch: int, q_len: int, max_len: int) -> list[int]:
    """Live requests interleaved with padded ones (KV length 0, or shorter than q_len)."""
    lens = []
    for b in range(batch):
        kind = b % 4
        if kind == 0:
            lens.append(max_len - (37 * b) % (max_len // 2))
        elif kind == 1:
            lens.append(0)
        elif kind == 2:
            lens.append(max_len)
        else:
            lens.append(q_len // 2)
    return lens


def _query_rows(kv_lens, q_lens) -> list[tuple[int, int]]:
    """(request, causal KV length) of every query row."""
    return [
        (batch, kv_len - q_len + index + 1)
        for batch, (kv_len, q_len) in enumerate(zip(kv_lens, q_lens, strict=True))
        for index in range(q_len)
    ]


def _blackwell_rows_reference(
    q_rows: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    rows: list[tuple[int, int]],
    page_size: int,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP32 attention of every query row ``[rows, heads, 576]``; a row of KV length <= 0 is zero."""
    device = q_rows.device
    queries = q_rows.float()
    kv_flat = kv_cache.reshape(-1, 576).float()
    tokens = torch.arange(page_size, device=device)
    output = torch.zeros(q_rows.shape[0], q_rows.shape[1], 512, device=device)
    by_batch: dict[int, list[tuple[int, int]]] = {}
    for row, (batch, length) in enumerate(rows):
        if length > 0:
            by_batch.setdefault(batch, []).append((row, length))
    for batch, items in by_batch.items():
        longest = max(length for _, length in items)
        pages = block_tables[batch, : -(-longest // page_size)].long()
        kv = kv_flat[
            (pages[:, None] * page_size + tokens[None, :]).reshape(-1)[:longest]
        ]
        index = torch.tensor([row for row, _ in items], device=device)
        lengths = torch.tensor([length for _, length in items], device=device)
        logits = torch.einsum("rhd,kd->rhk", queries[index], kv) * _BLACKWELL_BMM1
        visible = torch.arange(longest, device=device)[None, :] < lengths[:, None]
        logits = logits.masked_fill(~visible[:, None, :], float("-inf"))
        if sinks is not None:
            sink_logits = sinks.float()[None, :, None].expand(len(items), -1, 1)
            logits = torch.cat([logits, sink_logits], dim=-1)
        probabilities = torch.softmax(logits, dim=-1)[..., :longest]
        output[index] = torch.einsum("rhk,kd->rhd", probabilities, kv[:, :512])
    return output.to(torch.bfloat16)


def _assert_rows(output, reference, rows, dtype) -> None:
    """Padded rows are exactly zero; live rows match the reference."""
    output = output.reshape(len(rows), 128, 512)
    dead = torch.tensor([length <= 0 for _, length in rows], device=output.device)
    if bool(dead.any()):
        assert torch.equal(output[dead], torch.zeros_like(output[dead]))
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-1
    torch.testing.assert_close(
        output[~dead], reference[~dead], rtol=tolerance, atol=tolerance
    )


@pytest.fixture
def blackwell_domains(monkeypatch):
    """The domains ``trtllm_mla_blackwell_decode`` selects, in call order."""
    from flashinfer.mla import cake_trtllm_mla_blackwell as module

    seen: list[str] = []
    select = module.select_domain

    def spy(**kwargs):
        domain = select(**kwargs)
        seen.append(domain)
        return domain

    monkeypatch.setattr(module, "select_domain", spy)
    return seen


def test_trtllm_mla_blackwell_device_row_bindings() -> None:
    from flashinfer.mla.cake_trtllm_mla_blackwell import (
        _CLC_BINDING,
        _FP8_PAGE64_BINDING,
        _SPLITKV_BINDING,
        _VALUE_SPLIT_BINDINGS,
        FAMILIES,
        ROUTES,
    )

    # The launched binding decides the row inputs: every route of the generic CLC
    # domain, its one-wave programs and the exact-geometry routes of the sm_100a and
    # 152-SM sm_103a modules launch the device-row CLC binding.
    for route in (
        "mla_bf16_clc__sm_100a",
        "mla_bf16_clc__sm_103a",
        "mla_bf16_clc__sm_103a__152",
        "mla_bf16_clc_one_wave__sm_100a__148",
        "mla_bf16_clc_one_wave__sm_103a__152",
        "mla_bf16_clc_exact007826__sm_100a",
        "mla_bf16_clc_exact007826__sm_103a__152",
    ):
        assert FAMILIES[ROUTES[route]]["binding"] == _CLC_BINDING, route
    for route in (
        "mla_bf16_clc_exact007790__sm_100a",
        "mla_bf16_clc_exact007790__sm_103a",
        "mla_bf16_clc_exact007826__sm_103a",
        "mla_bf16_native_split8_pdl__sm_100a",
        "mla_bf16_native_split8_pdl__sm_103a",
    ):
        binding = FAMILIES[ROUTES[route]]["binding"]
        assert binding != _CLC_BINDING and binding not in _VALUE_SPLIT_BINDINGS, route
    for arch in ("sm_100a", "sm_103a"):
        assert (
            _VALUE_SPLIT_BINDINGS[
                FAMILIES[ROUTES[f"mla_bf16_vquarter__{arch}"]]["binding"]
            ]
            == 4
        )
        assert (
            _VALUE_SPLIT_BINDINGS[
                FAMILIES[ROUTES[f"mla_bf16_vhalf__{arch}"]]["binding"]
            ]
            == 2
        )
        assert (
            FAMILIES[ROUTES[f"mla_fp8_page64_pdl__{arch}"]]["binding"]
            == _FP8_PAGE64_BINDING
        )
        order = FAMILIES[ROUTES[f"mla_bf16_clc_order__{arch}"]]
        assert order["arch"] == arch and order["domain"] == "mla_bf16_clc_order"
        # The split-KV binding takes the caller's seq_lens and block_tables too and
        # links its mainloop and both reducer builds (prefetch_any, prefetch_all).
        splitkv = FAMILIES[ROUTES[f"mla_bf16_splitkv__{arch}"]]
        assert splitkv["binding"] == _SPLITKV_BINDING
        assert splitkv["arch"] == arch and splitkv["domain"] == "mla_bf16_splitkv"
        assert splitkv["dependencies"] == [
            "cake_trtllm_mla_blackwell_3e8ec0cdecccb3b5f371",
            "cake_trtllm_mla_blackwell_be9225dfb9d6e35904ed",
            "cake_trtllm_mla_blackwell_1e364888a6277ea9a2c3",
        ]
    assert _SPLITKV_BINDING != _CLC_BINDING
    assert _SPLITKV_BINDING not in _VALUE_SPLIT_BINDINGS
    assert _SPLITKV_BINDING != _FP8_PAGE64_BINDING


# name: (dtype, batch, q_len, page_size, sink, return_lse, max_len, domain)
_PADDED_ROW_CASES = {
    "vquarter": (torch.bfloat16, 4, 4, 32, False, False, 1024, "mla_bf16_vquarter"),
    "vquarter_page64_sink": (
        torch.bfloat16,
        3,
        2,
        64,
        True,
        False,
        1024,
        "mla_bf16_vquarter",
    ),
    "vhalf": (torch.bfloat16, 8, 4, 32, False, False, 1024, "mla_bf16_vhalf"),
    "clc_one_wave": (torch.bfloat16, 16, 4, 32, False, False, 1024, "mla_bf16_clc"),
    "clc_ordered": (torch.bfloat16, 64, 4, 32, False, False, 1024, "mla_bf16_clc"),
    "clc_unordered": (torch.bfloat16, 300, 4, 32, False, False, 256, "mla_bf16_clc"),
    "fp8_page64": (
        torch.float8_e4m3fn,
        4,
        2,
        64,
        False,
        False,
        1024,
        "mla_fp8_page64_pdl",
    ),
    # One split (plan_num_split): 128 and 40 query rows exceed half the
    # clusters of a wave; a 256-token max_seq_len has two KV tiles.
    "fp8_page64_one_split": (
        torch.float8_e4m3fn,
        128,
        1,
        64,
        False,
        False,
        1024,
        "mla_fp8_page64_pdl",
    ),
    "fp8_page64_q2_one_split": (
        torch.float8_e4m3fn,
        20,
        2,
        64,
        False,
        False,
        1024,
        "mla_fp8_page64_pdl",
    ),
    "fp8_page64_short_one_split": (
        torch.float8_e4m3fn,
        4,
        1,
        64,
        False,
        False,
        256,
        "mla_fp8_page64_pdl",
    ),
    # LSE output: the reducer (split-K) and the one-split producer write it.
    "fp8_page64_lse": (
        torch.float8_e4m3fn,
        8,
        1,
        64,
        False,
        True,
        1024,
        "mla_fp8_page64_pdl",
    ),
    "fp8_page64_lse_one_split": (
        torch.float8_e4m3fn,
        128,
        1,
        64,
        False,
        True,
        1024,
        "mla_fp8_page64_pdl",
    ),
    "bf16_tail": (torch.bfloat16, 4, 2, 32, False, True, 1024, "mla_bf16_tail"),
    "fp8_tail": (torch.float8_e4m3fn, 4, 2, 32, False, False, 1024, "mla_fp8_tail"),
    "splitkv": (torch.bfloat16, 4, 2, 32, False, False, 4096, "mla_bf16_splitkv"),
    "splitkv_page64_sink": (
        torch.bfloat16,
        3,
        2,
        64,
        True,
        False,
        2048,
        "mla_bf16_splitkv",
    ),
}


@pytest.mark.parametrize("case", sorted(_PADDED_ROW_CASES))
def test_trtllm_mla_blackwell_padded_rows_output_zero(
    case, blackwell_domains, monkeypatch
) -> None:
    """Rows of KV length <= 0 output exact zeros (LSE -inf); live rows are unaffected.

    The output is NaN-filled before every call, so a padded row the route leaves
    unwritten stays NaN.  Padded requests (KV length 0, or shorter than the query)
    are interleaved with live ones; a batch of padded requests only must complete.
    """
    _skip_unless_blackwell()
    dtype, batch, q_len, page_size, sink, return_lse, max_len, domain = (
        _PADDED_ROW_CASES[case]
    )
    _list_splitkv_device(domain, monkeypatch)
    kv_lens = _padded_kv_lens(batch, q_len, max_len)
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, page_size, dtype
    )
    sinks = torch.randn(128, device="cuda", dtype=torch.float32) * 2.0 if sink else None
    rows = _query_rows(kv_lens, [q_len] * batch)
    if domain == "mla_fp8_page64_pdl":
        from flashinfer.mla.cake_trtllm_mla_blackwell import plan_num_split

        num_sms = get_device_sm_count(torch.device("cuda"))
        one_split = plan_num_split(len(rows), max_len, num_sms) == 1
        if case.endswith("one_split"):
            assert one_split
        elif num_sms >= 148:
            # The other sizes split on full B200 / GB300 parts; a part with
            # fewer SMs (a MIG slice) may plan one split for them.
            assert not one_split
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size, sinks
    )
    out = torch.empty(batch, q_len, 128, 512, dtype=torch.bfloat16, device="cuda")
    first = None
    for _ in range(3):
        out.fill_(float("nan"))
        result = _blackwell_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_len,
            out=out,
            sinks=sinks,
            return_lse=return_lse,
        )
        _assert_rows(out, reference, rows, dtype)
        if return_lse:
            lse = result[1].reshape(len(rows), 128)
            dead = torch.tensor([length <= 0 for _, length in rows], device="cuda")
            assert bool((lse[dead] == float("-inf")).all())
            assert bool(lse[~dead].isfinite().all())
        if first is None:
            first = out.clone()
        else:
            assert torch.equal(out, first)
    assert blackwell_domains and set(blackwell_domains) == {domain}

    seq_lens.zero_()
    out.fill_(float("nan"))
    result = _blackwell_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        max_len,
        out=out,
        sinks=sinks,
        return_lse=return_lse,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))
    if return_lse:
        assert bool((result[1] == float("-inf")).all())


# name: (dtype, batch, q_len, page_size, max_len)
_GRAPH_CASES = {
    "vquarter": (torch.bfloat16, 4, 4, 32, 1024),
    "vhalf": (torch.bfloat16, 8, 4, 32, 1024),
    "vhalf_page64": (torch.bfloat16, 8, 4, 64, 1024),
    "clc_one_wave": (torch.bfloat16, 16, 4, 32, 1024),
    "clc_ordered": (torch.bfloat16, 64, 4, 32, 1024),
    "fp8_page64": (torch.float8_e4m3fn, 4, 2, 64, 1024),
    "fp8_page64_one_split": (torch.float8_e4m3fn, 64, 1, 64, 1024),
    "splitkv": (torch.bfloat16, 4, 2, 32, 4096),
    "splitkv_page64": (torch.bfloat16, 3, 1, 64, 2048),
}


@pytest.mark.parametrize("case", sorted(_GRAPH_CASES))
def test_trtllm_mla_blackwell_graph_replay_reads_current_rows(
    case, blackwell_domains, monkeypatch
) -> None:
    """A graph captured once follows later in-place changes of seq_lens and block_tables.

    Rows are derived on the device from the caller's tensors at every replay, so the
    replay equals an eager call on fresh tensors holding the same values.
    """
    _skip_unless_blackwell()
    dtype, batch, q_len, page_size, max_len = _GRAPH_CASES[case]
    if case.startswith("splitkv"):
        _list_splitkv_device("mla_bf16_splitkv", monkeypatch)
    width = max_len // page_size
    kv_lens = _padded_kv_lens(batch, q_len, max_len)
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, page_size, dtype, width=width
    )
    out = torch.full(
        (batch, q_len, 128, 512), float("nan"), dtype=torch.bfloat16, device="cuda"
    )

    def decode(lens, table, output):
        return _blackwell_decode(query, kv_cache, table, lens, max_len, out=output)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(seq_lens, block_tables, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(seq_lens, block_tables, out)

    generator = torch.Generator(device="cuda").manual_seed(1)
    for values in (
        [0] * batch,
        [max_len] * batch,
        _padded_kv_lens(batch, q_len, max_len)[::-1],
    ):
        seq_lens.copy_(torch.tensor(values, dtype=torch.int32))
        permutation = torch.randperm(batch * width, device="cuda", generator=generator)
        block_tables.copy_(block_tables.reshape(-1)[permutation].reshape(batch, width))
        out.fill_(float("nan"))
        graph.replay()
        eager = decode(seq_lens.clone(), block_tables.clone(), None)
        torch.cuda.synchronize()
        assert torch.equal(out, eager), values
        rows = _query_rows(values, [q_len] * batch)
        reference = _blackwell_rows_reference(
            query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
        )
        _assert_rows(out, reference, rows, dtype)
    if case.startswith("splitkv"):
        assert set(blackwell_domains) == {"mla_bf16_splitkv"}
    if case.startswith("fp8_page64"):
        assert set(blackwell_domains) == {"mla_fp8_page64_pdl"}


# name: (batch, q_len, page_size, ragged q_lens with the same total, domain)
_RAGGED_CASES = {
    "vquarter": (4, 4, 32, [1, 7, 4, 4], "mla_bf16_vquarter"),
    "vhalf": (8, 4, 32, [1, 7, 4, 4, 2, 6, 3, 5], "mla_bf16_vhalf"),
    "vquarter_page64": (4, 2, 64, [1, 3, 2, 2], "mla_bf16_vquarter"),
    "vhalf_page64": (8, 4, 64, [1, 7, 4, 4, 2, 6, 3, 5], "mla_bf16_vhalf"),
}


@pytest.mark.parametrize("case", sorted(_RAGGED_CASES))
def test_trtllm_mla_blackwell_ragged_value_split(case, blackwell_domains) -> None:
    """Ragged value-split calls read the host row plan; uniform ragged equals the fixed call."""
    _skip_unless_blackwell()
    batch, q_len, page_size, ragged_q_lens, domain = _RAGGED_CASES[case]
    assert sum(ragged_q_lens) == batch * q_len
    max_len = 1024
    kv_lens = _padded_kv_lens(batch, max(ragged_q_lens), max_len)
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, page_size, torch.bfloat16
    )
    compact = query.reshape(batch * q_len, 128, 576)

    def cumulative(q_lens):
        offsets = [0]
        for length in q_lens:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device="cuda")

    fixed = _blackwell_decode(query, kv_cache, block_tables, seq_lens, max_len)
    uniform = _blackwell_decode(
        compact,
        kv_cache,
        block_tables,
        seq_lens,
        max_len,
        cum_seq_lens_q=cumulative([q_len] * batch),
        max_q_len=q_len,
    )
    torch.cuda.synchronize()
    assert torch.equal(uniform.reshape(fixed.shape), fixed)

    out = torch.full(
        (batch * q_len, 128, 512), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    _blackwell_decode(
        compact,
        kv_cache,
        block_tables,
        seq_lens,
        max_len,
        out=out,
        cum_seq_lens_q=cumulative(ragged_q_lens),
        max_q_len=max(ragged_q_lens),
    )
    rows = _query_rows(kv_lens, ragged_q_lens)
    reference = _blackwell_rows_reference(
        compact, kv_cache, block_tables, rows, page_size
    )
    _assert_rows(out, reference, rows, torch.bfloat16)
    assert set(blackwell_domains) == {domain}


# name: (dtype, batch, q_len, page_size, domain)
_TABLE_WIDTH_CASES = {
    "vquarter": (torch.bfloat16, 2, 2, 32, "mla_bf16_vquarter"),
    "vhalf": (torch.bfloat16, 8, 4, 32, "mla_bf16_vhalf"),
    "clc_ordered": (torch.bfloat16, 64, 2, 32, "mla_bf16_clc"),
    "vquarter_page64": (torch.bfloat16, 2, 1, 64, "mla_bf16_vquarter"),
    "fp8_page64": (torch.float8_e4m3fn, 4, 2, 64, "mla_fp8_page64_pdl"),
}


# name: (dtype, batch, q_len, page_size, domain); widths whose capacity exceeds
# 1024 tokens, so the split-KV domain serves them where _SPLITKV_RANGE lists the
# device.
_SPLITKV_TABLE_WIDTH_CASES = {
    "splitkv": (torch.bfloat16, 2, 2, 32, "mla_bf16_splitkv"),
    "splitkv_page64": (torch.bfloat16, 2, 1, 64, "mla_bf16_splitkv"),
}


@pytest.mark.parametrize("table_end", ["poison", "allocation_end"])
@pytest.mark.parametrize("width", [17, 19, 33, 34, 35, 37])
@pytest.mark.parametrize("case", sorted(_SPLITKV_TABLE_WIDTH_CASES))
def test_trtllm_mla_blackwell_splitkv_reads_no_page_id_past_the_table(
    case, width, table_end, blackwell_domains, monkeypatch
) -> None:
    """The split-KV program reads page ids only inside the caller's table (see
    ``test_trtllm_mla_blackwell_reads_no_page_id_past_the_table``)."""
    _skip_unless_blackwell()
    dtype, batch, q_len, page_size, domain = _SPLITKV_TABLE_WIDTH_CASES[case]
    if (page_size == 64) != (width < 32):
        pytest.skip("page 32 takes widths 33-37, page 64 widths 17 and 19")
    _list_splitkv_device(domain, monkeypatch)
    _check_page_ids_inside_the_table(
        dtype, batch, q_len, page_size, domain, width, table_end, blackwell_domains
    )


@pytest.mark.parametrize("table_end", ["poison", "allocation_end"])
@pytest.mark.parametrize("case", sorted(_SPLITKV_TABLE_WIDTH_CASES))
def test_trtllm_mla_blackwell_splitkv_clamps_rows_to_the_table(
    case, table_end, blackwell_domains, monkeypatch
) -> None:
    """KV lengths past the caller's table width read no page id past the table.

    The first and the last request are 300 keys longer than their table rows hold
    (the last one ends at the table's allocation, or before poison entries naming a
    NaN page). The split-KV program clamps every row's causal KV length to its table
    row, so the output is finite and equals attention over the keys the table row
    holds; the other rows are unchanged.
    """
    _skip_unless_blackwell()
    dtype, _, q_len, page_size, domain = _SPLITKV_TABLE_WIDTH_CASES[case]
    _list_splitkv_device(domain, monkeypatch)
    width = 33 if page_size == 32 else 17
    capacity = width * page_size
    kv_lens = [capacity + 300, capacity - 7, capacity + 300]
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens,
        q_len,
        page_size,
        dtype,
        width=width,
        trailing_poison=8 if table_end == "poison" else 0,
    )
    if table_end == "allocation_end":
        assert block_tables.untyped_storage().nbytes() == block_tables.nbytes
    output = _blackwell_decode(query, kv_cache, block_tables, seq_lens, max(kv_lens))
    torch.cuda.synchronize()
    assert bool(output.isfinite().all())
    rows = [
        (b, min(kv_len - q_len + index + 1, capacity))
        for b, kv_len in enumerate(kv_lens)
        for index in range(q_len)
    ]
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    _assert_rows(output, reference, rows, dtype)
    assert set(blackwell_domains) == {domain}


@pytest.mark.parametrize("kv_len", [0, 500])
def test_trtllm_mla_blackwell_splitkv_empty_table_outputs_zero(
    kv_len, blackwell_domains, monkeypatch
) -> None:
    """A table of width 0 holds no key: every row outputs exact zeros and the
    program reads no page id, whatever ``seq_lens`` claims."""
    _skip_unless_blackwell()
    _list_splitkv_device("mla_bf16_splitkv", monkeypatch)
    batch, q_len, page_size = 2, 1, 32
    query, kv_cache, _, _ = _blackwell_inputs(
        [1] * batch, q_len, page_size, torch.bfloat16
    )
    block_tables = torch.empty((batch, 0), dtype=torch.int32, device="cuda")
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device="cuda")
    out = torch.full(
        (batch, q_len, 128, 512), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    _blackwell_decode(query, kv_cache, block_tables, seq_lens, 2048, out=out)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))
    assert set(blackwell_domains) == {"mla_bf16_splitkv"}


@pytest.mark.parametrize("table_end", ["poison", "allocation_end"])
@pytest.mark.parametrize("width", [1, 2, 3, 5])
@pytest.mark.parametrize("case", sorted(_TABLE_WIDTH_CASES))
def test_trtllm_mla_blackwell_reads_no_page_id_past_the_table(
    case, width, table_end, blackwell_domains
) -> None:
    """Page ids are read only inside the caller's table, whatever its width.

    A 128-token KV tile spans four page-32 (two page-64) entries, so a table width
    that is not a multiple of that leaves the last row's last tile short.  Every
    request's last page is its last entry.

    ``poison``: the entries right after the table name a NaN page, so a read of one
    that feeds a V tile turns the last request's rows non-finite.  The CLC route
    loads only the last tile's K pages there, whose columns softmax masks, so its
    over-read leaves the output finite.  ``allocation_end``: the table is its own
    allocation of exactly ``batch * width`` entries, so any read past it is out of
    bounds.  Run that variant under ``compute-sanitizer --tool memcheck`` with
    ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` (the caching allocator rounds the table's
    block up) to catch every route's over-read, the CLC one included.
    """
    _skip_unless_blackwell()
    dtype, batch, q_len, page_size, domain = _TABLE_WIDTH_CASES[case]
    _check_page_ids_inside_the_table(
        dtype, batch, q_len, page_size, domain, width, table_end, blackwell_domains
    )


def _check_page_ids_inside_the_table(
    dtype, batch, q_len, page_size, domain, width, table_end, blackwell_domains
) -> None:
    if page_size == 64 and width % 2 == 0:
        pytest.skip("page 64 reads two entries per tile: odd widths only")
    capacity = width * page_size
    kv_lens = [capacity - (7 * b) % min(capacity, 32) for b in range(batch)]
    kv_lens[-1] = capacity
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens,
        q_len,
        page_size,
        dtype,
        width=width,
        trailing_poison=8 if table_end == "poison" else 0,
    )
    if table_end == "allocation_end":
        assert block_tables.untyped_storage().nbytes() == block_tables.nbytes
    output = _blackwell_decode(query, kv_cache, block_tables, seq_lens, capacity)
    torch.cuda.synchronize()
    assert bool(output.isfinite().all())
    rows = _query_rows(kv_lens, [q_len] * batch)
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    _assert_rows(output, reference, rows, dtype)
    assert set(blackwell_domains) == {domain}


@pytest.mark.parametrize(
    "rows_offset", ["resident", "resident+1", "300", "513", "1024", "1025"]
)
@pytest.mark.parametrize("q_len", [1, 2])
def test_trtllm_mla_blackwell_clc_device_work_order(rows_offset, q_len) -> None:
    """The generic CLC launch takes the longest-first order written on the device.

    Launches of more rows than one wave of clusters and at most 1024 rows are
    preceded by the order program; its order equals the host argsort of the rows'
    KV tiles (padded rows count zero tiles, ties keep row order), and its second
    half holds the ordered rows' KV lengths clamped at zero.
    """
    _skip_unless_blackwell()
    from flashinfer.mla.cake_trtllm_mla_blackwell import (
        _MAX_ORDERED_ROWS,
        _WORK_ORDER_BUFFERS,
        longest_first_order,
        row_metadata,
    )

    resident = _blackwell_resident()
    total = {
        "resident": resident,
        "resident+1": resident + 1,
        "300": 300,
        "513": 513,
        "1024": 1024,
        "1025": 1025,
    }[rows_offset]
    if total % q_len:
        pytest.skip("row count is not a multiple of q_len")
    batch = total // q_len
    choices = [0, 1, 128, 129, 256, 300, 512]
    kv_lens = [choices[(5 * b + b // 3) % len(choices)] for b in range(batch)]
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, 32, torch.bfloat16
    )
    output = _blackwell_decode(query, kv_cache, block_tables, seq_lens, 512)
    stream = int(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if resident < total <= _MAX_ORDERED_ROWS:
        _, row_seq_lens = row_metadata(
            seq_lens, None, batch_size=batch, q_len=q_len, total_q=total
        )
        clamped = row_seq_lens.clamp_min(0)
        expected = longest_first_order(clamped)
        work_order = _WORK_ORDER_BUFFERS[(torch.cuda.current_device(), stream)]
        assert torch.equal(work_order[:total], expected)
        assert torch.equal(
            work_order[_MAX_ORDERED_ROWS : _MAX_ORDERED_ROWS + total],
            clamped[expected.long()].to(torch.int32),
        )
    rows = _query_rows(kv_lens, [q_len] * batch)
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, 32
    )
    _assert_rows(output, reference, rows, torch.bfloat16)


@pytest.mark.parametrize("padded", [False, True])
def test_trtllm_mla_blackwell_exact007826_geometry(padded, blackwell_domains) -> None:
    """The 512x16 geometry keeps its exact route unless a request is padded.

    On sm_100a and the 152-SM sm_103a the exact route launches the device-row CLC
    binding; a padded request (KV length below the query length) leaves the host
    row-plan programs for the generic CLC route.
    """
    _skip_unless_blackwell()
    batch, q_len, width, page_size = 512, 16, 32, 32
    kv_lens = [1024] * batch
    if padded:
        kv_lens[3] = 0
        kv_lens[100] = 7
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, page_size, torch.bfloat16, width=width
    )
    output = _blackwell_decode(query, kv_cache, block_tables, seq_lens, 1024)
    rows = _query_rows(kv_lens, [q_len] * batch)
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    _assert_rows(output, reference, rows, torch.bfloat16)
    expected = "mla_bf16_clc" if padded else "mla_bf16_clc_exact007826"
    assert blackwell_domains == [expected]


def test_trtllm_mla_blackwell_kv_lens_identity_outlives_its_tensor() -> None:
    """A KV-length read cached for a freed tensor never serves a new tensor.

    The cache entry holds the tensor it read, so a later ``seq_lens`` with other
    values cannot take its address and version counter; the padded request then
    leaves the exact-geometry route.
    """
    _skip_unless_blackwell()
    from flashinfer.mla.cake_trtllm_mla_blackwell import _host_kv_lens

    live = [1024] * 512
    padded = [0] + live[1:]
    first = torch.tensor(live, dtype=torch.int32, device="cuda")
    assert _host_kv_lens(first) == tuple(live)
    address = first.data_ptr()
    del first
    second = torch.tensor(padded, dtype=torch.int32, device="cuda")
    assert second.data_ptr() != address
    assert _host_kv_lens(second) == tuple(padded)


@pytest.mark.parametrize("layout", ["rows", "tail"])
def test_trtllm_mla_blackwell_row_plan_identity_outlives_its_tensor(layout) -> None:
    """A row plan cached for a freed ``seq_lens`` never serves a new tensor.

    The plan holds the tensors it was derived from, so a later ``seq_lens`` with
    other values cannot take its address, version counter, shape and strides.
    Each ``seq_lens`` is a view of its own 64 MiB block: the plan's small device
    temporaries never reuse that block, so without the held source the second
    block lands at the first one's address and the stale plan is returned.
    """
    _skip_unless_blackwell()
    from flashinfer.mla.cake_trtllm_mla_blackwell import _row_plan, row_metadata

    cum_seq_lens_q = torch.tensor([0, 1, 3, 4, 6], dtype=torch.int32, device="cuda")
    block_tables = torch.zeros(4, 4, dtype=torch.int32, device="cuda")
    plan_args = dict(
        cum_seq_lens_q=cum_seq_lens_q,
        block_tables=block_tables,
        batch_size=4,
        q_len=2,
        total_q=6,
        page_size=32,
        topk=0,
    )

    def seq_lens_in_own_block(values):
        block = torch.empty(16 << 20, dtype=torch.int32, device="cuda")
        block[: len(values)].copy_(torch.tensor(values, dtype=torch.int32))
        return block[: len(values)]

    first = seq_lens_in_own_block([10, 20, 30, 40])
    _row_plan(layout, seq_lens=first, **plan_args)
    address = first.data_ptr()
    del first
    second = seq_lens_in_own_block([5, 1, 70, 0])
    assert second.data_ptr() != address
    plan = _row_plan(layout, seq_lens=second, **plan_args)
    row_batches, row_seq_lens = row_metadata(
        second, cum_seq_lens_q, batch_size=4, q_len=2, total_q=6
    )
    if layout == "tail":
        row_seq_lens = row_seq_lens.clamp_min(0)
    assert torch.equal(plan["row_batches"], row_batches)
    assert torch.equal(plan["row_seq_lens"], row_seq_lens)


_PDL_PREDECESSOR_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Lets its programmatic dependents launch at once, then writes dst only after spinning,
// so a dependent that reads dst before its griddepcontrol.wait sees the old values.
__global__ void write_after_spin(int* dst, const int* src, int n, unsigned long long spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < spin_ns);
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

void write_after_spin_launch(torch::Tensor dst, torch::Tensor src, int64_t spin_ns) {
  write_after_spin<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      dst.data_ptr<int>(), src.data_ptr<int>(), static_cast<int>(dst.numel()),
      static_cast<unsigned long long>(spin_ns));
}
"""


@pytest.mark.parametrize(
    "dtype,batch,q_len,page_size,max_len",
    [
        (torch.bfloat16, 64, 4, 32, 1024),
        (torch.bfloat16, 16, 4, 32, 1024),
        (torch.float8_e4m3fn, 4, 2, 64, 1024),
        (torch.float8_e4m3fn, 64, 1, 64, 1024),
        (torch.bfloat16, 4, 2, 32, 2048),
        (torch.bfloat16, 3, 1, 64, 4096),
    ],
    ids=[
        "clc_ordered",
        "clc_one_wave",
        "fp8_page64",
        "fp8_page64_one_split",
        "splitkv",
        "splitkv_page64",
    ],
)
def test_trtllm_mla_blackwell_reads_rows_after_the_dependency_wait(
    dtype, batch, q_len, page_size, max_len, blackwell_domains, monkeypatch
) -> None:
    """The PDL programs read seq_lens only after their programmatic dependency wait.

    A predecessor that triggers its dependents at entry writes seq_lens 200 us later;
    a program reading it earlier sees all requests empty and outputs zeros.  In the
    split-KV cases (the mainloop and its PDL reducer) only the final output is
    checked, so a reducer reading partials early is caught only when it changes
    that output.
    """
    _skip_unless_blackwell()
    if max_len > 1024:
        _list_splitkv_device("mla_bf16_splitkv", monkeypatch)
    from torch.utils.cpp_extension import load_inline

    predecessor = load_inline(
        name="test_trtllm_mla_blackwell_pdl_predecessor",
        cpp_sources=[
            "void write_after_spin_launch(torch::Tensor dst, torch::Tensor src, int64_t spin_ns);"
        ],
        cuda_sources=[_PDL_PREDECESSOR_SOURCE],
        functions=["write_after_spin_launch"],
        verbose=False,
    )
    kv_lens = _padded_kv_lens(batch, q_len, max_len)
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, page_size, dtype
    )
    _blackwell_decode(query, kv_cache, block_tables, seq_lens.clone(), max_len)
    torch.cuda.synchronize()
    rows = _query_rows(kv_lens, [q_len] * batch)
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    for _ in range(4):
        stale = torch.zeros_like(seq_lens)
        predecessor.write_after_spin_launch(stale, seq_lens, 200_000)
        output = _blackwell_decode(query, kv_cache, block_tables, stale, max_len)
        _assert_rows(output, reference, rows, dtype)
    if max_len > 1024:
        assert set(blackwell_domains) == {"mla_bf16_splitkv"}


def test_trtllm_mla_blackwell_tail_guards_lengths_beyond_4096() -> None:
    """A tail row longer than 4096 keys outputs NaN (O and LSE); other rows are unchanged.

    The host knows only max_seq_len <= 4096; a captured graph replayed after
    seq_lens grew past it must flag the row instead of reading out of range.
    """
    _skip_unless_blackwell()
    page_size, width, q_len = 32, 130, 1
    kv_lens = [4096, 100, 0, 2000]
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, page_size, torch.bfloat16, width=width
    )
    out = torch.empty(4, q_len, 128, 512, dtype=torch.bfloat16, device="cuda")

    def decode(lens, output):
        return _blackwell_decode(
            query, kv_cache, block_tables, lens, 4096, out=output, return_lse=True
        )

    # Warm up on other tensors: the tail's row plan of an identity seen before
    # capture would be reused by the graph instead of recomputed at each replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(seq_lens.clone(), out.clone())
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, lse = decode(seq_lens, out)

    graph.replay()
    torch.cuda.synchronize()
    valid_out, valid_lse = out.clone(), lse.clone()
    rows = _query_rows(kv_lens, [q_len] * 4)
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    _assert_rows(out, reference, rows, torch.bfloat16)

    seq_lens[1] = 4097
    graph.replay()
    torch.cuda.synchronize()
    assert bool(out[1].isnan().all()) and bool(lse.reshape(4, 128)[1].isnan().all())
    for batch in (0, 2, 3):
        assert torch.equal(out[batch], valid_out[batch])
        assert torch.equal(lse.reshape(4, 128)[batch], valid_lse.reshape(4, 128)[batch])

    seq_lens[1] = 100
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, valid_out) and torch.equal(lse, valid_lse)


_TAIL_REQUEST = {
    "dtype": torch.bfloat16,
    "num_heads": 128,
    "qk_dim": 576,
    "value_dim": 512,
    "qk_nope_head_dim": 128,
    "page_size": 32,
    "topk": 0,
    "uses_shared_paged_kv_idx": True,
    "ragged_query": False,
    "enable_sink": False,
    "skip_softmax": False,
    "wants_lse": False,
}


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"wants_lse": True}, "BF16 with LSE output"),
        ({"wants_lse": True, "page_size": 64}, "BF16 with LSE output"),
        (
            {"num_heads": 32, "qk_dim": 320, "value_dim": 256},
            "32 heads of dimensions (128, 256, 64)",
        ),
        ({"topk": 64}, "sparse top-k"),
        ({"uses_shared_paged_kv_idx": False}, "separate K and V page indices"),
        ({"dtype": torch.float8_e4m3fn}, "FP8 with page size 32"),
        (
            {"dtype": torch.float8_e4m3fn, "wants_lse": True},
            "FP8 with page size 32 and LSE output",
        ),
        (
            {"dtype": torch.float8_e4m3fn, "page_size": 64, "enable_sink": True},
            "FP8 with sinks",
        ),
        (
            {"dtype": torch.float8_e4m3fn, "page_size": 64, "ragged_query": True},
            "FP8 with a compact variable-length query",
        ),
        (
            {"dtype": torch.float8_e4m3fn, "page_size": 64, "skip_softmax": True},
            "FP8 with skip-softmax",
        ),
        (
            {
                "dtype": torch.float8_e4m3fn,
                "page_size": 64,
                "wants_lse": True,
                "enable_sink": True,
            },
            "FP8 with sinks",
        ),
    ],
)
def test_trtllm_mla_blackwell_generic_tail_reason(overrides: dict, reason: str) -> None:
    """Every request ``select_domain`` sends to a generic tail program names the
    feature that took it there. No GPU needed."""
    from flashinfer.mla.cake_trtllm_mla_blackwell import (
        DOMAIN_BF16_TAIL,
        DOMAIN_FP8_TAIL,
        generic_tail_reason,
        select_domain,
    )

    request = dict(_TAIL_REQUEST, **overrides)
    domain = select_domain(
        **request,
        bmm2_scale=1.0,
        total_q=8,
        max_seq_len=1024,
        batch_size=8,
        q_len=1,
        table_width=32,
        kv_lens=None,
        num_sms=148,
        arch="sm_100a",
    )
    assert domain in (DOMAIN_BF16_TAIL, DOMAIN_FP8_TAIL)
    assert generic_tail_reason(**request) == reason


def test_trtllm_mla_blackwell_tail_warns_once_per_reason(monkeypatch) -> None:
    """An explicit call that takes a generic tail program logs one warning per
    reason and process; repeated calls log nothing and return the same bits,
    and calls served by a tuned program log nothing."""
    _skip_unless_blackwell()
    from flashinfer.mla import cake_trtllm_mla_blackwell as module

    warnings: list[tuple] = []

    class _Recorder:
        def warning(self, message, *args):
            warnings.append(args)

    monkeypatch.setattr(module, "_TAIL_WARNED", set())
    monkeypatch.setattr(module, "logger", _Recorder())
    kv_lens = [1024, 0, 700, 1024, 513, 1, 1024, 900]
    rows = _query_rows(kv_lens, [1] * len(kv_lens))
    for dtype, page_size, lse in (
        (torch.bfloat16, 32, True),
        (torch.float8_e4m3fn, 32, False),
    ):
        query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
            kv_lens, 1, page_size, dtype
        )
        outputs = [
            _blackwell_decode(
                query, kv_cache, block_tables, seq_lens, 1024, return_lse=lse
            )
            for _ in range(3)
        ]
        torch.cuda.synchronize()
        first = outputs[0][0] if lse else outputs[0]
        for repeated in outputs[1:]:
            if lse:
                assert torch.equal(repeated[0], first)
                assert torch.equal(repeated[1], outputs[0][1])
            else:
                assert torch.equal(repeated, first)
        reference = _blackwell_rows_reference(
            query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
        )
        _assert_rows(first, reference, rows, dtype)
    assert warnings == [("BF16 with LSE output",), ("FP8 with page size 32",)]
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, 1, 32, torch.bfloat16
    )
    _blackwell_decode(query, kv_cache, block_tables, seq_lens, 1024)
    torch.cuda.synchronize()
    assert len(warnings) == 2


def _blackwell_lse_reference(
    q_rows: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    rows: list[tuple[int, int]],
    page_size: int,
) -> torch.Tensor:
    """FP64 base-2 LSE ``[rows, heads]`` of the bmm1-scaled logits; ``-inf`` for
    a row of KV length <= 0."""
    device = q_rows.device
    queries = q_rows.double()
    kv_flat = kv_cache.reshape(-1, 576).double()
    tokens = torch.arange(page_size, device=device)
    lse = torch.full(
        (q_rows.shape[0], q_rows.shape[1]),
        float("-inf"),
        dtype=torch.float64,
        device=device,
    )
    for row, (batch, length) in enumerate(rows):
        if length <= 0:
            continue
        pages = block_tables[batch, : -(-length // page_size)].long()
        kv = kv_flat[
            (pages[:, None] * page_size + tokens[None, :]).reshape(-1)[:length]
        ]
        logits = queries[row] @ kv.T * _BLACKWELL_BMM1
        lse[row] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
    return lse


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_trtllm_mla_blackwell_lse_follows_return_lse_base(dtype) -> None:
    """Explicit ``backend="cake"`` on the Blackwell route returns base-2 LSE by
    default and with ``"base2"``, natural-log LSE with ``"basee"`` (also in a
    supplied buffer), equal to the FP64 reference on live rows and ``-inf`` on
    padded rows; the output does not depend on the base."""
    _skip_unless_blackwell()
    page_size = 32 if dtype == torch.bfloat16 else 64
    kv_lens = [1024, 0, 700, 1024, 513, 1, 300, 900]
    batch = len(kv_lens)
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, 1, page_size, dtype
    )
    rows = _query_rows(kv_lens, [1] * batch)
    reference = _blackwell_lse_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def decode(**kwargs):
        return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=1024,
            bmm1_scale=_BLACKWELL_BMM1,
            bmm2_scale=1.0,
            backend="cake",
            **kwargs,
        )

    out_default, lse_default = decode(return_lse=True)
    out_base2, lse_base2 = decode(return_lse=True, return_lse_base="base2")
    out_basee, lse_basee = decode(return_lse=True, return_lse_base="basee")
    supplied = torch.full(
        (batch, 128), float("nan"), dtype=torch.float32, device="cuda"
    )
    out_supplied = decode(lse=supplied, return_lse_base="basee")
    torch.cuda.synchronize()
    for output in (out_base2, out_basee, out_supplied):
        assert torch.equal(output, out_default)
    assert torch.equal(lse_base2, lse_default)
    lse_base2 = lse_base2.reshape(batch, 128)
    assert torch.equal(lse_basee.reshape(batch, 128), supplied)
    torch.testing.assert_close(
        supplied, lse_base2 * math.log(2.0), rtol=1e-6, atol=1e-6
    )
    live = torch.tensor([length > 0 for _, length in rows], device="cuda")
    assert bool(torch.isneginf(lse_base2[~live]).all())
    assert bool(torch.isneginf(supplied[~live]).all())
    torch.testing.assert_close(
        lse_base2[live].double(), reference[live], rtol=1e-3, atol=5e-3
    )
    torch.testing.assert_close(
        supplied[live].double(), reference[live] * math.log(2.0), rtol=1e-3, atol=5e-3
    )


@pytest.mark.parametrize(
    ("batch", "q_len", "max_len", "padded"),
    [
        (1, 1, 1024, False),  # split-K: the reducer writes the LSE
        (1, 1, 8192, False),  # beyond the generic tail's 4096 keys
        (8, 1, 1024, False),
        (8, 1, 4096, True),
        (32, 1, 1024, True),
        (128, 1, 1024, False),  # one split: the producer writes the LSE
        (128, 1, 4096, True),
        (20, 2, 1024, True),
    ],
)
def test_trtllm_mla_blackwell_fp8_page64_lse(
    batch, q_len, max_len, padded, blackwell_domains
) -> None:
    """An FP8 page-64 request with LSE output runs the FP8 page-64 program (one
    split and split-K): the output equals, bit for bit, the same request without
    LSE; the base-2 LSE, written into the caller's buffer of either accepted
    shape, matches the FP64 reference on live rows and is ``-inf`` on padded
    rows."""
    _skip_unless_blackwell()
    from flashinfer.mla.cake_trtllm_mla_blackwell import (
        _device_facts,
        plan_num_split,
    )

    page_size = 64
    if padded:
        kv_lens = _padded_kv_lens(batch, q_len, max_len)
    else:
        generator = random.Random(batch * 7 + max_len)
        kv_lens = [generator.randint(max_len // 2, max_len) for _ in range(batch)]
        kv_lens[-1] = max_len
    query, kv_cache, block_tables, seq_lens = _blackwell_inputs(
        kv_lens, q_len, page_size, torch.float8_e4m3fn
    )
    rows = _query_rows(kv_lens, [q_len] * batch)
    total_q = batch * q_len
    _, num_sms = _device_facts(torch.cuda.current_device())
    splits = plan_num_split(total_q, max_len, num_sms)

    plain = _blackwell_decode(query, kv_cache, block_tables, seq_lens, max_len)
    out, lse = _blackwell_decode(
        query, kv_cache, block_tables, seq_lens, max_len, return_lse=True
    )
    nested = torch.full(
        (batch, q_len, 128), float("nan"), dtype=torch.float32, device="cuda"
    )
    out_nested = _blackwell_decode(
        query, kv_cache, block_tables, seq_lens, max_len, lse=nested
    )
    torch.cuda.synchronize()
    assert blackwell_domains == ["mla_fp8_page64_pdl"] * 3, (blackwell_domains, splits)
    assert torch.equal(out, plain) and torch.equal(out_nested, plain)
    assert tuple(lse.shape) == (total_q, 128)
    assert torch.equal(nested.reshape(total_q, 128), lse)
    reference = _blackwell_rows_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    _assert_rows(out, reference, rows, torch.float8_e4m3fn)
    lse_reference = _blackwell_lse_reference(
        query.reshape(-1, 128, 576), kv_cache, block_tables, rows, page_size
    )
    live = torch.tensor([length > 0 for _, length in rows], device="cuda")
    assert bool(torch.isneginf(lse[~live]).all()), splits
    torch.testing.assert_close(
        lse[live].double(), lse_reference[live], rtol=1e-3, atol=5e-3
    )


@pytest.mark.parametrize(
    "layer_dimensions",
    supported_mla_layer_dimensions,
)
@pytest.mark.parametrize("batch_size", [1, 16, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("q_len_per_request", [1, 2])
def test_trtllm_batch_decode_mla_use_fp16_softmax(
    layer_dimensions: MLALayerDimensions,
    batch_size: int,
    dtype: torch.dtype,
    page_size: int,
    q_len_per_request: int,
):
    trtllm_batch_decode_mla(
        layer_dimensions=layer_dimensions,
        batch_size=batch_size,
        scale=1.0,
        dtype=dtype,
        page_size=page_size,
        q_len_per_request=q_len_per_request,
        dynamic_scale=False,
        enable_pdl=None,
        backend="trtllm-gen",
        MAX_SEQ_LEN=1024,
        skips_softmax=False,
        uses_shared_paged_kv_idx=True,
        use_fp16_softmax=True,
    )


@pytest.fixture
def functional_request():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA tensors")
    return dict(
        query=torch.empty(2, 1, 16, 576, dtype=torch.bfloat16, device="cuda"),
        kv_cache=torch.empty(4, 32, 576, dtype=torch.bfloat16, device="cuda"),
        workspace_buffer=torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda"),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.zeros(2, 4, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([17, 31], dtype=torch.int32, device="cuda"),
        max_seq_len=32,
        out=torch.empty(2, 1, 16, 512, dtype=torch.bfloat16, device="cuda"),
        bmm1_scale=0.25,
        bmm2_scale=0.75,
        enable_pdl=False,
        backend="auto",
    )


@pytest.fixture
def functional_dispatch(monkeypatch):
    state = SimpleNamespace(selected="trtllm-gen", cute_reason=None, runs=[], tuning=[])

    class Runner:
        @classmethod
        def from_functional(cls, **kwargs):
            return cls(**kwargs)

        def __init__(self, **kwargs):
            self.options = kwargs

        def __call__(self, *, inputs, tactic, **kwargs):
            state.runs.append((self, inputs, kwargs))
            # Real runners write the supplied output/LSE buffers; forwarding
            # alone would miss a lost view or a discarded caller allocation.
            inputs[3].fill_(self.value)
            if self.options["return_lse"]:
                self.options["lse"].fill_(self.value + 1)
                return inputs[3], self.options["lse"]
            return inputs[3]

    class TrtRunner(Runner):
        name, value = "trtllm-gen", 11

    class CuteRunner(Runner):
        name, value = "cute-dsl", 21

    def choose_one(self, op, runners, config, inputs):
        state.tuning.append(runners)
        return next(r for r in runners if r.name == state.selected), 7

    monkeypatch.setattr(core, "_BatchMLAPagedAttentionTrtllmGenBackend", TrtRunner)

    class ModularRunner(CuteRunner):
        value = 31

    monkeypatch.setattr(
        core, "_BatchMLAPagedAttentionCuteDslModularBackend", ModularRunner
    )
    monkeypatch.setattr(
        core, "_BatchMLAPagedAttentionCuteDslMonolithicBackend", CuteRunner
    )
    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(core, "get_device_sm_count", lambda device: 148)
    monkeypatch.setattr(
        core, "_cute_dsl_incompatibility_reason", lambda *a, **kw: state.cute_reason
    )
    return state


def _dispatch_counter():
    return torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(2, 16, 148),
        dtype=torch.uint8,
        device="cuda",
    )


def _dispatch_ragged(case):
    case.update(
        query=case["query"].view(2, 16, 576),
        out=case["out"].view(2, 16, 512),
        cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        max_q_len=1,
    )


@pytest.mark.parametrize(
    "requested,selected,expected",
    [
        ("auto", "trtllm-gen", ["trtllm-gen", "cute-dsl"]),
        ("auto", "cute-dsl", ["trtllm-gen", "cute-dsl"]),
        ("trtllm-gen", "trtllm-gen", ["trtllm-gen"]),
        ("cute-dsl", "cute-dsl", ["cute-dsl"]),
    ],
)
def test_functional_dispatch_tuning_preserves_outputs(
    functional_request, functional_dispatch, requested, selected, expected
):
    case = functional_request
    functional_dispatch.selected = selected
    lse = torch.empty(2, 1, 16, dtype=torch.float32, device="cuda")
    counter = _dispatch_counter() if "trtllm-gen" in expected else None
    case.update(
        backend=requested,
        return_lse=True,
        lse=lse,
        return_lse_base="basee",
        multi_ctas_kv_counter_buffer=counter,
    )

    with autotune(True):
        result = core.trtllm_batch_decode_with_kv_cache_mla(**case)

    assert result[0] is case["out"] and result[1] is lse
    assert len(functional_dispatch.tuning) == 1
    candidates = functional_dispatch.tuning[0]
    assert [r.name for r in candidates] == expected
    runner, _, kwargs = functional_dispatch.runs[0]
    assert runner.name == selected
    assert runner.options["return_lse_base"] == "basee"
    if selected == "trtllm-gen":
        assert kwargs["multi_ctas_kv_counter_buffer"] is counter
    else:
        assert "multi_ctas_kv_counter_buffer" not in kwargs
    torch.testing.assert_close(result[0], torch.full_like(result[0], runner.value))
    torch.testing.assert_close(lse, torch.full_like(lse, runner.value + 1))


@pytest.mark.parametrize("sparse", [False, True])
def test_functional_dispatch_explicit_trt_inference(
    functional_request, functional_dispatch, monkeypatch, sparse
):
    case = functional_request
    lse = torch.empty(2, 1, 16, dtype=torch.float32, device="cuda")
    counter = _dispatch_counter()
    case.update(
        backend="trtllm-gen",
        return_lse=True,
        lse=lse,
        multi_ctas_kv_counter_buffer=counter,
    )
    lengths = None
    scales = tuple(torch.tensor([v], device="cuda") for v in (0.25, 0.75))
    if sparse:
        lengths = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        case.update(
            query=torch.empty(2, 1, 16, 512, dtype=torch.float8_e4m3fn, device="cuda"),
            kv_cache=torch.empty(4, 32, 512, dtype=torch.float8_e4m3fn, device="cuda"),
            bmm1_scale=scales[0],
            bmm2_scale=scales[1],
            qk_rope_head_dim=0,
            sparse_mla_top_k=4,
            sparse_mla_top_k_lens=lengths,
            block_tables=torch.zeros(2, 1, 4, dtype=torch.int32, device="cuda"),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit TRT inference performed backend selection")

    monkeypatch.setattr(core, "_cute_dsl_incompatibility_reason", forbidden)
    monkeypatch.setattr(core, "_build_mla_decode_tuning_config", forbidden)
    monkeypatch.setattr(core.AutoTuner, "choose_one", forbidden)
    with autotune(False):
        result = core.trtllm_batch_decode_with_kv_cache_mla(**case)

    assert result[0] is case["out"] and result[1] is lse
    assert not functional_dispatch.tuning
    runner, inputs, kwargs = functional_dispatch.runs[0]
    assert runner.name == "trtllm-gen"
    assert kwargs["multi_ctas_kv_counter_buffer"] is counter
    if sparse:
        assert inputs[4] is lengths
        assert runner.options["bmm1_scale"] is scales[0]
        assert runner.options["bmm2_scale"] is scales[1]
        for scale, expected in zip(scales, (0.25, 0.75), strict=True):
            torch.testing.assert_close(scale, torch.full_like(scale, expected))
    torch.testing.assert_close(result[0], torch.full_like(result[0], runner.value))
    torch.testing.assert_close(lse, torch.full_like(lse, runner.value + 1))


def test_functional_dispatch_sm12_xqa_scales(
    monkeypatch, functional_request, functional_dispatch
):
    case = functional_request
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (12, 0))
    monkeypatch.setattr(core, "is_sm12x_supported", lambda device: True)
    first = torch.tensor([0.25], device="cuda")
    second = torch.tensor([0.75], device="cuda")
    case.update(bmm1_scale=first, bmm2_scale=second)
    received = []

    signature = inspect.signature(core.xqa_batch_decode_with_kv_cache_mla)

    def xqa(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        received.append(arguments)
        arguments["out"].fill_(31)
        return arguments["out"]

    monkeypatch.setattr(core, "xqa_batch_decode_with_kv_cache_mla", xqa)
    result = core.trtllm_batch_decode_with_kv_cache_mla(**case)
    assert result is case["out"]
    assert len(received) == 1 and not functional_dispatch.tuning
    assert received[0]["query"] is case["query"]
    assert received[0]["bmm1_scale"] is first and received[0]["bmm2_scale"] is second
    torch.testing.assert_close(result, torch.full_like(result, 31))


@pytest.mark.parametrize("rope_dim,packed_width", [(64, 656), (0, 528)])
def test_functional_dispatch_sm12_sparse_metadata(
    monkeypatch, functional_request, functional_dispatch, rope_dim, packed_width
):
    case = functional_request
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (12, 0))
    lse = torch.empty(2, 1, 16, dtype=torch.float32, device="cuda")
    case.update(
        query=torch.empty(
            2, 1, 16, 512 + rope_dim, dtype=torch.bfloat16, device="cuda"
        ),
        kv_cache=torch.empty(4, 32, packed_width, dtype=torch.uint8, device="cuda"),
        qk_rope_head_dim=rope_dim,
        block_tables=torch.zeros(2, 1, 4, dtype=torch.int32, device="cuda"),
        sparse_mla_top_k=4,
        seq_lens=None,
        bmm2_scale=1.0,
        kv_scale_format="arbitrary_fp32",
        lse=lse,
        return_lse=True,
        return_lse_base="basee",
    )
    received = []

    def sparse(**kwargs):
        received.append(kwargs)
        kwargs["out"].fill_(41)
        kwargs["lse"].fill_(42)
        return kwargs["out"], kwargs["lse"]

    # Keep the packed adapter's real validation and segment construction.
    monkeypatch.setattr(core, "_trtllm_batch_decode_sparse_mla_sm120", sparse)
    result = core.trtllm_batch_decode_with_kv_cache_mla(**case)
    assert result[0] is case["out"] and result[1] is lse
    assert len(received) == 1 and not functional_dispatch.tuning
    args = received[0]
    assert args["kv_cache"] is case["kv_cache"]
    assert args["sm_scale"] == 0.25
    assert args["lse_scale"] == pytest.approx(math.log(2))
    assert args["sparse_mla_segments"][0].indices is case["block_tables"]
    assert args["sparse_mla_segments"][0].lengths is None
    torch.testing.assert_close(result[0], torch.full_like(result[0], 41))
    torch.testing.assert_close(lse, torch.full_like(lse, 42))


def test_functional_dispatch_ragged_trt_without_host_reads(
    monkeypatch, functional_request, functional_dispatch
):
    case = functional_request
    _dispatch_ragged(case)
    counter = _dispatch_counter()
    case["multi_ctas_kv_counter_buffer"] = counter

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "ragged TRT must not tune, probe CuTe, or read CUDA metadata"
        )

    monkeypatch.setattr(core, "_cute_dsl_incompatibility_reason", forbidden)
    monkeypatch.setattr(core.AutoTuner, "choose_one", forbidden)
    with monkeypatch.context() as no_host_reads:
        no_host_reads.setattr(torch.Tensor, "cpu", forbidden)
        no_host_reads.setattr(torch.Tensor, "item", forbidden)
        result = core.trtllm_prefill_with_kv_cache_mla(**case)
    assert result is case["out"]
    runner, inputs, kwargs = functional_dispatch.runs[0]
    assert runner.name == "trtllm-gen"
    assert inputs[0] is case["query"]
    assert kwargs["cum_seq_lens_q"] is case["cum_seq_lens_q"]
    assert kwargs["max_q_len"] == 1
    assert kwargs["multi_ctas_kv_counter_buffer"] is counter


def test_functional_dispatch_ragged_lse_fallback(
    functional_request, functional_dispatch
):
    case = functional_request
    _dispatch_ragged(case)
    lse = torch.empty(2, 16, dtype=torch.float32, device="cuda")
    case.update(lse=lse, return_lse=True, return_lse_base="base2")
    result = core.trtllm_prefill_with_kv_cache_mla(**case)
    assert result[0] is case["out"] and result[1] is lse
    assert not functional_dispatch.tuning
    runner, inputs, kwargs = functional_dispatch.runs[0]
    assert runner.name == "cute-dsl"
    assert inputs[0] is case["query"] and inputs[3] is case["out"]
    assert kwargs["cum_seq_lens_q"] is case["cum_seq_lens_q"]
    assert kwargs["max_q_len"] == 1
    assert runner.options["softmax_scale"] == 0.25
    assert runner.options["return_lse_base"] == "base2"
    torch.testing.assert_close(result[0], torch.full_like(result[0], 21))
    torch.testing.assert_close(lse, torch.full_like(lse, 22))


def test_functional_dispatch_ragged_rejections(functional_request, functional_dispatch):
    case = functional_request
    _dispatch_ragged(case)
    case["return_lse"] = True
    functional_dispatch.cute_reason = "CuTe unavailable for this request"
    with pytest.raises(
        ValueError, match="auto: no backend supports this variable-Q"
    ) as exc:
        core.trtllm_prefill_with_kv_cache_mla(**case)
    assert "trtllm-gen MLA does not support return_lse/lse with cum_seq_lens_q" in str(
        exc.value
    )
    assert functional_dispatch.cute_reason in str(exc.value)
    assert not functional_dispatch.tuning


def test_functional_dispatch_cute_rejects_trt_counter(
    functional_request, functional_dispatch
):
    case = functional_request
    case.update(backend="cute-dsl", multi_ctas_kv_counter_buffer=_dispatch_counter())
    with pytest.raises(
        ValueError, match="only supported when a trtllm-gen runner is selected"
    ):
        core.trtllm_batch_decode_with_kv_cache_mla(**case)
    assert not functional_dispatch.tuning


@pytest.mark.parametrize("family", ["blackwell", "kimi"])
def test_functional_dispatch_cake_routes(functional_request, monkeypatch, family):
    import sys

    case = functional_request
    case.update(backend="cake")
    if family == "kimi":
        case.update(
            query=case["query"].to(torch.float8_e4m3fn),
            kv_cache=case["kv_cache"].to(torch.float8_e4m3fn),
        )
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        out = kwargs["out"] if family == "blackwell" else args[4]
        out.fill_(7)
        return out

    monkeypatch.setattr(
        core, "_cake_trtllm_mla_blackwell_supports", lambda *args: family == "blackwell"
    )
    module, function = (
        ("cake_trtllm_mla_blackwell", "trtllm_mla_blackwell_decode")
        if family == "blackwell"
        else ("cake_kimi_k3_mla", "run_cake_kimi_k3_mla_fp8_paged_attention")
    )
    monkeypatch.setitem(
        sys.modules, f"flashinfer.mla.{module}", SimpleNamespace(**{function: launch})
    )
    result = core.trtllm_batch_decode_with_kv_cache_mla(**case)
    assert result is case["out"]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert (kwargs["query"] if family == "blackwell" else args[0]) is case["query"]
    assert (kwargs["kv_cache"] if family == "blackwell" else args[1]) is case[
        "kv_cache"
    ]
    assert kwargs["bmm1_scale"] == case["bmm1_scale"]
    assert kwargs["bmm2_scale"] == case["bmm2_scale"]
    torch.testing.assert_close(result, torch.full_like(result, 7))


@pytest.mark.parametrize("caller_lse", [False, True])
@pytest.mark.parametrize("lse_base", [None, "basee", "base2"])
def test_functional_dispatch_cake_kimi_returns_lse(
    functional_request, monkeypatch, lse_base, caller_lse
):
    import sys

    case = functional_request
    case.update(
        backend="cake",
        query=case["query"].to(torch.float8_e4m3fn),
        kv_cache=case["kv_cache"].to(torch.float8_e4m3fn),
        return_lse=True,
        return_lse_base=lse_base,
    )
    user_lse = None
    if caller_lse:
        user_lse = torch.empty(
            case["query"].shape[:-1], dtype=torch.float32, device="cuda"
        )
        case.update(lse=user_lse)
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        args[4].fill_(7)
        kwargs["lse"].fill_(3)

    monkeypatch.setattr(
        core, "_cake_trtllm_mla_blackwell_supports", lambda *args: False
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.mla.cake_kimi_k3_mla",
        SimpleNamespace(run_cake_kimi_k3_mla_fp8_paged_attention=launch),
    )
    out, lse = core.trtllm_batch_decode_with_kv_cache_mla(**case)
    assert out is case["out"]
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["lse"] is lse
    if caller_lse:
        assert lse is user_lse
    assert kwargs["lse_base"] == ("2" if lse_base == "base2" else "e")
    assert lse.dtype == torch.float32
    assert lse.shape == case["query"].shape[:-1]
    torch.testing.assert_close(out, torch.full_like(out, 7))
    torch.testing.assert_close(lse, torch.full_like(lse, 3))


@pytest.mark.parametrize("caller_lse", [False, True])
@pytest.mark.parametrize("lse_base", [None, "basee", "base2"])
def test_functional_dispatch_cake_blackwell_lse_base(
    functional_request, monkeypatch, lse_base, caller_lse
):
    """The Blackwell Cake route writes base-2 LSE: ``return_lse_base="basee"``
    converts the returned LSE (the caller's buffer when given) to natural-log
    units in place; ``None`` and ``"base2"`` keep base 2."""
    import sys

    case = functional_request
    case.update(backend="cake", return_lse=True, return_lse_base=lse_base)
    user_lse = None
    if caller_lse:
        user_lse = torch.empty(
            case["query"].shape[:-1], dtype=torch.float32, device="cuda"
        )
        case.update(lse=user_lse)
    calls = []

    def launch(**kwargs):
        calls.append(kwargs)
        lse = kwargs["lse"]
        if lse is None:
            lse = torch.empty(
                (2, 16), dtype=torch.float32, device=kwargs["query"].device
            )
        kwargs["out"].fill_(7)
        lse.fill_(3)
        return kwargs["out"], lse

    monkeypatch.setattr(core, "_cake_trtllm_mla_blackwell_supports", lambda *args: True)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.mla.cake_trtllm_mla_blackwell",
        SimpleNamespace(trtllm_mla_blackwell_decode=launch),
    )
    out, lse = core.trtllm_batch_decode_with_kv_cache_mla(**case)
    assert out is case["out"] and len(calls) == 1
    assert calls[0]["lse"] is user_lse and calls[0]["return_lse"]
    if caller_lse:
        assert lse is user_lse
    expected = 3 * math.log(2.0) if lse_base == "basee" else 3.0
    torch.testing.assert_close(lse, torch.full_like(lse, expected))
    torch.testing.assert_close(out, torch.full_like(out, 7))


def test_functional_dispatch_cake_kimi_rejects_pdl(functional_request, monkeypatch):
    monkeypatch.setattr(
        core, "_cake_trtllm_mla_blackwell_supports", lambda *args: False
    )
    with pytest.raises(ValueError, match="backend='cake' does not support enable_pdl"):
        core.trtllm_batch_decode_with_kv_cache_mla(
            **{
                **functional_request,
                "backend": "cake",
                "query": functional_request["query"].to(torch.float8_e4m3fn),
                "kv_cache": functional_request["kv_cache"].to(torch.float8_e4m3fn),
                "return_lse": True,
                "enable_pdl": True,
            }
        )


@pytest.mark.parametrize(
    "implementation,expected", [("monolithic", 21), ("modular", 31)]
)
def test_functional_dispatch_cute_implementation(
    functional_request, functional_dispatch, implementation, expected
):
    functional_dispatch.selected = "cute-dsl"
    functional_request.update(backend="cute-dsl", cute_dsl_impl=implementation)
    out = core.trtllm_batch_decode_with_kv_cache_mla(**functional_request)
    assert out is functional_request["out"]
    torch.testing.assert_close(out, torch.full_like(out, expected))
