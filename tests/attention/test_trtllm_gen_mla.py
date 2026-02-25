import pytest
import torch
import random

import flashinfer
from flashinfer.utils import get_compute_capability

global_workspace_buffer = None  # can.be empty initialized
global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
workspace_size = 128 * 1024 * 1024


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


def trtllm_batch_decode_mla(
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
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "xqa":
        if compute_capability[0] != 12:
            pytest.skip("XQA MLA only supports SM120 GPUs")
        if q_len_per_request != 1 or dtype != torch.float8_e4m3fn:
            pytest.skip(
                "XQA MLA only supports q_len_per_request == 1 and dtype == torch.float8_e4m3fn"
            )
    if backend == "trtllm-gen":
        if compute_capability[0] != 10:
            pytest.skip("TRTLLM-GEN MLA only supports SM100 and SM103 GPUs")
    if dynamic_scale and dtype != torch.float8_e4m3fn:
        pytest.skip("Dynamic scale is not supported for non-fp8 dtype")

    if skips_softmax and backend != "trtllm-gen":
        pytest.skip("skips_softmax is only supported for trtllm-gen backend")

    torch.manual_seed(42)
    device = "cuda:0"

    # Deepseek attention config (decode-MLA)
    num_q_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512

    # Initialize tensors
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
    ).to(dtype)

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

    # Create interleaved KV cache
    # Allocate more than needed blocks, block_id is just enough, to mimick real-world cases
    kv_cache = torch.randn(
        size=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim), device=device
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
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    workspace_buffer = global_trtllm_gen_fmha_workspace_buffer
    workspace_buffer_ref = global_workspace_buffer

    # Using a tiny threshold should give the same output as standard attention
    skip_softmax_threshold_scale_factor = 1e-30 if skips_softmax else None

    # Run decode-MLA
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale / ((128 + 64) ** 0.5),
        bmm2_scale=1.0,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        backend=backend,
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()

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

    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * q_len_per_request
    )
    kv_indptr = torch.zeros_like(q_indptr)
    kv_indptr[1:] = torch.cumsum(blocks_per_seq, dim=0)
    kv_indices = all_block_ids.int()

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        seq_lens_tensor,
        num_q_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        True,
        sm_scale,
        query.dtype,
        kv_cache.dtype,
    )
    q_nope = query[..., :kv_lora_rank].view(
        batch_size * q_len_per_request, num_q_heads, kv_lora_rank
    )
    q_pe = query[..., kv_lora_rank:].view(
        batch_size * q_len_per_request, num_q_heads, qk_rope_head_dim
    )

    # todo: fix kv_cache
    ckv = kv_cache[..., :kv_lora_rank]
    kpe = kv_cache[..., kv_lora_rank:]

    o_ref = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    if backend == "trtllm-gen":
        # check is nan
        assert not torch.isnan(o_ref).any(), "o_ref is nan"
        assert not torch.isnan(output).any(), "output is nan"

        if dtype == torch.float8_e4m3fn:
            try:
                torch.testing.assert_close(
                    output,
                    o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
                    rtol=1e-1,
                    atol=1e-1,
                )  # todo: do reference with normal attention?
            except AssertionError as e:
                print("output:", output)
                print("o_ref:", o_ref)
                raise e
        else:
            try:
                torch.testing.assert_close(
                    output,
                    o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
                    rtol=1e-2,
                    atol=1e-2,
                )
            except AssertionError as e:
                print("output:", output)
                print("o_ref:", o_ref)
                raise e
    elif backend == "xqa":
        atol = 0.05
        rtol = 0.05

        diff_abs = torch.abs(
            o_ref.view(batch_size, q_len_per_request, num_q_heads, -1) - output
        )
        diff_rel = diff_abs / (
            torch.abs(o_ref.view(batch_size, q_len_per_request, num_q_heads, -1)) + 1e-8
        )

        within_tolerance = (diff_abs <= atol) | (diff_rel <= rtol)

        pass_ratio = within_tolerance.float().mean().item()

        required_ratio = 0.95
        assert pass_ratio >= required_ratio, (
            f"Total {o_ref.numel()} elements, only {pass_ratio:.1%} meet tolerance criteria, "
            f"require at least {required_ratio:.1%}"
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
@pytest.mark.parametrize("backend", ["trtllm-gen", "xqa"])
@pytest.mark.parametrize("skips_softmax", [False, True])
def test_trtllm_batch_decode_mla(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    page_size: int,
    q_len_per_request: int,
    dynamic_scale: bool,
    enable_pdl: bool,
    backend: str,
    skips_softmax: bool,
):
    trtllm_batch_decode_mla(
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
    )


@pytest.mark.parametrize(
    "batch_size",
    [2, 4, 8],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("q_len_per_request", [1, 2, 3])
@pytest.mark.parametrize("dynamic_scale", [False])
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("backend", ["trtllm-gen"])
@pytest.mark.parametrize("MAX_SEQ_LEN", [1024, 8960])
@pytest.mark.parametrize("skips_softmax", [False, True])
def test_dsr1_trtllm_mla(
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
):
    trtllm_batch_decode_mla(
        batch_size,
        scale,
        dtype,
        page_size,
        q_len_per_request,
        dynamic_scale,
        enable_pdl,
        backend,
        MAX_SEQ_LEN,
        skips_softmax,
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
def test_trtllm_batch_decode_mla_sparse(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    q_len_per_request: int,
    topk: int,
    is_varlen: bool,
    enable_pdl: bool,
    backend: str,
):
    """
    Test sparse MLA decoding with top-k attention.
    Based on FlashMLA test patterns from:
    https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_decoding.py
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "trtllm-gen":
        if compute_capability[0] != 10:
            pytest.skip("TRTLLM-GEN MLA only supports SM100 and SM103 GPUs")

    torch.manual_seed(42)
    device = "cuda:0"

    # Deepseek attention config (decode-MLA)
    num_q_heads = 128
    qk_nope_head_dim = 128
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
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    workspace_buffer = global_trtllm_gen_fmha_workspace_buffer
    # workspace_buffer_ref = global_workspace_buffer

    # Run sparse decode-MLA
    query_input = query.clone()
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query_input,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=indices_in_kvcache,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=topk,
        bmm1_scale=scale / ((qk_nope_head_dim + qk_rope_head_dim) ** 0.5),
        bmm2_scale=1.0,
        enable_pdl=enable_pdl,
        backend=backend,
    )

    # Check workspace buffer is zeroed
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()

    # For now, just check that output has correct shape and no NaNs
    expected_shape = (batch_size, q_len_per_request, num_q_heads, kv_lora_rank)
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

    out_ref, lse_ref = sparse_mla_reference_torch(
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
            print(f"Output sample: {output[0, 0, 0, :8]}")
            print(f"Reference sample: {out_ref[0, 0, 0, :8]}")
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
            print(f"Output sample: {output[0, 0, 0, :8]}")
            print(f"Output sample: {output[0, 1, 0, :8]}")
            print(f"Reference sample: {out_ref[0, 0, 0, :8]}")
            print(f"Reference sample: {out_ref[0, 1, 0, :8]}")
            raise e

    print(
        f"Sparse MLA test passed: batch_size={batch_size}, topk={topk}, "
        f"q_len={q_len_per_request}, varlen={is_varlen}, dtype={dtype}"
    )
