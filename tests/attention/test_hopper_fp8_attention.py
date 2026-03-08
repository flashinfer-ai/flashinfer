from typing import Tuple

import numpy as np
import pytest
import scipy as sp
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported


def get_fp8_dtype_minmax(dtype: torch.dtype) -> Tuple[float, float]:
    """Get min/max representable values for FP8 dtype."""
    if dtype == torch.float8_e4m3fn:
        return -448.0, 448.0
    elif dtype == torch.float8_e5m2:
        return -57344, 57344
    else:
        raise ValueError(f"Unsupported quantization dtype: {dtype}")


def per_head_symmetric_quant(
    x: torch.Tensor, quant_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor with per-head scale factors.

    Args:
        x: Input tensor of shape [seq_len, num_heads, head_dim]
        quant_dtype: FP8 dtype (e4m3 or e5m2)

    Returns:
        Tuple of (quantized tensor, per-head scales of shape [num_heads])
    """
    assert quant_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    o_min_val, o_max_val = get_fp8_dtype_minmax(quant_dtype)
    # Compute max per head: reduce over seq_len and head_dim
    x_max_val = x.abs().amax(dim=(0, 2)).to(dtype=torch.float32)

    s_out = torch.clamp(x_max_val / o_max_val, min=1e-6)
    s_out_broadcast = s_out.view(1, -1, 1)

    q_x_out = torch.clamp(
        x / s_out_broadcast,
        min=o_min_val,
        max=o_max_val,
    ).to(dtype=quant_dtype)

    assert not torch.any(torch.isnan(q_x_out))
    assert not torch.any(torch.isnan(s_out))

    return q_x_out, s_out


def per_tensor_symmetric_quant(
    x: torch.Tensor, quant_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor with a single per-tensor scale factor.

    Args:
        x: Input tensor of shape [seq_len, num_heads, head_dim]
        quant_dtype: FP8 dtype (e4m3 or e5m2)

    Returns:
        Tuple of (quantized tensor, per-tensor scale of shape [1])
    """
    assert quant_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    o_min_val, o_max_val = get_fp8_dtype_minmax(quant_dtype)
    # Compute max over entire tensor
    x_max_val = x.abs().amax().to(dtype=torch.float32)

    s_out = torch.clamp(x_max_val / o_max_val, min=1e-6).view(1)

    q_x_out = torch.clamp(
        x / s_out,
        min=o_min_val,
        max=o_max_val,
    ).to(dtype=quant_dtype)

    assert not torch.any(torch.isnan(q_x_out))
    assert not torch.any(torch.isnan(s_out))

    return q_x_out, s_out


def broadcast_scale_to_per_head(scale: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Broadcast per-tensor scale to per-head scale if needed.

    Args:
        scale: Scale tensor of shape [1] (per-tensor) or [num_heads] (per-head)
        num_heads: Number of heads

    Returns:
        Scale tensor of shape [num_heads]
    """
    if scale.numel() == 1:
        # Per-tensor scale: broadcast to all heads
        return scale.expand(num_heads).contiguous()
    else:
        # Already per-head scale
        assert scale.numel() == num_heads, (
            f"Scale size {scale.numel()} != num_heads {num_heads}"
        )
        return scale


def bsr_attention_ref(
    q,
    k,
    v,
    indptr,
    indices,
    mask_data,
):
    M = q.shape[0]
    N = k.shape[0]
    bsr = sp.sparse.bsr_matrix(
        (mask_data.cpu().numpy(), indices.cpu().numpy(), indptr.cpu().numpy()),
        shape=(M, N),
    )
    dense_mask = torch.tensor(bsr.toarray(), dtype=bool, device=q.device)
    o = flashinfer.prefill.single_prefill_with_kv_cache(
        q, k, v, custom_mask=dense_mask, backend="fa2"
    )
    return o


# Test single_prefill correctness: MSE should be below threshold
@pytest.mark.parametrize("seq_len", [117, 509, 1011, 2372, 7777, 12315])
@pytest.mark.parametrize("num_heads", [24, 32])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_single_prefill(seq_len, num_heads, causal, head_dim, dtype):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    # Prepare inputs
    o_dtype = torch.half
    num_qo_heads = num_kv_heads = num_heads
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    # Reference output
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa3"
    )

    # Quantize
    q_fp8, s_q = per_head_symmetric_quant(q, quant_dtype=dtype)
    k_fp8, s_k = per_head_symmetric_quant(k, quant_dtype=dtype)
    v_fp8, s_v = per_head_symmetric_quant(v, quant_dtype=dtype)

    # FP8 output
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q_fp8,
        k_fp8,
        v_fp8,
        s_q,
        s_k,
        s_v,
        causal=causal,
        backend="fa3",
        o_dtype=o_dtype,
    )

    # Compute MSE and assert
    # NOTE: This is not a strict correctness guarantee
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2)
    assert mse < 1.0, f"MSE too high: {mse.item()}"


# Test block sparse attention correctness: MSE should be below threshold
@pytest.mark.parametrize("R", [1, 4, 16])
@pytest.mark.parametrize("C", [1, 4, 16])
@pytest.mark.parametrize("M", [256, 512, 1024, 4096])
@pytest.mark.parametrize("N", [256, 512, 1024, 4096])
@pytest.mark.parametrize("num_heads", [1, 8, 24, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mask_inside_block", [False])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_block_sparse_attention(
    R, C, M, N, num_heads, head_dim, mask_inside_block, dtype
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    # print args
    print(
        f"Testing block sparse attention with R={R}, C={C}, M={M}, N={N}, num_heads={num_heads}, "
        f"head_dim={head_dim}, mask_inside_block={mask_inside_block}, dtype={dtype}"
    )
    # setup random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    # Build sparse mask
    MB = M // R
    NB = N // C
    rng = np.random.default_rng(seed=0)
    S = sp.sparse.random(MB, NB, density=0.25, random_state=rng).tocsr()
    indptr = torch.from_numpy(S.indptr).cuda()
    indices = torch.from_numpy(S.indices).cuda()
    nnz = S.nnz
    if mask_inside_block:
        data_mask = (torch.rand((nnz, R, C)) > 0.5).to(torch.bool).cuda()
    else:
        data_mask = torch.ones((nnz, R, C), dtype=torch.bool, device="cuda")

    # Random inputs
    q = torch.randn((M, num_heads, head_dim), dtype=torch.float16, device="cuda")
    k = torch.randn((N, num_heads, head_dim), dtype=torch.float16, device="cuda")
    v = torch.randn((N, num_heads, head_dim), dtype=torch.float16, device="cuda")

    # Reference output via dense mask
    o_ref = bsr_attention_ref(q, k, v, indptr, indices, data_mask)

    # Plan and run BlockSparseAttention
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    sparse_wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(
        workspace_buffer, backend="fa3"
    )
    sparse_wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        head_dim,
        mask=data_mask if mask_inside_block else None,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=torch.float16,
    )
    q_fp8, s_q = per_head_symmetric_quant(q, quant_dtype=dtype)
    k_fp8, s_k = per_head_symmetric_quant(k, quant_dtype=dtype)
    v_fp8, s_v = per_head_symmetric_quant(v, quant_dtype=dtype)
    o = sparse_wrapper.run(q_fp8, k_fp8, v_fp8, s_q, s_k, s_v)

    # Compute MSE and assert
    # NOTE: This is not a strict correctness guarantee
    mse = torch.mean((o_ref.float() - o.float()) ** 2)
    assert mse < 1.0, f"Block sparse MSE too high: {mse.item()}"


# Test batch prefill with ragged KV cache: MSE should be below threshold
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_ragged(batch_size, num_heads, head_dim, causal, dtype):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    print(
        f"Testing FP8 batch prefill ragged with batch_size={batch_size}, num_heads={num_heads}, "
        f"head_dim={head_dim}, causal={causal}, dtype={dtype}"
    )

    # Setup
    o_dtype = torch.half
    num_qo_heads = num_kv_heads = num_heads

    # Create variable length sequences
    torch.manual_seed(0)
    qo_lens = [128 * (i + 1) for i in range(batch_size)]
    kv_lens = [128 * (i + 1) for i in range(batch_size)]

    # Build ragged tensors
    qo_indptr = torch.tensor(
        [0] + [sum(qo_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.tensor(
        [0] + [sum(kv_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )

    total_qo_len = sum(qo_lens)
    total_kv_len = sum(kv_lens)

    # Create input tensors (fp16)
    q_fp16 = torch.randn(
        total_qo_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k_fp16 = torch.randn(
        total_kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v_fp16 = torch.randn(
        total_kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    # Get reference output using fp16
    wrapper_fp16 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp16.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        causal=causal,
    )
    o_ref = wrapper_fp16.run(q_fp16, k_fp16, v_fp16)

    # Quantize to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, quant_dtype=dtype)
    k_fp8, s_k = per_head_symmetric_quant(k_fp16, quant_dtype=dtype)
    v_fp8, s_v = per_head_symmetric_quant(v_fp16, quant_dtype=dtype)

    # Run FP8 batch prefill with ragged KV cache
    wrapper_fp8 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp8.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_dtype,
        causal=causal,
    )
    o_fp8 = wrapper_fp8.run(q_fp8, k_fp8, v_fp8, s_q, s_k, s_v)

    # Compute MSE
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2)
    assert mse < 1.0, f"MSE too high: {mse.item()}"


def create_per_head_varying_kv(
    shape: Tuple[int, ...],
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Create K/V tensor with per-head varying scale to reveal head offset bugs.

    Each head gets data with a slightly different scale factor: head i gets scale (1 + i*0.1).
    This ensures that if the kernel incorrectly reads head 0's data for head i,
    the output will have noticeably different magnitude, causing high MSE.
    Using small scale differences (0.1 step) to keep quantization error manageable.

    Args:
        shape: Tensor shape, should contain num_heads dimension
        num_heads: Number of heads
        head_dim: Head dimension
        dtype: Data type
        device: Device string

    Returns:
        Tensor with per-head varying scale
    """
    # Generate base random tensor
    tensor = torch.randn(shape, dtype=dtype, device=device)

    # Apply per-head scaling: head i gets multiplied by (1 + i*0.1)
    # This makes different heads have slightly different magnitudes
    # Using smaller scale differences to reduce quantization error while still detecting bugs
    # Shape handling: for paged KV (num_pages, page_size, num_heads, head_dim)
    # or for flat KV (seq_len, num_heads, head_dim)
    if len(shape) == 4:
        # Paged: (num_pages, page_size, num_heads, head_dim)
        scale = (1.0 + 0.1 * torch.arange(num_heads, dtype=dtype, device=device)).view(
            1, 1, num_heads, 1
        )
    else:
        # Flat: (seq_len, num_heads, head_dim)
        scale = (1.0 + 0.1 * torch.arange(num_heads, dtype=dtype, device=device)).view(
            1, num_heads, 1
        )

    return tensor * scale


# Test batch prefill with paged KV cache: MSE should be below threshold
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_paged(batch_size, num_heads, head_dim, causal, dtype):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    print(
        f"Testing FP8 batch prefill paged with batch_size={batch_size}, num_heads={num_heads}, "
        f"head_dim={head_dim}, causal={causal}, dtype={dtype}"
    )

    # Setup
    o_dtype = torch.half
    num_qo_heads = num_kv_heads = num_heads
    page_size = 16

    # Create variable length sequences
    torch.manual_seed(0)
    qo_lens = [128 * (i + 1) for i in range(batch_size)]
    kv_lens = [128 * (i + 1) for i in range(batch_size)]

    # Build indptr for Q
    qo_indptr = torch.tensor(
        [0] + [sum(qo_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )

    total_qo_len = sum(qo_lens)

    # Compute number of pages needed for each sequence
    kv_page_counts = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(kv_page_counts)

    # Build paged KV indptr and indices
    kv_indptr = torch.tensor(
        [0] + [sum(kv_page_counts[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    # Simple page indices: sequential allocation
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.tensor(
        [
            kv_len % page_size if kv_len % page_size != 0 else page_size
            for kv_len in kv_lens
        ],
        dtype=torch.int32,
        device="cuda",
    )

    # Create input tensors (fp16)
    q_fp16 = torch.randn(
        total_qo_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    # Paged KV cache: (num_pages, page_size, num_heads, head_dim)
    # Use per-head varying scale to reveal head offset bugs:
    # If kernel incorrectly reads head 0's data for all heads, MSE will be high
    paged_k_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )
    paged_v_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )

    # Get reference output using fp16
    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
    )
    o_ref = wrapper_fp16.run(q_fp16, (paged_k_fp16, paged_v_fp16))

    # Quantize to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, quant_dtype=dtype)
    # For paged KV, reshape to (total_tokens, num_heads, head_dim) for quantization
    k_flat = paged_k_fp16.view(-1, num_kv_heads, head_dim)
    v_flat = paged_v_fp16.view(-1, num_kv_heads, head_dim)
    k_fp8_flat, s_k = per_head_symmetric_quant(k_flat, quant_dtype=dtype)
    v_fp8_flat, s_v = per_head_symmetric_quant(v_flat, quant_dtype=dtype)
    paged_k_fp8 = k_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)
    paged_v_fp8 = v_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)

    # Run FP8 batch prefill with paged KV cache
    wrapper_fp8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp8.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_dtype,
        causal=causal,
    )
    o_fp8 = wrapper_fp8.run(q_fp8, (paged_k_fp8, paged_v_fp8), s_q, s_k, s_v)

    # Compute MSE - with per-head varying K/V data, head offset bugs will cause high MSE
    # because reading head 0's data for head i gives wrong scale magnitude
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2)
    assert mse < 1.0, f"MSE too high: {mse.item()}"


# Test batch prefill with paged KV cache and GQA (grouped query attention)
# GQA has num_qo_heads > num_kv_heads, this tests the head offset calculation more thoroughly
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 4), (8, 2)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_gqa(
    batch_size, num_qo_heads, num_kv_heads, head_dim, causal, dtype
):
    """Test FP8 batch prefill with paged KV cache using grouped query attention (GQA).

    GQA is important to test because:
    1. It exercises the head mapping logic (multiple Q heads share one KV head)
    2. It verifies kv_head_idx calculation is correct for different group sizes
    3. The per-head varying K/V data makes head offset bugs highly visible
    """
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    print(
        f"Testing FP8 batch prefill paged GQA with batch_size={batch_size}, "
        f"num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}, causal={causal}, dtype={dtype}"
    )

    # Setup
    o_dtype = torch.half
    page_size = 16

    # Create variable length sequences
    torch.manual_seed(0)
    qo_lens = [128 * (i + 1) for i in range(batch_size)]
    kv_lens = [128 * (i + 1) for i in range(batch_size)]

    # Build indptr for Q
    qo_indptr = torch.tensor(
        [0] + [sum(qo_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )

    total_qo_len = sum(qo_lens)

    # Compute number of pages needed for each sequence
    kv_page_counts = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(kv_page_counts)

    # Build paged KV indptr and indices
    kv_indptr = torch.tensor(
        [0] + [sum(kv_page_counts[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.tensor(
        [
            kv_len % page_size if kv_len % page_size != 0 else page_size
            for kv_len in kv_lens
        ],
        dtype=torch.int32,
        device="cuda",
    )

    # Create input tensors (fp16)
    q_fp16 = torch.randn(
        total_qo_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    # Paged KV cache: (num_pages, page_size, num_kv_heads, head_dim)
    # Use per-head varying scale to reveal head offset bugs
    paged_k_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )
    paged_v_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )

    # Get reference output using fp16
    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
    )
    o_ref = wrapper_fp16.run(q_fp16, (paged_k_fp16, paged_v_fp16))

    # Quantize to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, quant_dtype=dtype)
    k_flat = paged_k_fp16.view(-1, num_kv_heads, head_dim)
    v_flat = paged_v_fp16.view(-1, num_kv_heads, head_dim)
    k_fp8_flat, s_k = per_head_symmetric_quant(k_flat, quant_dtype=dtype)
    v_fp8_flat, s_v = per_head_symmetric_quant(v_flat, quant_dtype=dtype)
    paged_k_fp8 = k_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)
    paged_v_fp8 = v_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)

    # Run FP8 batch prefill with paged KV cache
    wrapper_fp8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp8.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_dtype,
        causal=causal,
    )
    o_fp8 = wrapper_fp8.run(q_fp8, (paged_k_fp8, paged_v_fp8), s_q, s_k, s_v)

    # Compute MSE
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2)
    assert mse < 1.0, f"MSE too high: {mse.item()}"


# Test batch decode with paged KV cache for FA3 backend with FP8 kv-cache.
# Under the hood, BatchDecodeWithPagedKVCacheWrapper actually uses the same
# backend module as BatchPrefillWithPagedKVCacheWrapper, so this test is just
# making sure that BatchDecodeWithPagedKVCacheWrapper interface works correctly.
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 4), (8, 2)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_decode_paged(batch_size, num_qo_heads, num_kv_heads, head_dim, dtype):
    """Test FP8 batch decode with paged KV cache using grouped query attention (GQA)."""
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    print(
        f"Testing FP8 batch decode paged GQA with batch_size={batch_size}, "
        f"num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}, dtype={dtype}"
    )

    # Setup
    o_dtype = torch.half
    page_size = 16

    # Create variable length sequences
    torch.manual_seed(0)
    kv_lens = [128 * (i + 1) for i in range(batch_size)]

    # Compute number of pages needed for each sequence
    kv_page_counts = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(kv_page_counts)

    # Build paged KV indptr and indices
    kv_indptr = torch.tensor(
        [0] + [sum(kv_page_counts[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.tensor(
        [
            kv_len % page_size if kv_len % page_size != 0 else page_size
            for kv_len in kv_lens
        ],
        dtype=torch.int32,
        device="cuda",
    )

    # Create input tensors (fp16)
    q_fp16 = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    # Paged KV cache: (num_pages, page_size, num_kv_heads, head_dim)
    # Use per-head varying scale to reveal head offset bugs
    paged_k_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )
    paged_v_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )

    # Get reference output using fp16
    wrapper_fp16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        use_tensor_cores=True,
        backend="fa3",
    )
    wrapper_fp16.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    o_ref = wrapper_fp16.run(q_fp16, (paged_k_fp16, paged_v_fp16))

    # Quantize to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, quant_dtype=dtype)
    k_flat = paged_k_fp16.view(-1, num_kv_heads, head_dim)
    v_flat = paged_v_fp16.view(-1, num_kv_heads, head_dim)
    k_fp8_flat, s_k = per_head_symmetric_quant(k_flat, quant_dtype=dtype)
    v_fp8_flat, s_v = per_head_symmetric_quant(v_flat, quant_dtype=dtype)
    paged_k_fp8 = k_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)
    paged_v_fp8 = v_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)

    # Run FP8 batch decode with paged KV cache
    wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        use_tensor_cores=True,
        backend="fa3",
    )
    wrapper_fp8.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_dtype,
    )
    o_fp8 = wrapper_fp8.run(q_fp8, (paged_k_fp8, paged_v_fp8), s_q, s_k, s_v)

    # Compute MSE
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2)
    assert mse < 0.01, f"MSE too high: {mse.item()}"


# Test both per-tensor and per-head scale types
@pytest.mark.parametrize("scale_type", ["per_head", "per_tensor"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_scale_types(scale_type, dtype):
    """Test FP8 batch prefill with both per-tensor and per-head scale types.

    This test verifies that:
    1. Per-head scale: shape [num_heads], each head has its own scale
    2. Per-tensor scale: shape [1], single scale broadcast to all heads
    """
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    print(
        f"Testing FP8 batch prefill paged with scale_type={scale_type}, dtype={dtype}"
    )

    # Setup
    batch_size = 2
    num_qo_heads = num_kv_heads = 8
    head_dim = 128
    o_dtype = torch.half
    page_size = 16
    causal = True

    # Create variable length sequences
    torch.manual_seed(0)
    qo_lens = [128 * (i + 1) for i in range(batch_size)]
    kv_lens = [128 * (i + 1) for i in range(batch_size)]

    # Build indptr for Q
    qo_indptr = torch.tensor(
        [0] + [sum(qo_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )

    total_qo_len = sum(qo_lens)

    # Compute number of pages needed for each sequence
    kv_page_counts = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(kv_page_counts)

    # Build paged KV indptr and indices
    kv_indptr = torch.tensor(
        [0] + [sum(kv_page_counts[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.tensor(
        [
            kv_len % page_size if kv_len % page_size != 0 else page_size
            for kv_len in kv_lens
        ],
        dtype=torch.int32,
        device="cuda",
    )

    # Create input tensors (fp16)
    q_fp16 = torch.randn(
        total_qo_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    paged_k_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )
    paged_v_fp16 = create_per_head_varying_kv(
        (total_pages, page_size, num_kv_heads, head_dim),
        num_kv_heads,
        head_dim,
        torch.half,
        "cuda",
    )

    # Get reference output using fp16
    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
    )
    o_ref = wrapper_fp16.run(q_fp16, (paged_k_fp16, paged_v_fp16))

    # Quantize to FP8 with the specified scale type
    if scale_type == "per_head":
        q_fp8, s_q = per_head_symmetric_quant(q_fp16, quant_dtype=dtype)
        k_flat = paged_k_fp16.view(-1, num_kv_heads, head_dim)
        v_flat = paged_v_fp16.view(-1, num_kv_heads, head_dim)
        k_fp8_flat, s_k = per_head_symmetric_quant(k_flat, quant_dtype=dtype)
        v_fp8_flat, s_v = per_head_symmetric_quant(v_flat, quant_dtype=dtype)
    else:  # per_tensor
        q_fp8, s_q = per_tensor_symmetric_quant(q_fp16, quant_dtype=dtype)
        k_flat = paged_k_fp16.view(-1, num_kv_heads, head_dim)
        v_flat = paged_v_fp16.view(-1, num_kv_heads, head_dim)
        k_fp8_flat, s_k = per_tensor_symmetric_quant(k_flat, quant_dtype=dtype)
        v_fp8_flat, s_v = per_tensor_symmetric_quant(v_flat, quant_dtype=dtype)

    paged_k_fp8 = k_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)
    paged_v_fp8 = v_fp8_flat.view(total_pages, page_size, num_kv_heads, head_dim)

    # Broadcast per-tensor scales to per-head if needed
    # The kernel expects per-head scales, so we broadcast [1] -> [num_heads]
    s_q_broadcast = broadcast_scale_to_per_head(s_q, num_qo_heads)
    s_k_broadcast = broadcast_scale_to_per_head(s_k, num_kv_heads)
    s_v_broadcast = broadcast_scale_to_per_head(s_v, num_kv_heads)

    # Run FP8 batch prefill with paged KV cache
    wrapper_fp8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    wrapper_fp8.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_dtype,
        causal=causal,
    )
    o_fp8 = wrapper_fp8.run(
        q_fp8, (paged_k_fp8, paged_v_fp8), s_q_broadcast, s_k_broadcast, s_v_broadcast
    )

    # Compute MSE
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2)
    assert mse < 1.0, f"MSE too high for scale_type={scale_type}: {mse.item()}"


if __name__ == "__main__":
    # Test batch prefill paged
    for batch_size in [2]:
        for num_heads in [8]:
            for head_dim in [128, 256]:
                for causal in [True, False]:
                    for dtype in [torch.float8_e4m3fn]:
                        test_batch_prefill_paged(
                            batch_size, num_heads, head_dim, causal, dtype
                        )

    # Test batch decode paged
    for batch_size in [2]:
        for num_qo_heads in [8]:
            for num_kv_heads in [2]:
                for head_dim in [128]:
                    for dtype in [torch.float8_e4m3fn]:
                        test_batch_decode_paged(
                            batch_size, num_qo_heads, num_kv_heads, head_dim, dtype
                        )

    # Test batch prefill ragged
    for batch_size in [2]:
        for num_heads in [8]:
            for head_dim in [128]:
                for causal in [True, False]:
                    for dtype in [torch.float8_e4m3fn]:
                        test_batch_prefill_ragged(
                            batch_size, num_heads, head_dim, causal, dtype
                        )

    # Test block sparse attention
    for R in [4]:
        for C in [1]:
            for M in [1024]:
                for N in [512]:
                    for num_heads in [8]:
                        for head_dim in [256]:
                            for mask_inside_block in [False]:
                                for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                                    test_block_sparse_attention(
                                        R,
                                        C,
                                        M,
                                        N,
                                        num_heads,
                                        head_dim,
                                        mask_inside_block,
                                        dtype,
                                    )
