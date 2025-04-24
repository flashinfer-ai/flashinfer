from typing import Tuple

import numpy as np
import pytest
import scipy as sp
import torch

import flashinfer


def per_head_symmetric_quant(
    x: torch.Tensor, quant_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [seq_len, num_heads, head_dim]
    assert quant_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    def get_dtype_minmax(dtype: torch.dtype) -> Tuple[float, float]:
        if dtype == torch.float8_e4m3fn:
            return -448.0, 448.0
        elif dtype == torch.float8_e5m2:
            return -57344, 57344
        else:
            raise ValueError(f"Unsupported quantization dtype: {dtype}")

    o_min_val, o_max_val = get_dtype_minmax(quant_dtype)
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


if __name__ == "__main__":
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
