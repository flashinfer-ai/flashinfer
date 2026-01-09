"""Tests for batch_invariant parameter in trtllm decode functions."""
import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

GPU_DEVICE = "cuda:0"

global_trtllm_gen_fmha_workspace_buffer = None
workspace_size = 256 * 1024 * 1024


def create_workspace_buffers(device):
    """Create workspace buffers for testing."""
    global global_trtllm_gen_fmha_workspace_buffer
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    return global_trtllm_gen_fmha_workspace_buffer


@pytest.mark.parametrize("kv_layout", ["HND"])
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 8, 4),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("max_in_kv_len", [2048])
@pytest.mark.parametrize("head_dim", [128])
def test_trtllm_batch_decode_batch_invariant(
    kv_layout,
    batch_size,
    q_len_per_req,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
    max_in_kv_len,
    head_dim,
):
    """Test that batch_invariant parameter produces consistent results across different batch sizes."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))

    # trtllm-gen backend requires SM100 and SM103 GPUs
    if compute_capability[0] != 10:
        pytest.skip("trtllm-gen backend requires SM100 and SM103 GPUs.")

    torch.manual_seed(42)  # Fixed seed for reproducibility

    num_qo_heads = num_kv_heads * head_grp_size
    q_dtype_torch = DTYPE_MAP[q_dtype]

    # Create two simple requests with the same content
    seq_len1 = 128  # Fixed KV seq length

    # Single request test
    q_single = torch.randn(1, num_qo_heads, head_dim, device=GPU_DEVICE, dtype=q_dtype_torch)
    seq_lens_single = torch.tensor([seq_len1], dtype=torch.int32, device=GPU_DEVICE)

    # Create KV cache for single request
    num_pages_single = (seq_len1 + page_size - 1) // page_size
    kv_cache_single = torch.randn(
        num_pages_single, 2, num_kv_heads, page_size, head_dim,
        device=GPU_DEVICE, dtype=q_dtype_torch
    )
    page_table_single = torch.arange(num_pages_single, dtype=torch.int32, device=GPU_DEVICE).unsqueeze(0)

    workspace_buffer_single = create_workspace_buffers(GPU_DEVICE)

    sm_scale = float(1.0 / (head_dim**0.5))
    bmm1_scale = sm_scale
    bmm2_scale = 1.0

    # Run with batch_invariant=True for single request
    output_single = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q_single,
        kv_cache_single,
        workspace_buffer_single,
        page_table_single,
        seq_lens_single,
        seq_len1,
        bmm1_scale,
        bmm2_scale,
        window_left,
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        backend="trtllm-gen",
        batch_invariant=True,
    )

    # Now test with a batch containing the same request replicated
    q_batch = q_single.repeat(batch_size, 1, 1)
    seq_lens_batch = torch.full((batch_size,), seq_len1, dtype=torch.int32, device=GPU_DEVICE)

    # Create KV cache for batch (replicate the same pages)
    kv_cache_batch = kv_cache_single.repeat(batch_size, 1, 1, 1, 1)

    # Create page table for batch
    page_table_batch = torch.zeros(batch_size, num_pages_single, dtype=torch.int32, device=GPU_DEVICE)
    for i in range(batch_size):
        page_table_batch[i] = torch.arange(
            i * num_pages_single, (i+1) * num_pages_single, dtype=torch.int32, device=GPU_DEVICE
        )

    workspace_buffer_batch = create_workspace_buffers(GPU_DEVICE)

    # Run with batch_invariant=True for batch
    output_batch = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q_batch,
        kv_cache_batch,
        workspace_buffer_batch,
        page_table_batch,
        seq_lens_batch,
        seq_len1,
        bmm1_scale,
        bmm2_scale,
        window_left,
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        backend="trtllm-gen",
        batch_invariant=True,
    )

    # Compare: the first output in the batch should match the single output
    rtol, atol = 1e-3, 1e-3  # Tight tolerance since we expect identical results

    torch.testing.assert_close(
        output_single[0],
        output_batch[0],
        rtol=rtol,
        atol=atol,
        msg="Output with batch_invariant=True should be identical for same request in different batch sizes"
    )

    # Also verify all batch outputs are identical (since we replicated the same request)
    for i in range(1, batch_size):
        torch.testing.assert_close(
            output_batch[0],
            output_batch[i],
            rtol=rtol,
            atol=atol,
            msg=f"All outputs in batch should be identical when using same input (batch index {i})"
        )


@pytest.mark.parametrize("kv_layout", ["HND"])
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 8, 4),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("max_in_kv_len", [2048])
@pytest.mark.parametrize("head_dim", [128])
def test_trtllm_mla_batch_decode_batch_invariant(
    kv_layout,
    batch_size,
    q_len_per_req,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
    max_in_kv_len,
    head_dim,
):
    """Test that batch_invariant parameter works for MLA decode functions."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))

    # MLA requires SM100+
    if compute_capability[0] < 10:
        pytest.skip("MLA attention requires SM100+ GPUs.")

    torch.manual_seed(42)

    num_qo_heads = num_kv_heads * head_grp_size
    q_dtype_torch = DTYPE_MAP[q_dtype]

    # MLA uses different head dims
    head_dim_qk = 192
    head_dim_vo = 128

    seq_len1 = 128

    # Single request
    q_single = torch.randn(1, num_qo_heads, head_dim_qk, device=GPU_DEVICE, dtype=q_dtype_torch)
    seq_lens_single = torch.tensor([seq_len1], dtype=torch.int32, device=GPU_DEVICE)

    num_pages_single = (seq_len1 + page_size - 1) // page_size
    # For MLA: K has head_dim_qk, V has head_dim_vo
    k_cache_single = torch.randn(
        num_pages_single, num_kv_heads, page_size, head_dim_qk,
        device=GPU_DEVICE, dtype=q_dtype_torch
    )
    v_cache_single = torch.randn(
        num_pages_single, num_kv_heads, page_size, head_dim_vo,
        device=GPU_DEVICE, dtype=q_dtype_torch
    )
    page_table_single = torch.arange(num_pages_single, dtype=torch.int32, device=GPU_DEVICE).unsqueeze(0)

    workspace_buffer_single = create_workspace_buffers(GPU_DEVICE)

    sm_scale = float(1.0 / (head_dim_qk**0.5))
    bmm1_scale = sm_scale
    bmm2_scale = 1.0

    # Run with batch_invariant=True for single request
    output_single = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        q_single,
        (k_cache_single, v_cache_single),
        workspace_buffer_single,
        page_table_single,
        seq_lens_single,
        seq_len1,
        bmm1_scale,
        bmm2_scale,
        window_left,
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        backend="trtllm-gen",
        batch_invariant=True,
    )

    # Batch test
    q_batch = q_single.repeat(batch_size, 1, 1)
    seq_lens_batch = torch.full((batch_size,), seq_len1, dtype=torch.int32, device=GPU_DEVICE)

    k_cache_batch = k_cache_single.repeat(batch_size, 1, 1, 1)
    v_cache_batch = v_cache_single.repeat(batch_size, 1, 1, 1)

    page_table_batch = torch.zeros(batch_size, num_pages_single, dtype=torch.int32, device=GPU_DEVICE)
    for i in range(batch_size):
        page_table_batch[i] = torch.arange(
            i * num_pages_single, (i+1) * num_pages_single, dtype=torch.int32, device=GPU_DEVICE
        )

    workspace_buffer_batch = create_workspace_buffers(GPU_DEVICE)

    output_batch = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        q_batch,
        (k_cache_batch, v_cache_batch),
        workspace_buffer_batch,
        page_table_batch,
        seq_lens_batch,
        seq_len1,
        bmm1_scale,
        bmm2_scale,
        window_left,
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        backend="trtllm-gen",
        batch_invariant=True,
    )

    rtol, atol = 1e-3, 1e-3

    torch.testing.assert_close(
        output_single[0],
        output_batch[0],
        rtol=rtol,
        atol=atol,
        msg="MLA output with batch_invariant=True should be identical for same request in different batch sizes"
    )

    for i in range(1, batch_size):
        torch.testing.assert_close(
            output_batch[0],
            output_batch[i],
            rtol=rtol,
            atol=atol,
            msg=f"All MLA outputs in batch should be identical when using same input (batch index {i})"
        )
