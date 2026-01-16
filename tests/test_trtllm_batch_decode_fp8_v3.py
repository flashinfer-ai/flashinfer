"""
Test trtllm_batch_decode_with_kv_cache FP8 correctness.

This test compares three variants:
- Variant 1 (Baseline): BF16 naive torch implementation using FP8 pseudo-quantized QKV
- Variant 2 (FP8): trtllm_batch_decode_with_kv_cache with FP8 QKV
- Variant 3 (FP4+FP8): trtllm_batch_decode_with_kv_cache with FP8 Q and FP4-quantized KV
                       KV path: BF16 -> nvFP4 -> BF16 -> FP8

All variants use pseudo-quantization so they have comparable quantization errors.

Test parameters:
- num_qo_heads: 16
- num_kv_heads: 2  (GQA with 8 query heads per KV head)
- head_dim: 256
- input_seq_len (isl): 1024
- output_seq_len (oisl): 1
- page_size: 64

Quantization:
- FP8: Uses torch.to() for FP8 conversion (scale=1.0)
- FP4: Uses nvfp4_quantize/e2m1_and_ufp8sf_scale_to_float for nvFP4 conversion
"""

import torch
import flashinfer
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from kvfp4_tensor import KVFP4QuantizeUtil


def naive_attention_decode_bf16(q, k_cache, v_cache, sm_scale=None):
    """
    Naive BF16 attention implementation for batch decode (single query per batch).

    Args:
        q: Query tensor [batch_size, num_qo_heads, head_dim]
        k_cache: Key cache [batch_size, seq_len, num_kv_heads, head_dim]
        v_cache: Value cache [batch_size, seq_len, num_kv_heads, head_dim]
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        output: [batch_size, num_qo_heads, head_dim]
    """
    batch_size, num_qo_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _ = k_cache.shape

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    # Expand KV heads to match QO heads (GQA)
    num_heads_per_group = num_qo_heads // num_kv_heads
    # k_cache: [batch_size, seq_len, num_kv_heads, head_dim]
    # -> [batch_size, seq_len, num_qo_heads, head_dim]
    k_expanded = k_cache.repeat_interleave(num_heads_per_group, dim=2)
    v_expanded = v_cache.repeat_interleave(num_heads_per_group, dim=2)

    # q: [batch_size, num_qo_heads, head_dim]
    # k_expanded: [batch_size, seq_len, num_qo_heads, head_dim]
    # Compute attention scores: [batch_size, num_qo_heads, seq_len]
    q_expanded = q.unsqueeze(2)  # [batch_size, num_qo_heads, 1, head_dim]
    k_transposed = k_expanded.transpose(1, 2).transpose(
        2, 3
    )  # [batch_size, num_qo_heads, head_dim, seq_len]
    scores = (
        torch.matmul(q_expanded, k_transposed).squeeze(2) * sm_scale
    )  # [batch_size, num_qo_heads, seq_len]

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_qo_heads, seq_len]

    # Apply attention to values
    # attn_weights: [batch_size, num_qo_heads, seq_len]
    # v_expanded: [batch_size, seq_len, num_qo_heads, head_dim]
    v_transposed = v_expanded.transpose(
        1, 2
    )  # [batch_size, num_qo_heads, seq_len, head_dim]
    output = torch.matmul(attn_weights.unsqueeze(2), v_transposed).squeeze(
        2
    )  # [batch_size, num_qo_heads, head_dim]

    return output


def fp8_pseudo_quantize(tensor):
    """
    Apply FP8 pseudo-quantization with scale=1.0 (using torch.to).

    Args:
        tensor: Input tensor in BF16

    Returns:
        Pseudo-quantized tensor in BF16 (BF16 -> FP8 -> BF16)
    """
    return tensor.to(torch.float8_e4m3fn).to(torch.bfloat16)


def verify_fp4_quantization(original, quantized, name="Tensor"):
    """
    Verify FP4 quantization/dequantization quality.

    Args:
        original: Original tensor before quantization
        quantized: Tensor after quantization and dequantization
        name: Name for printing (e.g., "K cache", "V cache")
    """
    original_f32 = original.float()
    quantized_f32 = quantized.float()

    # 1. Check for NaN/Inf
    has_nan_orig = torch.isnan(original_f32).any().item()
    has_inf_orig = torch.isinf(original_f32).any().item()
    has_nan_quant = torch.isnan(quantized_f32).any().item()
    has_inf_quant = torch.isinf(quantized_f32).any().item()

    print(f"\n  [{name}] Verification:")
    print(f"    NaN check: original={has_nan_orig}, quantized={has_nan_quant}")
    print(f"    Inf check: original={has_inf_orig}, quantized={has_inf_quant}")

    if has_nan_quant or has_inf_quant:
        print(f"    ⚠️  WARNING: Found NaN or Inf after quantization!")
        return False

    # 2. Statistics comparison
    orig_mean = original_f32.mean().item()
    orig_std = original_f32.std().item()
    orig_min = original_f32.min().item()
    orig_max = original_f32.max().item()

    quant_mean = quantized_f32.mean().item()
    quant_std = quantized_f32.std().item()
    quant_min = quantized_f32.min().item()
    quant_max = quantized_f32.max().item()

    print(
        f"    Original:  mean={orig_mean:+.6f}, std={orig_std:.6f}, range=[{orig_min:+.6f}, {orig_max:+.6f}]"
    )
    print(
        f"    Quantized: mean={quant_mean:+.6f}, std={quant_std:.6f}, range=[{quant_min:+.6f}, {quant_max:+.6f}]"
    )

    # 3. Error metrics
    abs_error = (original_f32 - quantized_f32).abs()
    rel_error = abs_error / (original_f32.abs() + 1e-8)

    print(
        f"    Abs error: mean={abs_error.mean().item():.6f}, max={abs_error.max().item():.6f}"
    )
    print(
        f"    Rel error: mean={rel_error.mean().item():.6f}, max={rel_error.max().item():.6f}"
    )

    # 4. Statistical ratio
    mean_ratio = quant_mean / (orig_mean + 1e-10)
    std_ratio = quant_std / (orig_std + 1e-10)

    print(f"    Mean ratio (quant/orig): {mean_ratio:.6f}")
    print(f"    Std ratio (quant/orig): {std_ratio:.6f}")

    # 5. Sample values check (show a few examples)
    flat_orig = original_f32.flatten()
    flat_quant = quantized_f32.flatten()

    # Sample 5 random indices
    sample_indices = torch.randperm(flat_orig.numel())[:5]
    print(f"    Sample values (5 random elements):")
    for i, idx in enumerate(sample_indices):
        orig_val = flat_orig[idx].item()
        quant_val = flat_quant[idx].item()
        error = abs(orig_val - quant_val)
        print(
            f"      [{i}] orig={orig_val:+.6f}, quant={quant_val:+.6f}, error={error:.6f}"
        )

    # 6. Value distribution check (histogram)
    # Count how many values fall into different ranges
    def count_in_range(tensor, low, high):
        return ((tensor >= low) & (tensor < high)).sum().item()

    total = original_f32.numel()
    ranges = [
        (-float("inf"), -1.0),
        (-1.0, -0.1),
        (-0.1, -0.01),
        (-0.01, 0.01),
        (0.01, 0.1),
        (0.1, 1.0),
        (1.0, float("inf")),
    ]

    print(f"    Value distribution (% of total):")
    for low, high in ranges:
        orig_count = count_in_range(original_f32, low, high)
        quant_count = count_in_range(quantized_f32, low, high)
        range_str = f"[{low:+.2f}, {high:+.2f})"
        print(
            f"      {range_str:20s}: orig={orig_count * 100 / total:5.1f}%, quant={quant_count * 100 / total:5.1f}%"
        )

    # 7. Check if std ratio is too small (indicates severe compression)
    if std_ratio < 0.1:
        print(f"    ⚠️  WARNING: Std ratio < 0.1, severe information loss!")
        return False

    return True


def prepare_paged_kv_cache(
    k_cache_flat, v_cache_flat, batch_size, seq_len, page_size, num_kv_heads, head_dim
):
    """
    Convert flat KV cache to paged format.

    Args:
        k_cache_flat: [batch_size, seq_len, num_kv_heads, head_dim]
        v_cache_flat: [batch_size, seq_len, num_kv_heads, head_dim]
        page_size: Number of tokens per page

    Returns:
        kv_cache: [num_pages, 2, num_kv_heads, page_size, head_dim]
        block_tables: [batch_size, num_pages_per_seq]
    """
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    device = k_cache_flat.device
    dtype = k_cache_flat.dtype

    # Create paged KV cache
    kv_cache = torch.zeros(
        total_num_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=dtype,
        device=device,
    )

    # Create block tables
    block_tables = torch.arange(
        0, total_num_pages, dtype=torch.int32, device=device
    ).view(batch_size, num_pages_per_seq)

    # Fill paged KV cache
    for b in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            page_id = b * num_pages_per_seq + page_idx
            start_idx = page_idx * page_size
            end_idx = min(start_idx + page_size, seq_len)
            valid_len = end_idx - start_idx

            # Copy K and V
            # k_cache_flat[b, start_idx:end_idx, :, :] has shape [valid_len, num_kv_heads, head_dim]
            # We need to transpose to [num_kv_heads, valid_len, head_dim]
            kv_cache[page_id, 0, :, :valid_len, :] = k_cache_flat[
                b, start_idx:end_idx, :, :
            ].transpose(0, 1)
            kv_cache[page_id, 1, :, :valid_len, :] = v_cache_flat[
                b, start_idx:end_idx, :, :
            ].transpose(0, 1)

    return kv_cache, block_tables


def test_trtllm_batch_decode_fp8():
    """Test trtllm_batch_decode_with_kv_cache FP8 correctness."""
    print("\n" + "=" * 80)
    print("Test: trtllm_batch_decode_with_kv_cache FP8 Correctness")
    print("=" * 80 + "\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Test parameters
    batch_size = 1
    num_qo_heads = 16
    num_kv_heads = 2
    head_dim = 256
    q_len = 1  # Query length (decode phase)
    kv_len = 64  # KV cache length (aligned to one page)
    page_size = 64
    device = torch.device("cuda:0")

    print(f"Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_qo_heads: {num_qo_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  q_len: {q_len}")
    print(f"  kv_len: {kv_len}")
    print(f"  page_size: {page_size}\n")

    # Create random input data in BF16
    q_bf16 = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k_cache_bf16 = torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache_bf16 = torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # Apply FP8 pseudo-quantization (scale=1.0)
    print("Step 1: Apply FP8 pseudo-quantization (scale=1.0)")
    q_pseudo_fp8 = fp8_pseudo_quantize(q_bf16)
    k_cache_pseudo_fp8 = fp8_pseudo_quantize(k_cache_bf16)
    v_cache_pseudo_fp8 = fp8_pseudo_quantize(v_cache_bf16)
    print("  Q, K, V quantized to FP8 and dequantized back to BF16\n")

    # Baseline: Naive BF16 attention with FP8 pseudo-quantized inputs
    print("Step 2: Run baseline - Naive BF16 attention")
    sm_scale = 1.0 / (head_dim**0.5)
    output_baseline = naive_attention_decode_bf16(
        q_pseudo_fp8, k_cache_pseudo_fp8, v_cache_pseudo_fp8, sm_scale=sm_scale
    )
    print(f"  Baseline output shape: {output_baseline.shape}")
    print(
        f"  Baseline output stats: mean={output_baseline.float().mean():.6f}, "
        f"std={output_baseline.float().std():.6f}, "
        f"min={output_baseline.float().min():.6f}, "
        f"max={output_baseline.float().max():.6f}\n"
    )

    # Test: trtllm_batch_decode_with_kv_cache with FP8
    print("Step 3: Run test - trtllm_batch_decode_with_kv_cache with FP8")

    # Prepare paged KV cache
    kv_cache_paged, block_tables = prepare_paged_kv_cache(
        k_cache_pseudo_fp8,
        v_cache_pseudo_fp8,
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
    )
    print(f"  Paged KV cache shape: {kv_cache_paged.shape}")
    print(f"  Block tables shape: {block_tables.shape}")

    # Convert to FP8
    q_fp8 = q_pseudo_fp8.to(torch.float8_e4m3fn)
    kv_cache_fp8 = kv_cache_paged.to(torch.float8_e4m3fn)

    # Prepare workspace buffer
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    workspace_size = 128 * 1024 * 1024  # 128MB workspace
    workspace_buffer = torch.zeros(workspace_size, dtype=torch.uint8, device=device)

    # Prepare seq_lens (refers to KV cache length)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    max_seq_len = kv_len  # max_seq_len refers to KV length

    # Calculate scales for FP8 (scale=1.0 for pseudo-quantization)
    q_scale = 1.0
    k_scale = 1.0
    v_scale = 1.0
    o_scale = 1.0
    bmm1_scale = q_scale * k_scale * sm_scale  # Q * K^T * sm_scale
    bmm2_scale = v_scale / o_scale  # P * V / o_scale

    print(
        f"  FP8 scales: q_scale={q_scale}, k_scale={k_scale}, v_scale={v_scale}, o_scale={o_scale}"
    )
    print(f"  bmm1_scale={bmm1_scale:.6f}, bmm2_scale={bmm2_scale:.6f}")

    # Run trtllm_batch_decode_with_kv_cache
    output_test = trtllm_batch_decode_with_kv_cache(
        #     query=q_pseudo_fp8,
        #     kv_cache=kv_cache_paged,
        query=q_fp8,
        kv_cache=kv_cache_fp8,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=-1,
        kv_layout="HND",
        backend="auto",
        q_len_per_req=1,
        o_scale=o_scale,
    )

    print(f"  Test output shape: {output_test.shape}")
    print(
        f"  Test output stats: mean={output_test.float().mean():.6f}, "
        f"std={output_test.float().std():.6f}, "
        f"min={output_test.float().min():.6f}, "
        f"max={output_test.float().max():.6f}\n"
    )

    # =========================================================================
    # Variant 4: trtllm_batch_decode_with_kv_cache with KVFP4QuantizeUtil
    # =========================================================================
    print(
        "\nStep 3c: Run variant 4 - trtllm_batch_decode_with_kv_cache with KVFP4QuantizeUtil"
    )
    print("  KV path: BF16 -> KVFP4 (packed uint8 + FP8 scales) -> BF16 -> FP8")
    print("  NOTE: Uses production-ready KVFP4QuantizeUtil.batched_quantize/dequantize")

    # Apply KVFP4 quantization to KV cache
    # Reshape for batched quantization [B, M, N] where B=batch_size, M=seq_len*num_kv_heads, N=head_dim
    print(f"\n  [KVFP4 Quantization Process - Using KVFP4QuantizeUtil]")

    # Process K cache
    print(f"  Quantizing K cache with KVFP4QuantizeUtil...")
    k_cache_for_kvfp4 = k_cache_bf16.reshape(
        batch_size, kv_len * num_kv_heads, head_dim
    )
    print(
        f"    K cache reshaped: {k_cache_for_kvfp4.shape} (B={batch_size}, M={kv_len * num_kv_heads}, N={head_dim})"
    )

    k_quant_packed, k_block_scales_fp8, k_global_scale = (
        KVFP4QuantizeUtil.batched_quantize(k_cache_for_kvfp4)
    )
    print(
        f"    K quantized data shape: {k_quant_packed.shape}, dtype: {k_quant_packed.dtype}"
    )
    print(
        f"    K block scales shape: {k_block_scales_fp8.shape}, dtype: {k_block_scales_fp8.dtype}"
    )
    print(f"    K global scale: {k_global_scale.item():.6f}")
    print(
        f"    K block scales range: [{k_block_scales_fp8.float().min().item():.2f}, {k_block_scales_fp8.float().max().item():.2f}]"
    )

    k_cache_kvfp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
        k_quant_packed, k_block_scales_fp8, k_global_scale, dtype=torch.bfloat16
    )
    k_cache_kvfp4_dequant = k_cache_kvfp4_dequant.reshape(
        batch_size, kv_len, num_kv_heads, head_dim
    )
    k_valid_kvfp4 = verify_fp4_quantization(
        k_cache_bf16, k_cache_kvfp4_dequant, "K cache (KVFP4)"
    )

    # Process V cache
    print(f"\n  Quantizing V cache with KVFP4QuantizeUtil...")
    v_cache_for_kvfp4 = v_cache_bf16.reshape(
        batch_size, kv_len * num_kv_heads, head_dim
    )
    print(
        f"    V cache reshaped: {v_cache_for_kvfp4.shape} (B={batch_size}, M={kv_len * num_kv_heads}, N={head_dim})"
    )

    v_quant_packed, v_block_scales_fp8, v_global_scale = (
        KVFP4QuantizeUtil.batched_quantize(v_cache_for_kvfp4)
    )
    print(
        f"    V quantized data shape: {v_quant_packed.shape}, dtype: {v_quant_packed.dtype}"
    )
    print(
        f"    V block scales shape: {v_block_scales_fp8.shape}, dtype: {v_block_scales_fp8.dtype}"
    )
    print(f"    V global scale: {v_global_scale.item():.6f}")
    print(
        f"    V block scales range: [{v_block_scales_fp8.float().min().item():.2f}, {v_block_scales_fp8.float().max().item():.2f}]"
    )

    v_cache_kvfp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
        v_quant_packed, v_block_scales_fp8, v_global_scale, dtype=torch.bfloat16
    )
    v_cache_kvfp4_dequant = v_cache_kvfp4_dequant.reshape(
        batch_size, kv_len, num_kv_heads, head_dim
    )
    v_valid_kvfp4 = verify_fp4_quantization(
        v_cache_bf16, v_cache_kvfp4_dequant, "V cache (KVFP4)"
    )

    if not k_valid_kvfp4 or not v_valid_kvfp4:
        print(f"\n  ⚠️  KVFP4 quantization quality check FAILED!")
    else:
        print(
            f"\n  ✓ KVFP4 quantization quality check PASSED (with expected precision loss)"
        )

    # Prepare paged KV cache with KVFP4-quantized data
    kv_cache_paged_kvfp4, block_tables_kvfp4 = prepare_paged_kv_cache(
        k_cache_kvfp4_dequant,
        v_cache_kvfp4_dequant,
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
    )
    print(f"  Paged KV cache (KVFP4) shape: {kv_cache_paged_kvfp4.shape}")

    # Convert to FP8
    kv_cache_fp8_kvfp4 = kv_cache_paged_kvfp4.to(torch.float8_e4m3fn)

    # Run trtllm_batch_decode_with_kv_cache with KVFP4-quantized KV
    output_test_kvfp4 = trtllm_batch_decode_with_kv_cache(
        query=q_fp8,
        kv_cache=kv_cache_fp8_kvfp4,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables_kvfp4,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=-1,
        kv_layout="HND",
        backend="auto",
        q_len_per_req=1,
        o_scale=o_scale,
    )

    print(f"  Variant 4 output shape: {output_test_kvfp4.shape}")
    print(
        f"  Variant 4 output stats: mean={output_test_kvfp4.float().mean():.6f}, "
        f"std={output_test_kvfp4.float().std():.6f}, "
        f"min={output_test_kvfp4.float().min():.6f}, "
        f"max={output_test_kvfp4.float().max():.6f}\n"
    )

    # Compare variant 4 (KVFP4) vs variant 3 (FP4 pseudo) - Output Range Comparison
    print("\n[Variant 4 (KVFP4) vs Variant 3 (FP4 Pseudo) - Output Range Comparison]")

    output_test_kvfp4_f32 = output_test_kvfp4.float()

    kvfp4_mean = output_test_kvfp4_f32.mean().item()
    kvfp4_std = output_test_kvfp4_f32.std().item()
    kvfp4_min = output_test_kvfp4_f32.min().item()
    kvfp4_max = output_test_kvfp4_f32.max().item()
    kvfp4_range = kvfp4_max - kvfp4_min

    print(
        f"  Variant 4 (KVFP4):   mean={kvfp4_mean:.6f}, std={kvfp4_std:.6f}, range=[{kvfp4_min:.6f}, {kvfp4_max:.6f}], span={kvfp4_range:.6f}"
    )

    # =========================================================================
    # Variant 5: Direct FP4 KV Cache (No Dequantization)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3d: Run variant 5 - Direct FP4 KV Cache (Native FP4 Kernel)")
    print("=" * 80)
    print("  KV path: BF16 -> KVFP4 (packed uint8 + FP8 scales) -> Direct FP4 Kernel")
    print("  NOTE: No dequantization! Directly pass FP4 quantized KV to kernel")
    print("  Scale fusion: k_global_scale -> bmm1_scale, v_global_scale -> bmm2_scale")
    print("  Scale adjustment: block_scale /= 6, global_scale *= 6 (for FP8 compute)")

    # Reuse quantization results from Variant 4
    print(f"\n  [Using Quantized Data from Variant 4]")
    print(
        f"    K quantized data shape: {k_quant_packed.shape}, dtype: {k_quant_packed.dtype}"
    )
    print(
        f"    K block scales shape: {k_block_scales_fp8.shape}, dtype: {k_block_scales_fp8.dtype}"
    )
    print(f"    K global scale (original): {k_global_scale.item():.6f}")
    print(
        f"    V quantized data shape: {v_quant_packed.shape}, dtype: {v_quant_packed.dtype}"
    )
    print(
        f"    V block scales shape: {v_block_scales_fp8.shape}, dtype: {v_block_scales_fp8.dtype}"
    )
    print(f"    V global scale (original): {v_global_scale.item():.6f}")

    # Step 1: Adjust scales (block_scale /= 6, global_scale *= 6)
    print(f"\n  [Step 1: Adjust Scales for FP8 Compute]")
    print(f"    Adjustment: block_scale /= 6, global_scale *= 6")
    print(f"    Reason: Kernel uses FP8 compute, need to prevent overflow")

    # Adjust K scales
    k_block_scales_adjusted = (k_block_scales_fp8.float() / 6.0).to(torch.float8_e4m3fn)
    k_global_scale_adjusted = k_global_scale * 6.0

    print(
        f"    K block scales adjusted range: [{k_block_scales_adjusted.float().min().item():.6f}, {k_block_scales_adjusted.float().max().item():.6f}]"
    )
    print(f"    K global scale adjusted: {k_global_scale_adjusted.item():.6f}")

    # Adjust V scales
    v_block_scales_adjusted = (v_block_scales_fp8.float() / 6.0).to(torch.float8_e4m3fn)
    v_global_scale_adjusted = v_global_scale * 6.0

    print(
        f"    V block scales adjusted range: [{v_block_scales_adjusted.float().min().item():.6f}, {v_block_scales_adjusted.float().max().item():.6f}]"
    )
    print(f"    V global scale adjusted: {v_global_scale_adjusted.item():.6f}")

    # Step 2: Prepare paged FP4 KV cache (uint8 packed format)
    print(f"\n  [Step 2: Prepare Paged FP4 KV Cache]")

    # Reshape K: [batch_size, seq_len*num_kv_heads, head_dim] -> [batch_size, seq_len, num_kv_heads, head_dim]
    # -> paged format
    k_quant_reshaped = k_quant_packed.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 2
    )
    v_quant_reshaped = v_quant_packed.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 2
    )

    print(f"    K quant reshaped: {k_quant_reshaped.shape}")
    print(f"    V quant reshaped: {v_quant_reshaped.shape}")

    # Create paged KV cache for FP4 data
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # KV cache: [num_pages, num_kv_heads, page_size, head_dim/2] dtype=uint8
    k_cache_fp4_paged = torch.zeros(
        total_num_pages,
        num_kv_heads,
        page_size,
        head_dim // 2,
        dtype=torch.uint8,
        device=device,
    )
    v_cache_fp4_paged = torch.zeros(
        total_num_pages,
        num_kv_heads,
        page_size,
        head_dim // 2,
        dtype=torch.uint8,
        device=device,
    )

    # Fill paged KV cache
    for b in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            page_id = b * num_pages_per_seq + page_idx
            start_idx = page_idx * page_size
            end_idx = min(start_idx + page_size, kv_len)
            valid_len = end_idx - start_idx

            # Transpose: [valid_len, num_kv_heads, head_dim/2] -> [num_kv_heads, valid_len, head_dim/2]
            k_cache_fp4_paged[page_id, :, :valid_len, :] = k_quant_reshaped[
                b, start_idx:end_idx, :, :
            ].transpose(0, 1)
            v_cache_fp4_paged[page_id, :, :valid_len, :] = v_quant_reshaped[
                b, start_idx:end_idx, :, :
            ].transpose(0, 1)
    kv_cache_fp4_paged = (k_cache_fp4_paged, v_cache_fp4_paged)

    print(
        f"    Paged FP4 KV cache shape: {k_cache_fp4_paged.shape}, dtype: {k_cache_fp4_paged.dtype}"
    )

    # Step 3: Prepare paged block scales
    print(f"\n  [Step 3: Prepare Paged Block Scales]")

    # Reshape block scales: [batch_size, seq_len*num_kv_heads*head_dim/16] -> [batch_size, seq_len, num_kv_heads, head_dim/16]
    k_block_scales_reshaped = k_block_scales_adjusted.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 16
    )
    v_block_scales_reshaped = v_block_scales_adjusted.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 16
    )

    print(f"    K block scales reshaped: {k_block_scales_reshaped.shape}")
    print(f"    V block scales reshaped: {v_block_scales_reshaped.shape}")

    # Create paged block scales: [num_pages, num_kv_heads, page_size, head_dim/16]
    k_block_scales_paged = torch.zeros(
        total_num_pages,
        num_kv_heads,
        page_size,
        head_dim // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    v_block_scales_paged = torch.zeros(
        total_num_pages,
        num_kv_heads,
        page_size,
        head_dim // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    # Fill paged block scales
    for b in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            page_id = b * num_pages_per_seq + page_idx
            start_idx = page_idx * page_size
            end_idx = min(start_idx + page_size, kv_len)
            valid_len = end_idx - start_idx

            # Transpose: [valid_len, num_kv_heads, head_dim/16] -> [num_kv_heads, valid_len, head_dim/16]
            k_block_scales_paged[page_id, :, :valid_len, :] = k_block_scales_reshaped[
                b, start_idx:end_idx, :, :
            ].transpose(0, 1)
            v_block_scales_paged[page_id, :, :valid_len, :] = v_block_scales_reshaped[
                b, start_idx:end_idx, :, :
            ].transpose(0, 1)
    kv_block_scales_paged = (k_block_scales_paged, v_block_scales_paged)
    print(
        f"    Paged block scales shape: {k_block_scales_paged.shape}, dtype: {k_block_scales_paged.dtype}"
    )

    # Step 4: Fuse global scales into bmm1_scale and bmm2_scale
    print(f"\n  [Step 4: Fuse Global Scales into BMM Scales]")
    print(f"    Original bmm1_scale: {bmm1_scale:.6f} (q_scale * k_scale * sm_scale)")
    print(f"    Original bmm2_scale: {bmm2_scale:.6f} (v_scale / o_scale)")

    # bmm1 = Q @ K^T, so fuse k_global_scale into bmm1_scale
    bmm1_scale_fp4 = bmm1_scale * k_global_scale_adjusted.item()

    # bmm2 = P @ V, so fuse v_global_scale into bmm2_scale
    bmm2_scale_fp4 = bmm2_scale * v_global_scale_adjusted.item()

    print(f"    Fused bmm1_scale (with K global): {bmm1_scale_fp4:.6f}")
    print(f"    Fused bmm2_scale (with V global): {bmm2_scale_fp4:.6f}")

    # Step 5: Create block tables (same as before)
    block_tables_fp4 = torch.arange(
        0, total_num_pages, dtype=torch.int32, device=device
    ).view(batch_size, num_pages_per_seq)
    print(f"    Block tables shape: {block_tables_fp4.shape}")

    # Step 6: Call trtllm_batch_decode_with_kv_cache with FP4 KV cache
    print(f"\n  [Step 5: Call Kernel with FP4 KV Cache]")
    print(f"    Note: is_nvfp4_kvcache will be auto-detected by the interface")

    output_test_fp4_native = trtllm_batch_decode_with_kv_cache(
        query=q_fp8,
        kv_cache=kv_cache_fp4_paged,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables_fp4,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale_fp4,
        bmm2_scale=bmm2_scale_fp4,
        window_left=-1,
        kv_layout="HND",
        backend="auto",
        q_len_per_req=1,
        o_scale=o_scale,
        kv_block_scales=kv_block_scales_paged,
    )

    print(f"  Variant 5 output shape: {output_test_fp4_native.shape}")
    print(
        f"  Variant 5 output stats: mean={output_test_fp4_native.float().mean():.6f}, "
        f"std={output_test_fp4_native.float().std():.6f}, "
        f"min={output_test_fp4_native.float().min():.6f}, "
        f"max={output_test_fp4_native.float().max():.6f}\n"
    )

    # Compare variant 5 (Native FP4) vs variant 4 (Dequantized FP4)
    print(
        "\n[Variant 5 (Native FP4) vs Variant 4 (Dequantized FP4) - Output Comparison]"
    )

    output_test_fp4_native_f32 = output_test_fp4_native.float()

    fp4_native_mean = output_test_fp4_native_f32.mean().item()
    fp4_native_std = output_test_fp4_native_f32.std().item()
    fp4_native_min = output_test_fp4_native_f32.min().item()
    fp4_native_max = output_test_fp4_native_f32.max().item()
    fp4_native_range = fp4_native_max - fp4_native_min

    print(
        f"  Variant 4 (Dequant):  mean={kvfp4_mean:.6f}, std={kvfp4_std:.6f}, range=[{kvfp4_min:.6f}, {kvfp4_max:.6f}], span={kvfp4_range:.6f}"
    )
    print(
        f"  Variant 5 (Native):   mean={fp4_native_mean:.6f}, std={fp4_native_std:.6f}, range=[{fp4_native_min:.6f}, {fp4_native_max:.6f}], span={fp4_native_range:.6f}"
    )

    # Calculate errors
    abs_diff_native_vs_dequant = torch.abs(
        output_test_fp4_native_f32 - output_test_kvfp4_f32
    )
    mean_abs_error_native = abs_diff_native_vs_dequant.mean().item()
    max_abs_error_native = abs_diff_native_vs_dequant.max().item()

    ss_res_native = torch.sum(
        (output_test_fp4_native_f32 - output_test_kvfp4_f32) ** 2
    ).item()
    ss_tot_native = torch.sum(
        (output_test_kvfp4_f32 - output_test_kvfp4_f32.mean()) ** 2
    ).item()
    r2_score_native = 1 - (ss_res_native / (ss_tot_native + 1e-10))

    print(f"\n  Variant 5 (Native) vs Variant 4 (Dequant):")
    print(f"    Mean absolute error: {mean_abs_error_native:.6f}")
    print(f"    Max absolute error: {max_abs_error_native:.6f}")
    print(f"    R² score: {r2_score_native:.6f} (1.0 = perfect match)")


    # Summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY: Comparison of All Variants")
    print("=" * 80)
    print(
        f"\n{'Variant':<35} {'Mean':<12} {'Std':<12} {'Range':<12} {'MAE vs Baseline':<18}"
    )
    print("-" * 89)
    print(
        f"{'Variant 4 (KVFP4 Dequant)':<35} {kvfp4_mean:<12.6f} {kvfp4_std:<12.6f} {kvfp4_range:<12.6f} {'-':<18}"
    )
    print(
        f"{'Variant 5 (KVFP4 Native)':<35} {fp4_native_mean:<12.6f} {fp4_native_std:<12.6f} {fp4_native_range:<12.6f} {'-':<18}"
    )

    print(f"\n{'Comparison':<35} {'MAE':<15} {'Max AE':<15} {'R² Score':<15}")
    print("-" * 80)
    print(
        f"{'V5 (KVFP4 Native) vs V4 (Dequant)':<35} {mean_abs_error_native:<15.6f} {max_abs_error_native:<15.6f} {r2_score_native:<15.6f}"
    )

    print("\n" + "=" * 80)

    return True


if __name__ == "__main__":
    success = test_trtllm_batch_decode_fp8()
    exit(0 if success else 1)
