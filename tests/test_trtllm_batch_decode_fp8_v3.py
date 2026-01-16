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
from kv_fp4_pseudo_quantizer import quantize_kv_cache_fp4_pseudo, dequantize_kv_cache_fp4_pseudo
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
        sm_scale = 1.0 / (head_dim ** 0.5)

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
    k_transposed = k_expanded.transpose(1, 2).transpose(2, 3)  # [batch_size, num_qo_heads, head_dim, seq_len]
    scores = torch.matmul(q_expanded, k_transposed).squeeze(2) * sm_scale  # [batch_size, num_qo_heads, seq_len]

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_qo_heads, seq_len]

    # Apply attention to values
    # attn_weights: [batch_size, num_qo_heads, seq_len]
    # v_expanded: [batch_size, seq_len, num_qo_heads, head_dim]
    v_transposed = v_expanded.transpose(1, 2)  # [batch_size, num_qo_heads, seq_len, head_dim]
    output = torch.matmul(attn_weights.unsqueeze(2), v_transposed).squeeze(2)  # [batch_size, num_qo_heads, head_dim]

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

    print(f"    Original:  mean={orig_mean:+.6f}, std={orig_std:.6f}, range=[{orig_min:+.6f}, {orig_max:+.6f}]")
    print(f"    Quantized: mean={quant_mean:+.6f}, std={quant_std:.6f}, range=[{quant_min:+.6f}, {quant_max:+.6f}]")

    # 3. Error metrics
    abs_error = (original_f32 - quantized_f32).abs()
    rel_error = abs_error / (original_f32.abs() + 1e-8)

    print(f"    Abs error: mean={abs_error.mean().item():.6f}, max={abs_error.max().item():.6f}")
    print(f"    Rel error: mean={rel_error.mean().item():.6f}, max={rel_error.max().item():.6f}")

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
        print(f"      [{i}] orig={orig_val:+.6f}, quant={quant_val:+.6f}, error={error:.6f}")

    # 6. Value distribution check (histogram)
    # Count how many values fall into different ranges
    def count_in_range(tensor, low, high):
        return ((tensor >= low) & (tensor < high)).sum().item()

    total = original_f32.numel()
    ranges = [
        (-float('inf'), -1.0),
        (-1.0, -0.1),
        (-0.1, -0.01),
        (-0.01, 0.01),
        (0.01, 0.1),
        (0.1, 1.0),
        (1.0, float('inf'))
    ]

    print(f"    Value distribution (% of total):")
    for low, high in ranges:
        orig_count = count_in_range(original_f32, low, high)
        quant_count = count_in_range(quantized_f32, low, high)
        range_str = f"[{low:+.2f}, {high:+.2f})"
        print(f"      {range_str:20s}: orig={orig_count*100/total:5.1f}%, quant={quant_count*100/total:5.1f}%")

    # 7. Check if std ratio is too small (indicates severe compression)
    if std_ratio < 0.1:
        print(f"    ⚠️  WARNING: Std ratio < 0.1, severe information loss!")
        return False

    return True


def fp4_pseudo_quantize_kv(k_cache_bf16, v_cache_bf16):
    """
    Apply FP4 pseudo-quantization to KV cache: BF16 -> nvFP4 -> BF16
    
    Args:
        k_cache_bf16: [batch_size, seq_len, num_kv_heads, head_dim] in BF16
        v_cache_bf16: [batch_size, seq_len, num_kv_heads, head_dim] in BF16
    
    Returns:
        k_cache_pseudo_fp4: [batch_size, seq_len, num_kv_heads, head_dim] in BF16 (after FP4 quant/dequant)
        v_cache_pseudo_fp4: [batch_size, seq_len, num_kv_heads, head_dim] in BF16 (after FP4 quant/dequant)
    """
    print(f"  [FP4 Quantization Process]")

    # Quantize and dequantize K cache
    print(f"  Quantizing K cache...")
    k_fp4_data, k_sf, k_global_sf, k_orig_shape, k_orig_dtype = quantize_kv_cache_fp4_pseudo(k_cache_bf16)
    print(f"    K global_sf: {k_global_sf.item():.6f}, max_abs: {k_cache_bf16.abs().max().item():.6f}")
    print(f"    K fp4_data shape: {k_fp4_data.shape}, sf shape: {k_sf.shape}")
    print(f"    K scale factor stats: min={k_sf.float().min().item():.2f}, max={k_sf.float().max().item():.2f}, mean={k_sf.float().mean().item():.2f}")

    k_cache_pseudo_fp4 = dequantize_kv_cache_fp4_pseudo(k_fp4_data, k_sf, k_global_sf, k_orig_shape, k_orig_dtype)
    k_valid = verify_fp4_quantization(k_cache_bf16, k_cache_pseudo_fp4, "K cache")

    # Quantize and dequantize V cache
    print(f"\n  Quantizing V cache...")
    v_fp4_data, v_sf, v_global_sf, v_orig_shape, v_orig_dtype = quantize_kv_cache_fp4_pseudo(v_cache_bf16)
    print(f"    V global_sf: {v_global_sf.item():.6f}, max_abs: {v_cache_bf16.abs().max().item():.6f}")
    print(f"    V fp4_data shape: {v_fp4_data.shape}, sf shape: {v_sf.shape}")
    print(f"    V scale factor stats: min={v_sf.float().min().item():.2f}, max={v_sf.float().max().item():.2f}, mean={v_sf.float().mean().item():.2f}")

    v_cache_pseudo_fp4 = dequantize_kv_cache_fp4_pseudo(v_fp4_data, v_sf, v_global_sf, v_orig_shape, v_orig_dtype)
    v_valid = verify_fp4_quantization(v_cache_bf16, v_cache_pseudo_fp4, "V cache")

    if not k_valid or not v_valid:
        print(f"\n  ⚠️  FP4 quantization quality check FAILED!")
    else:
        print(f"\n  ✓ FP4 quantization quality check PASSED (but with expected precision loss)")

    return k_cache_pseudo_fp4, v_cache_pseudo_fp4


def prepare_paged_kv_cache(k_cache_flat, v_cache_flat, batch_size, seq_len, page_size, num_kv_heads, head_dim):
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
        total_num_pages, 2, num_kv_heads, page_size, head_dim,
        dtype=dtype, device=device
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
            kv_cache[page_id, 0, :, :valid_len, :] = k_cache_flat[b, start_idx:end_idx, :, :].transpose(0, 1)
            kv_cache[page_id, 1, :, :valid_len, :] = v_cache_flat[b, start_idx:end_idx, :, :].transpose(0, 1)

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
    q_bf16 = torch.randn(batch_size,
                         num_qo_heads,
                         head_dim,
                         dtype=torch.bfloat16,
                         device=device)
    k_cache_bf16 = torch.randn(batch_size,
                               kv_len,
                               num_kv_heads,
                               head_dim,
                               dtype=torch.bfloat16,
                               device=device)
    v_cache_bf16 = torch.randn(batch_size,
                               kv_len,
                               num_kv_heads,
                               head_dim,
                               dtype=torch.bfloat16,
                               device=device)

    # Apply FP8 pseudo-quantization (scale=1.0)
    print("Step 1: Apply FP8 pseudo-quantization (scale=1.0)")
    q_pseudo_fp8 = fp8_pseudo_quantize(q_bf16)
    k_cache_pseudo_fp8 = fp8_pseudo_quantize(k_cache_bf16)
    v_cache_pseudo_fp8 = fp8_pseudo_quantize(v_cache_bf16)
    print("  Q, K, V quantized to FP8 and dequantized back to BF16\n")

    # Baseline: Naive BF16 attention with FP8 pseudo-quantized inputs
    print("Step 2: Run baseline - Naive BF16 attention")
    sm_scale = 1.0 / (head_dim**0.5)
    output_baseline = naive_attention_decode_bf16(q_pseudo_fp8,
                                                  k_cache_pseudo_fp8,
                                                  v_cache_pseudo_fp8,
                                                  sm_scale=sm_scale)
    print(f"  Baseline output shape: {output_baseline.shape}")
    print(
        f"  Baseline output stats: mean={output_baseline.float().mean():.6f}, "
        f"std={output_baseline.float().std():.6f}, "
        f"min={output_baseline.float().min():.6f}, "
        f"max={output_baseline.float().max():.6f}\n")

    # Test: trtllm_batch_decode_with_kv_cache with FP8
    print("Step 3: Run test - trtllm_batch_decode_with_kv_cache with FP8")

    # Prepare paged KV cache
    kv_cache_paged, block_tables = prepare_paged_kv_cache(
        k_cache_pseudo_fp8, v_cache_pseudo_fp8, batch_size, kv_len, page_size,
        num_kv_heads, head_dim)
    print(f"  Paged KV cache shape: {kv_cache_paged.shape}")
    print(f"  Block tables shape: {block_tables.shape}")

    # Convert to FP8
    q_fp8 = q_pseudo_fp8.to(torch.float8_e4m3fn)
    kv_cache_fp8 = kv_cache_paged.to(torch.float8_e4m3fn)

    # Prepare workspace buffer
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    workspace_size = 128 * 1024 * 1024  # 128MB workspace
    workspace_buffer = torch.zeros(workspace_size,
                                   dtype=torch.uint8,
                                   device=device)

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
    print(f"  Test output stats: mean={output_test.float().mean():.6f}, "
          f"std={output_test.float().std():.6f}, "
          f"min={output_test.float().min():.6f}, "
          f"max={output_test.float().max():.6f}\n")

    # Variant 3: trtllm_batch_decode_with_kv_cache with FP4-quantized KV cache
    print(
        "\nStep 3b: Run variant 3 - trtllm_batch_decode_with_kv_cache with FP4-quantized KV"
    )
    print("  KV path: BF16 -> nvFP4 -> BF16 -> FP8")
    print(
        "  NOTE: nvFP4 is an extremely lossy 4-bit format with limited precision"
    )

    # Apply FP4 pseudo-quantization to KV cache with detailed verification
    k_cache_pseudo_fp4, v_cache_pseudo_fp4 = fp4_pseudo_quantize_kv(
        k_cache_bf16, v_cache_bf16)

    # Prepare paged KV cache with FP4-quantized data
    kv_cache_paged_fp4, block_tables_fp4 = prepare_paged_kv_cache(
        k_cache_pseudo_fp4, v_cache_pseudo_fp4, batch_size, kv_len, page_size,
        num_kv_heads, head_dim)
    print(f"  Paged KV cache (FP4) shape: {kv_cache_paged_fp4.shape}")

    # Convert to FP8
    kv_cache_fp8_fp4 = kv_cache_paged_fp4.to(torch.float8_e4m3fn)

    # Run trtllm_batch_decode_with_kv_cache with FP4-quantized KV
    output_test_fp4 = trtllm_batch_decode_with_kv_cache(
        query=q_fp8,
        kv_cache=kv_cache_fp8_fp4,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables_fp4,
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

    print(f"  Variant 3 output shape: {output_test_fp4.shape}")
    print(
        f"  Variant 3 output stats: mean={output_test_fp4.float().mean():.6f}, "
        f"std={output_test_fp4.float().std():.6f}, "
        f"min={output_test_fp4.float().min():.6f}, "
        f"max={output_test_fp4.float().max():.6f}\n")

    # Compare results
    print("Step 4: Compare results")
    output_baseline_f32 = output_baseline.float()
    output_test_f32 = output_test.float()
    output_test_fp4_f32 = output_test_fp4.float()

    # Compare variant 2 (FP8) vs baseline (BF16)
    print("\n[Variant 2 (FP8) vs Baseline (BF16)]")
    abs_diff = torch.abs(output_test_f32 - output_baseline_f32)

    mean_abs_error = abs_diff.mean().item()
    max_abs_error = abs_diff.max().item()

    # Calculate R² score (coefficient of determination)
    ss_res = torch.sum((output_test_f32 - output_baseline_f32)**2).item()
    ss_tot = torch.sum(
        (output_baseline_f32 - output_baseline_f32.mean())**2).item()
    r2_score = 1 - (ss_res / (ss_tot + 1e-10))

    print(f"  Mean absolute error: {mean_abs_error:.6f}")
    print(f"  Max absolute error: {max_abs_error:.6f}")
    print(f"  R² score: {r2_score:.6f} (1.0 = perfect match)")

    # Compare variant 3 (FP4+FP8) vs variant 2 (FP8) - Focus on output range
    print(
        "\n[Variant 3 (FP4+FP8) vs Variant 2 (FP8) - Output Range Comparison]")

    # Baseline/FP8 output stats
    baseline_mean = output_baseline_f32.mean().item()
    baseline_std = output_baseline_f32.std().item()
    baseline_min = output_baseline_f32.min().item()
    baseline_max = output_baseline_f32.max().item()
    baseline_range = baseline_max - baseline_min

    fp8_mean = output_test_f32.mean().item()
    fp8_std = output_test_f32.std().item()
    fp8_min = output_test_f32.min().item()
    fp8_max = output_test_f32.max().item()
    fp8_range = fp8_max - fp8_min

    fp4_mean = output_test_fp4_f32.mean().item()
    fp4_std = output_test_fp4_f32.std().item()
    fp4_min = output_test_fp4_f32.min().item()
    fp4_max = output_test_fp4_f32.max().item()
    fp4_range = fp4_max - fp4_min

    print(
        f"  Baseline (BF16):  mean={baseline_mean:.6f}, std={baseline_std:.6f}, range=[{baseline_min:.6f}, {baseline_max:.6f}], span={baseline_range:.6f}"
    )
    print(
        f"  Variant 2 (FP8):  mean={fp8_mean:.6f}, std={fp8_std:.6f}, range=[{fp8_min:.6f}, {fp8_max:.6f}], span={fp8_range:.6f}"
    )
    print(
        f"  Variant 3 (FP4):  mean={fp4_mean:.6f}, std={fp4_std:.6f}, range=[{fp4_min:.6f}, {fp4_max:.6f}], span={fp4_range:.6f}"
    )

    # Calculate R² score for Variant 3 (FP4+FP8) vs Variant 2 (FP8)
    abs_diff_fp4_vs_fp8 = torch.abs(output_test_fp4_f32 - output_test_f32)
    mean_abs_error_fp4 = abs_diff_fp4_vs_fp8.mean().item()
    max_abs_error_fp4 = abs_diff_fp4_vs_fp8.max().item()

    ss_res_fp4 = torch.sum((output_test_fp4_f32 - output_test_f32)**2).item()
    ss_tot_fp4 = torch.sum(
        (output_test_f32 - output_test_f32.mean())**2).item()
    r2_score_fp4 = 1 - (ss_res_fp4 / (ss_tot_fp4 + 1e-10))

    print(f"\n  Variant 3 vs Variant 2:")
    print(f"    Mean absolute error: {mean_abs_error_fp4:.6f}")
    print(f"    Max absolute error: {max_abs_error_fp4:.6f}")
    print(f"    R² score: {r2_score_fp4:.6f} (1.0 = perfect match)")

    # =========================================================================
    # Variant 4: trtllm_batch_decode_with_kv_cache with KVFP4QuantizeUtil
    # =========================================================================
    print(
        "\nStep 3c: Run variant 4 - trtllm_batch_decode_with_kv_cache with KVFP4QuantizeUtil"
    )
    print(
        "  KV path: BF16 -> KVFP4 (packed uint8 + FP8 scales) -> BF16 -> FP8")
    print(
        "  NOTE: Uses production-ready KVFP4QuantizeUtil.batched_quantize/dequantize"
    )

    # Apply KVFP4 quantization to KV cache
    # Reshape for batched quantization [B, M, N] where B=batch_size, M=seq_len*num_kv_heads, N=head_dim
    print(f"\n  [KVFP4 Quantization Process - Using KVFP4QuantizeUtil]")

    # Process K cache
    print(f"  Quantizing K cache with KVFP4QuantizeUtil...")
    k_cache_for_kvfp4 = k_cache_bf16.reshape(batch_size, kv_len * num_kv_heads,
                                             head_dim)
    print(
        f"    K cache reshaped: {k_cache_for_kvfp4.shape} (B={batch_size}, M={kv_len * num_kv_heads}, N={head_dim})"
    )

    k_quant_packed, k_block_scales_fp8, k_global_scale = KVFP4QuantizeUtil.batched_quantize(
        k_cache_for_kvfp4)
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
        k_quant_packed,
        k_block_scales_fp8,
        k_global_scale,
        dtype=torch.bfloat16)
    k_cache_kvfp4_dequant = k_cache_kvfp4_dequant.reshape(
        batch_size, kv_len, num_kv_heads, head_dim)
    k_valid_kvfp4 = verify_fp4_quantization(k_cache_bf16,
                                            k_cache_kvfp4_dequant,
                                            "K cache (KVFP4)")

    # Process V cache
    print(f"\n  Quantizing V cache with KVFP4QuantizeUtil...")
    v_cache_for_kvfp4 = v_cache_bf16.reshape(batch_size, kv_len * num_kv_heads,
                                             head_dim)
    print(
        f"    V cache reshaped: {v_cache_for_kvfp4.shape} (B={batch_size}, M={kv_len * num_kv_heads}, N={head_dim})"
    )

    v_quant_packed, v_block_scales_fp8, v_global_scale = KVFP4QuantizeUtil.batched_quantize(
        v_cache_for_kvfp4)
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
        v_quant_packed,
        v_block_scales_fp8,
        v_global_scale,
        dtype=torch.bfloat16)
    v_cache_kvfp4_dequant = v_cache_kvfp4_dequant.reshape(
        batch_size, kv_len, num_kv_heads, head_dim)
    v_valid_kvfp4 = verify_fp4_quantization(v_cache_bf16,
                                            v_cache_kvfp4_dequant,
                                            "V cache (KVFP4)")

    if not k_valid_kvfp4 or not v_valid_kvfp4:
        print(f"\n  ⚠️  KVFP4 quantization quality check FAILED!")
    else:
        print(
            f"\n  ✓ KVFP4 quantization quality check PASSED (with expected precision loss)"
        )

    # Prepare paged KV cache with KVFP4-quantized data
    kv_cache_paged_kvfp4, block_tables_kvfp4 = prepare_paged_kv_cache(
        k_cache_kvfp4_dequant, v_cache_kvfp4_dequant, batch_size, kv_len,
        page_size, num_kv_heads, head_dim)
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
        f"max={output_test_kvfp4.float().max():.6f}\n")

    # Compare variant 4 (KVFP4) vs variant 3 (FP4 pseudo) - Output Range Comparison
    print(
        "\n[Variant 4 (KVFP4) vs Variant 3 (FP4 Pseudo) - Output Range Comparison]"
    )

    output_test_kvfp4_f32 = output_test_kvfp4.float()

    kvfp4_mean = output_test_kvfp4_f32.mean().item()
    kvfp4_std = output_test_kvfp4_f32.std().item()
    kvfp4_min = output_test_kvfp4_f32.min().item()
    kvfp4_max = output_test_kvfp4_f32.max().item()
    kvfp4_range = kvfp4_max - kvfp4_min

    print(
        f"  Baseline (BF16):     mean={baseline_mean:.6f}, std={baseline_std:.6f}, range=[{baseline_min:.6f}, {baseline_max:.6f}], span={baseline_range:.6f}"
    )
    print(
        f"  Variant 2 (FP8):     mean={fp8_mean:.6f}, std={fp8_std:.6f}, range=[{fp8_min:.6f}, {fp8_max:.6f}], span={fp8_range:.6f}"
    )
    print(
        f"  Variant 3 (FP4):     mean={fp4_mean:.6f}, std={fp4_std:.6f}, range=[{fp4_min:.6f}, {fp4_max:.6f}], span={fp4_range:.6f}"
    )
    print(
        f"  Variant 4 (KVFP4):   mean={kvfp4_mean:.6f}, std={kvfp4_std:.6f}, range=[{kvfp4_min:.6f}, {kvfp4_max:.6f}], span={kvfp4_range:.6f}"
    )

    # Calculate R² score and errors for Variant 4 vs Variant 3
    abs_diff_kvfp4_vs_fp4 = torch.abs(output_test_kvfp4_f32 -
                                      output_test_fp4_f32)
    mean_abs_error_kvfp4 = abs_diff_kvfp4_vs_fp4.mean().item()
    max_abs_error_kvfp4 = abs_diff_kvfp4_vs_fp4.max().item()

    ss_res_kvfp4 = torch.sum(
        (output_test_kvfp4_f32 - output_test_fp4_f32)**2).item()
    ss_tot_kvfp4 = torch.sum(
        (output_test_fp4_f32 - output_test_fp4_f32.mean())**2).item()
    r2_score_kvfp4 = 1 - (ss_res_kvfp4 / (ss_tot_kvfp4 + 1e-10))

    print(f"\n  Variant 4 (KVFP4) vs Variant 3 (FP4 Pseudo):")
    print(f"    Mean absolute error: {mean_abs_error_kvfp4:.6f}")
    print(f"    Max absolute error: {max_abs_error_kvfp4:.6f}")
    print(f"    R² score: {r2_score_kvfp4:.6f} (1.0 = perfect match)")

    # Calculate R² score for Variant 4 vs Variant 2 (for reference)
    abs_diff_kvfp4_vs_fp8 = torch.abs(output_test_kvfp4_f32 - output_test_f32)
    mean_abs_error_kvfp4_vs_fp8 = abs_diff_kvfp4_vs_fp8.mean().item()
    max_abs_error_kvfp4_vs_fp8 = abs_diff_kvfp4_vs_fp8.max().item()

    ss_res_kvfp4_vs_fp8 = torch.sum(
        (output_test_kvfp4_f32 - output_test_f32)**2).item()
    ss_tot_kvfp4_vs_fp8 = torch.sum(
        (output_test_f32 - output_test_f32.mean())**2).item()
    r2_score_kvfp4_vs_fp8 = 1 - (ss_res_kvfp4_vs_fp8 /
                                 (ss_tot_kvfp4_vs_fp8 + 1e-10))

    print(f"\n  Variant 4 (KVFP4) vs V2 (FP8):")
    print(f"    Mean absolute error: {mean_abs_error_kvfp4_vs_fp8:.6f}")
    print(f"    Max absolute error: {max_abs_error_kvfp4_vs_fp8:.6f}")
    print(f"    R² score: {r2_score_kvfp4_vs_fp8:.6f} (1.0 = perfect match)")









    # =========================================================================
    # Variant 5: Direct FP4 KV Cache (No Dequantization)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3d: Run variant 5 - Direct FP4 KV Cache (Native FP4 Kernel)")
    print("=" * 80)
    print(
        "  KV path: BF16 -> KVFP4 (packed uint8 + FP8 scales) -> Direct FP4 Kernel"
    )
    print(
        "  NOTE: No dequantization! Directly pass FP4 quantized KV to kernel")
    print(
        "  Scale fusion: k_global_scale -> bmm1_scale, v_global_scale -> bmm2_scale"
    )
    print(
        "  Scale adjustment: block_scale /= 6, global_scale *= 6 (for FP8 compute)"
    )

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
    k_block_scales_adjusted = (k_block_scales_fp8.float() / 6.0).to(
        torch.float8_e4m3fn)
    k_global_scale_adjusted = k_global_scale * 6.0

    print(
        f"    K block scales adjusted range: [{k_block_scales_adjusted.float().min().item():.6f}, {k_block_scales_adjusted.float().max().item():.6f}]"
    )
    print(f"    K global scale adjusted: {k_global_scale_adjusted.item():.6f}")

    # Adjust V scales
    v_block_scales_adjusted = (v_block_scales_fp8.float() / 6.0).to(
        torch.float8_e4m3fn)
    v_global_scale_adjusted = v_global_scale * 6.0

    print(
        f"    V block scales adjusted range: [{v_block_scales_adjusted.float().min().item():.6f}, {v_block_scales_adjusted.float().max().item():.6f}]"
    )
    print(f"    V global scale adjusted: {v_global_scale_adjusted.item():.6f}")

    # Step 2: Prepare paged FP4 KV cache (uint8 packed format)
    print(f"\n  [Step 2: Prepare Paged FP4 KV Cache]")

    # Reshape K: [batch_size, seq_len*num_kv_heads, head_dim] -> [batch_size, seq_len, num_kv_heads, head_dim]
    # -> paged format
    k_quant_reshaped = k_quant_packed.reshape(batch_size, kv_len, num_kv_heads,
                                              head_dim // 2)
    v_quant_reshaped = v_quant_packed.reshape(batch_size, kv_len, num_kv_heads,
                                              head_dim // 2)

    print(f"    K quant reshaped: {k_quant_reshaped.shape}")
    print(f"    V quant reshaped: {v_quant_reshaped.shape}")

    # Create paged KV cache for FP4 data
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # KV cache: [num_pages, num_kv_heads, page_size, head_dim/2] dtype=uint8
    k_cache_fp4_paged = torch.zeros(total_num_pages,
                                    num_kv_heads,
                                    page_size,
                                    head_dim // 2,
                                    dtype=torch.uint8,
                                    device=device)
    v_cache_fp4_paged = torch.zeros(total_num_pages,
                                    num_kv_heads,
                                    page_size,
                                    head_dim // 2,
                                    dtype=torch.uint8,
                                    device=device)

    # Fill paged KV cache
    for b in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            page_id = b * num_pages_per_seq + page_idx
            start_idx = page_idx * page_size
            end_idx = min(start_idx + page_size, kv_len)
            valid_len = end_idx - start_idx

            # Transpose: [valid_len, num_kv_heads, head_dim/2] -> [num_kv_heads, valid_len, head_dim/2]
            k_cache_fp4_paged[page_id, :, :valid_len, :] = k_quant_reshaped[
                b, start_idx:end_idx, :, :].transpose(0, 1)
            v_cache_fp4_paged[page_id, :, :valid_len, :] = v_quant_reshaped[
                b, start_idx:end_idx, :, :].transpose(0, 1)
    kv_cache_fp4_paged = (k_cache_fp4_paged, v_cache_fp4_paged)

    print(
        f"    Paged FP4 KV cache shape: {k_cache_fp4_paged.shape}, dtype: {k_cache_fp4_paged.dtype}"
    )

    # Step 3: Prepare paged block scales
    print(f"\n  [Step 3: Prepare Paged Block Scales]")

    # Reshape block scales: [batch_size, seq_len*num_kv_heads*head_dim/16] -> [batch_size, seq_len, num_kv_heads, head_dim/16]
    k_block_scales_reshaped = k_block_scales_adjusted.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 16)
    v_block_scales_reshaped = v_block_scales_adjusted.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 16)

    print(f"    K block scales reshaped: {k_block_scales_reshaped.shape}")
    print(f"    V block scales reshaped: {v_block_scales_reshaped.shape}")

    # Create paged block scales: [num_pages, num_kv_heads, page_size, head_dim/16]
    k_block_scales_paged = torch.zeros(total_num_pages,
                                       num_kv_heads,
                                       page_size,
                                       head_dim // 16,
                                       dtype=torch.float8_e4m3fn,
                                       device=device)
    v_block_scales_paged = torch.zeros(total_num_pages,
                                       num_kv_heads,
                                       page_size,
                                       head_dim // 16,
                                       dtype=torch.float8_e4m3fn,
                                       device=device)
    # Fill paged block scales
    for b in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            page_id = b * num_pages_per_seq + page_idx
            start_idx = page_idx * page_size
            end_idx = min(start_idx + page_size, kv_len)
            valid_len = end_idx - start_idx

            # Transpose: [valid_len, num_kv_heads, head_dim/16] -> [num_kv_heads, valid_len, head_dim/16]
            k_block_scales_paged[
                page_id, :, :valid_len, :] = k_block_scales_reshaped[
                    b, start_idx:end_idx, :, :].transpose(0, 1)
            v_block_scales_paged[
                page_id, :, :valid_len, :] = v_block_scales_reshaped[
                    b, start_idx:end_idx, :, :].transpose(0, 1)
    kv_block_scales_paged = (k_block_scales_paged, v_block_scales_paged)
    print(
        f"    Paged block scales shape: {k_block_scales_paged.shape}, dtype: {k_block_scales_paged.dtype}"
    )

    # Step 4: Fuse global scales into bmm1_scale and bmm2_scale
    print(f"\n  [Step 4: Fuse Global Scales into BMM Scales]")
    print(
        f"    Original bmm1_scale: {bmm1_scale:.6f} (q_scale * k_scale * sm_scale)"
    )
    print(f"    Original bmm2_scale: {bmm2_scale:.6f} (v_scale / o_scale)")

    # bmm1 = Q @ K^T, so fuse k_global_scale into bmm1_scale
    bmm1_scale_fp4 = bmm1_scale * k_global_scale_adjusted.item()

    # bmm2 = P @ V, so fuse v_global_scale into bmm2_scale
    bmm2_scale_fp4 = bmm2_scale * v_global_scale_adjusted.item()

    print(f"    Fused bmm1_scale (with K global): {bmm1_scale_fp4:.6f}")
    print(f"    Fused bmm2_scale (with V global): {bmm2_scale_fp4:.6f}")

    # Step 5: Create block tables (same as before)
    block_tables_fp4 = torch.arange(0,
                                    total_num_pages,
                                    dtype=torch.int32,
                                    device=device).view(
                                        batch_size, num_pages_per_seq)
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
        f"max={output_test_fp4_native.float().max():.6f}\n")

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
    abs_diff_native_vs_dequant = torch.abs(output_test_fp4_native_f32 -
                                           output_test_kvfp4_f32)
    mean_abs_error_native = abs_diff_native_vs_dequant.mean().item()
    max_abs_error_native = abs_diff_native_vs_dequant.max().item()

    ss_res_native = torch.sum(
        (output_test_fp4_native_f32 - output_test_kvfp4_f32)**2).item()
    ss_tot_native = torch.sum(
        (output_test_kvfp4_f32 - output_test_kvfp4_f32.mean())**2).item()
    r2_score_native = 1 - (ss_res_native / (ss_tot_native + 1e-10))

    print(f"\n  Variant 5 (Native) vs Variant 4 (Dequant):")
    print(f"    Mean absolute error: {mean_abs_error_native:.6f}")
    print(f"    Max absolute error: {max_abs_error_native:.6f}")
    print(f"    R² score: {r2_score_native:.6f} (1.0 = perfect match)")

    # Compare variant 5 vs variant 2 (FP8 baseline)
    abs_diff_native_vs_fp8 = torch.abs(output_test_fp4_native_f32 -
                                       output_test_f32)
    mean_abs_error_native_vs_fp8 = abs_diff_native_vs_fp8.mean().item()
    max_abs_error_native_vs_fp8 = abs_diff_native_vs_fp8.max().item()

    ss_res_native_vs_fp8 = torch.sum(
        (output_test_fp4_native_f32 - output_test_f32)**2).item()
    ss_tot_native_vs_fp8 = torch.sum(
        (output_test_f32 - output_test_f32.mean())**2).item()
    r2_score_native_vs_fp8 = 1 - (ss_res_native_vs_fp8 /
                                  (ss_tot_native_vs_fp8 + 1e-10))

    print(f"\n  Variant 5 (Native) vs Variant 2 (FP8):")
    print(f"    Mean absolute error: {mean_abs_error_native_vs_fp8:.6f}")
    print(f"    Max absolute error: {max_abs_error_native_vs_fp8:.6f}")
    print(f"    R² score: {r2_score_native_vs_fp8:.6f} (1.0 = perfect match)")

    # Summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY: Comparison of All Variants")
    print("=" * 80)
    print(
        f"\n{'Variant':<35} {'Mean':<12} {'Std':<12} {'Range':<12} {'MAE vs Baseline':<18}"
    )
    print("-" * 89)
    print(
        f"{'Baseline (BF16)':<35} {baseline_mean:<12.6f} {baseline_std:<12.6f} {baseline_range:<12.6f} {'N/A':<18}"
    )
    print(
        f"{'Variant 2 (FP8)':<35} {fp8_mean:<12.6f} {fp8_std:<12.6f} {fp8_range:<12.6f} {mean_abs_error:<18.6f}"
    )
    print(
        f"{'Variant 3 (FP4 Pseudo)':<35} {fp4_mean:<12.6f} {fp4_std:<12.6f} {fp4_range:<12.6f} {'-':<18}"
    )
    print(
        f"{'Variant 4 (KVFP4 Dequant)':<35} {kvfp4_mean:<12.6f} {kvfp4_std:<12.6f} {kvfp4_range:<12.6f} {'-':<18}"
    )
    print(
        f"{'Variant 5 (KVFP4 Native)':<35} {fp4_native_mean:<12.6f} {fp4_native_std:<12.6f} {fp4_native_range:<12.6f} {'-':<18}"
    )

    print(f"\n{'Comparison':<35} {'MAE':<15} {'Max AE':<15} {'R² Score':<15}")
    print("-" * 80)
    print(
        f"{'V2 (FP8) vs Baseline':<35} {mean_abs_error:<15.6f} {max_abs_error:<15.6f} {r2_score:<15.6f}"
    )
    print(
        f"{'V3 (FP4) vs V2 (FP8)':<35} {mean_abs_error_fp4:<15.6f} {max_abs_error_fp4:<15.6f} {r2_score_fp4:<15.6f}"
    )
    print(
        f"{'V4 (KVFP4 Dequant) vs V3 (FP4)':<35} {mean_abs_error_kvfp4:<15.6f} {max_abs_error_kvfp4:<15.6f} {r2_score_kvfp4:<15.6f}"
    )
    print(
        f"{'V4 (KVFP4 Dequant) vs V2 (FP8)':<35} {mean_abs_error_kvfp4_vs_fp8:<15.6f} {max_abs_error_kvfp4_vs_fp8:<15.6f} {r2_score_kvfp4_vs_fp8:<15.6f}"
    )
    print(
        f"{'V5 (KVFP4 Native) vs V4 (Dequant)':<35} {mean_abs_error_native:<15.6f} {max_abs_error_native:<15.6f} {r2_score_native:<15.6f}"
    )
    print(
        f"{'V5 (KVFP4 Native) vs V2 (FP8)':<35} {mean_abs_error_native_vs_fp8:<15.6f} {max_abs_error_native_vs_fp8:<15.6f} {r2_score_native_vs_fp8:<15.6f}"
    )








    # =========================================================================
    # Variant 6: Direct FP4 KV Cache (No Dequantization)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" Run variant 6 - Direct FP4 KV Cache (Native FP4 Kernel) + KV Merge Cache")
    print("=" * 80)

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
    k_block_scales_adjusted = (k_block_scales_fp8.float() / 6.0).to(
        torch.float8_e4m3fn)
    k_global_scale_adjusted = k_global_scale * 6.0

    # Adjust V scales
    v_block_scales_adjusted = (v_block_scales_fp8.float() / 6.0).to(
        torch.float8_e4m3fn)
    v_global_scale_adjusted = v_global_scale * 6.0

    # Reshape K: [batch_size, seq_len*num_kv_heads, head_dim] -> [batch_size, seq_len, num_kv_heads, head_dim]
    # -> paged format
    k_quant_reshaped = k_quant_packed.reshape(batch_size, kv_len, num_kv_heads,
                                              head_dim // 2)
    v_quant_reshaped = v_quant_packed.reshape(batch_size, kv_len, num_kv_heads,
                                              head_dim // 2)

    print(f"    K quant reshaped: {k_quant_reshaped.shape}")

    # Create paged KV cache for FP4 data
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # KV cache: [num_pages, 2, num_kv_heads, page_size, head_dim/2] dtype=uint8
    kv_cache_fp4_paged = torch.zeros(total_num_pages,
                                    2,
                                    num_kv_heads,
                                    page_size,
                                    head_dim // 2,
                                    dtype=torch.uint8,
                                    device=device)

    # Fill paged KV cache
    for b in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            page_id = b * num_pages_per_seq + page_idx
            start_idx = page_idx * page_size
            end_idx = min(start_idx + page_size, kv_len)
            valid_len = end_idx - start_idx

            # Transpose: [valid_len, num_kv_heads, head_dim/2] -> [num_kv_heads, valid_len, head_dim/2]
            kv_cache_fp4_paged[page_id, 0, :, :valid_len, :] = k_quant_reshaped[
                b, start_idx:end_idx, :, :].transpose(0, 1)
            kv_cache_fp4_paged[page_id, 1, :, :valid_len, :] = v_quant_reshaped[
                b, start_idx:end_idx, :, :].transpose(0, 1)

    print(
        f"    Paged FP4 KV cache shape: {kv_cache_fp4_paged.shape}, dtype: {kv_cache_fp4_paged.dtype}"
    )

    # Step 3: Prepare paged block scales
    print(f"\n  [Step 3: Prepare Paged Block Scales]")

    # Reshape block scales: [batch_size, seq_len*num_kv_heads*head_dim/16] -> [batch_size, seq_len, num_kv_heads, head_dim/16]
    k_block_scales_reshaped = k_block_scales_adjusted.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 16)
    v_block_scales_reshaped = v_block_scales_adjusted.reshape(
        batch_size, kv_len, num_kv_heads, head_dim // 16)

    print(f"    K block scales reshaped: {k_block_scales_reshaped.shape}")
    print(f"    V block scales reshaped: {v_block_scales_reshaped.shape}")


    # Create paged block scales: [num_pages, num_kv_heads, page_size, head_dim/16]
    k_block_scales_paged = torch.zeros(total_num_pages,
                                       num_kv_heads,
                                       page_size,
                                       head_dim // 16,
                                       dtype=torch.float8_e4m3fn,
                                       device=device)
    v_block_scales_paged = torch.zeros(total_num_pages,
                                       num_kv_heads,
                                       page_size,
                                       head_dim // 16,
                                       dtype=torch.float8_e4m3fn,
                                       device=device)
    # Fill paged block scales
    # for b in range(batch_size):
    #     for page_idx in range(num_pages_per_seq):
    #         page_id = b * num_pages_per_seq + page_idx
    #         start_idx = page_idx * page_size
    #         end_idx = min(start_idx + page_size, kv_len)
    #         valid_len = end_idx - start_idx

    #         # Transpose: [valid_len, num_kv_heads, head_dim/16] -> [num_kv_heads, valid_len, head_dim/16]
    #         k_block_scales_paged[
    #             page_id, :, :valid_len, :] = k_block_scales_reshaped[
    #                 b, start_idx:end_idx, :, :].transpose(0, 1)
    #         v_block_scales_paged[
    #             page_id, :, :valid_len, :] = v_block_scales_reshaped[
    #                 b, start_idx:end_idx, :, :].transpose(0, 1)
    # kv_block_scales_paged = (k_block_scales_paged, v_block_scales_paged)
    # print(
    #     f"    Paged block scales shape: {k_block_scales_paged.shape}, dtype: {k_block_scales_paged.dtype}"
    # )

    # Create paged block scales: [num_pages, num_kv_heads, page_size, head_dim/16]
    kv_block_scales_paged = torch.zeros(total_num_pages,
                                        2, 
                                       num_kv_heads,
                                       page_size,
                                       head_dim // 16,
                                       dtype=torch.float8_e4m3fn,
                                       device=device)
    # Fill paged block scales
    for b in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            page_id = b * num_pages_per_seq + page_idx
            start_idx = page_idx * page_size
            end_idx = min(start_idx + page_size, kv_len)
            valid_len = end_idx - start_idx

            # Transpose: [valid_len, num_kv_heads, head_dim/16] -> [num_kv_heads, valid_len, head_dim/16]
            kv_block_scales_paged[
                page_id, 0, :, :valid_len, :] = k_block_scales_reshaped[
                    b, start_idx:end_idx, :, :].transpose(0, 1)
            kv_block_scales_paged[
                page_id, 1, :, :valid_len, :] = v_block_scales_reshaped[
                    b, start_idx:end_idx, :, :].transpose(0, 1)
    print(
        f"    Paged block scales shape: {kv_block_scales_paged.shape}, dtype: {kv_block_scales_paged.dtype}"
    )

    # Step 4: Fuse global scales into bmm1_scale and bmm2_scale
    print(f"\n  [Step 4: Fuse Global Scales into BMM Scales]")
    print(
        f"    Original bmm1_scale: {bmm1_scale:.6f} (q_scale * k_scale * sm_scale)"
    )
    print(f"    Original bmm2_scale: {bmm2_scale:.6f} (v_scale / o_scale)")

    # bmm1 = Q @ K^T, so fuse k_global_scale into bmm1_scale
    bmm1_scale_fp4 = bmm1_scale * k_global_scale_adjusted.item()

    # bmm2 = P @ V, so fuse v_global_scale into bmm2_scale
    bmm2_scale_fp4 = bmm2_scale * v_global_scale_adjusted.item()

    print(f"    Fused bmm1_scale (with K global): {bmm1_scale_fp4:.6f}")
    print(f"    Fused bmm2_scale (with V global): {bmm2_scale_fp4:.6f}")

    # Step 5: Create block tables (same as before)
    block_tables_fp4 = torch.arange(0,
                                    total_num_pages,
                                    dtype=torch.int32,
                                    device=device).view(
                                        batch_size, num_pages_per_seq)
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

    print(f"  Variant 6 output shape: {output_test_fp4_native.shape}")
    print(
        f"  Variant 65 output stats: mean={output_test_fp4_native.float().mean():.6f}, "
        f"std={output_test_fp4_native.float().std():.6f}, "
        f"min={output_test_fp4_native.float().min():.6f}, "
        f"max={output_test_fp4_native.float().max():.6f}\n")

    # Compare variant 6 (Native FP4) vs variant 4 (Dequantized FP4)
    print(
        "\n[Variant 6 (Native FP4) vs Variant 4 (Dequantized FP4) - Output Comparison]"
    )

    output_test_fp4_native_f32 = output_test_fp4_native.float()

    fp4_native_mean = output_test_fp4_native_f32.mean().item()
    fp4_native_std = output_test_fp4_native_f32.std().item()
    fp4_native_min = output_test_fp4_native_f32.min().item()
    fp4_native_max = output_test_fp4_native_f32.max().item()
    fp4_native_range = fp4_native_max - fp4_native_min


    # Calculate errors
    abs_diff_native_vs_dequant = torch.abs(output_test_fp4_native_f32 -
                                           output_test_kvfp4_f32)
    mean_abs_error_native = abs_diff_native_vs_dequant.mean().item()
    max_abs_error_native = abs_diff_native_vs_dequant.max().item()

    ss_res_native = torch.sum(
        (output_test_fp4_native_f32 - output_test_kvfp4_f32)**2).item()
    ss_tot_native = torch.sum(
        (output_test_kvfp4_f32 - output_test_kvfp4_f32.mean())**2).item()
    r2_score_native = 1 - (ss_res_native / (ss_tot_native + 1e-10))

    print(f"\n  Variant 6 (Native) vs Variant 4 (Dequant):")
    print(f"    Mean absolute error: {mean_abs_error_native:.6f}")
    print(f"    Max absolute error: {max_abs_error_native:.6f}")
    print(f"    R² score: {r2_score_native:.6f} (1.0 = perfect match)")

    # Compare variant 5 vs variant 2 (FP8 baseline)
    abs_diff_native_vs_fp8 = torch.abs(output_test_fp4_native_f32 -
                                       output_test_f32)
    mean_abs_error_native_vs_fp8 = abs_diff_native_vs_fp8.mean().item()
    max_abs_error_native_vs_fp8 = abs_diff_native_vs_fp8.max().item()

    ss_res_native_vs_fp8 = torch.sum(
        (output_test_fp4_native_f32 - output_test_f32)**2).item()
    ss_tot_native_vs_fp8 = torch.sum(
        (output_test_f32 - output_test_f32.mean())**2).item()
    r2_score_native_vs_fp8 = 1 - (ss_res_native_vs_fp8 /
                                  (ss_tot_native_vs_fp8 + 1e-10))
    
    # Summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY: Comparison of All Variants")
    print("=" * 80)
    print(
        f"{'V6 (KVFP4 Native+kv_merge) vs V4 (Dequant)':<35} {mean_abs_error_native:<15.6f} {max_abs_error_native:<15.6f} {r2_score_native:<15.6f}"
    )
    print(
        f"{'V6 (KVFP4 Native+kv_merge) vs V2 (FP8)':<35} {mean_abs_error_native_vs_fp8:<15.6f} {max_abs_error_native_vs_fp8:<15.6f} {r2_score_native_vs_fp8:<15.6f}"
    )




    print("\n" + "=" * 80)

    # # =========================================================================
    # # Diagnostic Tests: V6 and V7 with Controlled Values
    # # =========================================================================
    # print("\n" + "="*80)
    # print("DIAGNOSTIC TESTS: V6 and V7 with Controlled Values")
    # print("="*80)
    # print("Goal: Isolate whether the issue is in KVFP4QuantizeUtil.batched_quantize")
    # print("      or in the kernel parameter passing")
    # print("\nStrategy:")
    # print("  - Use specific, controlled values for KV cache, scales")
    # print("  - V6: FP8/BF16 pseudo-quantization (ground truth)")
    # print("  - V7: Explicit FP4 binary representation (test native kernel)")
    # print("  - Compare V6 and V7 outputs to identify the issue")

    # # =========================================================================
    # # Step 1: Define controlled test values
    # # =========================================================================
    # print("\n[Step 1: Define Controlled Test Values]")

    # # Use simple values that are exactly representable in FP4/FP8
    # # nvFP4 format: 1 sign bit, 2 exp bits, 1 mantissa bit
    # # Representable values: 0, ±0.5, ±1, ±2, ±4, ±6 (and subnormals)

    # # For KV cache: use constant value 1.0 (exactly representable)
    # kv_test_value_bf16 = 1.0

    # # For FP4 quantized data: value 1.0 in nvFP4 = 0b0100 (sign=0, exp=10, mant=0)
    # # Packed as uint8: two FP4 values per byte, so 0b00100010 = 0x22
    # kv_fp4_packed_value = 0x22  # Two 1.0 values packed

    # # For block scales: use 1.0 (no scaling)
    # block_scale_value = 1.0

    # # For global scales: use 2.0 (simple scaling factor)
    # global_scale_value = 2.0

    # print(f"  Test configuration:")
    # print(f"    KV cache value (BF16): {kv_test_value_bf16}")
    # print(f"    KV FP4 packed value (uint8): 0x{kv_fp4_packed_value:02x}")
    # print(f"    Block scale value: {block_scale_value}")
    # print(f"    Global scale value: {global_scale_value}")

    # # =========================================================================
    # # Step 2: V6 - FP8/BF16 Pseudo-quantization (Ground Truth)
    # # =========================================================================
    # print("\n[Step 2: V6 - FP8/BF16 Pseudo-quantization (Ground Truth)]")
    # print("  Path: Controlled BF16 -> FP8 -> Attention")

    # # Create KV cache with constant value
    # k_cache_v6 = torch.full(
    #     (batch_size, kv_len, num_kv_heads, head_dim),
    #     kv_test_value_bf16, dtype=torch.bfloat16, device=device
    # )
    # v_cache_v6 = torch.full(
    #     (batch_size, kv_len, num_kv_heads, head_dim),
    #     kv_test_value_bf16, dtype=torch.bfloat16, device=device
    # )

    # # Apply global scale manually: KV_scaled = KV * global_scale
    # k_cache_v6_scaled = k_cache_v6 * global_scale_value
    # v_cache_v6_scaled = v_cache_v6 * global_scale_value

    # print(f"  K/V cache shape: {k_cache_v6.shape}")
    # print(f"  K/V cache value (before global scale): {k_cache_v6[0, 0, 0, 0].item():.6f}")
    # print(f"  K/V cache value (after global scale): {k_cache_v6_scaled[0, 0, 0, 0].item():.6f}")

    # # Prepare paged KV cache
    # kv_cache_paged_v6, block_tables_v6 = prepare_paged_kv_cache(
    #     k_cache_v6_scaled, v_cache_v6_scaled,
    #     batch_size, kv_len, page_size, num_kv_heads, head_dim
    # )

    # # Convert to FP8
    # kv_cache_fp8_v6 = kv_cache_paged_v6.to(torch.float8_e4m3fn)

    # print(f"  Paged KV cache (FP8) shape: {kv_cache_fp8_v6.shape}")

    # # Calculate scales for V6
    # # NOTE: KV cache already has global_scale applied in BF16 stage (k_cache_v6_scaled = k_cache_v6 * global_scale_value)
    # # So bmm scales should NOT include global_scale_value again
    # # bmm1 = Q @ K^T * sm_scale
    # bmm1_scale_v6 = q_scale * k_scale * sm_scale

    # # bmm2 = P @ V / o_scale
    # bmm2_scale_v6 = v_scale / o_scale

    # print(f"  bmm1_scale (no global_scale, already in KV): {bmm1_scale_v6:.6f}")
    # print(f"  bmm2_scale (no global_scale, already in KV): {bmm2_scale_v6:.6f}")

    # # Run V6
    # output_v6 = trtllm_batch_decode_with_kv_cache(
    #     query=q_fp8,
    #     kv_cache=kv_cache_fp8_v6,
    #     workspace_buffer=workspace_buffer,
    #     block_tables=block_tables_v6,
    #     seq_lens=seq_lens,
    #     max_seq_len=max_seq_len,
    #     bmm1_scale=bmm1_scale_v6,
    #     bmm2_scale=bmm2_scale_v6,
    #     window_left=-1,
    #     kv_layout="HND",
    #     backend="auto",
    #     q_len_per_req=1,
    #     o_scale=o_scale,
    # )

    # print(f"  V6 output shape: {output_v6.shape}")
    # print(f"  V6 output stats: mean={output_v6.float().mean():.6f}, "
    #       f"std={output_v6.float().std():.6f}, "
    #       f"min={output_v6.float().min():.6f}, "
    #       f"max={output_v6.float().max():.6f}")

    # # =========================================================================
    # # Step 3: V7 - Native FP4 Kernel with Explicit Binary Representation
    # # =========================================================================
    # print("\n[Step 3: V7 - Native FP4 Kernel with Explicit Binary Representation]")
    # print("  Path: Explicit FP4 uint8 -> Native FP4 Kernel -> Attention")

    # # Create FP4 packed KV cache with explicit binary representation
    # # Each uint8 contains two FP4 values (4 bits each)
    # # Value 1.0 in nvFP4 = 0b0100, so two 1.0s = 0x44
    # kv_cache_fp4_explicit = torch.full(
    #     (total_num_pages, 2, num_kv_heads, page_size, head_dim // 2),
    #     kv_fp4_packed_value, dtype=torch.uint8, device=device
    # )

    # print(f"  FP4 KV cache shape: {kv_cache_fp4_explicit.shape}, dtype: {kv_cache_fp4_explicit.dtype}")
    # print(f"  FP4 packed value (uint8): 0x{kv_cache_fp4_explicit[0, 0, 0, 0, 0].item():02x}")
    # print(f"  FP4 interpretation: two values of 1.0 (0b0100 + 0b0100)")

    # # Create block scales with constant value
    # # Note: We need to apply the scale adjustment (block_scale /= 6, global_scale *= 6)
    # block_scale_adjusted = block_scale_value / 6.0
    # global_scale_adjusted_v7 = global_scale_value * 6.0

    # kv_block_scales_explicit = torch.full(
    #     (total_num_pages, 2, num_kv_heads, page_size, head_dim // 16),
    #     block_scale_adjusted, dtype=torch.float8_e4m3fn, device=device
    # )

    # print(f"  Block scales shape: {kv_block_scales_explicit.shape}, dtype: {kv_block_scales_explicit.dtype}")
    # print(f"  Block scale value (adjusted): {kv_block_scales_explicit[0, 0, 0, 0, 0].float().item():.6f}")
    # print(f"  Global scale (adjusted): {global_scale_adjusted_v7:.6f}")

    # # Calculate scales for V7 (fuse global scales)
    # bmm1_scale_v7 = bmm1_scale * global_scale_adjusted_v7
    # bmm2_scale_v7 = bmm2_scale * global_scale_adjusted_v7

    # print(f"  bmm1_scale (fused): {bmm1_scale_v7:.6f}")
    # print(f"  bmm2_scale (fused): {bmm2_scale_v7:.6f}")

    # # Run V7
    # output_v7 = trtllm_batch_decode_with_kv_cache(
    #     query=q_fp8,
    #     kv_cache=kv_cache_fp4_explicit,
    #     workspace_buffer=workspace_buffer,
    #     block_tables=block_tables_fp4,
    #     seq_lens=seq_lens,
    #     max_seq_len=max_seq_len,
    #     bmm1_scale=bmm1_scale_v7,
    #     bmm2_scale=bmm2_scale_v7,
    #     window_left=-1,
    #     kv_layout="HND",
    #     backend="auto",
    #     q_len_per_req=1,
    #     o_scale=o_scale,
    #     kv_block_scales=kv_block_scales_explicit,
    # )

    # print(f"  V7 output shape: {output_v7.shape}")
    # print(f"  V7 output stats: mean={output_v7.float().mean():.6f}, "
    #       f"std={output_v7.float().std():.6f}, "
    #       f"min={output_v7.float().min():.6f}, "
    #       f"max={output_v7.float().max():.6f}")

    # # =========================================================================
    # # Step 4: Compare V6 and V7
    # # =========================================================================
    # print("\n[Step 4: Compare V6 and V7]")

    # output_v6_f32 = output_v6.float()
    # output_v7_f32 = output_v7.float()

    # v6_mean = output_v6_f32.mean().item()
    # v6_std = output_v6_f32.std().item()
    # v6_min = output_v6_f32.min().item()
    # v6_max = output_v6_f32.max().item()

    # v7_mean = output_v7_f32.mean().item()
    # v7_std = output_v7_f32.std().item()
    # v7_min = output_v7_f32.min().item()
    # v7_max = output_v7_f32.max().item()

    # print(f"  V6 (Pseudo FP8):  mean={v6_mean:.6f}, std={v6_std:.6f}, range=[{v6_min:.6f}, {v6_max:.6f}]")
    # print(f"  V7 (Native FP4):  mean={v7_mean:.6f}, std={v7_std:.6f}, range=[{v7_min:.6f}, {v7_max:.6f}]")

    # # Calculate errors
    # abs_diff_v7_v6 = torch.abs(output_v7_f32 - output_v6_f32)
    # mean_abs_error_v7 = abs_diff_v7_v6.mean().item()
    # max_abs_error_v7 = abs_diff_v7_v6.max().item()
    # rel_error_v7 = abs_diff_v7_v6 / (output_v6_f32.abs() + 1e-8)
    # mean_rel_error_v7 = rel_error_v7.mean().item()

    # ss_res_v7 = torch.sum((output_v7_f32 - output_v6_f32) ** 2).item()
    # ss_tot_v7 = torch.sum((output_v6_f32 - output_v6_f32.mean()) ** 2).item()
    # r2_score_v7 = 1 - (ss_res_v7 / (ss_tot_v7 + 1e-10))

    # print(f"\n  V7 (Native FP4) vs V6 (Pseudo FP8):")
    # print(f"    Mean absolute error: {mean_abs_error_v7:.6f}")
    # print(f"    Max absolute error: {max_abs_error_v7:.6f}")
    # print(f"    Mean relative error: {mean_rel_error_v7:.6f}")
    # print(f"    R² score: {r2_score_v7:.6f} (1.0 = perfect match)")

    # # Diagnostic conclusion
    # print("\n[Diagnostic Conclusion]")
    # if r2_score_v7 > 0.99:
    #     print("  ✓ PASS: V6 and V7 outputs match closely (R² > 0.99)")
    #     print("  → The FP4 format and kernel are working correctly")
    #     print("  → Issue likely in KVFP4QuantizeUtil.batched_quantize")
    # elif r2_score_v7 > 0.9:
    #     print("  ⚠ PARTIAL: V6 and V7 outputs have moderate match (R² > 0.9)")
    #     print("  → Minor issues in either quantization or kernel")
    #     print("  → Need further investigation")
    # else:
    #     print("  ✗ FAIL: V6 and V7 outputs differ significantly (R² < 0.9)")
    #     print("  → Major issue in FP4 format or kernel parameter passing")
    #     print("  → Check kernel implementation and scale calculations")

    # # Additional diagnostics: Show sample values
    # print("\n  Sample output values (first 5 elements of first batch, first head):")
    # print(f"    {'Index':<8} {'V6 (Pseudo)':<15} {'V7 (Native)':<15} {'Abs Diff':<15} {'Rel Diff':<15}")
    # print("    " + "-"*68)
    # for i in range(min(5, head_dim)):
    #     v6_val = output_v6_f32[0, 0, i].item()
    #     v7_val = output_v7_f32[0, 0, i].item()
    #     abs_diff = abs(v6_val - v7_val)
    #     rel_diff = abs_diff / (abs(v6_val) + 1e-8)
    #     print(f"    {i:<8} {v6_val:<15.6f} {v7_val:<15.6f} {abs_diff:<15.6f} {rel_diff:<15.6f}")

    print("\n" + "=" * 80)

    return True


if __name__ == "__main__":
    success = test_trtllm_batch_decode_fp8()
    exit(0 if success else 1)
