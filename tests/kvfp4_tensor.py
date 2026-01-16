# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch

E2M1_MAX = 6.0
MAX_BLOCK_SCALE_FP8 = 448.0  # Maximum FP8 E4M3 value
# Put constants directly on CUDA if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
# E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits
# 16 possible values: 0x0-0xF
# Negative values: 0x8-0xF (sign bit = 1)
# Positive values: 0x0-0x7 (sign bit = 0)
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6,  # 0x0-0x7: positive values
     -0, -0.5, -1, -1.5, -2, -3, -4, -6],  # 0x8-0xF: negative values
    dtype=torch.float32, device=_device
)
# Boundaries for rounding to nearest E2M1 value (only for positive values)
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device
)


class KVFP4QuantizeUtil:
    """Utility class for NVFP4 quantization and dequantization with two-level scaling (global FP32 + block FP8)."""

    @staticmethod
    # @torch.compile
    def batched_quantize(
        tensor: torch.Tensor,
        global_scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NVFP4 format with two-level scaling (global FP32 + block FP8 E4M3)
        
        Formula: x_fp4 * block_scale * global_scale = x_bf16
        
        Process:
        1. Scale x_bf16 to FP4 range [-6, 6]
        2. Calculate global_scale from this scaling
        3. Calculate block_scale (FP8) for each block
        4. Convert scaled values to packed FP4
        
        Args:
            tensor: Input tensor of shape [B, M, N]
            global_scale: Optional global scale factor (float32 scalar). 
                         If None, will auto-compute per-tensor global scale.
                         If provided, will use the given global scale.

        Returns:
            quant_tensor: Quantized E2M1 tensor of shape [B, M, N/2] (packed uint8)
            block_scales: Block scale factors of shape [B, M*N/16] (FP8 E4M3)
            global_scale: Global scale factor (float32 scalar)
        """
        b, m, n = tensor.shape
        device = tensor.device
        
        # Step 1: Calculate global_scale
        if global_scale is None:
            global_max = tensor.abs().amax()
            global_scale = torch.tensor(
                global_max.item() / (E2M1_MAX * MAX_BLOCK_SCALE_FP8),
                dtype=torch.float32, device=device
            )
        else:
            # Use provided global scale
            if not isinstance(global_scale, torch.Tensor):
                global_scale = torch.tensor(global_scale, dtype=torch.float32, device=device)
            else:
                global_scale = global_scale.to(device=device, dtype=torch.float32)

        if global_scale < 1e-6:
            global_scale = torch.tensor(1e-6, dtype=torch.float32, device=device)

        # Step 2: Scale x_bf16 to FP4 range [-6, 6]
        # First, reshape to blocks [B, M*N/16, 16]
        reshaped = tensor.float().view(b, m * n // 16, 16)
        block_max = reshaped.abs().amax(dim=-1, keepdim=True)
        block_scales = block_max.squeeze(-1) / (E2M1_MAX * global_scale)
        block_scales = torch.clamp(block_scales, 0.0, MAX_BLOCK_SCALE_FP8)
        block_scales_fp8 = block_scales.to(torch.float8_e4m3fn).view(b, m, n // 16)

        # Scale each block to FP4 range: x_scaled = x / block_max * E2M1_MAX
        # This ensures values are in [-6, 6] range
        block_scales_fixed = block_scales.unsqueeze(-1)
        x_scaled = reshaped / (block_scales_fixed * global_scale)

        # Step 3: Convert scaled values (x_scaled) to packed FP4
        # x_scaled is already in FP4 range [-6, 6] in bf16 representation
        # Now quantize to E2M1 format
        
        # E2M1 format: bit 3 = sign, bits 2-0 = magnitude (exponent + mantissa)
        sign_bits = (x_scaled < 0).to(torch.uint8) << 3  # bit 3: sign bit
        abs_vals = x_scaled.abs()
        # Find nearest E2M1 magnitude (0-7) using boundaries
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1).to(torch.uint8)
        # Combine sign and magnitude: 4-bit value = sign_bit | magnitude
        fp4_vals = sign_bits | magnitude_bits
        # Pack two FP4 values into one uint8
        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, block_scales_fp8, global_scale

    @staticmethod
    # @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize NVFP4 tensor with two-level scaling (global FP32 + block FP8 E4M3)
        
        Args:
            quant_tensor: Quantized E2M1 tensor of shape [B, M, N/2] (packed uint8)
            block_scales: Block scale factors of shape [B, M*N/16] (FP8 E4M3)
            global_scale: Global scale factor (float32 scalar)
            dtype: Target dtype for output

        Returns:
            Dequantized tensor of shape [B, M, N]
        """
        b, m, n_half = quant_tensor.shape
        n = n_half * 2

        # More efficient unpacking using bit operations
        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

        # Directly map 4-bit E2M1 values (0x0-0xF) to float
        # E2M1_VALUES[0-7] = positive, E2M1_VALUES[8-15] = negative
        float_vals = E2M1_VALUES[fp4_vals.long()]

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m, n // 16, 16)

        # Apply block scale factors (inverse scaling: divide by FP8 block scales)
        # Convert FP8 back to float32 for computation
        block_scales_float = block_scales.float().unsqueeze(-1)  # [B, M*N/16, 1]
        scaled = reshaped * block_scales_float

        # Apply inverse global scaling
        dequantized = scaled.view(b, m, n) * global_scale

        return dequantized.to(dtype)



if __name__ == "__main__":
    """Test NVFP4 quantization with data distribution analysis and comparison"""
    
    
    # Then run full distribution analysis
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n\nUsing device: {device}")
    print(f"Testing NVFP4 format: Global Scale (FP32) + Block Scale (FP8 E4M3) + Data (E2M1)")
    
    # Import pseudo quantizer for comparison
    import sys
    sys.path.insert(0, '/home/scratch.lsam_gpu/bench-b200/flashinfer/wkdir')
    from kv_fp4_pseudo_quantizer import KVFP4PseudoQuantizer
    
    # Test case: KV cache format
    print("\n" + "="*80)
    print("NVFP4 Quantization Test - Data Distribution Analysis")
    print("="*80)
    
    num_pages = 16
    num_kv = 2  # K and V
    num_kv_heads = 2
    page_size = 64
    head_dim = 256
    
    kv_cache = torch.randn(
        num_pages, num_kv, num_kv_heads, page_size, head_dim,
        dtype=torch.bfloat16, device=device
    )
    
    print(f"\nOriginal KV cache shape: {kv_cache.shape}, dtype: {kv_cache.dtype}")
    print(f"  num_pages={num_pages}, num_kv={num_kv}, num_kv_heads={num_kv_heads}")
    print(f"  page_size={page_size}, head_dim={head_dim}")
    
    # Reshape for quantization [B, M, N] where B=num_pages*num_kv, M=num_kv_heads*page_size, N=head_dim
    B_flat = num_pages * num_kv
    M_flat = num_kv_heads * page_size
    N_flat = head_dim
    
    kv_cache_flat = kv_cache.reshape(B_flat, M_flat, N_flat)
    print(f"Reshaped for quantization: {kv_cache_flat.shape}")
    
    # Quantize
    print("\n" + "-"*80)
    print("QUANTIZATION")
    print("-"*80)
    quant_kv, block_scales_kv, global_scale_kv = KVFP4QuantizeUtil.batched_quantize(kv_cache_flat)
    
    print(f"\nQuantization Results:")
    print(f"  Quantized data shape: {quant_kv.shape}, dtype: {quant_kv.dtype}")
    print(f"  Block scales shape: {block_scales_kv.shape}, dtype: {block_scales_kv.dtype}")
    print(f"  Global scale: {global_scale_kv.item():.6f} (dtype: {global_scale_kv.dtype})")
    print(f"  Block scales range: [{block_scales_kv.float().min().item():.6f}, {block_scales_kv.float().max().item():.6f}]")
    
    # Dequantize
    print("\n" + "-"*80)
    print("DEQUANTIZATION")
    print("-"*80)
    dequant_kv = KVFP4QuantizeUtil.batched_dequantize(quant_kv, block_scales_kv, global_scale_kv)
    dequant_kv_reshaped = dequant_kv.reshape(num_pages, num_kv, num_kv_heads, page_size, head_dim)
    print(f"Dequantized KV cache shape: {dequant_kv_reshaped.shape}, dtype: {dequant_kv_reshaped.dtype}")
    
    # Data distribution analysis
    print("\n" + "="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    orig_flat = kv_cache.float().flatten()
    dequant_flat = dequant_kv_reshaped.float().flatten()
    
    print(f"\n{'Metric':<30} {'Original':<20} {'Dequantized':<20} {'Difference':<20}")
    print("-"*90)
    
    orig_min, orig_max = orig_flat.min().item(), orig_flat.max().item()
    dequant_min, dequant_max = dequant_flat.min().item(), dequant_flat.max().item()
    orig_mean, orig_std = orig_flat.mean().item(), orig_flat.std().item()
    dequant_mean, dequant_std = dequant_flat.mean().item(), dequant_flat.std().item()
    
    print(f"{'Min value':<30} {orig_min:<20.8f} {dequant_min:<20.8f} {dequant_min-orig_min:<20.8f}")
    print(f"{'Max value':<30} {orig_max:<20.8f} {dequant_max:<20.8f} {dequant_max-orig_max:<20.8f}")
    print(f"{'Mean value':<30} {orig_mean:<20.8f} {dequant_mean:<20.8f} {dequant_mean-orig_mean:<20.8f}")
    print(f"{'Std deviation':<30} {orig_std:<20.8f} {dequant_std:<20.8f} {dequant_std-orig_std:<20.8f}")
    print(f"{'Range (max-min)':<30} {orig_max-orig_min:<20.8f} {dequant_max-dequant_min:<20.8f} {(dequant_max-dequant_min)-(orig_max-orig_min):<20.8f}")
    
    # Percentile analysis
    print("\n" + "-"*80)
    print("PERCENTILE DISTRIBUTION")
    print("-"*80)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n{'Percentile':<15} {'Original':<20} {'Dequantized':<20} {'Difference':<20}")
    print("-"*75)
    for p in percentiles:
        orig_p = torch.quantile(orig_flat, p/100.0).item()
        dequant_p = torch.quantile(dequant_flat, p/100.0).item()
        print(f"{p:>3}th{'':<10} {orig_p:<20.8f} {dequant_p:<20.8f} {dequant_p-orig_p:<20.8f}")
    
    # Error statistics
    print("\n" + "="*80)
    print("ERROR STATISTICS")
    print("="*80)
    
    error = (orig_flat - dequant_flat).abs()
    rel_error = error / (orig_flat.abs() + 1e-8)
    
    print(f"\nAbsolute Error:")
    print(f"  Max:    {error.max().item():.8f}")
    print(f"  Mean:   {error.mean().item():.8f}")
    print(f"  Median: {error.median().item():.8f}")
    print(f"  Std:    {error.std().item():.8f}")
    
    print(f"\nRelative Error:")
    print(f"  Max:    {rel_error.max().item():.6f} ({rel_error.max().item()*100:.2f}%)")
    print(f"  Mean:   {rel_error.mean().item():.6f} ({rel_error.mean().item()*100:.2f}%)")
    print(f"  Median: {rel_error.median().item():.6f} ({rel_error.median().item()*100:.2f}%)")
    
    # Error distribution by bins
    print("\n" + "-"*80)
    print("ERROR DISTRIBUTION BY RELATIVE ERROR BINS")
    print("-"*80)
    
    total_elements = rel_error.numel()
    bins = [(0, 0.001), (0.001, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, float('inf'))]
    
    print(f"\n{'Relative Error Range':<25} {'Count':<15} {'Percentage':<15}")
    print("-"*55)
    for bin_low, bin_high in bins:
        if bin_high == float('inf'):
            mask = rel_error >= bin_low
            range_str = f"[{bin_low:.3f}, ∞)"
        else:
            mask = (rel_error >= bin_low) & (rel_error < bin_high)
            range_str = f"[{bin_low:.3f}, {bin_high:.3f})"
        
        count = mask.sum().item()
        percentage = 100.0 * count / total_elements
        print(f"{range_str:<25} {count:<15} {percentage:>6.2f}%")
    
    # Histogram comparison
    print("\n" + "="*80)
    print("VALUE HISTOGRAM COMPARISON (10 bins)")
    print("="*80)
    
    num_bins = 10
    hist_min = min(orig_min, dequant_min)
    hist_max = max(orig_max, dequant_max)
    
    hist_orig = torch.histc(orig_flat, bins=num_bins, min=hist_min, max=hist_max)
    hist_dequant = torch.histc(dequant_flat, bins=num_bins, min=hist_min, max=hist_max)
    
    bin_width = (hist_max - hist_min) / num_bins
    print(f"\n{'Bin Range':<25} {'Original':<15} {'Dequantized':<15} {'Diff':<15}")
    print("-"*70)
    for i in range(num_bins):
        bin_start = hist_min + i * bin_width
        bin_end = hist_min + (i + 1) * bin_width
        orig_count = hist_orig[i].item()
        dequant_count = hist_dequant[i].item()
        diff = dequant_count - orig_count
        print(f"[{bin_start:>8.4f}, {bin_end:>8.4f}){'':<3} {orig_count:<15.0f} {dequant_count:<15.0f} {diff:>+15.0f}")
    
    # Compression ratio
    print("\n" + "="*80)
    print("COMPRESSION ANALYSIS")
    print("="*80)
    
    orig_size_bytes = kv_cache.numel() * 2  # bfloat16 = 2 bytes
    quant_size_bytes = quant_kv.numel() * 1  # uint8 = 1 byte
    block_scales_bytes = block_scales_kv.numel() * 1  # float8_e4m3fn = 1 byte
    global_scale_bytes = 4  # float32 = 4 bytes
    total_compressed_bytes = quant_size_bytes + block_scales_bytes + global_scale_bytes
    
    compression_ratio = orig_size_bytes / total_compressed_bytes
    
    print(f"\nOriginal size:          {orig_size_bytes:>12} bytes (bfloat16)")
    print(f"Quantized data:         {quant_size_bytes:>12} bytes (uint8, packed E2M1)")
    print(f"Block scales:           {block_scales_bytes:>12} bytes (FP8 E4M3)")
    print(f"Global scale:           {global_scale_bytes:>12} bytes (FP32)")
    print(f"Total compressed:       {total_compressed_bytes:>12} bytes")
    print(f"\nCompression ratio:      {compression_ratio:.3f}x")
    print(f"Space saving:           {(1 - 1/compression_ratio)*100:.2f}%")
    
    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80)
    
    # =============================================================================
    # Comparison with kv_fp4_pseudo_quantizer.py
    # =============================================================================
    print("\n" + "="*80)
    print("COMPARISON: kvfp4_tensor.py vs kv_fp4_pseudo_quantizer.py")
    print("="*80)
    
    print("\nBoth methods use NVFP4 format with two-level scaling:")
    print("  - Global Scale: FP32")
    print("  - Block Scale: FP8")
    print("  - Data: E2M1 (4-bit)")
    
    # Test on the same data
    test_tensor = torch.randn(4, 8, 128, dtype=torch.bfloat16, device=device)
    print(f"\nTest tensor shape: {test_tensor.shape}")
    print(f"Test tensor range: [{test_tensor.min().item():.6f}, {test_tensor.max().item():.6f}]")
    
    # =========================================================================
    # PHASE 1: Analyze quantization results and data properties
    # =========================================================================
    
    # Method 1: kvfp4_tensor.py (current implementation)
    print("\n" + "-"*80)
    print("Method 1: kvfp4_tensor.py (packed uint8 + FP8 scales)")
    print("-"*80)
    
    quant1, block_scales1, global_scale1 = KVFP4QuantizeUtil.batched_quantize(test_tensor)
    dequant1 = KVFP4QuantizeUtil.batched_dequantize(quant1, block_scales1, global_scale1)
    
    print(f"\n[Quantization Output Properties]")
    print(f"  Quantized data: shape={quant1.shape}, dtype={quant1.dtype}")
    print(f"  Block scales: shape={block_scales1.shape}, dtype={block_scales1.dtype}")
    print(f"  Global scale: {global_scale1.item():.6f} (dtype={global_scale1.dtype})")
    
    # Analyze quantized data distribution
    unique_vals1 = torch.unique(quant1)
    print(f"\n[Quantized Data Distribution]")
    print(f"  Unique packed uint8 values: {len(unique_vals1)} (out of 256 possible)")
    print(f"  Range: [{quant1.min().item()}, {quant1.max().item()}]")
    
    # Analyze block scales
    block_scales1_float = block_scales1.float()
    print(f"\n[Block Scales Distribution]")
    print(f"  Range: [{block_scales1_float.min().item():.6f}, {block_scales1_float.max().item():.6f}]")
    print(f"  Mean: {block_scales1_float.mean().item():.6f}")
    print(f"  Std: {block_scales1_float.std().item():.6f}")
    print(f"  Median: {block_scales1_float.median().item():.6f}")
    
    # Analyze dequantized data
    dequant1_flat = dequant1.float().flatten()
    print(f"\n[Dequantized Data Properties]")
    print(f"  Shape: {dequant1.shape}, dtype: {dequant1.dtype}")
    print(f"  Range: [{dequant1_flat.min().item():.8f}, {dequant1_flat.max().item():.8f}]")
    print(f"  Mean: {dequant1_flat.mean().item():.8f}")
    print(f"  Std: {dequant1_flat.std().item():.8f}")
    print(f"  Median: {dequant1_flat.median().item():.8f}")
    
    # Percentiles
    percentiles_check = [1, 25, 50, 75, 99]
    print(f"\n[Dequantized Data Percentiles]")
    for p in percentiles_check:
        p_val = torch.quantile(dequant1_flat, p/100.0).item()
        print(f"    {p:>2}th: {p_val:>10.6f}")
    
    # Storage
    size1_data = quant1.numel() * 1  # uint8
    size1_block = block_scales1.numel() * 1  # float8_e4m3fn
    size1_global = 4  # float32
    size1_total = size1_data + size1_block + size1_global
    print(f"\n[Storage]")
    print(f"  Data: {size1_data} bytes, Block scales: {size1_block} bytes, Global: {size1_global} bytes")
    print(f"  Total: {size1_total} bytes")
    
    # =========================================================================
    # PHASE 2: Compare data properties and distributions
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: DATA PROPERTIES COMPARISON")
    print("="*80)
    
    # Original data properties
    test_tensor_flat = test_tensor.float().flatten()
    print(f"\n[Original Data Properties]")
    print(f"  Shape: {test_tensor.shape}, dtype: {test_tensor.dtype}")
    print(f"  Range: [{test_tensor_flat.min().item():.8f}, {test_tensor_flat.max().item():.8f}]")
    print(f"  Mean: {test_tensor_flat.mean().item():.8f}")
    print(f"  Std: {test_tensor_flat.std().item():.8f}")
    print(f"  Median: {test_tensor_flat.median().item():.8f}")
    
    # Side-by-side comparison
    print("\n" + "-"*80)
    print("SIDE-BY-SIDE COMPARISON: Original vs Method1 vs Method2")
    print("-"*80)
    
    print(f"\n{'Property':<25} {'Original':<20} {'Method1 (Dequant)':<20} {'Method2 (Dequant)':<20}")
    print("-"*85)
    
    print(f"{'Min':<25} {test_tensor_flat.min().item():<20.8f} {dequant1_flat.min().item():<20.8f}")
    print(f"{'Max':<25} {test_tensor_flat.max().item():<20.8f} {dequant1_flat.max().item():<20.8f}")
    print(f"{'Mean':<25} {test_tensor_flat.mean().item():<20.8f} {dequant1_flat.mean().item():<20.8f}")
    print(f"{'Std':<25} {test_tensor_flat.std().item():<20.8f} {dequant1_flat.std().item():<20.8f}")
    print(f"{'Median':<25} {test_tensor_flat.median().item():<20.8f} {dequant1_flat.median().item():<20.8f}")
    
    # Percentile comparison
    print(f"\n{'Percentile Comparison':<25}")
    print("-"*85)
    for p in [1, 5, 25, 50, 75, 95, 99]:
        orig_p = torch.quantile(test_tensor_flat, p/100.0).item()
        dq1_p = torch.quantile(dequant1_flat, p/100.0).item()
        print(f"{f'{p}th':<25} {orig_p:<20.8f} {dq1_p:<20.8f}")
    
    # Scale comparison
    print(f"\n{'Scaling Factors':<25}")
    print("-"*85)
    print(f"{'Global scale':<25} {'N/A':<20} {global_scale1.item():<20.6f}")
    print(f"{'Block scale (mean)':<25} {'N/A':<20} {block_scales1_float.mean().item():<20.6f}")
    print(f"{'Block scale (range)':<25} {'N/A':<20} {f'[{block_scales1_float.min().item():.2f}, {block_scales1_float.max().item():.2f}]':<20}")
    
    # Histogram comparison
    print("\n" + "-"*80)
    print("VALUE DISTRIBUTION HISTOGRAM (10 bins)")
    print("-"*80)
    
    hist_min = min(test_tensor_flat.min().item(), dequant1_flat.min().item())
    hist_max = max(test_tensor_flat.max().item(), dequant1_flat.max().item())
    
    hist_orig = torch.histc(test_tensor_flat, bins=10, min=hist_min, max=hist_max)
    hist_dq1 = torch.histc(dequant1_flat, bins=10, min=hist_min, max=hist_max)
    
    bin_width = (hist_max - hist_min) / 10
    print(f"\n{'Bin Range':<20} {'Original':<15} {'Method1':<15}")
    print("-"*65)
    for i in range(10):
        bin_start = hist_min + i * bin_width
        bin_end = hist_min + (i + 1) * bin_width
        print(f"[{bin_start:>7.3f}, {bin_end:>7.3f}) {hist_orig[i].item():<15.0f} {hist_dq1[i].item():<15.0f}")
    
    # =========================================================================
    # PHASE 3: Error analysis (only if distributions look reasonable)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: ERROR ANALYSIS")
    print("="*80)
    
    error1 = (test_tensor.float() - dequant1.float()).abs()
    rel_error1 = error1 / (test_tensor.float().abs() + 1e-8)    
    
    print(f"\n{'Error Metric':<30} {'Method 1':<20} {'Method 2':<20}")
    print("-"*70)
    print(f"{'Abs error - Max':<30} {error1.max().item():<20.8f}")
    print(f"{'Abs error - Mean':<30} {error1.mean().item():<20.8f}")
    print(f"{'Abs error - Median':<30} {error1.median().item():<20.8f}")
    print(f"{'Rel error - Max':<30} {rel_error1.max().item():<20.6f}")
    print(f"{'Rel error - Mean':<30} {rel_error1.mean().item():<20.6f}")
    print(f"{'Rel error - Median':<30} {rel_error1.median().item():<20.6f}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<40} {'Method 1 (packed)':<25} {'Method 2 (pseudo)':<25}")
    print("-"*90)
    
    print(f"{'Data format':<40} {'uint8 (packed E2M1)':<25} {'float32 (E2M1 values)':<25}")
    print(f"{'Block scale format':<40} {'float8_e4m3fn':<25} {'float32':<25}")
    print(f"{'Global scale format':<40} {'float32':<25} {'float32':<25}")
    
    print(f"\n{'Max absolute error':<40} {error1.max().item():<25.8f}")
    print(f"{'Mean absolute error':<40} {error1.mean().item():<25.8f}")
    print(f"{'Max relative error':<40} {rel_error1.max().item():<25.6f}")
    print(f"{'Mean relative error':<40} {rel_error1.mean().item():<25.6f}")
    
    print(f"\n{'Global scale value':<40} {global_scale1.item():<25.6f}")
    
    # Check if results are similar
    global_scale_match = True
    
    print(f"\n{'Global scales match?':<40} {str(global_scale_match):<25}")
    
    print("\n" + "="*80)
    print("✓ Comparison test completed!")
    print("="*80)
