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

"""Pure-PyTorch NVFP4 KV cache quantization utility.

This module provides a reference (non-kernel) implementation of NVFP4
quantization with two-level scaling (global FP32 + per-block FP8).
It works on any GPU and is used for testing and benchmarking.

For the production GPU kernel-based quantizer (SM100 only), see
:func:`flashinfer.nvfp4_batched_quantize`.
"""

import torch

E2M1_MAX = 6.0
MAX_BLOCK_SCALE_FP8 = 448.0  # Maximum FP8 E4M3 value

# E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits
# 16 possible values: 0x0-0xF
# Negative values: 0x8-0xF (sign bit = 1)
# Positive values: 0x0-0x7 (sign bit = 0)
_E2M1_VALUES_CPU = torch.tensor(
    [
        0,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        6,
        -0,
        -0.5,
        -1,
        -1.5,
        -2,
        -3,
        -4,
        -6,
    ],
    dtype=torch.float32,
)
# Boundaries for rounding to nearest E2M1 value (only for positive values)
_E2M1_BOUNDS_CPU = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5],
    dtype=torch.float32,
)

# Per-device cache to avoid repeated .to() calls
_e2m1_values_cache: dict[torch.device, torch.Tensor] = {}
_e2m1_bounds_cache: dict[torch.device, torch.Tensor] = {}


def _get_e2m1_values(device: torch.device) -> torch.Tensor:
    if device not in _e2m1_values_cache:
        _e2m1_values_cache[device] = _E2M1_VALUES_CPU.to(device)
    return _e2m1_values_cache[device]


def _get_e2m1_bounds(device: torch.device) -> torch.Tensor:
    if device not in _e2m1_bounds_cache:
        _e2m1_bounds_cache[device] = _E2M1_BOUNDS_CPU.to(device)
    return _e2m1_bounds_cache[device]


class KVFP4QuantizeUtil:
    """Utility class for NVFP4 quantization and dequantization with two-level scaling (global FP32 + block FP8)."""

    @staticmethod
    def batched_quantize(
        tensor: torch.Tensor, global_scale: torch.Tensor | None = None
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
                dtype=torch.float32,
                device=device,
            )
        else:
            # Use provided global scale
            if not isinstance(global_scale, torch.Tensor):
                global_scale = torch.tensor(
                    global_scale, dtype=torch.float32, device=device
                )
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
        # E2M1 format: bit 3 = sign, bits 2-0 = magnitude (exponent + mantissa)
        sign_bits = (x_scaled < 0).to(torch.uint8) << 3  # bit 3: sign bit
        abs_vals = x_scaled.abs()
        # Find nearest E2M1 magnitude (0-7) using boundaries
        magnitude_bits = torch.sum(
            abs_vals.unsqueeze(-1) >= _get_e2m1_bounds(device), dim=-1
        ).to(torch.uint8)
        # Combine sign and magnitude: 4-bit value = sign_bit | magnitude
        fp4_vals = sign_bits | magnitude_bits
        # Pack two FP4 values into one uint8
        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, block_scales_fp8, global_scale

    @staticmethod
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
        float_vals = _get_e2m1_values(quant_tensor.device)[fp4_vals.long()]

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m, n // 16, 16)

        # Apply block scale factors (inverse scaling: divide by FP8 block scales)
        # Convert FP8 back to float32 for computation
        block_scales_float = block_scales.float().unsqueeze(-1)  # [B, M*N/16, 1]
        scaled = reshaped * block_scales_float

        # Apply inverse global scaling
        dequantized = scaled.view(b, m, n) * global_scale

        return dequantized.to(dtype)

    @staticmethod
    def quantize_paged_kv_cache(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        float,
        float,
    ]:
        """Quantize paged KV cache (HND layout) to NVFP4 format with scale adjustment.

        Takes BF16 K/V caches in HND layout, quantizes to NVFP4 with two-level scaling,
        and applies the /6 block-scale and *6 global-scale adjustment for FP8 compute.

        Args:
            k_cache: Key cache, shape [num_pages, num_kv_heads, page_size, head_dim].
            v_cache: Value cache, shape [num_pages, num_kv_heads, page_size, head_dim].

        Returns:
            kv_cache_fp4: Tuple of (k_fp4, v_fp4), each
                [num_pages, num_kv_heads, page_size, head_dim//2] dtype=uint8.
            kv_block_scales: Tuple of (k_scales, v_scales), each
                [num_pages, num_kv_heads, page_size, head_dim//16] dtype=float8_e4m3fn.
            k_global_scale: Adjusted global scale for K (float).
            v_global_scale: Adjusted global scale for V (float).
        """
        num_pages, num_kv_heads, page_size, head_dim = k_cache.shape

        k_flat = k_cache.reshape(num_pages, num_kv_heads * page_size, head_dim)
        v_flat = v_cache.reshape(num_pages, num_kv_heads * page_size, head_dim)

        k_packed, k_blk_scales, k_gs = KVFP4QuantizeUtil.batched_quantize(k_flat)
        v_packed, v_blk_scales, v_gs = KVFP4QuantizeUtil.batched_quantize(v_flat)

        # Adjust scales for FP8 compute: block_scale /= 6, global_scale *= 6
        k_blk_scales = (k_blk_scales.float() / 6.0).to(torch.float8_e4m3fn)
        v_blk_scales = (v_blk_scales.float() / 6.0).to(torch.float8_e4m3fn)

        kv_cache_fp4 = (
            k_packed.reshape(num_pages, num_kv_heads, page_size, head_dim // 2),
            v_packed.reshape(num_pages, num_kv_heads, page_size, head_dim // 2),
        )
        kv_block_scales = (
            k_blk_scales.reshape(num_pages, num_kv_heads, page_size, head_dim // 16),
            v_blk_scales.reshape(num_pages, num_kv_heads, page_size, head_dim // 16),
        )

        return kv_cache_fp4, kv_block_scales, (k_gs * 6.0).item(), (v_gs * 6.0).item()
