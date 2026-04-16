"""
FlashInfer-optimized implementation of WanTransformer3DModel.

This module provides an inference-optimized version of the Wan video transformer
using FlashInfer kernels for attention, normalization, and GEMM operations.

Original model: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py

Optimizations:
- RMSNorm: torch.nn.RMSNorm -> flashinfer.rmsnorm
- Attention: F.scaled_dot_product_attention -> flashinfer.single_prefill_with_kv_cache
- Activations: Fused GELU when applicable
- Linear: nn.Linear -> FlashInferLinear with mm_bf16/mm_fp8/mm_fp4 backends
- Sparse Attention: Optional skip-softmax sparse attention via trtllm_batch_context_with_kv_cache

Note: The 3D RoPE (WanRotaryPosEmbed) is kept as-is since it's specialized for video
(combining temporal, height, width frequencies) and not directly supported by FlashInfer's
standard RoPE APIs.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import flashinfer
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import get_compute_capability


class GEMMBackend(Enum):
    """GEMM backend selection for FlashInferLinear."""

    TORCH = "torch"  # Standard PyTorch (fallback, any GPU)
    BF16 = "bf16"  # FlashInfer mm_bf16 (SM100+)
    FP8 = "fp8"  # FlashInfer mm_fp8 with TRT-LLM backend (SM89+)
    FP8_SM90 = "fp8_sm90"  # FlashInfer fp8_blockscale_gemm_sm90 (SM90 only)
    BMM_FP8 = "bmm_fp8"  # FlashInfer bmm_fp8 with cublas backend (SM89+)
    FP4 = "fp4"  # FlashInfer mm_fp4 (SM100+)
    BMM_BF16 = "bmm_bf16"  # FlashInfer bmm_bf16 (SM100+)
    MXFP8 = "mxfp8"  # FlashInfer mm_mxfp8 (SM100+/SM120+)
    BMM_MXFP8 = "bmm_mxfp8"  # FlashInfer bmm_mxfp8 (SM100+/SM120+)


def _check_gemm_backend_support(backend: GEMMBackend, device: torch.device) -> bool:
    """Check if a GEMM backend is supported on the current device."""
    if backend == GEMMBackend.TORCH:
        return True

    major, minor = get_compute_capability(device)
    sm = major * 10 + minor

    if backend == GEMMBackend.BF16:
        # mm_bf16 requires SM100+ (Blackwell)
        return sm >= 100
    elif backend == GEMMBackend.FP8:
        # mm_fp8 (TRT-LLM low-latency) requires SM89+
        # Note: On SM90, prefer FP8_SM90 (fp8_blockscale_gemm_sm90) as it's more reliable
        # TRT-LLM has strict dimension constraints that may cause failures
        return sm >= 89 and sm != 90  # Disable on SM90, use FP8_SM90 instead
    elif backend == GEMMBackend.FP8_SM90:
        # fp8_blockscale_gemm_sm90 requires exactly SM90
        return sm == 90
    elif backend == GEMMBackend.BMM_FP8:
        # bmm_fp8 with cublas backend works on SM89+
        return sm >= 89
    elif backend == GEMMBackend.FP4:
        # mm_fp4 requires SM100+ (Blackwell)
        return sm >= 100
    elif backend == GEMMBackend.BMM_BF16:
        # bmm_bf16 requires SM100+ (Blackwell)
        return sm >= 100
    elif backend == GEMMBackend.MXFP8:
        # mm_mxfp8 requires SM100+ or SM120+
        return sm >= 100
    elif backend == GEMMBackend.BMM_MXFP8:
        # bmm_mxfp8 requires SM100+ or SM120+
        return sm >= 100

    return False


def _get_best_available_backend(device: torch.device) -> GEMMBackend:
    """Get the best available GEMM backend for the current device.

    Backend selection:
    - SM100+ (Blackwell): mm_bf16 (most stable)
    - SM90 (Hopper): fp8_blockscale_gemm_sm90 (optimized for Hopper)
    - SM89 (Ada): TORCH (TRT-LLM has strict constraints)
    - SM < 89: TORCH
    """
    major, minor = get_compute_capability(device)
    sm = major * 10 + minor

    if sm >= 100:
        return GEMMBackend.BF16  # Prefer BF16 on Blackwell for stability
    elif sm == 90:
        return GEMMBackend.FP8_SM90  # Use FP8 blockscale on Hopper
    else:
        return GEMMBackend.TORCH  # Fallback for other GPUs


class FlashInferLinear(nn.Module):
    """
    Linear layer using FlashInfer's optimized GEMM kernels.

    This module provides drop-in replacement for nn.Linear with support for
    multiple precision backends:
    - TORCH: Standard PyTorch (works on any GPU)
    - BF16: FlashInfer mm_bf16 with CUTLASS/cuDNN (SM100+)
    - FP8: FlashInfer mm_fp8 with TRT-LLM low-latency backend (SM89+)
    - FP4: FlashInfer mm_fp4 with block quantization (SM100+)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias (only supported with TORCH backend)
        backend: GEMM backend to use ("auto", "torch", "bf16", "fp8", "fp8_sm90", "bmm_fp8", "fp4", "bmm_bf16", "mxfp8", "bmm_mxfp8")
        device: Device to place the module on
        dtype: Data type for weights (for TORCH/BF16 backends)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend: Literal[
            "auto",
            "torch",
            "bf16",
            "fp8",
            "fp8_sm90",
            "bmm_fp8",
            "fp4",
            "bmm_bf16",
            "mxfp8",
            "bmm_mxfp8",
        ] = "auto",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Resolve device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Resolve backend
        if backend == "auto":
            self._backend = _get_best_available_backend(device)
        else:
            self._backend = GEMMBackend(backend)

        # Validate backend support
        if not _check_gemm_backend_support(self._backend, device):
            print(
                f"Warning: {self._backend.value} not supported on this device, falling back to TORCH"
            )
            self._backend = GEMMBackend.TORCH

        # Check dimension constraints for FP8 backends
        if self._backend == GEMMBackend.FP8_SM90:
            # fp8_blockscale_gemm_sm90 requires: N % 64 == 0, K % 128 == 0
            if out_features % 64 != 0 or in_features % 128 != 0:
                print(
                    f"Warning: FP8_SM90 requires N%64==0 and K%128==0, got N={out_features}, K={in_features}, falling back to TORCH"
                )
                self._backend = GEMMBackend.TORCH

        # Check dimension constraints for MXFP8 backends
        if self._backend in (GEMMBackend.MXFP8, GEMMBackend.BMM_MXFP8):
            # MXFP8 uses block size 32, so K must be divisible by 32
            if in_features % 32 != 0:
                print(
                    f"Warning: MXFP8 requires K%32==0, got K={in_features}, falling back to TORCH"
                )
                self._backend = GEMMBackend.TORCH

        # Handle bias - only TORCH backend supports bias directly
        # Note: bias is added separately in forward() for non-TORCH backends

        # Initialize weights based on backend
        if self._backend == GEMMBackend.FP8:
            # FP8 weights need special preparation (TRT-LLM)
            self._init_fp8_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.FP8_SM90:
            # FP8 blockscale for SM90 (Hopper)
            self._init_fp8_sm90_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.BMM_FP8:
            # BMM FP8 with cublas backend
            self._init_bmm_fp8_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.FP4:
            # FP4 weights need quantization
            self._init_fp4_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.MXFP8:
            # MXFP8 weights need quantization
            self._init_mxfp8_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.BMM_MXFP8:
            # BMM MXFP8 weights need quantization
            self._init_bmm_mxfp8_weights(dtype or torch.bfloat16)
        else:
            # TORCH, BF16, and BMM_BF16 use standard weight layout
            self._init_standard_weights(dtype or torch.bfloat16)

    def _init_standard_weights(self, dtype: torch.dtype):
        """Initialize weights for TORCH/BF16 backends."""
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, device=self.device, dtype=dtype
            )
        )
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=self.device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # For BF16 backend, cache the transposed weight
        self._weight_t_cache = None
        self._fp8_prepared = False

    def _init_fp8_weights(self, dtype: torch.dtype):
        """Initialize weights for FP8 backend."""
        # Store original weight for conversion
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, device=self.device, dtype=dtype
            )
        )
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=self.device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # FP8 specific attributes (will be computed during first forward or prepare_weights)
        self.register_buffer("_weight_fp8", None)
        self.register_buffer("_weight_scale", None)
        self._permutation_cache: Dict[torch.Size, torch.Tensor] = {}
        self._fp8_prepared = False

    def _init_fp4_weights(self, dtype: torch.dtype):
        """Initialize weights for FP4 backend."""
        # Store original weight for conversion
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, device=self.device, dtype=dtype
            )
        )
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=self.device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # FP4 specific attributes
        self.register_buffer("_weight_fp4", None)
        self.register_buffer("_weight_descale", None)
        self._fp4_prepared = False

    def _init_fp8_sm90_weights(self, dtype: torch.dtype):
        """Initialize weights for FP8_SM90 backend (fp8_blockscale_gemm_sm90)."""
        # Store original weight - will be quantized to FP8 with block scales
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, device=self.device, dtype=dtype
            )
        )
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=self.device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # FP8 SM90 specific attributes (128x128 block scales)
        self.register_buffer("_weight_fp8_sm90", None)
        self.register_buffer("_weight_scale_sm90", None)
        self._fp8_sm90_prepared = False

    def _init_bmm_fp8_weights(self, dtype: torch.dtype):
        """Initialize weights for BMM_FP8 backend (bmm_fp8 with cublas)."""
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, device=self.device, dtype=dtype
            )
        )
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=self.device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # BMM FP8 specific attributes
        self.register_buffer("_weight_fp8_bmm", None)
        self.register_buffer("_weight_scale_bmm", None)
        self._bmm_fp8_prepared = False

    def _init_mxfp8_weights(self, dtype: torch.dtype):
        """Initialize weights for MXFP8 backend (mm_mxfp8)."""
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, device=self.device, dtype=dtype
            )
        )
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=self.device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # MXFP8 specific attributes (block size 32)
        self.register_buffer("_weight_mxfp8", None)
        self.register_buffer("_weight_scale_mxfp8", None)
        self._mxfp8_prepared = False

    def _init_bmm_mxfp8_weights(self, dtype: torch.dtype):
        """Initialize weights for BMM_MXFP8 backend (bmm_mxfp8)."""
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, device=self.device, dtype=dtype
            )
        )
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, device=self.device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # BMM MXFP8 specific attributes
        self.register_buffer("_weight_mxfp8_bmm", None)
        self.register_buffer("_weight_scale_mxfp8_bmm", None)
        self._bmm_mxfp8_prepared = False

    @torch.no_grad()
    def prepare_weights(self):
        """Prepare weights for optimized inference (call after loading weights)."""
        if self._backend == GEMMBackend.FP8:
            self._prepare_fp8_weights()
        elif self._backend == GEMMBackend.FP8_SM90:
            self._prepare_fp8_sm90_weights()
        elif self._backend == GEMMBackend.BMM_FP8:
            self._prepare_bmm_fp8_weights()
        elif self._backend == GEMMBackend.FP4:
            self._prepare_fp4_weights()
        elif self._backend == GEMMBackend.BF16:
            self._prepare_bf16_weights()
        elif self._backend == GEMMBackend.BMM_BF16:
            self._prepare_bmm_bf16_weights()
        elif self._backend == GEMMBackend.MXFP8:
            self._prepare_mxfp8_weights()
        elif self._backend == GEMMBackend.BMM_MXFP8:
            self._prepare_bmm_mxfp8_weights()

    @torch.no_grad()
    def _prepare_bf16_weights(self):
        """Prepare weights for BF16 backend (pre-transpose)."""
        # mm_bf16 expects weight in column-major layout (k, n)
        # Our weight is (out_features, in_features) = (n, k), need to transpose to (k, n)
        self._weight_t_cache = self.weight.data.t().contiguous()

    @torch.no_grad()
    def _prepare_fp8_weights(self):
        """Prepare weights for FP8 backend."""
        if self._fp8_prepared:
            return

        # Convert weight to FP8
        weight = self.weight.data
        finfo = torch.finfo(torch.float8_e4m3fn)

        # Compute scale
        amax = weight.abs().max().clamp(min=1e-12)
        scale = finfo.max / amax

        # Quantize to FP8
        weight_scaled = (weight * scale).clamp(finfo.min, finfo.max)
        weight_fp8 = weight_scaled.to(torch.float8_e4m3fn)

        # Prepare for TRT-LLM low latency GEMM
        # Weight shape: (n, k) -> needs to be (k // block_size, n, block_size)
        prepared_weight = flashinfer.prepare_low_latency_gemm_weights(
            weight_fp8, self._permutation_cache
        )

        self._weight_fp8 = prepared_weight
        self._weight_scale = 1.0 / scale
        self._fp8_prepared = True

    @torch.no_grad()
    def _prepare_fp4_weights(self):
        """Prepare weights for FP4 backend."""
        if self._fp4_prepared:
            return

        # Use FlashInfer's nvfp4_quantize for FP4 conversion
        weight = self.weight.data
        # nvfp4_quantize expects (m, k) layout for row-major
        # Our weight is (out_features, in_features) = (n, k)

        # For mm_fp4, weight needs to be column-major (k, n)
        weight_t = weight.t().contiguous()  # (k, n)

        # Quantize to NV-FP4
        weight_fp4, weight_descale = flashinfer.nvfp4_quantize(weight_t)

        self._weight_fp4 = weight_fp4
        self._weight_descale = weight_descale
        self._fp4_prepared = True

    @torch.no_grad()
    def _prepare_fp8_sm90_weights(self):
        """Prepare weights for FP8_SM90 backend (fp8_blockscale_gemm_sm90).

        Uses 128x128 block-scale quantization for optimal SM90 performance.
        """
        if self._fp8_sm90_prepared:
            return

        weight = self.weight.data  # (N, K)
        n, k = weight.shape

        # Pad to multiples of 128 for block quantization
        n_pad = ((n + 127) // 128) * 128
        k_pad = ((k + 127) // 128) * 128

        weight_padded = torch.zeros(
            n_pad, k_pad, dtype=weight.dtype, device=weight.device
        )
        weight_padded[:n, :k] = weight

        # Reshape for 128x128 block quantization
        # Shape: (N_blocks, 128, K_blocks, 128)
        weight_view = weight_padded.view(n_pad // 128, 128, k_pad // 128, 128)

        # Compute per-block max for scaling
        block_amax = (
            weight_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
        )

        # FP8 E4M3 max value is ~448
        fp8_max = 448.0
        scale = block_amax / fp8_max

        # Quantize to FP8
        weight_scaled = (weight_view / scale).to(torch.float8_e4m3fn)

        # Store unpadded weight and scales
        self._weight_fp8_sm90 = weight_scaled.view(n_pad, k_pad)[:n, :k].contiguous()
        # Scales shape: (N // 128, K // 128) for 128x128 blocks
        self._weight_scale_sm90 = scale.view(n_pad // 128, k_pad // 128)[
            : ((n + 127) // 128), : ((k + 127) // 128)
        ].contiguous()
        self._fp8_sm90_prepared = True

    @torch.no_grad()
    def _prepare_bmm_fp8_weights(self):
        """Prepare weights for BMM_FP8 backend (bmm_fp8 with cublas).

        Uses per-tensor scaling for FP8 quantization.
        """
        if self._bmm_fp8_prepared:
            return

        weight = self.weight.data  # (N, K)

        # Per-tensor FP8 quantization
        finfo = torch.finfo(torch.float8_e4m3fn)
        amax = weight.abs().max().clamp(min=1e-12)
        scale = finfo.max / amax

        # Quantize to FP8
        weight_scaled = (weight * scale).clamp(finfo.min, finfo.max)
        weight_fp8 = weight_scaled.to(torch.float8_e4m3fn)

        # bmm_fp8 expects B in (batch, K, N) column-major format
        # Create as (1, N, K) then transpose to get column-major (1, K, N)
        # DO NOT call .contiguous() - column-major means non-contiguous transpose view
        self._weight_fp8_bmm = weight_fp8.unsqueeze(0).transpose(
            -2, -1
        )  # (1, K, N) column-major
        self._weight_scale_bmm = (1.0 / scale).to(torch.float32)
        self._bmm_fp8_prepared = True

    @torch.no_grad()
    def _prepare_bmm_bf16_weights(self):
        """Prepare weights for BMM_BF16 backend (bmm_bf16).

        Pre-transposes weight to column-major layout for bmm_bf16.
        """
        # bmm_bf16 expects B in (batch, K, N) column-major format
        # Our weight is (N, K), create as (1, N, K) then transpose to (1, K, N)
        # DO NOT call .contiguous() - column-major means non-contiguous transpose view
        self._weight_t_cache = self.weight.data.unsqueeze(0).transpose(
            -2, -1
        )  # (1, K, N) column-major

    @torch.no_grad()
    def _prepare_mxfp8_weights(self):
        """Prepare weights for MXFP8 backend (mm_mxfp8).

        Uses FlashInfer's mxfp8_quantize with block size 32.
        """
        if self._mxfp8_prepared:
            return

        weight = self.weight.data  # (N, K)
        # mm_mxfp8 expects weight in column-major (K, N) format
        weight_t = weight.t().contiguous()  # (K, N)

        # Quantize weight to MXFP8 using FlashInfer
        # mxfp8_quantize expects (M, K) layout, so we pass (K, N) as our "M, K"
        weight_mxfp8, weight_scale = flashinfer.mxfp8_quantize(
            weight_t, is_sf_swizzled_layout=True
        )

        self._weight_mxfp8 = weight_mxfp8
        self._weight_scale_mxfp8 = weight_scale
        self._mxfp8_prepared = True

    @torch.no_grad()
    def _prepare_bmm_mxfp8_weights(self):
        """Prepare weights for BMM_MXFP8 backend (bmm_mxfp8).

        Uses FlashInfer's mxfp8_quantize with block size 32.
        """
        if self._bmm_mxfp8_prepared:
            return

        weight = self.weight.data  # (N, K)
        # bmm_mxfp8 expects B in (batch, K, N) column-major format
        weight_t = weight.t().contiguous()  # (K, N)

        # Quantize weight to MXFP8 using FlashInfer
        weight_mxfp8, weight_scale = flashinfer.mxfp8_quantize(
            weight_t, is_sf_swizzled_layout=True
        )

        # Add batch dimension
        self._weight_mxfp8_bmm = weight_mxfp8.unsqueeze(0)  # (1, K, N)
        self._weight_scale_mxfp8_bmm = weight_scale.unsqueeze(0)  # (1, ...)
        self._bmm_mxfp8_prepared = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized GEMM.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Flatten batch dimensions
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        if self._backend == GEMMBackend.TORCH:
            out = self._forward_torch(x_2d)
        elif self._backend == GEMMBackend.BF16:
            out = self._forward_bf16(x_2d)
        elif self._backend == GEMMBackend.FP8:
            out = self._forward_fp8(x_2d)
        elif self._backend == GEMMBackend.FP8_SM90:
            out = self._forward_fp8_sm90(x_2d)
        elif self._backend == GEMMBackend.BMM_FP8:
            out = self._forward_bmm_fp8(x_2d)
        elif self._backend == GEMMBackend.FP4:
            out = self._forward_fp4(x_2d)
        elif self._backend == GEMMBackend.BMM_BF16:
            out = self._forward_bmm_bf16(x_2d)
        elif self._backend == GEMMBackend.MXFP8:
            out = self._forward_mxfp8(x_2d)
        elif self._backend == GEMMBackend.BMM_MXFP8:
            out = self._forward_bmm_mxfp8(x_2d)
        else:
            out = self._forward_torch(x_2d)

        # Restore batch dimensions
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using standard PyTorch."""
        return F.linear(x, self.weight, self.bias)

    def _forward_bf16(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer mm_bf16."""
        # Ensure BF16
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x

        # Cache transposed weight
        if self._weight_t_cache is None:
            self._prepare_bf16_weights()

        weight_t = self._weight_t_cache.to(torch.bfloat16)

        # mm_bf16: a @ b where a is (m, k), b is (k, n) column-major
        out = flashinfer.mm_bf16(x_bf16, weight_t)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer mm_fp8."""
        if not self._fp8_prepared:
            self._prepare_fp8_weights()

        # Quantize input to FP8
        finfo = torch.finfo(torch.float8_e4m3fn)
        x_amax = x.abs().max().clamp(min=1e-12)
        x_scale = finfo.max / x_amax
        x_fp8 = (x * x_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)

        # Compute output scale
        alpha = (1.0 / x_scale) * self._weight_scale

        # Run FP8 GEMM
        out = flashinfer.mm_fp8(x_fp8, self._weight_fp8, alpha=alpha)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out

    def _forward_fp4(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer mm_fp4."""
        if not self._fp4_prepared:
            self._prepare_fp4_weights()

        # Quantize input to FP4
        x_fp4, x_descale = flashinfer.nvfp4_quantize(x.contiguous())

        # Run FP4 GEMM
        out = flashinfer.mm_fp4(
            x_fp4,
            self._weight_fp4,
            x_descale,
            self._weight_descale,
            out_dtype=x.dtype,
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out

    def _forward_fp8_sm90(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer fp8_blockscale_gemm_sm90.

        This uses the DeepGEMM backend optimized for SM90 (Hopper) GPUs.
        Supports BF16 input with internal quantization or pre-quantized FP8 weights.
        """
        if not self._fp8_sm90_prepared:
            self._prepare_fp8_sm90_weights()

        # Ensure BF16 input (kernel does internal quantization)
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x

        # fp8_blockscale_gemm_sm90: input (M, K), weight (N, K) -> output (M, N)
        # With FP8 weight and block scales
        out = flashinfer.gemm.fp8_blockscale_gemm_sm90(
            x_bf16,
            self._weight_fp8_sm90,
            input_scale=None,  # BF16 input, no scale needed
            weight_scale=self._weight_scale_sm90,
            out_dtype=torch.bfloat16,
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_bmm_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer bmm_fp8 with cublas backend.

        This uses per-tensor FP8 quantization with cublas.
        """
        if not self._bmm_fp8_prepared:
            self._prepare_bmm_fp8_weights()

        # Quantize input to FP8 with per-tensor scale
        finfo = torch.finfo(torch.float8_e4m3fn)
        x_amax = x.abs().max().clamp(min=1e-12)
        x_scale = finfo.max / x_amax
        x_fp8 = (x * x_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)

        # bmm_fp8 expects (batch, M, K) input
        x_fp8_batch = x_fp8.unsqueeze(0)  # (1, M, K)

        # Compute scales for output
        a_scale = (1.0 / x_scale).to(torch.float32).unsqueeze(0)  # (1,)
        b_scale = self._weight_scale_bmm.unsqueeze(0)  # (1,)

        # Run bmm_fp8: (1, M, K) @ (1, K, N) -> (1, M, N)
        out = flashinfer.gemm.bmm_fp8(
            x_fp8_batch,
            self._weight_fp8_bmm,
            a_scale,
            b_scale,
            dtype=torch.bfloat16,
            backend="cublas",
        )

        # Remove batch dimension
        out = out.squeeze(0)  # (M, N)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_bmm_bf16(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer bmm_bf16.

        Uses batched matrix multiply with BF16 precision.
        """
        # Prepare weights if not already done
        if self._weight_t_cache is None:
            self._prepare_bmm_bf16_weights()

        # Ensure BF16
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x

        # bmm_bf16 expects (batch, M, K) input
        x_batch = x_bf16.unsqueeze(0)  # (1, M, K)
        weight_bf16 = self._weight_t_cache.to(torch.bfloat16)

        # Run bmm_bf16: (1, M, K) @ (1, K, N) -> (1, M, N)
        out = flashinfer.bmm_bf16(x_batch, weight_bf16)

        # Remove batch dimension
        out = out.squeeze(0)  # (M, N)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_mxfp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer mm_mxfp8.

        Uses MXFP8 quantization with block size 32.
        """
        if not self._mxfp8_prepared:
            self._prepare_mxfp8_weights()

        # Quantize input to MXFP8
        x_mxfp8, x_scale = flashinfer.mxfp8_quantize(
            x.contiguous(), is_sf_swizzled_layout=True
        )

        # Run mm_mxfp8: (M, K) @ (K, N) -> (M, N)
        out = flashinfer.mm_mxfp8(
            x_mxfp8,
            self._weight_mxfp8,
            x_scale,
            self._weight_scale_mxfp8,
            out_dtype=torch.bfloat16,
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_bmm_mxfp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer bmm_mxfp8.

        Uses batched MXFP8 quantization with block size 32.
        """
        if not self._bmm_mxfp8_prepared:
            self._prepare_bmm_mxfp8_weights()

        # Quantize input to MXFP8
        x_mxfp8, x_scale = flashinfer.mxfp8_quantize(
            x.contiguous(), is_sf_swizzled_layout=True
        )

        # bmm_mxfp8 expects (batch, M, K) input
        x_batch = x_mxfp8.unsqueeze(0)  # (1, M, K)
        x_scale_batch = x_scale.unsqueeze(0)  # (1, ...)

        # Run bmm_mxfp8: (1, M, K) @ (1, K, N) -> (1, M, N)
        out = flashinfer.bmm_mxfp8(
            x_batch,
            self._weight_mxfp8_bmm,
            x_scale_batch,
            self._weight_scale_mxfp8_bmm,
            dtype=torch.bfloat16,
        )

        # Remove batch dimension
        out = out.squeeze(0)  # (M, N)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    @property
    def backend(self) -> str:
        """Return current backend name."""
        return self._backend.value

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.has_bias}, backend={self._backend.value}"
        )


@dataclass
class WanTransformer3DConfig:
    """Configuration for WanTransformer3DModel.

    Added fields for FlashInfer optimizations:
    - gemm_backend: GEMM backend for linear layers ("auto", "torch", "bf16", "fp8", "fp4")
    - use_skip_softmax_sparse: Whether to use skip-softmax sparse attention
    - skip_softmax_threshold_scale_factor: Threshold scale factor for skip-softmax sparsity
    """

    patch_size: Tuple[int, ...] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: Optional[int] = None
    added_kv_proj_dim: Optional[int] = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: Optional[int] = None

    # FlashInfer optimization options
    gemm_backend: Literal[
        "auto",
        "torch",
        "bf16",
        "fp8",
        "fp8_sm90",
        "bmm_fp8",
        "fp4",
        "bmm_bf16",
        "mxfp8",
        "bmm_mxfp8",
    ] = "auto"
    use_skip_softmax_sparse: bool = False  # Enable skip-softmax sparse attention
    skip_softmax_threshold_scale_factor: float = 1.0  # Threshold scale factor for skip-softmax (higher = more sparse, less accurate)

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


def create_linear_layer(
    in_features: int,
    out_features: int,
    bias: bool = True,
    gemm_backend: str = "auto",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """
    Factory function to create a linear layer with the specified GEMM backend.

    If gemm_backend is "torch", returns standard nn.Linear.
    Otherwise, returns FlashInferLinear with the specified backend.
    """
    if gemm_backend == "torch":
        return nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
    else:
        return FlashInferLinear(
            in_features,
            out_features,
            bias=bias,
            backend=gemm_backend,
            device=device,
            dtype=dtype,
        )


def get_1d_rotary_pos_embed(
    dim: int,
    pos: int,
    theta: float = 10000.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate 1D rotary position embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim))
    t = torch.arange(pos, dtype=freqs_dtype)
    freqs = torch.outer(t, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1)
    else:
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()

    return freqs_cos.float(), freqs_sin.float()


class FlashInferRMSNorm(nn.Module):
    """RMSNorm using FlashInfer kernel."""

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            # FlashInfer rmsnorm only supports FP16/BF16, cast if needed
            if x.dtype == torch.float32:
                x_bf16 = x.to(torch.bfloat16)
                weight_bf16 = self.weight.to(torch.bfloat16)
                out = flashinfer.rmsnorm(x_bf16.contiguous(), weight_bf16, self.eps)
                return out.to(torch.float32)
            return flashinfer.rmsnorm(x.contiguous(), self.weight, self.eps)
        else:
            # Fallback for non-affine case
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps)


class FlashInferFP32LayerNorm(nn.Module):
    """LayerNorm computed in FP32 using FlashInfer kernel when applicable."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()

        if self.elementwise_affine and x.dim() == 2:
            # Use FlashInfer layernorm for 2D tensors with affine params
            # FlashInfer layernorm expects weight/bias in float32
            weight = (
                self.weight.float()
                if self.weight is not None
                else torch.ones(
                    self.normalized_shape, device=x.device, dtype=torch.float32
                )
            )
            bias = (
                self.bias.float()
                if self.bias is not None
                else torch.zeros(
                    self.normalized_shape, device=x.device, dtype=torch.float32
                )
            )
            out = flashinfer.layernorm(
                x_float.to(torch.bfloat16), weight, bias, self.eps
            )
            return out.to(orig_dtype)
        else:
            # Fallback for 3D+ tensors or non-affine
            out = F.layer_norm(
                x_float,
                (self.normalized_shape,),
                self.weight.float() if self.weight is not None else None,
                self.bias.float() if self.bias is not None else None,
                self.eps,
            )
            return out.to(orig_dtype)


class FlashInferFeedForward(nn.Module):
    """Feed-forward network with GELU activation and FlashInfer GEMM support.

    Note: The original Wan model uses simple GELU without gating (up->gelu->down).
    FlashInfer's gelu_and_mul is for gated FFN (gelu(gate) * up), which is different.
    We use PyTorch's F.gelu here since FlashInfer doesn't have a standalone GELU kernel.

    Args:
        dim: Input/output dimension
        inner_dim: Hidden dimension
        activation_fn: Activation function ("gelu-approximate", "gelu")
        bias: Whether to use bias in linear layers
        use_gating: Whether to use gated FFN (gelu(x) * gate)
        gemm_backend: GEMM backend for linear layers ("auto", "torch", "bf16", "fp8", "fp4")
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
        use_gating: bool = False,  # Whether to use gated FFN (gelu(x) * gate)
        gemm_backend: str = "auto",
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.use_gating = use_gating
        self.gemm_backend = gemm_backend

        if use_gating:
            # Gated FFN: fuse gate and up projections
            # gelu(Wx) * Ux - output is [..., 2 * inner_dim] split in half
            self.proj_up = create_linear_layer(
                dim, 2 * inner_dim, bias=bias, gemm_backend=gemm_backend
            )
        else:
            # Simple FFN: up -> activation -> down
            self.proj_up = create_linear_layer(
                dim, inner_dim, bias=bias, gemm_backend=gemm_backend
            )

        self.proj_down = create_linear_layer(
            inner_dim, dim, bias=bias, gemm_backend=gemm_backend
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gating:
            # Gated FFN using FlashInfer fused gelu_and_mul
            hidden = self.proj_up(x)

            # Check alignment requirement (16-byte aligned)
            if hidden.shape[-1] * hidden.dtype.itemsize % 16 == 0:
                # FlashInfer activation only supports FP16/BF16
                orig_dtype = hidden.dtype
                if orig_dtype == torch.float32:
                    hidden = hidden.to(torch.bfloat16)

                if self.activation_fn == "gelu-approximate":
                    hidden = flashinfer.gelu_tanh_and_mul(hidden.contiguous())
                else:
                    hidden = flashinfer.gelu_and_mul(hidden.contiguous())

                if orig_dtype == torch.float32:
                    hidden = hidden.to(orig_dtype)
            else:
                # Fallback for non-aligned
                gate, up = hidden.chunk(2, dim=-1)
                hidden = F.gelu(gate, approximate="tanh") * up
        else:
            # Simple FFN: up -> activation -> down
            hidden = self.proj_up(x)
            hidden = F.gelu(hidden, approximate="tanh")

        return self.proj_down(hidden)


class WanRotaryPosEmbed(nn.Module):
    """3D Rotary Position Embedding for video transformers.

    This is kept from the original implementation as it's specialized for video
    (combining temporal, height, width position encodings) and not directly
    supported by FlashInfer's standard RoPE APIs.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )

        return freqs_cos, freqs_sin


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to hidden states."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class FlashInferWanAttention(nn.Module):
    """Attention module using FlashInfer kernels with optimized GEMM backends.

    Args:
        dim: Input dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        eps: Epsilon for normalization
        dropout: Dropout rate
        added_kv_proj_dim: Dimension for additional KV projection (I2V)
        cross_attention_dim_head: Head dimension for cross attention
        is_cross_attention: Whether this is cross attention
        gemm_backend: GEMM backend for linear layers ("auto", "torch", "bf16", "fp8", "fp4")
        use_skip_softmax_sparse: Whether to use skip-softmax sparse attention
        skip_softmax_threshold_scale_factor: Threshold scale factor for skip-softmax sparsity
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        is_cross_attention: Optional[bool] = None,
        gemm_backend: str = "auto",
        use_skip_softmax_sparse: bool = False,
        skip_softmax_threshold_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )
        self.gemm_backend = gemm_backend
        self.use_skip_softmax_sparse = use_skip_softmax_sparse
        self.skip_softmax_threshold_scale_factor = skip_softmax_threshold_scale_factor

        # Use FlashInfer-optimized linear layers
        self.to_q = create_linear_layer(
            dim, self.inner_dim, bias=True, gemm_backend=gemm_backend
        )
        self.to_k = create_linear_layer(
            dim, self.kv_inner_dim, bias=True, gemm_backend=gemm_backend
        )
        self.to_v = create_linear_layer(
            dim, self.kv_inner_dim, bias=True, gemm_backend=gemm_backend
        )
        self.to_out = nn.ModuleList(
            [
                create_linear_layer(
                    self.inner_dim, dim, bias=True, gemm_backend=gemm_backend
                ),
                nn.Dropout(dropout),
            ]
        )

        # Use FlashInfer RMSNorm for Q/K normalization
        self.norm_q = FlashInferRMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.norm_k = FlashInferRMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = create_linear_layer(
                added_kv_proj_dim, self.inner_dim, bias=True, gemm_backend=gemm_backend
            )
            self.add_v_proj = create_linear_layer(
                added_kv_proj_dim, self.inner_dim, bias=True, gemm_backend=gemm_backend
            )
            self.norm_added_k = FlashInferRMSNorm(
                dim_head * heads, eps=eps, elementwise_affine=False
            )

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if self.add_k_proj is not None and encoder_hidden_states is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Apply RMSNorm to Q and K using FlashInfer
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape to (batch, seq, heads, head_dim)
        batch_size = hidden_states.shape[0]
        seq_len_q = hidden_states.shape[1]
        seq_len_kv = encoder_hidden_states.shape[1]

        query = query.view(batch_size, seq_len_q, self.heads, self.dim_head)
        key = key.view(batch_size, seq_len_kv, self.heads, self.dim_head)
        value = value.view(batch_size, seq_len_kv, self.heads, self.dim_head)

        # Apply rotary embedding
        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # FlashInfer attention only supports FP16/BF16, cast if needed
        orig_dtype = query.dtype
        needs_cast = orig_dtype == torch.float32
        if needs_cast:
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        # I2V task: handle image cross-attention
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = self.add_k_proj(encoder_hidden_states_img)
            value_img = self.add_v_proj(encoder_hidden_states_img)
            key_img = self.norm_added_k(key_img)

            seq_len_img = encoder_hidden_states_img.shape[1]
            key_img = key_img.view(batch_size, seq_len_img, self.heads, self.dim_head)
            value_img = value_img.view(
                batch_size, seq_len_img, self.heads, self.dim_head
            )
            if needs_cast:
                key_img = key_img.to(torch.bfloat16)
                value_img = value_img.to(torch.bfloat16)

            # Use FlashInfer attention for image cross-attention
            # Process each batch item separately for FlashInfer
            img_outputs = []
            for b in range(batch_size):
                out = flashinfer.single_prefill_with_kv_cache(
                    query[b].contiguous(),  # (seq_len_q, heads, head_dim)
                    key_img[b].contiguous(),  # (seq_len_img, heads, head_dim)
                    value_img[b].contiguous(),  # (seq_len_img, heads, head_dim)
                    causal=False,
                )
                img_outputs.append(out)
            hidden_states_img = torch.stack(img_outputs, dim=0)
            hidden_states_img = hidden_states_img.view(batch_size, seq_len_q, -1)

        # Main attention using FlashInfer
        # Check if skip-softmax sparse attention is supported (requires SM100 or SM103 only)
        device = query.device
        major, minor = get_compute_capability(device)
        sm_version = major * 10 + minor
        # trtllm_batch_context_with_kv_cache only supports SM100 and SM103
        use_skip_softmax = self.use_skip_softmax_sparse and sm_version in (100, 103)

        if self.use_skip_softmax_sparse and not use_skip_softmax:
            import warnings

            warnings.warn(
                f"Skip-softmax sparse attention requires SM100 or SM103 (Blackwell B100/B200), "
                f"but current GPU is SM{sm_version}. Falling back to standard attention.",
                RuntimeWarning,
                stacklevel=2,
            )

        if use_skip_softmax:
            # Use trtllm_batch_context_with_kv_cache with skip-softmax sparse attention
            # Set up paged KV cache: one page per sequence, page_size = seq_len_kv
            # Create paged KV cache: [num_pages, 2, num_heads, page_size, head_dim] in HND layout
            # Transpose from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]
            key_paged = key.permute(
                0, 2, 1, 3
            ).contiguous()  # [batch, heads, seq, head_dim]
            value_paged = value.permute(
                0, 2, 1, 3
            ).contiguous()  # [batch, heads, seq, head_dim]

            # Stack K and V: [num_pages, 2, num_heads, page_size, head_dim]
            kv_cache = torch.stack([key_paged, value_paged], dim=1)

            # Block table: each sequence uses one page (its own index)
            block_tables = torch.arange(
                batch_size, dtype=torch.int32, device=device
            ).unsqueeze(1)

            # Sequence lengths (all same length)
            seq_lens = torch.full(
                (batch_size,), seq_len_kv, dtype=torch.int32, device=device
            )

            # Cumulative sequence lengths
            cum_seq_lens_q = torch.arange(
                0,
                (batch_size + 1) * seq_len_q,
                seq_len_q,
                dtype=torch.int32,
                device=device,
            )
            cum_seq_lens_kv = torch.arange(
                0,
                (batch_size + 1) * seq_len_kv,
                seq_len_kv,
                dtype=torch.int32,
                device=device,
            )

            # Workspace buffer (needs to be zeroed on first use)
            workspace_size = 128 * 1024 * 1024  # 128 MB
            workspace = torch.zeros(workspace_size, dtype=torch.uint8, device=device)

            # Reshape query for batch processing: [total_tokens, num_heads, head_dim]
            query_flat = query.reshape(
                batch_size * seq_len_q, self.heads, self.dim_head
            ).contiguous()

            # Scale factors for attention
            sm_scale = 1.0 / math.sqrt(self.dim_head)

            hidden_states = trtllm_batch_context_with_kv_cache(
                query=query_flat,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=seq_len_q,
                max_kv_len=seq_len_kv,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                batch_size=batch_size,
                cum_seq_lens_q=cum_seq_lens_q,
                cum_seq_lens_kv=cum_seq_lens_kv,
                kv_layout="HND",
                skip_softmax_threshold_scale_factor=self.skip_softmax_threshold_scale_factor,
            )
            hidden_states = hidden_states.view(batch_size, seq_len_q, -1)
        else:
            # Standard attention using single_prefill_with_kv_cache
            outputs = []
            for b in range(batch_size):
                out = flashinfer.single_prefill_with_kv_cache(
                    query[b].contiguous(),  # (seq_len_q, heads, head_dim)
                    key[b].contiguous(),  # (seq_len_kv, heads, head_dim)
                    value[b].contiguous(),  # (seq_len_kv, heads, head_dim)
                    causal=False,
                )
                outputs.append(out)
            hidden_states = torch.stack(outputs, dim=0)
            hidden_states = hidden_states.view(batch_size, seq_len_q, -1)

        # Cast back to original dtype if needed
        if needs_cast:
            hidden_states = hidden_states.to(orig_dtype)
            if hidden_states_img is not None:
                hidden_states_img = hidden_states_img.to(orig_dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class FlashInferWanTransformerBlock(nn.Module):
    """Transformer block using FlashInfer optimizations.

    Args:
        dim: Hidden dimension
        ffn_dim: Feed-forward hidden dimension
        num_heads: Number of attention heads
        qk_norm: QK normalization type
        cross_attn_norm: Whether to use cross attention normalization
        eps: Epsilon for normalization
        added_kv_proj_dim: Dimension for additional KV projection
        gemm_backend: GEMM backend for linear layers
        use_skip_softmax_sparse: Whether to use skip-softmax sparse attention
        skip_softmax_threshold_scale_factor: Threshold scale factor for skip-softmax
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        gemm_backend: str = "auto",
        use_skip_softmax_sparse: bool = False,
        skip_softmax_threshold_scale_factor: float = 1.0,
    ):
        super().__init__()

        # 1. Self-attention (with optional skip-softmax sparse attention)
        self.norm1 = FlashInferFP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = FlashInferWanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            gemm_backend=gemm_backend,
            use_skip_softmax_sparse=use_skip_softmax_sparse,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )

        # 2. Cross-attention (always full attention, no sparse attention)
        self.attn2 = FlashInferWanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            gemm_backend=gemm_backend,
            use_skip_softmax_sparse=False,  # No sparse attention for cross-attention
        )
        self.norm2 = (
            FlashInferFP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward with FlashInfer GELU fusion and optimized GEMM
        self.ffn = FlashInferFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu-approximate",
            gemm_backend=gemm_backend,
        )
        self.norm3 = FlashInferFP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        return hidden_states


class WanImageEmbedding(nn.Module):
    """Image embedding module with FlashInfer GEMM support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pos_embed_seq_len: Optional[int] = None,
        gemm_backend: str = "auto",
    ):
        super().__init__()

        self.norm1 = FlashInferFP32LayerNorm(in_features)
        self.ff = FlashInferFeedForward(
            in_features, out_features, activation_fn="gelu", gemm_backend=gemm_backend
        )
        self.norm2 = FlashInferFP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_seq_len, in_features)
            )
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class Timesteps(nn.Module):
    """Timestep embedding."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            0, half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = timesteps[:, None].float() * exponent[None, :].exp()
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding projection with FlashInfer GEMM support."""

    def __init__(
        self, in_channels: int, time_embed_dim: int, gemm_backend: str = "auto"
    ):
        super().__init__()
        self.linear_1 = create_linear_layer(
            in_channels, time_embed_dim, bias=True, gemm_backend=gemm_backend
        )
        self.act = nn.SiLU()
        self.linear_2 = create_linear_layer(
            time_embed_dim, time_embed_dim, bias=True, gemm_backend=gemm_backend
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaTextProjection(nn.Module):
    """Text projection with GELU-tanh activation and FlashInfer GEMM support."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        act_fn: str = "gelu_tanh",
        gemm_backend: str = "auto",
    ):
        super().__init__()
        self.linear_1 = create_linear_layer(
            in_features, hidden_size, bias=True, gemm_backend=gemm_backend
        )
        self.linear_2 = create_linear_layer(
            hidden_size, hidden_size, bias=True, gemm_backend=gemm_backend
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    """Combined time, text, and image embedding with FlashInfer GEMM support."""

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
        gemm_backend: str = "auto",
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim, gemm_backend=gemm_backend
        )
        self.act_fn = nn.SiLU()
        self.time_proj = create_linear_layer(
            dim, time_proj_dim, gemm_backend=gemm_backend
        )
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh", gemm_backend=gemm_backend
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim,
                dim,
                pos_embed_seq_len=pos_embed_seq_len,
                gemm_backend=gemm_backend,
            )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None and self.image_embedder is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class FlashInferWanTransformer3DModel(nn.Module):
    """
    FlashInfer-optimized Wan Transformer for video generation.

    This is an inference-optimized implementation of WanTransformer3DModel
    using FlashInfer kernels for:
    - RMSNorm normalization
    - Scaled dot-product attention
    - Fused GELU activations
    - Optimized GEMM backends (bf16/fp8/fp4) for linear layers
    - Optional skip-softmax sparse attention

    Args:
        config: WanTransformer3DConfig or compatible config object

    Config options for optimization:
        gemm_backend: "auto", "torch", "bf16", "fp8", "fp4"
        use_skip_softmax_sparse: Whether to use skip-softmax sparse attention
        skip_softmax_threshold_scale_factor: Threshold scale factor for skip-softmax
    """

    def __init__(self, config: Union[WanTransformer3DConfig, Any]):
        super().__init__()

        # Handle both dataclass config and HuggingFace config
        if hasattr(config, "patch_size"):
            self.config = config
        else:
            self.config = WanTransformer3DConfig()

        patch_size = getattr(config, "patch_size", (1, 2, 2))
        num_attention_heads = getattr(config, "num_attention_heads", 40)
        attention_head_dim = getattr(config, "attention_head_dim", 128)
        in_channels = getattr(config, "in_channels", 16)
        out_channels = getattr(config, "out_channels", in_channels)
        text_dim = getattr(config, "text_dim", 4096)
        freq_dim = getattr(config, "freq_dim", 256)
        ffn_dim = getattr(config, "ffn_dim", 13824)
        num_layers = getattr(config, "num_layers", 40)
        cross_attn_norm = getattr(config, "cross_attn_norm", True)
        qk_norm = getattr(config, "qk_norm", "rms_norm_across_heads")
        eps = getattr(config, "eps", 1e-6)
        image_dim = getattr(config, "image_dim", None)
        added_kv_proj_dim = getattr(config, "added_kv_proj_dim", None)
        rope_max_seq_len = getattr(config, "rope_max_seq_len", 1024)
        pos_embed_seq_len = getattr(config, "pos_embed_seq_len", None)

        # FlashInfer optimization options
        gemm_backend = getattr(config, "gemm_backend", "auto")
        use_skip_softmax_sparse = getattr(config, "use_skip_softmax_sparse", False)
        skip_softmax_threshold_scale_factor = getattr(
            config, "skip_softmax_threshold_scale_factor", 1.0
        )

        inner_dim = num_attention_heads * attention_head_dim

        # Store config values for forward pass
        self.patch_size = patch_size
        self.gemm_backend = gemm_backend
        self.use_skip_softmax_sparse = use_skip_softmax_sparse
        self.skip_softmax_threshold_scale_factor = skip_softmax_threshold_scale_factor

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings with optimized GEMM
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
            gemm_backend=gemm_backend,
        )

        # 3. Transformer blocks with FlashInfer optimizations
        self.blocks = nn.ModuleList(
            [
                FlashInferWanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    gemm_backend=gemm_backend,
                    use_skip_softmax_sparse=use_skip_softmax_sparse,
                    skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FlashInferFP32LayerNorm(
            inner_dim, eps, elementwise_affine=False
        )
        self.proj_out = create_linear_layer(
            inner_dim, out_channels * math.prod(patch_size), gemm_backend=gemm_backend
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

    def prepare_weights(self):
        """Prepare all linear layer weights for optimized inference.

        Call this after loading pretrained weights to convert weights
        to the appropriate format for the selected GEMM backend.
        """
        for module in self.modules():
            if isinstance(module, FlashInferLinear):
                module.prepare_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass of the FlashInfer-optimized Wan Transformer.

        Args:
            hidden_states: Input video latents, shape (B, C, T, H, W)
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings
            encoder_hidden_states_image: Optional image embeddings for I2V
            return_dict: Whether to return a dict or tuple

        Returns:
            Output tensor or dict with 'sample' key
        """
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    use_reentrant=False,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (
                self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (
                self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return {"sample": output}

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load model from a pretrained HuggingFace checkpoint.

        Note: This requires the original diffusers model to be installed
        for config loading.
        """
        try:
            from diffusers import WanTransformer3DModel as OriginalModel
        except ImportError as e:
            raise ImportError(
                "Please install diffusers to load from pretrained: "
                "pip install diffusers"
            ) from e

        # Load original model to get config and weights
        original_model = OriginalModel.from_pretrained(model_path, **kwargs)

        # Create FlashInfer model with same config
        model = cls(original_model.config)

        # Copy weights
        model.load_state_dict(original_model.state_dict(), strict=False)

        return model


# Convenience function to convert original model
def convert_to_flashinfer(original_model) -> FlashInferWanTransformer3DModel:
    """
    Convert an existing WanTransformer3DModel to FlashInfer-optimized version.

    Args:
        original_model: The original diffusers WanTransformer3DModel

    Returns:
        FlashInferWanTransformer3DModel with copied weights
    """
    model = FlashInferWanTransformer3DModel(original_model.config)
    model.load_state_dict(original_model.state_dict(), strict=False)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FlashInfer WAN Transformer")
    parser.add_argument(
        "--gemm-backend",
        type=str,
        default="auto",
        choices=[
            "auto",
            "torch",
            "bf16",
            "fp8",
            "fp8_sm90",
            "bmm_fp8",
            "fp4",
            "bmm_bf16",
            "mxfp8",
            "bmm_mxfp8",
        ],
        help="GEMM backend for linear layers",
    )
    parser.add_argument(
        "--skip-softmax-sparse",
        action="store_true",
        help="Enable skip-softmax sparse attention",
    )
    parser.add_argument(
        "--skip-softmax-threshold",
        type=float,
        default=1.0,
        help="Threshold scale factor for skip-softmax (higher = more sparse)",
    )
    args = parser.parse_args()

    print("Testing FlashInferWanTransformer3DModel...")
    print(f"GEMM Backend: {args.gemm_backend}")
    print(f"Skip-Softmax Sparse: {args.skip_softmax_sparse}")
    if args.skip_softmax_sparse:
        print(f"Threshold Scale Factor: {args.skip_softmax_threshold}")

    # Check available GEMM backend support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        best_backend = _get_best_available_backend(device)
        print(f"Best available GEMM backend for this GPU: {best_backend.value}")

    config = WanTransformer3DConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=8,
        attention_head_dim=64,
        in_channels=4,
        out_channels=4,
        text_dim=512,
        freq_dim=256,
        ffn_dim=1024,
        num_layers=2,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=256,
        # FlashInfer optimization options
        gemm_backend=args.gemm_backend,
        use_skip_softmax_sparse=args.skip_softmax_sparse,
        skip_softmax_threshold_scale_factor=args.skip_softmax_threshold,
    )

    model = FlashInferWanTransformer3DModel(config).cuda().half()

    # Print model info
    linear_count = sum(1 for m in model.modules() if isinstance(m, FlashInferLinear))
    torch_linear_count = sum(1 for m in model.modules() if type(m) is nn.Linear)
    print(f"FlashInferLinear layers: {linear_count}")
    print(f"Standard nn.Linear layers: {torch_linear_count}")

    # Test inputs
    batch_size = 1
    num_frames = 4
    height = 32
    width = 32
    in_channels = 4
    text_seq_len = 64

    hidden_states = torch.randn(
        batch_size,
        in_channels,
        num_frames,
        height,
        width,
        device="cuda",
        dtype=torch.float16,
    )
    timestep = torch.randint(0, 1000, (batch_size,), device="cuda")
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, 512, device="cuda", dtype=torch.float16
    )

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(hidden_states, timestep, encoder_hidden_states, return_dict=False)

    # Benchmark
    import time

    torch.cuda.synchronize()
    start = time.time()
    num_iters = 10
    with torch.no_grad():
        for _ in range(num_iters):
            output = model(
                hidden_states, timestep, encoder_hidden_states, return_dict=False
            )
            torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output[0].shape}")
    print(f"Average time per forward pass: {elapsed / num_iters * 1000:.2f} ms")
    print("Test passed!")
