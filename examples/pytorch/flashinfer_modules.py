# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model-independent FlashInfer modules shared by PyTorch examples."""

import math
import warnings
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import flashinfer
from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.gemm import (
    batch_deepgemm_fp8_nt_groupwise,
    gemm_fp8_nt_blockscaled,
    gemm_fp8_nt_groupwise,
)
from flashinfer.prefill import (
    single_prefill_with_kv_cache,
    trtllm_batch_context_with_kv_cache,
)
from flashinfer.utils import get_compute_capability

_DEFAULT_WORKSPACE_SIZE = 128 * 1024 * 1024
# "torch" routes attention through ``torch.nn.functional.scaled_dot_product_attention``
# and is always available; it acts as the universal fallback when the FlashInfer
# attention kernels are unsupported on the current GPU.
_VALID_ATTENTION_BACKENDS = {"auto", "single", "cudnn", "trtllm", "torch"}


def _split_backend_choice(value: str) -> Tuple[str, Optional[str]]:
    """Split a ``"<base>[-<kernel>]"`` backend string.

    Examples:
        ``"fp4"``         -> ``("fp4", None)``
        ``"fp4-cudnn"``   -> ``("fp4", "cudnn")``
        ``"single-fa3"``  -> ``("single", "fa3")``

    The kernel suffix (when present) is forwarded to the corresponding
    FlashInfer kernel's own ``backend`` kwarg — see each kernel's docstring
    for the allowed values. ``None`` means "use the kernel's own default".
    """
    base, dash, kernel = value.partition("-")
    return base, (kernel if dash else None)


class GEMMBackend(Enum):
    """GEMM backend selection for FlashInferLinear."""

    TORCH = "torch"  # Standard PyTorch (fallback, any GPU)
    BF16 = "bf16"  # FlashInfer mm_bf16 (SM100+)
    FP8 = "fp8"  # FlashInfer mm_fp8 with TRT-LLM backend (SM89+)
    FP8_SM90 = "fp8_sm90"  # FlashInfer fp8_blockscale_gemm_sm90 (SM90 only)
    BMM_FP8 = "bmm_fp8"  # FlashInfer bmm_fp8 with cublas backend (SM89+)
    FP8_GROUPWISE = "fp8_groupwise"  # FlashInfer gemm_fp8_nt_groupwise (SM100+)
    FP8_BLOCKSCALED = "fp8_blockscaled"  # FlashInfer gemm_fp8_nt_blockscaled (SM100+)
    BATCH_DEEPGEMM_FP8 = (
        "batch_deepgemm_fp8"  # FlashInfer batch_deepgemm_fp8_nt_groupwise (SM100/SM103)
    )
    FP4 = "fp4"  # FlashInfer mm_fp4 (SM100+)
    BMM_BF16 = "bmm_bf16"  # FlashInfer bmm_bf16 (SM100+)
    MXFP8 = "mxfp8"  # FlashInfer mm_mxfp8 (SM100+/SM120+)
    BMM_MXFP8 = "bmm_mxfp8"  # FlashInfer bmm_mxfp8 (SM100+/SM120+)


# Human-readable per-backend SM requirement, used by the fallback warning.
# The actual support check is below in _check_gemm_backend_support; keep the
# two in sync.
_GEMM_BACKEND_REQUIREMENT_STR: Dict[str, str] = {
    "bf16": "SM100+ (Blackwell)",
    "fp8": "SM89/SM100+ (use FP8_SM90 on SM90)",
    "fp8_sm90": "SM90 (Hopper) only",
    "bmm_fp8": "SM89+",
    "fp8_groupwise": "SM100+ (Blackwell)",
    "fp8_blockscaled": "SM100+ (Blackwell)",
    "batch_deepgemm_fp8": "SM100/SM103",
    "fp4": "SM100+ (Blackwell)",
    "bmm_bf16": "SM100+ (Blackwell)",
    "mxfp8": "SM100+ (Blackwell)",
    "bmm_mxfp8": "SM100+ (Blackwell)",
}


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
    elif backend == GEMMBackend.FP8_GROUPWISE:
        # gemm_fp8_nt_groupwise requires SM100+ (Blackwell)
        return sm >= 100
    elif backend == GEMMBackend.FP8_BLOCKSCALED:
        # gemm_fp8_nt_blockscaled requires SM100+ (Blackwell)
        return sm >= 100
    elif backend == GEMMBackend.BATCH_DEEPGEMM_FP8:
        # batch_deepgemm_fp8_nt_groupwise requires SM100 or SM103
        return sm in (100, 103)
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


def _gemm_backend_requirement_str(backend: GEMMBackend) -> str:
    return _GEMM_BACKEND_REQUIREMENT_STR.get(backend.value, "unspecified")


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
        backend: GEMM backend string. The base name selects the FlashInfer
            code path (see ``GEMMBackend`` enum for the list); an optional
            ``"-<kernel>"`` suffix picks the kernel's own ``backend`` kwarg.
            Examples: ``"fp4"`` (kernel default), ``"fp4-cudnn"``,
            ``"fp4-cutlass"``, ``"mxfp8-cute-dsl"``, ``"bmm_fp8-cublas"``,
            ``"bmm_bf16-cutlass"``, ``"fp8_groupwise-trtllm"``. Backends
            with no per-kernel choice (``torch``, ``bf16``, ``fp8``,
            ``fp8_sm90``, ``batch_deepgemm_fp8``, ``fp8_blockscaled``)
            ignore the suffix.
        device: Device to place the module on
        dtype: Data type for weights (for TORCH/BF16 backends)
        online_act_quant: True for online activation scale computation, False
            for fixed default scale (used by FP8/FP4 paths). Ignored when the
            chosen GEMM backend does its own quantization (e.g. BF16, MXFP8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend: str = "torch",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        online_act_quant: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.online_act_quant = online_act_quant
        # ``backend`` may carry a "-<kernel>" suffix; split it once here so the
        # enum lookup and the validators all see the canonical base name.
        base_backend, self.kernel_backend = _split_backend_choice(backend)
        # Lazily-populated offline-quant scale caches. Allocated on first
        # forward (during warmup) and reused thereafter so cuda-graph capture
        # doesn't see fresh tensor allocations / CPU->GPU copies each call.
        self._offline_per_tensor_scale: Optional[torch.Tensor] = None
        self._offline_blockwise_scale: Optional[torch.Tensor] = None
        self._offline_fp4_global_sf: Optional[torch.Tensor] = None

        # Resolve device — FlashInfer kernels require CUDA. We accept None (default
        # to current CUDA device) but reject CPU since none of the GEMM backends
        # except TORCH would actually function there.
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "FlashInferLinear requires CUDA, but no CUDA device is available."
                )
            device = torch.device("cuda")
        elif torch.device(device).type != "cuda":
            raise RuntimeError(
                f"FlashInferLinear requires a CUDA device, got {device!r}."
            )
        self.device = device

        self._backend = GEMMBackend(base_backend)

        # Validate backend support
        if not _check_gemm_backend_support(self._backend, device):
            major, minor = get_compute_capability(device)
            warnings.warn(
                f"{self._backend.value} GEMM backend requires "
                f"{_gemm_backend_requirement_str(self._backend)}, "
                f"but device is SM{major * 10 + minor}; falling back to TORCH.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._backend = GEMMBackend.TORCH

        # Check dimension constraints for FP8 backends
        if self._backend == GEMMBackend.FP8_SM90:
            # fp8_blockscale_gemm_sm90 requires: N % 64 == 0, K % 128 == 0
            if out_features % 64 != 0 or in_features % 128 != 0:
                warnings.warn(
                    f"FP8_SM90 requires N%64==0 and K%128==0, got N={out_features}, "
                    f"K={in_features}; falling back to TORCH.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._backend = GEMMBackend.TORCH

        # Check dimension constraints for MXFP8 backends
        if self._backend in (GEMMBackend.MXFP8, GEMMBackend.BMM_MXFP8):
            # MXFP8 uses block size 32, so K must be divisible by 32
            if in_features % 32 != 0:
                warnings.warn(
                    f"MXFP8 requires K%32==0, got K={in_features}; falling back to TORCH.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._backend = GEMMBackend.TORCH
            # CUTLASS MXFP8 tile constraint: n>=128 and k>=128.
            # mm_mxfp8 raises if these aren't met (see flashinfer.gemm
            # _check_mm_mxfp8_problem_size). WAN has a few small projection
            # layers (e.g. Linear(1536, 64)) that hit this; fall back to torch.
            elif out_features < 128 or in_features < 128:
                warnings.warn(
                    f"{self._backend.value} requires N>=128 and K>=128, "
                    f"got N={out_features}, K={in_features}; falling back to TORCH.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._backend = GEMMBackend.TORCH

        # Check dimension constraints for NT groupwise/blockscaled FP8
        if self._backend in (
            GEMMBackend.FP8_GROUPWISE,
            GEMMBackend.FP8_BLOCKSCALED,
            GEMMBackend.BATCH_DEEPGEMM_FP8,
        ):
            if in_features % 128 != 0:
                warnings.warn(
                    f"{self._backend.value} requires K%128==0, got K={in_features}; "
                    "falling back to TORCH.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._backend = GEMMBackend.TORCH

        # Handle bias - only TORCH backend supports bias directly
        # Note: bias is added separately in forward() for non-TORCH backends

        # Initialize weights based on backend
        if self._backend == GEMMBackend.FP8:
            self._init_fp8_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.FP8_SM90:
            self._init_fp8_sm90_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.BMM_FP8:
            self._init_bmm_fp8_weights(dtype or torch.bfloat16)
        elif self._backend in (
            GEMMBackend.FP8_GROUPWISE,
            GEMMBackend.FP8_BLOCKSCALED,
            GEMMBackend.BATCH_DEEPGEMM_FP8,
        ):
            self._init_fp8_nt_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.FP4:
            self._init_fp4_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.MXFP8:
            self._init_mxfp8_weights(dtype or torch.bfloat16)
        elif self._backend == GEMMBackend.BMM_MXFP8:
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

    def _init_fp8_nt_weights(self, dtype: torch.dtype):
        """Initialize weights for FP8_GROUPWISE / FP8_BLOCKSCALED / BATCH_DEEPGEMM_FP8 backends.

        These use NT-format GEMM: a(m,k) @ b(n,k)^T with block-scale quantization.
        """
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

        # NT FP8 specific attributes (weight stored as (N, K) in FP8)
        self.register_buffer("_weight_fp8_nt", None)
        self.register_buffer("_weight_scale_nt", None)
        self._fp8_nt_prepared = False

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
        elif self._backend in (
            GEMMBackend.FP8_GROUPWISE,
            GEMMBackend.FP8_BLOCKSCALED,
            GEMMBackend.BATCH_DEEPGEMM_FP8,
        ):
            self._prepare_fp8_nt_weights()
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

        # Our weight is (N, K) = (out_features, in_features). Quantize in this
        # natural shape: nvfp4_quantize packs along the last dim, giving
        # weight_fp4 with shape (N, K/2). mm_fp4 expects ``b`` to look like
        # (K, N) column-major (see tests/gemm/test_mm_fp4.py), which we get by
        # passing ``weight_fp4.T`` at forward time.
        weight = self.weight.data  # (N, K)

        # NVFP4 per-tensor global scale: (448 * 6) / amax, broadcast to the
        # scale factor's e4m3fn range. This convention is documented in
        # mm_fp4's docstring and matches the FP4 tests in tests/gemm/.
        # amax in the weight dtype is sufficient (and far cheaper than
        # casting weight to fp32 first); weight prep runs once per layer.
        weight_amax = weight.abs().amax().to(torch.float32).clamp(min=1e-12)
        weight_global_sf = (448.0 * 6.0) / weight_amax

        weight_fp4, weight_descale = flashinfer.nvfp4_quantize(weight, weight_global_sf)

        self._weight_fp4 = weight_fp4
        self._weight_descale = weight_descale
        self._weight_global_sf = weight_global_sf
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
    def _prepare_fp8_nt_weights(self):
        """Prepare weights for FP8_GROUPWISE / FP8_BLOCKSCALED / BATCH_DEEPGEMM_FP8.

        Quantizes weight (N, K) to FP8 with 128x128 block scales.
        The NT format keeps weight as (N, K) — no transpose needed.
        """
        if self._fp8_nt_prepared:
            return

        weight = self.weight.data  # (N, K)
        n, k = weight.shape

        # Block quantization with 128x128 tiles
        block_size = 128
        n_pad = ((n + block_size - 1) // block_size) * block_size
        k_pad = ((k + block_size - 1) // block_size) * block_size

        weight_padded = torch.zeros(
            n_pad, k_pad, dtype=weight.dtype, device=weight.device
        )
        weight_padded[:n, :k] = weight

        weight_view = weight_padded.view(
            n_pad // block_size, block_size, k_pad // block_size, block_size
        )

        block_amax = (
            weight_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
        )

        fp8_max = 448.0
        scale = block_amax / fp8_max

        weight_scaled = (weight_view / scale).to(torch.float8_e4m3fn)

        self._weight_fp8_nt = weight_scaled.view(n_pad, k_pad)[:n, :k].contiguous()
        # scale shape: (n_blocks, k_blocks) stored as (k_blocks, n_blocks) for MN-major
        raw_scale = scale.squeeze(1).squeeze(-1)  # (n_blocks, k_blocks)
        n_blocks = (n + block_size - 1) // block_size
        k_blocks = (k + block_size - 1) // block_size
        self._weight_scale_nt = (
            raw_scale[:n_blocks, :k_blocks].t().contiguous().to(torch.float32)
        )
        self._fp8_nt_prepared = True

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

        # Our weight is (N, K) = (out_features, in_features). Quantize in this
        # natural shape; mm_mxfp8 expects ``b`` to look like (k, n) column-major
        # which we get by passing ``weight_mxfp8.T`` at forward time
        # (see tests/gemm/test_mm_mxfp8.py — the kernel reads mat2's underlying
        # storage as (n, k)).
        weight = self.weight.data.contiguous()  # (N, K)
        weight_mxfp8, weight_scale = flashinfer.mxfp8_quantize(
            weight, is_sf_swizzled_layout=True
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
        elif self._backend == GEMMBackend.FP8_GROUPWISE:
            out = self._forward_fp8_groupwise(x_2d)
        elif self._backend == GEMMBackend.FP8_BLOCKSCALED:
            out = self._forward_fp8_blockscaled(x_2d)
        elif self._backend == GEMMBackend.BATCH_DEEPGEMM_FP8:
            out = self._forward_batch_deepgemm_fp8(x_2d)
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

    def _quantize_activation_fp8_per_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activation to FP8 with per-tensor scale.

        When online_act_quant=True, computes scale from the actual tensor.
        When online_act_quant=False, uses a fixed default scale (1.0).
        """
        finfo = torch.finfo(torch.float8_e4m3fn)
        if self.online_act_quant:
            x_amax = x.abs().max().clamp(min=1e-12)
            x_scale = finfo.max / x_amax
        else:
            # Cache the constant scale tensor: allocating fresh each call
            # would trigger a CPU->GPU copy during CUDA-graph capture
            # (``torch.tensor(1.0, device=...)`` reads from a CPU scalar).
            # Warmup runs before capture, so by the time the graph records
            # this op, the buffer already exists.
            if (
                self._offline_per_tensor_scale is None
                or self._offline_per_tensor_scale.device != x.device
            ):
                self._offline_per_tensor_scale = torch.ones(
                    (), device=x.device, dtype=torch.float32
                )
            x_scale = self._offline_per_tensor_scale
        x_fp8 = (x * x_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        return x_fp8, x_scale

    def _quantize_activation_fp8_blockwise(
        self, x: torch.Tensor, block_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activation to FP8 with block-wise (per-token-group) scales.

        When online_act_quant=True, computes scales from actual data.
        When online_act_quant=False, uses a fixed default scale.

        Returns (x_fp8, x_scale) where x_scale shape is (m, k // block_size).
        """
        m, k = x.shape
        fp8_max = 448.0
        k_pad = ((k + block_size - 1) // block_size) * block_size

        if self.online_act_quant:
            # Skip the padding+copy path when K is already a multiple of
            # block_size (the common case for WAN: K ∈ {1536, 5120, 13824}
            # are all divisible by 128).
            if k == k_pad:
                x_view = x.view(m, -1, block_size)
                # amax in the activation dtype; promoting to fp32 first
                # doubles memory traffic for no precision benefit.
                x_amax = x_view.abs().amax(dim=2).to(torch.float32).clamp(min=1e-4)
                x_scale = x_amax / fp8_max  # (m, k_blocks)
                x_scaled = (x_view / x_scale.unsqueeze(2)).to(torch.float8_e4m3fn)
                x_fp8 = x_scaled.view(m, k_pad).contiguous()
            else:
                x_padded = torch.zeros(m, k_pad, dtype=x.dtype, device=x.device)
                x_padded[:, :k] = x
                x_view = x_padded.view(m, -1, block_size)
                x_amax = x_view.abs().amax(dim=2).to(torch.float32).clamp(min=1e-4)
                x_scale = x_amax / fp8_max
                x_scaled = (x_view / x_scale.unsqueeze(2)).to(torch.float8_e4m3fn)
                x_fp8 = x_scaled.view(m, k_pad)[:, :k].contiguous()
        else:
            default_scale_val = 1.0 / fp8_max
            x_fp8 = (x * fp8_max).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            num_blocks = k_pad // block_size
            # Same caching trick as the per-tensor path: ``torch.full(...)``
            # allocates new GPU memory each call, which breaks cuda-graph
            # replay. Reuse a single buffer keyed on the (m, num_blocks)
            # shape (constant across replays for a fixed input shape).
            cache = self._offline_blockwise_scale
            if (
                cache is None
                or cache.shape != (m, num_blocks)
                or cache.device != x.device
            ):
                cache = torch.full(
                    (m, num_blocks),
                    default_scale_val,
                    dtype=torch.float32,
                    device=x.device,
                )
                self._offline_blockwise_scale = cache
            x_scale = cache

        return x_fp8, x_scale

    def _forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer mm_fp8."""
        if not self._fp8_prepared:
            self._prepare_fp8_weights()

        x_fp8, x_scale = self._quantize_activation_fp8_per_tensor(x)

        # Compute output scale
        alpha = (1.0 / x_scale) * self._weight_scale

        # Run FP8 GEMM
        out = flashinfer.mm_fp8(x_fp8, self._weight_fp8, alpha=alpha)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        # Cast back to input dtype for consistency with other backends and to
        # avoid unexpected upcasts in the surrounding model graph.
        return out.to(x.dtype)

    def _forward_fp4(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer mm_fp4."""
        if not self._fp4_prepared:
            self._prepare_fp4_weights()

        # ``.is_contiguous()`` is a cheap stride check; we only pay for the
        # actual copy when needed. The reshape inside ``forward`` already
        # produces a contiguous tensor for typical inputs.
        x_contig = x if x.is_contiguous() else x.contiguous()

        # NVFP4 needs a per-tensor global scale factor. Online: recompute
        # from the current activation's amax (.float()/.nan_to_num() were
        # the culprit in the original wrapper; .abs().amax() is already
        # cheap, see the wrapper-hot-paths note in wan/BENCHMARK.md).
        # Offline: use a fixed default global SF — the same shape as the
        # FP8 offline path, so calibration / overrides can later swap in
        # a real per-layer constant.
        if self.online_act_quant:
            x_amax = x_contig.abs().amax().to(torch.float32).clamp(min=1e-12)
            x_global_sf = (448.0 * 6.0) / x_amax
        else:
            # Cache the constant SF tensor so cuda-graph capture doesn't
            # re-allocate or copy from CPU on each call. The value is
            # NVFP4's E4M3 absmax (448 * 6) and would be a per-layer
            # calibration constant in a real offline-quant pipeline.
            if (
                self._offline_fp4_global_sf is None
                or self._offline_fp4_global_sf.device != x.device
            ):
                self._offline_fp4_global_sf = torch.full(
                    (), 448.0 * 6.0, device=x.device, dtype=torch.float32
                )
            x_global_sf = self._offline_fp4_global_sf

        # Quantize input to FP4 — shape (M, K) -> (M, K/2)
        x_fp4, x_descale = flashinfer.nvfp4_quantize(x_contig, x_global_sf)

        # alpha = 1 / (a_global_sf * b_global_sf); see mm_fp4 docstring.
        alpha = (1.0 / (x_global_sf * self._weight_global_sf)).to(torch.float32)

        # mm_fp4 wants ``b`` shape (K, N) column-major and ``b_descale``
        # transposed to match — both via .T views (see tests/gemm/test_mm_fp4.py).
        # The ``backend`` kwarg defaults to "auto" inside mm_fp4; only pass
        # an explicit value when the caller asked for one to keep the kernel's
        # own dispatch logic intact.
        mm_fp4_kwargs = {}
        if self.kernel_backend is not None:
            mm_fp4_kwargs["backend"] = self.kernel_backend
        out = flashinfer.mm_fp4(
            x_fp4,
            self._weight_fp4.T,
            x_descale,
            self._weight_descale.T,
            alpha=alpha,
            out_dtype=x.dtype,
            **mm_fp4_kwargs,
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out

    def _forward_fp8_sm90(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer fp8_blockscale_gemm_sm90 (FP8+FP8 W8A8).

        Input is assumed BF16. We manually pre-quantize it to FP8 with
        per-token 128-block scales (the only input-scale layout this kernel
        accepts) — this honors ``self.online_act_quant`` the same way the
        groupwise/blockscaled paths do, instead of going through the
        kernel's internal BF16 quantization (which can't be steered from
        Python).
        """
        if not self._fp8_sm90_prepared:
            self._prepare_fp8_sm90_weights()

        x_fp8, input_scale = self._quantize_activation_fp8_blockwise(x, block_size=128)
        out = flashinfer.gemm.fp8_blockscale_gemm_sm90(
            x_fp8,
            self._weight_fp8_sm90,
            input_scale=input_scale,
            weight_scale=self._weight_scale_sm90,
            out_dtype=torch.bfloat16,
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_bmm_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer bmm_fp8.

        Uses per-tensor FP8 quantization. The ``backend`` argument
        (``cublas`` (kernel default) / ``cudnn`` / ``cutlass`` / ``auto``) is
        picked from ``self.kernel_backend`` when set, otherwise we keep
        ``"cublas"`` which is the long-standing default for this example.
        """
        if not self._bmm_fp8_prepared:
            self._prepare_bmm_fp8_weights()

        x_fp8, x_scale = self._quantize_activation_fp8_per_tensor(x)

        # bmm_fp8 expects (batch, M, K) input
        x_fp8_batch = x_fp8.unsqueeze(0)  # (1, M, K)

        # Compute scales for output
        a_scale = (1.0 / x_scale).to(torch.float32).unsqueeze(0)  # (1,)
        b_scale = self._weight_scale_bmm.unsqueeze(0)  # (1,)

        backend = self.kernel_backend if self.kernel_backend is not None else "cublas"
        # Run bmm_fp8: (1, M, K) @ (1, K, N) -> (1, M, N)
        out = flashinfer.gemm.bmm_fp8(
            x_fp8_batch,
            self._weight_fp8_bmm,
            a_scale,
            b_scale,
            dtype=torch.bfloat16,
            backend=backend,
        )

        # Remove batch dimension
        out = out.squeeze(0)  # (M, N)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_fp8_groupwise(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer gemm_fp8_nt_groupwise.

        NT GEMM: a(m,k) @ b(n,k)^T with groupwise (1, 128, 128) block scales.
        The cutlass backend requires a non-None ``scale_major_mode``; we use
        ``"MN"``, which means scales are laid out with k-block as the leading
        dimension. ``_weight_scale_nt`` is already prepared in MN-major layout
        (``(k_blocks, n_blocks)``) by ``_prepare_fp8_nt_weights``; we transpose
        the per-token activation scale to match.
        """
        if not self._fp8_nt_prepared:
            self._prepare_fp8_nt_weights()

        x_fp8, x_scale = self._quantize_activation_fp8_blockwise(x, block_size=128)
        # x_scale is (m, k_blocks) K-major; convert to (k_blocks, m) MN-major.
        x_scale_mn = x_scale.t().contiguous()

        # gemm_fp8_nt_groupwise's ``backend`` accepts ``cutlass`` (default in
        # the API; the historical pick here) or ``trtllm``. Honor an explicit
        # ``self.kernel_backend`` if set.
        backend = self.kernel_backend if self.kernel_backend is not None else "cutlass"
        out = gemm_fp8_nt_groupwise(
            x_fp8,
            self._weight_fp8_nt,
            x_scale_mn,
            self._weight_scale_nt,
            scale_major_mode="MN",
            scale_granularity_mnk=(1, 128, 128),
            out_dtype=torch.bfloat16,
            backend=backend,
        )

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_fp8_blockscaled(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer gemm_fp8_nt_blockscaled.

        Same as groupwise but fixed (128, 128, 128) scale granularity.
        """
        if not self._fp8_nt_prepared:
            self._prepare_fp8_nt_weights()

        x_fp8, x_scale = self._quantize_activation_fp8_blockwise(x, block_size=128)

        # gemm_fp8_nt_blockscaled: wrapper over groupwise with (128,128,128) granularity
        # a_scale for blockscaled: MN-major -> (k_blocks, m_blocks)
        m = x.shape[0]
        block_size = 128
        m_blocks = (m + block_size - 1) // block_size
        k_blocks = x_scale.shape[1]

        # Reshape per-token scale to per-block scale for 128x128 blocking
        x_scale_padded = torch.zeros(
            m_blocks * block_size,
            k_blocks,
            dtype=torch.float32,
            device=x.device,
        )
        x_scale_padded[:m] = x_scale
        x_scale_block = (
            x_scale_padded.view(m_blocks, block_size, k_blocks).amax(
                dim=1
            )  # (m_blocks, k_blocks)
        )
        # MN-major: (k_blocks, m_blocks)
        a_scale_mn = x_scale_block.t().contiguous()

        out = gemm_fp8_nt_blockscaled(
            x_fp8,
            self._weight_fp8_nt,
            a_scale_mn,
            self._weight_scale_nt,
            scale_major_mode="MN",
            out_dtype=torch.bfloat16,
        )

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_batch_deepgemm_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer batch_deepgemm_fp8_nt_groupwise.

        Treats the linear as a single-group batched GEMM (batch_size=1).
        """
        if not self._fp8_nt_prepared:
            self._prepare_fp8_nt_weights()

        m = x.shape[0]
        x_fp8, x_scale = self._quantize_activation_fp8_blockwise(x, block_size=128)

        # batch_deepgemm expects (batch, m, k) inputs
        a = x_fp8.unsqueeze(0)  # (1, m, k)
        b = self._weight_fp8_nt.unsqueeze(0)  # (1, n, k)
        a_scale_batch = x_scale.unsqueeze(0)  # (1, m, k_blocks)
        b_scale_batch = self._weight_scale_nt.unsqueeze(
            0
        )  # (1, k_blocks, n_blocks) -> need (1, n_blocks, k_blocks)
        # batch_deepgemm b_scale expects (batch, n_blocks, k_blocks)
        b_scale_batch = b_scale_batch.transpose(1, 2).contiguous()
        masked_m = torch.tensor([m], dtype=torch.int32, device=x.device)

        out = batch_deepgemm_fp8_nt_groupwise(
            a,
            b,
            a_scale_batch,
            b_scale_batch,
            masked_m=masked_m,
            expected_m=m,
            scale_granularity_mnk=(1, 128, 128),
            out_dtype=torch.bfloat16,
        )

        out = out.squeeze(0)  # (m, n)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_bmm_bf16(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer bmm_bf16.

        ``backend`` accepts ``cudnn`` (kernel default), ``cutlass``, or
        ``auto``; pass ``self.kernel_backend`` through when set.
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
        bmm_bf16_kwargs = {}
        if self.kernel_backend is not None:
            bmm_bf16_kwargs["backend"] = self.kernel_backend
        out = flashinfer.bmm_bf16(x_batch, weight_bf16, **bmm_bf16_kwargs)

        # Remove batch dimension
        out = out.squeeze(0)  # (M, N)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_mxfp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer mm_mxfp8.

        Uses MXFP8 quantization with block size 32. ``backend`` accepts
        ``cutlass``, ``cute-dsl``, ``trtllm``, or ``auto`` (kernel default);
        pass ``self.kernel_backend`` through when set.
        """
        if not self._mxfp8_prepared:
            self._prepare_mxfp8_weights()

        # Quantize input to MXFP8 — shape (M, K)
        x_mxfp8, x_scale = flashinfer.mxfp8_quantize(
            x.contiguous(), is_sf_swizzled_layout=True
        )

        # mm_mxfp8 wants ``b`` shape (K, N) column-major; pass weight.T.
        mm_mxfp8_kwargs = {}
        if self.kernel_backend is not None:
            mm_mxfp8_kwargs["backend"] = self.kernel_backend
        out = flashinfer.mm_mxfp8(
            x_mxfp8,
            self._weight_mxfp8.T,
            x_scale,
            self._weight_scale_mxfp8,
            out_dtype=torch.bfloat16,
            **mm_mxfp8_kwargs,
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(x.dtype)

    def _forward_bmm_mxfp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FlashInfer bmm_mxfp8.

        Uses batched MXFP8 quantization with block size 32. ``backend``
        accepts ``cudnn``, ``cutlass``, or ``auto`` (kernel default); pass
        ``self.kernel_backend`` through when set.
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
        bmm_mxfp8_kwargs = {}
        if self.kernel_backend is not None:
            bmm_mxfp8_kwargs["backend"] = self.kernel_backend
        out = flashinfer.bmm_mxfp8(
            x_batch,
            self._weight_mxfp8_bmm,
            x_scale_batch,
            self._weight_scale_mxfp8_bmm,
            dtype=torch.bfloat16,
            **bmm_mxfp8_kwargs,
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


def create_linear_layer(
    in_features: int,
    out_features: int,
    bias: bool = True,
    gemm_backend: str = "torch",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    online_act_quant: bool = True,
) -> nn.Module:
    """
    Factory function to create a linear layer with the specified GEMM backend.

    If gemm_backend (after stripping any ``"-<kernel>"`` suffix) is ``"torch"``,
    returns a standard ``nn.Linear``. Otherwise, returns ``FlashInferLinear``
    with the full ``gemm_backend`` string — see ``FlashInferLinear``'s
    docstring for the accepted ``"base-kernel"`` syntax.
    """
    base_backend, _ = _split_backend_choice(gemm_backend)
    if base_backend == "torch":
        return nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
    return FlashInferLinear(
        in_features,
        out_features,
        bias=bias,
        backend=gemm_backend,
        device=device,
        dtype=dtype,
        online_act_quant=online_act_quant,
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

    FlashInfer's gelu_and_mul is for gated FFN (gelu(gate) * up). For
    non-gated FFNs, this uses PyTorch's F.gelu because FlashInfer doesn't have
    a standalone GELU kernel.

    Args:
        dim: Input/output dimension
        inner_dim: Hidden dimension
        activation_fn: Activation function ("gelu-approximate", "gelu")
        bias: Whether to use bias in linear layers
        use_gating: Whether to use gated FFN (gelu(x) * gate)
        gemm_backend: GEMM backend for linear layers (see GEMMBackend enum)
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
        use_gating: bool = False,  # Whether to use gated FFN (gelu(x) * gate)
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.use_gating = use_gating
        self.gemm_backend = gemm_backend

        linear_kwargs = dict(
            bias=bias,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )

        if use_gating:
            self.proj_up = create_linear_layer(dim, 2 * inner_dim, **linear_kwargs)
        else:
            self.proj_up = create_linear_layer(dim, inner_dim, **linear_kwargs)

        self.proj_down = create_linear_layer(inner_dim, dim, **linear_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        approximate = "tanh" if self.activation_fn == "gelu-approximate" else "none"
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
                hidden = F.gelu(gate, approximate=approximate) * up
        else:
            # Simple FFN: up -> activation -> down
            hidden = self.proj_up(x)
            hidden = F.gelu(hidden, approximate=approximate)

        return self.proj_down(hidden)


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


class FlashInferAttentionDispatcher(nn.Module):
    """Model-independent FlashInfer attention backend dispatcher.

    The ``attention_backend`` string picks one of the four supported paths:
    ``"single"`` / ``"cudnn"`` / ``"trtllm"`` (FlashInfer kernels) and
    ``"torch"`` (``F.scaled_dot_product_attention``). ``"auto"`` selects
    ``"single"`` for ``batch_size == 1`` and ``"cudnn"`` otherwise — that
    pick remains the default because it matches the original WAN example.
    ``"torch"`` is provided as a hardware-agnostic fallback for GPUs where
    the FlashInfer attention kernels aren't supported, and can be selected
    explicitly via ``attention_backend="torch"``.

    A ``"-<kernel>"`` suffix on the ``"single"`` path is forwarded as the
    underlying ``single_prefill_with_kv_cache`` kernel's own ``backend``
    kwarg (the kernel itself routes over FA2 / FA3 / cuDNN / cutlass /
    trtllm-gen / ...). Examples: ``"single"`` (kernel-default ``"auto"``),
    ``"single-fa3"``, ``"single-fa2"``, ``"single-cudnn"``. The other
    paths ignore the suffix.
    """

    def __init__(
        self,
        heads: int = 8,
        dim_head: int = 64,
        attention_backend: str = "auto",
        use_skip_softmax_sparse: bool = False,
        skip_softmax_threshold_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        base_backend, self.single_backend = _split_backend_choice(attention_backend)
        if base_backend not in _VALID_ATTENTION_BACKENDS:
            raise ValueError(
                f"Unsupported attention backend {attention_backend!r}; "
                f"expected base name in {sorted(_VALID_ATTENTION_BACKENDS)} "
                f"with an optional '-<kernel>' suffix."
            )
        if self.single_backend is not None and base_backend != "single":
            warnings.warn(
                f"Attention backend {attention_backend!r}: the '-<kernel>' "
                f"suffix only takes effect on the 'single' path; ignoring "
                f"the {self.single_backend!r} hint on {base_backend!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.single_backend = None
        self.attention_backend = base_backend
        self.use_skip_softmax_sparse = use_skip_softmax_sparse
        self.skip_softmax_threshold_scale_factor = skip_softmax_threshold_scale_factor

        self._workspace_buffer: Optional[torch.Tensor] = None

    def _get_workspace_buffer(self, device: torch.device) -> torch.Tensor:
        if self._workspace_buffer is None or self._workspace_buffer.device != device:
            self._workspace_buffer = torch.zeros(
                _DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device
            )
        return self._workspace_buffer

    def _resolve_attention_backend(
        self, batch_size: int, device: torch.device
    ) -> Tuple[str, bool]:
        backend = self.attention_backend
        if backend == "auto":
            backend = "single" if batch_size == 1 else "cudnn"
        use_sparse = False

        # "torch" has no SM-version restriction and doesn't interact with
        # the sparse path; short-circuit before the FlashInfer-specific
        # checks below.
        if backend == "torch":
            return "torch", False

        major, minor = get_compute_capability(device)
        sm_version = major * 10 + minor

        if self.use_skip_softmax_sparse:
            if sm_version in (100, 103):
                return "trtllm", True
            warnings.warn(
                f"Skip-softmax sparse attention requires SM100 or SM103 "
                f"(Blackwell B100/B200), but current GPU is SM{sm_version}. "
                f"Falling back to {backend!r} with standard (non-sparse) attention.",
                RuntimeWarning,
                stacklevel=2,
            )

        # The TRT-LLM FMHA runner (include/flashinfer/trtllm/fmha/fmhaRunner.cuh)
        # only supports SM100/SM103 (Blackwell). On any other SM, constructing
        # the runner raises "Unsupported architecture", so silently swap to a
        # supported backend instead of crashing.
        if backend == "trtllm" and sm_version not in (100, 103):
            fallback = "single" if batch_size == 1 else "cudnn"
            warnings.warn(
                f"'trtllm' attention backend requires SM100/SM103 (Blackwell), "
                f"but current GPU is SM{sm_version}; falling back to {fallback!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = fallback

        return backend, use_sparse

    def _attention_single(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
    ) -> torch.Tensor:
        if batch_size != 1:
            raise ValueError(
                "'single' attention backend requires batch_size == 1. "
                "Use 'cudnn' or 'trtllm' for batched inputs."
            )
        single_kwargs = {}
        if self.single_backend is not None:
            single_kwargs["backend"] = self.single_backend
        out = single_prefill_with_kv_cache(
            query.squeeze(0).contiguous(),
            key.squeeze(0).contiguous(),
            value.squeeze(0).contiguous(),
            causal=False,
            **single_kwargs,
        )
        return out.view(1, seq_len_q, -1)

    def _attention_torch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
    ) -> torch.Tensor:
        """Reference path using ``F.scaled_dot_product_attention``.

        Inputs are ``(B, S, H, D)`` to match the FlashInfer-side layout. SDPA
        expects ``(B, H, S, D)``, so we transpose in / out. This path is
        intended as the universal fallback when none of the FlashInfer
        attention backends are available for the current GPU; it is also
        the most fusable target for ``torch.compile`` since dynamo can
        trace right through SDPA.
        """
        # (B, S, H, D) -> (B, H, S, D)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # (B, H, S, D) -> (B, S, H, D) -> (B, S, H*D)
        return out.transpose(1, 2).contiguous().reshape(batch_size, seq_len_q, -1)

    def _attention_cudnn_batch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
    ) -> torch.Tensor:
        device = query.device

        query_flat = query.reshape(
            batch_size * seq_len_q, self.heads, self.dim_head
        ).contiguous()
        k_flat = key.reshape(
            batch_size * seq_len_kv, self.heads, self.dim_head
        ).contiguous()
        v_flat = value.reshape(
            batch_size * seq_len_kv, self.heads, self.dim_head
        ).contiguous()

        # cudnn expects the per-sequence length tensors as 4-D `(batch, 1, 1, 1)`
        # (see tests/attention/test_cudnn_prefill.py). Passing 1-D `(batch,)`
        # triggers "seqLenQDesc.getNbDims() != 4, CUDNN_STATUS_BAD_PARAM".
        actual_seq_lens_q = torch.full(
            (batch_size, 1, 1, 1), seq_len_q, dtype=torch.int32, device=device
        )
        actual_seq_lens_kv = torch.full(
            (batch_size, 1, 1, 1), seq_len_kv, dtype=torch.int32, device=device
        )

        workspace = self._get_workspace_buffer(device)
        sm_scale = 1.0 / math.sqrt(self.dim_head)

        # cudnn ragged-offset descriptors must have length batch_size + 1
        # (i.e. cumulative start offsets of each request plus the final endpoint).
        # Passing length batch_size hits CUDNN_STATUS_BAD_PARAM in
        # _build_prefill_graph: "ragged dim should match dim value + 1 of original tensor".
        batch_offsets_q = torch.arange(
            0,
            (batch_size + 1) * seq_len_q,
            seq_len_q,
            dtype=torch.int32,
            device=device,
        )
        batch_offsets_kv = torch.arange(
            0,
            (batch_size + 1) * seq_len_kv,
            seq_len_kv,
            dtype=torch.int32,
            device=device,
        )

        out, _ = cudnn_batch_prefill_with_kv_cache(
            q=query_flat,
            k_cache=k_flat,
            v_cache=v_flat,
            scale=sm_scale,
            workspace_buffer=workspace,
            max_token_per_sequence=seq_len_q,
            max_sequence_kv=seq_len_kv,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            causal=False,
            return_lse=False,
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_q,
            batch_offsets_k=batch_offsets_kv,
            batch_offsets_v=batch_offsets_kv,
        )

        return out.view(batch_size, seq_len_q, -1)

    # TRT-LLM gen FMHA cubins for context attention with head_dim=128 are
    # compiled for a small fixed set of page sizes (16/32/64). Picking 64 keeps
    # the page count low for typical diffusion-model sequence lengths.
    _TRTLLM_PAGE_SIZE = 64

    def _attention_trtllm_batch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
        use_sparse: bool = False,
    ) -> torch.Tensor:
        device = query.device

        # Chunk seq_len_kv into pages of _TRTLLM_PAGE_SIZE so the kv_cache shape
        # matches the trtllm-gen cubin layout (num_pages, 2, H, page_size, D).
        # The previous "one page per request" layout produced numTokensPerPage
        # == seq_len_kv (e.g. 1024 for WAN), which has no matching cubin.
        page_size = self._TRTLLM_PAGE_SIZE
        padded_seq_kv = ((seq_len_kv + page_size - 1) // page_size) * page_size
        num_pages_per_seq = padded_seq_kv // page_size

        if padded_seq_kv != seq_len_kv:
            # pad along the seq dim (dim=1) with zeros; cubin still reads the
            # full page but seq_lens tells it the real length.
            pad_amount = padded_seq_kv - seq_len_kv
            # key/value are (B, seq_kv, H, D); pad last-1 dim from the right
            key = F.pad(key, (0, 0, 0, 0, 0, pad_amount))
            value = F.pad(value, (0, 0, 0, 0, 0, pad_amount))

        num_kv_heads = key.shape[2]
        head_dim = key.shape[3]

        # (B, padded_seq, H, D) -> (B, num_pages, page_size, H, D)
        # -> (B, num_pages, H, page_size, D) -> (B * num_pages, H, page_size, D)
        key_pages = (
            key.view(batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(batch_size * num_pages_per_seq, num_kv_heads, page_size, head_dim)
        )
        value_pages = (
            value.view(batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(batch_size * num_pages_per_seq, num_kv_heads, page_size, head_dim)
        )
        # (num_pages_total, 2, H, page_size, D), HND layout
        kv_cache = torch.stack([key_pages, value_pages], dim=1)

        # Each batch element references its contiguous slice of the page table.
        block_tables = torch.arange(
            batch_size * num_pages_per_seq, dtype=torch.int32, device=device
        ).view(batch_size, num_pages_per_seq)

        # seq_lens carries the real (unpadded) KV length per request.
        seq_lens = torch.full(
            (batch_size,), seq_len_kv, dtype=torch.int32, device=device
        )

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

        workspace = self._get_workspace_buffer(device)
        query_flat = query.reshape(
            batch_size * seq_len_q, self.heads, self.dim_head
        ).contiguous()

        sm_scale = 1.0 / math.sqrt(self.dim_head)
        skip_threshold = (
            self.skip_softmax_threshold_scale_factor if use_sparse else None
        )

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
            skip_softmax_threshold_scale_factor=skip_threshold,
        )
        return hidden_states.view(batch_size, seq_len_q, -1)

    def _dispatch_attention(
        self,
        backend: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
        use_sparse: bool = False,
    ) -> torch.Tensor:
        if backend == "single":
            return self._attention_single(
                query, key, value, batch_size, seq_len_q, seq_len_kv
            )
        if backend == "trtllm":
            return self._attention_trtllm_batch(
                query, key, value, batch_size, seq_len_q, seq_len_kv, use_sparse
            )
        if backend == "cudnn":
            return self._attention_cudnn_batch(
                query, key, value, batch_size, seq_len_q, seq_len_kv
            )
        if backend == "torch":
            return self._attention_torch(
                query, key, value, batch_size, seq_len_q, seq_len_kv
            )
        raise ValueError(
            f"Unsupported resolved attention backend {backend!r}; "
            f"expected one of {sorted(_VALID_ATTENTION_BACKENDS - {'auto'})}."
        )
