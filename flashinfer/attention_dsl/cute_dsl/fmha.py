# Copyright (c) 2025 by FlashInfer team.
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
"""
CuTe DSL FMHA Kernel Loader (Cubin Distribution)
=================================================

Loads pre-compiled FMHA kernel .so artifacts via ExternalBinaryModule.
The .so files are compiled offline from the proprietary DSL kernel source
and distributed through the cubin_publishing pipeline.

Runtime flow:
    1. get_artifact() downloads .so from artifactory (or local cache)
    2. ExternalBinaryModule loads .so and extracts callable kernel
    3. cute_dsl_fmha_prefill() wraps the kernel with a PyTorch-friendly API
"""

import functools
import logging
import math
import os
from typing import Optional

import torch

logger = logging.getLogger("flashinfer.attention.cute_dsl.fmha")


# =============================================================================
# Artifact configuration
# =============================================================================

# These will be updated when cubins are published to artifactory.
# For now they serve as placeholders for local development/testing.
DSL_FMHA_ARTIFACT_PATH = os.environ.get(
    "FLASHINFER_DSL_FMHA_ARTIFACT_PATH",
    "",  # Will be set to e.g. "<commit_hash>/fmha/cute-dsl/" once published
)

# Map: variant_name -> sha256 of the .so file
# Updated by cubin publishing pipeline.
DSL_FMHA_CHECKSUMS: dict[str, str] = {
    # Example: "cute_dsl_fmha_fp16_h128_causal_persistent": "<sha256>",
}


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch dtype to short string matching compile_cute_dsl_fmha.py naming."""
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float8_e4m3fn: "e4m3",
    }[dtype]


def _get_variant_name(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    is_persistent: bool = True,
    varlen: bool = False,
) -> str:
    """Generate the variant name matching compile_cute_dsl_fmha.py naming convention."""
    in_str = _dtype_to_str(in_dtype)
    out_str = _dtype_to_str(out_dtype)
    # Only include out_dtype in name when it differs from in_dtype (mixed precision)
    dtype_str = f"{in_str}_{out_str}" if in_dtype != out_dtype else in_str
    causal_str = "causal" if is_causal else "nocausal"
    persist_str = "persistent" if is_persistent else "nonpersistent"
    varlen_str = "_varlen" if varlen else ""
    return (
        f"cute_dsl_fmha_{dtype_str}_h{head_dim}_{causal_str}_{persist_str}{varlen_str}"
    )


# =============================================================================
# Loading: ExternalBinaryModule path
# =============================================================================


def _load_from_artifact(variant_name: str, enable_tvm_ffi: bool = False):
    """Download .so from artifactory and load via ExternalBinaryModule.

    This is the production path used when cubins are published.

    Parameters
    ----------
    variant_name : str
        The kernel variant name (matches function_prefix used during export).
    enable_tvm_ffi : bool
        If False (default), load with CuTe native ABI.
        If True, load with TVM-FFI ABI (TODO: compile-side support pending).
    """
    import cutlass.cute as cute
    from flashinfer.jit.cubin_loader import get_artifact
    from flashinfer.jit.env import FLASHINFER_CUBIN_DIR

    so_filename = f"{variant_name}.so"
    artifact_path = f"{DSL_FMHA_ARTIFACT_PATH}/{so_filename}"
    sha256 = DSL_FMHA_CHECKSUMS.get(variant_name, "")

    if not sha256:
        raise RuntimeError(
            f"No checksum registered for DSL FMHA variant '{variant_name}'. "
            f"Available variants: {list(DSL_FMHA_CHECKSUMS.keys())}"
        )

    # Download to local cache
    local_path = FLASHINFER_CUBIN_DIR / artifact_path
    data = get_artifact(artifact_path, sha256)
    if not data:
        raise RuntimeError(f"Failed to download DSL FMHA artifact: {artifact_path}")

    # Ensure .so is written to disk (get_artifact caches in CUBIN_DIR)
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)

    module = cute.runtime.load_module(str(local_path), enable_tvm_ffi=enable_tvm_ffi)
    return getattr(module, variant_name)


def _load_from_local(variant_name: str, local_dir: str, enable_tvm_ffi: bool = False):
    """Load .so or .o from a local directory (for development/testing).

    Set FLASHINFER_DSL_FMHA_LOCAL_DIR to use this path.

    Parameters
    ----------
    variant_name : str
        The kernel variant name (matches function_prefix used during export).
    local_dir : str
        Directory containing the compiled .so/.o files.
    enable_tvm_ffi : bool
        If False (default), load with CuTe native ABI.
        If True, load with TVM-FFI ABI (TODO: compile-side support pending).
    """
    import cutlass.cute as cute

    # Try .so first, then .o
    so_path = os.path.join(local_dir, f"{variant_name}.so")
    o_path = os.path.join(local_dir, f"{variant_name}.o")
    if os.path.exists(so_path):
        load_path = so_path
    elif os.path.exists(o_path):
        load_path = o_path
    else:
        raise FileNotFoundError(
            f"DSL FMHA .so/.o not found at {local_dir}/{variant_name}.[so|o]. "
            f"Run compile_cute_dsl_fmha.py to generate it."
        )

    module = cute.runtime.load_module(load_path, enable_tvm_ffi=enable_tvm_ffi)
    return getattr(module, variant_name)


@functools.cache
def get_cute_dsl_fmha_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    is_persistent: bool = True,
    enable_tvm_ffi: bool = False,
    varlen: bool = False,
):
    """Get a compiled DSL FMHA kernel function.

    Checks local directory first (FLASHINFER_DSL_FMHA_LOCAL_DIR env var),
    then falls back to artifact download.

    Parameters
    ----------
    in_dtype : torch.dtype
        Input data type (torch.float16, torch.bfloat16, or torch.float8_e4m3fn).
    out_dtype : torch.dtype
        Output data type. Same as in_dtype for non-mixed precision.
    head_dim : int
        Head dimension (e.g., 64, 128, 192). Note: 192 only supports FP8.
    is_causal : bool
        Whether to use causal masking.
    is_persistent : bool
        Whether to use persistent kernel mode.
    enable_tvm_ffi : bool
        If False (default), load with CuTe native ABI — kernel accepts
        cute Pointer/Tensor args (same calling convention as JIT mode).
        If True, load with TVM-FFI ABI — kernel accepts torch.Tensor
        directly (TODO: compile-side support pending).

    Returns
    -------
    callable
        The compiled kernel function.
    """
    variant_name = _get_variant_name(
        in_dtype, out_dtype, head_dim, is_causal, is_persistent, varlen
    )

    # Check for local .so directory (development mode)
    local_dir = os.environ.get("FLASHINFER_DSL_FMHA_LOCAL_DIR")
    if local_dir:
        logger.info(
            f"Loading DSL FMHA kernel from local dir: {local_dir} (tvm_ffi={enable_tvm_ffi})"
        )
        return _load_from_local(variant_name, local_dir, enable_tvm_ffi=enable_tvm_ffi)

    # Production path: download from artifactory
    logger.info(
        f"Loading DSL FMHA kernel variant: {variant_name} (tvm_ffi={enable_tvm_ffi})"
    )
    return _load_from_artifact(variant_name, enable_tvm_ffi=enable_tvm_ffi)


# =============================================================================
# PyTorch API wrapper
# =============================================================================


def _to_cute_tensor(t: torch.Tensor, assumed_align: int = 16):
    """Convert a torch tensor to a cute tensor, handling FP8 dtypes.

    DLPack does not support FP8 types directly, so we view as int8 first,
    then override the element_type on the cute side.
    """
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    _TORCH_TO_CUTLASS_FP8 = {
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }

    if t.dtype in _TORCH_TO_CUTLASS_FP8:
        cutlass_dtype = _TORCH_TO_CUTLASS_FP8[t.dtype]
        cute_t = from_dlpack(t.view(torch.int8), assumed_align=assumed_align)
        cute_t.element_type = cutlass_dtype
        return cute_t
    return from_dlpack(t, assumed_align=assumed_align)


def cute_dsl_fmha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    window_right: int = -1,
    lse: Optional[torch.Tensor] = None,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    scale_o: float = 1.0,
) -> None:
    """Run DSL FMHA prefill kernel on the given tensors.

    Note: The DSL FMHA kernel only supports per-tensor scalar scales, not
    per-head scale tensors.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape (B, S_q, H_q, D).
    k : torch.Tensor
        Key tensor, shape (B, S_k, H_k, D).
    v : torch.Tensor
        Value tensor, shape (B, S_k, H_k, D_v).
    o : torch.Tensor
        Output tensor, shape (B, S_q, H_q, D_v). Modified in-place.
    is_causal : bool
        Whether to apply causal masking.
    sm_scale : float, optional
        Softmax scale factor. Defaults to 1/sqrt(D).
    window_left : int
        Left sliding window size. -1 means no window (attend to all left).
    window_right : int
        Right sliding window size. -1 means no window. 0 for causal.
    lse : torch.Tensor, optional
        Log-sum-exp output tensor, shape (B, H_q, S_q). None to skip.
    scale_q : float
        Per-tensor scale for query (FP8 calibration). Default 1.0.
    scale_k : float
        Per-tensor scale for key (FP8 calibration). Default 1.0.
    scale_v : float
        Per-tensor scale for value (FP8 calibration). Default 1.0.
    scale_o : float
        Per-tensor scale for output (FP8 calibration). Default 1.0.
    """
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.typing import Int32, Float32
    import cutlass.torch as cutlass_torch

    B, S_q, H_q, D = q.shape
    _, S_k, H_k, _ = k.shape
    D_v = v.shape[-1]

    head_dim = D
    in_dtype = q.dtype
    out_dtype = o.dtype

    # Get compiled kernel
    kernel_fn = get_cute_dsl_fmha_kernel(in_dtype, out_dtype, head_dim, is_causal)

    # Compute scale factors
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    log2_e = math.log2(math.exp(1.0))
    scale_softmax = scale_q * scale_k * sm_scale
    scale_softmax_log2 = scale_softmax * log2_e
    scale_output = scale_v / scale_o

    # Convert tensors to cute iterators
    q_cute = from_dlpack(q, assumed_align=16)
    k_cute = from_dlpack(k, assumed_align=16)
    v_cute = from_dlpack(v, assumed_align=16)
    o_cute = from_dlpack(o, assumed_align=16)

    # Problem size: (B, S_q, S_lse, S_k, H_q, H_k, D, D_v)
    problem_size = (B, S_q, S_q, S_k, H_q, H_k, D, D_v)

    # LSE
    lse_iter = None
    if lse is not None:
        lse_cute = from_dlpack(lse, assumed_align=16)
        lse_iter = lse_cute.iterator

    # Window sizes
    ws_left = None if window_left == -1 else Int32(window_left)
    ws_right = None if window_right == -1 else Int32(window_right)
    if is_causal and ws_right is None:
        ws_right = Int32(0)

    # Stream
    stream = cutlass_torch.default_stream()

    # Launch kernel
    kernel_fn(
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        o_cute.iterator,
        problem_size,
        None,  # cum_seqlen_q (fixed-length batch)
        None,  # cum_seqlen_k (fixed-length batch)
        lse_iter,
        Float32(scale_softmax_log2),
        Float32(scale_softmax),
        Float32(scale_output),
        None,  # skip_softmax_threshold_log2
        ws_left,
        ws_right,
        None,  # skip_softmax_count
        None,  # total_softmax_count
        stream,
    )


def cute_dsl_fmha_ragged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    window_right: int = -1,
    lse: Optional[torch.Tensor] = None,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    scale_o: float = 1.0,
) -> None:
    """Run DSL FMHA prefill kernel on ragged (variable-length) tensors.

    Note: The DSL FMHA kernel only supports per-tensor scalar scales, not
    per-head scale tensors.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape (total_q_tokens, H_q, D).
    k : torch.Tensor
        Key tensor, shape (total_kv_tokens, H_k, D).
    v : torch.Tensor
        Value tensor, shape (total_kv_tokens, H_k, D_v).
    o : torch.Tensor
        Output tensor, shape (total_q_tokens, H_q, D_v). Modified in-place.
    qo_indptr : torch.Tensor
        Cumulative sequence lengths for Q/O, shape (batch_size + 1,).
        Same as cum_seqlen_q in DSL FMHA kernel.
    kv_indptr : torch.Tensor
        Cumulative sequence lengths for K/V, shape (batch_size + 1,).
        Same as cum_seqlen_k in DSL FMHA kernel.
    is_causal : bool
        Whether to apply causal masking.
    sm_scale : float, optional
        Softmax scale factor. Defaults to 1/sqrt(D).
    window_left : int
        Left sliding window size. -1 means no window.
    window_right : int
        Right sliding window size. -1 means no window. 0 for causal.
    lse : torch.Tensor, optional
        Log-sum-exp output tensor. None to skip.
    scale_q : float
        Per-tensor scale for query (FP8 calibration). Default 1.0.
    scale_k : float
        Per-tensor scale for key (FP8 calibration). Default 1.0.
    scale_v : float
        Per-tensor scale for value (FP8 calibration). Default 1.0.
    scale_o : float
        Per-tensor scale for output (FP8 calibration). Default 1.0.
    """
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.typing import Int32, Float32
    import cutlass.torch as cutlass_torch

    total_q, H_q, D = q.shape
    total_kv, H_k, _ = k.shape
    D_v = v.shape[-1]

    batch_size = len(qo_indptr) - 1
    in_dtype = q.dtype
    out_dtype = o.dtype

    # Get compiled kernel (varlen=True for variable-length support)
    kernel_fn = get_cute_dsl_fmha_kernel(in_dtype, out_dtype, D, is_causal, varlen=True)

    # Compute scale factors
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    log2_e = math.log2(math.exp(1.0))
    scale_softmax = scale_q * scale_k * sm_scale
    scale_softmax_log2 = scale_softmax * log2_e
    scale_output = scale_v / scale_o

    # Compute max seq lengths for problem_size
    qo_indptr_cpu = qo_indptr.cpu()
    kv_indptr_cpu = kv_indptr.cpu()
    max_s_q = int((qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).max().item())
    max_s_k = int((kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]).max().item())

    # DSL FMHA kernel expects 4D tensor (B, S, H, D).
    # For variable length: B=1, S=total_tokens, with cum_seqlen indicating boundaries.
    # (matches example's convention: q_shape = (1, sum(s_q), H, D) for varlen)
    q_4d = q.unsqueeze(0)  # (1, total_q, H_q, D)
    k_4d = k.unsqueeze(0)  # (1, total_kv, H_k, D)
    v_4d = v.unsqueeze(0)  # (1, total_kv, H_k, D_v)
    o_4d = o.unsqueeze(0)  # (1, total_q, H_q, D_v)

    q_cute = from_dlpack(q_4d, assumed_align=16)
    k_cute = from_dlpack(k_4d, assumed_align=16)
    v_cute = from_dlpack(v_4d, assumed_align=16)
    o_cute = from_dlpack(o_4d, assumed_align=16)

    # cum_seqlen tensors
    cum_seqlen_q_cute = from_dlpack(qo_indptr.to(torch.int32), assumed_align=16)
    cum_seqlen_k_cute = from_dlpack(kv_indptr.to(torch.int32), assumed_align=16)

    # problem_size: (B, max_s_q, s_lse, max_s_k, H_q, H_k, D, D_v)
    s_lse = total_q  # for variable length, s_lse = total tokens
    problem_size = (batch_size, max_s_q, s_lse, max_s_k, H_q, H_k, D, D_v)

    # LSE
    lse_iter = None
    if lse is not None:
        lse_cute = from_dlpack(lse, assumed_align=16)
        lse_iter = lse_cute.iterator

    # Window sizes
    ws_left = None if window_left == -1 else Int32(window_left)
    ws_right = None if window_right == -1 else Int32(window_right)
    if is_causal and ws_right is None:
        ws_right = Int32(0)

    # Stream
    stream = cutlass_torch.default_stream()

    # Launch kernel
    kernel_fn(
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        o_cute.iterator,
        problem_size,
        cum_seqlen_q_cute,
        cum_seqlen_k_cute,
        lse_iter,
        Float32(scale_softmax_log2),
        Float32(scale_softmax),
        Float32(scale_output),
        None,  # skip_softmax_threshold_log2
        ws_left,
        ws_right,
        None,  # skip_softmax_count
        None,  # total_softmax_count
        stream,
    )
