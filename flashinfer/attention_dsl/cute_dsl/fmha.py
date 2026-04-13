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
    3. cute_dsl_fmha_ragged_prefill() wraps the kernel with a PyTorch-friendly API
"""

import functools
import logging
import math
import os
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda_driver
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32

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
    with_lse: bool = False,
    enable_tvm_ffi: bool = False,
) -> str:
    """Generate the variant name matching compile_cute_dsl_fmha.py naming convention."""
    in_str = _dtype_to_str(in_dtype)
    out_str = _dtype_to_str(out_dtype)
    # Only include out_dtype in name when it differs from in_dtype (mixed precision)
    dtype_str = f"{in_str}_{out_str}" if in_dtype != out_dtype else in_str
    causal_str = "causal" if is_causal else "nocausal"
    persist_str = "persistent" if is_persistent else "nonpersistent"
    varlen_str = "_varlen" if varlen else ""
    lse_str = "_lse" if with_lse else ""
    ffi_str = "_tvmffi" if enable_tvm_ffi else ""
    return f"cute_dsl_fmha_{dtype_str}_h{head_dim}_{causal_str}_{persist_str}{varlen_str}{lse_str}{ffi_str}"


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
    with_lse: bool = False,
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
        If True, load with TVM-FFI ABI — Pointer args accept data_ptr(),
        Tensor args accept torch.Tensor directly, stream uses env stream.

    Returns
    -------
    callable
        The compiled kernel function.
    """
    variant_name = _get_variant_name(
        in_dtype,
        out_dtype,
        head_dim,
        is_causal,
        is_persistent,
        varlen,
        with_lse,
        enable_tvm_ffi,
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
    enable_tvm_ffi: bool = True,
    max_qo_len: Optional[int] = None,
    max_kv_len: Optional[int] = None,
    kernel_fn=None,
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
    enable_tvm_ffi : bool
        If True, use TVM-FFI ABI (pass data_ptr() for Pointer args, torch.Tensor
        for Tensor args, no explicit stream). Default False (CuTe native ABI).
    max_qo_len : int, optional
        Maximum query sequence length. Computed from qo_indptr if not provided.
        Pass this from plan() to avoid D2H copy during CUDA graph capture.
    max_kv_len : int, optional
        Maximum KV sequence length. Computed from kv_indptr if not provided.
    """
    total_q, H_q, D = q.shape
    total_kv, H_k, _ = k.shape
    D_v = v.shape[-1]

    batch_size = len(qo_indptr) - 1

    if kernel_fn is None:
        kernel_fn = get_cute_dsl_fmha_kernel(
            q.dtype,
            o.dtype,
            D,
            is_causal,
            is_persistent=not is_causal,
            varlen=True,
            enable_tvm_ffi=enable_tvm_ffi,
            with_lse=lse is not None,
        )

    # Compute scale factors
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    log2_e = math.log2(math.exp(1.0))
    scale_softmax = scale_q * scale_k * sm_scale
    scale_softmax_log2 = scale_softmax * log2_e
    scale_output = scale_v / scale_o

    # Max seq lengths for problem_size (prefer pre-computed values for CUDA graph compat)
    if max_qo_len is None:
        max_s_q = int((qo_indptr.cpu()[1:] - qo_indptr.cpu()[:-1]).max().item())
    else:
        max_s_q = max_qo_len
    if max_kv_len is None:
        max_s_k = int((kv_indptr.cpu()[1:] - kv_indptr.cpu()[:-1]).max().item())
    else:
        max_s_k = max_kv_len

    # problem_size: (B, max_s_q, s_lse, max_s_k, H_q, H_k, D, D_v)
    s_lse = total_q  # for variable length, s_lse = total tokens
    problem_size = (batch_size, max_s_q, s_lse, max_s_k, H_q, H_k, D, D_v)

    # Window sizes
    ws_left = None if window_left == -1 else Int32(window_left)
    ws_right = None if window_right == -1 else Int32(window_right)
    if is_causal and ws_right is None:
        ws_right = Int32(0)

    if enable_tvm_ffi:
        # TVM-FFI: Pointer args accept data_ptr(), Tensor args accept torch.Tensor,
        # no explicit stream (env stream).
        # Kernel expects 4D pointers; unsqueeze to (1, total, H, D).
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        o_4d = o.unsqueeze(0)

        kernel_fn(
            q_4d.data_ptr(),
            k_4d.data_ptr(),
            v_4d.data_ptr(),
            o_4d.data_ptr(),
            problem_size,
            qo_indptr.to(torch.int32),  # cum_seqlen_q: Tensor arg
            kv_indptr.to(torch.int32),  # cum_seqlen_k: Tensor arg
            lse.data_ptr() if lse is not None else None,
            Float32(scale_softmax_log2),
            Float32(scale_softmax),
            Float32(scale_output),
            None,  # skip_softmax_threshold_log2
            ws_left,
            ws_right,
            None,  # skip_softmax_count
            None,  # total_softmax_count
            q_4d,  # q_tensor for env stream device detection
        )
    else:
        # CuTe native ABI: convert to cute tensors, pass iterators + explicit stream.

        # DSL FMHA kernel expects 4D tensor (B, S, H, D).
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        o_4d = o.unsqueeze(0)

        is_fp8_in = q.dtype == torch.float8_e4m3fn
        is_fp8_out = o.dtype == torch.float8_e4m3fn
        if is_fp8_in:
            q_cute = from_dlpack(
                q_4d.view(torch.int8), assumed_align=16
            ).mark_layout_dynamic(leading_dim=3)
            q_cute.element_type = cutlass.Float8E4M3FN
            k_cute = from_dlpack(
                k_4d.view(torch.int8), assumed_align=16
            ).mark_layout_dynamic(leading_dim=3)
            k_cute.element_type = cutlass.Float8E4M3FN
            v_cute = from_dlpack(
                v_4d.view(torch.int8), assumed_align=16
            ).mark_layout_dynamic(leading_dim=3)
            v_cute.element_type = cutlass.Float8E4M3FN
        else:
            q_cute = from_dlpack(q_4d, assumed_align=16).mark_layout_dynamic(
                leading_dim=3
            )
            k_cute = from_dlpack(k_4d, assumed_align=16).mark_layout_dynamic(
                leading_dim=3
            )
            v_cute = from_dlpack(v_4d, assumed_align=16).mark_layout_dynamic(
                leading_dim=3
            )
        if is_fp8_out:
            o_cute = from_dlpack(
                o_4d.view(torch.int8), assumed_align=16
            ).mark_layout_dynamic(leading_dim=3)
            o_cute.element_type = cutlass.Float8E4M3FN
        else:
            o_cute = from_dlpack(o_4d, assumed_align=16).mark_layout_dynamic(
                leading_dim=3
            )

        cum_seqlen_q_cute = from_dlpack(
            qo_indptr.to(torch.int32), assumed_align=16
        ).mark_layout_dynamic(leading_dim=0)
        cum_seqlen_k_cute = from_dlpack(
            kv_indptr.to(torch.int32), assumed_align=16
        ).mark_layout_dynamic(leading_dim=0)

        lse_iter = None
        if lse is not None:
            # TODO: lse's shape?
            lse_cute = from_dlpack(lse, assumed_align=16).mark_layout_dynamic(
                leading_dim=2
            )
            lse_iter = lse_cute.iterator

        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

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
            None,  # q_tensor (unused, for TVM-FFI env stream)
            stream,
        )
