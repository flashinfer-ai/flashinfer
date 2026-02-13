"""CuTe DSL Flash Attention v2 backend for SM120 (Blackwell GeForce / DGX Spark).

This module provides an optional attention backend using NVIDIA's CuTe DSL
(CUTLASS Python DSL) which JIT-compiles flash attention kernels specifically
for SM120 GPUs that lack tcgen05 instructions.

Requirements:
    - ``nvidia-cutlass-dsl`` package (pip install nvidia-cutlass-dsl)
    - CUTLASS repository with the Blackwell GeForce flash_attention_v2 example

Usage::

    from flashinfer.cute_dsl_attention import cute_dsl_prefill_sm120

    out = cute_dsl_prefill_sm120(q, k, v, causal=True)

The CuTe DSL kernel uses SM80-compatible HMMA tensor core instructions
(mma.sync 16x8x16) which work on SM120, combined with TMA (cp.async.bulk)
for efficient memory movement. This achieves ~94% of peak performance on
RTX 5090 without requiring tcgen05.
"""

import functools
import importlib
import math
import os
from typing import Optional

import torch

from .api_logging import flashinfer_api
from .utils import is_sm120a_supported, is_sm121a_supported


def _find_cute_dsl_fa_module():
    """Find and import the CuTe DSL flash attention module.

    Searches for the FlashAttentionForwardSm120 class in these locations:
    1. CUTLASS_FA_SM120_PATH environment variable
    2. CUTLASS repo at standard paths
    """
    # Check env var first
    custom_path = os.environ.get("CUTLASS_FA_SM120_PATH")
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Check common CUTLASS repo locations
    candidates = [
        os.path.expanduser(
            "~/cutlass/examples/python/CuTeDSL/blackwell_geforce/flash_attention_v2.py"
        ),
        "/usr/local/cutlass/examples/python/CuTeDSL/blackwell_geforce/flash_attention_v2.py",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def _load_fa_module(module_path: str):
    """Load the flash attention module from a file path."""
    spec = importlib.util.spec_from_file_location("flash_attention_v2", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _get_compiled_kernel(head_dim, m_block, n_block, num_threads, is_causal, dtype_str):
    """Cache compiled CuTe DSL flash attention kernels."""
    module_path = _find_cute_dsl_fa_module()
    if module_path is None:
        raise ImportError(
            "CuTe DSL flash attention module not found. Set CUTLASS_FA_SM120_PATH "
            "to the path of flash_attention_v2.py from the CUTLASS repository, or "
            "clone CUTLASS to ~/cutlass/."
        )

    fa_module = _load_fa_module(module_path)

    fa2_fwd = fa_module.FlashAttentionForwardSm120(
        head_dim, m_block, n_block, num_threads, is_causal
    )

    return fa2_fwd, fa_module


@flashinfer_api
def cute_dsl_prefill_sm120(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    m_block_size: int = 64,
    n_block_size: int = 64,
    num_threads: int = 128,
) -> torch.Tensor:
    """Flash Attention v2 prefill using CuTe DSL kernels on SM120 GPUs.

    This function JIT-compiles a flash attention kernel using NVIDIA's CuTe DSL
    (CUTLASS Python DSL) that runs on SM120 GPUs using SM80-compatible HMMA
    tensor core instructions.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    key : torch.Tensor
        Key tensor with shape [batch_size, seqlen_k, num_heads, head_dim].
    value : torch.Tensor
        Value tensor with shape [batch_size, seqlen_k, num_heads, head_dim].
    causal : bool
        Whether to use causal masking.
    sm_scale : Optional[float]
        Softmax scale factor. If None, defaults to 1/sqrt(head_dim).
    m_block_size : int
        Query block tile size (default 64, can be 64 or 128).
    n_block_size : int
        Key/Value block tile size (default 64, can be 64 or 128).
    num_threads : int
        Number of threads per CTA (default 128).

    Returns
    -------
    out : torch.Tensor
        Output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    """
    if not (is_sm120a_supported(query.device) or is_sm121a_supported(query.device)):
        raise ValueError(
            "cute_dsl_prefill_sm120 is only supported on SM12x GPUs "
            "(RTX 5090, DGX Spark GB10)."
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Only BF16 and FP16 are supported, got {query.dtype}.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key, and value must have the same dtype.")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key, and value must be on the same device.")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value must have the same batch size.")
    if key.shape[1] != value.shape[1]:
        raise ValueError("key and value must have the same sequence length.")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key and value must have the same number of KV heads.")
    head_dim = query.shape[-1]
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim must be divisible by 8, got {head_dim}.")

    try:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except ImportError as e:
        raise ImportError(
            "nvidia-cutlass-dsl package is required for CuTe DSL attention. "
            "Install with: pip install nvidia-cutlass-dsl"
        ) from e

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    dtype_str = str(query.dtype).split(".")[-1]  # "float16" or "bfloat16"

    fa2_fwd, fa_module = _get_compiled_kernel(
        head_dim, m_block_size, n_block_size, num_threads, causal, dtype_str
    )

    out = torch.empty_like(query)

    dtype_bits = 16  # FP16 and BF16 are both 16-bit
    align_divisibility = 128 // dtype_bits  # 8 elements for 16-byte alignment

    def _get_stride_order(t):
        """Get stride order, with fallback for PyTorch < 2.7."""
        if hasattr(t, "dim_order"):
            return t.dim_order()
        return tuple(range(t.ndim))

    def to_cute_tensor(t):
        """Convert a torch tensor to CuTe tensor with proper alignment hints."""
        ct = from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        ct = ct.mark_compact_shape_dynamic(
            mode=3,
            stride_order=_get_stride_order(t),
            divisibility=align_divisibility,
        )
        return ct

    q_cute = to_cute_tensor(query)
    k_cute = to_cute_tensor(key)
    v_cute = to_cute_tensor(value)
    o_cute = to_cute_tensor(out)

    # Get CUDA stream
    try:
        from cuda import cuda
    except ImportError as e:
        raise ImportError(
            "cuda-python package is required for CuTe DSL attention. "
            "Install with: pip install cuda-python"
        ) from e

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # cute.compile has no internal memoization, so we cache the compiled
    # executor keyed by (shape, dtype, causal, sm_scale, block sizes).
    cache_key = (
        query.shape,
        key.shape,
        dtype_str,
        causal,
        sm_scale,
        m_block_size,
        n_block_size,
    )
    compiled = _compile_cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            fa2_fwd, q_cute, k_cute, v_cute, o_cute, sm_scale, stream
        )
        _compile_cache[cache_key] = compiled

    compiled(q_cute, k_cute, v_cute, o_cute, sm_scale, stream)

    return out


# Module-level cache for cute.compile results
_compile_cache: dict = {}
