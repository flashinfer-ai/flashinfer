"""
Blockwise Extend Attention with Tile-Level Skip Optimization

Optimization Principle:

Use native MaskMode::kBlockExpanding to trigger kernel's built-in tile-level skip optimization:

1. num_iterations calculation: Precisely calculate KV tiles to iterate based on Block Expanding boundaries
   kv_valid_end = ((q_tile_end - 1) / dllm_block_size + 1) * dllm_block_size
   Completely invisible KV tiles are skipped directly, not loaded or computed

2. mask_iteration calculation: Determine the first iteration that needs mask checking
   kv_fully_visible_end = (q_tile_start / dllm_block_size + 1) * dllm_block_size
   Tiles before this are fully visible, no per-element check needed

3. Native mask calculation: Use (q_block >= k_block) rule on boundary tiles

Block Extend Mask Rules:
  mask[q, k] = (q / dllm_block_size) >= (k / dllm_block_size)
  Bidirectional visibility within the same block, can see previous blocks, cannot see subsequent blocks

Usage:
    from flashinfer.dllm import block_extend_attention_with_offset
    o = block_extend_attention_with_offset(q, k, v, dllm_block_size=32)
"""

import math
import torch
from pathlib import Path
from typing import Optional, Union, Tuple

from ..jit import env as jit_env
from ..jit.attention import gen_customize_single_prefill_module
from ..prefill import single_prefill_with_kv_cache_with_jit_module
from ..utils import MaskMode, is_sm90a_supported
from ..api_logging import flashinfer_api

BLOCK_EXTEND_V2_WITH_OFFSET_VARIANT_DECL = r"""
// For incremental Chunk Prefill scenarios:
//   - Each chunk's Q has global offset q_offset
//   - Kernel retrieves offset via params.get_q_block_expanding_offset()
//   - position_mask internally calculates: (q_global_block >= k_block)

struct BlockExtendAttentionV2WithOffset : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t window_left;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ BlockExtendAttentionV2WithOffset(const Params& params, uint32_t batch_idx,
                                                           uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    sm_scale_log2 = params.sm_scale * math::log2e;
    window_left = kv_len;  // No sliding window
  }

  // CUDA kernel natively supports MaskMode::kBlockExpanding:
  //   - q_offset retrieved via params.get_q_block_expanding_offset(batch_idx)
  //   - position_mask internally handles: (q_global_block >= k_block)
  //
  // Therefore LogitsMask only needs to return true

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return true;  // kernel's position_mask already handles Block Expanding + q_offset logic
  });

  // No additional logits transformation needed
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  });
};
"""

def _get_aot_path(uri: str) -> Path:
    """Get AOT precompiled path (unified interface)"""
    return jit_env.FLASHINFER_AOT_DIR / uri / f"{uri}.so"


def _check_aot_available(uri: str) -> bool:
    """Check if AOT kernel is available (unified interface)"""
    import os
    if os.environ.get("FLASHINFER_FORCE_JIT", "0") == "1":
        return False
    return _get_aot_path(uri).exists()


def _get_dtype_str(dtype: torch.dtype) -> str:
    """Get dtype string representation (unified interface)"""
    _dtype_map = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    if dtype not in _dtype_map:
        raise ValueError(
            f"Unsupported dtype {dtype} for Block Extend Attention. "
            f"Supported: {list(_dtype_map.keys())}"
        )
    return _dtype_map[dtype]


def _get_module_uri_with_offset(head_dim: int, dtype: torch.dtype, backend: str) -> str:
    """Generate unique identifier for with offset module
    
    v2: 4 scalar params (sm_scale, dllm_block_size, q_block_expanding_offset,
                         kv_block_expanding_offset)
    Old version (without _v2 suffix) only has 3 scalars, will automatically match new URI when recompilation is needed.
    """
    return f"block_expanding_{backend}_with_offset_v2_hdim{head_dim}_{_get_dtype_str(dtype)}"


_MODULE_CACHE_WITH_OFFSET = {}

# V3 FA3 Variant Definition: Block Expanding Attention for Hopper (SM90) architecture

# FA3 uses different variant interface:
#   - Constructor receives MainloopParams and BlockCoord
#   - Requires GetAttentionUpdater() template function
#   - Access custom parameters via params.additional_params.xxx
#
# FA3 kernel natively supports kBlockExpanding, so LogitsTransform only needs to return logits

BLOCK_EXTEND_V3_WITH_OFFSET_VARIANT_DECL = r"""
// FA3 kernel natively supports MaskMode::kBlockExpanding:
//   - get_num_kv_tiles(): Precisely calculates KV valid range based on Block Expanding boundaries
//   - mma_f16(): BLOCK_EXPANDING template parameter controls n_masking_steps and col_limit
//   - position_mask: (q_global_block >= k_block) && (kv_idx < kv_len)
//
// Therefore LogitsTransform only needs to return logits, letting kernel's native mask logic take effect

struct BlockExtendAttentionV3WithOffset : AttentionVariantBase {
  float sm_scale_log2;
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ BlockExtendAttentionV3WithOffset(
      const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
  }
  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(sm_scale_log2);
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;  // kernel's native BLOCK_EXPANDING mask already handles this
  });
};
"""


def get_block_extend_module_with_offset(
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    backend: str = "fa2",
    device: Optional[torch.device] = None,
):
    """
    Get Block Extend Attention module with q_offset/kv_offset support
    
    Args:
        head_dim: Head dimension
        dtype: Data type
        backend: "fa2" or "fa3"
        device: Target CUDA device (default: current CUDA device)
    
    Returns:
        Compiled module
    
    Raises:
        RuntimeError: If backend="fa3" but GPU doesn't support SM90
    """
    import os
    import tvm_ffi
    
    if device is None:
        device = torch.device("cuda")
    
    # FA3 requires SM90 support
    if backend == "fa3" and not is_sm90a_supported(device):
        raise RuntimeError(
            "FA3 backend requires SM90 (Hopper) architecture. "
            "Use backend='fa2' for older architectures."
        )
    
    cache_key = (head_dim, dtype, backend, device)
    if cache_key in _MODULE_CACHE_WITH_OFFSET:
        return _MODULE_CACHE_WITH_OFFSET[cache_key]
    
    uri = _get_module_uri_with_offset(head_dim, dtype, backend)
    
    # AOT mode
    if _check_aot_available(uri):
        aot_path = _get_aot_path(uri)
        module = tvm_ffi.load_module(str(aot_path))
        _MODULE_CACHE_WITH_OFFSET[cache_key] = module
        return module
    
    # AOT not available, check if JIT is disabled
    if os.environ.get("FLASHINFER_DISABLE_JIT", "0") == "1":
        raise RuntimeError(
            f"JIT compilation is disabled via FLASHINFER_DISABLE_JIT environment variable, "
            f"but the required AOT module is not found at: {_get_aot_path(uri)}."
        )
    
    # JIT mode
    if backend == "fa3":
        variant_name = "BlockExtendAttentionV3WithOffset"
        variant_decl = BLOCK_EXTEND_V3_WITH_OFFSET_VARIANT_DECL
    else:
        variant_name = "BlockExtendAttentionV2WithOffset"
        variant_decl = BLOCK_EXTEND_V2_WITH_OFFSET_VARIANT_DECL
    
    spec = gen_customize_single_prefill_module(
        backend=backend,
        uri=uri,
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_o=dtype,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=["sm_scale", "dllm_block_size", "q_block_expanding_offset", "kv_block_expanding_offset"],
        additional_scalar_dtypes=["double", "int64_t", "int64_t", "int64_t"],
        variant_name=variant_name,
        variant_decl=variant_decl,
        mask_modes=[4],  # kBlockExpanding = 4
    )
    module = spec.build_and_load()
    
    _MODULE_CACHE_WITH_OFFSET[cache_key] = module
    return module

@flashinfer_api
def block_extend_attention_with_offset(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dllm_block_size: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    backend: str = "auto",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Block Extend Attention with Q and KV Offset Support
    
    Supports incremental Chunk Prefill and Cascade Current Chunk scenarios.
    
    Args:
        q: Query tensor [qo_len, num_heads, head_dim]
        k: Key tensor [kv_len, num_heads, head_dim]
        v: Value tensor [kv_len, num_heads, head_dim]
        dllm_block_size: DLLM block size (must be power of 2)
        q_offset: Q's global starting position (default 0)
        kv_offset: KV's global starting position (default 0)
        sm_scale: Softmax scale (default 1/sqrt(head_dim))
        return_lse: Whether to return log-sum-exp
        backend: "auto" (auto-select), "fa2" or "fa3"
    
    Returns:
        Output tensor [qo_len, num_heads, head_dim]
    
    Example:
        >>> # Incremental chunk prefill
        >>> o = block_extend_attention_with_offset(
        ...     q, k_cumul, v_cumul,
        ...     dllm_block_size=32,
        ...     q_offset=i * chunk_len,
        ... )
        >>> 
        >>> # Cascade Current Chunk
        >>> o = block_extend_attention_with_offset(
        ...     q, k_current, v_current,
        ...     dllm_block_size=256,
        ...     q_offset=prefix_len,
        ...     kv_offset=prefix_len,
        ... )
    """
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
        "q, k, v must be 3D tensors [seq_len, num_heads, head_dim]"
    assert dllm_block_size > 0 and (dllm_block_size & (dllm_block_size - 1)) == 0, \
        f"dllm_block_size must be a positive power of 2, got {dllm_block_size}"
    
    head_dim = q.size(-1)
    dtype = q.dtype
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # backend selection
    if backend == "auto":
        backend = "fa3" if is_sm90a_supported(q.device) else "fa2"
    
    module = get_block_extend_module_with_offset(head_dim=head_dim, dtype=dtype, backend=backend, device=q.device)
    
    return single_prefill_with_kv_cache_with_jit_module(
        module,
        q, k, v,
        sm_scale,
        dllm_block_size,
        q_offset,
        kv_offset,
        mask_mode=MaskMode.BLOCK_EXPANDING.value,
        return_lse=return_lse,
    )

# FA2 Cascade version: Current Chunk (causal=True) + Prefix (causal=False) + merge_state
# Similar to SGLang's Cascade Attention implementation:
#   - When chunk_size = dllm_block_size, causal mask ≡ block extend mask
#   - Uses standard FlashInfer API, no custom_mask needed
#   - Prefix fully visible (Q's block >= all prefix's blocks)
@flashinfer_api
def block_extend_cascade(
    q: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    k_prefix: Optional[torch.Tensor] = None,
    v_prefix: Optional[torch.Tensor] = None,
    dllm_block_size: int = 64,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    backend: str = "auto",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Cascade Block Extend Attention (Single Request)
    
    Implements Cascade Attention using native Block Extend mask.
    
    Three-stage Cascade:
    1. Current Chunk: Q attend to K_current/V_current (Block Expanding mask)
    2. Prefix: Q attend to K_prefix/V_prefix (causal=False, fully visible)
    3. Merge State: Merge softmax states from both stages
    
    Args:
        q: Query tensor [qo_len, num_heads, head_dim] - Q of current chunk
        k_current: Key tensor [curr_kv_len, num_kv_heads, head_dim] - K of current chunk
        v_current: Value tensor [curr_kv_len, num_kv_heads, head_dim] - V of current chunk
        k_prefix: Key tensor [prefix_len, num_kv_heads, head_dim] - Prefix K (optional)
        v_prefix: Value tensor [prefix_len, num_kv_heads, head_dim] - Prefix V (optional)
        dllm_block_size: DLLM block size (must be power of 2)
        sm_scale: Softmax scale (default 1/sqrt(head_dim))
        return_lse: Whether to return log-sum-exp
        backend: Backend selection ("auto", "fa2", "fa3")
    
    Returns:
        If return_lse=False: Output tensor [qo_len, num_heads, head_dim]
        If return_lse=True: (output, lse) tuple
    
    Example:
        >>> # Incremental chunk prefill
        >>> # Step 0: Q=[0:64), K=[0:64), V=[0:64)
        >>> o0 = block_extend_cascade(q0, k0, v0, dllm_block_size=64)
        >>> 
        >>> # Step 1: Q=[64:128), K_curr=[64:128), K_prefix=[0:64)
        >>> o1 = block_extend_cascade(q1, k1, v1, k_prefix=k0, v_prefix=v0, dllm_block_size=64)
    """
    from ..cascade import merge_state_in_place
    from ..prefill import single_prefill_with_kv_cache
    
    assert q.dim() == 3 and k_current.dim() == 3 and v_current.dim() == 3, \
        "q, k_current, v_current must be 3D tensors [seq_len, num_heads, head_dim]"
    
    head_dim = q.size(-1)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    has_prefix = k_prefix is not None and v_prefix is not None
    prefix_len = k_prefix.size(0) if has_prefix else 0

    # Stage 1: Current Chunk (Block Expanding mask)
    #
    # Use native Block Expanding mask with offset support:
    #   - q_offset = prefix_len: Q's global position starts from prefix_len
    #   - kv_offset = prefix_len: K_current's global position also starts from prefix_len
    #   - mask[q, k] = ((q_offset + q_idx) / B) >= ((kv_offset + kv_idx) / B)
    #
    # Return directly when no prefix, force return_lse=True when prefix exists for merge
    if not has_prefix:
        return block_extend_attention_with_offset(
            q, k_current, v_current,
            dllm_block_size=dllm_block_size,
            q_offset=prefix_len,  # = 0
            kv_offset=prefix_len,  # = 0
            sm_scale=sm_scale,
            return_lse=return_lse,
            backend=backend,
        )
    
    o1, s1 = block_extend_attention_with_offset(
        q, k_current, v_current,
        dllm_block_size=dllm_block_size,
        q_offset=prefix_len,
        kv_offset=prefix_len,
        sm_scale=sm_scale,
        return_lse=True,  # merge requires lse
        backend=backend,
    )

    # Stage 2: Prefix (causal=False, fully visible)
    #
    # Q's global position >= prefix's end position, so:
    #   - Q_block >= all prefix's K_block (since q_offset = prefix_len)
    #   - Block Expanding mask is all 1 for prefix, equivalent to causal=False
    #
    o2, s2 = single_prefill_with_kv_cache(
        q, k_prefix, v_prefix,
        causal=False,  # prefix fully visible
        sm_scale=sm_scale,
        return_lse=True,
    )

    # Stage 3: Merge State (in-place merge)
    merge_state_in_place(o1, s1, o2, s2)
    
    if return_lse:
        return o1, s1
    else:
        return o1
