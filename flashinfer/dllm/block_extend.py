"""
Blockwise Extend Attention with Tile-Level Skip Optimization

优化原理:
═══════════════════════════════════════════════════════════════════════════════
使用原生 MaskMode::kBlockExpanding 触发 kernel 内置的 tile 级跳过优化：

1. num_iterations 计算：根据 Block Expanding 边界精确计算需要迭代的 KV tiles
   kv_valid_end = ((q_tile_end - 1) / dllm_block_size + 1) * dllm_block_size
   完全不可见的 KV tiles 直接跳过，不加载、不计算

2. mask_iteration 计算：确定第一个需要 mask 检查的迭代
   kv_fully_visible_end = (q_tile_start / dllm_block_size + 1) * dllm_block_size
   在此之前的 tiles 完全可见，无需逐元素检查

3. 原生 mask 计算：在边界 tile 上使用 (q_block >= k_block) 规则

Block Extend Mask 规则:
───────────────────────────────────────────────────────────────────────────────
  mask[q, k] = (q / dllm_block_size) >= (k / dllm_block_size)
  同一 block 内双向可见，可以看见之前的 blocks，不能看见后续 blocks

使用方法:
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


# ════════════════════════════════════════════════════════════════════════════════════
# Variant 定义：使用原生 MaskMode::kBlockExpanding
# ════════════════════════════════════════════════════════════════════════════════════
#
# CUDA kernel 支持 MaskMode::kBlockExpanding 的 tile 级跳过：
#   - num_iterations: block_expanding_num_iterations() 精确计算 KV 有效范围
#   - mask_iteration: block_expanding_mask_iteration() 确定无需 mask 检查的 tiles
#   - position_mask: (q_block >= k_block) && (kv_idx < chunk_end)
#
# 因此 variant 的 LogitsMask 只需返回 true，让 kernel 的 position_mask 生效
#
# ════════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════════
# V2 with offset Variant 定义：支持 q_offset 的 Block Expanding Attention
# ════════════════════════════════════════════════════════════════════════════════════
#
# 用于增量 Chunk Prefill 场景，每个 chunk 的 Q 有全局偏移
# q_offset 通过 SinglePrefillParams.q_block_expanding_offset 传入
# kernel 内部通过 get_q_block_expanding_offset(batch_idx) 获取偏移
#
# ════════════════════════════════════════════════════════════════════════════════════

BLOCK_EXTEND_V2_WITH_OFFSET_VARIANT_DECL = r"""
// ════════════════════════════════════════════════════════════════════════════════════
// BlockExtendAttentionV2WithOffset: 支持 q_offset 的 Block Expanding Attention
// ════════════════════════════════════════════════════════════════════════════════════
//
// 用于增量 Chunk Prefill 场景：
//   - 每个 chunk 的 Q 有全局偏移 q_offset
//   - kernel 通过 params.get_q_block_expanding_offset() 获取偏移
//   - position_mask 内部计算：(q_global_block >= k_block)
//
// ════════════════════════════════════════════════════════════════════════════════════

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
    window_left = kv_len;  // 不使用 sliding window
  }

  // ════════════════════════════════════════════════════════════════════════════════
  // LogitsMask: 直接返回 true
  // ════════════════════════════════════════════════════════════════════════════════
  //
  // CUDA kernel 已原生支持 MaskMode::kBlockExpanding:
  //   - q_offset 通过 params.get_q_block_expanding_offset(batch_idx) 获取
  //   - position_mask 内部处理：(q_global_block >= k_block)
  //
  // 因此 LogitsMask 只需返回 true
  //
  // ════════════════════════════════════════════════════════════════════════════════
  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return true;  // kernel 的 position_mask 已处理 Block Expanding + q_offset 逻辑
  });

  // 不需要额外的 logits 变换
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  });
};
"""


# ════════════════════════════════════════════════════════════════════════════════════
# 统一的 AOT 工具函数
# ════════════════════════════════════════════════════════════════════════════════════

def _get_aot_path(uri: str) -> Path:
    """获取 AOT 预编译路径 (统一接口)"""
    return jit_env.FLASHINFER_AOT_DIR / uri / f"{uri}.so"


def _check_aot_available(uri: str) -> bool:
    """检查 AOT kernel 是否可用 (统一接口)"""
    import os
    if os.environ.get("FLASHINFER_FORCE_JIT", "0") == "1":
        return False
    return _get_aot_path(uri).exists()


def _get_dtype_str(dtype: torch.dtype) -> str:
    """获取 dtype 字符串表示 (统一接口)"""
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(dtype, "fp16")


# ════════════════════════════════════════════════════════════════════════════════════
# 模块 URI 生成函数
# ════════════════════════════════════════════════════════════════════════════════════

def _get_module_uri_with_offset(head_dim: int, dtype: torch.dtype, backend: str) -> str:
    """生成 with offset 模块的唯一标识符
    
    v2: 4 scalar params (sm_scale, dllm_block_size, q_block_expanding_offset,
                         kv_block_expanding_offset)
    旧版本 (无 _v2 后缀) 只有 3 scalar，需要重新编译时会自动匹配新 URI。
    """
    return f"block_expanding_{backend}_with_offset_v2_hdim{head_dim}_{_get_dtype_str(dtype)}"


# ════════════════════════════════════════════════════════════════════════════════════
# 模块缓存
# ════════════════════════════════════════════════════════════════════════════════════
_MODULE_CACHE_WITH_OFFSET = {}  # key = (head_dim, dtype, backend)


# ════════════════════════════════════════════════════════════════════════════════════
# V3 FA3 Variant 定义：Hopper (SM90) 架构的 Block Expanding Attention
# ════════════════════════════════════════════════════════════════════════════════════
#
# FA3 使用不同的 variant 接口：
#   - 构造函数接收 MainloopParams 和 BlockCoord
#   - 需要 GetAttentionUpdater() 模板函数
#   - 通过 params.additional_params.xxx 访问自定义参数
#
# FA3 kernel 已原生支持 kBlockExpanding，因此 LogitsTransform 只需返回 logits
#
# ════════════════════════════════════════════════════════════════════════════════════

BLOCK_EXTEND_V3_WITH_OFFSET_VARIANT_DECL = r"""
// ════════════════════════════════════════════════════════════════════════════════════
// BlockExtendAttentionV3WithOffset: FA3 (Hopper SM90) 版本的 Block Expanding Attention
// ════════════════════════════════════════════════════════════════════════════════════
//
// FA3 kernel 已原生支持 MaskMode::kBlockExpanding:
//   - get_num_kv_tiles(): 根据 Block Expanding 边界精确计算 KV 有效范围
//   - mma_f16(): BLOCK_EXPANDING 模板参数控制 n_masking_steps 和 col_limit
//   - position_mask: (q_global_block >= k_block) && (kv_idx < kv_len)
//
// 因此 LogitsTransform 只需返回 logits，让 kernel 的原生 mask 逻辑生效
//
// ════════════════════════════════════════════════════════════════════════════════════

struct BlockExtendAttentionV3WithOffset : AttentionVariantBase {
  float sm_scale_log2;

  // FA3 构造函数签名
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ BlockExtendAttentionV3WithOffset(
      const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
  }

  // FA3 需要 GetAttentionUpdater
  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(sm_scale_log2);
  }

  // ════════════════════════════════════════════════════════════════════════════════
  // LogitsTransform: 直接返回 logits
  // ════════════════════════════════════════════════════════════════════════════════
  //
  // FA3 kernel 已原生支持 MaskMode::kBlockExpanding:
  //   - BLOCK_EXPANDING 模板参数启用后，mma_f16 会计算 block_expanding_col_limit
  //   - n_masking_steps 类似 CAUSAL，只在边界 tile 上检查 mask
  //   - 完全不可见的 KV tiles 被 get_num_kv_tiles() 跳过
  //
  // 因此 LogitsTransform 只需返回 logits
  //
  // ════════════════════════════════════════════════════════════════════════════════
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;  // kernel 的原生 BLOCK_EXPANDING mask 已处理
  });
};
"""


def get_block_extend_module_with_offset(
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    backend: str = "fa2",
):
    """
    获取支持 q_offset/kv_offset 的 Block Extend Attention 模块
    
    Args:
        head_dim: Head 维度
        dtype: 数据类型
        backend: "fa2" 或 "fa3"
    
    Returns:
        编译好的模块
    
    Raises:
        RuntimeError: 如果 backend="fa3" 但 GPU 不支持 SM90
    """
    import os
    import tvm_ffi
    
    # FA3 需要 SM90 支持
    if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
        raise RuntimeError(
            "FA3 backend requires SM90 (Hopper) architecture. "
            "Use backend='fa2' for older architectures."
        )
    
    cache_key = (head_dim, dtype, backend)
    if cache_key in _MODULE_CACHE_WITH_OFFSET:
        return _MODULE_CACHE_WITH_OFFSET[cache_key]
    
    uri = _get_module_uri_with_offset(head_dim, dtype, backend)
    
    # AOT 模式
    if _check_aot_available(uri):
        aot_path = _get_aot_path(uri)
        module = tvm_ffi.load_module(str(aot_path))
        _MODULE_CACHE_WITH_OFFSET[cache_key] = module
        return module
    
    # AOT 不存在，检查是否禁用了 JIT
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        raise RuntimeError(
            f"JIT compilation is disabled via FLASHINFER_DISABLE_JIT environment variable, "
            f"but the required AOT module is not found at: {_get_aot_path(uri)}."
        )
    
    # JIT 模式
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
    
    支持增量 Chunk Prefill 和 Cascade Current Chunk 场景。
    
    Args:
        q: Query tensor [qo_len, num_heads, head_dim]
        k: Key tensor [kv_len, num_heads, head_dim]
        v: Value tensor [kv_len, num_heads, head_dim]
        dllm_block_size: DLLM block 大小 (必须是 2 的幂)
        q_offset: Q 的全局起始位置 (默认 0)
        kv_offset: KV 的全局起始位置 (默认 0)
        sm_scale: Softmax scale (默认 1/sqrt(head_dim))
        return_lse: 是否返回 log-sum-exp
        backend: "auto" (自动选择), "fa2" 或 "fa3"
    
    Returns:
        Output tensor [qo_len, num_heads, head_dim]
    
    Example:
        >>> # 增量 chunk prefill
        >>> o = block_expanding_attention_with_offset(
        ...     q, k_cumul, v_cumul,
        ...     dllm_block_size=32,
        ...     q_offset=i * chunk_len,
        ... )
        >>> 
        >>> # Cascade Current Chunk
        >>> o = block_expanding_attention_with_offset(
        ...     q, k_current, v_current,
        ...     dllm_block_size=256,
        ...     q_offset=prefix_len,
        ...     kv_offset=prefix_len,
        ... )
    """
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
        "q, k, v must be 3D tensors [seq_len, num_heads, head_dim]"
    assert (dllm_block_size & (dllm_block_size - 1)) == 0, \
        f"dllm_block_size must be power of 2, got {dllm_block_size}"
    
    head_dim = q.size(-1)
    dtype = q.dtype
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # backend 选择
    if backend == "auto":
        backend = "fa3" if is_sm90a_supported(q.device) else "fa2"
    
    module = get_block_extend_module_with_offset(head_dim=head_dim, dtype=dtype, backend=backend)
    
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


# ════════════════════════════════════════════════════════════════════════════════════
# FA2 Cascade 版本：Current Chunk (causal=True) + Prefix (causal=False) + merge_state
# ════════════════════════════════════════════════════════════════════════════════════
#
# 类似 SGLang 的 Cascade Attention 实现：
#   - 当 chunk_size = dllm_block_size 时，causal mask ≡ block extend mask
#   - 使用标准 FlashInfer API，无需 custom_mask
#   - 前缀全可见 (Q 的 block >= 所有 prefix 的 block)
#
# ════════════════════════════════════════════════════════════════════════════════════

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
    
    使用原生 Block Extend mask 实现 Cascade Attention。
    
    三阶段 Cascade:
    1. Current Chunk: Q attend to K_current/V_current (Block Expanding mask)
    2. Prefix: Q attend to K_prefix/V_prefix (causal=False, 全可见)
    3. Merge State: 合并两个阶段的 softmax 状态
    
    Args:
        q: Query tensor [qo_len, num_heads, head_dim] - 当前 chunk 的 Q
        k_current: Key tensor [curr_kv_len, num_kv_heads, head_dim] - 当前 chunk 的 K
        v_current: Value tensor [curr_kv_len, num_kv_heads, head_dim] - 当前 chunk 的 V
        k_prefix: Key tensor [prefix_len, num_kv_heads, head_dim] - 前缀的 K (可选)
        v_prefix: Value tensor [prefix_len, num_kv_heads, head_dim] - 前缀的 V (可选)
        dllm_block_size: DLLM block 大小 (必须是 2 的幂)
        sm_scale: Softmax scale (默认 1/sqrt(head_dim))
        return_lse: 是否返回 log-sum-exp
        backend: 后端选择 ("auto", "fa2", "fa3")
    
    Returns:
        如果 return_lse=False: Output tensor [qo_len, num_heads, head_dim]
        如果 return_lse=True: (output, lse) 元组
    
    Example:
        >>> # 增量 chunk prefill
        >>> # Step 0: Q=[0:64), K=[0:64), V=[0:64)
        >>> o0 = block_expanding_cascade(q0, k0, v0, dllm_block_size=64)
        >>> 
        >>> # Step 1: Q=[64:128), K_curr=[64:128), K_prefix=[0:64)
        >>> o1 = block_expanding_cascade(q1, k1, v1, k_prefix=k0, v_prefix=v0, dllm_block_size=64)
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
    
    # ════════════════════════════════════════════════════════════════════════════
    # 阶段 1: Current Chunk (Block Expanding mask)
    # ════════════════════════════════════════════════════════════════════════════
    #
    # 使用原生 Block Expanding mask:
    #   - q_offset = prefix_len (Q 的全局位置从 prefix_len 开始)
    #   - kv_offset = prefix_len (K_current 的全局位置也从 prefix_len 开始)
    #   - mask[q, k] = (q_global / B) >= (kv_global / B)
    #
    # ════════════════════════════════════════════════════════════════════════════
    
    # 没有 prefix 时直接返回，有 prefix 时需要 merge 所以强制 return_lse=True
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
        return_lse=True,  # merge 需要 lse
        backend=backend,
    )
    
    # ════════════════════════════════════════════════════════════════════════════
    # 阶段 2: Prefix (causal=False, 全可见)
    # ════════════════════════════════════════════════════════════════════════════
    #
    # Q 的全局位置 >= prefix 的末尾位置，所以：
    #   - Q_block >= 所有 prefix 的 K_block
    #   - Block Expanding mask 对 prefix 全部为 1
    #   - 使用 causal=False (全可见)
    #
    # ════════════════════════════════════════════════════════════════════════════
    o2, s2 = single_prefill_with_kv_cache(
        q, k_prefix, v_prefix,
        causal=False,  # prefix 全可见
        sm_scale=sm_scale,
        return_lse=True,
    )
    
    # ════════════════════════════════════════════════════════════════════════════
    # 阶段 3: Merge State (in-place 合并)
    # ════════════════════════════════════════════════════════════════════════════
    merge_state_in_place(o1, s1, o2, s2)
    
    if return_lse:
        return o1, s1
    else:
        return o1
