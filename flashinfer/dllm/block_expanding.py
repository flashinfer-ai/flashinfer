"""
Block Expanding Attention for DLLM - AOT/JIT 双模式支持

使用方法:
    from flashinfer.dllm import block_expanding_attention
    
    # 自动检测: 如果 AOT kernel 存在则直接加载，否则 JIT 编译
    o = block_expanding_attention(q, k, v, dllm_block_size=32)
    
环境变量:
    FLASHINFER_FORCE_JIT=1        强制使用 JIT 编译
    FLASHINFER_DISABLE_JIT=1      禁用 JIT，仅使用 AOT
"""

import functools
import math
import torch
from pathlib import Path
from typing import Optional, Union, Tuple

from ..jit import env as jit_env
from ..jit.attention import gen_customize_single_prefill_module
from ..prefill import single_prefill_with_kv_cache_with_jit_module
from ..utils import MaskMode



BLOCK_EXPANDING_VARIANT_DECL = r"""
struct BlockExpandingAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t log2_block_size;
  uint32_t window_left;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ BlockExpandingAttention(const Params& params, uint32_t batch_idx,
                                               uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    uint32_t block_size = static_cast<uint32_t>(params.dllm_block_size);
#ifdef __CUDA_ARCH__
    log2_block_size = __popc(block_size - 1);
#else
    log2_block_size = 0;
    while ((1u << log2_block_size) < block_size) ++log2_block_size;
#endif
    sm_scale_log2 = params.sm_scale * math::log2e;
    window_left = kv_len;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (qo_idx >= qo_len || kv_idx >= kv_len) {
      return false;
    }
    const uint32_t q_block_id = qo_idx >> log2_block_size;
    const uint32_t k_block_id = kv_idx >> log2_block_size;
    return q_block_id >= k_block_id;
  });

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  });
};
"""


_MODULE_CACHE = {}


def _get_module_uri(head_dim: int, dtype: torch.dtype) -> str:
    """生成模块的唯一标识符"""
    dtype_str = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(dtype, "fp16")
    return f"single_prefill_block_expanding_hd{head_dim}_{dtype_str}"


def _get_aot_path(uri: str) -> Path:
    """获取 AOT 预编译路径"""
    return jit_env.FLASHINFER_AOT_DIR / uri / f"{uri}.so"


def _check_aot_available(uri: str) -> bool:
    """检查 AOT kernel 是否可用"""
    import os
    if os.environ.get("FLASHINFER_FORCE_JIT", "0") == "1":
        return False
    return _get_aot_path(uri).exists()


def get_block_expanding_module(
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
):
    """
    获取 Block Expanding Attention 模块
    
    自动检测:
    - 如果 AOT kernel 存在 → 直接加载（跳过代码生成）
    - 否则 → JIT 编译
    
    Args:
        head_dim: Head 维度
        dtype: 数据类型
    
    Returns:
        编译好的模块
    """
    import tvm_ffi
    
    cache_key = (head_dim, dtype)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    
    uri = _get_module_uri(head_dim, dtype)
    
    # 优先检查 AOT kernel，避免不必要的代码生成
    if _check_aot_available(uri):
        aot_path = _get_aot_path(uri)
        module = tvm_ffi.load_module(str(aot_path))
        _MODULE_CACHE[cache_key] = module
        return module
    
    # Fallback: JIT 编译（会触发代码生成）
    spec = gen_customize_single_prefill_module(
        backend="fa2",
        uri=uri,
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_o=dtype,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=["sm_scale", "dllm_block_size"],
        additional_scalar_dtypes=["double", "int64_t"],
        variant_name="BlockExpandingAttention",
        variant_decl=BLOCK_EXPANDING_VARIANT_DECL,
    )
    module = spec.build_and_load()
    
    _MODULE_CACHE[cache_key] = module
    return module


def block_expanding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dllm_block_size: int,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Block Expanding Attention for DLLM (Draft-based LLM)
    
    术语说明:
    - dllm_block_size: DLLM 中每个 block 的 token 数量（双向可见组）
                       注意: 这与 PageAttention 的 page_size 是完全不同的概念！
    
    语义:
    - 同一个 DLLM block 内的 token 双向可见
    - 可以看见之前所有 blocks
    - 不能看见后续 blocks
    
    Args:
        q: Query tensor [qo_len, num_heads, head_dim]
        k: Key tensor [kv_len, num_heads, head_dim]
        v: Value tensor [kv_len, num_heads, head_dim]
        dllm_block_size: DLLM block 大小（必须是 2 的幂）
        sm_scale: Softmax scale, 默认 1/sqrt(head_dim)
        return_lse: 是否返回 log-sum-exp
    
    Returns:
        如果 return_lse=False: Output tensor [qo_len, num_heads, head_dim]
        如果 return_lse=True: (output, lse) 元组
    
    Example:
        >>> o = block_expanding_attention(q, k, v, dllm_block_size=32)
    """
    # 参数验证
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
        "q, k, v must be 3D tensors [seq_len, num_heads, head_dim]"
    assert (dllm_block_size & (dllm_block_size - 1)) == 0, \
        f"dllm_block_size must be power of 2, got {dllm_block_size}"
    
    head_dim = q.size(-1)
    dtype = q.dtype
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # 获取模块
    module = get_block_expanding_module(head_dim=head_dim, dtype=dtype)
    
    # 调用 kernel
    # mask_mode = CUSTOM (2) 确保每次迭代都调用 LogitsMask
    return single_prefill_with_kv_cache_with_jit_module(
        module,
        q, k, v,
        sm_scale,
        dllm_block_size,
        mask_mode=MaskMode.CUSTOM.value,
        return_lse=return_lse,
    )


def list_available_aot_kernels():
    """列出所有可用的 AOT 预编译 kernel"""
    aot_dir = jit_env.FLASHINFER_AOT_DIR
    if not aot_dir.exists():
        print(f"AOT directory not found: {aot_dir}")
        return []
    
    kernels = []
    for subdir in aot_dir.iterdir():
        if subdir.is_dir() and "block_expanding" in subdir.name:
            so_file = subdir / f"{subdir.name}.so"
            if so_file.exists():
                kernels.append(subdir.name)
    
    return kernels


def print_kernel_status():
    """打印 kernel 状态（AOT vs JIT）"""
    print("=" * 60)
    print("Block Expanding Attention Kernel Status")
    print("=" * 60)
    print(f"AOT directory: {jit_env.FLASHINFER_AOT_DIR}")
    print(f"JIT directory: {jit_env.FLASHINFER_JIT_DIR}")
    print()
    
    # 常见配置
    configs = [
        (64, torch.float16),
        (128, torch.float16),
        (256, torch.float16),
        (128, torch.bfloat16),
    ]
    
    for head_dim, dtype in configs:
        uri = _get_module_uri(head_dim, dtype)
        aot_available = _check_aot_available(uri)
        status = "AOT ✓" if aot_available else "JIT (will compile on first use)"
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        print(f"  head_dim={head_dim}, dtype={dtype_str}: {status}")
    
    print("=" * 60)
