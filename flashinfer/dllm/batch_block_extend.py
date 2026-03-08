"""
Batch Block Extend Attention for DLLM (Diffusion LLM)

Block Extend Mask Rules:
  q_global = q_offset + q_idx
  kv_global = kv_offset + kv_idx
  mask[q, k] = (q_global / dllm_block_size) >= (kv_global / dllm_block_size)
  Bidirectional visibility within the same block, can see previous blocks, cannot see subsequent blocks

Usage:
    from flashinfer.dllm import BatchBlockExtendRaggedOffsetWrapper
        wrapper = BatchBlockExtendRaggedOffsetWrapper(workspace, kv_layout="NHD", dllm_block_size=32)
    wrapper.plan(qo_indptr, kv_indptr, num_heads, num_kv_heads, head_dim)
    output = wrapper.run(q, k, v)
"""

from __future__ import annotations

import math
import os
import torch
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

from ..prefill import (
    BatchPrefillWithRaggedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from ..jit import gen_customize_batch_prefill_module
from ..jit import env as jit_env
from ..utils import MaskMode


def check_jit_environment() -> dict:
    """Check if JIT compilation environment is working properly"""
    results = {
        "tvm_ffi_ok": False,
        "device_guard_ok": False,
        "nvcc_ok": False,
        "issues": [],
    }
    
    try:
        import tvm_ffi
        results["tvm_ffi_ok"] = True
        include_path = tvm_ffi.libinfo.find_include_path()
        device_guard_path = Path(include_path) / "tvm" / "ffi" / "extra" / "cuda" / "device_guard.h"
        results["device_guard_ok"] = device_guard_path.exists()
        if not results["device_guard_ok"]:
            results["issues"].append(f"Missing TVM header: {device_guard_path}")
    except ImportError:
        results["issues"].append("tvm_ffi package not installed")
    except Exception as e:
        results["issues"].append(f"Error checking tvm_ffi: {e}")
    
    import subprocess
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        results["nvcc_ok"] = result.returncode == 0
    except FileNotFoundError:
        results["nvcc_ok"] = False
        results["issues"].append("nvcc not found in PATH")
    
    return results


def check_kernel_availability(uri: str) -> tuple:
    """Check availability of specified kernel"""
    aot_path = jit_env.FLASHINFER_AOT_DIR / uri / f"{uri}.so"
    aot_available = aot_path.exists()
    
    jit_env_check = check_jit_environment()
    jit_available = (
        jit_env_check["tvm_ffi_ok"] and 
        jit_env_check["device_guard_ok"] and 
        jit_env_check["nvcc_ok"]
    )
    
    return aot_available, jit_available, aot_path


def select_best_backend(head_dim: int, dtype: torch.dtype, preferred_backend: str = "auto", device: torch.device = None) -> str:
    """Select backend based on kernel availability and compute capability"""
    from ..utils import is_sm90a_supported
    
    base_uri = _get_batch_be_module_uri(head_dim, dtype)
    fa2_uri = base_uri + "_ragged_offset"
    fa3_uri = base_uri + "_ragged_offset_fa3"
    
    fa2_aot, fa2_jit, _ = check_kernel_availability(fa2_uri)
    fa3_aot, fa3_jit, _ = check_kernel_availability(fa3_uri)
    
    fa2_available = fa2_aot or fa2_jit
    fa3_available = fa3_aot or fa3_jit
    
    if preferred_backend == "auto":
        if device is None:
            device = torch.device("cuda")
        is_hopper = is_sm90a_supported(device)
        
        if is_hopper:
            if fa3_available:
                return "fa3"
            if fa2_available:
                return "fa2"
        else:
            if fa2_available:
                return "fa2"
            if fa3_available:
                return "fa3"
        
        raise RuntimeError(
            f"No Block Extend kernel available for head_dim={head_dim}, dtype={dtype}. "
            f"FA2: AOT={fa2_aot}, JIT={fa2_jit}; FA3: AOT={fa3_aot}, JIT={fa3_jit}"
        )
    
    if preferred_backend == "fa2":
        if fa2_available:
            return "fa2"
        raise RuntimeError(f"FA2 kernel '{fa2_uri}' not available")
    
    if preferred_backend == "fa3":
        if fa3_available:
            return "fa3"
        raise RuntimeError(f"FA3 kernel '{fa3_uri}' not available")
    
    raise ValueError(f"Unknown backend: {preferred_backend}")


def select_best_backend_paged(head_dim: int, dtype: torch.dtype, preferred_backend: str = "auto", device: torch.device = None) -> str:
    """Select backend based on Paged kernel availability and compute capability"""
    from ..utils import is_sm90a_supported
    
    base_uri = _get_batch_be_module_uri(head_dim, dtype)
    fa2_uri = base_uri + "_paged_offset"
    fa3_uri = base_uri + "_paged_offset_fa3"
    
    fa2_aot, fa2_jit, _ = check_kernel_availability(fa2_uri)
    fa3_aot, fa3_jit, _ = check_kernel_availability(fa3_uri)
    
    fa2_available = fa2_aot or fa2_jit
    fa3_available = fa3_aot or fa3_jit
    
    if preferred_backend == "auto":
        if device is None:
            device = torch.device("cuda")
        is_hopper = is_sm90a_supported(device)
        
        if is_hopper:
            if fa3_available:
                return "fa3"
            if fa2_available:
                return "fa2"
        else:
            if fa2_available:
                return "fa2"
            if fa3_available:
                return "fa3"
        
        raise RuntimeError(
            f"No Paged Block Extend kernel available for head_dim={head_dim}, dtype={dtype}"
        )
    
    if preferred_backend == "fa2":
        if fa2_available:
            return "fa2"
        raise RuntimeError(f"FA2 paged kernel '{fa2_uri}' not available")
    
    if preferred_backend == "fa3":
        if fa3_available:
            return "fa3"
        raise RuntimeError(f"FA3 paged kernel '{fa3_uri}' not available")
    
    raise ValueError(f"Unknown backend: {preferred_backend}")


_BATCH_BE_MODULE_CACHE = {}

def _get_batch_be_module_uri(head_dim: int, dtype: torch.dtype) -> str:
    dtype_str = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp16")
    return f"batch_prefill_block_expanding_hd{head_dim}_{dtype_str}"


def _get_batch_be_aot_path(uri: str) -> Path:
    return jit_env.FLASHINFER_AOT_DIR / uri / f"{uri}.so"


def _check_batch_be_aot_available(uri: str) -> bool:
    if os.environ.get("FLASHINFER_FORCE_JIT", "0") == "1":
        return False
    return _get_batch_be_aot_path(uri).exists()


# FA2 Offset Variant
_BATCH_BE_OFFSET_VARIANT_DECL = r"""
struct BatchBlockExtendOffsetAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t window_left;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ BatchBlockExtendOffsetAttention(const Params& params, uint32_t batch_idx,
                                                           uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    sm_scale_log2 = params.sm_scale * math::log2e;
    window_left = kv_len;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return true;
  });

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  });
};
"""

# FA3 Offset Variant
_BATCH_BE_OFFSET_VARIANT_DECL_FA3 = r"""
struct BatchBlockExtendOffsetAttentionFA3 : AttentionVariantBase {
  float sm_scale_log2;

  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ BatchBlockExtendOffsetAttentionFA3(
      const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(sm_scale_log2);
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
  });
};
"""


class BatchBlockExtendPagedOffsetWrapper:
    """Batch Block Extend Paged Attention with Offset Support"""
    
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        dllm_block_size: int = 256,
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        q_offsets_buf: Optional[torch.Tensor] = None,
        kv_offsets_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        assert (dllm_block_size & (dllm_block_size - 1)) == 0, \
            f"dllm_block_size must be power of 2, got {dllm_block_size}"
        
        self._dllm_block_size = dllm_block_size
        self._kv_layout = kv_layout
        self._backend = backend
        self._device = float_workspace_buffer.device
        self._dtype: Optional[torch.dtype] = None
        self._head_dim: Optional[int] = None
        
        self._float_workspace_buffer = float_workspace_buffer
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._q_offsets_buf = q_offsets_buf
        self._kv_offsets_buf = kv_offsets_buf
        
        self._inner_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
        self._q_offsets: Optional[torch.Tensor] = None
        self._kv_offsets: Optional[torch.Tensor] = None
    
    def _create_inner_wrapper(self, dtype: torch.dtype, head_dim: int, idtype: torch.dtype = torch.int32) -> None:
        effective_backend = select_best_backend_paged(head_dim, dtype, self._backend, self._device)
        if effective_backend != self._backend:
            self._backend = effective_backend
        
        if self._backend == "fa3":
            uri = _get_batch_be_module_uri(head_dim, dtype) + "_paged_offset_fa3"
            variant_name = "BatchBlockExtendOffsetAttentionFA3"
            variant_decl = _BATCH_BE_OFFSET_VARIANT_DECL_FA3
        else:
            uri = _get_batch_be_module_uri(head_dim, dtype) + "_paged_offset"
            variant_name = "BatchBlockExtendOffsetAttention"
            variant_decl = _BATCH_BE_OFFSET_VARIANT_DECL
        
        jit_args = [
            uri, dtype, dtype, dtype, idtype, head_dim, head_dim,
            ["maybe_q_block_expanding_offset", "maybe_kv_block_expanding_offset"],
            [dtype_map_for_idtype(idtype), dtype_map_for_idtype(idtype)],
            ["sm_scale", "dllm_block_size"], ["double", "int64_t"],
            variant_name, variant_decl,
        ]
        jit_kwargs = {
            "pos_encoding_mode": 0, "use_sliding_window": False,
            "use_logits_soft_cap": False, "use_fp16_qk_reduction": False,
            "mask_modes": [0, 1, 2, 3, 4],
        }
        
        self._inner_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._float_workspace_buffer, kv_layout=self._kv_layout,
            use_cuda_graph=self._use_cuda_graph, qo_indptr_buf=self._qo_indptr_buf,
            paged_kv_indptr_buf=self._paged_kv_indptr_buf,
            paged_kv_indices_buf=self._paged_kv_indices_buf,
            paged_kv_last_page_len_buf=self._paged_kv_last_page_len_buf,
            backend=self._backend, jit_args=jit_args, jit_kwargs=jit_kwargs,
        )
        self._dtype = dtype
        self._head_dim = head_dim
    
    def plan(
        self, qo_indptr: torch.Tensor, paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor, paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int, num_kv_heads: int, head_dim: int, page_size: int,
        q_data_type: torch.dtype = torch.float16, sm_scale: Optional[float] = None,
        q_offsets: Optional[torch.Tensor] = None, kv_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        if self._inner_wrapper is None or self._head_dim != head_dim or self._dtype != q_data_type:
            self._create_inner_wrapper(q_data_type, head_dim, qo_indptr.dtype)
        
        self._sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
        
        if self._use_cuda_graph:
            if q_offsets is not None:
                if self._q_offsets_buf is None:
                    raise ValueError("q_offsets_buf must be provided in CUDA Graph mode")
                self._q_offsets_buf[:len(q_offsets)].copy_(q_offsets, non_blocking=True)
                self._q_offsets = self._q_offsets_buf[:len(q_offsets)]
            else:
                self._q_offsets = None
            
            if kv_offsets is not None:
                if self._kv_offsets_buf is None:
                    raise ValueError("kv_offsets_buf must be provided in CUDA Graph mode")
                self._kv_offsets_buf[:len(kv_offsets)].copy_(kv_offsets, non_blocking=True)
                self._kv_offsets = self._kv_offsets_buf[:len(kv_offsets)]
            else:
                self._kv_offsets = None
        else:
            self._q_offsets = q_offsets
            self._kv_offsets = kv_offsets
        
        self._inner_wrapper.plan(
            qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices, paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, head_dim_vo=head_dim, page_size=page_size,
            causal=False, pos_encoding_mode="NONE",
            q_data_type=q_data_type, mask_mode=MaskMode.BLOCK_EXPANDING.value,
        )
    
    def run(
        self, q: torch.Tensor, paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        sm_scale: Optional[float] = None, return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert self._inner_wrapper is not None, "Must call plan() before run()"
        effective_sm_scale = sm_scale if sm_scale is not None else self._sm_scale
        return self._inner_wrapper.run(
            q, paged_kv_cache, self._q_offsets, self._kv_offsets,
            effective_sm_scale, self._dllm_block_size, return_lse=return_lse,
        )
    
    @property
    def dllm_block_size(self) -> int:
        return self._dllm_block_size


def dtype_map_for_idtype(idtype: torch.dtype) -> str:
    return {torch.int32: "int32_t", torch.int64: "int64_t"}.get(idtype, "int32_t")


class BatchBlockExtendRaggedOffsetWrapper:
    """Batch Block Extend Ragged Attention with Offset Support"""
    
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        dllm_block_size: int = 256,
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_offsets_buf: Optional[torch.Tensor] = None,
        kv_offsets_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        assert (dllm_block_size & (dllm_block_size - 1)) == 0, \
            f"dllm_block_size must be power of 2, got {dllm_block_size}"
        
        self._dllm_block_size = dllm_block_size
        self._kv_layout = kv_layout
        self._backend = backend
        self._device = float_workspace_buffer.device
        self._dtype: Optional[torch.dtype] = None
        self._head_dim: Optional[int] = None
        
        self._float_workspace_buffer = float_workspace_buffer
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._q_offsets_buf = q_offsets_buf
        self._kv_offsets_buf = kv_offsets_buf
        
        self._inner_wrapper: Optional[BatchPrefillWithRaggedKVCacheWrapper] = None
        self._q_offsets: Optional[torch.Tensor] = None
        self._kv_offsets: Optional[torch.Tensor] = None
    
    def _create_inner_wrapper(self, dtype: torch.dtype, head_dim: int, idtype: torch.dtype = torch.int32) -> None:
        effective_backend = select_best_backend(head_dim, dtype, self._backend, self._device)
        if effective_backend != self._backend:
            self._backend = effective_backend
        
        if self._backend == "fa3":
            uri = _get_batch_be_module_uri(head_dim, dtype) + "_ragged_offset_fa3"
            variant_name = "BatchBlockExtendOffsetAttentionFA3"
            variant_decl = _BATCH_BE_OFFSET_VARIANT_DECL_FA3
        else:
            uri = _get_batch_be_module_uri(head_dim, dtype) + "_ragged_offset"
            variant_name = "BatchBlockExtendOffsetAttention"
            variant_decl = _BATCH_BE_OFFSET_VARIANT_DECL
        
        jit_args = [
            uri, dtype, dtype, dtype, idtype, head_dim, head_dim,
            ["maybe_q_block_expanding_offset", "maybe_kv_block_expanding_offset"],
            [dtype_map_for_idtype(idtype), dtype_map_for_idtype(idtype)],
            ["sm_scale", "dllm_block_size"], ["double", "int64_t"],
            variant_name, variant_decl,
        ]
        jit_kwargs = {
            "pos_encoding_mode": 0, "use_sliding_window": False,
            "use_logits_soft_cap": False, "use_fp16_qk_reduction": False,
            "mask_modes": [0, 1, 2, 3, 4],
        }
        
        self._inner_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self._float_workspace_buffer, kv_layout=self._kv_layout,
            use_cuda_graph=self._use_cuda_graph, qo_indptr_buf=self._qo_indptr_buf,
            kv_indptr_buf=self._kv_indptr_buf, backend=self._backend,
            jit_args=jit_args, jit_kwargs=jit_kwargs,
        )
        self._dtype = dtype
        self._head_dim = head_dim
    
    def plan(
        self, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor,
        num_qo_heads: int, num_kv_heads: int, head_dim: int,
        q_data_type: torch.dtype = torch.float16, sm_scale: Optional[float] = None,
        q_offsets: Optional[torch.Tensor] = None, kv_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        if self._inner_wrapper is None or self._head_dim != head_dim or self._dtype != q_data_type:
            self._create_inner_wrapper(q_data_type, head_dim, qo_indptr.dtype)
        
        self._sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
        
        if self._use_cuda_graph:
            if q_offsets is not None:
                if self._q_offsets_buf is None:
                    raise ValueError("q_offsets_buf must be provided in CUDA Graph mode")
                self._q_offsets_buf[:len(q_offsets)].copy_(q_offsets, non_blocking=True)
                self._q_offsets = self._q_offsets_buf[:len(q_offsets)]
            else:
                self._q_offsets = None
            
            if kv_offsets is not None:
                if self._kv_offsets_buf is None:
                    raise ValueError("kv_offsets_buf must be provided in CUDA Graph mode")
                self._kv_offsets_buf[:len(kv_offsets)].copy_(kv_offsets, non_blocking=True)
                self._kv_offsets = self._kv_offsets_buf[:len(kv_offsets)]
            else:
                self._kv_offsets = None
        else:
            self._q_offsets = q_offsets
            self._kv_offsets = kv_offsets
        
        self._inner_wrapper.plan(
            qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, head_dim_vo=head_dim,
            causal=False, pos_encoding_mode="NONE",
            q_data_type=q_data_type, mask_mode=MaskMode.BLOCK_EXPANDING.value,
        )
    
    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        sm_scale: Optional[float] = None, return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert self._inner_wrapper is not None, "Must call plan() before run()"
        effective_sm_scale = sm_scale if sm_scale is not None else self._sm_scale
        return self._inner_wrapper.run(
            q, k, v, self._q_offsets, self._kv_offsets,
            effective_sm_scale, self._dllm_block_size, return_lse=return_lse,
        )
    
    @property
    def dllm_block_size(self) -> int:
        return self._dllm_block_size


def batch_block_extend_cascade(
    q: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_curr_indptr: torch.Tensor,
    paged_kv_cache: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    paged_kv_indptr: Optional[torch.Tensor] = None,
    paged_kv_indices: Optional[torch.Tensor] = None,
    paged_kv_last_page_len: Optional[torch.Tensor] = None,
    page_size: int = 16,
    dllm_block_size: int = 256,
    q_offsets: Optional[torch.Tensor] = None,
    kv_offsets: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    backend: str = "auto",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Batch Block Extend Cascade Attention (Current Chunk + Prefix + Merge State)"""
    from ..cascade import merge_state
    
    assert q.dim() == 3 and k_current.dim() == 3 and v_current.dim() == 3
    
    device = q.device
    head_dim = q.size(-1)
    num_qo_heads = q.size(1)
    num_kv_heads = k_current.size(1)
    batch_size = qo_indptr.size(0) - 1
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    if backend == "auto":
        from ..utils import is_sm90a_supported
        actual_backend = "fa3" if is_sm90a_supported(device) else "fa2"
    else:
        actual_backend = backend
    
    if workspace_buffer is None:
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    
    has_prefix = (
        paged_kv_cache is not None and paged_kv_indptr is not None and
        paged_kv_indices is not None and paged_kv_last_page_len is not None and
        paged_kv_indices.size(0) > 0
    )
    
    if q_offsets is None:
        q_offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    if kv_offsets is None:
        kv_offsets = q_offsets
    
    # Stage 1: Current Chunk (Ragged)
    current_wrapper = BatchBlockExtendRaggedOffsetWrapper(
        workspace_buffer, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=actual_backend,
    )
    current_wrapper.plan(
        qo_indptr=qo_indptr, kv_indptr=kv_curr_indptr,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        q_data_type=q.dtype, sm_scale=sm_scale, q_offsets=q_offsets, kv_offsets=kv_offsets,
    )
    
    if has_prefix:
        o1, s1 = current_wrapper.run(q, k_current, v_current, return_lse=True)
    else:
        return current_wrapper.run(q, k_current, v_current, return_lse=return_lse)
    
    # Stage 2: Prefix (Paged)
    prefix_wrapper = BatchBlockExtendPagedOffsetWrapper(
        workspace_buffer, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=actual_backend,
    )
    prefix_wrapper.plan(
        qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices, paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        page_size=page_size, q_data_type=q.dtype, sm_scale=sm_scale,
        q_offsets=q_offsets, kv_offsets=None,
    )
    o2, s2 = prefix_wrapper.run(q, paged_kv_cache, return_lse=True)
    
    # Stage 3: Merge State
    o, s = merge_state(o1, s1, o2, s2)
    return (o, s) if return_lse else o


def sglang_style_cascade_attention(
    q: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_curr_indptr: torch.Tensor,
    paged_kv_cache: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    paged_kv_indptr: Optional[torch.Tensor] = None,
    paged_kv_indices: Optional[torch.Tensor] = None,
    paged_kv_last_page_len: Optional[torch.Tensor] = None,
    page_size: int = 16,
    workspace_buffer: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    return_lse: bool = False,
    backend: str = "fa2",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """SGLang style Cascade Attention (causal + merge)"""
    from ..cascade import merge_state
    from ..prefill import BatchPrefillWithRaggedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
    
    assert q.dim() == 3 and k_current.dim() == 3 and v_current.dim() == 3
    
    device = q.device
    head_dim = q.size(-1)
    num_qo_heads = q.size(1)
    num_kv_heads = k_current.size(1)
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    if workspace_buffer is None:
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    
    has_prefix = (
        paged_kv_cache is not None and paged_kv_indptr is not None and
        paged_kv_indices is not None and paged_kv_last_page_len is not None and
        paged_kv_indices.size(0) > 0
    )
    
    # Stage 1: Current Chunk (Ragged, causal=True)
    ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, kv_layout="NHD", backend=backend)
    ragged_wrapper.plan(
        qo_indptr=qo_indptr, kv_indptr=kv_curr_indptr,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim, q_data_type=q.dtype, causal=False,
    )
    
    if has_prefix:
        o1, s1 = ragged_wrapper.run(q, k_current, v_current, return_lse=True)
    else:
        return ragged_wrapper.run(q, k_current, v_current, return_lse=return_lse)
    
    # Stage 2: Prefix (Paged, causal=False)
    paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD", backend=backend)
    paged_wrapper.plan(
        qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices, paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim, page_size=page_size,
        q_data_type=q.dtype, causal=False,
    )
    o2, s2 = paged_wrapper.run(q, paged_kv_cache, return_lse=True)
    
    # Stage 3: Merge State
    o, s = merge_state(o1, s1, o2, s2)
    return (o, s) if return_lse else o
