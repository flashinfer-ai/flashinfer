"""Private JIT wiring for native Block Extend batch prefill."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from .utils import MaskMode


_FA2_VARIANT_DECL = r"""
struct BlockExtendBatchAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t window_left;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ BlockExtendBatchAttention(
      const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
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


_FA3_VARIANT_DECL = r"""
struct BlockExtendBatchAttentionFA3 : AttentionVariantBase {
  float sm_scale_log2;

  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ BlockExtendBatchAttentionFA3(
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


def _get_uri(
    head_dim_qk: int,
    dtype_q: torch.dtype,
    idtype: torch.dtype,
    head_dim_vo: int,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
) -> str:
    dtype_uri = {torch.float16: "fp16", torch.bfloat16: "bf16"}
    idtype_uri = {torch.int32: "i32", torch.int64: "i64"}
    if head_dim_qk not in (64, 128) or head_dim_vo not in (64, 128):
        raise ValueError("block_extend only supports head dimensions 64 and 128")
    if (
        dtype_q not in dtype_uri
        or dtype_kv not in dtype_uri
        or dtype_o not in dtype_uri
    ):
        raise ValueError("block_extend only supports fp16 and bf16")
    if idtype not in idtype_uri:
        raise ValueError("block_extend only supports int32 and int64 indptr")
    return (
        f"batch_prefill_block_extend_hd{head_dim_qk}_{dtype_uri[dtype_q]}_"
        f"idx{idtype_uri[idtype]}_vo{head_dim_vo}_{dtype_uri[dtype_kv]}_"
        f"{dtype_uri[dtype_o]}"
    )


def build_block_extend_jit_args(
    head_dim: int,
    dtype: torch.dtype,
    idtype: torch.dtype,
    backend: str,
    layout: str,
    head_dim_vo: Optional[int] = None,
    dtype_kv: Optional[torch.dtype] = None,
    dtype_o: Optional[torch.dtype] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Return fixed-mode JIT arguments for the existing batch wrappers."""
    if layout not in ("ragged", "paged"):
        raise ValueError(f"layout must be 'ragged' or 'paged', got {layout!r}")
    if backend == "fa2":
        suffix, variant_name, variant_decl = (
            f"_{layout}_offset",
            "BlockExtendBatchAttention",
            _FA2_VARIANT_DECL,
        )
    elif backend == "fa3":
        suffix, variant_name, variant_decl = (
            f"_{layout}_offset_fa3",
            "BlockExtendBatchAttentionFA3",
            _FA3_VARIANT_DECL,
        )
    else:
        raise ValueError(f"backend must be 'fa2' or 'fa3', got {backend!r}")

    head_dim_vo = head_dim if head_dim_vo is None else head_dim_vo
    dtype_kv = dtype if dtype_kv is None else dtype_kv
    dtype_o = dtype if dtype_o is None else dtype_o
    uri = _get_uri(head_dim, dtype, idtype, head_dim_vo, dtype_kv, dtype_o) + suffix
    idtype_name = {torch.int32: "int32_t", torch.int64: "int64_t"}[idtype]
    jit_args = [
        uri,
        dtype,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim,
        head_dim_vo,
        ["maybe_q_block_extend_offset", "maybe_kv_block_extend_offset"],
        [idtype_name, idtype_name],
        ["sm_scale", "dllm_block_size"],
        ["double", "int64_t"],
        variant_name,
        variant_decl,
    ]
    jit_kwargs = {
        "pos_encoding_mode": 0,
        "use_sliding_window": False,
        "use_logits_soft_cap": False,
        "use_fp16_qk_reduction": False,
        "mask_modes": [MaskMode.BLOCK_EXTEND.value],
    }
    return jit_args, jit_kwargs
