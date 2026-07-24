"""Private JIT wiring for native Block Extend single prefill."""

from typing import Optional

import torch

from .jit.attention.modules import gen_customize_block_extend_single_prefill_module
from .utils import is_sm90a_supported


_FA2_VARIANT_DECL = r"""
struct BlockExtendSingleAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t window_left;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ BlockExtendSingleAttention(
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
struct BlockExtendSingleAttentionFA3 : AttentionVariantBase {
  float sm_scale_log2;

  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ BlockExtendSingleAttentionFA3(
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


def _dtype_uri(dtype: torch.dtype) -> str:
    names = {torch.float16: "fp16", torch.bfloat16: "bf16"}
    try:
        return names[dtype]
    except KeyError as error:
        raise ValueError("block_extend only supports fp16 and bf16") from error


def _get_uri(
    head_dim_qk: int,
    head_dim_vo: int,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    backend: str,
) -> str:
    if head_dim_qk not in (64, 128) or head_dim_vo not in (64, 128):
        raise ValueError("block_extend only supports head dimensions 64 and 128")
    return (
        f"block_extend_{backend}_hd{head_dim_qk}_vo{head_dim_vo}_"
        f"{_dtype_uri(dtype_q)}_{_dtype_uri(dtype_kv)}_{_dtype_uri(dtype_o)}"
    )


def get_block_extend_single_prefill_module(
    head_dim: int,
    dtype: torch.dtype,
    backend: str,
    device: torch.device,
    head_dim_vo: Optional[int] = None,
    dtype_kv: Optional[torch.dtype] = None,
    dtype_o: Optional[torch.dtype] = None,
):
    """Build the fixed-mode module used by the native single-prefill option."""
    if backend not in ("fa2", "fa3"):
        raise ValueError(f"backend must be 'fa2' or 'fa3', got {backend!r}")
    if backend == "fa3" and not is_sm90a_supported(device):
        raise RuntimeError("block_extend fa3 backend requires SM90/Hopper architecture")

    head_dim_vo = head_dim if head_dim_vo is None else head_dim_vo
    dtype_kv = dtype if dtype_kv is None else dtype_kv
    dtype_o = dtype if dtype_o is None else dtype_o
    if backend == "fa2":
        variant_name, variant_decl = "BlockExtendSingleAttention", _FA2_VARIANT_DECL
    else:
        variant_name, variant_decl = (
            "BlockExtendSingleAttentionFA3",
            _FA3_VARIANT_DECL,
        )

    spec = gen_customize_block_extend_single_prefill_module(
        backend=backend,
        uri=_get_uri(head_dim, head_dim_vo, dtype, dtype_kv, dtype_o, backend),
        dtype_q=dtype,
        dtype_kv=dtype_kv,
        dtype_o=dtype_o,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim_vo,
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=[
            "sm_scale",
            "dllm_block_size",
            "q_block_extend_offset",
            "kv_block_extend_offset",
        ],
        additional_scalar_dtypes=["double", "int64_t", "int64_t", "int64_t"],
        variant_name=variant_name,
        variant_decl=variant_decl,
    )
    return spec.build_and_load()
