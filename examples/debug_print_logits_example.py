import functools
import math

import torch

from flashinfer.jit.attention import gen_customize_single_prefill_module
from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module
from flashinfer.utils import MaskMode, is_sm90a_supported


def debug_print_logits_sm80():
    torch.manual_seed(42)
    variant_decl = r"""
struct DebugPrintLogits : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ DebugPrintLogits(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (logits >= 5) {
      printf("Large logits at qo_idx=%d, kv_idx=%d, qo_head_idx=%d, kv_head_idx=%d: %.3f\n",
              qo_idx, kv_idx, qo_head_idx, kv_head_idx, float(logits));
    }
    return logits;
  });
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa2",  # backend
        "batch_prefill_debug_print_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "DebugPrintLogits",
        variant_decl,
    ).build_and_load()

    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(128, 32, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1023, 32, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1023, 32, 128, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)

    p = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * sm_scale
    o_ref = torch.einsum("hmn,nhd->mhd", torch.softmax(p, dim=-1), v.float()).half()
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    print("SM80 debug print logits example passed!")


def debug_print_logits_sm90():
    if not is_sm90a_supported(torch.device("cuda")):
        print("SM90A is not supported, skipping...")
        return

    torch.manual_seed(42)
    variant_decl = r"""
struct DebugPrintLogits : AttentionVariantBase {
  float sm_scale_log2;
  int qo_len, kv_len;

  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ DebugPrintLogits(const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    auto [_, __, ___, ____, _____, qo_len_, kv_len_, batch_idx] =
        block_coord;

    qo_len = qo_len_;
    kv_len = kv_len_;
  }


  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE*/false>(sm_scale_log2);
  }


  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (qo_idx < qo_len && kv_idx < kv_len) {
        printf(
            "---> LOGITS DEBUG: "
            "qo_idx=%-5d "
            "kv_idx=%-5d "
            "sm_scale_log2=%-12.5f "
            "logits=%-12.5f "
            "\n",
            qo_idx,
            kv_idx,
            sm_scale_log2,
            static_cast<float>(logits));
    }
    logits *= sm_scale_log2;
    return logits;
  })
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa3",  # backend
        "debug_print_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "DebugPrintLogits",
        variant_decl,
    ).build_and_load()

    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(16, 2, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(16, 1, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(16, 1, 128, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)

    p = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * sm_scale
    o_ref = torch.einsum("hmn,nhd->mhd", torch.softmax(p, dim=-1), v.float()).half()
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    print("SM90 debug print logits example passed!")


if __name__ == "__main__":
    print("Running SM80 debug print logits example...")
    debug_print_logits_sm80()
    print("\nRunning SM90 debug print logits example...")
    debug_print_logits_sm90()
