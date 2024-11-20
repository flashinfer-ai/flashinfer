import functools
import math

import flashinfer.jit
import torch
from flashinfer.decode import single_decode_with_kv_cache_with_jit_module
from flashinfer.jit.attention import (
    gen_customize_single_decode_module,
    gen_customize_single_prefill_module,
    single_decode_suffix,
    single_prefill_suffix,
)
from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module
from flashinfer.utils import MaskMode

import flashinfer


def test_single_decode_mask():
    torch.manual_seed(42)
    variant_decl = r"""
template <typename ParamsT_>
struct SingleDecodeWithCustomMask {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t window_left, qo_len, kv_len;

  // Create closure
  __device__ __host__ SingleDecodeWithCustomMask(const ParamsT& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    custom_mask_ptr = params.custom_mask;
    qo_len = 1;
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    const uint32_t offset = kv_idx;
    return ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
  }
};
"""
    jit_module = gen_customize_single_decode_module(
        "single_decode_with_custom_mask",
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # head_dim
        ["custom_mask"],  # additional_input_tensor_var_names
        ["uint8_t"],  # additional_input_tensor_var_types
        ["sm_scale"],  # # additional_input_scalar_var_names
        ["float"],  # additional_input_scalar_var_types
        "SingleDecodeWithCustomMask",
        variant_decl,
    )

    f = functools.partial(single_decode_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(32, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(254, 32, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(254, 32, 128, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)

    custom_mask = torch.randint(0, 2, (254,), dtype=torch.uint8, device="cuda")
    packed_custom_mask = flashinfer.packbits(custom_mask, bitorder="little")

    o = f(q, k, v, packed_custom_mask, sm_scale)

    p = torch.einsum("hd,nhd->hn", q.float(), k.float()) * sm_scale
    p[:, torch.nonzero(torch.logical_not(custom_mask)).squeeze()] = -float("inf")
    o_ref = torch.einsum("hn,nhd->hd", torch.softmax(p, dim=-1), v.float()).half()
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


def test_flash_sigmoid():
    torch.manual_seed(42)
    variant_decl = r"""
template <typename ParamsT_>
struct FlashSigmoid {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_softmax = false;

  uint32_t window_left, qo_len, kv_len;
  float sigmoid_bias_log2e;

  // Create closure
  __device__ __host__ FlashSigmoid(const ParamsT& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sigmoid_bias_log2e = params.sigmoid_bias * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.logits_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits + sigmoid_bias_log2e)));
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};
"""
    jit_module = gen_customize_single_prefill_module(
        "flash_sigmoid",
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim
        [],  # additional_input_tensor_var_names
        [],  # additional_input_tensor_var_types
        ["logits_scale", "sigmoid_bias"],  # additional_input_scalar_var_names
        ["float", "float"],  # additional_input_scalar_var_types
        "FlashSigmoid",
        variant_decl,
    )

    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(128, 8, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1027, 8, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1027, 8, 128, dtype=torch.float16, device="cuda")
    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25
    o = f(q, k, v, logits_scale, sigmoid_bias, mask_mode=MaskMode.NON_CAUSAL.value)

    p = torch.sigmoid(
        torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * logits_scale + sigmoid_bias
    )
    o_ref = torch.einsum("hmn,nhd->mhd", p, v.float()).half()
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


def test_dump_logits():
    torch.manual_seed(42)
    variant_decl = r"""
template <typename ParamsT_>
struct DumpLogits {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  __device__ __host__ DumpLogits(const ParamsT& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if (qo_idx < qo_len && kv_idx < kv_len) {
      params.output_logits[qo_head_idx * (qo_len * kv_len) + qo_idx * kv_len + kv_idx] = logits * math::loge2;
    }
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};
"""
    jit_module = gen_customize_single_prefill_module(
        "dump_logits",
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim
        ["output_logits"],  # additional_input_tensor_var_names
        ["float"],  # additional_input_tensor_var_types
        ["sm_scale"],  # additional_input_scalar_var_names
        ["float"],  # additional_input_scalar_var_types
        "DumpLogits",
        variant_decl,
    )

    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(128, 32, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1023, 32, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1023, 32, 128, dtype=torch.float16, device="cuda")
    logits = torch.empty(32, 128, 1023, dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, logits, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)

    p = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * sm_scale
    o_ref = torch.einsum("hmn,nhd->mhd", torch.softmax(p, dim=-1), v.float()).half()
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(logits, p, rtol=2e-2, atol=2e-2)


def test_debug_print_logits():
    torch.manual_seed(42)
    variant_decl = r"""
template <typename ParamsT_>
struct DebugPrintLogits {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  __device__ __host__ DebugPrintLogits(const ParamsT& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if (logits >= 5) {
      printf("Large logits at qo_idx=%d, kv_idx=%d, qo_head_idx=%d, kv_head_idx=%d: %.3f\n",
             qo_idx, kv_idx, qo_head_idx, kv_head_idx, float(logits));
    }
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};
"""
    jit_module = gen_customize_single_prefill_module(
        "debug_print_logits",
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim
        [],  # additional_input_tensor_var_names
        [],  # additional_input_tensor_var_types
        ["sm_scale"],  # additional_input_scalar_var_names
        ["float"],  # additional_input_scalar_var_types
        "DebugPrintLogits",
        variant_decl,
    )

    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(128, 32, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1023, 32, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1023, 32, 128, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)

    p = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * sm_scale
    o_ref = torch.einsum("hmn,nhd->mhd", torch.softmax(p, dim=-1), v.float()).half()
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_single_decode_mask()
    test_flash_sigmoid()
    test_dump_logits()
    test_debug_print_logits()
