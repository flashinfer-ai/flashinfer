
import torch
import flashinfer
import flashinfer.jit
import functools
from flashinfer.utils import MaskMode
from flashinfer.jit.attention import get_customize_single_decode_cu_str, get_customize_single_prefill_cu_str
from flashinfer.decode import single_decode_with_kv_cache_with_jit_module
from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module


def test_single_decode_mask():
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
    cuda_ops_str = get_customize_single_decode_cu_str(
        torch.float16,
        torch.float16,
        torch.float16,
        128,
        ["sm_scale"],  # # additional_input_scalar_var_names
        ["float"],  # additional_input_scalar_var_types
        ["custom_mask"],  # additional_input_tensor_var_names
        ["uint8_t"],  # additional_input_tensor_var_types
        "SingleDecodeWithCustomMask",
        variant_decl,
    )
    gen_directory = flashinfer.jit.FLASHINFER_GEN_SRC_DIR
    flashinfer.jit.utils.write_if_different(
        gen_directory / "single_decode_with_custom_mask.cu",
        cuda_ops_str,
    )
    
    jit_module = flashinfer.jit.load_cuda_ops(
        "single_decode_with_custom_mask",
        [gen_directory / "single_decode_with_custom_mask.cu"],
    )

    f = functools.partial(single_decode_with_kv_cache_with_jit_module, jit_module)

def test_flash_sigmoid():
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

  // Create closure
  __device__ __host__ FlashSigmoid(const ParamsT& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits)));
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};
"""
    cuda_ops_str = get_customize_single_prefill_cu_str(
        torch.float16,
        torch.float16,
        torch.float16,
        128,
        MaskMode.NON_CAUSAL.value,
        [],  # additional_input_scalar_var_names
        [],  # additional_input_scalar_var_types
        [],  # additional_input_tensor_var_names
        [],  # additional_input_tensor_var_types
        "FlashSigmoid",
        variant_decl,
    )

    gen_directory = flashinfer.jit.FLASHINFER_GEN_SRC_DIR
    flashinfer.jit.utils.write_if_different(
        gen_directory / "flash_sigmoid.cu",
        cuda_ops_str,
    )
    
    jit_module = flashinfer.jit.load_cuda_ops(
        "flash_sigmoid",
        [gen_directory / "flash_sigmoid.cu"],
    )

    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(128, 1, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(254, 1, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(254, 1, 128, dtype=torch.float16, device="cuda")
    o = f(q, k, v)

    p = torch.sigmoid(torch.einsum("mhd,nhd->hmn", q, k).float())
    o_ref = torch.einsum("hmn,nhd->mhd", p.half(), v)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=2e-3)


if __name__ == "__main__":
    test_single_decode_mask()
    test_flash_sigmoid()
