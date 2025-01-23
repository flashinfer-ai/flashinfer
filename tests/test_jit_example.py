import functools
import math

import pytest
import torch

import flashinfer
import flashinfer.jit
from flashinfer.decode import single_decode_with_kv_cache_with_jit_module
from flashinfer.jit.attention import (
    gen_customize_single_decode_module,
    gen_customize_single_prefill_module,
)
from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module
from flashinfer.utils import MaskMode


def test_single_decode_mask():
    torch.manual_seed(42)
    variant_decl = r"""
struct SingleDecodeWithCustomMask {
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t window_left, qo_len, kv_len;

  // Create closure
  template <typename Params>
  __device__ __host__ SingleDecodeWithCustomMask(const Params& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    custom_mask_ptr = params.custom_mask;
    qo_len = 1;
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    const uint32_t offset = kv_idx;
    return ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
  }
};
"""
    jit_module = gen_customize_single_decode_module(
        "single_decode_custom_mask",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # head_dim
        ["custom_mask"],  # additional_tensor_names
        ["uint8_t"],  # additional_tensor_dtypes
        ["sm_scale"],  # # additional_scalar_names
        ["float"],  # additional_scalar_dtypes
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


flash_sigmoid_sm80_decl = r"""
struct FlashSigmoid {
  static constexpr bool use_softmax = false;

  uint32_t window_left, qo_len, kv_len;
  float sigmoid_bias_log2e;

  // Create closure
  template <typename Params>
  __device__ __host__ FlashSigmoid(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sigmoid_bias_log2e = params.sigmoid_bias * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.logits_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits + sigmoid_bias_log2e)));
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};
"""

flash_sigmoid_sm90_decl = r"""
struct FlashSigmoid {
  template <int NUM_ROWS_PER_THREAD>
  using Updater = DefaultUpdater<NUM_ROWS_PER_THREAD>;

  float logits_scale_log2, sigmoid_bias_log2e;
  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ FlashSigmoid(const MainloopParams& params, const BlockCoord& block_coord) {
    logits_scale_log2 = params.additional_params.logits_scale * math::log2e;
    sigmoid_bias_log2e = params.additional_params.sigmoid_bias * math::log2e;
  }

  template <typename MainloopParams, typename T>
  __device__ __forceinline__ T LogitsTransform(const MainloopParams& params, T logits,
                                               int batch_idx,
                                               int qo_idx, int kv_idx,
                                               int qo_head_idx, int kv_head_idx) {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits * logits_scale_log2 + sigmoid_bias_log2e)));
  }
};
"""


def test_flash_sigmoid():
    torch.manual_seed(42)
    variant_decl = flash_sigmoid_sm80_decl
    jit_module = gen_customize_single_prefill_module(
        "fa2",  # backend
        "single_prefill_flash_sigmoid",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["float", "float"],  # additional_scalar_dtypes
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
struct DumpLogits {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  template <typename Params>
  __device__ __host__ DumpLogits(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if (qo_idx < qo_len && kv_idx < kv_len) {
      params.output_logits[qo_head_idx * (qo_len * kv_len) + qo_idx * kv_len + kv_idx] = logits * math::loge2;
    }
    return logits;
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa2",  # backend
        "single_prefill_dump_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim
        ["output_logits"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["float"],  # additional_scalar_dtypes
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


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_flash_sigmoid(use_tensor_cores):
    torch.manual_seed(42)
    variant_decl = flash_sigmoid_sm80_decl
    jit_args = (
        "batch_decode_flash_sigmoid_sm80",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["float", "float"],  # additional_scalar_dtypes
        "FlashSigmoid",
        variant_decl,
    )

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_tensor_cores=use_tensor_cores,
        jit_args=jit_args,
    )

    batch_size = 128
    seq_len_per_request = 1024
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )
    page_size = 1
    kv_indices_host = torch.arange(
        0, batch_size * seq_len_per_request, dtype=torch.int32
    )
    last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128

    wrapper.plan(
        kv_indptr_host,
        kv_indices_host,
        last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    q = torch.randn(
        batch_size,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k_cache = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    v_cache = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25

    o = wrapper.run(q, (k_cache, v_cache), logits_scale, sigmoid_bias)
    p = torch.sigmoid(
        torch.einsum(
            "bhd,bnhd->bhn",
            q.view(batch_size, num_qo_heads, head_dim).float(),
            k_cache.view(
                batch_size, seq_len_per_request, num_kv_heads, head_dim
            ).float(),
        )
        * logits_scale
        + sigmoid_bias
    )
    o_ref = (
        torch.einsum(
            "bhn,bnhd->bhd",
            p,
            v_cache.view(
                batch_size, seq_len_per_request, num_kv_heads, head_dim
            ).float(),
        )
        .half()
        .reshape(batch_size, num_qo_heads, head_dim)
    )

    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


def test_batch_prefill_flash_sigmoid():
    torch.manual_seed(42)
    variant_decl = flash_sigmoid_sm80_decl
    jit_args = (
        "batch_prefill_flash_sigmoid_sm80",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["float", "float"],  # additional_scalar_dtypes
        "FlashSigmoid",
        variant_decl,
    )

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )

    batch_size = 128
    seq_len_per_request = 1024
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )

    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=False,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    q = torch.randn(
        batch_size * seq_len_per_request,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    v = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25

    o = wrapper.run(q, k, v, logits_scale, sigmoid_bias)

    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    kv_indices_host = torch.arange(
        0,
        batch_size * seq_len_per_request,
        dtype=torch.int32,
    )
    paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )
    o_paged = wrapper_paged.run(q, (k, v), logits_scale, sigmoid_bias)

    p = torch.sigmoid(
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, seq_len_per_request, num_qo_heads, head_dim).float(),
            k.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).float(),
        )
        * logits_scale
        + sigmoid_bias
    )
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).float(),
        )
        .half()
        .reshape(batch_size * seq_len_per_request, num_qo_heads, head_dim)
    )
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(o_paged, o_ref, rtol=2e-2, atol=2e-2)


def test_batch_prefill_sm90_flash_sigmoid():
    torch.manual_seed(42)
    variant_decl = flash_sigmoid_sm90_decl
    jit_args = (
        "batch_prefill_flash_sigmoid",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["float", "float"],  # additional_scalar_dtypes
        "FlashSigmoid",
        variant_decl,
    )

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa3", jit_args=jit_args
    )

    batch_size = 128
    seq_len_per_request = 1024
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )

    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=False,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    q = torch.randn(
        batch_size * seq_len_per_request,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    v = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25

    o = wrapper.run(q, k, v, logits_scale, sigmoid_bias)
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa3", jit_args=jit_args
    )
    kv_indices_host = torch.arange(
        0,
        batch_size * seq_len_per_request,
        dtype=torch.int32,
    )
    paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )
    o_paged = wrapper_paged.run(q, (k, v), logits_scale, sigmoid_bias)

    p = torch.sigmoid(
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, seq_len_per_request, num_qo_heads, head_dim).float(),
            k.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).float(),
        )
        * logits_scale
        + sigmoid_bias
    )
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).float(),
        )
        .half()
        .reshape(batch_size * seq_len_per_request, num_qo_heads, head_dim)
    )
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(o_paged, o_ref, rtol=2e-2, atol=2e-2)


def test_debug_print_logits():
    torch.manual_seed(42)
    variant_decl = r"""
struct DebugPrintLogits {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  template <typename Params>
  __device__ __host__ DebugPrintLogits(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if (logits >= 5) {
      printf("Large logits at qo_idx=%d, kv_idx=%d, qo_head_idx=%d, kv_head_idx=%d: %.3f\n",
             qo_idx, kv_idx, qo_head_idx, kv_head_idx, float(logits));
    }
    return logits;
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa2",  # backend
        "batch_prefill_debug_print_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["float"],  # additional_scalar_dtypes
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


def test_sm90_debug_print_logits():
    torch.manual_seed(42)
    variant_decl = r"""
struct DebugPrintLogits {

  template <int NUM_ROWS_PER_THREAD>
  using Updater = OnlineSoftmaxWithoutScale<NUM_ROWS_PER_THREAD>;

  float sm_scale_log2;
  int qo_len, kv_len;

  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ DebugPrintLogits(const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    auto [_, __, ___, ____, _____, qo_len_, kv_len_] =
        block_coord;

    qo_len = qo_len_;
    kv_len = kv_len_;
  }

  template <typename MainloopParams, typename T>
  __device__ __forceinline__ T LogitsTransform(const MainloopParams& params, T logits,
                                               int batch_idx,
                                               int qo_idx, int kv_idx,
                                               int qo_head_idx, int kv_head_idx) {
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
  }
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa3",  # backend
        "debug_print_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # hidden_dim
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["float"],  # additional_scalar_dtypes
        "DebugPrintLogits",
        variant_decl,
    )

    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)

    q = torch.randn(16, 2, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(16, 1, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(16, 1, 128, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)

    p = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * sm_scale
    o_ref = torch.einsum("hmn,nhd->mhd", torch.softmax(p, dim=-1), v.float()).half()
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # test_single_decode_mask()
    # test_flash_sigmoid()
    # test_dump_logits()
    # test_debug_print_logits()
    # test_sm90_debug_print_logits()
    # test_batch_decode_flash_sigmoid(False)
    # test_batch_decode_flash_sigmoid(True)
    # test_batch_prefill_flash_sigmoid()
    test_batch_prefill_sm90_flash_sigmoid()
