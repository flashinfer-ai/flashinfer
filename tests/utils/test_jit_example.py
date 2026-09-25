import functools
import math

import pytest
import torch

import flashinfer
from flashinfer.decode import single_decode_with_kv_cache_with_jit_module
from flashinfer.jit.attention import (
    gen_customize_single_decode_module,
    gen_customize_single_prefill_module,
)
from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module
from flashinfer.utils import (
    SINGLE_KERNEL_TMP_SIZE,
    MaskMode,
    TensorLayout,
    get_compute_capability,
    is_sm90a_supported,
)


def test_single_decode_mask():
    torch.manual_seed(42)
    variant_decl = r"""
struct SingleDecodeWithCustomMask : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ SingleDecodeWithCustomMask(const Params& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    custom_mask_ptr = params.custom_mask;
    qo_len = 1;
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    const uint32_t offset = kv_idx;
    return ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
  })

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
    return output * d_rcp;
  })
};
"""
    jit_module = gen_customize_single_decode_module(
        "single_decode_custom_mask",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # head_dim_qk
        128,  # head_dim_vo
        ["custom_mask"],  # additional_tensor_names
        ["uint8_t"],  # additional_tensor_dtypes
        ["sm_scale"],  # # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "SingleDecodeWithCustomMask",
        variant_decl,
    ).build_and_load()

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
struct FlashSigmoid : AttentionVariantBase {
  static constexpr bool use_softmax = false;

  uint32_t window_left, qo_len, kv_len;
  float sigmoid_scale_log2;
  float sigmoid_bias_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ FlashSigmoid(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sigmoid_bias_log2 = params.sigmoid_bias * math::log2e;
    sigmoid_scale_log2 = params.logits_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits * sigmoid_scale_log2 + sigmoid_bias_log2)));
  });

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    return output;
  })
};
"""

flash_sigmoid_sm90_decl = r"""
struct FlashSigmoid : AttentionVariantBase {
  float logits_scale_log2, sigmoid_bias_log2e;
  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ FlashSigmoid(const MainloopParams& params, const BlockCoord& block_coord) {
    logits_scale_log2 = params.additional_params.logits_scale * math::log2e;
    sigmoid_bias_log2e = params.additional_params.sigmoid_bias * math::log2e;
  }


  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return DefaultUpdater<NUM_ROWS_PER_THREAD>();
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits * logits_scale_log2 + sigmoid_bias_log2e)));
  });
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
        128,  # head_dim_qk
        128,  # head_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["double", "double"],  # additional_scalar_dtypes
        "FlashSigmoid",
        variant_decl,
    ).build_and_load()

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
struct DumpLogits : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ DumpLogits(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (qo_idx < qo_len && kv_idx < kv_len) {
      params.output_logits[qo_head_idx * (qo_len * kv_len) + qo_idx * kv_len + kv_idx] = logits * params.sm_scale;
    }
    return logits;
  });
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa2",  # backend
        "single_prefill_dump_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        128,  # head_dim_qk
        128,  # head_dim_vo
        ["output_logits"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "DumpLogits",
        variant_decl,
    ).build_and_load()

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
        f"batch_decode_flash_sigmoid_sm80_{use_tensor_cores}",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["double", "double"],  # additional_scalar_dtypes
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
        backend="fa2",
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
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["double", "double"],  # additional_scalar_dtypes
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


variant_owned_window_decl = r"""
struct WindowOwnedMask : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ WindowOwnedMask(const Params& params, uint32_t batch_idx,
                                      uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    // The mask owns the window: keep KV traversal un-pruned.
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    // Clamp the CTA_TILE_Q padding lanes whose results are discarded.
    const uint32_t q_local = qo_idx < qo_len ? qo_idx : qo_len - 1;
    const uint32_t q_abs = kv_len - qo_len + q_local;
    return (kv_idx <= q_abs) && (q_abs - kv_idx < uint32_t(params.mask_window));
  })
};
"""


def _owned_mask_jit_args(uri):
    return (
        uri,
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["mask_window", "sm_scale"],  # additional_scalar_names
        ["double", "double"],  # additional_scalar_dtypes
        "WindowOwnedMask",
        variant_owned_window_decl,
    )


def test_batch_prefill_variant_owns_mask():
    """A JIT variant that owns the full mask must get MaskMode::CUSTOM.

    Without a mask tensor, ``causal=False`` selects MaskMode::kNone, under
    which the FA2 kernel only evaluates ``LogitsMask`` on boundary KV tiles.
    ``variant_owns_mask=True`` selects MaskMode::CUSTOM without a mask tensor
    so every interior tile is masked as well. The sequences here are long
    enough to have interior tiles for every CTA_TILE_KV configuration.
    """
    if get_compute_capability(torch.device("cuda")) < (7, 5):
        pytest.skip("FA2 prefill JIT requires SM75 or newer.")
    torch.manual_seed(42)
    jit_args = _owned_mask_jit_args("batch_prefill_variant_owns_mask")

    num_qo_heads = 8
    num_kv_heads = 8
    head_dim = 128
    mask_window = 32.0
    sm_scale = 1.0 / math.sqrt(head_dim)
    lens = [(128, 2048), (64, 1024)]

    qo_indptr_host = torch.tensor(
        [0] + list(torch.tensor([q for q, _ in lens]).cumsum(0)), dtype=torch.int32
    )
    kv_indptr_host = torch.tensor(
        [0] + list(torch.tensor([kv for _, kv in lens]).cumsum(0)), dtype=torch.int32
    )
    total_q = int(qo_indptr_host[-1])
    total_kv = int(kv_indptr_host[-1])
    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(
        total_kv, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
    )
    v = torch.randn(
        total_kv, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
    )

    def ref_output():
        outs = []
        for i, (qo_len, kv_len) in enumerate(lens):
            qs = q[qo_indptr_host[i] : qo_indptr_host[i + 1]].float()
            ks = k[kv_indptr_host[i] : kv_indptr_host[i + 1]].float()
            vs = v[kv_indptr_host[i] : kv_indptr_host[i + 1]].float()
            q_abs = torch.arange(kv_len - qo_len, kv_len, device="cuda").view(-1, 1)
            kv_pos = torch.arange(kv_len, device="cuda").view(1, -1)
            mask = (kv_pos <= q_abs) & (q_abs - kv_pos < mask_window)
            scores = torch.einsum("qhd,khd->hqk", qs, ks) * sm_scale
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
            outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vs))
        return torch.cat(outs, dim=0)

    o_ref = ref_output()
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )

    def plan_ragged(wrapper):
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

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend="fa2",
        jit_args=jit_args,
        variant_owns_mask=True,
    )
    plan_ragged(wrapper)
    o = wrapper.run(q, k, v, mask_window, sm_scale)
    torch.testing.assert_close(o.float(), o_ref, rtol=2e-2, atol=2e-2)

    # Same variant without the flag: MaskMode::kNone skips LogitsMask on
    # interior KV tiles, so the window must NOT be applied there.
    wrapper_none = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend="fa2",
        jit_args=jit_args,
    )
    plan_ragged(wrapper_none)
    o_none = wrapper_none.run(q, k, v, mask_window, sm_scale)
    assert (o_none.float() - o_ref).abs().max() > 1e-2, (
        "MaskMode::kNone unexpectedly applied the variant mask on interior "
        "tiles; variant_owns_mask would be redundant"
    )

    # Paged wrapper, page_size=1 identity table.
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend="fa2",
        jit_args=jit_args,
        variant_owns_mask=True,
    )
    kv_indices_host = torch.arange(0, total_kv, dtype=torch.int32)
    paged_kv_last_page_len_host = torch.full((len(lens),), 1, dtype=torch.int32)
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=False,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_paged = wrapper_paged.run(q, (k, v), mask_window, sm_scale)
    torch.testing.assert_close(o_paged.float(), o_ref, rtol=2e-2, atol=2e-2)


def test_variant_owns_mask_requires_jit_module():
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    for cls in (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper,
    ):
        with pytest.raises(ValueError, match="variant_owns_mask requires"):
            cls(
                float_workspace_buffer,
                kv_layout="NHD",
                backend="fa2",
                variant_owns_mask=True,
            )
        # Rejected before any JIT build: the SM90 batch prefill kernels
        # return cudaErrorNotSupported under MaskMode.CUSTOM.
        with pytest.raises(ValueError, match="only supported on the fa2 backend"):
            cls(
                float_workspace_buffer,
                kv_layout="NHD",
                backend="fa3",
                variant_owns_mask=True,
            )


@pytest.mark.parametrize("paged", [True, False])
def test_variant_owns_mask_rejects_multi_item_scoring(paged):
    """prefix_len_ptr selects MULTIITEMSCORING, which would override CUSTOM."""
    if get_compute_capability(torch.device("cuda")) < (7, 5):
        pytest.skip("FA2 prefill JIT requires SM75 or newer.")
    jit_args = _owned_mask_jit_args("batch_prefill_variant_owns_mask")
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 8], dtype=torch.int32)
    prefix_len_ptr = torch.zeros(1, dtype=torch.uint32, device="cuda")
    if paged:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
            jit_args=jit_args,
            variant_owns_mask=True,
        )
        with pytest.raises(ValueError, match="incompatible"):
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                torch.arange(0, 8, dtype=torch.int32),
                torch.full((1,), 1, dtype=torch.int32),
                8,
                8,
                128,
                1,
                causal=False,
                q_data_type=torch.float16,
                kv_data_type=torch.float16,
                prefix_len_ptr=prefix_len_ptr,
            )
    else:
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
            jit_args=jit_args,
            variant_owns_mask=True,
        )
        with pytest.raises(ValueError, match="incompatible"):
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                8,
                8,
                128,
                causal=False,
                q_data_type=torch.float16,
                kv_data_type=torch.float16,
                prefix_len_ptr=prefix_len_ptr,
            )


def test_batch_prefill_sm90_flash_sigmoid():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    torch.manual_seed(42)
    variant_decl = flash_sigmoid_sm90_decl
    jit_args = (
        "batch_prefill_flash_sigmoid",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        ["logits_scale", "sigmoid_bias"],  # additional_scalar_names
        ["double", "double"],  # additional_scalar_dtypes
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


def test_batch_prefill_jit_wellknown_mask_buffers():
    """Issue #1044: JIT variants using well-known additional tensor names
    (maybe_custom_mask, maybe_mask_indptr) should auto-inject internal buffers
    without the user having to pass them via *args.
    Verifies both argument injection AND numerical correctness of the mask."""
    torch.manual_seed(42)

    variant_decl = r"""
struct FlashCustomMask : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint8_t* custom_mask_ptr;
  uint32_t qo_len, kv_len;
  float sm_scale_log2;
  uint32_t window_left;

  template <typename Params>
  __device__ __host__ FlashCustomMask(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    custom_mask_ptr = params.maybe_custom_mask + params.maybe_mask_indptr[batch_idx];
    sm_scale_log2 = math::log2e;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = true;
    const uint32_t offset = qo_idx * kv_len + kv_idx;
    mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
    return mask;
  })
};
"""
    num_qo_heads = 8
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    batch_size = 1
    seq_len = 16

    jit_args = (
        "batch_prefill_flash_custom_mask_wellknown",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        head_dim,
        head_dim,
        ["maybe_custom_mask", "maybe_mask_indptr"],
        ["uint8_t", "int32_t"],
        [],
        [],
        "FlashCustomMask",
        variant_decl,
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda"
    )

    # Use causal (lower-triangular) mask to verify mask is actually applied
    custom_mask = torch.tril(
        torch.full((batch_size, seq_len, seq_len), True, device="cuda")
    )

    # --- Test paged wrapper ---
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    num_pages = (seq_len + page_size - 1) // page_size
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor(
        [seq_len - (num_pages - 1) * page_size], dtype=torch.int32, device="cuda"
    )
    kv_cache = torch.randn(
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    wrapper_paged.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask,
        causal=False,
    )
    o_masked = wrapper_paged.run(q, kv_cache)

    # Run without mask (non-causal) for comparison
    wrapper_paged_nomask = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        backend="fa2",
    )
    wrapper_paged_nomask.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
    )
    o_nomask = wrapper_paged_nomask.run(q, kv_cache)

    assert o_masked.shape == (batch_size * seq_len, num_qo_heads, head_dim)
    assert not torch.allclose(o_masked, o_nomask, rtol=1e-2, atol=1e-2), (
        "Masked and unmasked outputs should differ, mask was not applied"
    )

    # --- Test ragged wrapper ---
    k_flat = kv_cache[:, 0].reshape(-1, num_kv_heads, head_dim)[:seq_len]
    v_flat = kv_cache[:, 1].reshape(-1, num_kv_heads, head_dim)[:seq_len]
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    wrapper_ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    wrapper_ragged.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        custom_mask=custom_mask,
        causal=False,
    )
    o_ragged_masked = wrapper_ragged.run(q, k_flat, v_flat)

    wrapper_ragged_nomask = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        backend="fa2",
    )
    wrapper_ragged_nomask.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=False,
    )
    o_ragged_nomask = wrapper_ragged_nomask.run(q, k_flat, v_flat)

    assert o_ragged_masked.shape == (batch_size * seq_len, num_qo_heads, head_dim)
    assert not torch.allclose(o_ragged_masked, o_ragged_nomask, rtol=1e-2, atol=1e-2), (
        "Masked and unmasked outputs should differ, mask was not applied"
    )


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_jit_wellknown_alibi_buffer(use_tensor_cores):
    """Issue #1044 (decode): JIT variants using well-known additional tensor name
    (maybe_alibi_slopes) should auto-inject the internal buffer without the user
    having to pass it via *args."""
    torch.manual_seed(42)

    variant_decl = r"""
struct FlashAlibiDecode : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ FlashAlibiDecode(const Params& params, uint32_t batch_idx,
                                       uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    float bias = 0.f;
    if (params.maybe_alibi_slopes != nullptr) {
      bias = params.maybe_alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(kv_len) + 1);
    }
    return logits + bias;
  });
};
"""
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128
    batch_size = 4
    seq_len = 128
    page_size = 1

    jit_args = (
        f"batch_decode_alibi_wellknown_{use_tensor_cores}",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        head_dim,
        head_dim,
        ["maybe_alibi_slopes"],
        ["float"],
        ["sm_scale"],
        ["double"],
        "FlashAlibiDecode",
        variant_decl,
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        use_tensor_cores=use_tensor_cores,
        jit_args=jit_args,
        backend="fa2",
    )

    kv_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len, dtype=torch.int32)
    kv_indices = torch.arange(0, batch_size * seq_len, dtype=torch.int32)
    last_page_len = torch.full((batch_size,), 1, dtype=torch.int32)

    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda"
    )
    k_cache = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
    )
    v_cache = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
    )

    sm_scale = 1.0 / math.sqrt(head_dim)
    o = wrapper.run(q, (k_cache, v_cache), sm_scale)
    assert o.shape == (batch_size, num_qo_heads, head_dim)


# Issue #2765: REGISTER_OUTPUT_TRANSFORM on the Hopper (fa3) variant helper. The variants add a
# per-(query, head) correction to the normalized output, o[q, h, :] += corr[q, h] * hvec[h, :]
# with q the global query row, which exercises the row, head and column coordinates of the hook.

corrected_attention_sm90_decl = r"""
struct CorrectedAttention : AttentionVariantBase {
  float sm_scale_log2;
  uint32_t qo_start, corr_stride;

  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ CorrectedAttention(const MainloopParams& params,
                                         const BlockCoord& block_coord) {
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    qo_start = qo_indptr;
    corr_stride = params.additional_params.corr_stride;
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(sm_scale_log2);
  }

  // qo_idx is request-local; qo_start makes it the global row that indexes corr.
  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, d_idx, lse, {
    const float corr =
        params.additional_params.corr[(qo_start + qo_idx) * corr_stride + qo_head_idx];
    const float h = params.additional_params.hvec[qo_head_idx * HEAD_DIM_VO + d_idx];
    return output + corr * h;
  })
};
"""

# The same correction through the FA2 hook, which also owns the softmax normalization and has
# no column index, so this twin adds a per-(query, head) term only.
corrected_attention_sm80_decl = r"""
struct CorrectedAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ CorrectedAttention(const Params& params, uint32_t batch_idx,
                                         uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
    return output * d_rcp + params.corr[qo_idx * params.corr_stride + qo_head_idx];
  })
};
"""

# FP8 twin: the built-in FP8 variant plus the hook, to cover the FP8 epilogue.
corrected_attention_fp8_sm90_decl = r"""
#include <flashinfer/attention/hopper/variants.cuh>

struct CorrectedFP8Attention : StandardFP8Attention {
  uint32_t qo_start, corr_stride;

  template <typename MainloopParams, typename BlockCoord>
  __device__ CorrectedFP8Attention(const MainloopParams& params, const BlockCoord& block_coord)
      : StandardFP8Attention(params, block_coord) {
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    qo_start = qo_indptr;
    corr_stride = params.additional_params.corr_stride;
  }

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, d_idx, lse, {
    const float corr =
        params.additional_params.corr[(qo_start + qo_idx) * corr_stride + qo_head_idx];
    const float h = params.additional_params.hvec[qo_head_idx * HEAD_DIM_VO + d_idx];
    return output + corr * h;
  })
};
"""

# (tensor names, tensor dtypes, scalar names, scalar dtypes) shared by the fa3 variants above.
_OUTPUT_TRANSFORM_PARAMS = (
    ["corr", "hvec"],
    ["float", "float"],
    ["sm_scale", "corr_stride"],
    ["double", "int64_t"],
)


def _skip_unless_sm90a():
    """Skip unless the current device runs the fa3 (SM90a) kernels."""
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")


def _random_correction(nnz_qo, num_qo_heads, head_dim):
    """Per-(query, head) factors and per-(head, column) vectors, uniform in [-1, 1)."""
    corr = torch.rand(nnz_qo, num_qo_heads, dtype=torch.float32, device="cuda") * 2 - 1
    hvec = (
        torch.rand(num_qo_heads, head_dim, dtype=torch.float32, device="cuda") * 2 - 1
    )
    return corr, hvec


def _attention_reference(q, k, v, causal, sm_scale):
    """fp32 softmax attention with GQA by head repetition and a causal mask aligned to the
    bottom right. Returns (o, lse) with lse in the log2 domain, which is what the kernels
    store. An empty KV yields zeros and -inf."""
    qo_len, kv_len, num_qo_heads = q.size(0), k.size(0), q.size(1)
    if kv_len == 0:
        o = torch.zeros(
            qo_len, num_qo_heads, v.size(-1), dtype=torch.float32, device=q.device
        )
        lse = torch.full((qo_len, num_qo_heads), float("-inf"), device=q.device)
        return o, lse
    group_size = num_qo_heads // k.size(1)
    kf = k.float().repeat_interleave(group_size, dim=1)
    vf = v.float().repeat_interleave(group_size, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kf) * sm_scale
    if causal:
        keep = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device)
        s = s.masked_fill(~keep.tril(kv_len - qo_len), float("-inf"))
    o = torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), vf)
    lse = torch.logsumexp(s, dim=-1).t() * math.log2(math.e)
    return o, lse


def _sm90_output_transform_module(dtype):
    """fa3 single-prefill module for CorrectedAttention with head_dim 128, cached by uri."""
    return gen_customize_single_prefill_module(
        "fa3",
        f"single_prefill_output_transform_{str(dtype).split('.')[-1]}",
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        128,  # head_dim_qk
        128,  # head_dim_vo
        *_OUTPUT_TRANSFORM_PARAMS,
        "CorrectedAttention",
        corrected_attention_sm90_decl,
    ).build_and_load()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_sm90_output_transform(dtype, causal):
    """fa3 single prefill through a variant whose REGISTER_OUTPUT_TRANSFORM adds
    corr[q, h] * hvec[h, :] to the normalized output. The output must match an fp32 reference
    and the LSE must be untouched, so it is compared with the built-in fa3 kernel."""
    _skip_unless_sm90a()
    torch.manual_seed(42)
    # 333 rows = two full 128-row tiles plus a partial one; 8 query heads over 2 kv heads.
    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = 333, 1027, 8, 2, 128
    f = functools.partial(
        single_prefill_with_kv_cache_with_jit_module,
        _sm90_output_transform_module(dtype),
    )

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    corr, hvec = _random_correction(qo_len, num_qo_heads, head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)
    mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value

    o, lse = f(
        q,
        k,
        v,
        corr,
        hvec,
        sm_scale,
        corr.stride(0),
        mask_mode=mask_mode,
        return_lse=True,
    )

    o_ref, _ = _attention_reference(q, k, v, causal, sm_scale)
    o_ref = o_ref + corr[:, :, None] * hvec[None, :, :]
    _, lse_builtin = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa3", return_lse=True
    )
    # Values are O(1); the error budget is dominated by the 16-bit output rounding.
    tol = 2e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(o.float(), o_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(lse, lse_builtin, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_sm90_output_transform(causal):
    """fa3 batch prefill, ragged and paged, through the same variant. The batch mixes a partial
    128-row tile, a request with an empty KV (its rows come from the epilogue's zero path, so
    the output must be exactly the correction term and the LSE -inf), a decode-like request
    and a request spanning two tiles. corr is indexed by the global query row."""
    _skip_unless_sm90a()
    torch.manual_seed(42)
    dtype = torch.float16
    num_qo_heads, num_kv_heads, head_dim, page_size = 8, 2, 128, 16
    qo_lens = [77, 300, 5, 130]
    kv_lens = [77, 0, 640, 130]
    nnz_qo, nnz_kv = sum(qo_lens), sum(kv_lens)
    jit_args = (
        "batch_prefill_output_transform",  # uri
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        torch.int32,  # idtype
        head_dim,  # hidden_dim_qk
        head_dim,  # hidden_dim_vo
        *_OUTPUT_TRANSFORM_PARAMS,
        "CorrectedAttention",
        corrected_attention_sm90_decl,
    )

    q = torch.randn(nnz_qo, num_qo_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(nnz_kv, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(nnz_kv, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    corr, hvec = _random_correction(nnz_qo, num_qo_heads, head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qo_indptr = torch.tensor([0] + qo_lens).cumsum(0).int()
    kv_indptr = torch.tensor([0] + kv_lens).cumsum(0).int()

    o_ref = torch.empty(
        nnz_qo, num_qo_heads, head_dim, dtype=torch.float32, device="cuda"
    )
    lse_ref = torch.empty(nnz_qo, num_qo_heads, dtype=torch.float32, device="cuda")
    for i in range(len(qo_lens)):
        q0, q1 = qo_indptr[i].item(), qo_indptr[i + 1].item()
        k0, k1 = kv_indptr[i].item(), kv_indptr[i + 1].item()
        o_ref[q0:q1], lse_ref[q0:q1] = _attention_reference(
            q[q0:q1], k[k0:k1], v[k0:k1], causal, sm_scale
        )
    o_ref += corr[:, :, None] * hvec[None, :, :]

    ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        backend="fa3",
        jit_args=jit_args,
    )
    ragged.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = ragged.run(q, k, v, corr, hvec, sm_scale, corr.stride(0), return_lse=True)
    torch.testing.assert_close(o.float(), o_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)

    # The same KV in 16-token pages; the empty request owns no page.
    num_pages = [(n + page_size - 1) // page_size for n in kv_lens]
    page_indptr = torch.tensor([0] + num_pages).cumsum(0).int()
    last_page_len = torch.tensor(
        [
            n - (p - 1) * page_size if p > 0 else 0
            for n, p in zip(kv_lens, num_pages, strict=True)
        ],
        dtype=torch.int32,
    )
    k_paged = torch.zeros(
        sum(num_pages), page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v_paged = torch.zeros_like(k_paged)
    for i, n in enumerate(kv_lens):
        k0, t0 = kv_indptr[i].item(), page_indptr[i].item() * page_size
        k_paged.view(-1, num_kv_heads, head_dim)[t0 : t0 + n] = k[k0 : k0 + n]
        v_paged.view(-1, num_kv_heads, head_dim)[t0 : t0 + n] = v[k0 : k0 + n]
    paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        backend="fa3",
        jit_args=jit_args,
    )
    paged.plan(
        qo_indptr,
        page_indptr,
        torch.arange(sum(num_pages), dtype=torch.int32),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
    )
    o_paged, lse_paged = paged.run(
        q, (k_paged, v_paged), corr, hvec, sm_scale, corr.stride(0), return_lse=True
    )
    torch.testing.assert_close(o_paged.float(), o_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse_paged, lse_ref, rtol=1e-2, atol=1e-2)

    # Rows of the empty request: exactly the correction term (up to fp16 rounding) and -inf.
    q0, q1 = qo_indptr[1].item(), qo_indptr[2].item()
    expected = corr[q0:q1, :, None] * hvec[None, :, :]
    for out, out_lse in ((o, lse), (o_paged, lse_paged)):
        torch.testing.assert_close(out[q0:q1].float(), expected, rtol=2e-3, atol=2e-3)
        assert torch.isneginf(out_lse[q0:q1]).all()


@pytest.mark.parametrize("causal", [False, True])
def test_output_transform_fa2_fa3_parity(causal):
    """The same per-(query, head) correction written against the FA2 hook (which also
    normalizes) and the fa3 hook (post-normalization, hvec set to ones) must agree with each
    other and with the fp32 reference."""
    _skip_unless_sm90a()
    torch.manual_seed(42)
    dtype = torch.float16
    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = 333, 1027, 8, 2, 128
    fa2_module = gen_customize_single_prefill_module(
        "fa2",
        "single_prefill_output_transform_fa2",
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        head_dim,  # head_dim_qk
        head_dim,  # head_dim_vo
        ["corr"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale", "corr_stride"],  # additional_scalar_names
        ["double", "int64_t"],  # additional_scalar_dtypes
        "CorrectedAttention",
        corrected_attention_sm80_decl,
    ).build_and_load()
    fa3_module = _sm90_output_transform_module(dtype)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    corr, _ = _random_correction(qo_len, num_qo_heads, head_dim)
    ones = torch.ones(num_qo_heads, head_dim, dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(head_dim)
    mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value

    o_fa2 = single_prefill_with_kv_cache_with_jit_module(
        fa2_module, q, k, v, corr, sm_scale, corr.stride(0), mask_mode=mask_mode
    )
    o_fa3 = single_prefill_with_kv_cache_with_jit_module(
        fa3_module, q, k, v, corr, ones, sm_scale, corr.stride(0), mask_mode=mask_mode
    )

    o_ref, _ = _attention_reference(q, k, v, causal, sm_scale)
    o_ref = o_ref + corr[:, :, None]
    torch.testing.assert_close(o_fa2.float(), o_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(o_fa3.float(), o_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(o_fa2, o_fa3, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_sm90_fp8_output_transform(causal):
    """The hook also runs in the fa3 FP8 epilogue, which stores the accumulator through its own
    column handling. With the built-in FP8 variant extended by the hook, the transformed output
    minus the built-in FP8 output must be exactly the correction term."""
    _skip_unless_sm90a()
    torch.manual_seed(42)
    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = 333, 1027, 8, 2, 128
    fp8, out_dtype = torch.float8_e4m3fn, torch.float16
    jit_module = gen_customize_single_prefill_module(
        "fa3",
        "single_prefill_output_transform_fp8",
        fp8,  # dtype_q
        fp8,  # dtype_kv
        out_dtype,  # dtype_o
        head_dim,  # head_dim_qk
        head_dim,  # head_dim_vo
        *_OUTPUT_TRANSFORM_PARAMS,
        "CorrectedFP8Attention",
        corrected_attention_fp8_sm90_decl,
        fp8_enabled=True,  # select the FP8 kernel template
    ).build_and_load()

    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda").to(fp8)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda").to(fp8)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda").to(fp8)
    corr, hvec = _random_correction(qo_len, num_qo_heads, head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)
    mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value

    # single_prefill_with_kv_cache_with_jit_module allocates the output in q's dtype, so call
    # the module directly with a 16-bit output buffer.
    o = torch.empty(qo_len, num_qo_heads, head_dim, dtype=out_dtype, device="cuda")
    tmp = torch.empty(SINGLE_KERNEL_TMP_SIZE, dtype=torch.uint8, device="cuda")
    jit_module.run(
        q,
        k,
        v,
        tmp,
        o,
        None,
        mask_mode,
        TensorLayout.NHD.value,
        -1,
        corr,
        hvec,
        sm_scale,
        corr.stride(0),
    )
    o_builtin = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa3", o_dtype=out_dtype
    )
    torch.testing.assert_close(
        o.float() - o_builtin.float(),
        corr[:, :, None] * hvec[None, :, :],
        rtol=2e-2,
        atol=2e-2,
    )


if __name__ == "__main__":
    test_single_decode_mask()
    test_flash_sigmoid()
    test_dump_logits()
    test_batch_decode_flash_sigmoid(False)
    test_batch_decode_flash_sigmoid(True)
    test_batch_prefill_flash_sigmoid()
    test_batch_prefill_sm90_flash_sigmoid()
    test_batch_prefill_jit_wellknown_mask_buffers()
    test_batch_decode_jit_wellknown_alibi_buffer(False)
    test_batch_decode_jit_wellknown_alibi_buffer(True)
    test_single_prefill_sm90_output_transform(torch.float16, False)
    test_batch_prefill_sm90_output_transform(False)
    test_output_transform_fa2_fa3_parity(False)
    test_single_prefill_sm90_fp8_output_transform(False)
