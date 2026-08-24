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
from flashinfer.utils import MaskMode, get_compute_capability, is_sm90a_supported


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
    if get_compute_capability(torch.device("cuda")) < (8, 0):
        pytest.skip("variant_owns_mask FA2 JIT test is validated on SM80+.")
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
    if get_compute_capability(torch.device("cuda")) < (8, 0):
        pytest.skip("variant_owns_mask FA2 JIT test is validated on SM80+.")
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
