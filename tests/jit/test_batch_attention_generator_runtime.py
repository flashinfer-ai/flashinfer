"""Exercise public/custom persistent modules through their direct plan/run ABI."""

import math

import pytest
import torch

from flashinfer.jit.attention.modules import (
    gen_batch_attention_module,
    gen_customize_batch_attention_module,
)
from flashinfer.utils import MaskMode, TensorLayout, get_compute_capability
from tests.test_helpers.paged_kv import make_paged_kv_cache_pair


_VARIANT_DECL = """
struct TestScaledAttention : AttentionVariantBase {
  float sm_scale_log2;
  static constexpr bool use_logits_soft_cap = false;
  template <typename Params>
  __device__ __host__ TestScaledAttention(
      const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    sm_scale_log2 = params.sm_scale * float(params.test_logits_scale) * math::log2e;
  }
};
"""


@pytest.fixture(scope="module", params=["public", "custom"])
def batch_attention_module(request):
    if request.param == "public":
        spec = gen_batch_attention_module(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.int32,
            128,
            128,
            0,
            False,
            False,
        )
        assert spec.name == (
            "batch_attention_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_"
            "dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_"
            "use_logits_soft_cap_false_use_profiler_false"
        )
    else:
        uri = "test_batch_attention_custom_scaled_bf16_d128"
        spec = gen_customize_batch_attention_module(
            uri,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.int32,
            128,
            128,
            ["maybe_k_cache_sf", "maybe_v_cache_sf"],
            ["uint8_t", "uint8_t"],
            ["test_logits_scale"],
            ["double"],
            "TestScaledAttention",
            _VARIANT_DECL,
        )
        assert spec.name == uri
    return request.param, spec.build_and_load()


def _reference(q, k, v, qptr, kptr, indices, lengths, causal, scale):
    """CPU FP32 paged attention with GQA, bottom-right mask and base-2 LSE."""
    q, k, v = (tensor.detach().float().cpu() for tensor in (q, k, v))
    output, lse = [], []
    for request, length in enumerate(lengths):
        pages = indices[kptr[request] : kptr[request + 1]]
        kr = k[pages].reshape(-1, 2, 128)[:length].repeat_interleave(4, dim=1)
        vr = v[pages].reshape(-1, 2, 128)[:length].repeat_interleave(4, dim=1)
        qr = q[qptr[request] : qptr[request + 1]]
        scores = torch.einsum("qhd,khd->hqk", qr, kr) * (scale / math.sqrt(128))
        if causal:
            allowed = torch.arange(length)[None, :] <= (
                torch.arange(qr.shape[0])[:, None] + length - qr.shape[0]
            )
            scores.masked_fill_(~allowed[None], -torch.inf)
        output.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vr))
        lse.append(torch.logsumexp(scores, -1).transpose(0, 1) / math.log(2))
    return torch.cat(output), torch.cat(lse)


@pytest.mark.parametrize("causal", [False, True])
def test_batch_attention_generator_direct_plan_run(batch_attention_module, causal):
    kind, module = batch_attention_module
    torch.manual_seed(20260917)
    lengths, qptr, kptr = [35, 147], [0, 1, 130], [0, 3, 13]
    indices = list(reversed(list(range(1, 13)) + [0]))
    q = torch.randn(130, 8, 128, dtype=torch.bfloat16, device="cuda") * 0.2
    dense_k = torch.randn(13, 16, 2, 128, dtype=q.dtype, device=q.device)
    dense_v = torch.randn_like(dense_k)
    caches = [
        make_paged_kv_cache_pair(dense_k, dense_v, "NHD", mode, 8)
        for mode in ("padded", "head")
    ]
    scales = (0.5, 1.25) if kind == "custom" else (1.0,)
    expected = {
        scale: _reference(
            q, dense_k, dense_v, qptr, kptr, indices, lengths, causal, scale
        )
        for scale in scales
    }
    if kind == "custom":
        # An implementation that ignores the additional scalar must fail.
        assert not torch.allclose(
            expected[0.5][0], expected[1.25][0], rtol=1e-2, atol=1e-2
        )
        unscaled = _reference(
            q, dense_k, dense_v, qptr, kptr, indices, lengths, causal, 1.0
        )
        assert all(
            not torch.allclose(expected[scale][0], unscaled[0], rtol=1e-2, atol=1e-2)
            for scale in scales
        )

    float_workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    int_workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    pinned_workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    plan = module.plan(
        float_workspace,
        int_workspace,
        pinned_workspace,
        torch.tensor(qptr, dtype=torch.int32),
        torch.tensor(kptr, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        2,
        8,
        2,
        128,
        causal,
    )
    indices_tensor = torch.tensor(indices, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    for k, v in caches:
        for scale in scales:
            out.fill_(torch.nan)
            lse.fill_(torch.nan)
            # This fixture exceeds the SM120/121 cooperative-grid limit.
            # Avoid leaving a CUDA launch error for the next test.
            if q.shape[-1] == 128 and get_compute_capability(q.device)[0] == 12:
                pytest.xfail(
                    "SM120/121 persistent BatchAttention cooperative-launch limit"
                )
            module.run(
                float_workspace,
                int_workspace,
                plan,
                q,
                k,
                v,
                indices_tensor,
                out,
                lse,
                MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value,
                TensorLayout.NHD.value,
                8,
                2,
                16,
                1.0,
                1.0 / math.sqrt(128),
                0.0,
                None,
                None,
                *([scale] if kind == "custom" else []),
            )
            for actual, reference in zip((out, lse), expected[scale], strict=True):
                torch.testing.assert_close(
                    actual.float().cpu(), reference, rtol=1e-2, atol=1e-2
                )
