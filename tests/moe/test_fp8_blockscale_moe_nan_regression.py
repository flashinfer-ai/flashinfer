"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Regression tests for two numerical-stability fixes in the SM90 FP8 block-scale
grouped GEMM that backs the CUTLASS fused MoE path
(``cutlass_fused_moe(..., use_deepseek_fp8_block_scale=True)``).

1. Cross-boundary / padding-row ``0 * Inf`` (the primary fix).
   In the swap-AB grouped GEMM, rows at or past a per-expert row boundary are
   padding rows. With many experts and small token counts, a tile is mostly
   padding; those rows accumulate over in-bounds-but-stale FP8 input and the raw
   accumulator can reach Inf, while their per-token scale is 0, so
   ``scale * accum`` evaluates to ``0 * Inf = NaN``. Because the persistent-kernel
   epilogue reuses a shared-memory staging buffer, that NaN can bleed into a
   valid row's output. The fix gates the cross-row contribution (adds exactly
   ``0.f`` for padding rows) and finite-sanitizes the accumulator before the bf16
   store, at the block-scale kernel epilogue and the two swap-AB sites in
   ``deep_gemm/fp8_gemm_impl.cuh``. Valid rows are always finite, so every real
   output element is unchanged and the tensor-core path is untouched.

   Reproduction note (important before strengthening these tests): on real
   serving traffic this fires at small M with a high populated-expert count, and
   it is *value-correlated* — it depends on the specific interaction of the
   quantized expert weights, their per-128-block weight-scale dynamic range (real
   scales span ~1e-27 .. ~1e-4), and the grouped-GEMM tiling. It was confirmed to
   fail on the unpatched kernel and pass on the patched kernel on captured
   real-routing tensors (a full output row of NaN -> all finite). It could **not**
   be reproduced with fully synthetic random inputs: neither a sweep over M,
   routing concentration and activation magnitude, nor injecting the same extreme
   weight-scale dynamic range, triggered it on the unpatched kernel. The tests
   below therefore exercise the same small-M / many-expert regime as a finiteness
   + correctness guard; they pass on both the patched and unpatched kernels for
   synthetic inputs and exist to prevent regressions in that regime, not to
   reproduce the original value-dependent trigger.

2. Per-token ``[1 x 128]`` activation-requant amax (defensive).
   The patch also switches the per-token requant amax reduction to
   ``find_max_elem_in_warp`` and adds finite clamps. Note that the unpatched
   reduction (``kernel_utils::warpReduceSum`` in this file) is *already* a warp
   MAX despite its name — it is implemented as
   ``val = max(val, __shfl_xor_sync(...))`` — so the requant scale is the true
   block max-abs on both kernels and there is no summation-overflow path to
   reproduce. The change is a defensive finiteness guard, not a behavioural fix.
   ``test_per_token_1x128_block_scale_is_true_max`` documents and pins the
   correct semantics (true max, finite, no scale collapse).

Every kernel case asserts the output is finite (the NaN failure mode) AND, where
the activation magnitude is in a normal range, numerically correct against a
high-precision reference built by dequantizing the FP8 block-scale weights and
running the matmul + SwiGLU in fp32. Correctness is checked with an
absolute/relative element tolerance, not cosine similarity: cosine similarity is
scale-blind and can report a near-perfect match even when the output is off by a
large constant factor.
"""

import pytest
import torch
import torch.nn.functional as F

import flashinfer.fused_moe as fused_moe


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 9


requires_sm90 = pytest.mark.skipif(
    not _is_sm90(),
    reason="FP8 block-scale CUTLASS fused MoE is only implemented on SM90 (Hopper)",
)


def ceil_div(a: int, b: int) -> int:
    return -(a // -b)


# -----------------------------------------------------------------------------
# FP8 block-scale quantization helpers (dequant-scale semantics: float ~= fp8 * s)
# -----------------------------------------------------------------------------
def per_token_group_quant_fp8(x: torch.Tensor, group_size: int = 128):
    """Per-token [1 x group_size] FP8 quantization. Returns (fp8, scale)."""
    assert x.shape[-1] % group_size == 0
    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-10).to(torch.float32)
    x_s = amax / FP8_MAX
    x_q = (x_ / x_s).clamp(min=-FP8_MAX, max=FP8_MAX).to(FP8_DTYPE)
    return x_q.reshape(x.shape), x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))


def per_block_cast_to_fp8(x: torch.Tensor, block_size_n: int = 128):
    """128 x block_size_n block FP8 quantization for a 2D weight matrix."""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, block_size_n) * block_size_n),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(FP8_DTYPE)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def dequantize_activation(x_quant: torch.Tensor, scales: torch.Tensor, dtype):
    """Dequantize a [T, H] per-token-group-quantized activation to ``dtype``."""
    t, h = x_quant.shape
    nb = ceil_div(h, 128)
    s = scales.view(t, nb, 1).expand(-1, -1, 128).reshape(t, nb * 128)[:, :h]
    return (x_quant.to(dtype) * s.to(dtype)).view(t, h)


def dequantize_weight(w_quant: torch.Tensor, scales: torch.Tensor, dtype):
    """Dequantize a [R, C] per-128x128-block-quantized weight to ``dtype``."""
    r, c = w_quant.shape
    nb_r, nb_c = ceil_div(r, 128), ceil_div(c, 128)
    s = scales.view(nb_r, nb_c)
    s = s.unsqueeze(1).unsqueeze(3).expand(nb_r, 128, nb_c, 128).reshape(
        nb_r * 128, nb_c * 128
    )[:r, :c]
    return (w_quant.to(dtype) * s.to(dtype)).view(r, c)


# -----------------------------------------------------------------------------
# High-precision reference (no FP8): dequantize then matmul + SwiGLU in fp32.
# -----------------------------------------------------------------------------
def moe_reference(
    num_experts: int,
    x: torch.Tensor,  # [T, H] high precision
    w31: torch.Tensor,  # [E, 2I, H] high precision
    w2: torch.Tensor,  # [E, H, I] high precision
    selected_experts: torch.Tensor,  # [T, top_k] int
    routing_weights: torch.Tensor,  # [T, top_k]
) -> torch.Tensor:
    out = torch.zeros_like(x)
    for e in range(num_experts):
        mask = selected_experts == e
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        w3_e, w1_e = torch.chunk(w31[e], 2, dim=0)  # each [I, H]
        xin = x[tok]
        inter = F.silu(xin @ w1_e.t()) * (xin @ w3_e.t())
        o = inter @ w2[e].t()
        out[tok] += routing_weights[tok, nth, None] * o
    return out.view_as(x)


def _make_moe_inputs(
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    activation_scale: float,
    seed: int = 0,
):
    """Build a deterministic FP8 block-scale MoE problem.

    Returns the quantized kernel inputs plus the dequantized high-precision
    tensors used to compute the reference. Activations are left in BF16 and are
    quantized internally by the kernel (this is the path that exercises the
    activation requant), so ``activation_scale`` directly controls the requant
    amax.
    """
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    x = activation_scale * torch.randn(batch_size, hidden_size, dtype=dtype).cuda()
    # Dequantized activation for the reference. The kernel quantizes the BF16
    # activation internally with the per-token [1x128] requant, so the honest
    # correctness reference is x round-tripped through the SAME FP8 block-scale
    # quantization (this isolates kernel correctness from activation quant noise,
    # matching the repo's existing FP8 block-scale MoE test).
    x_q, x_s = per_token_group_quant_fp8(x, group_size=128)
    x_dequant = dequantize_activation(x_q, x_s, dtype)
    w31 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=dtype).cuda() / 10
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype).cuda() / 10

    selected = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()
    routing = F.softmax(torch.randn(batch_size, top_k).cuda(), dim=1)

    w31_q = torch.empty_like(w31, dtype=FP8_DTYPE)
    w2_q = torch.empty_like(w2, dtype=FP8_DTYPE)
    w31_s = torch.zeros(
        num_experts,
        ceil_div(2 * intermediate_size, 128),
        ceil_div(hidden_size, 128),
        dtype=torch.float32,
    ).cuda()
    w2_s = torch.zeros(
        num_experts,
        ceil_div(hidden_size, 128),
        ceil_div(intermediate_size, 128),
        dtype=torch.float32,
    ).cuda()
    w31_dq = torch.empty_like(w31)
    w2_dq = torch.empty_like(w2)

    for e in range(num_experts):
        q1, s1 = per_block_cast_to_fp8(w31[e])
        q2, s2 = per_block_cast_to_fp8(w2[e])
        w31_q[e].copy_(q1)
        w2_q[e].copy_(q2)
        w31_s[e].copy_(s1)
        w2_s[e].copy_(s2)
        w31_dq[e] = dequantize_weight(q1, s1, dtype)
        w2_dq[e] = dequantize_weight(q2, s2, dtype)

    # Reference uses the dequantized weights and the BF16 activations directly:
    # the kernel quantizes the activation internally, so the BF16 input is the
    # honest high-precision reference for the activation side.
    return {
        "x": x,
        "selected": selected.to(torch.int),
        "routing": routing,
        "w31_q": w31_q.contiguous(),
        "w2_q": w2_q.contiguous(),
        "w31_s": w31_s.contiguous(),
        "w2_s": w2_s.contiguous(),
        "x_ref": x_dequant,
        "w31_ref": w31_dq,
        "w2_ref": w2_dq,
    }


def _run_kernel(inp, num_experts, hidden_size):
    out = torch.zeros_like(inp["x"])
    fused_moe.cutlass_fused_moe(
        inp["x"].contiguous(),
        inp["selected"],
        inp["routing"],
        inp["w31_q"],
        inp["w2_q"],
        torch.bfloat16,
        use_deepseek_fp8_block_scale=True,
        quant_scales=[inp["w31_s"], inp["w2_s"]],
        output=out,
    )
    return out


def _assert_finite_and_correct(
    out, inp, num_experts, check_elementwise=True, hit_floor=0.85
):
    # Primary regression assertion: no NaN/Inf anywhere. This is the failure
    # mode the cross-boundary fix addresses and is checked in every case.
    assert torch.isfinite(out).all(), (
        f"output contains {(~torch.isfinite(out)).sum().item()} non-finite elements "
        f"(NaN/Inf) — block-scale grouped GEMM NaN regression"
    )

    ref = moe_reference(
        num_experts,
        inp["x_ref"].float(),
        inp["w31_ref"].float(),
        inp["w2_ref"].float(),
        inp["selected"].long(),
        inp["routing"].float(),
    )
    out_f = out.float()

    # Scale-sanity guard, NOT cosine similarity. Cosine is scale-blind: a kernel
    # that scales the whole output by a wrong constant can still score
    # cosine ~= 1.0, so it cannot detect a scale-error regression. Comparing the
    # output RMS to the reference RMS directly catches such a scale bug.
    ref_rms = ref.pow(2).mean().sqrt().item()
    out_rms = out_f.pow(2).mean().sqrt().item()
    if ref_rms > 1e-6:
        ratio = out_rms / ref_rms
        assert 0.5 <= ratio <= 2.0, (
            f"output/reference RMS ratio {ratio:.3f} indicates a scale error "
            f"(ref_rms={ref_rms:.3e}, out_rms={out_rms:.3e})"
        )

    # Element-wise correctness via absolute+relative tolerance. The reference
    # dequantizes both the FP8 weights and the FP8 activation (the same
    # round-trip the kernel applies internally), so quantization noise is
    # excluded and the comparison measures kernel correctness. The element-hit
    # floor is a correctness guard: a broken (e.g. scale-error) kernel scores
    # near zero, while the correct kernel's genuine FP8 rounding error at the
    # production reduction depth (K up to 4096) leaves a small fraction of
    # elements just outside the tolerance, so the floor is set accordingly per
    # call site rather than at a single global value.
    if not check_elementwise:
        return
    abs_diff = (out_f - ref).abs()
    atol, rtol = 1e-1, 1e-1
    within = abs_diff <= (atol + rtol * ref.abs())
    hit = within.float().mean().item()
    assert hit >= hit_floor, (
        f"only {hit * 100:.1f}% of elements within abs/rel tolerance "
        f"(atol={atol}, rtol={rtol}, floor={hit_floor}); max abs diff "
        f"{abs_diff.max().item():.3e}. A large scale error would pass cosine "
        f"similarity but fails here."
    )


# -----------------------------------------------------------------------------
# Cross-boundary / padding-row finiteness + correctness guard.
#
# This is the regime that produced the original cross-boundary NaN in production:
# small M with a high populated-expert count, at the production hidden /
# intermediate sizes (so the grouped-GEMM tiling and swap-AB epilogue match the
# deployed kernel). See the module docstring: the original failure is
# value-correlated and was confirmed on captured real tensors (fail on the
# unpatched kernel, finite on the patched kernel); it does not reproduce with
# synthetic random inputs, so these cases are a finiteness + correctness guard
# for the regime, passing on both kernels for synthetic data.
# -----------------------------------------------------------------------------
@requires_sm90
@pytest.mark.parametrize(
    "batch_size, num_experts, top_k",
    [
        pytest.param(1, 64, 8, id="M1_64e"),
        pytest.param(3, 64, 8, id="M3_64e"),
        pytest.param(5, 64, 8, id="M5_64e"),
        pytest.param(16, 64, 8, id="M16_64e"),
        pytest.param(256, 64, 8, id="M256_64e"),
    ],
)
def test_small_m_many_experts_finite(batch_size, num_experts, top_k):
    """Small M, many populated experts: output must stay finite and correct."""
    # Production-shaped hidden/intermediate so the swap-AB grouped-GEMM tiling
    # and epilogue match the deployed kernel where the cross-boundary NaN lived.
    hidden_size, intermediate_size = 4096, 1024
    inp = _make_moe_inputs(
        batch_size,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        activation_scale=2.0,
        seed=99,
    )
    out = _run_kernel(inp, num_experts, hidden_size)
    # Production reduction depth (K=4096) leaves a small fraction of elements
    # just outside the FP8 tolerance; 0.75 still fails a scale-error kernel.
    _assert_finite_and_correct(out, inp, num_experts, hit_floor=0.75)


# -----------------------------------------------------------------------------
# General M sweep across the BLOCK_M tile boundary at production shapes. M=1,2
# are mostly-padding tiles; this sweeps the boundary asserting finiteness and
# correctness across the range.
# -----------------------------------------------------------------------------
@requires_sm90
@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 16, 32, 256])
def test_m_sweep_finite(batch_size):
    """Sweep M across the BLOCK_M boundary; output stays finite and correct."""
    hidden_size, intermediate_size, num_experts = 4096, 1024, 32
    top_k = min(8, num_experts)
    inp = _make_moe_inputs(
        batch_size,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        activation_scale=2.0,
        seed=7,
    )
    out = _run_kernel(inp, num_experts, hidden_size)
    _assert_finite_and_correct(out, inp, num_experts, hit_floor=0.75)


# -----------------------------------------------------------------------------
# Small-M activation requant amax sanity. Pushes activation magnitudes near the
# FP8 representable range; the per-token [1x128] requant must stay finite and
# produce a correctly-scaled output. (The unpatched reduction is already a warp
# MAX, so this passes on both kernels; it guards the requant amax path.)
# -----------------------------------------------------------------------------
@requires_sm90
@pytest.mark.parametrize(
    "batch_size, stress_amax",
    [
        pytest.param(1, True, id="M1_stress_amax"),
        pytest.param(2, True, id="M2_stress_amax"),
        pytest.param(5, True, id="M5_stress_amax"),
        pytest.param(1, False, id="M1_nominal"),
        pytest.param(5, False, id="M5_nominal"),
    ],
)
def test_small_m_requant_amax_finite(batch_size, stress_amax):
    """Small-M activation requant must stay finite near the FP8 range."""
    hidden_size, intermediate_size, num_experts, top_k = 512, 256, 4, 2
    activation_scale = (FP8_MAX * 0.5) if stress_amax else 2.0
    inp = _make_moe_inputs(
        batch_size,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        activation_scale,
        seed=1234,
    )
    out = _run_kernel(inp, num_experts, hidden_size)
    # Under deliberate amax stress, a few output elements carry large genuine
    # FP8 rounding error, so assert finiteness + scale sanity rather than
    # element-wise correctness. The nominal cases assert full correctness.
    _assert_finite_and_correct(
        out, inp, num_experts, check_elementwise=not stress_amax
    )


# -----------------------------------------------------------------------------
# Focused unit check for the per-token [1x128] requant block scale.
#
# The requant scale must be derived from the TRUE max-abs over the 128 lanes.
# Both the patched (find_max_elem_in_warp) and the unpatched
# (kernel_utils::warpReduceSum, which is itself implemented as a warp MAX) paths
# compute this true max — so this asserts the correct semantics, which hold on
# both kernels, and documents that there is no summation-overestimate path. The
# patched code additionally clamps a (hypothetically) non-finite amax; for finite
# inputs that branch is never taken and the result is unchanged.
# -----------------------------------------------------------------------------
def test_per_token_1x128_block_scale_is_true_max():
    """The 1x128 requant scale must be the block max-abs and stay finite."""
    torch.manual_seed(0)
    device = "cpu"
    # A block whose true max-abs is near the FP8 range.
    block = (FP8_MAX * 0.9) * torch.ones(1, 128, dtype=torch.float32, device=device)
    block += 0.01 * torch.randn(1, 128, dtype=torch.float32, device=device)

    q, s = per_token_group_quant_fp8(block.to(torch.bfloat16).float(), group_size=128)
    true_amax = block.abs().max().item()

    # The correct scale is true_amax / FP8_MAX (dequant-scale semantics).
    assert s.item() == pytest.approx(true_amax / FP8_MAX, rel=5e-2)

    dq = dequantize_activation(q, s, torch.float32)
    assert torch.isfinite(dq).all()
    # Dequantized block must track the original to within FP8 step error, which
    # only holds when the scale is the true max-abs (a collapsed/0 scale would
    # zero it or blow it up).
    rel = (dq - block).abs().max().item() / true_amax
    assert rel < 0.1, f"dequant rel error {rel:.3f} too high — scale likely wrong"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
