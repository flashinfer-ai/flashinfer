# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VC-Attention (bf16 or E4M3 Q/K with E4M3 V tile residuals) for the PrimTS context kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0a0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0a0",
)

from flashinfer.attention.prims_ts import (
    BatchPrefillTSWrapper,
    VCAttentionConfig,
    VCAttentionPreprocessor,
    batch_prefill,
    vc_attention as vca,
)

from tests.attention.test_attention_ts_context import _REQUIRES_CONTEXT_GPU

_FP8 = torch.float8_e4m3fn
_HEAD_DIM = 128


@pytest.fixture(autouse=True)
def _exact_fp32_matmul():
    """The torch reference needs true fp32 matmuls; NGC containers export
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which would turn them into TF32."""
    prev_prec = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(prev_prec)
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@dataclass(frozen=True)
class _VCCase:
    """One VC problem: geometry and launch path."""

    name: str
    batch_size: int
    seq_len: int
    num_heads: int
    one_shot: bool = False
    # bf16 selects VC-Attention-QK16, E4M3 VC-Attention-QK8.
    qk_dtype: torch.dtype = torch.bfloat16
    q_block_size: int = 128


_CASES = (
    _VCCase("single-tile", 1, 512, 1),
    _VCCase("partial-tile", 1, 2000, 2),
    # Odd Q-tile count (cluster padding), partial last K/V tile, two batches.
    _VCCase("two-batches", 2, 2120, 2),
    _VCCase("one-shot", 1, 2000, 2, one_shot=True),
    _VCCase("qk8", 2, 2048, 2, qk_dtype=_FP8),
    _VCCase("qk8-partial-tile", 1, 2000, 2, qk_dtype=_FP8),
    # Q scale blocks narrower than the Q tile.
    _VCCase("qk8-q-block-64", 1, 1024, 1, qk_dtype=_FP8, q_block_size=64),
    _VCCase("qk8-one-shot", 1, 2000, 2, one_shot=True, qk_dtype=_FP8),
)


def _structured_values(shape, device):
    """Values with a strong per-token component so tile means matter."""
    b, s, h, d = shape
    return torch.randn(shape, device=device) * 0.3 + torch.randn(
        b, s, h, 1, device=device
    ) * torch.randn(b, 1, h, d, device=device)


def _random_inputs(batch_size, seq_len, num_heads, *, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn(batch_size, seq_len, num_heads, _HEAD_DIM, device=device).to(
        torch.bfloat16
    )
    k = torch.randn_like(q)
    v = _structured_values(q.shape, device).to(torch.bfloat16)
    return q, k, v


def _plan(
    wrapper,
    *,
    batch_size,
    seq_len,
    num_heads,
    sm_scale=None,
    qk_dtype=torch.bfloat16,
    q_block_size=128,
    **overrides,
):
    arguments = dict(
        device=torch.device("cuda"),
        batch_size=batch_size,
        max_seq_len_q=seq_len,
        max_kv_len=seq_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=_HEAD_DIM,
        q_dtype=qk_dtype,
        k_dtype=qk_dtype,
        v_dtype=_FP8,
        out_dtype=torch.bfloat16,
        sm_scale=sm_scale,
        vc_config=VCAttentionConfig(q_block_size=q_block_size),
    )
    wrapper.plan(**{**arguments, **overrides})


def _quantize(case: _VCCase, q, k, v):
    if case.qk_dtype == _FP8:
        return vca.vc_quantize_fp8(q, k, v, q_block_size=case.q_block_size)
    return vca.vc_quantize(k, v)


def _run_case(case: _VCCase):
    q, k, v = _random_inputs(case.batch_size, case.seq_len, case.num_heads)
    ops = _quantize(case, q, k, v)
    q_run = q if ops.q is None else ops.q
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    if case.one_shot:
        out = batch_prefill(
            q_run,
            ops.k,
            ops.v,
            sm_scale=sm_scale,
            out_dtype=torch.bfloat16,
            vc=ops.params,
            vc_config=VCAttentionConfig(q_block_size=case.q_block_size),
        )
    else:
        wrapper = BatchPrefillTSWrapper()
        _plan(
            wrapper,
            batch_size=case.batch_size,
            seq_len=case.seq_len,
            num_heads=case.num_heads,
            sm_scale=sm_scale,
            qk_dtype=case.qk_dtype,
            q_block_size=case.q_block_size,
        )
        out = wrapper.run(q_run, ops.k, ops.v, vc=ops.params)
    reference = vca.vc_reference(q, ops, sm_scale=sm_scale)
    exact = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    ).transpose(1, 2)
    return out.float(), reference, exact


def _relative_error(out: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(out - reference) / torch.linalg.vector_norm(reference)
    )


@_REQUIRES_CONTEXT_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_vc_attention_matches_dequantized_reference(case: _VCCase):
    out, reference, exact = _run_case(case)
    assert torch.isfinite(out).all()
    # ExpCast-FP8 codes P with Mitchell's log2 approximation (<= 7.5% per
    # element, bounded total variation), plus bf16 output rounding.
    assert _relative_error(out, reference) < 6e-2
    torch.testing.assert_close(out, reference, rtol=1e-1, atol=1e-1)
    # The recipe itself stays close to unquantized attention.
    assert _relative_error(out, exact) < 1e-1


@_REQUIRES_CONTEXT_GPU
@pytest.mark.arch_blackwell
def test_vc_mean_step():
    """A constant tile mean shifts every output by exactly that constant. With zero
    tile means, demean=False (mean UMMA skipped) equals demean=True bit for bit and
    differs from a run whose means are non-zero."""
    q, k, v = _random_inputs(1, 1024, 2)
    ops = vca.vc_quantize(k, v)
    wrapper = BatchPrefillTSWrapper()
    _plan(wrapper, batch_size=1, seq_len=1024, num_heads=2)
    outs = {}
    for name, value in (("zero", 0.0), ("one", 1.0)):
        mean = torch.full_like(ops.mean, value)
        mu = vca.pack_vc_tile_means(mean, ops.v_scale)
        outs[name] = wrapper.run(q, ops.k, ops.v, vc=replace(ops.params, tile_means=mu))
    torch.testing.assert_close(
        outs["one"].float() - outs["zero"].float(),
        torch.ones_like(outs["zero"], dtype=torch.float32),
        rtol=0,
        atol=1.6e-2,
    )
    plain = vca.vc_quantize(k, v, perm=ops.perm, demean=False)
    assert plain.params.demean is False
    out_off = wrapper.run(q, plain.k, plain.v, vc=plain.params)
    out_on = wrapper.run(q, plain.k, plain.v, vc=replace(plain.params, demean=True))
    assert torch.equal(out_on, out_off)
    out = wrapper.run(q, ops.k, ops.v, vc=ops.params)
    assert not torch.equal(
        out, wrapper.run(q, ops.k, ops.v, vc=replace(ops.params, demean=False))
    )
    assert _relative_error(out.float(), vca.vc_reference(q, ops)) < 6e-2


@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    (
        pytest.param({"v_dtype": torch.bfloat16}, ValueError, "E4M3 V", id="bf16-v"),
        pytest.param(
            {"q_dtype": _FP8, "k_dtype": _FP8, "v_dtype": torch.bfloat16},
            NotImplementedError,
            "same dtype",
            id="qk8-bf16-v",
        ),
        pytest.param(
            {"mask_type": "causal"}, ValueError, "dense contiguous", id="causal"
        ),
        pytest.param({"num_qo_heads": 2}, ValueError, "equal Q and K/V head", id="gqa"),
        pytest.param(
            {"vc_config": {"k_block_size": 128}},
            TypeError,
            "VCAttentionConfig instance",
            id="dict-config",
        ),
        pytest.param(
            {"vc_config": lambda: VCAttentionConfig(k_block_size=64)},
            ValueError,
            "k_block_size must be",
            id="k-block-64",
        ),
        pytest.param(
            {"vc_config": lambda: VCAttentionConfig(q_block_size=96)},
            ValueError,
            "q_block_size must be",
            id="q-block-96",
        ),
    ),
)
def test_vc_plan_rejects_unsupported_recipes(overrides, error, match):
    """A VC plan takes the dense contiguous D128 MHA kernel with bf16 or E4M3 Q/K
    and E4M3 V."""
    with pytest.raises(error, match=match):
        _plan(
            BatchPrefillTSWrapper(),
            batch_size=1,
            seq_len=256,
            num_heads=1,
            **{k: v() if callable(v) else v for k, v in overrides.items()},
        )


def test_vc_kernel_config_rejects_invalid_profiles():
    """The kernel config owns the VC recipe rules."""
    from cutlass import BFloat16, Float32, Float8E4M3FN

    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import FmhaTs

    arguments = dict(
        in_qk_dtype=BFloat16,
        in_pv_dtype=Float8E4M3FN,
        qk_acc_dtype=Float32,
        pv_acc_dtype=Float32,
        d=_HEAD_DIM,
        d_v=_HEAD_DIM,
        is_persistent=False,
        is_causal=False,
        is_clc_dynamic=False,
        two_cta_umma=True,
    )
    FmhaTs(**arguments, vc_k_block_size=128, vc_num_q_heads=1)
    qk8 = {**arguments, "in_qk_dtype": Float8E4M3FN}
    FmhaTs(**qk8, vc_k_block_size=128, vc_num_q_heads=1, vc_q_block_log2=6)
    with pytest.raises(ValueError, match="vc_q_block_log2 must be"):
        FmhaTs(**qk8, vc_k_block_size=128, vc_num_q_heads=1, vc_q_block_log2=9)
    with pytest.raises(ValueError, match="vc_k_block_size must equal"):
        FmhaTs(**arguments, vc_k_block_size=64, vc_num_q_heads=1)
    with pytest.raises(ValueError, match="requires vc_num_q_heads"):
        FmhaTs(**arguments, vc_k_block_size=128)
    with pytest.raises(ValueError, match="requires vc_k_block_size"):
        FmhaTs(**arguments, vc_num_q_heads=1)


@_REQUIRES_CONTEXT_GPU
def test_vc_run_rejects_mismatched_operands():
    q, k, v = _random_inputs(1, 1024, 2)
    ops = vca.vc_quantize(k, v)
    wrapper = BatchPrefillTSWrapper()
    _plan(wrapper, batch_size=1, seq_len=1024, num_heads=2)
    good = ops.params
    with pytest.raises(ValueError, match="v_scale must have shape"):
        wrapper.run(q, ops.k, ops.v, vc=replace(good, v_scale=good.v_scale[:, :, :-1]))
    with pytest.raises(ValueError, match="v_scale must have dtype"):
        wrapper.run(q, ops.k, ops.v, vc=replace(good, v_scale=good.v_scale.half()))
    with pytest.raises(ValueError, match="tile_means must be contiguous"):
        wrapper.run(
            q,
            ops.k,
            ops.v,
            vc=replace(
                good, tile_means=good.tile_means.repeat(1, 1, 1, 1, 2)[..., ::2]
            ),
        )
    with pytest.raises(TypeError, match="VCAttentionParams instance"):
        wrapper.run(q, ops.k, ops.v, vc={"v_scale": good.v_scale})
    with pytest.raises(ValueError, match="required by a VC plan"):
        wrapper.run(q, ops.k, ops.v)
    plain = BatchPrefillTSWrapper()
    _plan(plain, batch_size=1, seq_len=1024, num_heads=2, vc_config=None)
    with pytest.raises(ValueError, match="rejected by a plan without"):
        plain.run(q, ops.k, ops.v, vc=good)
    ops8 = vca.vc_quantize_fp8(q, k, v)
    with pytest.raises(ValueError, match="belong to VC-Attention-QK8"):
        wrapper.run(q, ops.k, ops.v, vc=replace(good, k_scale=ops8.k_scale))
    qk8 = BatchPrefillTSWrapper()
    _plan(qk8, batch_size=1, seq_len=1024, num_heads=2, qk_dtype=_FP8)
    good8 = ops8.params
    with pytest.raises(ValueError, match="k_scale must have shape"):
        qk8.run(
            ops8.q, ops8.k, ops8.v, vc=replace(good8, k_scale=good8.k_scale[:, :-1])
        )
    with pytest.raises(ValueError, match="q_scale must have dtype"):
        qk8.run(ops8.q, ops8.k, ops8.v, vc=replace(good8, q_scale=good8.q_scale.half()))
    with pytest.raises(ValueError, match="require vc.q_scale"):
        qk8.run(ops8.q, ops8.k, ops8.v, vc=good, validate=False)


def test_one_shot_vc_config_requires_the_operands():
    """A one-shot recipe without operands has nothing to run."""
    q = torch.empty((1, 256, 1, _HEAD_DIM), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="vc_config requires"):
        batch_prefill(q, q, q.to(_FP8), vc_config=VCAttentionConfig())


@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize("batch,seq_len,heads", [(1, 2000, 2), (2, 4096, 4)])
def test_vc_quantize_fused_matches_reference(batch, seq_len, heads):
    _, k, v = _random_inputs(batch, seq_len, heads)
    perm = vca.vc_token_permutation(v)
    ref = vca.vc_quantize(k, v, perm=perm)
    fused = vca.vc_quantize_fused(k, v, perm=perm)
    assert torch.equal(fused.k, ref.k)
    torch.testing.assert_close(fused.v_scale, ref.v_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.mean, ref.mean, rtol=1e-5, atol=1e-6)
    assert torch.equal(fused.mu, ref.mu)
    # E4M3 rounding ties may differ by one code between the two paths.
    mismatch = (fused.v.float() != ref.v.float()).float().mean().item()
    assert mismatch < 1e-3


@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize("batch,seq_len,heads", [(1, 2000, 2), (2, 4096, 4)])
def test_vc_quantize_fp8_fused_matches_reference(batch, seq_len, heads):
    q, k, v = _random_inputs(batch, seq_len, heads)
    perm = vca.vc_token_permutation(v)
    ref = vca.vc_quantize_fp8(q, k, v, perm=perm)
    fused = vca.vc_quantize_fp8_fused(q, k, v, perm=perm)
    torch.testing.assert_close(fused.q_scale, ref.q_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.k_scale, ref.k_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.v_scale, ref.v_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.mean, ref.mean, rtol=1e-5, atol=1e-6)
    assert torch.equal(fused.mu, ref.mu)
    for name in ("q", "k", "v"):
        # E4M3 rounding ties may differ by one code between the two paths.
        mismatch = (
            (getattr(fused, name).float() != getattr(ref, name).float()).float().mean()
        )
        assert mismatch.item() < 1e-3, name


def test_vc_flat_scale_layout_roundtrip():
    from flashinfer.attention.prims_ts.sage import flat_scale_numel, flat_scale_slot

    batch, seq_len, heads, block = 3, 2000, 2, 128
    nb = (seq_len + block - 1) // block
    scale = torch.rand(batch, heads, nb) + 0.5
    flat = vca.flat_block_scales(scale, seq_len, block)
    assert flat.shape == (heads, flat_scale_numel(batch, seq_len, block))
    for b in range(batch):
        for j in range(nb):
            slot = flat_scale_slot(b, j * block, seq_len, 7)
            assert torch.equal(flat[:, slot], scale[b, :, j])
    assert torch.equal(vca.block_scales_from_flat(flat, batch, seq_len, block), scale)


def test_vc_hadamard_is_orthonormal_and_qk_invariant():
    hm = vca.hadamard_matrix(_HEAD_DIM, torch.device("cpu"))
    torch.testing.assert_close(hm @ hm.T, torch.eye(_HEAD_DIM), atol=1e-5, rtol=0)
    torch.manual_seed(0)
    q = torch.randn(1, 512, 2, _HEAD_DIM)
    k = torch.randn_like(q)
    a = torch.einsum("bqhd,bkhd->bhqk", q, k)
    b = torch.einsum("bqhd,bkhd->bhqk", q @ hm, k @ hm)
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-4)


@_REQUIRES_CONTEXT_GPU
@pytest.mark.arch_blackwell
def test_vc_preprocessor_step_schedule():
    """First 25% of steps: grouping + demeaning (refreshed every 4 steps); afterwards demeaning
    is off and the permutation the window left behind is kept (paper Section 3.4)."""
    q, k, v = _random_inputs(1, 1024, 2)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    wrapper = BatchPrefillTSWrapper()
    _plan(wrapper, batch_size=1, seq_len=1024, num_heads=2, sm_scale=sm_scale)
    prep = VCAttentionPreprocessor()
    exact = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    ).transpose(1, 2)
    step0 = prep.prepare(k, v, denoise_step=(0, 40))
    assert step0.demean
    out = wrapper.run(q, step0.k, step0.v, vc=step0.params)
    assert (
        _relative_error(out.float(), vca.vc_reference(q, step0, sm_scale=sm_scale))
        < 6e-2
    )
    step1 = prep.prepare(k, v, denoise_step=(1, 40))
    assert torch.equal(step1.perm, step0.perm)  # no refresh at step 1
    step4 = prep.prepare(k, v.flip(1), denoise_step=(4, 40))
    assert not torch.equal(step4.perm, step0.perm)  # refreshed at step 4
    late = prep.prepare(k, v, denoise_step=(20, 40))  # V-Smooth off
    assert not late.demean
    assert torch.equal(late.perm, step4.perm)
    out_late = wrapper.run(q, late.k, late.v, vc=late.params)
    assert _relative_error(out_late.float(), exact) < 1e-1
    # The same schedule drives the QK8 preparation, which also returns E4M3 Q.
    qk8 = VCAttentionPreprocessor().prepare(k, v, q=q, denoise_step=(0, 40))
    assert qk8.q.dtype == _FP8 and qk8.q_scale is not None and qk8.demean
    assert torch.equal(qk8.v_scale, vca.vc_quantize_fp8(q, k, v, perm=qk8.perm).v_scale)
