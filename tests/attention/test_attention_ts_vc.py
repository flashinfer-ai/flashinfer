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

"""VC-Attention-QK16 (bf16 Q/K with E4M3 V tile residuals) for the PrimTS context kernel."""

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
_MEAN_CONFIG = VCAttentionConfig()


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
    # V repair budget as a fraction of tokens. 0 means tile means.
    repair_budget: float = 0.0


_CASES = (
    _VCCase("single-tile", 1, 512, 1),
    _VCCase("partial-tile", 1, 2000, 2),
    # Odd Q-tile count (cluster padding), partial last K/V tile, two batches.
    _VCCase("two-batches", 2, 2120, 2),
    _VCCase("one-shot", 1, 2000, 2, one_shot=True),
    # V repair, one repair tile before the masked partial tail, two tiles with no tail.
    _VCCase("repair-partial-tile", 1, 2000, 2, repair_budget=0.02),
    _VCCase("repair-two-batches", 2, 4096, 2, repair_budget=0.05),
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
    vc_config=_MEAN_CONFIG,
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
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        v_dtype=_FP8,
        out_dtype=torch.bfloat16,
        sm_scale=sm_scale,
        vc_config=vc_config,
    )
    wrapper.plan(**{**arguments, **overrides})


def _run_case(case: _VCCase):
    q, k, v = _random_inputs(case.batch_size, case.seq_len, case.num_heads)
    if case.repair_budget:
        ops = vca.vc_quantize_repair(k, v, budget=case.repair_budget)
    else:
        ops = vca.vc_quantize(k, v)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    if case.one_shot:
        out = batch_prefill(q, ops.k, ops.v, sm_scale=sm_scale, vc=ops.params)
    else:
        wrapper = BatchPrefillTSWrapper()
        _plan(
            wrapper,
            batch_size=case.batch_size,
            seq_len=case.seq_len,
            num_heads=case.num_heads,
            sm_scale=sm_scale,
            vc_config=VCAttentionConfig(repair_tiles=ops.repair_tiles),
        )
        out = wrapper.run(q, ops.k, ops.v, vc=ops.params)
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


@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    (
        pytest.param(
            {"q_dtype": _FP8, "k_dtype": _FP8, "v_dtype": _FP8},
            ValueError,
            "16-bit Q/K",
            id="fp8-qk",
        ),
        pytest.param({"v_dtype": torch.bfloat16}, ValueError, "E4M3 V", id="bf16-v"),
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
    ),
)
def test_vc_plan_rejects_unsupported_recipes(overrides, error, match):
    """A VC plan takes the dense contiguous D128 MHA kernel with bf16 Q/K and E4M3 V."""
    with pytest.raises(error, match=match):
        _plan(
            BatchPrefillTSWrapper(),
            batch_size=1,
            seq_len=256,
            num_heads=1,
            **{k: v() if callable(v) else v for k, v in overrides.items()},
        )


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
    with pytest.raises(ValueError, match="vc_config requires"):
        batch_prefill(q, k, v.to(_FP8), vc_config=VCAttentionConfig())
    repair = vca.vc_quantize_repair(k, v, budget=0.02)
    with pytest.raises(NotImplementedError, match="does not run V repair"):
        batch_prefill(q, repair.k, repair.v, vc=repair.params)
    with pytest.raises(NotImplementedError, match="does not run V repair"):
        batch_prefill(
            q,
            repair.k,
            repair.v,
            vc=repair.params,
            vc_config=VCAttentionConfig(repair_tiles=repair.repair_tiles),
        )


@_REQUIRES_CONTEXT_GPU
@pytest.mark.arch_blackwell
def test_vc_preprocessor_step_schedule():
    """First 25% of steps: grouping + demeaning (refreshed every 4 steps); afterwards demeaning
    is off and the permutation the window left behind is kept (paper Section 3.4)."""
    _, k, v = _random_inputs(1, 1024, 2)
    prep = VCAttentionPreprocessor()
    step0 = prep.prepare(k, v, denoise_step=(0, 40))
    assert step0.demean
    step1 = prep.prepare(k, v, denoise_step=(1, 40))
    assert torch.equal(step1.perm, step0.perm)  # no refresh at step 1
    step4 = prep.prepare(k, v.flip(1), denoise_step=(4, 40))
    assert not torch.equal(step4.perm, step0.perm)  # refreshed at step 4
    late = prep.prepare(k, v, denoise_step=(20, 40))  # V-Smooth off
    assert not late.demean
    assert torch.equal(late.perm, step4.perm)
