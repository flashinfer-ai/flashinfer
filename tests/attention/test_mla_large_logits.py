"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""FA2 MLA decode with large-magnitude QK logits.

The online softmax computed the P exponent as "s * sm_scale - max * sm_scale",
which nvcc contracts into an FMA.  For the selected element that expression must
be exactly 0 (or exactly the negative gap to the maximum), but the contracted
form evaluates to the rounding error of "max * sm_scale".  The error grows with
the logit magnitude: once P leaves the fp16 range the row's denominator becomes
inf and the output NaN.  The history rescale had the same shape, so an unchanged
maximum rescaled the running state by that same error on every KV tile.  Both
are computed as "(s - max) * sm_scale" now.
"""

import math

import pytest
import torch

import flashinfer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

CKV_DIM = 512
KPE_DIM = 64
SM_SCALE = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)
HEADS = 16
PAGE_SIZE = 16


def _reference(q_nope, q_pe, ckv, kpe):
    """float64 reference for a single request with contiguous KV pages."""
    query = torch.cat([q_nope[0].float(), q_pe[0].float()], dim=-1)
    key = torch.cat(
        [ckv.reshape(-1, CKV_DIM).float(), kpe.reshape(-1, KPE_DIM).float()], dim=-1
    )
    scores = (query.double() @ key.double().T) * SM_SCALE
    probs = torch.softmax(scores, dim=-1)
    return probs @ ckv.reshape(-1, CKV_DIM).double()


def _run_fa2_mla(q_nope, q_pe, ckv, kpe):
    """Run one FA2 MLA decode request over contiguous pages and return its output."""
    seq_len = ckv.shape[0] * ckv.shape[1]
    num_pages = ckv.shape[0]
    device = q_nope.device
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    metadata = flashinfer.mla.MLAPlanMetadata.csr(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        torch.arange(num_pages, dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=HEADS,
        head_dim_ckv=CKV_DIM,
        head_dim_kpe=KPE_DIM,
        page_size=PAGE_SIZE,
        causal=False,
        sm_scale=SM_SCALE,
        q_data_type=q_nope.dtype,
        kv_data_type=ckv.dtype,
        query_layout="split",
        kv_cache_layout="split",
    )
    out = wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv, kpe))
    torch.cuda.synchronize()
    return out


def _make_inputs(seq_len, magnitude, constant):
    """Build one request worth of q and kv tensors on cuda, scaled by magnitude.

    Constant inputs fill every element with the same value, so each KV position
    scores identically; otherwise the values come from a fixed-seed normal
    distribution clamped to the fp16 range.
    """
    num_pages = seq_len // PAGE_SIZE
    if constant:
        # Every KV position gets the same score, so the exact output is V itself
        # and any deviation comes from the softmax exponent path.
        q_nope = torch.full((1, HEADS, CKV_DIM), magnitude, dtype=torch.float16)
        q_pe = torch.zeros(1, HEADS, KPE_DIM, dtype=torch.float16)
        ckv = torch.full(
            (num_pages, PAGE_SIZE, CKV_DIM), magnitude, dtype=torch.float16
        )
        kpe = torch.zeros(num_pages, PAGE_SIZE, KPE_DIM, dtype=torch.float16)
    else:
        generator = torch.Generator().manual_seed(0)
        q_nope = (
            torch.randn(1, HEADS, CKV_DIM, generator=generator) * magnitude
        ).clamp(-65504, 65504)
        q_pe = torch.zeros(1, HEADS, KPE_DIM)
        ckv = (
            torch.randn(num_pages, PAGE_SIZE, CKV_DIM, generator=generator) * magnitude
        ).clamp(-65504, 65504)
        kpe = torch.zeros(num_pages, PAGE_SIZE, KPE_DIM)
    device = torch.device("cuda:0")
    return (
        q_nope.to(device=device, dtype=torch.float16),
        q_pe.to(device=device, dtype=torch.float16),
        ckv.to(device=device, dtype=torch.float16),
        kpe.to(device=device, dtype=torch.float16),
    )


@pytest.mark.parametrize("magnitude", [6000.0, 12000.0, 30000.0])
def test_constant_scores_large_logits_are_exact(magnitude):
    """A uniform-score row must return V itself, not NaN, for large scaled maxima."""
    # 512 * magnitude * magnitude * SM_SCALE is ~7.7e8 for magnitude 6000: the
    # fp32 ulp of the scaled maximum is already 64, so exp2(rounding error)
    # overflowed the fp16 P and the whole row came back as NaN.
    q_nope, q_pe, ckv, kpe = _make_inputs(128, magnitude, constant=True)
    out = _run_fa2_mla(q_nope, q_pe, ckv, kpe)
    assert torch.isfinite(out).all(), f"{int(torch.isnan(out).sum())} NaNs"
    torch.testing.assert_close(
        out, torch.full_like(out, magnitude), rtol=1e-3, atol=1e-3
    )


def test_random_large_logits_match_fp64_reference():
    """Random large-magnitude inputs must stay finite and match the fp64 reference."""
    # 32 KV tiles of online-softmax history rescale on top of the per-tile
    # exponent; this reproduces the NaN end to end.
    q_nope, q_pe, ckv, kpe = _make_inputs(2048, 20000.0, constant=False)
    out = _run_fa2_mla(q_nope, q_pe, ckv, kpe)
    reference = _reference(q_nope, q_pe, ckv, kpe)
    assert torch.isfinite(out).all(), f"{int(torch.isnan(out).sum())} NaNs"
    relative = (out.double() - reference).abs().max().item() / (
        reference.abs().max().item() + 1e-6
    )
    assert relative < 5e-3, f"relative error {relative:.4g}"


def test_normal_logits_still_match_fp64_reference():
    """Control: ordinary logits keep the fp16-level accuracy of the old exponent path."""
    q_nope, q_pe, ckv, kpe = _make_inputs(1024, 1.0, constant=False)
    out = _run_fa2_mla(q_nope, q_pe, ckv, kpe)
    reference = _reference(q_nope, q_pe, ckv, kpe)
    relative = (out.double() - reference).abs().max().item() / (
        reference.abs().max().item() + 1e-6
    )
    assert relative < 5e-3, f"relative error {relative:.4g}"
