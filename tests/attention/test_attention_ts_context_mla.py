# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-absorbed MLA: compact separate Q/K=192, V/O=128 on Blackwell."""

import math

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")

from flashinfer.attention.prims_ts import BatchPrefillTSWrapper
from flashinfer.utils import is_sm100a_supported

pytestmark = [
    pytest.mark.arch_blackwell,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
        reason="requires Blackwell",
    ),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_mla_prefill(dtype, out_dtype, packed, causal):
    torch.manual_seed(123)
    q_lens = (65, 129) if packed else (257,)
    k_lens = (129, 257) if packed else (257,)
    hq, hkv = (8, 2) if packed else (96, 1)
    q = torch.randn(sum(q_lens), hq, 192, device="cuda")
    k = torch.randn(sum(k_lens), hkv, 192, device="cuda")
    v = torch.randn(sum(k_lens), hkv, 128, device="cuda")
    # Exercise descales with large FP8 operands, as in the benchmark.
    operand_scale = 16 if dtype == torch.float8_e4m3fn else 1
    q = (q * operand_scale).to(dtype)
    k = (k * operand_scale).to(dtype)
    v = (v * operand_scale).to(dtype)
    sm_scale = 1 / (math.sqrt(192) * operand_scale**2)
    output_scale = 0.75 / operand_scale
    qo = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    ko = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    # Compute the reference from the actual quantized operands before planning.
    expected = []
    q_offset = k_offset = 0
    for nq, nk in zip(q_lens, k_lens, strict=True):
        qr = q[q_offset : q_offset + nq].float().transpose(0, 1)
        kr = k[k_offset : k_offset + nk].float().transpose(0, 1)
        vr = v[k_offset : k_offset + nk].float().transpose(0, 1)
        kr = kr.repeat_interleave(hq // hkv, dim=0)
        vr = vr.repeat_interleave(hq // hkv, dim=0)
        scores = (qr @ kr.transpose(-1, -2)) * sm_scale
        if causal:
            mask = torch.arange(nk, device="cuda")[None, :] > (
                nk - nq + torch.arange(nq, device="cuda")[:, None]
            )
            scores.masked_fill_(mask, -torch.inf)
        expected.append((scores.softmax(-1) @ vr).transpose(0, 1) * output_scale)
        q_offset += nq
        k_offset += nk
    expected = torch.cat(expected)
    if not packed:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        expected = expected.unsqueeze(0)
    out = torch.empty((*q.shape[:-1], 128), dtype=out_dtype, device="cuda")
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        q,
        k,
        v,
        qo_indptr=qo if packed else None,
        kv_indptr=ko if packed else None,
        mask_type="causal" if causal else "dense",
        sm_scale=sm_scale,
        output_scale=output_scale,
        out_dtype=out_dtype,
    )
    actual = wrapper.run(q, k, v, out=out)
    assert actual is out
    assert actual.shape == expected.shape
    has_fp8 = torch.float8_e4m3fn in (dtype, out_dtype)
    atol, rtol = (0.13, 0.05) if has_fp8 else (0.01, 0.02)
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
    relative_l2 = torch.linalg.vector_norm(
        actual.float() - expected
    ) / torch.linalg.vector_norm(expected)
    assert relative_l2.item() < (0.05 if has_fp8 else 0.01)
    allocated = wrapper.run(q, k, v)
    assert allocated.shape == out.shape and allocated.dtype == out_dtype
    torch.testing.assert_close(allocated.float(), expected, atol=atol, rtol=rtol)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out)
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out.float(), expected, atol=atol, rtol=rtol)
    with pytest.raises(ValueError, match="out must have shape"):
        wrapper.run(q, k, v, out=torch.empty_like(q, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="v must have"):
        wrapper.run(q, k, torch.empty_like(k))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mla_prefill_single_tile_uses_qk_tail(dtype):
    # Isolate the final 64 Q/K columns so dropping them cannot look correct.
    torch.manual_seed(321)
    q = torch.zeros(1, 65, 96, 192, device="cuda")
    k = torch.zeros(1, 65, 1, 192, device="cuda")
    q[..., 128:] = torch.randn_like(q[..., 128:])
    k[..., 128:] = torch.randn_like(k[..., 128:])
    q, k = q.to(dtype), k.to(dtype)
    v = torch.randn(1, 65, 1, 128, device="cuda").to(dtype)
    scores = (q[0].float().transpose(0, 1) @ k[0, :, 0].float().T) / math.sqrt(192)
    mask = torch.ones(65, 65, device="cuda", dtype=torch.bool).triu(1)
    expected = (
        (scores.masked_fill(mask, -torch.inf).softmax(-1) @ v[0, :, 0].float())
        .transpose(0, 1)
        .unsqueeze(0)
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(q, k, v, mask_type="causal", out_dtype=torch.bfloat16)
    actual = wrapper.run(q, k, v)
    atol, rtol = (0.13, 0.05) if dtype == torch.float8_e4m3fn else (0.01, 0.02)
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("seq_len", [256, 257])
def test_mla_prefill_negative_scores(dtype, seq_len):
    # Every score is -192. A fused maximum incorrectly initialized to zero
    # would underflow FP8 probabilities; an unmasked K tail would dominate
    # these negative scores. Uniform attention must return the constant V.
    # Aligned K uses fused LDTM.STAT; a partial K tile uses masked reduction.
    q = torch.ones(1, seq_len, 96, 192, device="cuda", dtype=torch.bfloat16).to(dtype)
    k = -torch.ones(1, seq_len, 1, 192, device="cuda")
    k = k.to(dtype)
    v = torch.ones(1, seq_len, 1, 128, device="cuda", dtype=torch.bfloat16).to(dtype)
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        q, k, v, mask_type="dense", output_scale=0.75, out_dtype=torch.bfloat16
    )
    actual = wrapper.run(q, k, v)
    torch.testing.assert_close(
        actual.float(), torch.full_like(actual.float(), 0.75), atol=1e-3, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mla_prefill_variable_window(dtype):
    # Variable windows use the single-query schedule with two K stages and
    # one V stage. Cross K tiles with row-dependent bounds and partial tails.
    torch.manual_seed(456)
    q = torch.randn(1, 257, 4, 192, device="cuda").to(dtype)
    k = torch.randn(1, 385, 2, 192, device="cuda").to(dtype)
    v = torch.randn(1, 385, 2, 128, device="cuda").to(dtype)
    positions = torch.arange(257, device="cuda", dtype=torch.int32)
    starts = (positions - 33).clamp(min=0).unsqueeze(0).contiguous()
    ends = (positions + 129).clamp(max=384).unsqueeze(0).contiguous()
    kr = k[0].float().transpose(0, 1).repeat_interleave(2, dim=0)
    vr = v[0].float().transpose(0, 1).repeat_interleave(2, dim=0)
    scores = (q[0].float().transpose(0, 1) @ kr.transpose(-1, -2)) / math.sqrt(192)
    keys = torch.arange(385, device="cuda")[None, :]
    mask = (keys < starts[0, :, None]) | (keys > ends[0, :, None])
    expected = (
        (scores.masked_fill(mask, -torch.inf).softmax(-1) @ vr)
        .transpose(0, 1)
        .unsqueeze(0)
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        q,
        k,
        v,
        mask_type="variable_window",
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
        out_dtype=torch.bfloat16,
    )
    actual = wrapper.run(q, k, v)
    atol, rtol = (0.13, 0.05) if dtype == torch.float8_e4m3fn else (0.01, 0.02)
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
