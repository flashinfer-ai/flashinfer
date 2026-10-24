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
"""SM120 (GB202) FP8 MiniMax-H3 packed-varlen attention against an FP32 exact-softmax oracle.

Q / K / P / V are E4M3 tensor-core operands (per-token Q scale, per-128-key-block K scale after
the segment-mean shift, 2^8-biased P, per-(segment, channel) V scale) with an FP32 softmax, so the
BF16 output carries FP8 quantization noise: the package tolerance is ``atol = rtol = 0.1`` on every
element plus a relative-L2 bound, checked on packed boundaries, tails, empty segments and
multi-segment streams.
"""

import math

import pytest
import torch

from flashinfer.diffusion_ops import minimax_h3_sm120_varlen_attention_fp8
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    MiniMaxH3VarlenPlan,
    assign_unit_slots,
    normalize_cu_seqlens,
    workspace_bytes,
)
from flashinfer.utils import get_compute_capability

ATOL = 0.1
RTOL = 0.1
REL_L2_MAX = 0.08
# Packed boundaries: single tokens, exact / one-past tile edges, empty segments, ragged
# multi-segment streams, and a multi-tile single segment.
CU_SEQLENS = [
    (0, 1),
    (0, 128),
    (0, 129),
    (0, 0, 129, 129, 500),
    (0, 133, 300, 900),
    (0, 257, 4567, 4824),
    (0, 4097),
]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = get_compute_capability(torch.device("cuda:0"))
    return major == 12


requires_sm120 = pytest.mark.skipif(
    not _supported(), reason="requires an SM120 (RTX 5090 / RTX PRO 6000 Blackwell) GPU"
)


def synthetic_inputs(tokens: int, heads: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        torch.randn(
            tokens, heads, MINIMAX_H3_HEAD_DIM, generator=generator, dtype=torch.float32
        )
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )


def fp32_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: tuple,
    query_chunk: int = 1024,
) -> torch.Tensor:
    """Exact-softmax FP32 attention per segment and head over the BF16 operands (TF32 off)."""
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for a, b in zip(cu_seqlens, cu_seqlens[1:], strict=False):
            if b == a:
                continue
            # One head at a time: a [query_chunk, L] FP32 score block stays below 0.5 GiB even at
            # the 110k-token rows, so the oracle fits next to the operands on a 32 GiB board.
            for h in range(q.shape[1]):
                kf = k[a:b, h].float()  # [L, D]
                vf = v[a:b, h].float()
                for start in range(a, b, query_chunk):
                    stop = min(start + query_chunk, b)
                    scores = (
                        torch.matmul(q[start:stop, h].float(), kf.transpose(0, 1))
                        * scale
                    )
                    out[start:stop, h] = torch.matmul(torch.softmax(scores, dim=-1), vf)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def _check(actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected).abs()
    bound = ATOL + RTOL * expected.abs()
    assert torch.isfinite(actual.float()).all(), "non-finite output"
    bad = int((diff > bound).sum().item())
    assert bad == 0, (
        f"{bad} elements outside atol={ATOL} rtol={RTOL} (max abs err {diff.max().item():.4f})"
    )
    rel_l2 = (diff.norm() / expected.norm().clamp_min(1e-6)).item()
    assert rel_l2 < REL_L2_MAX, f"relative L2 error {rel_l2:.4f} exceeds {REL_L2_MAX}"


@requires_sm120
@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("heads", [MINIMAX_H3_NUM_HEADS, 3])
def test_minimax_h3_sm120_varlen_attention_fp8_matches_fp32_oracle(
    cu_seqlens: tuple, heads: int
) -> None:
    if heads != MINIMAX_H3_NUM_HEADS and cu_seqlens[-1] > 1000:
        pytest.skip("small-head sweep covers the boundary shapes only")
    device = torch.device("cuda:0")
    tokens = cu_seqlens[-1]
    q, k, v = synthetic_inputs(tokens, heads, seed=tokens + heads, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, cu_seqlens_host=cu_seqlens)
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    _check(out, fp32_oracle(q, k, v, cu_seqlens))
    # Idempotence: the quantized operands and scales are rebuilt every call, bit-exactly.
    again = minimax_h3_sm120_varlen_attention_fp8(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    assert torch.equal(out, again)


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_preallocated_output_and_device_cu() -> (
    None
):
    device = torch.device("cuda:0")
    cu_seqlens = (0, 300, 700)
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=7, device=device
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    # No host copy of cu_seqlens: the plan reads the bounds from the device once.
    result = minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, out)
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    _check(out, fp32_oracle(q, k, v, cu_seqlens))


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_fp8_custom_scale_and_empty_stream() -> None:
    device = torch.device("cuda:0")
    cu_seqlens = (0, 200)
    q, k, v = synthetic_inputs(200, 4, seed=11, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    scale = 0.05
    out = minimax_h3_sm120_varlen_attention_fp8(
        q, k, v, cu, cu_seqlens_host=cu_seqlens, softmax_scale=scale
    )
    torch.cuda.synchronize()
    expected = torch.softmax(
        torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * scale, dim=-1
    )
    expected = torch.einsum("hmn,nhd->mhd", expected, v.float())
    _check(out, expected)
    # Every segment empty: nothing is written and no kernel is launched.
    empty = torch.zeros(0, 4, MINIMAX_H3_HEAD_DIM, dtype=torch.bfloat16, device=device)
    cu0 = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
    assert (
        minimax_h3_sm120_varlen_attention_fp8(empty, empty, empty, cu0).shape
        == empty.shape
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_sm120_varlen_attention_fp8_rejects_bad_inputs() -> None:
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(300, 4, seed=3, device=device)
    cu = torch.tensor([0, 300], dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q.float(), k, v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k[:299], v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu.to(torch.int64))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, cu_seqlens_host=(0, 299))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, cu_seqlens_host=(5, 300))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(
            q, k, v, cu, cu_seqlens_host=(0, 200, 100, 300)
        )
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_fp8(
            q, k, v, cu, out=torch.empty_like(q)[:, :3]
        )


def test_minimax_h3_sm120_varlen_plan_covers_every_unit_once() -> None:
    bounds = normalize_cu_seqlens((0, 0, 129, 129, 500, 4824))
    heads = 5
    plan = MiniMaxH3VarlenPlan(bounds, heads, torch.device("cpu"), num_ctas=170)
    assert plan.num_segments == 5 and plan.total_tokens == 4824
    # Q tiles per segment: 0, 2, 0, 3, 34 -> 39 tiles, one unit per (non-empty tile, head).
    assert plan.num_tiles == 39 and plan.num_units == 39 * heads
    assert plan.grid == 170
    assert plan.seg_tile_begin.tolist() == [0, 0, 2, 2, 5, 39]
    table = plan.unit_table.view(-1, 2).tolist()
    units = {(seg, packed >> 16, packed & 0xFFFF) for seg, packed in table}
    assert len(units) == plan.num_units
    for seg, head, tile in units:
        assert 0 <= head < heads
        assert bounds[seg + 1] > bounds[seg]
        assert tile * 128 < bounds[seg + 1] - bounds[seg]
    # Slot order: the heaviest units land in the first full round.
    first_round = [seg for seg, _ in table[: plan.grid]]
    assert all(seg == 4 for seg in first_round)
    assert workspace_bytes(4824, heads, 5) > 0
    assert assign_unit_slots([], 8) == []
    assert sorted(assign_unit_slots([3, 1, 2, 5, 4], 2)) == [0, 1, 2, 3, 4]
