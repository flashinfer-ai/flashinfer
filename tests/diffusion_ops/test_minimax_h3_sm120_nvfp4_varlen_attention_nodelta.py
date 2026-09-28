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
"""Tests for the opt-in UNCOMPENSATED variant of the SM120 NVFP4 MiniMax-H3 packed-varlen attention.

``minimax_h3_sm120_varlen_attention_nvfp4_nodelta`` shares the contract, signature, workspace and
pre-processing launches of ``minimax_h3_sm120_varlen_attention_nvfp4`` but does not add the Q
block-mean term ``qm K^T`` back to the scores.  It trades a measurably larger FP4 error (relative L2
about 0.209 vs 0.191 on Gaussian inputs) for a faster attention launch.  These tests pin that
trade-off: the FP4 tolerance ``atol = 1.0, rtol = 0.1`` holds on every element, the relative L2 stays
within ``1.15x`` the default route's on the same inputs, the output is deterministic, and the routes
share their validation.  Every test needs an SM120 GPU.
"""

import pytest
import torch

from flashinfer.diffusion_ops import (
    minimax_h3_sm120_varlen_attention_nvfp4,
    minimax_h3_sm120_varlen_attention_nvfp4_nodelta,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    MINIMAX_H3_NUM_HEADS,
)

# tests/diffusion_ops is not a package: pytest's default import mode puts this directory on sys.path,
# so the default route's fixtures import by module name.
from test_minimax_h3_sm120_nvfp4_varlen_attention import (  # noqa: I001
    ATOL,
    RTOL,
    CU_SEQLENS,
    _check,
    fp32_oracle,
    requires_sm120,
    synthetic_inputs,
)

# rel-L2(nodelta) / rel-L2(default) on the same inputs: measured 1.09 on Gaussian inputs at every
# contract shape; the bound leaves headroom for the boundary shapes without accepting a broken kernel.
REL_L2_RATIO_MAX = 1.15


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-6)).item()
    )


@requires_sm120
@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("heads", [MINIMAX_H3_NUM_HEADS, 3])
def test_minimax_h3_sm120_varlen_attention_nvfp4_nodelta_matches_fp32_oracle(
    cu_seqlens, heads
) -> None:
    """FP4 tolerance vs the FP32 oracle on every element, relative L2 within 1.15x the default
    route's on the same inputs, and a deterministic second call."""
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(cu_seqlens[-1], heads, seed=11, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    expected = fp32_oracle(q, k, v, cu_seqlens)
    out = minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    base = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    _check(out, expected)
    if cu_seqlens[-1] > 0:
        ratio = _rel_l2(out, expected) / max(_rel_l2(base, expected), 1e-6)
        assert ratio <= REL_L2_RATIO_MAX, (
            f"uncompensated route relative L2 is {ratio:.4f}x the default route's "
            f"(bound {REL_L2_RATIO_MAX})"
        )
    again = minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    assert torch.equal(again, out), "second call is not bit-identical to the first"


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_nodelta_differs_from_default_only_by_the_term() -> (
    None
):
    """The two routes agree within the FP4 tolerance of each other (they differ only by the
    uncompensated block-mean term) and both stay within the tolerance of the oracle."""
    device = torch.device("cuda:0")
    cu_seqlens = (0, 257, 4567, 4824)
    q, k, v = synthetic_inputs(
        cu_seqlens[-1], MINIMAX_H3_NUM_HEADS, seed=5, device=device
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    expected = fp32_oracle(q, k, v, cu_seqlens)
    out_nodelta = minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    out_default = minimax_h3_sm120_varlen_attention_nvfp4(
        q, k, v, cu, cu_seqlens_host=cu_seqlens
    )
    torch.cuda.synchronize()
    _check(out_nodelta, expected)
    _check(out_default, expected)
    diff = (out_nodelta.float() - out_default.float()).abs()
    bound = ATOL + RTOL * out_default.float().abs()
    assert int((diff > bound).sum().item()) == 0, (
        f"routes differ beyond the FP4 tolerance (max abs {diff.max().item():.4f})"
    )
    # The uncompensated route is the less accurate one by construction; the bound is the pinned ratio.
    ratio = _rel_l2(out_nodelta, expected) / _rel_l2(out_default, expected)
    assert 1.0 <= ratio <= REL_L2_RATIO_MAX, ratio


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_nodelta_preallocated_output_and_custom_scale() -> (
    None
):
    """Pre-allocated ``out`` is written in place and returned; a custom softmax scale is honoured;
    an all-empty stream returns the (empty) output."""
    device = torch.device("cuda:0")
    cu_seqlens = (0, 300)
    q, k, v = synthetic_inputs(300, 5, seed=2, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    result = minimax_h3_sm120_varlen_attention_nvfp4_nodelta(q, k, v, cu, out)
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    _check(out, fp32_oracle(q, k, v, cu_seqlens))
    scaled = minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
        q, k, v, cu, cu_seqlens_host=cu_seqlens, softmax_scale=0.02
    )
    torch.cuda.synchronize()
    assert not torch.equal(scaled, out), "softmax_scale had no effect"
    empty = torch.empty(0, 5, q.shape[-1], dtype=torch.bfloat16, device=device)
    cu0 = torch.tensor((0, 0), dtype=torch.int32, device=device)
    assert minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
        empty, empty, empty, cu0
    ).shape == (0, 5, q.shape[-1])


@requires_sm120
def test_minimax_h3_sm120_varlen_attention_nvfp4_nodelta_rejects_bad_inputs() -> None:
    """The variant validates its inputs exactly like the default route."""
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(300, 4, seed=3, device=device)
    cu = torch.tensor((0, 300), dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4_nodelta(q.float(), k, v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4_nodelta(q, k[:299], v, cu)
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4_nodelta(q, k, v, cu.to(torch.int64))
    with pytest.raises(ValueError):
        minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
            q, k, v, cu, cu_seqlens_host=(0, 299)
        )
