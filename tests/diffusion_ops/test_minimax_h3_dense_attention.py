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
import pytest
import torch

from flashinfer.diffusion_ops import minimax_h3_dense_attention
from flashinfer.diffusion_ops.cake_minimax_h3_dense_attention import (
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    MINIMAX_H3_QUERY_SCALE_BF16,
    MINIMAX_H3_WIDTH,
)
from flashinfer.utils import get_compute_capability

# Package tolerance of the MiniMax-H3 attention repro (fixed atol/rtol on the full output).
PACKAGE_ATOL = 0.01
PACKAGE_RTOL = 0.02
# Tail lengths around the 64-token tile, plus a multi-tile shape with a partial last tile.
TOKENS = [1, 63, 64, 65, 128, 257, 1023, 4097]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = get_compute_capability(torch.device("cuda:0"))
    return major in (9, 10, 12)


def synthetic_inputs(tokens: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        torch.randn(tokens, MINIMAX_H3_WIDTH, generator=generator, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )


def scaled_query(q: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(
        MINIMAX_H3_QUERY_SCALE_BF16, dtype=torch.bfloat16, device=q.device
    )
    return q * scale  # BF16 x BF16 -> BF16 (rounded product), as in the model graph


def fp32_oracle(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_chunk: int = 1024
) -> torch.Tensor:
    """Exact-softmax FP32 attention over the BF16 operands, per head, TF32 disabled."""

    tokens = q.shape[0]
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qh = (
            scaled_query(q)
            .view(tokens, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM)
            .transpose(0, 1)
            .float()
        )
        kh = (
            k.view(tokens, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM)
            .transpose(0, 1)
            .float()
        )
        vh = (
            v.view(tokens, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM)
            .transpose(0, 1)
            .float()
        )
        out = torch.empty_like(qh)
        for start in range(0, tokens, query_chunk):
            stop = min(start + query_chunk, tokens)
            scores = qh[:, start:stop] @ kh.transpose(1, 2)
            out[:, start:stop] = torch.softmax(scores, dim=-1) @ vh
        return out.transpose(0, 1).reshape(tokens, MINIMAX_H3_WIDTH).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(
    not _supported(),
    reason="requires a CUDA GPU with compute capability 9.0, 10.x or 12.x",
)
@pytest.mark.parametrize("tokens", TOKENS)
def test_minimax_h3_dense_attention_matches_fp32_oracle(tokens: int) -> None:
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(tokens, seed=0, device=device)
    out = minimax_h3_dense_attention(q, k, v)
    torch.cuda.synchronize()
    expected = fp32_oracle(q, k, v)
    assert out.shape == (tokens, MINIMAX_H3_WIDTH) and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        out.float(), expected.float(), atol=PACKAGE_ATOL, rtol=PACKAGE_RTOL
    )


@pytest.mark.skipif(
    not _supported(),
    reason="requires a CUDA GPU with compute capability 9.0, 10.x or 12.x",
)
def test_minimax_h3_dense_attention_preallocated_output() -> None:
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(300, seed=1, device=device)
    expected = fp32_oracle(q, k, v)
    out = torch.empty_like(q)
    returned = minimax_h3_dense_attention(q, k, v, out)
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not _supported(),
    reason="requires a CUDA GPU with compute capability 9.0, 10.x or 12.x",
)
def test_minimax_h3_dense_attention_repeated_launches_reuse_workspace() -> None:
    """Back-to-back launches of different sizes share the per-device workspace.

    The last-wave items are split across idle CTAs and merged in-kernel through the
    workspace; the kernel rewinds its arrival counters, so the second launch of every
    size must match the first.
    """

    device = torch.device("cuda:0")
    for tokens in (1023, 257, 1023, 4097):
        q, k, v = synthetic_inputs(tokens, seed=2, device=device)
        first = minimax_h3_dense_attention(q, k, v)
        second = minimax_h3_dense_attention(q, k, v)
        torch.cuda.synchronize()
        expected = fp32_oracle(q, k, v)
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            first.float(), expected.float(), atol=1e-2, rtol=1e-2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_dense_attention_rejects_bad_inputs() -> None:
    device = torch.device("cuda:0")
    q, k, v = synthetic_inputs(64, seed=2, device=device)
    with pytest.raises(ValueError):
        minimax_h3_dense_attention(q.float(), k, v)
    with pytest.raises(ValueError):
        minimax_h3_dense_attention(q[:, :64], k, v)
    with pytest.raises(ValueError):
        minimax_h3_dense_attention(q.t().contiguous().t(), k, v)
