"""Correctness tests for adaptive sparse block-mask selection."""

import math

import pytest
import torch

from flashinfer.sparse import adaptive_sparse_block_mask


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _reference(
    logits: torch.Tensor,
    q_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    *,
    block_size: int,
    alpha: float,
    initial_blocks: int,
    window_size: int,
    medium_rate: float = 0.2,
    medium_bias: int = 30,
    large_rate: float = 0.1,
    large_bias: int = 30,
) -> torch.Tensor:
    output = torch.zeros_like(logits, dtype=torch.bool)
    for batch in range(logits.shape[0]):
        q_blocks = min(
            math.ceil(max(int(q_lens[batch]), 0) / block_size), logits.shape[2]
        )
        k_blocks = min(
            math.ceil(max(int(kv_lens[batch]), 0) / block_size), logits.shape[3]
        )
        prompt_blocks = max(1, math.ceil(max(int(prompt_lens[batch]), 0) / block_size))
        offset = math.ceil(
            max(int(kv_lens[batch]) - int(q_lens[batch]), 0) / block_size
        )
        if prompt_blocks < 56:
            max_budget = prompt_blocks
        elif prompt_blocks < 160:
            max_budget = int(prompt_blocks * medium_rate) + medium_bias
        else:
            max_budget = int(prompt_blocks * large_rate) + large_bias
        max_budget = min(max(max_budget, 1), prompt_blocks)

        for head in range(logits.shape[1]):
            for row in range(q_blocks):
                decay_len = prompt_blocks - max_budget
                if row + offset < max_budget or decay_len <= 1:
                    budget = max_budget
                else:
                    t = (row + offset - max_budget) / (decay_len - 1)
                    budget = math.floor(
                        max_budget + t * (max_budget * alpha - max_budget)
                    )
                    budget = min(max(budget, 1), max_budget)

                values = logits[batch, head, row, :k_blocks].float()
                finite = torch.isfinite(values)
                selected = torch.zeros(k_blocks, dtype=torch.bool, device=logits.device)
                finite_count = int(finite.sum())
                if budget >= finite_count:
                    selected |= finite
                elif budget > 0:
                    threshold = torch.topk(values[finite], budget).values[-1]
                    selected |= finite & (values >= threshold)
                diagonal = min(row + offset, k_blocks - 1)
                selected[: min(initial_blocks, k_blocks)] = True
                selected[max(0, diagonal - window_size + 1) : diagonal + 1] = True
                output[batch, head, row, :k_blocks] = selected
    return output


@pytest.mark.parametrize("blocks", [32, 192, 1100, 2100])
def test_adaptive_sparse_block_mask_matches_reference(blocks: int) -> None:
    torch.manual_seed(7 + blocks)
    block_size = 128
    batch, heads = 2, 2
    q_blocks = min(blocks, 24)
    logits = torch.randn(
        batch, heads, q_blocks, blocks, device="cuda", dtype=torch.bfloat16
    )
    q_lens = torch.tensor(
        [q_blocks * block_size, (q_blocks - 1) * block_size + 1],
        device="cuda",
        dtype=torch.int32,
    )
    kv_lens = torch.tensor(
        [blocks * block_size, (blocks - 1) * block_size + 1],
        device="cuda",
        dtype=torch.int32,
    )
    prompt_lens = kv_lens.clone()
    logits[0, 0, 0, -1] = -math.inf

    actual = adaptive_sparse_block_mask(
        logits,
        q_lens,
        kv_lens,
        prompt_lens,
        block_size=block_size,
        alpha=0.5,
    )
    expected = _reference(
        logits,
        q_lens,
        kv_lens,
        prompt_lens,
        block_size=block_size,
        alpha=0.5,
        initial_blocks=4,
        window_size=4,
    )
    torch.testing.assert_close(actual, expected)


def test_adaptive_sparse_block_mask_reuses_output() -> None:
    logits = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([512], device="cuda", dtype=torch.int32)
    out = torch.ones_like(logits, dtype=torch.bool)
    result = adaptive_sparse_block_mask(logits, lengths, lengths, lengths, out=out)
    assert result.data_ptr() == out.data_ptr()
    assert (~out).any()


def test_adaptive_sparse_block_mask_is_experimental() -> None:
    assert adaptive_sparse_block_mask.is_experimental
