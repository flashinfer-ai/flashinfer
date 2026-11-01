"""Determinism tests for the DSA indexer top-k selection.

The 32K canary flake (~27% degenerate repetition) is hypothesized to come from
nondeterministic top-k selection over tied attention scores: identical prompt,
varying read plan. These tests probe whether the selection kernels are stable
across repeated calls on input with ties at the top-k boundary.
"""

import torch
import pytest

torch_device = "cuda"


def _torch_topk_tied_nondeterministic() -> None:
    """torch.topk(sorted=False) on tied input can return different index sets."""
    # 64 candidates, all score 1.0 (total tie), topk=8.
    x = torch.full((64,), 1.0, device=torch_device)
    sets = set()
    for _ in range(20):
        idx = torch.topk(x, 8, sorted=False).indices
        sets.add(tuple(idx.cpu().tolist()))
    # With a deterministic selection there is exactly one set.
    assert len(sets) == 1, (
        f"torch.topk(sorted=False) is nondeterministic on ties: {len(sets)} distinct sets"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_topk_sorted_false_tie_determinism():
    _torch_topk_tied_nondeterministic()


def _run_tiled_topk_once(
    tile_logits, *, topk, block_q, block_k, lengths
) -> torch.Tensor:
    from b12x.attention.dsa_indexer.tiled_topk import run_tiled_topk

    values, indices = run_tiled_topk(
        tile_logits=tile_logits,
        k_start=None,
        lengths=lengths,
        topk=topk,
        block_q=block_q,
        block_k=block_k,
        zero_row_start=True,
        is_first=True,
    )
    return indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_run_tiled_topk_tied_determinism():
    """run_tiled_topk must return identical index sets across calls on tied input."""
    from b12x.attention.dsa_indexer.tiled_topk import run_tiled_topk

    # Kernel only supports topk in {512, 1024, 2048}; use the smallest with
    # enough candidates (1024) that the top-k boundary is a total tie.
    block_q, block_k, topk = 1, 4, 512
    num_k_tiles = 256
    # Layout: [num_q_tiles, num_k_tiles, block_q, block_k] flattened.
    # All scores equal -> every selection is a tie.
    tile_logits = torch.zeros(
        num_k_tiles * block_k, dtype=torch.float32, device=torch_device
    )
    lengths = torch.tensor(
        [num_k_tiles * block_k], dtype=torch.int32, device=torch_device
    )

    sets = set()
    for _ in range(20):
        idx = _run_tiled_topk_once(
            tile_logits, topk=topk, block_q=block_q, block_k=block_k, lengths=lengths
        )
        # idx shape [out_rows, topk]; row 0 is the single query row.
        sets.add(tuple(sorted(idx[0].cpu().tolist())))
    assert len(sets) == 1, (
        f"run_tiled_topk nondeterministic on ties: {len(sets)} distinct sets, "
        f"examples={list(sets)[:3]}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_run_row_topk_tied_determinism():
    """run_row_topk must return identical index sets across calls on tied input."""
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk

    topk = 512
    width = 1024
    # Single query row, all scores equal -> every selection is a tie.
    row_logits = torch.zeros(1, width, dtype=torch.float32, device=torch_device)
    lengths = torch.tensor([width], dtype=torch.int32, device=torch_device)

    sets = set()
    for _ in range(20):
        values, indices = run_row_topk(
            row_logits=row_logits,
            lengths=lengths,
            topk=topk,
        )
        # indices shape [rows, topk]; row 0 is the single query row.
        sets.add(tuple(sorted(indices[0].cpu().tolist())))
    assert len(sets) == 1, (
        f"run_row_topk nondeterministic on ties: {len(sets)} distinct sets, "
        f"examples={list(sets)[:3]}"
    )
