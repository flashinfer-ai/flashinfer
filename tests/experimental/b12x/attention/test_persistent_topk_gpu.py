"""GPU regressions for the CUTLASS 4.6 persistent top-k flat launch ABI."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention.dsa_indexer._impl import clear_indexer_caches
from b12x.attention.dsa_indexer.persistent_topk import (
    persistent_topk2048_scratch_nbytes,
    run_persistent_topk2048,
)
from b12x._lib.compiler import compile_cache_info


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for persistent top-k coverage",
)

_WIDTH = 33_792
_TOPK = 512
_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


def _allocator_counters(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {name: int(stats.get(name, 0)) for name in _ALLOCATOR_COUNTERS}


def _expected_indices(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    columns = torch.arange(_WIDTH, dtype=torch.int64, device=logits.device)
    masked = torch.where(
        columns.unsqueeze(0) < lengths.unsqueeze(1),
        logits,
        torch.full_like(logits, float("-inf")),
    )
    selected = torch.topk(masked, _TOPK, dim=1, largest=True, sorted=False)
    return selected.values, torch.gather(page_table, 1, selected.indices)


def _assert_exact(
    output: torch.Tensor,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
) -> None:
    expected_values, expected_indices = _expected_indices(
        logits,
        lengths,
        page_table,
    )
    assert bool(torch.isfinite(expected_values).all().item())
    assert bool((expected_values != 0).any().item())
    assert bool((output >= 0).all().item())
    for row in range(int(output.shape[0])):
        assert set(output[row].tolist()) == set(expected_indices[row].tolist())


@pytest.mark.parametrize("rows", (1, 2, 3, 4))
def test_persistent_topk_multirow_eager_and_changed_graph_gpu(rows: int) -> None:
    """Rows stay isolated in eager launch and changed-input graph replay."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(84_000 + rows)
    scenario_logits = tuple(
        torch.randn((2, rows, _WIDTH), generator=generator, dtype=torch.float32)
        .to(device=device)
        .unbind(0)
    )
    row_offsets = torch.arange(rows, dtype=torch.int32, device=device)
    scenario_lengths = (
        _WIDTH - 1 - row_offsets * 53,
        _WIDTH - 17 - torch.flip(row_offsets, dims=(0,)) * 67,
    )
    logical = torch.arange(_WIDTH, dtype=torch.int32, device=device)
    scenario_tables = tuple(
        torch.stack(
            tuple(
                torch.roll(logical, shifts=scenario * 97 + row * 31)
                + scenario * 1_000_000
                + row * 100_000
                for row in range(rows)
            )
        )
        for scenario in range(2)
    )

    logits = torch.empty_like(scenario_logits[0])
    lengths = torch.empty_like(scenario_lengths[0])
    page_table = torch.empty_like(scenario_tables[0])
    output = torch.empty((rows, _TOPK), dtype=torch.int32, device=device)
    scratch_nbytes = persistent_topk2048_scratch_nbytes(
        rows,
        _WIDTH,
        device=device,
    )
    scratch = torch.empty(
        (scratch_nbytes // torch.empty((), dtype=torch.int32).element_size(),),
        dtype=torch.int32,
        device=device,
    )
    stable_ptrs = (output.data_ptr(), scratch.data_ptr())

    def install(index: int) -> None:
        logits.copy_(scenario_logits[index])
        lengths.copy_(scenario_lengths[index])
        page_table.copy_(scenario_tables[index])

    def run() -> torch.Tensor:
        return run_persistent_topk2048(
            logits,
            lengths,
            page_table_1=page_table,
            output_indices=output,
            scratch=scratch,
            max_seq_len=_WIDTH,
            topk=_TOPK,
        )

    clear_indexer_caches()
    install(0)
    eager = run()
    torch.cuda.synchronize(device)
    assert eager.data_ptr() == output.data_ptr()
    _assert_exact(output, logits, lengths, page_table)
    warm_compile_misses = int(compile_cache_info()["compile_misses"])

    with kernel_resolution_guard('persistent top-k graph must reuse the eager kernel'):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
    assert captured.data_ptr() == output.data_ptr()
    assert int(compile_cache_info()["compile_misses"]) == warm_compile_misses

    install(1)
    output.fill_(-1)
    before = _allocator_counters(device)
    graph.replay()
    torch.cuda.synchronize(device)
    assert _allocator_counters(device) == before
    assert (output.data_ptr(), scratch.data_ptr()) == stable_ptrs
    _assert_exact(output, logits, lengths, page_table)
