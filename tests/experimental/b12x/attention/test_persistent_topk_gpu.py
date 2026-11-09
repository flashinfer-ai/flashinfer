"""GPU regressions for the CUTLASS 4.6 persistent top-k flat launch ABI."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention.dsa_indexer import persistent_topk as persistent_topk_module
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

    with kernel_resolution_guard("persistent top-k graph must reuse the eager kernel"):
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


def _tied_logits(pattern: str, rows: int, generator: torch.Generator) -> torch.Tensor:
    if pattern == "all_tied":
        return torch.zeros((rows, _WIDTH), dtype=torch.float32)
    if pattern == "quantized":
        return torch.randint(0, 4, (rows, _WIDTH), generator=generator).to(
            torch.float32
        )
    return (torch.randn((rows, _WIDTH), generator=generator) * 4).half().float()


def _lowest_column_reference(
    logits: torch.Tensor, lengths: torch.Tensor, page_table: torch.Tensor
) -> torch.Tensor:
    columns = torch.arange(_WIDTH, dtype=torch.int64, device=logits.device)
    masked = torch.where(
        columns.unsqueeze(0) < lengths.unsqueeze(1),
        logits,
        torch.full_like(logits, float("-inf")),
    )
    order = torch.argsort(-masked, dim=1, stable=True)[:, :_TOPK]
    return torch.gather(page_table, 1, order)


@pytest.mark.parametrize("rows", (1, 3))
@pytest.mark.parametrize("pattern", ("all_tied", "quantized", "fp16_rounded"))
def test_persistent_topk_pivot_ties_select_lowest_columns_gpu(
    rows: int, pattern: str
) -> None:
    """Pivot ties resolve to the lowest logical columns, identically on every launch."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(91_000 + rows)
    logits = _tied_logits(pattern, rows, generator).to(device=device)
    lengths = (
        _WIDTH - 1 - torch.arange(rows, dtype=torch.int32, device=device) * 53
    ).to(torch.int32)
    logical = torch.arange(_WIDTH, dtype=torch.int32, device=device)
    page_table = torch.stack(
        tuple(
            torch.roll(logical, shifts=row * 31) + row * 100_000 for row in range(rows)
        )
    )

    columns = torch.arange(_WIDTH, device=device)
    masked = torch.where(
        columns.unsqueeze(0) < lengths.unsqueeze(1),
        logits,
        torch.full_like(logits, float("-inf")),
    )
    kth = torch.topk(masked, _TOPK, dim=1).values[:, -1:]
    slots_at_pivot = _TOPK - (masked > kth).sum(dim=1)
    ties_at_pivot = (masked == kth).sum(dim=1)
    if pattern == "fp16_rounded":
        assert bool((ties_at_pivot == slots_at_pivot).all().item())
    else:
        assert bool((ties_at_pivot > slots_at_pivot).all().item())

    expected = _lowest_column_reference(logits, lengths, page_table)
    expected_sets = [set(expected[row].tolist()) for row in range(rows)]
    first: torch.Tensor | None = None
    for _ in range(8):
        output = run_persistent_topk2048(
            logits, lengths, page_table_1=page_table, max_seq_len=_WIDTH, topk=_TOPK
        )
        torch.cuda.synchronize(device)
        assert bool((output >= 0).all().item())
        for row in range(rows):
            assert set(output[row].tolist()) == expected_sets[row]
        sorted_output = torch.sort(output, dim=1).values
        if first is None:
            first = sorted_output
        assert torch.equal(sorted_output, first)


# Score whose radix key has byte 2 == 0xFF, so a stale round-2 histogram lands in bucket 255.
_HIGH_BYTE2_SCORE = torch.tensor(0x3F80FF00, dtype=torch.int32).view(torch.float32)
_ROTATION_ROWS = (
    ("high_byte2_tied", _WIDTH - 1),
    ("short", _TOPK),
    ("all_tied", _WIDTH - 54),
    ("short", 37),
    ("high_byte2_tied", _WIDTH - 107),
    ("short", _TOPK),
    ("quantized", _WIDTH - 160),
    ("short", 300),
)


def _rotation_row(kind: str, generator: torch.Generator) -> torch.Tensor:
    if kind == "high_byte2_tied":
        return _HIGH_BYTE2_SCORE.expand(_WIDTH).clone()
    if kind == "quantized":
        return torch.randint(0, 4, (_WIDTH,), generator=generator).to(torch.float32)
    return torch.zeros((_WIDTH,), dtype=torch.float32)


def test_persistent_topk_one_group_rotation_with_skipped_rows_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One group walks short and tied long rows; the histogram rotation stays clean across skipped rows."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(92_000)
    rows = len(_ROTATION_ROWS)
    logits = torch.stack(
        tuple(_rotation_row(kind, generator) for kind, _ in _ROTATION_ROWS)
    ).to(device=device)
    lengths = torch.tensor(
        [length for _, length in _ROTATION_ROWS], dtype=torch.int32, device=device
    )
    logical = torch.arange(_WIDTH, dtype=torch.int32, device=device)
    page_table = torch.stack(
        tuple(
            torch.roll(logical, shifts=row * 31) + row * 100_000 for row in range(rows)
        )
    )

    real_resolve = persistent_topk_module._resolve_launch_config

    def one_group(num_rows: int, stride: int, dev: torch.device):
        launch = real_resolve(num_rows, stride, dev)
        return dataclasses.replace(
            launch, num_groups=1, total_ctas=launch.ctas_per_group
        )

    monkeypatch.setattr(persistent_topk_module, "_resolve_launch_config", one_group)
    assert persistent_topk2048_scratch_nbytes(rows, _WIDTH, device=device) == (
        persistent_topk_module._STATE_WORDS * 4
    )

    columns = torch.arange(_WIDTH, device=device)
    masked = torch.where(
        columns.unsqueeze(0) < lengths.unsqueeze(1),
        logits,
        torch.full_like(logits, float("-inf")),
    )
    long_rows = lengths > _TOPK
    kth = torch.topk(masked, _TOPK, dim=1).values[:, -1:]
    slots_at_pivot = _TOPK - (masked > kth).sum(dim=1)
    ties_at_pivot = (masked == kth).sum(dim=1)
    assert bool((ties_at_pivot[long_rows] > slots_at_pivot[long_rows]).all().item())
    assert int((~long_rows).sum().item()) >= 3

    expected = _lowest_column_reference(logits, lengths, page_table)
    for row in range(rows):
        length = int(lengths[row].item())
        if length <= _TOPK:
            expected[row, :length] = page_table[row, :length]
            expected[row, length:] = -1
    expected_sorted = torch.sort(expected, dim=1).values

    output = torch.full((rows, _TOPK), -1, dtype=torch.int32, device=device)
    first: torch.Tensor | None = None
    for _ in range(4):
        output.fill_(-1)
        result = run_persistent_topk2048(
            logits,
            lengths,
            page_table_1=page_table,
            output_indices=output,
            max_seq_len=_WIDTH,
            topk=_TOPK,
        )
        torch.cuda.synchronize(device)
        assert result.data_ptr() == output.data_ptr()
        sorted_output = torch.sort(output, dim=1).values
        unwritten = ((output < 0).any(dim=1) & long_rows).nonzero().flatten().tolist()
        assert unwritten == [], f"long rows left unwritten: {unwritten}"
        bad_rows = [
            row
            for row in range(rows)
            if not torch.equal(sorted_output[row], expected_sorted[row])
        ]
        assert bad_rows == [], f"rows differing from the lowest-column set: {bad_rows}"
        if first is None:
            first = sorted_output
        assert torch.equal(sorted_output, first)
