"""Prepared fixed-split QSA: reference arithmetic and CUDA-graph capacity invariance."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention import qsa
from b12x.attention.qsa.reference import sparse_paged_gqa_reference

from ..conftest import require_b12x
from .test_qsa_contract import (
    _allocate_binding,
    _caps,
    _cuda_graph,
    _draft_selection_state,
    _dynamic_inputs,
    _rebind,
    _retain_qsa_test_resources,
)


# Match test_qsa_production_page_boundary_after_speculative_rollback_graph_replay.
_REFERENCE_RTOL = 0.0
_REFERENCE_ATOL = 2e-2


def _random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator).to(torch.bfloat16)


def _binding(device, rows: int, splits: int, *, dcp_rank: int | None = None):
    distributed = dcp_rank is not None
    caps = _caps(
        device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=rows,
        max_seq_len=8192 if distributed else 4096,
        num_main_cache_pages=64,
        num_compressed_cache_pages=64,
        main_page_size=64,
        compressed_page_size=16,
        q_heads=24,
        kv_heads=2,
        index_heads=4,
        attention_splits=splits,
        max_speculative_tokens=3,
        dcp_size=2 if distributed else 1,
        dcp_rank=0 if dcp_rank is None else dcp_rank,
        cp_kv_cache_interleave_size=4,
    )
    binding = _allocate_binding(caps)
    for name, seed in (
        ("main_k_cache", 911),
        ("main_v_cache", 912),
        ("compressed_k_cache", 913),
    ):
        tensor = getattr(binding, name)
        tensor.copy_(_random(tuple(tensor.shape), seed))
    for table in (binding.main_block_table, binding.compressed_block_table):
        table[0].copy_(torch.arange(table.shape[1] - 1, -1, -1, device=device))
    binding.raw_k_ring.zero_()
    binding.raw_logical_positions.fill_(-1)
    binding.raw_rope_positions.fill_(-1)
    return binding


def _persistent(binding) -> tuple[torch.Tensor, ...]:
    return (
        binding.compressed_k_cache,
        binding.raw_k_ring,
        binding.raw_logical_positions,
        binding.raw_rope_positions,
        binding.raw_interval_start_positions,
    )


def _snapshot(tensors):
    snapshots = tuple(tensor.clone() for tensor in tensors)

    def restore():
        for tensor, saved in zip(tensors, snapshots, strict=True):
            tensor.copy_(saved)

    return snapshots, restore


def _inputs(binding, start: int, live: int):
    rows = binding.state.caps.max_q_rows
    dynamic = _dynamic_inputs(
        binding,
        positions=tuple(range(start, start + live)) + (-1,) * (rows - live),
        request_ids=(0,) * live + (-1,) * (rows - live),
        is_prefilling=(True,),
    )
    for name, shape, seed in (
        ("query", (128, 24, 256), 921),
        ("index_query", (128, 4, 128), 922),
        ("raw_index_key", (128, 128), 923),
    ):
        dynamic[name].copy_(_random(shape, seed)[:rows])
    return dynamic


def _set_live(dynamic, start: int, live: int):
    rows = dynamic["query"].shape[0]
    device = dynamic["query"].device
    positions = torch.arange(start, start + rows, device=device, dtype=torch.int64)
    positions[live:] = -1
    dynamic["query_positions"].copy_(positions)
    dynamic["rope_positions"].copy_(positions[:, None])
    dynamic["request_ids"].fill_(-1)
    dynamic["request_ids"][:live].zero_()
    dynamic["sequence_lengths"].fill_(start + live)
    dynamic["query_start_loc"][1] = live


def _replay_without_allocations(graph, binding):
    device = binding.output.device
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (binding.output, binding.selected_positions, binding.scratch)
    )
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_stats(device)
    graph.replay()
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_stats(device)
    for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
        assert before[key] == after[key], key
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (binding.output, binding.selected_positions, binding.scratch)
    )


def _check_reference(binding, dynamic, output, live):
    selected = binding.selected_positions[: output.shape[0]]
    global_positions = selected.long()
    caps = binding.state.caps
    if caps.dcp_size > 1:
        global_positions = (
            global_positions // 4 * caps.dcp_size + caps.dcp_rank
        ) * 4 + global_positions % 4
    assert not torch.any(
        (selected >= 0) & (global_positions > dynamic["query_positions"][:, None])
    )
    expected = sparse_paged_gqa_reference(
        dynamic["query"],
        binding.main_k_cache,
        binding.main_v_cache,
        binding.main_block_table,
        dynamic["request_ids"],
        selected,
        dynamic["query_positions"],
    )
    torch.testing.assert_close(
        output, expected, rtol=_REFERENCE_RTOL, atol=_REFERENCE_ATOL
    )
    assert torch.isfinite(output).all()
    assert torch.all(output[live:] == 0)
    assert torch.all(selected[live:] == -1)


def _check_lse_reference(binding, dynamic, lse, row):
    positions = binding.selected_positions[row].long()
    positions = positions[positions >= 0]
    pages = binding.main_block_table[0, positions // 64].long()
    keys = binding.main_k_cache[pages, positions % 64]
    keys = keys.repeat_interleave(12, dim=1)
    scores = (
        torch.einsum("hd,khd->hk", dynamic["query"][row].float(), keys.float()) / 16
    )
    expected = torch.logsumexp(scores, dim=-1)
    # Match test_qsa_dcp4_shards_and_lse_merge_match_global_attention.
    torch.testing.assert_close(lse[row], expected, rtol=1e-5, atol=1e-5)


def _capture(invoke: Callable, restore: Callable):
    invoke()
    restore()
    graph = _cuda_graph()
    with kernel_resolution_guard("fixed-split QSA uses only prepared programs"):
        with torch.cuda.graph(graph):
            result = invoke()
    restore()
    return graph, result


@pytest.mark.parametrize("splits", [1, 16, 64])
def test_fixed_splits_combined_capacity_and_changed_live_graph_replay(splits):
    device = require_b12x()
    results = {}
    for rows in (16, 65, 128):
        live = min(rows, 67)
        binding = _binding(device, rows, splits)
        dynamic = _inputs(binding, 2048, live)
        _, restore = _snapshot(_persistent(binding))
        graph, output = _capture(lambda: qsa.run(binding, **dynamic), restore)
        for variant, live_rows in enumerate((live, live - 1)):
            _set_live(dynamic, 2048, live_rows)
            dynamic["query"].copy_(
                (_random((128, 24, 256), 921).float() * (1 - variant * 0.25)).to(
                    torch.bfloat16
                )[:rows]
            )
            repeated = None
            for _ in range(2):
                restore()
                binding.output.fill_(float("nan"))
                binding.selected_positions.fill_(-2)
                _replay_without_allocations(graph, binding)
                actual = output.cpu()
                if repeated is not None:
                    assert torch.equal(actual, repeated)
                repeated = actual
            _check_reference(binding, dynamic, output, live_rows)
            # Position 2050 spills three causal tail entries past the 2048 budget.
            assert int((binding.selected_positions[2] >= 0).sum()) == 2051
            assert binding.selected_positions[2, -3:].tolist() == [2048, 2049, 2050]
            assert int((binding.selected_positions[6] >= 0).sum()) == 2051
            selected = binding.selected_positions.cpu()
            key = variant
            if key in results:
                prior_output, prior_selected, prior_live = results[key]
                common = min(prior_live, live_rows)
                assert torch.equal(actual[:common], prior_output[:common])
                assert torch.equal(selected[:common], prior_selected[:common])
            results[key] = (actual, selected, live_rows)


@pytest.mark.parametrize("splits", [1, 16, 64])
def test_fixed_splits_dcp_select_attend_capacity_and_lse(splits):
    """One rank's public select/attend contract; no distributed merge is simulated."""
    device = require_b12x()
    results = {}
    for rows in (65, 128):
        binding = _binding(device, rows, splits, dcp_rank=1)
        live = min(rows, 67)
        dynamic = _inputs(binding, 4096, live)
        _, restore = _snapshot(_persistent(binding))
        global_ids = torch.empty(
            (rows, binding.state.caps.group_budget), device=device, dtype=torch.int32
        )

        def invoke():
            local = qsa.select(binding, **dynamic)
            # With interleave=4 and rank=1, local group g belongs to global 2*g+1.
            torch.mul(local.group_ids, 2, out=global_ids)
            global_ids.add_(1)
            return qsa.attend(
                binding,
                query=dynamic["query"],
                request_ids=dynamic["request_ids"],
                query_positions=dynamic["query_positions"],
                selection=qsa.LocalSelection(group_ids=global_ids, scores=local.scores),
            )

        graph, (output, lse) = _capture(invoke, restore)
        for variant, live_rows in enumerate((64, live)):
            _set_live(dynamic, 4096, live_rows)
            for _ in range(2):
                restore()
                binding.output.fill_(float("nan"))
                binding.output_lse.fill_(float("nan"))
                _replay_without_allocations(graph, binding)
            _check_reference(binding, dynamic, output, live_rows)
            assert torch.isfinite(lse[:live_rows]).all()
            assert torch.isneginf(lse[live_rows:]).all()
            for row in (0, 64):
                if row < live_rows:
                    _check_lse_reference(binding, dynamic, lse, row)
            actual = (output.cpu(), lse.cpu(), binding.selected_positions.cpu())
            if variant in results:
                previous_values, previous_live = results[variant]
                common = min(live_rows, previous_live)
                for value, previous in zip(actual, previous_values, strict=True):
                    assert torch.equal(value[:common], previous[:common])
            results[variant] = (actual, live_rows)
        # Row 64 belongs to the second attention chunk and must receive its own LSE.
        assert torch.isfinite(lse[64]).all()


@pytest.mark.parametrize("splits", [1, 16, 64])
def test_fixed_splits_draft_reuse_causal_tail_and_invalid_sources(splits):
    device = require_b12x()
    binding = _binding(device, 4, splits)
    state = _draft_selection_state(binding)
    binding = _rebind(binding, draft_selection=state)
    dynamic = _inputs(binding, 2048, 3)
    qsa.run(binding, **dynamic)
    assert int(state.num_source_rows) == 4
    assert int(state.logical_positions[2]) == 2050
    source = torch.tensor([2], device=device, dtype=torch.int64)
    query = dynamic["query"][:1].clone()
    requests = torch.zeros(1, device=device, dtype=torch.int64)
    positions = torch.tensor([2051], device=device, dtype=torch.int64)
    persistent = _persistent(binding) + (
        state.selected_positions,
        state.logical_positions,
        state.num_source_rows,
    )
    snapshots, _ = _snapshot(persistent)
    anchor_selection = state.selected_positions[2].clone()

    def invoke():
        return qsa.run(
            binding,
            query=query,
            request_ids=requests,
            query_positions=positions,
            reuse=qsa.DraftSelectionReuse(source),
        )

    graph, output = _capture(invoke, lambda: None)
    for position in (2051, 2053):
        positions.fill_(position)
        for _ in range(2):
            _replay_without_allocations(graph, binding)
        selected = torch.full(
            (1, binding.state.caps.selection_width + 3),
            -1,
            device=device,
            dtype=torch.int32,
        )
        selected[0, : anchor_selection.numel()].copy_(anchor_selection)
        tail = torch.arange(2051, position + 1, device=device, dtype=torch.int32)
        selected[
            0, anchor_selection.numel() : anchor_selection.numel() + tail.numel()
        ] = tail
        expected = sparse_paged_gqa_reference(
            query,
            binding.main_k_cache,
            binding.main_v_cache,
            binding.main_block_table,
            requests,
            selected,
            positions,
        )
        torch.testing.assert_close(
            output, expected, rtol=_REFERENCE_RTOL, atol=_REFERENCE_ATOL
        )
        assert torch.equal(binding._draft_work_positions[:1], selected)
    for invalid_source, invalid_request, invalid_position in (
        (3, 0, 2051),
        (4, 0, 2051),
        (2, -1, 2051),
        (2, 1, 2051),
        (2, 0, 2054),
    ):
        source.fill_(invalid_source)
        requests.fill_(invalid_request)
        positions.fill_(invalid_position)
        _replay_without_allocations(graph, binding)
        assert torch.all(output == 0)
        assert torch.all(binding._draft_work_positions[:1] == -1)
    for tensor, saved in zip(persistent, snapshots, strict=True):
        assert torch.equal(tensor, saved)
