from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from b12x.attention.qsa import _contract as contract
from b12x.attention.qsa._contract import Caps
from b12x.attention.qsa._dcp import (
    global_to_local,
    local_length,
    local_to_global,
    validate_geometry,
)

from ..conftest import require_b12x as require_sm120


@pytest.mark.parametrize("size", [2, 4])
def test_qsa_dcp_group_ownership_round_trips(size: int) -> None:
    for global_group in range(257):
        owner, local_group = global_to_local(
            global_group,
            size=size,
            interleave=1,
        )
        assert local_to_global(
            local_group,
            size=size,
            rank=owner,
            interleave=1,
        ) == global_group


def test_qsa_dcp4_packs_complete_c4_groups() -> None:
    global_tokens = 262_144
    local_tokens = [
        local_length(global_tokens, size=4, rank=rank, interleave=4)
        for rank in range(4)
    ]
    local_groups = [
        local_length(global_tokens // 4, size=4, rank=rank, interleave=1)
        for rank in range(4)
    ]
    assert local_tokens == [65_536] * 4
    assert local_groups == [16_384] * 4
    assert [tokens // 4 for tokens in local_tokens] == local_groups


def test_qsa_dcp_rejects_split_compression_groups() -> None:
    with pytest.raises(ValueError, match="divisible by compress_ratio"):
        validate_geometry(size=4, rank=0, token_interleave=1, compress_ratio=4)


def test_qsa_dcp1_preserves_unsharded_geometry() -> None:
    caps = Caps(
        device="cuda:0",
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=64,
        num_main_cache_pages=1,
        num_compressed_cache_pages=1,
        main_page_size=64,
        compressed_page_size=16,
    )

    assert caps.cp_kv_cache_interleave_size == 1
    assert caps.max_local_seq_len == caps.max_seq_len
    assert caps.max_local_groups == caps.max_global_groups


def test_qsa_dcp_handles_partial_final_round() -> None:
    assert [
        local_length(18, size=4, rank=rank, interleave=4) for rank in range(4)
    ] == [6, 4, 4, 4]


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_qsa_dcp_rejects_plan_without_local_compressed_capacity(rank: int) -> None:
    # Live empty ranks are valid; this checks a plan that can never score a
    # complete local group, for which the score workspace has no chunk width.
    with pytest.raises(ValueError, match="at least one local compressed group"):
        caps = Caps(
            device="cuda:0", max_batch=1, max_raw_state_slots=1, max_q_rows=1,
            max_seq_len=4, num_main_cache_pages=1, num_compressed_cache_pages=1,
            main_page_size=64, compressed_page_size=16,
            dcp_size=4, dcp_rank=rank, cp_kv_cache_interleave_size=4,
        )
        contract._scratch_layout(caps)


def _boundary_binding(dcp_size: int) -> contract.Binding:
    """CPU metadata fixture whose native launch must be rejected by the API."""
    caps = SimpleNamespace(
        device=torch.device("cpu"), dcp_size=dcp_size, max_q_rows=2,
        max_batch=2, q_heads=24, head_dim=256, group_budget=512,
        dcp_rank=0, cp_kv_cache_interleave_size=4,
        max_speculative_tokens=3,
    )
    programs = object.__new__(contract.QsaPrograms)
    for field in fields(contract.QsaPrograms):
        object.__setattr__(programs, field.name, None if field.name == "score" else {})
    values = {field.name: None for field in fields(contract.Binding)}
    values.update(
        state=SimpleNamespace(
            caps=caps, programs=programs,
            _layout=SimpleNamespace(draft_positions_offset_bytes=0),
        ),
        plan=object.__new__(contract.Plan),
        output=torch.empty(2, 24, 256, dtype=torch.bfloat16),
        output_lse=torch.full((2, 24), torch.nan),
        scratch=torch.empty(0, dtype=torch.uint8),
        selected_positions=torch.empty(2, 2051, dtype=torch.int32),
        draft_selection=SimpleNamespace(
            _storage=torch.empty(0, dtype=torch.uint8),
            plan=SimpleNamespace(max_source_rows=2, selection_width=2051),
        ),
    )
    return contract.Binding(**values)


@pytest.mark.parametrize("position_rows", [0, 2, 3])
def test_qsa_dcp_attend_rejects_position_rows_before_expansion(
    monkeypatch: pytest.MonkeyPatch, position_rows: int,
) -> None:
    from b12x.attention.qsa import _kernels

    def unexpected_launch(**kwargs):
        pytest.fail("invalid query positions reached global selection expansion")

    monkeypatch.setattr(_kernels, "launch_expand_global_selected_groups", unexpected_launch)
    binding = _boundary_binding(dcp_size=2)
    selection = contract.LocalSelection(
        group_ids=torch.zeros(1, 512, dtype=torch.int32),
        scores=torch.empty(1, 512),
    )
    with pytest.raises(ValueError, match="query_positions must have shape"):
        contract.attend(
            binding, query=torch.empty(1, 24, 256, dtype=torch.bfloat16),
            request_ids=torch.zeros(1, dtype=torch.int32),
            query_positions=torch.zeros(position_rows, dtype=torch.int64),
            selection=selection,
        )


def test_qsa_dcp_attend_reuse_rejects_unsharded_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.attention.qsa import _draft_selection

    monkeypatch.setattr(_draft_selection, "validate_buffers", lambda *args: None)
    monkeypatch.setattr(_draft_selection, "prepare_selection", lambda *args: None)
    monkeypatch.setattr(contract, "_run_attention", lambda binding, **kwargs: binding.output[:1])
    with pytest.raises(ValueError, match="context-parallel QSA"):
        contract.attend_reuse(
            _boundary_binding(dcp_size=1),
            query=torch.empty(1, 24, 256, dtype=torch.bfloat16),
            request_ids=torch.zeros(1, dtype=torch.int32),
            query_positions=torch.zeros(1, dtype=torch.int64),
            reuse=contract.DraftSelectionReuse(torch.zeros(2, dtype=torch.int32)),
        )


@pytest.mark.parametrize("global_tokens", [4, 64])
def test_qsa_dcp4_shards_and_lse_merge_match_global_attention(
    global_tokens: int,
) -> None:
    from b12x.attention.qsa._kernels import launch_expand_global_selected_groups
    from b12x.attention.qsa._sparse_gqa import launch_sparse_paged_gqa

    device = require_sm120()
    torch.manual_seed(20260919)
    dcp_size = 4
    interleave = 4
    q_heads = 24
    kv_heads = 2
    head_dim = 256
    selection_width = 2051
    splits = 64
    scale = head_dim**-0.5

    query = torch.randn(
        1, q_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    global_keys = torch.randn(
        1, global_tokens, kv_heads, head_dim, device=device
    ).to(torch.float8_e4m3fn)
    global_values = torch.randn(
        1, global_tokens, kv_heads, head_dim, device=device
    ).to(torch.float8_e4m3fn)
    request_ids = torch.zeros(1, dtype=torch.int32, device=device)
    query_positions = torch.full(
        (1,), global_tokens - 1, dtype=torch.int64, device=device
    )
    descale = torch.ones(1, dtype=torch.float32, device=device)

    def attend(
        keys: torch.Tensor,
        values: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(query)
        output_lse = torch.empty(1, q_heads, dtype=torch.float32, device=device)
        partial_output = torch.empty(
            1, splits, q_heads, head_dim, dtype=torch.float32, device=device
        )
        partial_lse = torch.empty(
            1, splits, q_heads, dtype=torch.float32, device=device
        )
        launch_sparse_paged_gqa(
            query=query,
            key_cache=keys,
            value_cache=values,
            k_descale=descale,
            v_descale=descale,
            block_table=torch.zeros(1, 1, dtype=torch.int32, device=device),
            request_ids=request_ids,
            selected_positions=selected,
            query_positions=query_positions,
            output=output,
            output_lse=output_lse,
            partial_output=partial_output,
            partial_lse=partial_lse,
            softmax_scale=scale,
            block_n=16,
            splits=splits,
        )
        return output, output_lse

    global_selected = torch.full(
        (1, selection_width), -1, dtype=torch.int32, device=device
    )
    global_selected[0, :global_tokens] = torch.arange(
        global_tokens, dtype=torch.int32, device=device
    )
    expected, expected_lse = attend(global_keys, global_values, global_selected)

    topk_groups = torch.full((1, 512), -1, dtype=torch.int32, device=device)
    topk_groups[0, : global_tokens // 4] = torch.arange(
        global_tokens // 4, dtype=torch.int32, device=device
    )
    local_outputs = []
    local_lses = []
    for rank in range(dcp_size):
        local_tokens = local_length(
            global_tokens,
            size=dcp_size,
            rank=rank,
            interleave=interleave,
        )
        local_keys = torch.empty(
            1,
            max(local_tokens, 4),
            kv_heads,
            head_dim,
            dtype=global_keys.dtype,
            device=device,
        )
        local_values = torch.empty_like(local_keys)
        for global_position in range(global_tokens):
            owner, local_position = global_to_local(
                global_position,
                size=dcp_size,
                interleave=interleave,
            )
            if owner == rank:
                local_keys[0, local_position].copy_(global_keys[0, global_position])
                local_values[0, local_position].copy_(
                    global_values[0, global_position]
                )

        selected = torch.empty(
            1, selection_width, dtype=torch.int32, device=device
        )
        launch_expand_global_selected_groups(
            topk_group_ids=topk_groups,
            query_positions=query_positions,
            selected_positions=selected,
            caps=SimpleNamespace(
                group_budget=512,
                compress_ratio=4,
                selection_width=selection_width,
                dcp_size=dcp_size,
                dcp_rank=rank,
                cp_kv_cache_interleave_size=interleave,
            ),
        )
        output, output_lse = attend(local_keys, local_values, selected)
        local_outputs.append(output.float())
        local_lses.append(output_lse)

    stacked_lse = torch.stack(local_lses)
    merged_lse = torch.logsumexp(stacked_lse, dim=0)
    weights = torch.exp(stacked_lse - merged_lse)
    merged = (
        torch.stack(local_outputs) * weights.unsqueeze(-1)
    ).sum(dim=0)

    torch.testing.assert_close(merged_lse, expected_lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(merged, expected.float(), rtol=1e-2, atol=1e-2)
