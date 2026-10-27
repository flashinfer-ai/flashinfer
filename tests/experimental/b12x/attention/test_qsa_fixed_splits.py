"""CPU contract checks for caller-selected QSA attention partitions."""

from dataclasses import fields, replace

import pytest
import torch

from b12x.attention import qsa
from b12x.attention.qsa import _contract as contract
from b12x.attention.qsa._tuning import QsaQuery, TUNING
from b12x.preparation import FrozenMapping


def _caps(**changes):
    values = dict(
        device="cuda:0",
        max_batch=4,
        max_raw_state_slots=4,
        max_q_rows=512,
        max_seq_len=4096,
        num_main_cache_pages=64,
        num_compressed_cache_pages=64,
        main_page_size=64,
        compressed_page_size=16,
    )
    values.update(changes)
    return qsa.Caps(**values)


@pytest.mark.parametrize("splits", [0, 1, 2, 4, 8, 16, 32, 64])
def test_split_policy_survives_query_round_trip(splits):
    caps = _caps(attention_splits=splits)
    query = contract._query_from_caps(caps, FrozenMapping())
    restored = QsaQuery(**query.to_dict())
    assert restored.attention_splits == splits
    assert contract._caps_from_query(restored, ordinal=0) == caps
    TUNING.validate_query(restored, None)
    assert TUNING.encode_query(restored)["attention_splits"] == splits


@pytest.mark.parametrize("value", [None, True, False, 1.0, "16"])
def test_split_policy_rejects_non_integer(value):
    with pytest.raises(TypeError, match="attention_splits"):
        _caps(attention_splits=value)
    query = contract._query_from_caps(_caps(), FrozenMapping())
    with pytest.raises(TypeError, match="attention_splits"):
        TUNING.validate_query(replace(query, attention_splits=value), None)


@pytest.mark.parametrize("value", [-1, 3, 63, 65, 128])
def test_split_policy_rejects_out_of_range_integer(value):
    with pytest.raises(ValueError, match="attention_splits"):
        _caps(attention_splits=value)
    query = contract._query_from_caps(_caps(), FrozenMapping())
    with pytest.raises(ValueError, match="attention_splits"):
        TUNING.validate_query(replace(query, attention_splits=value), None)


def test_split_policy_changes_plan_identity_without_changing_config_schema():
    from b12x.preparation.session import _declaration_key

    declarations = [qsa.plan(_caps(attention_splits=s)) for s in (0, 1, 16, 64)]
    assert len({_declaration_key(plan) for plan in declarations}) == 4
    assert TUNING.query_schema_version == 23
    assert TUNING.config_schema_version == 22
    assert len({TUNING.default_config(plan.query, None) for plan in declarations}) == 1


@pytest.mark.parametrize(
    "rows,splits,product",
    [
        (1, 64, 64),
        (4, 32, 128),
        (16, 16, 256),
        (64, 16, 1024),
        (128, 16, 2048),
        (129, 1, 2048),
        (512, 1, 2048),
    ],
)
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_default_policy_retains_partition_and_workspace(
    rows, splits, product, kv_heads
):
    if kv_heads == 2:
        splits = 1 if rows > 64 else splits
        product = min(product, 1024)
    caps = _caps(max_batch=1, max_q_rows=rows, kv_heads=kv_heads)
    assert contract._target_splits(caps, rows) == (16, splits)
    assert contract._scratch_layout(caps)[4:] == (product, rows)


@pytest.mark.parametrize("splits", [1, 16, 64])
@pytest.mark.parametrize("rows", [1, 2, 4, 16, 64, 65, 128, 512])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_fixed_policy_bounds_only_partial_scratch(rows, splits, kv_heads):
    caps = _caps(
        max_batch=1, max_q_rows=rows, attention_splits=splits, kv_heads=kv_heads
    )
    layout, score_chunk, score_width, score_chunks, product, work_rows = (
        contract._scratch_layout(caps)
    )
    default = contract._scratch_layout(replace(caps, attention_splits=0))
    assert contract._target_splits(caps, rows) == (16, splits)
    assert work_rows == rows
    assert (score_chunk, score_width, score_chunks) == default[1:4]
    assert product == min(rows, 128 if kv_heads == 1 else 64) * splits
    assert layout.partial_output_nbytes == product * 24 * 256 * 4
    assert layout.partial_lse_nbytes == product * 24 * 4
    for name in ("prepared_query_nbytes", "score_nbytes", "topk_nbytes"):
        assert getattr(layout, name) == getattr(default[0], name)


def _operands(caps, device="meta"):
    def tensor(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    rows = caps.max_q_rows
    layout, score_chunk, score_width, score_chunks, product, work_rows = (
        contract._scratch_layout(caps)
    )
    operands = dict(
        query=tensor(rows, caps.q_heads, caps.head_dim),
        index_query=tensor(rows, caps.index_heads, caps.index_head_dim),
        raw_index_key=tensor(rows, caps.index_head_dim),
        request_ids=tensor(rows, dtype=torch.int64),
        query_positions=tensor(rows, dtype=torch.int64),
        rope_positions=tensor(rows, 1, dtype=torch.int64),
        sequence_lengths=tensor(caps.max_batch, dtype=torch.int64),
        query_start_loc=tensor(caps.max_batch + 1, dtype=torch.int32),
        num_accepted_tokens=tensor(caps.max_batch, dtype=torch.int32),
        is_prefilling=tensor(caps.max_batch, dtype=torch.bool),
        scratch=tensor(layout.total_nbytes, dtype=torch.uint8),
        main_k_cache=tensor(64, 64, caps.kv_heads, caps.head_dim),
        main_v_cache=tensor(64, 64, caps.kv_heads, caps.head_dim),
        k_descale=None,
        v_descale=None,
        main_block_table=tensor(caps.max_batch, 64, dtype=torch.int32),
        compressed_k_cache=tensor(64, 16, caps.index_head_dim),
        compressed_block_table=tensor(caps.max_batch, 64, dtype=torch.int32),
        raw_k_ring=tensor(
            caps.max_raw_state_slots, caps.raw_ring_capacity, caps.index_head_dim
        ),
        raw_logical_positions=tensor(
            caps.max_raw_state_slots, caps.raw_ring_capacity, dtype=torch.int64
        ),
        raw_rope_positions=tensor(
            caps.max_raw_state_slots, caps.raw_ring_capacity, 1, dtype=torch.int64
        ),
        raw_interval_start_positions=tensor(
            caps.max_raw_state_slots, dtype=torch.int64
        ),
        raw_state_slot_ids=tensor(caps.max_batch, dtype=torch.int64),
        index_q_norm_weight=tensor(caps.index_head_dim, dtype=torch.float32),
        index_k_norm_weight=tensor(caps.index_head_dim, dtype=torch.float32),
        rope_cos=tensor(caps.max_seq_len, 32, dtype=torch.float32),
        rope_sin=tensor(caps.max_seq_len, 32, dtype=torch.float32),
        output=tensor(rows, caps.q_heads, caps.head_dim),
        selected_positions=tensor(rows, caps.selection_width, dtype=torch.int32),
        sparse_gqa_direct_kv_warps=2,
        max_seq_len=caps.max_seq_len,
        max_speculative_tokens=caps.max_speculative_tokens,
        compress_ratio=caps.compress_ratio,
        budget=caps.budget,
        index_rotary_dim=caps.index_rotary_dim,
        mrope_section_0=0,
        mrope_section_1=0,
        mrope_section_2=0,
        mrope_interleaved=False,
        rms_norm_eps=caps.rms_norm_eps,
        score_chunk_groups=score_chunk,
        score_workspace_width=score_width,
        num_score_chunks=score_chunks,
        max_split_row_product=product,
        workspace_q_rows=work_rows,
        dcp_size=caps.dcp_size,
        dcp_rank=caps.dcp_rank,
        cp_kv_cache_interleave_size=caps.cp_kv_cache_interleave_size,
        attention_splits=caps.attention_splits,
    )
    operands.update(
        {
            field.name: getattr(layout, field.name)
            for field in fields(layout)
            if field.name.endswith("_offset_bytes")
            and field.name != "draft_positions_offset_bytes"
        }
    )
    return operands, layout


def _assert_launch_slices(calls, operands, caps):
    rows = caps.max_q_rows
    limit = 128 if caps.kv_heads == 1 else 64
    expected_rows = (
        [min(limit, rows - start) for start in range(0, rows, limit)]
        if caps.attention_splits
        else [rows]
    )
    assert [call["query"].shape[0] for call in calls] == expected_rows
    start = 0
    for call, count in zip(calls, expected_rows, strict=True):
        assert call["splits"] == (
            caps.attention_splits or contract._qwen_row_splits(count, caps.kv_heads)
        )
        for name in (
            "query",
            "request_ids",
            "query_positions",
            "selected_positions",
            "output",
        ):
            tensor = call[name]
            source = operands[name]
            assert tensor.shape[0] == count
            assert tensor.untyped_storage()._cdata == source.untyped_storage()._cdata
            assert (
                tensor.storage_offset()
                == source.storage_offset() + start * source.stride(0)
            )
        if caps.attention_splits or count <= limit:
            assert call["partial_output"].shape == (
                count,
                call["splits"],
                caps.q_heads,
                caps.head_dim,
            )
            assert call["partial_lse"].shape == (count, call["splits"], caps.q_heads)
        else:
            assert call["partial_output"] is call["partial_lse"] is None
        if caps.dcp_size > 1:
            assert call["output_lse"].shape == (count, caps.q_heads)
        else:
            assert call["output_lse"] is None
        start += count
    partials = [
        call["partial_output"] for call in calls if call["partial_output"] is not None
    ]
    assert len({tensor.storage_offset() for tensor in partials}) <= 1


@pytest.mark.parametrize("splits", [0, 1, 16, 64])
@pytest.mark.parametrize("rows", [1, 4, 16, 64, 65, 128, 129, 512])
@pytest.mark.parametrize("dcp_size", [1, 2])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_combined_slices_attention_without_slicing_selector(
    monkeypatch, splits, rows, dcp_size, kv_heads
):
    from b12x.attention.qsa import _kernels, _sparse_gqa

    caps = _caps(
        kv_heads=kv_heads,
        max_batch=1,
        max_q_rows=rows,
        attention_splits=splits,
        dcp_size=dcp_size,
        cp_kv_cache_interleave_size=4,
    )
    operands, _ = _operands(caps)
    selector_rows = []
    calls = []
    for name in (
        "launch_compress_completed_groups",
        "launch_commit_raw_ring",
        "launch_expand_selected_groups",
        "launch_expand_global_selected_groups",
        "launch_score_representatives_tiled",
        "launch_stabilize_topk",
        "launch_stage_topk_carry",
        "launch_topk_groups",
    ):
        monkeypatch.setattr(_kernels, name, lambda **kwargs: None)
    monkeypatch.setattr(
        _kernels,
        "launch_prepare_index_query",
        lambda **kwargs: selector_rows.append(kwargs["index_query"].shape[0]),
    )
    monkeypatch.setattr(
        _sparse_gqa, "launch_sparse_paged_gqa", lambda **kwargs: calls.append(kwargs)
    )
    contract._qsa_decode_impl(**operands)
    assert selector_rows == [rows]
    _assert_launch_slices(calls, operands, caps)


@pytest.mark.parametrize("splits", [0, 1, 16, 64])
@pytest.mark.parametrize("rows", [1, 4, 16, 64, 65, 128, 129, 512])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_separate_and_draft_attention_use_identical_partition_contract(
    monkeypatch, splits, rows, draft, kv_heads
):
    from b12x.attention.qsa import _sparse_gqa

    caps = _caps(
        kv_heads=kv_heads,
        max_batch=rows if draft else 1,
        max_raw_state_slots=max(4, rows),
        max_q_rows=rows,
        attention_splits=splits,
        max_speculative_tokens=3,
    )
    operands, layout = _operands(caps, device="cpu")
    query = contract._query_from_caps(caps, FrozenMapping())
    state = contract._materialize(caps, query.abi, TUNING.default_config(query, None))
    programs = object.__new__(contract.QsaPrograms)
    object.__setattr__(programs, "sparse", {"path": "selected"})
    object.__setattr__(programs, "sparse_draft", {"path": "draft"})
    state = replace(state, programs=programs)
    monkeypatch.setattr(contract, "plan_from_handle", lambda handle: None)
    monkeypatch.setattr(contract, "require_prepared", lambda *args: state)
    calls = []
    monkeypatch.setattr(
        _sparse_gqa, "launch_sparse_paged_gqa", lambda **kwargs: calls.append(kwargs)
    )
    selected = operands["selected_positions"]
    width = caps.selection_width + caps.max_speculative_tokens
    if draft:
        selected = contract._scratch_view(
            operands["scratch"],
            offset_bytes=layout.draft_positions_offset_bytes,
            shape=(rows, width),
            dtype=torch.int32,
        )
    contract._qsa_attention_op._init_fn(
        plan_handle=0,
        **{
            name: operands[name]
            for name in (
                "query",
                "request_ids",
                "query_positions",
                "main_k_cache",
                "main_v_cache",
                "k_descale",
                "v_descale",
                "main_block_table",
                "scratch",
                "output",
            )
        },
        selected_positions=None if draft else selected,
        work_rows=state.workspace_q_rows,
        max_split_row_product=state.max_split_row_product,
        partial_output_offset=layout.partial_output_offset_bytes,
        partial_lse_offset=layout.partial_lse_offset_bytes,
        output_lse_offset=layout.output_lse_offset_bytes,
        direct_kv_warps=2,
        draft_positions_offset=layout.draft_positions_offset_bytes,
        draft_width=width,
        draft_capacity=rows,
    )
    _assert_launch_slices(calls, {**operands, "selected_positions": selected}, caps)
    assert all(
        call["_prepared"] == {"path": "draft" if draft else "selected"}
        for call in calls
    )
