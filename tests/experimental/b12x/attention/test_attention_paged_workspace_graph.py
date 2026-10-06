from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from b12x.attention.paged import graph_replay
from b12x.attention.paged._forward import (
    _capture_decode_graph_replay_metadata_if_needed,
)
from b12x.attention.paged._scratch import B12XPagedAttentionBinding
from b12x.attention.paged.workspace import PagedAttentionWorkspace


def _make_cpu_decode_graph_workspace(
    *,
    work_items: int = 32,
    partial_rows: int = 4,
) -> PagedAttentionWorkspace:
    batch = 2
    q_heads = 128
    head_dim = 256
    return PagedAttentionWorkspace(
        mode="decode",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=q_heads,
        num_kv_heads=1,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=64,
        use_cuda_graph=True,
        request_indices=torch.empty(work_items, dtype=torch.int32),
        qo_tile_indices=torch.empty(work_items, dtype=torch.int32),
        kv_tile_indices=torch.empty(work_items, dtype=torch.int32),
        merge_indptr=torch.empty(batch + 1, dtype=torch.int32),
        o_indptr=torch.empty(batch + 1, dtype=torch.int32),
        kv_chunk_size_ptr=torch.empty(1, dtype=torch.int32),
        kv_window_start_tokens=torch.empty(batch, dtype=torch.int32),
        total_num_rows_ptr=torch.empty(1, dtype=torch.int32),
        block_valid_mask=torch.empty(work_items, dtype=torch.int32),
        cache_seqlens=torch.full((batch,), 4096, dtype=torch.int32),
        tmp_output=torch.empty(
            (partial_rows, q_heads, head_dim), dtype=torch.bfloat16
        ),
        tmp_lse=torch.empty((partial_rows, q_heads), dtype=torch.float32),
        _plan=SimpleNamespace(
            split_kv=True,
            gqa_group_size=128,
            cta_tile_q=16,
            num_q_heads=q_heads,
            head_dim_vo=head_dim,
            window_left=-1,
            page_table_shape=(batch, 64),
            total_q=batch,
        ),
        _decode_graph_chunk_pages_lut=torch.ones(65, dtype=torch.int32),
        _decode_graph_max_chunks_per_req=2,
        _decode_graph_max_q_tiles_per_req=8,
    )


def test_workspace_decode_graph_capacity_counts_query_tiles_and_partial_rows() -> None:
    workspace = _make_cpu_decode_graph_workspace()

    # Exact-plane decode uses CTA16, so GQA128 needs eight work tiles per
    # request for every KV chunk,
    # while merge scratch remains one partial row per request and KV chunk.
    workspace._validate_decode_graph_replay_capacity(batch=2)

    workspace.request_indices = torch.empty(16, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="capacity is too small"):
        workspace._validate_decode_graph_replay_capacity(batch=2)

    workspace = _make_cpu_decode_graph_workspace(partial_rows=3)
    with pytest.raises(RuntimeError, match="partial-row capacity is too small"):
        workspace._validate_decode_graph_replay_capacity(batch=2)

    workspace = _make_cpu_decode_graph_workspace(work_items=24)
    with pytest.raises(RuntimeError, match="batch/query-tile bucket"):
        workspace._validate_decode_graph_replay_capacity(batch=2)

    workspace = _make_cpu_decode_graph_workspace()
    workspace._decode_graph_max_q_tiles_per_req = 2
    with pytest.raises(RuntimeError, match="does not match the prepared Plan"):
        workspace._validate_decode_graph_replay_capacity(batch=2)


def test_workspace_runtime_length_updater_receives_query_tile_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_cpu_decode_graph_workspace()
    received: dict[str, object] = {}

    def fake_update_decode_graph_chunk_metadata(**kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(
        graph_replay,
        "update_decode_graph_chunk_metadata",
        fake_update_decode_graph_chunk_metadata,
    )

    workspace.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()

    assert received["max_q_tiles_per_req"] == 8
    assert received["cache_seqlens"] is workspace.cache_seqlens
    assert received["request_indices"] is workspace.request_indices


def test_workspace_runtime_page_table_updater_receives_query_tile_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_cpu_decode_graph_workspace()
    workspace.page_table = torch.empty((2, 64), dtype=torch.int32)
    workspace.cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    received: dict[str, object] = {}

    def fake_update_decode_graph_replay_metadata(**kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(
        graph_replay,
        "update_decode_graph_replay_metadata",
        fake_update_decode_graph_replay_metadata,
    )

    workspace.update_decode_graph_replay_metadata(
        req_to_token=torch.empty((2, 4096), dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
    )

    assert received["max_q_tiles_per_req"] == 8
    assert received["request_indices"] is workspace.request_indices
    assert received["page_table"] is workspace.page_table


@pytest.mark.parametrize("captured", [False, True])
def test_workspace_rejects_decode_graph_repreparation(
    captured: bool,
) -> None:
    workspace = _make_cpu_decode_graph_workspace()
    captured_lut = workspace._decode_graph_chunk_pages_lut
    workspace._decode_graph_metadata_captured_in_graph = captured

    with pytest.raises(RuntimeError, match="cannot replace decode graph replay state"):
        workspace.prepare_decode_graph_replay_state(
            batch=2,
            max_page_table_width=64,
            max_cache_page_count=64,
        )

    assert workspace._decode_graph_chunk_pages_lut is captured_lut
    assert workspace._decode_graph_metadata_captured_in_graph is captured


def test_workspace_rejects_decode_graph_preparation_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_cpu_decode_graph_workspace()
    workspace._decode_graph_chunk_pages_lut = None
    workspace.device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        workspace.prepare_decode_graph_replay_state(
            batch=2,
            max_page_table_width=64,
            max_cache_page_count=64,
        )


def test_workspace_rejects_decode_graph_state_after_generic_prepare() -> None:
    workspace = _make_cpu_decode_graph_workspace()
    workspace._decode_graph_chunk_pages_lut = None

    with pytest.raises(RuntimeError, match="cannot replace decode graph replay state"):
        workspace.prepare_decode_graph_replay_state(
            batch=1,
            max_page_table_width=64,
            max_cache_page_count=64,
        )


def test_fixed_workspace_requires_exact_decode_graph_bucket() -> None:
    workspace = PagedAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=64,
        head_dim_vo=64,
        page_size=64,
        max_total_q=2,
        max_batch=2,
        max_page_table_width=64,
        max_work_items=16,
        max_partial_rows=4,
        num_cache_pages=64,
        use_cuda_graph=True,
    )

    with pytest.raises(ValueError, match="batch must exactly match"):
        workspace.prepare_decode_graph_replay_state(
            batch=1,
            max_page_table_width=64,
            max_cache_page_count=64,
        )
    with pytest.raises(ValueError, match="page-table width must exactly match"):
        workspace.prepare_decode_graph_replay_state(
            batch=2,
            max_page_table_width=32,
            max_cache_page_count=32,
        )


@pytest.mark.parametrize(
    ("page_table_shape", "cache_shape", "cu_shape", "match"),
    [
        ((1, 64), (2,), (3,), "page_table must exactly match"),
        ((2, 64), (1,), (3,), "cache_seqlens must exactly match"),
        ((2, 64), (2,), (2,), "cu_seqlens_q must exactly match"),
    ],
)
def test_workspace_rejects_partial_decode_graph_runtime_metadata(
    page_table_shape: tuple[int, int],
    cache_shape: tuple[int],
    cu_shape: tuple[int],
    match: str,
) -> None:
    workspace = _make_cpu_decode_graph_workspace()

    with pytest.raises(ValueError, match=match):
        workspace.bind_cuda_graph_runtime_metadata(
            torch.empty(page_table_shape, dtype=torch.int32),
            torch.empty(cache_shape, dtype=torch.int32),
            torch.empty(cu_shape, dtype=torch.int32),
        )


def test_workspace_accepts_narrower_decode_graph_page_table_capacity() -> None:
    workspace = _make_cpu_decode_graph_workspace()
    page_table = torch.empty((2, 32), dtype=torch.int32)

    workspace.bind_cuda_graph_runtime_metadata(
        page_table,
        torch.empty((2,), dtype=torch.int32),
        torch.empty((3,), dtype=torch.int32),
    )

    assert workspace.page_table is page_table


@pytest.mark.parametrize("invalid_kind", ["dtype", "contiguity"])
def test_workspace_decode_graph_metadata_requires_fixed_dtype_and_layout(
    invalid_kind: str,
) -> None:
    workspace = _make_cpu_decode_graph_workspace()
    page_table = torch.empty((2, 64), dtype=torch.int32)
    if invalid_kind == "dtype":
        page_table = page_table.to(torch.int64)
        expected = "torch.int32"
    else:
        page_table = torch.empty((2, 128), dtype=torch.int32)[:, ::2]
        assert not page_table.is_contiguous()
        expected = "contiguous"

    with pytest.raises((TypeError, ValueError), match=expected):
        workspace.bind_cuda_graph_runtime_metadata(
            page_table,
            torch.empty((2,), dtype=torch.int32),
            torch.empty((3,), dtype=torch.int32),
        )


def test_workspace_rejects_decode_metadata_binding_before_replay_prepare() -> None:
    workspace = _make_cpu_decode_graph_workspace()
    workspace._plan = None
    workspace._decode_graph_chunk_pages_lut = None

    with pytest.raises(RuntimeError, match="after prepare_decode_graph_replay_state"):
        workspace.bind_cuda_graph_runtime_metadata(
            torch.empty((2, 64), dtype=torch.int32),
            torch.empty((2,), dtype=torch.int32),
            torch.empty((3,), dtype=torch.int32),
        )


def test_workspace_run_requires_exact_decode_graph_q_and_output_bucket() -> None:
    workspace = _make_cpu_decode_graph_workspace()
    q = torch.empty((2, 128, 256), dtype=torch.bfloat16)
    k_cache = torch.empty((1, 64, 1, 256), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    with pytest.raises(ValueError, match="q total_q must exactly match"):
        workspace.run(
            q[:1],
            k_cache,
            v_cache,
            output=torch.empty_like(q),
        )

    with pytest.raises(ValueError, match="output total_q must exactly match"):
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=torch.empty((3, 128, 256), dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("invalid_kind", ["q", "output"])
def test_public_binding_cannot_bypass_exact_decode_graph_bucket(
    invalid_kind: str,
) -> None:
    workspace = _make_cpu_decode_graph_workspace()
    workspace.page_table = torch.empty((2, 64), dtype=torch.int32)
    workspace.cu_seqlens_q = torch.arange(3, dtype=torch.int32)
    q = torch.empty((2, 128, 256), dtype=torch.bfloat16)
    k_cache = torch.empty((1, 64, 1, 256), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    bound_q = q[:1] if invalid_kind == "q" else q
    bound_output = (
        torch.empty((3, 128, 256), dtype=torch.bfloat16)
        if invalid_kind == "output"
        else torch.empty_like(q)
    )
    with pytest.raises(
        ValueError,
        match=f"{invalid_kind} total_q must exactly match",
    ):
        from b12x.attention.paged._forward import paged_attention_forward

        paged_attention_forward(
            binding=B12XPagedAttentionBinding(
                scratch=workspace,
                q=bound_q,
                k_cache=k_cache,
                v_cache=v_cache,
                output=bound_output,
            )
        )


def test_every_decode_graph_capture_inserts_a_device_metadata_updater(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Scratch:
        use_cuda_graph = True
        mode = "decode"
        _decode_graph_chunk_pages_lut = object()
        _plan = object()
        _owner_scratch_plan = None
        _decode_graph_metadata_captured_in_graph = False

        def update_decode_graph_replay_metadata_from_runtime_cache_seqlens(
            self,
        ) -> None:
            nonlocal calls
            calls += 1

    scratch = Scratch()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    _capture_decode_graph_replay_metadata_if_needed(scratch)
    assert scratch._decode_graph_metadata_captured_in_graph
    # Model a second capture of the same prepared state.  The lifetime flag is
    # already true, but it must not suppress the new graph's updater node.
    _capture_decode_graph_replay_metadata_if_needed(scratch)

    assert calls == 2


def test_decode_graph_capture_fails_closed_without_device_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = SimpleNamespace(
        use_cuda_graph=True,
        mode="decode",
        _decode_graph_chunk_pages_lut=None,
        _plan=object(),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="prepare_decode_graph_replay_state"):
        _capture_decode_graph_replay_metadata_if_needed(scratch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_public_workspace_prepares_exact_gqa128_decode_graph_capacity() -> None:
    workspace = PagedAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=128,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        max_total_q=2,
        max_batch=2,
        max_page_table_width=64,
        max_work_items=32,
        max_partial_rows=4,
        num_cache_pages=64,
        use_cuda_graph=True,
    )

    workspace.prepare_decode_graph_replay_state(
        batch=2,
        max_page_table_width=64,
        max_cache_page_count=64,
    )

    assert workspace.plan.gqa_group_size == 128
    assert workspace.plan.cta_tile_q == 16
    assert workspace._decode_graph_max_q_tiles_per_req == 8
    assert workspace._decode_graph_max_chunks_per_req == 2
    assert workspace.plan.padded_batch_size == 2 * 8 * 2
    assert workspace.plan.total_num_partial_rows == 2 * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_public_workspace_fails_closed_below_direct_query_tile_capacity() -> None:
    workspace = PagedAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=128,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        max_total_q=2,
        max_batch=2,
        max_page_table_width=64,
        max_work_items=15,
        max_partial_rows=0,
        num_cache_pages=64,
        use_cuda_graph=True,
    )

    with pytest.raises(ValueError, match="needs 16 work items"):
        workspace.prepare_decode_graph_replay_state(
            batch=2,
            max_page_table_width=64,
            max_cache_page_count=64,
        )
