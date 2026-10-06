from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import b12x.attention.dsa_indexer._impl as indexer_impl
import b12x.attention._shared.mla.api as sparse_mla_impl
import b12x.attention._shared.mla.compressed_api as compressed_sparse_mla_impl
import b12x.attention._shared.mla.kernel as compressed_sparse_mla_kernel
from b12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
    compressed_sparse_mla_page_nbytes,
)
from b12x.attention._shared.workspace import B12XAttentionArena, B12XAttentionWorkspace
from b12x.attention.dsa_indexer.scratch import (
    INDEXER_SOURCE_LAYOUT_CONTIGUOUS,
    INDEXER_SOURCE_LAYOUT_PAGED,
    B12XIndexerContiguousBinding,
    B12XIndexerPagedBinding,
    B12XIndexerPagedScratch,
    B12XIndexerScratchCaps,
    plan_indexer_scratch,
)
from b12x.attention.dsa_indexer.scratch import (
    B12XIndexerContiguousScratchCaps,
    B12XIndexerPagedScratchCaps,
    plan_indexer_contiguous_scratch,
    plan_indexer_paged_scratch,
)
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLABinding,
    B12XCompressedSparseMLAScratchCaps,
)
from b12x.attention.compressed_sparse_mla import (
    plan as plan_compressed_sparse_mla_scratch,
)
from b12x.attention.sparse_mla._scratch import (
    B12XSparseMLABinding,
    B12XSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)


def _workspace(
    *,
    num_q_heads: int = 2,
    indexer_num_q_heads: int = 2,
    max_total_q: int = 4,
    max_paged_q_rows: int = 4,
    topk: int = 8,
    max_page_table_width: int = 8,
) -> B12XAttentionWorkspace:
    return B12XAttentionWorkspace(
        mode="decode",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_q_heads,
        indexer_num_q_heads=indexer_num_q_heads,
        head_dim=512,
        v_head_dim=512,
        topk=topk,
        indexer_topk=topk,
        max_page_table_width=max_page_table_width,
        max_total_q=max_total_q,
        max_batch=max_total_q,
        max_paged_q_rows=max_paged_q_rows,
        max_kv_rows=0,
        fixed_capacity=True,
        max_chunks_per_row=4,
    )


def _one_scratch(plan):
    (spec,) = plan.scratch_specs()
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def test_compressed_sparse_mla_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_compressed_sparse_mla_scratch(
        B12XCompressedSparseMLAScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_width=8,
            max_page_table_width=16,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "compressed_sparse_mla.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.nbytes == specs[0].nbytes


@pytest.mark.parametrize("cache_format", ["deepseek_v4", "deepseek_v41"])
def test_compressed_sparse_mla_bind_is_views_only(monkeypatch, cache_format) -> None:
    plan = plan_compressed_sparse_mla_scratch(
        B12XCompressedSparseMLAScratchCaps(
            device="cpu", num_q_heads=8, max_q_rows=4, max_width=8,
            swa_width=4, indexed_width=4, max_page_table_width=2,
            cache_format=cache_format,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.full(spec.shape, 165, dtype=spec.dtype, device=spec.device)
    q = torch.empty((4, 8, 512), dtype=torch.bfloat16)
    indices = torch.zeros((4, 4), dtype=torch.int32)
    lengths = torch.ones((4,), dtype=torch.int32)
    page_table = torch.ones((1, 2), dtype=torch.int32).expand(4, -1)

    def reject_allocation(*args, **kwargs):
        raise AssertionError("bind must not allocate tensor storage")

    with monkeypatch.context() as guarded:
        guarded.setattr(torch, "empty", reject_allocation)
        guarded.setattr(torch, "zeros", reject_allocation)
        guarded.setattr(torch, "full", reject_allocation)
        binding = plan.bind(
            scratch=scratch, q=q, swa_indices=indices, swa_lengths=lengths,
            indexed_indices=indices, indexed_lengths=lengths, indexed_page_table=page_table,
        )
    assert torch.all(scratch == 165)
    # Mutation through the consumer-visible final LSE is within caller storage.
    binding.scratch.final_lse.zero_()
    assert torch.count_nonzero(scratch != 165).item() == 4 * 8 * 4


def test_indexer_paged_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_page_table_width=16,
            topk=8,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=512,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "paged_indexer.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.nbytes == specs[0].nbytes
    assert plan.layout.supertile_tokens == 512
    assert plan.layout.route == "paged_tiled"
    assert plan.layout.fused_pack_elements == 0
    assert plan.layout.fused_state_words == 0


def test_indexer_paged_scratch_bind_does_not_call_workspace_or_arena_factory(
    monkeypatch,
) -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_page_table_width=16,
            topk=8,
            reserve_paged_logits=False,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    real_page_table = torch.empty((4, 16), dtype=torch.int32)
    cache_seqlens = torch.empty((4,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)

    def fail_factory(*args, **kwargs):
        raise AssertionError("scratch binding must not call workspace/arena factories")

    monkeypatch.setattr(B12XAttentionArena, "make_workspace", fail_factory)
    monkeypatch.setattr(B12XAttentionArena, "from_shared_arena", fail_factory)
    monkeypatch.setattr(B12XAttentionArena, "_make_workspace_views", fail_factory)

    binding = plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
    )

    assert isinstance(binding, B12XIndexerPagedBinding)
    assert isinstance(binding.scratch, B12XIndexerPagedScratch)
    assert binding.scratch.shared_scratch.data_ptr() == scratch.data_ptr()
    assert binding.real_page_table is real_page_table
    assert binding.active_width is active_width
    assert binding.scratch.indexer_contiguous_tile_logits is not None
    assert binding.scratch.indexer_contiguous_topk_values is not None
    assert binding.scratch.indexer_contiguous_topk_indices is not None
    assert binding.scratch.route == "paged_tiled"
    assert binding.scratch.fused_indexer_pack_values is None
    assert binding.scratch.fused_indexer_pack_indices is None
    assert binding.scratch.fused_indexer_merge_state is None


def test_indexer_common_plan_chooses_layout_from_source_contract() -> None:
    paged_plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=2,
            max_q_rows=4,
            max_page_table_width=16,
            topk=8,
            mode="decode",
        )
    )
    contiguous_plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_CONTIGUOUS,
            num_q_heads=2,
            max_q_rows=4,
            max_k_rows=1024,
            topk=8,
        )
    )

    assert paged_plan.source_layout == INDEXER_SOURCE_LAYOUT_PAGED
    assert paged_plan.layout.route in {"paged_tiled", "paged_fused"}
    assert contiguous_plan.source_layout == INDEXER_SOURCE_LAYOUT_CONTIGUOUS
    assert contiguous_plan.layout.max_k_rows == 1024


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 64])
def test_indexer_common_plan_selects_tiled_for_c4_decode_buckets(rows) -> None:
    # C4 routing is hardware-specific. A CPU plan has no Blackwell capability
    # metadata, so it conservatively retains the streamed tiled route.
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=64,
            max_q_rows=rows,
            max_page_table_width=4160,
            topk=512,
            mode="decode",
        )
    )

    assert plan.layout.route == "paged_tiled"


@pytest.mark.parametrize("minor,sm_count", [(0, 188), (1, 48)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32])
def test_indexer_common_plan_selects_sm12x_c4_decode_routes(
    monkeypatch, minor, sm_count, rows
) -> None:
    props = type(
        "Props",
        (),
        {
            "major": 12,
            "minor": minor,
            "multi_processor_count": sm_count,
            "name": "test SM12x device",
        },
    )()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)

    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cuda",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=64,
            max_q_rows=rows,
            max_page_table_width=1024,
            topk=512,
            mode="decode",
        )
    )

    assert plan.layout.route == ("paged_fused" if rows <= 16 else "paged_tiled")


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 64])
def test_indexer_common_plan_selects_measured_glm_decode_routes(rows) -> None:
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=32,
            max_q_rows=rows,
            max_page_table_width=256,
            topk=2048,
            mode="decode",
        )
    )

    # GLM keeps fused through rows 16; rows >= 32 measured faster on the
    # streamed tiled route at every capacity bucket.
    expected = "paged_fused" if rows <= 16 else "paged_tiled"
    assert plan.layout.route == expected


@pytest.mark.parametrize("rows", [1024, 2048, 4096, 8192])
def test_indexer_common_plan_selects_bk512_for_c4_prefill_buckets(rows) -> None:
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=64,
            max_q_rows=rows,
            max_page_table_width=4160,
            topk=512,
            mode="prefill",
            shared_page_table=True,
        )
    )

    assert plan.layout.route == "packed_contiguous"
    assert plan.layout.prefill_block_k == 512


@pytest.mark.parametrize("rows", [1024, 2048, 4096, 8192])
def test_indexer_common_plan_selects_bk512_for_glm_prefill_buckets(rows) -> None:
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=32,
            max_q_rows=rows,
            max_page_table_width=256,
            topk=2048,
            mode="prefill",
            shared_page_table=True,
        )
    )

    assert plan.layout.route == "packed_contiguous"
    assert plan.layout.prefill_block_k == 512


def test_indexer_paged_default_supertile_is_capped_by_fixed_capacity(
    monkeypatch,
) -> None:
    monkeypatch.delenv("B12X_PAGED_INDEX_SUPERTILE_K", raising=False)
    common = dict(
        device="cpu",
        source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
        num_q_heads=32,
        max_q_rows=4096,
        max_page_table_width=256,
        topk=2048,
        mode="prefill",
        shared_page_table=True,
    )
    automatic = plan_indexer_scratch(B12XIndexerScratchCaps(**common))
    explicit = plan_indexer_scratch(B12XIndexerScratchCaps(**common, supertile_k=32768))

    assert automatic.layout.supertile_tokens == 16384
    assert automatic.layout.gather_k_rows == 16384
    assert explicit.layout.supertile_tokens == 32768
    assert explicit.layout.gather_k_rows == 32768
    assert automatic.layout.nbytes < explicit.layout.nbytes


def test_indexer_common_packed_scratch_sizes_from_indexer_k_rows() -> None:
    context_tokens = 256 * 1024
    supertile_context_tokens = 32 * 1024
    c4_tokens_per_k = 4

    glm_k_rows = context_tokens
    c4_k_rows = (context_tokens + c4_tokens_per_k - 1) // c4_tokens_per_k
    glm_supertile_k_rows = supertile_context_tokens
    c4_supertile_k_rows = (
        supertile_context_tokens + c4_tokens_per_k - 1
    ) // c4_tokens_per_k

    glm_plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=4,
            max_q_rows=128,
            max_k_rows=glm_k_rows,
            topk=8,
            supertile_k=glm_supertile_k_rows,
            mode="prefill",
            shared_page_table=True,
        )
    )
    c4_plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device="cpu",
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=4,
            max_q_rows=128,
            max_k_rows=c4_k_rows,
            topk=8,
            supertile_k=c4_supertile_k_rows,
            mode="prefill",
            shared_page_table=True,
        )
    )

    assert glm_plan.layout.route == "packed_contiguous"
    assert c4_plan.layout.route == "packed_contiguous"
    assert glm_plan.layout.max_chunks == (
        (glm_k_rows + glm_supertile_k_rows - 1) // glm_supertile_k_rows
    )
    assert c4_plan.layout.max_chunks == (
        (c4_k_rows + c4_supertile_k_rows - 1) // c4_supertile_k_rows
    )
    assert glm_plan.layout.gather_k_rows == glm_supertile_k_rows
    assert c4_plan.layout.gather_k_rows == c4_supertile_k_rows
    assert c4_plan.layout.gather_k_rows * c4_tokens_per_k == (
        glm_plan.layout.gather_k_rows
    )
    assert c4_plan.layout.max_chunks == glm_plan.layout.max_chunks


def test_indexer_paged_scratch_plan_exposes_one_opaque_arena_spec() -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_page_table_width=16,
            topk=8,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=512,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "paged_indexer.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.nbytes == specs[0].nbytes
    assert plan.layout.supertile_tokens == 512


def test_indexer_paged_scratch_bind_does_not_call_workspace_factory(
    monkeypatch,
) -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_page_table_width=16,
            topk=8,
            reserve_paged_logits=False,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    real_page_table = torch.empty((4, 16), dtype=torch.int32)
    cache_seqlens = torch.empty((4,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)

    def fail_make_workspace(*args, **kwargs):
        raise AssertionError("scratch binding must not call the workspace factory")

    monkeypatch.setattr(B12XAttentionArena, "make_workspace", fail_make_workspace)

    binding = plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
    )

    assert isinstance(binding, B12XIndexerPagedBinding)
    assert binding.real_page_table is real_page_table
    assert binding.metadata.real_page_table is real_page_table
    assert binding.metadata.cache_seqlens_int32 is cache_seqlens
    assert binding.active_width is active_width


def test_indexer_contiguous_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_k_rows=1024,
            topk=8,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "indexer_contiguous.arena"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.nbytes == specs[0].nbytes
    assert plan.layout.max_k_rows == 1024
    assert plan.layout.tile_logits_elements > 0


def test_indexer_contiguous_scratch_bind_does_not_call_workspace_factory(
    monkeypatch,
) -> None:
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_k_rows=1024,
            topk=8,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    k_start = torch.zeros((4,), dtype=torch.int32)
    k_end = torch.full((4,), 64, dtype=torch.int32)

    def fail_make_workspace(*args, **kwargs):
        raise AssertionError("scratch binding must not call the workspace factory")

    monkeypatch.setattr(B12XAttentionArena, "make_workspace", fail_make_workspace)

    binding = plan.bind(scratch=scratch, k_start=k_start, k_end=k_end)

    assert isinstance(binding, B12XIndexerContiguousBinding)
    assert binding.metadata.k_start is k_start
    assert binding.metadata.k_end is k_end


def test_sparse_mla_scratch_plan_exposes_one_opaque_arena_spec() -> None:
    plan = plan_sparse_mla_scratch(
        B12XSparseMLAScratchCaps(
            softmax_scale=1.0,
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_width=8,
            max_page_table_width=16,
            head_dim=512,
            v_head_dim=512,
            mode="extend",
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "sparse_mla.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.nbytes == specs[0].nbytes


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_sparse_mla_scratch_can_expose_head_major_output(mode: str) -> None:
    caps = B12XSparseMLAScratchCaps(
        softmax_scale=1.0,
        device="cpu",
        num_q_heads=8,
        max_q_rows=6,
        max_width=32,
        head_dim=576,
        v_head_dim=512,
        mode=mode,
        max_chunks_per_row=4,
        head_major_output=True,
    )
    plan = plan_sparse_mla_scratch(caps)
    scratch = _one_scratch(plan)
    q = torch.empty((6, 8, 576), dtype=torch.bfloat16)
    binding = plan.bind(
        scratch=scratch,
        q=q,
        selected_indices=torch.empty((6, 32), dtype=torch.int32),
        cache_seqlens_int32=torch.empty((6,), dtype=torch.int32),
        nsa_cache_seqlens_int32=torch.empty((6,), dtype=torch.int32),
    )

    output = binding.scratch.output_buffer
    assert output is not None
    assert output.shape == (6, 8, 512)
    assert output.stride() == (512, 6 * 512, 1)
    assert not output.is_contiguous()
    assert binding.scratch.head_major_output
    if mode == "decode":
        assert binding.scratch.tmp_output is not None
        assert output.data_ptr() == binding.scratch.tmp_output[:, :, 0, :].data_ptr()


def test_sparse_mla_scratch_bind_does_not_call_workspace_factory(
    monkeypatch,
) -> None:
    plan = plan_sparse_mla_scratch(
        B12XSparseMLAScratchCaps(
            softmax_scale=1.0,
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_width=8,
            max_page_table_width=16,
            head_dim=512,
            v_head_dim=512,
            mode="extend",
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty((4, 2, 512), dtype=torch.bfloat16)
    selected_indices = torch.empty((4, 8), dtype=torch.int32)
    cache_seqlens = torch.empty((4,), dtype=torch.int32)
    active_counts = torch.empty((4,), dtype=torch.int32)

    def fail_make_workspace(*args, **kwargs):
        raise AssertionError("scratch binding must not call the workspace factory")

    monkeypatch.setattr(B12XAttentionArena, "make_workspace", fail_make_workspace)

    binding = plan.bind(
        scratch=scratch,
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active_counts,
    )

    assert isinstance(binding, B12XSparseMLABinding)
    assert binding.q.data_ptr() == q.data_ptr()
    assert binding.selected_indices is selected_indices


def test_workspace_bind_compressed_sparse_mla_returns_common_binding_type() -> None:
    workspace = _workspace(topk=6, max_page_table_width=5)
    q = torch.empty((4, 2, 512), dtype=torch.bfloat16)
    swa_indices = torch.empty((4, 2), dtype=torch.int32)
    swa_lengths = torch.empty((4,), dtype=torch.int32)
    indexed_indices = torch.empty((4, 4), dtype=torch.int32)
    indexed_lengths = torch.empty((4,), dtype=torch.int32)
    indexed_page_table = torch.empty((4, 5), dtype=torch.int32)

    binding = workspace.bind_compressed_sparse_mla(
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
    )

    assert isinstance(binding, B12XCompressedSparseMLABinding)
    assert binding.scratch is workspace
    assert binding.q.data_ptr() == q.data_ptr()
    assert binding.indexed_page_table is indexed_page_table


def test_compressed_sparse_mla_bind_accepts_row_shared_page_table() -> None:
    workspace = _workspace(topk=6, max_page_table_width=5)
    q = torch.empty((4, 2, 512), dtype=torch.bfloat16)
    swa_indices = torch.empty((4, 2), dtype=torch.int32)
    swa_lengths = torch.empty((4,), dtype=torch.int32)
    indexed_indices = torch.empty((4, 4), dtype=torch.int32)
    indexed_lengths = torch.empty((4,), dtype=torch.int32)
    indexed_page_table = torch.empty((1, 5), dtype=torch.int32).expand(4, -1)

    binding = workspace.bind_compressed_sparse_mla(
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
    )

    assert binding.indexed_page_table is indexed_page_table
    assert binding.indexed_page_table.stride() == (0, 1)


def test_workspace_bind_sparse_mla_returns_common_binding_type() -> None:
    workspace = _workspace(topk=6, max_page_table_width=5)
    q = torch.empty((4, 2, 512), dtype=torch.bfloat16)
    selected_indices = torch.empty((4, 6), dtype=torch.int32)
    cache_seqlens = torch.empty((3,), dtype=torch.int32)
    active_counts = torch.empty((4,), dtype=torch.int32)

    binding = workspace.bind_sparse_mla(
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active_counts,
    )

    assert isinstance(binding, B12XSparseMLABinding)
    assert not hasattr(binding, "workspace")
    assert binding.scratch is workspace
    assert binding.q.data_ptr() == q.data_ptr()
    assert binding.selected_indices is selected_indices
    assert binding.cache_seqlens_int32 is cache_seqlens


def test_indexer_paged_plan_bind_returns_common_binding_type() -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=3,
            max_q_rows=4,
            max_page_table_width=7,
            topk=8,
            reserve_paged_logits=False,
            mode="prefill",
            shared_page_table=True,
        )
    )
    scratch = _one_scratch(plan)
    real_page_table = torch.empty((4, 7), dtype=torch.int32)
    cache_seqlens = torch.empty((4,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)
    schedule = torch.empty((2, 2), dtype=torch.int32)

    binding = plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
        schedule_metadata=schedule,
        expected_num_q_heads=3,
        shared_page_table=True,
    )

    assert isinstance(binding, B12XIndexerPagedBinding)
    assert not hasattr(binding, "workspace")
    assert isinstance(binding.scratch, B12XIndexerPagedScratch)
    assert binding.real_page_table is real_page_table
    assert binding.active_width is active_width
    assert binding.expected_num_q_heads == 3
    assert binding.shared_page_table is True


def test_indexer_paged_decode_plan_bind_returns_common_binding_type() -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=3,
            max_q_rows=4,
            max_page_table_width=7,
            topk=8,
            reserve_paged_logits=False,
            route="paged_tiled",
        )
    )
    scratch = _one_scratch(plan)
    real_page_table = torch.empty((4, 7), dtype=torch.int32)
    cache_seqlens = torch.empty((4,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)
    schedule = torch.empty((2, 2), dtype=torch.int32)

    binding = plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
        schedule_metadata=schedule,
    )

    assert isinstance(binding, B12XIndexerPagedBinding)
    assert not hasattr(binding, "workspace")
    assert isinstance(binding.scratch, B12XIndexerPagedScratch)
    assert binding.metadata.real_page_table is real_page_table
    assert binding.metadata.paged_mqa_schedule_metadata is schedule
    assert binding.active_width is active_width


def test_indexer_contiguous_plan_bind_returns_common_binding_type() -> None:
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device="cpu",
            num_q_heads=3,
            max_q_rows=4,
            max_k_rows=64,
            topk=8,
        )
    )
    scratch = _one_scratch(plan)
    k_start = torch.zeros((3,), dtype=torch.int32)
    k_end = torch.full((3,), 64, dtype=torch.int32)

    binding = plan.bind(scratch=scratch, k_start=k_start, k_end=k_end, topk=3)

    assert isinstance(binding, B12XIndexerContiguousBinding)
    assert not hasattr(binding, "workspace")
    assert binding.metadata.k_start is k_start
    assert binding.metadata.k_end is k_end
    assert binding.topk == 3
    assert binding.tile_logits is not None
    assert binding.output_values.shape == (3, 3)
    assert binding.output_indices.shape == (3, 3)
    assert binding.candidate_values.shape == (2, 3, 3)
    assert binding.candidate_indices.shape == (2, 3, 3)
    assert binding.merge_positions is None
    assert binding.lengths.shape == (3,)


def test_compressed_sparse_mla_decode_binding_supplies_runtime_tensors(
    monkeypatch,
) -> None:
    workspace = _workspace(max_total_q=1, topk=2, max_page_table_width=2)
    workspace.fixed_capacity = False
    workspace.use_cuda_graph = True
    workspace.tmp_output = torch.empty((1, 2, 4, 512), dtype=torch.bfloat16)
    workspace.tmp_lse = torch.empty((1, 2, 4), dtype=torch.float32)
    workspace.output_buffer = workspace.tmp_output[:, :, 0, :]
    workspace.final_lse = torch.empty((1, 2), dtype=torch.float32)
    workspace.kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32)
    workspace.num_chunks_ptr = torch.empty((1,), dtype=torch.int32)

    q = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
    swa_indices = torch.zeros((1, 2), dtype=torch.int32)
    swa_lengths = torch.zeros((1,), dtype=torch.int32)
    binding = workspace.bind_compressed_sparse_mla(
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
    )
    swa_cache = torch.empty(
        (1, compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
    )
    calls = {}

    def fail_stage(**kwargs):
        raise AssertionError("binding path should not stage compressed MLA inputs")

    def fake_forward(**kwargs):
        calls["q_all"] = kwargs["q_all"]
        calls["swa_indices"] = kwargs["swa_indices"]
        calls["swa_lengths"] = kwargs["swa_topk_lengths"]
        return kwargs["workspace"].output_buffer

    monkeypatch.setattr(
        compressed_sparse_mla_impl,
        "_stage_fixed_compressed_sparse_mla_inputs",
        fail_stage,
    )
    monkeypatch.setattr(
        compressed_sparse_mla_impl, "_use_sm120_sparse_mla", lambda **_: True
    )
    monkeypatch.setattr(
        compressed_sparse_mla_kernel, "run_unified_decode", fake_forward
    )

    out = compressed_sparse_mla_impl.compressed_sparse_mla_decode_forward(
        binding=binding,
        swa_k_cache=swa_cache,
        sm_scale=1.0,
    )

    assert calls["q_all"].data_ptr() == q.data_ptr()
    assert calls["swa_indices"].data_ptr() == swa_indices.data_ptr()
    assert calls["swa_lengths"].data_ptr() == swa_lengths.data_ptr()
    assert out.shape == (1, 2, 512)


def test_sparse_mla_decode_binding_supplies_runtime_tensors(monkeypatch) -> None:
    workspace = _workspace(max_total_q=1, topk=2, max_page_table_width=2)
    q = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
    selected_indices = torch.zeros((1, 2), dtype=torch.int32)
    cache_seqlens = torch.zeros((1,), dtype=torch.int32)
    active_counts = torch.zeros((1,), dtype=torch.int32)
    binding = workspace.bind_sparse_mla(
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active_counts,
    )
    kv_cache = torch.empty((1, 576), dtype=torch.bfloat16)
    calls = {}

    def fake_run_sparse_mla(**kwargs):
        calls.update(kwargs)
        return torch.empty((1, 2, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(sparse_mla_impl, "_run_sparse_mla", fake_run_sparse_mla)

    out = sparse_mla_impl.sparse_mla_decode_forward(
        binding=binding,
        kv_cache=kv_cache,
        sm_scale=1.0,
    )

    assert calls["q_all"].data_ptr() == q.data_ptr()
    assert calls["selected_indices"].data_ptr() == selected_indices.data_ptr()
    assert calls["cache_seqlens_int32"].data_ptr() == cache_seqlens.data_ptr()
    assert calls["active_token_counts"].data_ptr() == active_counts.data_ptr()
    assert calls["workspace"] is workspace
    assert calls["v_head_dim"] == workspace.v_head_dim
    assert out.shape == (1, 2, 512)


def test_indexer_paged_decode_binding_supplies_metadata(monkeypatch) -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=3,
            max_page_table_width=5,
            topk=4,
            reserve_paged_logits=False,
            route="paged_tiled",
        )
    )
    scratch = _one_scratch(plan)
    real_page_table = torch.zeros((3, 5), dtype=torch.int32)
    cache_seqlens = torch.zeros((3,), dtype=torch.int32)
    active_width = torch.tensor([0], dtype=torch.int32)
    binding = plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
    )
    q_fp8 = torch.empty((3, 2, 128), dtype=torch.uint8)
    weights = torch.empty((3, 2), dtype=torch.float32)
    index_k_cache = torch.empty((8, 64 * 128), dtype=torch.uint8)
    calls = {}

    def fake_supports(**kwargs):
        return True

    def fake_uses_schedule(**kwargs):
        return False

    def fake_run_kernel(**kwargs):
        calls.update(kwargs)
        return torch.empty((3, 320), dtype=torch.float32)

    monkeypatch.setattr(indexer_impl, "supports_paged_logits_kernel", fake_supports)
    monkeypatch.setattr(indexer_impl, "uses_paged_mqa_schedule", fake_uses_schedule)
    monkeypatch.setattr(indexer_impl, "run_paged_logits_kernel", fake_run_kernel)

    logits = indexer_impl.paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
    )

    assert calls["real_page_table"] is real_page_table
    assert calls["seqlens_per_query"] is cache_seqlens
    assert calls["active_width"] is active_width
    assert "workspace" not in calls
    assert logits.shape == (3, 320)


def test_indexer_contiguous_logits_binding_supplies_metadata(monkeypatch) -> None:
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=3,
            max_k_rows=64,
            topk=4,
        )
    )
    scratch = _one_scratch(plan)
    k_start = torch.zeros((3,), dtype=torch.int32)
    k_end = torch.full((3,), 64, dtype=torch.int32)
    binding = plan.bind(scratch=scratch, k_start=k_start, k_end=k_end, topk=2)
    q_fp8 = torch.empty((3, 2, 128), dtype=torch.uint8)
    weights = torch.empty((3, 2), dtype=torch.float32)
    k_quant = torch.empty((64, 128), dtype=torch.uint8)
    k_scale = torch.empty((64,), dtype=torch.float32)
    calls = {}

    def fake_supports(**kwargs):
        return True

    def fake_run_kernel(**kwargs):
        calls.update(kwargs)
        return torch.empty((3, 64), dtype=torch.float32)

    monkeypatch.setattr(
        indexer_impl, "supports_contiguous_logits_kernel", fake_supports
    )
    monkeypatch.setattr(indexer_impl, "run_contiguous_logits_kernel", fake_run_kernel)

    logits = indexer_impl.contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=(k_quant, k_scale),
        binding=binding,
    )

    assert calls["k_start"] is k_start
    assert calls["k_end"] is k_end
    assert "workspace" not in calls
    assert "contract_phantoms" not in calls
    assert logits.shape == (3, 64)


def test_indexer_contiguous_tiled_topk_binding_supplies_topk_and_metadata(
    monkeypatch,
) -> None:
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=3,
            max_k_rows=4,
            topk=4,
        )
    )
    scratch = _one_scratch(plan)
    k_start = torch.zeros((3,), dtype=torch.int32)
    k_end = torch.full((3,), 4, dtype=torch.int32)
    binding = replace(
        plan.bind(scratch=scratch, k_start=k_start, k_end=k_end, topk=2),
        strict=False,
    )
    q_fp8 = torch.empty((3, 2, 128), dtype=torch.uint8)
    weights = torch.empty((3, 2), dtype=torch.float32)
    k_quant = torch.empty((4, 128), dtype=torch.uint8)
    k_scale = torch.empty((4,), dtype=torch.float32)
    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0, 2.0],
            [4.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 5.0, 4.0],
        ],
        dtype=torch.float32,
    )
    calls = {}

    def fake_supports(**kwargs):
        return False

    def fake_reference(**kwargs):
        calls.update(kwargs)
        return logits

    monkeypatch.setattr(
        indexer_impl, "supports_contiguous_logits_kernel", fake_supports
    )
    monkeypatch.setattr(indexer_impl, "contiguous_logits_reference", fake_reference)

    indices = indexer_impl.contiguous_tiled_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=(k_quant, k_scale),
        binding=binding,
    )

    assert calls["k_start"] is k_start
    assert calls["k_end"] is k_end
    assert indices.tolist() == [[1, 3], [0, 3], [2, 3]]
