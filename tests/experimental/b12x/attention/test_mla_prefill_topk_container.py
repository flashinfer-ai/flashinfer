"""Model-specific sparse-MLA top-k row-width contracts."""

from __future__ import annotations

import pytest
import torch

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.prefill import (
    _topk_container,
    run_unified_prefill,
)
from b12x.attention._shared.mla.traits import ModelType


@pytest.mark.parametrize(
    ("topk", "expected"),
    [
        (1, 512),
        (100, 512),
        (128, 128),
        (192, 512),
        (512, 512),
        (513, 1024),
        (1024, 1024),
        (1500, 2048),
        (2048, 2048),
        (2051, 2051),
        (2112, 2112),
        (4096, 4096),
    ],
)
def test_topk_container_rounds_up_to_supported_widths(topk: int, expected: int) -> None:
    """A 192-token prefill clamps top-k to 192; the kernel needs a 512 row."""
    assert _topk_container(ModelType.DSV4, topk) == expected


@pytest.mark.parametrize("model_type", [ModelType.GLM_NSA, ModelType.GLM_NEXT])
@pytest.mark.parametrize("topk", [1, 100, 192, 513, 1500])
def test_glm_topk_container_preserves_plan_width(
    model_type: ModelType, topk: int
) -> None:
    """GLM plan bindings retain the caller-owned top-k row width."""
    assert _topk_container(model_type, topk) == topk


@pytest.mark.parametrize("topk", [512, 1024, 2048, 2051, 2088, 2176])
def test_kvarn_topk_container_preserves_plan_width(topk: int) -> None:
    """KVarN MG extends bind the caller-planned selection row unchanged."""
    assert _topk_container(ModelType.KVARN_MLA, topk) == topk


def test_kvarn_prefill_rejects_noncontainer_topk_width() -> None:
    """The KVarN MG gate accepts packed containers only. A width between
    containers hard-fails rather than widening the caller-owned row."""
    q = torch.zeros((2, 16, 512), dtype=torch.bfloat16)
    kv_cache = torch.zeros((2, 4352, 536), dtype=torch.uint8)
    topk_indices = torch.full((2, 1025), -1, dtype=torch.int32)
    with pytest.raises(ValueError, match="prefill"):
        run_unified_prefill(
            q=q,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            sm_scale=0.1,
            page_block_size=4352,
            model_type=int(ModelType.KVARN_MLA),
            scale_format=3,
        )


@pytest.mark.parametrize(
    ("model_type", "head_dim", "planned_width", "short_width"),
    [
        (ModelType.GLM_NSA, 576, 2048, 1024),
        (None, 576, 2048, 1024),
        (ModelType.GLM_NEXT, 512, 2051, 2048),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="sparse_mla plan/bind preparation requires a CUDA session",
)
def test_glm_binding_requires_the_planned_topk_width(
    model_type: int | None,
    head_dim: int,
    planned_width: int,
    short_width: int,
) -> None:
    rows = 2
    caps = sparse_mla.Caps(
        device="cuda",
        num_q_heads=8,
        max_q_rows=rows,
        max_width=planned_width,
        softmax_scale=256**-0.5,
        kv_dtype=torch.uint8,
        head_dim=head_dim,
        v_head_dim=512,
        page_size=64,
        model_type=model_type,
        mode="extend",
    )
    plan = sparse_mla.plan(caps)
    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device="cuda")
    q = torch.empty((rows, 8, head_dim), dtype=torch.bfloat16, device="cuda")
    cache_seqlens = torch.full((rows,), 4096, dtype=torch.int32, device="cuda")
    active_counts = torch.full((rows,), 128, dtype=torch.int32, device="cuda")
    kv_cache = torch.empty(
        (1, 64, caps.cache_record_bytes), dtype=torch.uint8, device="cuda"
    )

    with pytest.raises(
        ValueError, match=f"must match the planned top-k width {planned_width}"
    ):
        sparse_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=kv_cache,
            selected_indices=torch.empty(
                (rows, short_width), dtype=torch.int32, device="cuda"
            ),
            cache_lengths=cache_seqlens,
            selected_lengths=active_counts,
        )

    selected_indices = torch.empty(
        (rows, planned_width), dtype=torch.int32, device="cuda"
    )
    binding = sparse_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=kv_cache,
        selected_indices=selected_indices,
        cache_lengths=cache_seqlens,
        selected_lengths=active_counts,
    )
    assert binding.runtime.selected_indices is selected_indices


def test_glm_prefill_dispatch_keeps_the_bound_index_storage(monkeypatch) -> None:
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    captured_indices: list[torch.Tensor] = []
    captured_lengths: list[torch.Tensor] = []

    def fake_run_unified_prefill_mg(**kwargs) -> None:
        captured_indices.append(kwargs["topk_indices"])
        captured_lengths.append(kwargs["topk_length"])

    monkeypatch.setattr(
        prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg
    )

    rows, heads, planned_width = 2, 8, 2051
    q = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 528), dtype=torch.uint8)
    selected_indices = torch.empty((rows, planned_width), dtype=torch.int32)
    active_counts = torch.full((rows,), 128, dtype=torch.int32)
    output = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    lse = torch.empty((rows, heads), dtype=torch.float32)

    for live_count in (128, 2048):
        active_counts.fill_(live_count)
        actual_output, actual_lse = run_unified_prefill(
            q=q,
            kv_cache=kv_cache,
            topk_indices=selected_indices,
            topk_length=active_counts,
            sm_scale=256**-0.5,
            page_block_size=64,
            output=output,
            lse_out=lse,
            model_type=ModelType.GLM_NEXT,
        )

        assert actual_output is output
        assert actual_lse is lse

    assert len(captured_indices) == 2
    assert all(item is selected_indices for item in captured_indices)
    assert len(captured_lengths) == 2
    assert all(item is active_counts for item in captured_lengths)


def test_glm_nsa_prefill_dispatch_accepts_fixed_512_pool_tail_width(
    monkeypatch,
) -> None:
    """GLM_NSA accepts the fixed select_k=512-pool + tail container (2051).

    Upstream's fixed budget is topk_tokens//index_kpool = 2048//4 = 512 pools;
    expand_pools_and_append_tail widens that to 512*4+3 = 2051 tokens. The MG
    gate must admit 2051 for GLM_NSA (not only GLM_NEXT) and stream it via
    per-row topk_length.
    """
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    captured_topk: list[int] = []

    def fake_run_unified_prefill_mg(**kwargs) -> None:
        captured_topk.append(kwargs["topk_indices"].shape[1])

    monkeypatch.setattr(
        prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg
    )

    rows, heads, planned_width = 2, 8, 2051
    q = torch.empty((rows, heads, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 656), dtype=torch.uint8)
    selected_indices = torch.empty((rows, planned_width), dtype=torch.int32)
    active_counts = torch.full((rows,), 2051, dtype=torch.int32)
    output = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    lse = torch.empty((rows, heads), dtype=torch.float32)

    run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=selected_indices,
        topk_length=active_counts,
        sm_scale=256**-0.5,
        page_block_size=64,
        output=output,
        lse_out=lse,
        model_type=ModelType.GLM_NSA,
    )

    assert captured_topk == [2051]


@pytest.mark.parametrize(
    ("model_type", "head_dim", "record_bytes", "topk"),
    [
        # Non-kvarn GLM-5.3 DSA: GLM_NSA with the native 2048 selection width.
        (ModelType.GLM_NSA, 576, 656, 2048),
        # Non-kvarn GLM-5.3-Flash C4 pooled indexer: GLM_NEXT with the fixed
        # 2051 container (index_topk + index_kpool - 1 = 2048 + 3).
        (ModelType.GLM_NEXT, 512, 528, 2051),
        # Non-kvarn GLM_NEXT alignment-only container.
        (ModelType.GLM_NEXT, 512, 528, 2112),
    ],
)
def test_nonkvarn_glm_prefill_dispatch_unchanged(
    monkeypatch, model_type, head_dim, record_bytes, topk
) -> None:
    """Upstream GLM dispatch clauses admit non-kvarn widths unchanged.

    The kvarn restructure must not alter the non-kvarn GLM-5.3 DSA (GLM_NSA
    topk=2048) or GLM-5.3-Flash C4 (GLM_NEXT topk=2051/2112) paths: those
    widths are admitted by the upstream clauses byte-identical to clean b12x,
    not by the kvarn extended-container clause.
    """
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    captured_topk: list[int] = []

    def fake_run_unified_prefill_mg(**kwargs) -> None:
        captured_topk.append(kwargs["topk_indices"].shape[1])

    monkeypatch.setattr(
        prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg
    )

    rows, heads = 2, 8
    q = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, record_bytes), dtype=torch.uint8)
    selected_indices = torch.empty((rows, topk), dtype=torch.int32)
    active_counts = torch.full((rows,), topk, dtype=torch.int32)
    output = torch.empty((rows, heads, 512), dtype=torch.bfloat16)
    lse = torch.empty((rows, heads), dtype=torch.float32)

    run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=selected_indices,
        topk_length=active_counts,
        sm_scale=256**-0.5,
        page_block_size=64,
        output=output,
        lse_out=lse,
        model_type=model_type,
    )

    assert captured_topk == [topk]


def test_glm_prefill_rejects_an_unplanned_width_without_padding(monkeypatch) -> None:
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    def fail_pad(*args, **kwargs):
        raise AssertionError("GLM prefill must not allocate a padded index row")

    monkeypatch.setattr(torch.nn.functional, "pad", fail_pad)

    rows, heads, unplanned_width = 1, 8, 513
    with pytest.raises(ValueError, match="unsupported shape"):
        run_unified_prefill(
            q=torch.empty((rows, heads, 576), dtype=torch.bfloat16),
            kv_cache=torch.empty((1, 64, 656), dtype=torch.uint8),
            topk_indices=torch.empty((rows, unplanned_width), dtype=torch.int32),
            topk_length=torch.full((rows,), 128, dtype=torch.int32),
            sm_scale=512**-0.5,
            page_block_size=64,
            output=torch.empty((rows, heads, 512), dtype=torch.bfloat16),
            lse_out=torch.empty((rows, heads), dtype=torch.float32),
            model_type=ModelType.GLM_NSA,
        )
