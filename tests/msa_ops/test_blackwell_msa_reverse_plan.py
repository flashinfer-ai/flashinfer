"""CPU-only tests for standalone exact reverse-prefill planning."""

import ast
from pathlib import Path

import pytest
import torch

from flashinfer.msa_ops import _blackwell_sm100_reverse_plan as plan


def _frozen_q2k(shape, seed: int) -> torch.Tensor:
    q2k = torch.full(
        (shape.num_kv_heads, shape.total_q, shape.topk),
        -1,
        dtype=torch.int32,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    blocks = (shape.seqlen_kv + 127) // 128
    for batch_idx in range(shape.batch_size):
        q_base = batch_idx * shape.seqlen_q
        offset = shape.seqlen_kv - shape.seqlen_q
        for local_q in range(shape.seqlen_q):
            visible = min(blocks, (offset + local_q + 128) // 128)
            for head in range(shape.num_kv_heads):
                selected = (
                    torch.randperm(visible, generator=generator)[: shape.topk]
                    .sort()
                    .values
                )
                q2k[head, q_base + local_q] = selected.to(torch.int32)
    return q2k.contiguous()


def test_topk8_qagg_plan_has_exact_geometry_and_cohorts() -> None:
    shape = plan._FP8_TOPK8_SHAPE
    q2k = _frozen_q2k(shape, 71)
    for sm_count in (148, 160):
        value = plan.build_fp8_topk8_qagg_plan(q2k, sm_count=sm_count)
        geometry = value["geometry"]
        assert (geometry.schedule_capacity, geometry.work_count) == (677, 384)
        assert value["scheduler_metadata"].shape == (677, 6)
        assert value["k2q_row_ptr"].shape == (2, 193)
        assert value["k2q_qsplit_indices"].shape == (2, 24576)
        assert value["split_counts"].shape == (3072, 2)
        assert torch.all(value["split_counts"] == 8)
        assert value["q_order"].shape == (3072,)
        assert torch.equal(
            torch.sort(value["q_order"]).values,
            torch.arange(3072, dtype=torch.int32),
        )
        assert value["contributor_work_ids"].shape == (3072, 2, 8)


def test_topk4_qload4_plan_has_exact_geometry_and_segments() -> None:
    shape = plan._BF16_TOPK4_SHAPE
    q2k = _frozen_q2k(shape, 73)
    for sm_count in (148, 160):
        value = plan.build_bf16_paged_topk4_plan(q2k, sm_count=sm_count)
        geometry = value["geometry"]
        assert (
            geometry.schedule_capacity,
            geometry.work_count,
            geometry.target_q_per_cta,
        ) == (640, 389, 384)
        assert value["scheduler_metadata"].shape == (640, 6)
        assert value["k2q_row_ptr"].shape == (2, 193)
        assert value["k2q_qsplit_indices"].shape == (2, 49152)
        assert value["split_counts"].shape == (12288, 2)
        assert torch.all(value["split_counts"] == 4)
        assert len(value["group_segment_ends"]) == 20
        assert tuple(value["group_segment_ends"]) == tuple(
            sorted(value["group_segment_ends"])
        )


def test_standalone_planner_does_not_import_source_generation_packages() -> None:
    source = plan.__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    import_roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    import_roots.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert import_roots <= {"__future__", "dataclasses", "typing", "torch"}


def test_qagg_cache_hit_does_not_snapshot_cuda_metadata(monkeypatch) -> None:
    q2k = torch.empty((2, 3072, 8), dtype=torch.int32)
    cu_q = torch.empty((4,), dtype=torch.int32)
    cu_k = torch.empty((4,), dtype=torch.int32)
    owners = (q2k, cu_q, cu_k)
    state = {
        "signature": plan._uploaded_plan_signature(
            "fp8_topk8_qagg_pdl",
            sm_count=148,
            stream_id=17,
            owners=owners,
        ),
        "owners": owners,
    }
    monkeypatch.setattr(plan, "_require_cuda_i32", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plan,
        "_require_exact_values",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit attempted a host snapshot")
        ),
    )
    assert (
        plan.prepare_fp8_topk8_qagg_plan(
            q2k,
            cu_q,
            cu_k,
            sm_count=148,
            stream_id=17,
            state=state,
        )
        is state
    )


def test_warmed_topk4_capture_hit_does_not_read_host(monkeypatch) -> None:
    q2k = torch.empty((2, 12288, 4), dtype=torch.int32)
    cu_q = torch.empty((4,), dtype=torch.int32)
    cu_k = torch.empty((4,), dtype=torch.int32)
    page_table = torch.empty((3, 64), dtype=torch.int32)
    seqused_k = torch.empty((3,), dtype=torch.int32)
    owners = (q2k, cu_q, cu_k, page_table, seqused_k)
    state = {
        "signature": plan._uploaded_plan_signature(
            "bf16_paged_topk4_qload4",
            sm_count=160,
            stream_id=23,
            owners=owners,
        ),
        "owners": owners,
    }
    monkeypatch.setattr(plan, "_require_cuda_i32", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plan,
        "_require_exact_values",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("warmed capture hit attempted a host snapshot")
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert (
        plan.prepare_bf16_paged_topk4_plan(
            q2k,
            cu_q,
            cu_k,
            page_table,
            seqused_k,
            sm_count=160,
            stream_id=23,
            state=state,
        )
        is state
    )


def test_qagg_capture_miss_fails_before_host_snapshot(monkeypatch) -> None:
    q2k = torch.empty((2, 3072, 8), dtype=torch.int32)
    cu_q = torch.empty((4,), dtype=torch.int32)
    cu_k = torch.empty((4,), dtype=torch.int32)
    monkeypatch.setattr(plan, "_require_cuda_i32", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plan,
        "_require_exact_values",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("capture miss attempted a host snapshot")
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prepared before CUDA graph capture"):
        plan.prepare_fp8_topk8_qagg_plan(
            q2k,
            cu_q,
            cu_k,
            sm_count=148,
            stream_id=29,
            state={},
        )
