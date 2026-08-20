"""CPU-only contract tests for the SM100/SM103 TopK16 serving portfolio."""

from argparse import Namespace
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from benchmarks import bench_blackwell_msa_sm100 as benchmark


def test_manifest_freezes_the_serving_portfolio():
    stable_ids = tuple(shape.stable_id for shape in benchmark.SHAPE_MANIFEST)

    assert stable_ids == benchmark.FROZEN_SHAPE_IDS
    assert len(stable_ids) == 13
    assert tuple(benchmark.SHAPES_BY_ID) == stable_ids

    with pytest.raises(FrozenInstanceError):
        benchmark.SHAPE_MANIFEST[0].seed = 0


def test_manifest_names_public_semantic_entrypoints():
    assert benchmark.SEMANTIC_ENTRYPOINTS == (
        "flashinfer.msa_ops.msa_sparse_attention",
        "flashinfer.msa_ops.msa_sparse_decode_attention",
    )


def test_manifest_records_complete_provenance_and_baseline_support():
    assert set(benchmark.CORRECTNESS_TOLERANCES) == {
        "bfloat16",
        "float16",
        "float8_e4m3fn",
    }
    for shape in benchmark.SHAPE_MANIFEST:
        assert shape.stable_id
        assert shape.source
        assert shape.provenance
        assert shape.selection_rationale
        assert shape.selection_mode == "random_valid_bottom_right_causal"

        if shape.q_dtype == "float16":
            assert shape.baseline_mode == "candidate_only_fp16"
            assert not shape.baseline_comparable
        else:
            assert shape.baseline_mode == "minimax_public"
            assert shape.baseline_comparable

    official_rows = [
        shape
        for shape in benchmark.SHAPE_MANIFEST
        if shape.source.startswith("minimax_official_")
    ]
    assert official_rows
    assert all("MiniMax-AI/MSA@80434d7f" in shape.provenance for shape in official_rows)


def test_manifest_covers_audited_shape_axes_and_route_complements():
    shapes = benchmark.SHAPE_MANIFEST

    assert {shape.kv_layout for shape in shapes} == {"flat_varlen", "paged"}
    assert {shape.topk for shape in shapes} == {16}
    assert {shape.batch_size for shape in shapes}.issuperset({1, 2, 3, 32, 64, 128})
    assert {shape.seqlen_q for shape in shapes}.issuperset({1, 4, 8, 16, 1024, 4096})
    assert {shape.seqlen_kv for shape in shapes}.issuperset({1921, 4096, 8192, 65536})
    assert {shape.num_q_heads // shape.num_kv_heads for shape in shapes}.issuperset(
        {4, 8, 16}
    )

    routes = {
        (shape.operation, shape.q_dtype, shape.kv_dtype, shape.kv_layout)
        for shape in shapes
    }
    assert routes.issuperset(
        {
            ("sparse_decode", "bfloat16", "bfloat16", "paged"),
            ("sparse_decode", "float16", "float16", "paged"),
            ("sparse_decode", "bfloat16", "float8_e4m3fn", "flat_varlen"),
            ("sparse_prefill", "bfloat16", "float8_e4m3fn", "flat_varlen"),
        }
    )


def test_shape_filter_defaults_to_all_and_preserves_manifest_order():
    assert benchmark._selected_shapes(Namespace(shapes=None)) is (
        benchmark.SHAPE_MANIFEST
    )

    requested = [benchmark.FROZEN_SHAPE_IDS[-1], benchmark.FROZEN_SHAPE_IDS[0]]
    selected = benchmark._selected_shapes(Namespace(shapes=requested))
    assert tuple(shape.stable_id for shape in selected) == (
        benchmark.FROZEN_SHAPE_IDS[0],
        benchmark.FROZEN_SHAPE_IDS[-1],
    )

    with pytest.raises(ValueError, match="duplicate stable IDs"):
        benchmark._selected_shapes(Namespace(shapes=[requested[0], requested[0]]))


def test_public_shape_schema_exposes_stable_metadata():
    shape = benchmark.SHAPE_MANIFEST[-1]
    public = benchmark._public_shape(shape)

    assert public["label"] == shape.stable_id
    assert public["stable_id"] == shape.stable_id
    assert public["source"] == shape.source
    assert public["provenance"] == shape.provenance
    assert public["selection_rationale"] == shape.selection_rationale
    assert public["baseline_mode"] == "minimax_public"


def test_paged_cache_builder_pads_partial_final_page_without_changing_tokens():
    seqlen_kv = 1921
    num_kv_heads = 1
    head_dim = 2
    block_size = 128
    logical = torch.arange(seqlen_kv * head_dim, dtype=torch.float32).reshape(
        seqlen_kv, num_kv_heads, head_dim
    )

    paged, page_table = benchmark._make_paged_cache(
        torch,
        logical,
        batch_size=1,
        seqlen_kv=seqlen_kv,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )

    logical_pages = paged[page_table[0].long()].permute(0, 2, 1, 3)
    reconstructed = logical_pages.reshape(-1, num_kv_heads, head_dim)
    torch.testing.assert_close(reconstructed[:seqlen_kv], logical)
    assert torch.count_nonzero(reconstructed[seqlen_kv:]).item() == 0


def test_independent_reference_is_layout_invariant_for_partial_fp16_page():
    base = benchmark.SHAPE_MANIFEST[4]
    common = {
        "batch_size": 1,
        "seqlen_q": 2,
        "seqlen_kv": 1921,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "topk": 16,
    }
    flat = replace(base, kv_layout="flat_varlen", **common)
    paged = replace(base, kv_layout="paged", **common)
    flat_inputs = benchmark._make_inputs(torch, flat, torch.device("cpu"))
    paged_inputs = benchmark._make_inputs(torch, paged, torch.device("cpu"))

    flat_output = benchmark._candidate_reference_output(torch, flat, flat_inputs)
    paged_output = benchmark._candidate_reference_output(torch, paged, paged_inputs)

    torch.testing.assert_close(flat_output, paged_output, atol=0, rtol=0)
