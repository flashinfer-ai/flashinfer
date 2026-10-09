"""CPU-only host checks for the CAKE DSv4 NVFP4 route scaffold.

Covers the public guard pairing ``kv_cache_format`` with the resolved backend,
the Cake ``plan()`` reproduction (``_nvfp4_plan``), the workspace layout
extension and the explicit ``NotImplementedError`` raised while the generated
NVFP4 sources are not exported. No GPU is needed.
"""

import pytest

torch = pytest.importorskip("torch")

from flashinfer.jit.cake_dsv4 import _ARCH_REGISTRATIONS
from flashinfer.mla import cake_dsv4 as cake
from flashinfer.mla._core import _check_dsv4_kv_cache_format_backend
from flashinfer.mla.cake_dsv4 import (
    _nvfp4_plan,
    _require_nvfp4_variant,
    cake_dsv4_workspace_layout,
    get_cake_dsv4_workspace_bytes,
)

_NVFP4_VARIANTS = cake._NVFP4_VARIANTS


@pytest.mark.parametrize("backend", ["cake", "sparse"])
def test_nvfp4_format_accepts_cake_and_sparse(backend):
    _check_dsv4_kv_cache_format_backend("nvfp4", backend)


@pytest.mark.parametrize("backend", ["trtllm-gen", "cute-dsl"])
def test_nvfp4_format_rejects_other_backends(backend):
    with pytest.raises(ValueError, match="kv_cache_format='nvfp4'"):
        _check_dsv4_kv_cache_format_backend("nvfp4", backend)


@pytest.mark.parametrize("fmt", ["fp8_dsv41", "fp8_dsv41_fp4_ca"])
def test_dsv41_formats_still_sparse_only(fmt):
    _check_dsv4_kv_cache_format_backend(fmt, "sparse")
    with pytest.raises(ValueError, match="requires backend='sparse'"):
        _check_dsv4_kv_cache_format_backend(fmt, "cake")


def test_fp8_format_accepts_every_backend():
    for backend in ("trtllm-gen", "cute-dsl", "sparse", "cake"):
        _check_dsv4_kv_cache_format_backend("fp8", backend)
    with pytest.raises(ValueError, match="kv_cache_format must be"):
        _check_dsv4_kv_cache_format_backend("int4", "cake")


# (num_query_tokens, num_heads, sparse_topk, extra_topk, sm_count) -> (variant, num_splits, tiles_per_split, grid, merge_heads_per_cta)
# generated from the Cake production plans of the 76 contract rows of the DSv4 NVFP4 shape ledger at the exportable SM counts
_CAKE_PLAN_TABLE = {
    (1, 16, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 4, 1),
    (1, 16, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 4, 1),
    (1, 16, 512, 0, 148): ("nvfp4_decode_swap_n16_oc4", 4, 1, 16, 1),
    (1, 16, 512, 0, 152): ("nvfp4_decode_swap_n16_oc4", 4, 1, 16, 1),
    (1, 32, 128, 0, 148): ("nvfp4_decode_swap_n32_oc4", 1, 1, 4, 1),
    (1, 32, 128, 0, 152): ("nvfp4_decode_swap_n32_oc4", 1, 1, 4, 1),
    (1, 32, 512, 0, 148): ("nvfp4_decode_swap_n32_oc4", 4, 1, 16, 1),
    (1, 32, 512, 0, 152): ("nvfp4_decode_swap_n32_oc4", 4, 1, 16, 1),
    (1, 64, 128, 0, 148): ("nvfp4_decode_tile_oc4", 1, 1, 4, 1),
    (1, 64, 128, 0, 152): ("nvfp4_decode_tile_oc4", 1, 1, 4, 1),
    (1, 64, 512, 0, 148): ("nvfp4_decode_tile_oc4", 4, 1, 16, 1),
    (1, 64, 512, 0, 152): ("nvfp4_decode_tile_oc4", 4, 1, 16, 1),
    (1, 128, 128, 0, 148): ("nvfp4_decode_tile_oc4", 1, 1, 4, 1),
    (1, 128, 128, 0, 152): ("nvfp4_decode_tile_oc4", 1, 1, 4, 1),
    (1, 128, 512, 0, 148): ("nvfp4_decode_tile_oc4", 4, 1, 16, 1),
    (1, 128, 512, 0, 152): ("nvfp4_decode_tile_oc4", 4, 1, 16, 1),
    (3, 64, 128, 132, 148): ("nvfp4_decode_tile_oc4", 3, 1, 36, 2),
    (3, 64, 128, 132, 152): ("nvfp4_decode_tile_oc4", 3, 1, 36, 2),
    (4, 32, 128, 0, 148): ("nvfp4_decode_swap_n32_oc4", 1, 1, 16, 1),
    (4, 32, 128, 0, 152): ("nvfp4_decode_swap_n32_oc4", 1, 1, 16, 1),
    (5, 16, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 20, 1),
    (5, 16, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 20, 1),
    (5, 16, 128, 128, 148): ("nvfp4_decode_swap_n16_oc4", 2, 1, 40, 1),
    (5, 16, 128, 128, 152): ("nvfp4_decode_swap_n16_oc4", 2, 1, 40, 1),
    (7, 16, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 28, 1),
    (7, 16, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 28, 1),
    (8, 8, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 32, 1),
    (8, 8, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 32, 1),
    (8, 16, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 32, 1),
    (8, 16, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 32, 1),
    (8, 16, 128, 132, 148): ("nvfp4_decode_swap_n16_oc4", 3, 1, 96, 1),
    (8, 16, 128, 132, 152): ("nvfp4_decode_swap_n16_oc4", 3, 1, 96, 1),
    (8, 16, 128, 512, 148): ("nvfp4_decode_swap_n16_oc2", 5, 1, 80, 1),
    (8, 16, 128, 512, 152): ("nvfp4_decode_swap_n16_oc2", 5, 1, 80, 1),
    (8, 16, 512, 0, 148): ("nvfp4_decode_swap_n16_oc4", 4, 1, 128, 1),
    (8, 16, 512, 0, 152): ("nvfp4_decode_swap_n16_oc4", 4, 1, 128, 1),
    (8, 16, 512, 512, 148): ("nvfp4_decode_swap_n16_oc2", 8, 1, 128, 1),
    (8, 16, 512, 512, 152): ("nvfp4_decode_swap_n16_oc2", 8, 1, 128, 1),
    (8, 32, 128, 0, 148): ("nvfp4_decode_swap_n32_oc4", 1, 1, 32, 2),
    (8, 32, 128, 0, 152): ("nvfp4_decode_swap_n32_oc4", 1, 1, 32, 2),
    (8, 32, 512, 0, 148): ("nvfp4_decode_swap_n32_oc4", 4, 1, 128, 2),
    (8, 32, 512, 0, 152): ("nvfp4_decode_swap_n32_oc4", 4, 1, 128, 2),
    (8, 64, 128, 0, 148): ("nvfp4_decode_tile_oc4", 1, 1, 32, 4),
    (8, 64, 128, 0, 152): ("nvfp4_decode_tile_oc4", 1, 1, 32, 4),
    (8, 64, 512, 0, 148): ("nvfp4_decode_tile_oc2", 4, 1, 64, 4),
    (8, 64, 512, 0, 152): ("nvfp4_decode_tile_oc2", 4, 1, 64, 4),
    (8, 128, 128, 0, 148): ("nvfp4_decode_tile_oc4", 1, 1, 32, 8),
    (8, 128, 128, 0, 152): ("nvfp4_decode_tile_oc4", 1, 1, 32, 8),
    (8, 128, 128, 132, 148): ("nvfp4_decode_tile_oc2", 3, 1, 48, 8),
    (8, 128, 128, 132, 152): ("nvfp4_decode_tile_oc2", 3, 1, 48, 8),
    (8, 128, 128, 512, 148): ("nvfp4_decode_tile_oc1", 5, 1, 40, 8),
    (8, 128, 128, 512, 152): ("nvfp4_decode_tile_oc1", 5, 1, 40, 8),
    (8, 128, 512, 0, 148): ("nvfp4_decode_tile_oc2", 4, 1, 64, 8),
    (8, 128, 512, 0, 152): ("nvfp4_decode_tile_oc2", 4, 1, 64, 8),
    (8, 128, 512, 512, 148): ("nvfp4_decode_tile_oc1", 8, 1, 64, 8),
    (8, 128, 512, 512, 152): ("nvfp4_decode_tile_oc1", 8, 1, 64, 8),
    (9, 8, 128, 512, 148): ("nvfp4_decode_swap_n16_oc2", 5, 1, 90, 1),
    (9, 8, 128, 512, 152): ("nvfp4_decode_swap_n16_oc2", 5, 1, 90, 1),
    (12, 16, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 48, 2),
    (12, 16, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 48, 2),
    (12, 16, 128, 512, 148): ("nvfp4_decode_swap_n16_oc2", 5, 1, 120, 2),
    (12, 16, 128, 512, 152): ("nvfp4_decode_swap_n16_oc2", 5, 1, 120, 2),
    (12, 32, 128, 0, 148): ("nvfp4_decode_swap_n32_oc4", 1, 1, 48, 4),
    (12, 32, 128, 0, 152): ("nvfp4_decode_swap_n32_oc4", 1, 1, 48, 4),
    (12, 32, 128, 512, 148): ("nvfp4_decode_swap_n32_oc2", 5, 1, 120, 4),
    (12, 32, 128, 512, 152): ("nvfp4_decode_swap_n32_oc2", 5, 1, 120, 4),
    (12, 64, 128, 0, 148): ("nvfp4_decode_tile_oc4", 1, 1, 48, 8),
    (12, 64, 128, 0, 152): ("nvfp4_decode_tile_oc4", 1, 1, 48, 8),
    (12, 64, 128, 512, 148): ("nvfp4_decode_tile_oc1", 5, 1, 60, 8),
    (12, 64, 128, 512, 152): ("nvfp4_decode_tile_oc1", 5, 1, 60, 8),
    (12, 128, 128, 0, 148): ("nvfp4_decode_tile_oc4", 1, 1, 48, 16),
    (12, 128, 128, 0, 152): ("nvfp4_decode_tile_oc4", 1, 1, 48, 16),
    (12, 128, 128, 512, 148): ("nvfp4_decode_tile_oc1", 5, 1, 60, 16),
    (12, 128, 128, 512, 152): ("nvfp4_decode_tile_oc1", 5, 1, 60, 16),
    (32, 8, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 128, 2),
    (32, 8, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 128, 2),
    (32, 16, 128, 0, 148): ("nvfp4_decode_swap_n16_oc4", 1, 1, 128, 4),
    (32, 16, 128, 0, 152): ("nvfp4_decode_swap_n16_oc4", 1, 1, 128, 4),
    (32, 16, 128, 132, 148): ("nvfp4_decode_swap_n16_oc1", 3, 1, 96, 4),
    (32, 16, 128, 132, 152): ("nvfp4_decode_swap_n16_oc1", 3, 1, 96, 4),
    (32, 16, 128, 512, 148): ("nvfp4_decode_pv_n16_oc1", 3, 2, 96, 4),
    (32, 16, 128, 512, 152): ("nvfp4_decode_pv_n16_oc1", 3, 2, 96, 4),
    (32, 16, 256, 0, 148): ("nvfp4_decode_swap_n16_oc2", 2, 1, 128, 4),
    (32, 16, 256, 0, 152): ("nvfp4_decode_swap_n16_oc2", 2, 1, 128, 4),
    (32, 16, 512, 0, 148): ("nvfp4_decode_swap_n16_oc1", 4, 1, 128, 4),
    (32, 16, 512, 0, 152): ("nvfp4_decode_swap_n16_oc1", 4, 1, 128, 4),
    (32, 16, 512, 512, 148): ("nvfp4_decode_pv_n16_oc1", 4, 2, 128, 4),
    (32, 16, 512, 512, 152): ("nvfp4_decode_pv_n16_oc1", 4, 2, 128, 4),
    (32, 32, 128, 0, 148): ("nvfp4_decode_swap_n32_oc4", 1, 1, 128, 8),
    (32, 32, 128, 0, 152): ("nvfp4_decode_swap_n32_oc4", 1, 1, 128, 8),
    (32, 32, 512, 0, 148): ("nvfp4_decode_swap_n32_oc1", 4, 1, 128, 8),
    (32, 32, 512, 0, 152): ("nvfp4_decode_swap_n32_oc1", 4, 1, 128, 8),
    (32, 64, 128, 0, 148): ("nvfp4_decode_tile_oc2", 1, 1, 64, 16),
    (32, 64, 128, 0, 152): ("nvfp4_decode_tile_oc2", 1, 1, 64, 16),
    (32, 64, 512, 0, 148): ("nvfp4_decode_tile_oc1", 4, 1, 128, 16),
    (32, 64, 512, 0, 152): ("nvfp4_decode_tile_oc1", 4, 1, 128, 16),
    (32, 128, 128, 0, 148): ("nvfp4_decode_tile_oc2", 1, 1, 64, 16),
    (32, 128, 128, 0, 152): ("nvfp4_decode_tile_oc2", 1, 1, 64, 16),
    (32, 128, 128, 132, 148): ("nvfp4_decode_t64_n64_oc1", 1, 3, 64, 16),
    (32, 128, 128, 132, 152): ("nvfp4_decode_t64_n64_oc1", 1, 3, 64, 16),
    (32, 128, 128, 512, 148): ("nvfp4_decode_t64_n64_oc1", 1, 5, 64, 16),
    (32, 128, 128, 512, 152): ("nvfp4_decode_t64_n64_oc1", 1, 5, 64, 16),
    (32, 128, 256, 0, 148): ("nvfp4_decode_t64_n64_oc1", 1, 2, 64, 16),
    (32, 128, 256, 0, 152): ("nvfp4_decode_t64_n64_oc1", 1, 2, 64, 16),
    (32, 128, 512, 0, 148): ("nvfp4_decode_tile_oc1", 4, 1, 128, 16),
    (32, 128, 512, 0, 152): ("nvfp4_decode_tile_oc1", 4, 1, 128, 16),
    (32, 128, 512, 512, 148): ("nvfp4_decode_persistent", 4, 2, 128, 16),
    (32, 128, 512, 512, 152): ("nvfp4_decode_persistent", 4, 2, 128, 16),
    (72, 128, 256, 0, 148): ("nvfp4_decode_cluster", 1, 2, 144, 16),
    (72, 128, 256, 0, 152): ("nvfp4_decode_cluster", 1, 2, 144, 16),
    (128, 16, 128, 0, 148): ("nvfp4_decode_swap_n16_oc1", 1, 1, 128, 16),
    (128, 16, 128, 0, 152): ("nvfp4_decode_swap_n16_oc1", 1, 1, 128, 16),
    (128, 16, 512, 0, 148): ("nvfp4_decode_pv_n16_oc1", 1, 4, 128, 16),
    (128, 16, 512, 0, 152): ("nvfp4_decode_pv_n16_oc1", 1, 4, 128, 16),
    (128, 32, 128, 0, 148): ("nvfp4_decode_swap_n32_oc1", 1, 1, 128, 16),
    (128, 32, 128, 0, 152): ("nvfp4_decode_swap_n32_oc1", 1, 1, 128, 16),
    (128, 32, 512, 0, 148): ("nvfp4_decode_pv_n32_oc1", 1, 4, 128, 16),
    (128, 32, 512, 0, 152): ("nvfp4_decode_pv_n32_oc1", 1, 4, 128, 16),
    (128, 64, 128, 0, 148): ("nvfp4_decode_tile_oc1", 1, 1, 128, 16),
    (128, 64, 128, 0, 152): ("nvfp4_decode_tile_oc1", 1, 1, 128, 16),
    (128, 64, 512, 0, 148): ("nvfp4_decode_t64_n64_oc1", 1, 4, 128, 16),
    (128, 64, 512, 0, 152): ("nvfp4_decode_t64_n64_oc1", 1, 4, 128, 16),
    (128, 128, 128, 0, 148): ("nvfp4_decode_tile_oc1", 1, 1, 128, 16),
    (128, 128, 128, 0, 152): ("nvfp4_decode_tile_oc1", 1, 1, 128, 16),
    (128, 128, 512, 0, 148): ("nvfp4_decode_persistent", 1, 4, 128, 16),
    (128, 128, 512, 0, 152): ("nvfp4_decode_persistent", 1, 4, 128, 16),
    (128, 128, 512, 512, 148): ("nvfp4_decode_persistent", 1, 8, 128, 16),
    (128, 128, 512, 512, 152): ("nvfp4_decode_persistent", 1, 8, 128, 16),
}


@pytest.mark.parametrize("key", sorted(_CAKE_PLAN_TABLE))
def test_nvfp4_plan_matches_cake_plan(key):
    tokens, num_heads, topk, extra_topk, sm_count = key
    variant, expected_splits, expected_tiles_per_split, expected_grid, expected_hpc = (
        _CAKE_PLAN_TABLE[key]
    )
    plan = _nvfp4_plan(
        num_query_tokens=tokens,
        num_heads=num_heads,
        sparse_topk=topk,
        extra_topk=extra_topk,
        sm_count=sm_count,
    )
    assert plan.num_main_tiles == -(-topk // 128)
    assert plan.num_extra_tiles == -(-extra_topk // 128)
    assert plan.total_tiles == plan.num_main_tiles + plan.num_extra_tiles
    assert (
        plan.variant,
        plan.num_splits,
        plan.tiles_per_split,
        plan.grid,
        plan.merge_heads_per_cta,
    ) == (
        variant,
        expected_splits,
        expected_tiles_per_split,
        expected_grid,
        expected_hpc,
    )
    assert plan.variant in cake._NVFP4_DECODE_VARIANTS
    # Every split owns tiles_per_split tiles and the splits cover all tiles.
    assert plan.tiles_per_split * (plan.num_splits - 1) < plan.total_tiles
    assert plan.tiles_per_split * plan.num_splits >= plan.total_tiles
    head_tile = 64 if plan.member == "t64" else 128
    assert plan.num_head_tiles == -(-num_heads // head_tile)
    assert plan.merge_groups == tokens * plan.num_head_tiles
    assert plan.merge_grid == (tokens, -(-num_heads // plan.merge_heads_per_cta), 1)
    if plan.num_splits > 1:
        assert (
            tokens * -(-num_heads // plan.merge_heads_per_cta) <= sm_count
            or plan.merge_heads_per_cta == 16
        )


def test_nvfp4_variant_names():
    assert cake._nvfp4_variant_name("persistent") == "nvfp4_decode_persistent"
    assert (
        cake._nvfp4_variant_name("swap", tile_n=16, o_chunks=4)
        == "nvfp4_decode_swap_n16_oc4"
    )
    assert cake._nvfp4_variant_name("tile", o_chunks=2) == "nvfp4_decode_tile_oc2"
    assert (
        cake._nvfp4_variant_name("t64", tile_n=64, o_chunks=1)
        == "nvfp4_decode_t64_n64_oc1"
    )
    with pytest.raises(ValueError, match="variant knob"):
        cake._nvfp4_variant_name("pv", o_chunks=1)
    with pytest.raises(ValueError, match="unknown NVFP4 family member"):
        cake._nvfp4_variant_name("merge")
    assert len(set(_NVFP4_VARIANTS)) == len(_NVFP4_VARIANTS) == 15


def test_nvfp4_plan_head_tiles_and_caps():
    # Head tiles are 128 wide (64 on the tile64 member); one-tile rows form one head tile on every member.
    for heads in (8, 16, 32, 64, 128):
        plan = _nvfp4_plan(
            num_query_tokens=4,
            num_heads=heads,
            sparse_topk=128,
            extra_topk=0,
            sm_count=148,
        )
        assert plan.num_head_tiles == 1
        assert plan.member == ("swap" if heads <= 32 else "tile")
    with pytest.raises(ValueError, match="num_heads"):
        _nvfp4_plan(
            num_query_tokens=4,
            num_heads=48,
            sparse_topk=128,
            extra_topk=0,
            sm_count=148,
        )
    # 13 candidate tiles exceed the split cap (12) of every member.
    with pytest.raises(ValueError, match="exceeds the NVFP4 route cap"):
        _nvfp4_plan(
            num_query_tokens=1,
            num_heads=128,
            sparse_topk=13 * 128,
            extra_topk=0,
            sm_count=148,
        )
    with pytest.raises(ValueError, match="extra_topk"):
        _nvfp4_plan(
            num_query_tokens=1,
            num_heads=128,
            sparse_topk=128,
            extra_topk=-1,
            sm_count=148,
        )


def test_workspace_layout_lse_region_is_opt_in():
    base = cake_dsv4_workspace_layout(32, 128, 4)
    with_lse = cake_dsv4_workspace_layout(32, 128, 4, with_lse=True)
    assert base.lse == (base.total_bytes, 0)
    assert (base.partial_o, base.partial_lse) == (
        with_lse.partial_o,
        with_lse.partial_lse,
    )
    assert with_lse.lse[0] == base.total_bytes
    assert with_lse.lse[1] == -(-(32 * 128 * 4) // 128) * 128
    assert with_lse.total_bytes == base.total_bytes + with_lse.lse[1]
    assert with_lse.lse[0] % 128 == 0


def test_workspace_bytes_nvfp4_bound():
    fp8_bytes = get_cake_dsv4_workspace_bytes(32, 128, 512, torch.bfloat16)
    assert fp8_bytes == cake_dsv4_workspace_layout(32, 128, 5).total_bytes
    nvfp4_bytes = get_cake_dsv4_workspace_bytes(
        32, 128, 512, torch.bfloat16, kv_cache_format="nvfp4", extra_topk=512
    )
    assert (
        nvfp4_bytes == cake_dsv4_workspace_layout(32, 128, 8, with_lse=True).total_bytes
    )
    capped = get_cake_dsv4_workspace_bytes(
        1, 128, 2048, torch.bfloat16, kv_cache_format="nvfp4", extra_topk=2048
    )
    assert capped == cake_dsv4_workspace_layout(1, 128, 12, with_lse=True).total_bytes
    pinned = get_cake_dsv4_workspace_bytes(
        32, 128, 512, torch.bfloat16, kv_cache_format="nvfp4", num_splits=2
    )
    assert pinned == cake_dsv4_workspace_layout(32, 128, 2, with_lse=True).total_bytes
    with pytest.raises(ValueError, match="BF16 query"):
        get_cake_dsv4_workspace_bytes(
            32, 128, 512, torch.float8_e4m3fn, kv_cache_format="nvfp4"
        )


def test_nvfp4_argument_vocabulary():
    for name in (
        "q_rows",
        "main_cache",
        "extra_cache",
        "main_indices",
        "extra_indices",
        "main_lengths",
        "extra_lengths",
        "lse_out",
        "partial_O",
        "partial_lse",
    ):
        assert cake.is_bindable_arg("buffer", name)
    assert cake.canonical_arg_name("tma_buffer", "tmap_out") == "partial_O_tiles"
    assert cake.is_bindable_arg("tma_buffer", "tmap_out")
    assert cake.is_bindable_arg("tma_buffer", "tmap_q")
    for name, canonical in (
        ("tmap_g4d", "main_cache_g4d"),
        ("tmap_g4f", "main_cache_g4f"),
        ("tmap_g4dx", "extra_cache_g4d"),
        ("tmap_g4fx", "extra_cache_g4f"),
    ):
        assert cake.canonical_arg_name("tma_buffer", name) == canonical
        assert cake.is_bindable_arg("tma_buffer", name)
    assert cake.is_bindable_arg("parameter", "heads_per_cta")
    for name in (
        "tiles_per_split",
        "total_tiles",
        "main_page_stride",
        "lse_partial_scale",
        "lse_scale",
    ):
        assert cake.is_bindable_arg("parameter", name)
    values = {"lse_scale": 0.5, "total_tiles": 3}
    assert (
        cake._bind_argument(
            values, "parameter", "lse_scale", variant="v", grid={}, descriptor_slab=None
        )
        == 0.5
    )
    assert (
        cake._bind_argument(
            values,
            "parameter",
            "total_tiles",
            variant="v",
            grid={},
            descriptor_slab=None,
        )
        == 3
    )
    with pytest.raises(TypeError, match="must be an int"):
        cake._bind_argument(
            {"total_tiles": 2.0},
            "parameter",
            "total_tiles",
            variant="v",
            grid={},
            descriptor_slab=None,
        )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("variant", _NVFP4_VARIANTS)
def test_missing_nvfp4_variant_raises_not_implemented(monkeypatch, arch, variant):
    # The registration dict is shared with the cached metadata view, so
    # removing the key (if a later commit registered it) restores the
    # pre-export state for this test.
    monkeypatch.delitem(_ARCH_REGISTRATIONS[arch]["variants"], variant, raising=False)
    with pytest.raises(NotImplementedError, match="not yet exported"):
        _require_nvfp4_variant(variant, arch=arch)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_registered_variant_passes_the_guard(arch):
    registered = next(iter(_ARCH_REGISTRATIONS[arch]["variants"]))
    _require_nvfp4_variant(registered, arch=arch)
