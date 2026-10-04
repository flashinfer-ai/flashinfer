# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake SM120 DeepSeek-V4.1 mixed-cache sparse-MLA decode.

``backend="cake"`` with ``kv_cache_format="fp8_dsv41_fp4_ca"`` (public API) or
``SparseMLASm120Wrapper(backend="cake", kv_cache_format="fp8",
kv_scale_format="ue8m0_g32", extra_kv_fp4=True)``: a 528-byte FP8 + UE8M0
group-32 main cache and a 288-byte V41_FP4 extra cache.

The CPU half checks the host contract without a GPU: format facts, the
footer-layout page-geometry rules for both caches, planner purity and
invariants, compute-precision normalisation and the public-API / wrapper
gates (compute capability monkeypatched). The GPU half compares the generated
family against the fp32 reference over the exactly dequantized caches across
the hardening list (page sizes 61/53/48/2, padded page strides, all ``-1``,
lengths 0/1/63/65, odd token counts, empty rows with and without sinks,
HND/NHD/3-D views, ``lse_scale``, workspace reuse across shapes, graph
replay, poisoned slot 0, head counts outside the mandatory set) and skips
until the generated family is present in the tree and an SM120/SM121 device
is available. Tolerances follow the kernel's acceptance table: the BF16 route
at BF16 level (output and LSE ``atol = rtol = 1e-2``), the FP8 route at FP8
level (``0.1``).
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math

import pytest
import torch

import flashinfer
import flashinfer.mla._core as mla_core
from flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed import (
    cake_sparse_mla_sm120_dsv41_mixed_available,
    cake_sparse_mla_sm120_dsv41_mixed_manifest,
)
from flashinfer.mla import (
    SparseMLASm120Wrapper,
    dsv41_fp4_quantize_pack_sparse_mla_cache,
    dsv41_fp8_quantize_pack_sparse_mla_cache,
)
from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as route
from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
    BINDING_PARAMS,
    EXTRA_BYTES_PER_TOKEN,
    MAIN_BYTES_PER_TOKEN,
    PROVISIONAL_GEOMETRY,
    cache_page_geometry,
    cake_sparse_mla_sm120_dsv41_mixed_decode,
    cake_sparse_mla_sm120_dsv41_mixed_format_info,
    cake_sparse_mla_sm120_dsv41_mixed_num_chunks,
    cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles,
    cake_sparse_mla_sm120_dsv41_mixed_plan_splits,
    cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes,
    cake_sparse_mla_sm120_dsv41_mixed_supported_heads,
    get_cake_sparse_mla_sm120_dsv41_mixed_module,
    kernel_geometry,
    normalize_compute_precision,
)
from flashinfer.utils import is_sm12x_supported
from tests.attention.sparse_mla_test_utils import (
    _reference_sparse_attention,
    dequantize_kv_dsv4_1,
    dequantize_kv_dsv4_1_fp4,
)

_D = 512
_SM_SCALE = _D**-0.5
_CPU = torch.device("cpu")
# Wrapper keywords of the mixed cache; ``backend`` and ``compute_precision`` vary per test.
_MIXED = dict(kv_cache_format="fp8", kv_scale_format="ue8m0_g32", extra_kv_fp4=True)
_TOL = {
    "bf16": dict(atol=1e-2, rtol=1e-2),
    "fp8": dict(atol=1e-1, rtol=1e-1),
}
_LSE_TOL = {
    "bf16": dict(atol=1e-2, rtol=1e-2),
    "fp8": dict(atol=1e-1, rtol=1e-1),
}
_LAZY_EXPORTS = (
    "cake_sparse_mla_sm120_dsv41_mixed_decode",
    "cake_sparse_mla_sm120_dsv41_mixed_format_info",
    "cake_sparse_mla_sm120_dsv41_mixed_num_chunks",
    "cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles",
    "cake_sparse_mla_sm120_dsv41_mixed_plan_splits",
    "cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes",
    "cake_sparse_mla_sm120_dsv41_mixed_supported_heads",
)
# Geometry with two-tile instances, to exercise the head-tile rules the provisional geometry disables.
_TWO_TILE_GEOMETRY = dataclasses.replace(
    PROVISIONAL_GEOMETRY,
    head_counts=(8, 16, 32, 48, 64, 80, 96, 112, 128),
    two_tile_head_counts=(32, 64, 96, 128),
)


# ---------------------------------------------------------------------------
# CPU: format facts, geometry rules, planners, gates.
# ---------------------------------------------------------------------------


def test_format_info() -> None:
    info = cake_sparse_mla_sm120_dsv41_mixed_format_info()
    assert info["query_dim"] == _D and info["value_dim"] == _D
    assert info["main_bytes_per_token"] == MAIN_BYTES_PER_TOKEN == 528
    assert info["main_data_bytes"] + info["main_scale_bytes"] == 528
    assert (
        info["main_data_bytes"] // info["main_scale_group"] == info["main_scale_bytes"]
    )
    assert info["extra_bytes_per_token"] == EXTRA_BYTES_PER_TOKEN == 288
    assert info["extra_data_bytes"] + info["extra_scale_bytes"] == 288
    assert _D // info["extra_scale_group"] == info["extra_scale_bytes"]
    assert info["runtime_page"] and info["runtime_extra_page"]
    assert set(info["heads"]) >= {8, 16, 32, 64, 128}
    assert cake_sparse_mla_sm120_dsv41_mixed_supported_heads() == info["heads"]
    assert set(info["two_tile_heads"]) <= set(info["heads"])
    assert info["default_compute_precision"] == "bf16"
    assert "bf16" in info["compute_precisions"]
    assert set(info["compute_precisions"]) <= {"bf16", "fp8"}
    assert info["kernels_available"] == cake_sparse_mla_sm120_dsv41_mixed_available()
    assert info["heads_per_block"] >= 1 and info["max_chunks_per_block"] >= 1
    assert info["chunk_width"] >= 1 and info["extra_chunk_width"] >= 1


def test_num_chunks_counts_both_caches() -> None:
    chunks = functools.partial(
        cake_sparse_mla_sm120_dsv41_mixed_num_chunks, geometry=PROVISIONAL_GEOMETRY
    )
    assert chunks(128) == 2
    assert chunks(1) == 1
    assert chunks(130, 1) == 4
    assert chunks(128, 512) == 10
    assert chunks(0, 512) == 8
    assert chunks(512, 1024) == 24
    narrow = dataclasses.replace(
        PROVISIONAL_GEOMETRY, candidates_per_chunk=32, extra_candidates_per_chunk=128
    )
    assert cake_sparse_mla_sm120_dsv41_mixed_num_chunks(128, 512, geometry=narrow) == 8
    assert cake_sparse_mla_sm120_dsv41_mixed_num_chunks(130, 129, geometry=narrow) == 7


def test_normalize_compute_precision() -> None:
    assert normalize_compute_precision("default") == "bf16"
    assert normalize_compute_precision("bf16") == "bf16"
    assert normalize_compute_precision("fp8") == "fp8"
    for bad in ("nvfp4", "auto", "fp16", ""):
        with pytest.raises(ValueError, match="compute_precision"):
            normalize_compute_precision(bad)


@pytest.mark.parametrize("bytes_per_token", [528, 288])
@pytest.mark.parametrize("page_size", [1, 2, 48, 53, 61, 64, 128])
def test_cache_page_geometry_accepts_footer_views(
    bytes_per_token: int, page_size: int
) -> None:
    pages = 3
    payload = page_size * bytes_per_token
    flat = torch.empty(pages, payload, dtype=torch.uint8)
    expect = (pages, page_size, payload)
    views = (
        flat,
        flat.view(pages, page_size, bytes_per_token),
        flat.view(pages, 1, page_size, bytes_per_token),
        flat.view(pages, page_size, 1, bytes_per_token),
    )
    for view in views:
        assert (
            cache_page_geometry(
                view.shape, view.stride(), bytes_per_token=bytes_per_token, name="c"
            )
            == expect
        )
    # Padded pools: the page stride exceeds the payload by a 16-byte multiple.
    stride = payload + 32
    padded = torch.empty(pages * stride, dtype=torch.uint8).as_strided(
        (pages, 1, page_size, bytes_per_token), (stride, stride, bytes_per_token, 1)
    )
    assert cache_page_geometry(
        padded.shape, padded.stride(), bytes_per_token=bytes_per_token, name="c"
    ) == (pages, page_size, stride)


def test_cache_page_geometry_rejects_bad_views() -> None:
    bpt = 528
    geometry = functools.partial(cache_page_geometry, bytes_per_token=bpt, name="kv")
    # Rows of another format (the 384-byte NVFP4 cache) in a 4-D view.
    with pytest.raises(ValueError, match="528"):
        geometry((2, 1, 64, 384), (64 * 384, 64 * 384, 384, 1))
    # 2-D page bytes that are not a whole number of rows.
    with pytest.raises(ValueError, match="multiple of 528"):
        geometry((2, 64 * bpt + 8), (64 * bpt + 8, 1))
    # Per-token record stride: the footer layout keeps rows packed inside a page.
    with pytest.raises(ValueError, match="packed"):
        geometry((2, 1, 64, bpt), (64 * 544, 64 * 544, 544, 1))
    # Overlapping pages.
    with pytest.raises(ValueError, match="smaller than"):
        geometry((2, 1, 64, bpt), (32 * bpt, 32 * bpt, bpt, 1))
    # Page stride that is not a 16-byte multiple.
    with pytest.raises(ValueError, match="multiple of 16"):
        geometry((2, 1, 64, bpt), (64 * bpt + 8, 64 * bpt + 8, bpt, 1))
    # Two latent heads.
    with pytest.raises(ValueError, match="singleton"):
        geometry((2, 2, 64, bpt), (2 * 64 * bpt, 64 * bpt, bpt, 1))
    # Strided byte dimension.
    with pytest.raises(ValueError, match="contiguous"):
        geometry((2, 64 * bpt), (2 * 64 * bpt, 2))
    # Ranks the route does not accept.
    with pytest.raises(ValueError):
        geometry((2 * 64 * bpt,), (1,))
    with pytest.raises(ValueError):
        geometry((2, 1, 1, 64, bpt), (64 * bpt, 64 * bpt, 64 * bpt, bpt, 1))
    with pytest.raises(ValueError, match="rank"):
        geometry((2, 1, 64, bpt), (64 * bpt, bpt, 1))
    # Empty pools.
    with pytest.raises(ValueError, match="at least one page"):
        geometry((0, 64 * bpt), (64 * bpt, 1))
    # The extra cache rejects main-cache rows and vice versa.
    with pytest.raises(ValueError, match="288"):
        cache_page_geometry(
            (2, 1, 64, 528), (64 * 528, 64 * 528, 528, 1), bytes_per_token=288, name="x"
        )
    with pytest.raises(ValueError, match="528"):
        cache_page_geometry(
            (2, 1, 64, 288), (64 * 288, 64 * 288, 288, 1), bytes_per_token=528, name="m"
        )


def test_plan_head_tiles_rules() -> None:
    plan = functools.partial(
        cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles, geometry=_TWO_TILE_GEOMETRY
    )
    # The tile count is fixed per head count by the exported kernels: two tiles for every head count
    # the manifest lists, one tile otherwise, whatever the token / chunk counts.
    for heads in (8, 16, 48, 80, 112):
        for tokens, topk in ((1, 128), (128, 512), (8, 1024)):
            assert plan(num_tokens=tokens, num_heads=heads, topk=topk, num_sms=188) == 1
    for heads in (32, 64, 96, 128):
        for tokens, topk, sms in (
            (1, 128, 188),
            (128, 512, 170),
            (8, 1024, 48),
            (16, 64, 128),
        ):
            assert plan(num_tokens=tokens, num_heads=heads, topk=topk, num_sms=sms) == 2
    # The provisional geometry exports no two-tile instance at all.
    for heads in (32, 64, 128):
        assert (
            cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles(
                num_tokens=128,
                num_heads=heads,
                topk=512,
                extra_topk=512,
                num_sms=188,
                geometry=PROVISIONAL_GEOMETRY,
            )
            == 1
        )
    with pytest.raises(ValueError, match="num_sms"):
        plan(num_tokens=1, num_heads=8, topk=128, num_sms=0)


# Exported-kernel geometry for the planner tests: two tiles from 32 heads up, a 16-chunk index table.
_FITTED_GEOMETRY = _TWO_TILE_GEOMETRY

# Fitted to the Cake split sweeps (S = 1, 2, 3, 4, 5, 8, 10 timed per row) on RTX PRO 6000 (188 SMs),
# RTX 5090 (170 SMs) and GB10 (48 SMs, unified memory); the kernel module's planner test asserts the same table.
# (num_tokens, num_heads, topk, extra_topk, num_sms, unified_memory) -> (num_splits, chunks_per_block)
_FITTED_PLANS = {
    (1, 64, 128, 512, 188, False): (10, 1),
    (8, 64, 128, 512, 188, False): (10, 1),
    (32, 64, 128, 512, 188, False): (3, 4),
    (64, 64, 128, 512, 188, False): (1, 10),
    (128, 64, 128, 512, 188, False): (2, 5),
    (8, 8, 128, 512, 188, False): (10, 1),
    (32, 8, 128, 512, 188, False): (5, 2),
    (128, 8, 128, 512, 188, False): (1, 10),
    (8, 16, 128, 512, 188, False): (10, 1),
    (32, 16, 128, 512, 188, False): (5, 2),
    (128, 16, 128, 512, 188, False): (1, 10),
    (8, 32, 128, 512, 188, False): (10, 1),
    (32, 32, 128, 512, 188, False): (5, 2),
    (128, 32, 128, 512, 188, False): (1, 10),
    (8, 128, 128, 512, 188, False): (5, 2),
    (32, 128, 128, 512, 188, False): (1, 10),
    (128, 128, 128, 512, 188, False): (1, 10),
    (8, 64, 128, 0, 188, False): (2, 1),
    (32, 64, 128, 0, 188, False): (1, 2),
    (128, 64, 128, 0, 188, False): (1, 2),
    (1, 64, 128, 512, 170, False): (10, 1),
    (8, 64, 128, 512, 170, False): (10, 1),
    (32, 64, 128, 512, 170, False): (3, 4),
    (64, 64, 128, 512, 170, False): (1, 10),
    (128, 64, 128, 512, 170, False): (3, 4),
    (8, 8, 128, 512, 170, False): (10, 1),
    (32, 8, 128, 512, 170, False): (5, 2),
    (128, 8, 128, 512, 170, False): (1, 10),
    (8, 16, 128, 512, 170, False): (10, 1),
    (32, 16, 128, 512, 170, False): (5, 2),
    (128, 16, 128, 512, 170, False): (1, 10),
    (8, 32, 128, 512, 170, False): (10, 1),
    (32, 32, 128, 512, 170, False): (5, 2),
    (128, 32, 128, 512, 170, False): (1, 10),
    (8, 128, 128, 512, 170, False): (5, 2),
    (32, 128, 128, 512, 170, False): (1, 10),
    (128, 128, 128, 512, 170, False): (1, 10),
    (8, 64, 128, 0, 170, False): (2, 1),
    (32, 64, 128, 0, 170, False): (1, 2),
    (128, 64, 128, 0, 170, False): (1, 2),
    (1, 64, 128, 512, 48, True): (5, 2),
    (8, 64, 128, 512, 48, True): (2, 5),
    (32, 64, 128, 512, 48, True): (2, 5),
    (64, 64, 128, 512, 48, True): (1, 10),
    (128, 64, 128, 512, 48, True): (1, 10),
    (8, 8, 128, 512, 48, True): (2, 5),
    (32, 8, 128, 512, 48, True): (1, 10),
    (128, 8, 128, 512, 48, True): (1, 10),
    (8, 16, 128, 512, 48, True): (2, 5),
    (32, 16, 128, 512, 48, True): (1, 10),
    (128, 16, 128, 512, 48, True): (1, 10),
    (8, 32, 128, 512, 48, True): (2, 5),
    (32, 32, 128, 512, 48, True): (1, 10),
    (128, 32, 128, 512, 48, True): (1, 10),
    (8, 128, 128, 512, 48, True): (1, 10),
    (32, 128, 128, 512, 48, True): (1, 10),
    (128, 128, 128, 512, 48, True): (1, 10),
    (8, 64, 128, 0, 48, True): (1, 2),
    (32, 64, 128, 0, 48, True): (1, 2),
    (128, 64, 128, 0, 48, True): (1, 2),
}


def test_plan_splits_matches_the_fitted_sweep_table() -> None:
    for (
        tokens,
        heads,
        topk,
        extra_topk,
        sms,
        unified,
    ), expected in _FITTED_PLANS.items():
        tiles = cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles(
            num_tokens=tokens,
            num_heads=heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=sms,
            geometry=_FITTED_GEOMETRY,
        )
        plan = cake_sparse_mla_sm120_dsv41_mixed_plan_splits(
            num_tokens=tokens,
            num_heads=heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=sms,
            head_tiles=tiles,
            unified_memory=unified,
            geometry=_FITTED_GEOMETRY,
        )
        assert plan == expected, (tokens, heads, topk, extra_topk, sms, unified)


def test_plan_splits_rules() -> None:
    plan = functools.partial(
        cake_sparse_mla_sm120_dsv41_mixed_plan_splits, geometry=_FITTED_GEOMETRY
    )
    # Two chunks split only on a discrete-memory card whose grid is below a tenth of the SMs.
    assert plan(num_tokens=1, num_heads=8, topk=128, num_sms=188) == (2, 1)
    assert plan(num_tokens=18, num_heads=8, topk=128, num_sms=188) == (2, 1)
    assert plan(num_tokens=19, num_heads=8, topk=128, num_sms=188) == (1, 2)
    assert plan(num_tokens=1, num_heads=8, topk=128, num_sms=188, max_splits=1) == (
        1,
        2,
    )
    assert plan(num_tokens=128, num_heads=128, topk=128, num_sms=188) == (1, 2)
    assert plan(
        num_tokens=1, num_heads=8, topk=128, num_sms=48, unified_memory=True
    ) == (1, 2)
    # Discrete-memory SM120: the fewest chunks per CTA whose split grid stays within ~1.15 waves ...
    assert plan(num_tokens=1, num_heads=16, topk=512, num_sms=188) == (8, 1)
    assert plan(num_tokens=16, num_heads=16, topk=512, num_sms=188) == (8, 1)
    assert plan(num_tokens=32, num_heads=16, topk=512, num_sms=188) == (4, 2)
    assert plan(num_tokens=8, num_heads=128, topk=512, num_sms=170, head_tiles=2) == (
        4,
        2,
    )
    assert plan(num_tokens=64, num_heads=16, topk=512, num_sms=188) == (3, 3)
    assert plan(
        num_tokens=32, num_heads=64, topk=128, extra_topk=512, num_sms=170, head_tiles=2
    ) == (3, 4)
    # ... and otherwise unsplit, except a >= 8-chunk full grid: the smallest of two or three splits whose
    # grid ends in a half-full wave (256 x 2 = 512 CTAs: 136 of 188 on the PRO 6000, 2 of 170 on the 5090;
    # 256 x 3 = 768 CTAs: 88 of 170 on the 5090).
    assert plan(num_tokens=128, num_heads=16, topk=512, num_sms=188) == (1, 8)
    assert plan(num_tokens=256, num_heads=16, topk=512, num_sms=188) == (2, 4)
    assert plan(num_tokens=256, num_heads=16, topk=512, num_sms=170) == (3, 3)
    assert plan(num_tokens=512, num_heads=16, topk=512, num_sms=170) == (1, 8)
    # Unified memory (GB10): doubling only while the grid stays small, never one chunk per CTA.
    gb10 = functools.partial(plan, num_sms=48, unified_memory=True)
    assert gb10(num_tokens=1, num_heads=16, topk=512) == (4, 2)
    assert gb10(num_tokens=8, num_heads=16, topk=512) == (2, 4)
    assert gb10(num_tokens=16, num_heads=16, topk=512) == (2, 4)
    assert gb10(num_tokens=32, num_heads=16, topk=512) == (1, 8)
    # ... except a >= 8-chunk grid between one and 4/3 waves unsplit, which runs in two splits.
    assert gb10(num_tokens=64, num_heads=16, topk=512) == (2, 4)
    assert gb10(num_tokens=49, num_heads=16, topk=512) == (2, 4)
    assert gb10(num_tokens=65, num_heads=16, topk=512) == (1, 8)
    assert gb10(num_tokens=48, num_heads=16, topk=512) == (1, 8)
    # Beyond the index table every grid splits; a max_splits that cannot honour it raises.
    splits, cpb = plan(
        num_tokens=128, num_heads=128, topk=512, extra_topk=1024, num_sms=188
    )
    assert cpb <= 16 and splits * cpb >= 24 and splits >= 2
    with pytest.raises(ValueError, match="at most 16"):
        plan(
            num_tokens=128,
            num_heads=128,
            topk=512,
            extra_topk=1024,
            num_sms=188,
            max_splits=1,
        )
    assert plan(num_tokens=4, num_heads=128, topk=512, num_sms=188, max_splits=1) == (
        1,
        8,
    )
    # Two head tiles halve the grid the split planner sees.
    assert plan(
        num_tokens=8, num_heads=32, topk=128, extra_topk=512, num_sms=188, head_tiles=2
    ) == plan(num_tokens=8, num_heads=16, topk=128, extra_topk=512, num_sms=188)
    with pytest.raises(ValueError, match="head_tiles"):
        plan(num_tokens=8, num_heads=64, topk=512, num_sms=188, head_tiles=3)
    with pytest.raises(ValueError, match="at least one candidate"):
        plan(num_tokens=8, num_heads=64, topk=0, num_sms=188)
    with pytest.raises(ValueError, match="num_sms"):
        plan(num_tokens=8, num_heads=64, topk=512, num_sms=0)


def test_planners_are_pure_and_bounded() -> None:
    """Same inputs, same plan; every plan honours the index table and covers every chunk."""

    g = PROVISIONAL_GEOMETRY
    grid = itertools.product(
        (1, 5, 8, 32, 128),
        g.head_counts,
        (1, 64, 128, 130, 512, 1024),
        (0, 2, 512, 1024),
        (48, 170, 188),
    )
    for num_tokens, num_heads, topk, extra_topk, num_sms in grid:
        kwargs = dict(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=num_sms,
            geometry=g,
        )
        chunks = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(
            topk, extra_topk, geometry=g
        )
        tiles = cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles(**kwargs)
        assert tiles == cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles(**kwargs) == 1
        plan = cake_sparse_mla_sm120_dsv41_mixed_plan_splits(**kwargs, head_tiles=tiles)
        assert plan == cake_sparse_mla_sm120_dsv41_mixed_plan_splits(
            **kwargs, head_tiles=tiles
        )
        splits, cpb = plan
        assert 1 <= cpb <= g.max_chunks_per_block, kwargs
        assert splits == -(-chunks // cpb) and (splits - 1) * cpb < chunks, kwargs
        assert -(-chunks // g.max_chunks_per_block) <= splits <= 16, kwargs
        if chunks == 1:
            assert splits == 1, kwargs
        if chunks == 2:
            # two chunks split only when the unsplit grid is below a tenth of the SMs (discrete memory)
            head_blocks = -(-num_heads // g.heads_per_block)
            assert splits == (2 if num_tokens * head_blocks * 10 <= num_sms else 1), (
                kwargs
            )
        scratch = cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes(
            num_tokens, num_heads, topk, extra_topk
        )
        rows = num_tokens * num_heads
        assert scratch >= rows * splits * (_D * 2 + 4) + rows * 4, kwargs


def test_scratch_bytes_formula() -> None:
    chunks = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(512, 0)
    assert (
        cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes(2, 128, 512)
        == 2 * 128 * chunks * (_D * 2 + 4) + 2 * 128 * 4 + 48
    )


def test_wrapper_accepts_cake_mixed_cache() -> None:
    for word, expect in (("default", "bf16"), ("bf16", "bf16"), ("fp8", "fp8")):
        wrapper = SparseMLASm120Wrapper(
            backend="cake", compute_precision=word, device=_CPU, **_MIXED
        )
        assert wrapper._cake_compute_precision == expect
    # Other backends and the NVFP4 Cake route keep their own precision handling.
    for backend in ("auto", "sparse"):
        wrapper = SparseMLASm120Wrapper(
            backend=backend, compute_precision="bf16", device=_CPU, **_MIXED
        )
        assert wrapper._cake_compute_precision is None
    nvfp4 = SparseMLASm120Wrapper(backend="cake", kv_cache_format="nvfp4", device=_CPU)
    assert nvfp4._cake_compute_precision is None


def test_wrapper_rejects_incomplete_cake_mixed_cache() -> None:
    # The FP8 default without the mixed-cache flags names both Cake formats.
    with pytest.raises(ValueError, match="kv_cache_format='nvfp4'"):
        SparseMLASm120Wrapper(backend="cake", device=_CPU)
    with pytest.raises(ValueError, match="extra_kv_fp4=True"):
        SparseMLASm120Wrapper(
            backend="cake",
            kv_cache_format="fp8",
            kv_scale_format="ue8m0_g32",
            device=_CPU,
        )
    with pytest.raises(ValueError, match="ue8m0_g32"):
        SparseMLASm120Wrapper(
            backend="cake", kv_cache_format="fp8", kv_scale_format="auto", device=_CPU
        )
    with pytest.raises(ValueError, match="d_v"):
        SparseMLASm120Wrapper(backend="cake", d_v=1024, device=_CPU, **_MIXED)
    with pytest.raises(ValueError, match="nvfp4"):
        SparseMLASm120Wrapper(
            backend="cake", compute_precision="nvfp4", device=_CPU, **_MIXED
        )


def test_wrapper_run_checks_call_shape_before_planning() -> None:
    wrapper = SparseMLASm120Wrapper(
        max_num_tokens=2, max_num_heads=16, backend="cake", device=_CPU, **_MIXED
    )
    q = torch.zeros(2, 16, _D, dtype=torch.bfloat16)
    cache = torch.zeros(1, 1, 64, MAIN_BYTES_PER_TOKEN, dtype=torch.uint8)
    indices = torch.zeros(2, 128, dtype=torch.int32)
    with pytest.raises(ValueError, match="decode-only"):
        wrapper.run(q, cache, indices, torch.empty_like(q), 1.0, prefill_impl="swapab")
    with pytest.raises(ValueError, match=r"\[T,H,D\]"):
        wrapper.run(q[0], cache, indices, torch.empty_like(q[0]), 1.0)
    big = torch.zeros(3, 16, _D, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_num_tokens"):
        wrapper.run(big, cache, torch.zeros(3, 128, dtype=torch.int32), big, 1.0)


def _public_api_call(
    backend: str,
    *,
    kv_cache_format: str = "fp8_dsv41_fp4_ca",
    num_heads: int = 64,
    **extra,
):
    """CPU metadata call: the SM120 dispatch is monkeypatched or raises before touching a device."""

    q = torch.zeros(2, num_heads, _D, dtype=torch.bfloat16)
    return flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=torch.zeros(4, 1, 64, MAIN_BYTES_PER_TOKEN, dtype=torch.uint8),
        workspace_buffer=torch.empty(1, dtype=torch.int8),
        sparse_indices=torch.zeros(2, 128, dtype=torch.int32),
        compressed_kv_cache=torch.zeros(
            4, 1, 64, EXTRA_BYTES_PER_TOKEN, dtype=torch.uint8
        ),
        swa_topk_lens=torch.full((2,), 128, dtype=torch.int32),
        extra_sparse_indices=torch.zeros(2, 512, dtype=torch.int32),
        extra_sparse_topk_lens=torch.full((2,), 512, dtype=torch.int32),
        bmm1_scale=_SM_SCALE,
        kv_cache_format=kv_cache_format,
        backend=backend,
        **extra,
    )


def test_public_api_gate_dispatches_cake_mixed_on_sm120(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_sm120(**kwargs):
        calls.append(kwargs)
        return kwargs["query"]

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        mla_core, "_trtllm_batch_decode_sparse_mla_dsv4_sm120", fake_sm120
    )
    _public_api_call("cake")
    assert calls[-1]["backend"] == "cake"
    assert calls[-1]["kv_cache_format"] == "fp8_dsv41_fp4_ca"
    # Default routing is unchanged: "auto" and "sparse" keep the hand-written SM120 kernels.
    _public_api_call("auto")
    assert calls[-1]["backend"] == "sparse"
    _public_api_call("sparse")
    assert calls[-1]["backend"] == "sparse"
    assert len(calls) == 3
    # SM121 is part of the family.
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda device: (12, 1))
    _public_api_call("cake")
    assert calls[-1]["backend"] == "cake"


def test_public_api_gate_rejects_cake_mixed_off_sm120(monkeypatch) -> None:
    for cc in ((10, 0), (10, 3)):
        monkeypatch.setattr(
            mla_core, "get_compute_capability", lambda device, cc=cc: cc
        )
        with pytest.raises(ValueError, match="backend='cake' on SM120/SM121"):
            _public_api_call("cake")
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda device: (9, 0))
    with pytest.raises(ValueError, match="backend='cake' requires"):
        _public_api_call("cake")


def test_public_api_cake_rejects_other_fp8_forms_and_head_counts(monkeypatch) -> None:
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda device: (12, 0))
    with pytest.raises(ValueError, match="fp8_dsv41_fp4_ca"):
        _public_api_call("cake", kv_cache_format="fp8")
    with pytest.raises(ValueError, match="requires backend='sparse'"):
        _public_api_call("cake", kv_cache_format="fp8_dsv41")
    unsupported = next(
        h
        for h in (24, 40, 48, 72)
        if h not in cake_sparse_mla_sm120_dsv41_mixed_supported_heads()
    )
    with pytest.raises(ValueError, match="query heads"):
        _public_api_call("cake", num_heads=unsupported)


def test_public_api_enable_pdl_gate(monkeypatch) -> None:
    """The mixed-cache Cake route forwards ``enable_pdl`` (None = device default); the NVFP4 Cake route still rejects it."""

    calls: list[dict] = []

    def fake_sm120(**kwargs):
        calls.append(kwargs)
        return kwargs["query"]

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        mla_core, "_trtllm_batch_decode_sparse_mla_dsv4_sm120", fake_sm120
    )
    _public_api_call("cake")
    assert calls[-1]["enable_pdl"] is None
    _public_api_call("cake", enable_pdl=True)
    assert calls[-1]["enable_pdl"] is True
    _public_api_call("cake", enable_pdl=False)
    assert calls[-1]["enable_pdl"] is False
    # The hand-written SM120 kernels do not take the flag.
    _public_api_call("sparse", enable_pdl=True)
    assert calls[-1]["enable_pdl"] is None
    with pytest.raises(ValueError, match="does not support enable_pdl"):
        _public_api_call("cake", kv_cache_format="nvfp4", enable_pdl=True)
    assert route._resolve_enable_pdl(True, _CPU) is True
    assert route._resolve_enable_pdl(False, _CPU) is False
    assert route._resolve_enable_pdl(None, _CPU) is False  # no PDL on a CPU device


def test_core_selects_functional_route_by_format() -> None:
    from flashinfer.mla._sparse_mla_sm120 import _cake_dsv4_nvfp4

    assert (
        mla_core._cake_sm120_functional_run("fp8_dsv41_fp4_ca") is route.functional_run
    )
    assert (
        mla_core._cake_sm120_functional_run("nvfp4") is _cake_dsv4_nvfp4.functional_run
    )


def test_lazy_exports() -> None:
    assert set(_LAZY_EXPORTS) <= set(dir(flashinfer.mla))
    for name in _LAZY_EXPORTS:
        assert getattr(flashinfer.mla, name) is getattr(route, name)
    missing = "cake_sparse_mla_sm120_dsv41_mixed_missing"
    with pytest.raises(AttributeError):
        getattr(flashinfer.mla, missing)


def test_binding_params_are_a_well_formed_contract() -> None:
    assert len(set(BINDING_PARAMS)) == len(BINDING_PARAMS)
    assert BINDING_PARAMS[:3] == ("q", "kv_cache", "indices")
    for name in ("extra_kv_cache", "extra_indices", "mid_out", "mid_lse", "precision"):
        assert name in BINDING_PARAMS
    assert (
        "head_tiles" not in BINDING_PARAMS
    )  # no two-tile instance in the mixed-cache family
    assert BINDING_PARAMS.index("page_size") < BINDING_PARAMS.index("extra_page_size")
    assert BINDING_PARAMS[-1] == "enable_pdl"  # launch attribute of the split merge


def test_generated_family_manifest_or_absence() -> None:
    """Before the export the family is reported absent with a precise error; after it, manifest and host agree."""

    if not cake_sparse_mla_sm120_dsv41_mixed_available():
        with pytest.raises(
            FileNotFoundError, match="cake_sparse_mla_dsv41_mixed_manifest.json"
        ):
            cake_sparse_mla_sm120_dsv41_mixed_manifest()
        with pytest.raises(FileNotFoundError):
            get_cake_sparse_mla_sm120_dsv41_mixed_module()
        assert kernel_geometry().provisional
        assert not cake_sparse_mla_sm120_dsv41_mixed_format_info()["kernels_available"]
        return
    manifest = cake_sparse_mla_sm120_dsv41_mixed_manifest()
    geometry = kernel_geometry()
    assert not geometry.provisional
    assert tuple(int(h) for h in manifest["head_counts"]) == geometry.head_counts
    assert tuple(manifest.get("binding_params", BINDING_PARAMS)) == BINDING_PARAMS
    assert int(manifest["main_bytes_per_token"]) == MAIN_BYTES_PER_TOKEN
    assert int(manifest["extra_bytes_per_token"]) == EXTRA_BYTES_PER_TOKEN
    assert "bf16" in manifest["precisions"]
    assert manifest["entry"] and "nvfp4" not in manifest["entry"]
    assert manifest["sources"], "the manifest must list the translation units"
    for name in manifest["sources"]:
        assert name.endswith((".cu", ".h")) and "nvfp4" not in name, name


# ---------------------------------------------------------------------------
# GPU: correctness of the generated family (skips until it is present).
# ---------------------------------------------------------------------------


def _require_family() -> None:
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("SM120/SM121 GPU required")
    if not cake_sparse_mla_sm120_dsv41_mixed_available():
        pytest.skip(
            "generated Cake SM120 DSv4.1 mixed-cache family not present in this tree"
        )


def _require_precision(precision: str) -> None:
    if precision not in kernel_geometry().precisions:
        pytest.skip(f"compute_precision={precision!r} not exported by this family")


def _latent(num_pages: int, page_size: int) -> torch.Tensor:
    return (
        torch.randn(num_pages, page_size, _D, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)


def _main_pool(num_pages: int, page_size: int) -> torch.Tensor:
    return dsv41_fp8_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))


def _extra_pool(num_pages: int, page_size: int) -> torch.Tensor:
    return dsv41_fp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))


def _query(num_tokens: int, num_heads: int) -> torch.Tensor:
    return (
        torch.randn(num_tokens, num_heads, _D, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)


def _indices(num_tokens: int, topk: int, num_slots: int) -> torch.Tensor:
    return torch.randint(
        0, num_slots, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )


def _lengths(num_tokens: int, topk: int, *, low: int | None = None) -> torch.Tensor:
    low = topk // 2 if low is None else low
    return torch.randint(low, topk + 1, (num_tokens,), dtype=torch.int32, device="cuda")


def _mask_tail(indices: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    ref = indices.clone()
    if lengths is not None:
        for token in range(indices.shape[0]):
            ref[token, int(lengths[token].item()) :] = -1
    return ref


def _dequant_main(cache_hnd: torch.Tensor) -> torch.Tensor:
    """HND ``[P, 1, page, 528]`` -> bf16 rows ``[P * page, 512]``."""

    return dequantize_kv_dsv4_1(cache_hnd.permute(0, 2, 1, 3).contiguous()).reshape(
        -1, _D
    )


def _dequant_extra(cache_hnd: torch.Tensor) -> torch.Tensor:
    return dequantize_kv_dsv4_1_fp4(cache_hnd.permute(0, 2, 1, 3).contiguous()).reshape(
        -1, _D
    )


def _reference(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    *,
    main_lengths: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    lse_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 attention over the exactly dequantized caches; LSE (fp32) in base 2 times ``lse_scale``."""

    kv = _dequant_main(main_cache).float()
    ref_indices = _mask_tail(main_indices, main_lengths)
    if extra_cache is not None:
        assert extra_indices is not None
        main_rows = kv.shape[0]
        kv = torch.cat((kv, _dequant_extra(extra_cache).float()), dim=0)
        ref_extra = _mask_tail(extra_indices, extra_lengths)
        ref_indices = torch.cat(
            (ref_indices, torch.where(ref_extra < 0, ref_extra, ref_extra + main_rows)),
            dim=1,
        )
    # Poisoned (non-finite) rows are never valid candidates (the kernel zero-fills masked slots); keep
    # them out of the reference's P x V, where a 0 x NaN would otherwise poison every output row.
    kv = torch.nan_to_num(kv, nan=0.0, posinf=0.0, neginf=0.0)
    output, lse = _reference_sparse_attention(
        q.float(), kv.reshape(1, -1, 1, _D), ref_indices, _SM_SCALE, attn_sink=attn_sink
    )
    return output, (lse * lse_scale).float()


def _wrapper(precision: str = "bf16", **kwargs) -> SparseMLASm120Wrapper:
    return SparseMLASm120Wrapper(
        backend="cake",
        compute_precision=precision,
        device=torch.device("cuda"),
        **_MIXED,
        **kwargs,
    )


def _run(
    wrapper: SparseMLASm120Wrapper,
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    *,
    main_lengths: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    lse_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(q)
    lse = wrapper.run(
        q,
        main_cache,
        main_indices,
        output,
        _SM_SCALE,
        topk_length=main_lengths,
        attn_sink=attn_sink,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        return_lse=True,
        lse_scale=lse_scale,
    )
    assert lse is not None
    return output, lse.clone()


def _padded_view(cache_hnd: torch.Tensor, pad_bytes: int) -> torch.Tensor:
    """Copy an HND cache into a pool whose page stride exceeds the page payload."""

    num_pages, _, page_size, bpt = cache_hnd.shape
    page_bytes = page_size * bpt
    stride = page_bytes + pad_bytes
    storage = torch.full(
        (num_pages * stride + 16,), 0xAB, dtype=torch.uint8, device="cuda"
    )
    base = (-storage.data_ptr()) % 16
    rows = storage[base : base + num_pages * stride].view(num_pages, stride)
    rows[:, :page_bytes] = cache_hnd.reshape(num_pages, page_bytes)
    return rows.as_strided(
        (num_pages, 1, page_size, bpt), (stride, page_bytes, bpt, 1), base
    )


def _layout_view(cache_hnd: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "HND":
        return cache_hnd
    if layout == "NHD":
        return cache_hnd.permute(0, 2, 1, 3)
    if layout == "3D":
        return cache_hnd.squeeze(1)
    if layout == "padded":
        return _padded_view(cache_hnd, 128)
    raise ValueError(layout)


def _check(precision: str, actual: tuple, expected: tuple) -> None:
    torch.testing.assert_close(actual[0], expected[0], **_TOL[precision])
    torch.testing.assert_close(actual[1], expected[1], **_LSE_TOL[precision])


@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("page_size", [61, 64])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64, 128])
def test_gpu_main_only_matches_reference(
    num_heads: int, page_size: int, with_sink: bool
) -> None:
    _require_family()
    torch.manual_seed(20261101 + num_heads + page_size)
    num_tokens, topk, pages = 8, 128, 16
    q = _query(num_tokens, num_heads)
    cache = _main_pool(pages, page_size)
    indices = _indices(num_tokens, topk, pages * page_size)
    lengths = _lengths(num_tokens, topk)
    sink = torch.randn(num_heads, device="cuda") if with_sink else None
    actual = _run(_wrapper(), q, cache, indices, main_lengths=lengths, attn_sink=sink)
    expected = _reference(q, cache, indices, main_lengths=lengths, attn_sink=sink)
    _check("bf16", actual, expected)


@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize(
    "main_page,extra_page",
    [(64, 64), (32, 64), (64, 32), (61, 53), (64, 2), (48, 48), (128, 128)],
)
def test_gpu_dual_cache_matches_reference(
    num_heads: int, main_page: int, extra_page: int
) -> None:
    _require_family()
    torch.manual_seed(20261102 + num_heads + main_page + extra_page)
    num_tokens, topk, extra_topk = 8, 128, 512
    main_pages = max(8, -(-4096 // main_page))
    extra_pages = max(8, -(-4096 // extra_page))
    q = _query(num_tokens, num_heads)
    cache = _main_pool(main_pages, main_page)
    extra = _extra_pool(extra_pages, extra_page)
    indices = _indices(num_tokens, topk, main_pages * main_page)
    extra_indices = _indices(num_tokens, extra_topk, extra_pages * extra_page)
    # Ragged lengths on both caches plus explicit -1 tails inside the active range.
    lengths = _lengths(num_tokens, topk)
    extra_lengths = _lengths(num_tokens, extra_topk)
    indices[:, topk - 8 :] = -1
    extra_indices[::2, :16] = -1
    sink = torch.randn(num_heads, device="cuda")
    actual = _run(
        _wrapper(),
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=sink,
    )
    expected = _reference(
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=sink,
    )
    _check("bf16", actual, expected)


@pytest.mark.parametrize("length", [0, 1, 63, 65])
def test_gpu_length_edge_values(length: int) -> None:
    _require_family()
    torch.manual_seed(20261103 + length)
    num_tokens, num_heads, topk, extra_topk = 4, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(16, 64)
    extra = _extra_pool(16, 64)
    indices = _indices(num_tokens, topk, 16 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 16 * 64)
    lengths = torch.full((num_tokens,), length, dtype=torch.int32, device="cuda")
    extra_lengths = torch.full((num_tokens,), length, dtype=torch.int32, device="cuda")
    for sink in (None, torch.randn(num_heads, device="cuda")):
        actual = _run(
            _wrapper(),
            q,
            cache,
            indices,
            main_lengths=lengths,
            extra_cache=extra,
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
            attn_sink=sink,
        )
        expected = _reference(
            q,
            cache,
            indices,
            main_lengths=lengths,
            extra_cache=extra,
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
            attn_sink=sink,
        )
        _check("bf16", actual, expected)


@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize(
    "mode", ["main_empty", "extra_empty", "both_empty", "all_minus_one"]
)
def test_gpu_empty_rows(mode: str, with_sink: bool) -> None:
    """Empty rows: zero output, LSE -inf, or the sink's mass alone when a sink is present."""

    _require_family()
    torch.manual_seed(20261104)
    num_tokens, num_heads, topk, extra_topk = 3, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(8, 64)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64)
    zero = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    full = torch.full((num_tokens,), topk, dtype=torch.int32, device="cuda")
    extra_full = torch.full((num_tokens,), extra_topk, dtype=torch.int32, device="cuda")
    lengths, extra_lengths = full, extra_full
    if mode == "main_empty":
        lengths = zero
    elif mode == "extra_empty":
        extra_lengths = zero
    elif mode == "both_empty":
        lengths, extra_lengths = zero, zero
    else:
        indices.fill_(-1)
        extra_indices.fill_(-1)
    sink = torch.randn(num_heads, device="cuda") if with_sink else None
    lse_scale = 0.5
    actual = _run(
        _wrapper(),
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=sink,
        lse_scale=lse_scale,
    )
    expected = _reference(
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=sink,
        lse_scale=lse_scale,
    )
    _check("bf16", actual, expected)
    if mode in ("both_empty", "all_minus_one"):
        assert torch.equal(actual[0], torch.zeros_like(actual[0]))
        if sink is None:
            assert torch.isneginf(actual[1]).all()
        else:
            expected_lse = (sink * math.log2(math.e) * lse_scale).expand(
                num_tokens, num_heads
            )
            torch.testing.assert_close(actual[1], expected_lse, **_LSE_TOL["bf16"])


@pytest.mark.parametrize("layout", ["NHD", "3D", "padded"])
def test_gpu_cache_layouts_match_hnd(layout: str) -> None:
    _require_family()
    torch.manual_seed(20261105)
    num_tokens, num_heads, topk, extra_topk = 5, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(16, 32)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 16 * 32)
    wrapper = _wrapper()
    base = _run(
        wrapper, q, cache, indices, extra_cache=extra, extra_indices=extra_indices
    )
    other = _run(
        wrapper,
        q,
        _layout_view(cache, layout),
        indices,
        extra_cache=_layout_view(extra, layout),
        extra_indices=extra_indices,
    )
    assert torch.equal(base[0], other[0]) and torch.equal(base[1], other[1])
    _check(
        "bf16",
        base,
        _reference(q, cache, indices, extra_cache=extra, extra_indices=extra_indices),
    )


def test_gpu_odd_token_count_and_lse_scale() -> None:
    _require_family()
    torch.manual_seed(20261106)
    num_tokens, num_heads, topk, extra_topk = 7, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(8, 64)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64)
    wrapper = _wrapper()
    base = _run(
        wrapper, q, cache, indices, extra_cache=extra, extra_indices=extra_indices
    )
    _check(
        "bf16",
        base,
        _reference(q, cache, indices, extra_cache=extra, extra_indices=extra_indices),
    )
    for scale in (0.5, 2.0, math.log(2)):
        scaled = _run(
            wrapper,
            q,
            cache,
            indices,
            extra_cache=extra,
            extra_indices=extra_indices,
            lse_scale=scale,
        )
        assert torch.equal(scaled[0], base[0])
        torch.testing.assert_close(scaled[1], base[1] * scale, atol=1e-6, rtol=1e-6)


def test_gpu_bitwise_repeatable_and_graph_replay() -> None:
    _require_family()
    torch.manual_seed(20261107)
    num_tokens, num_heads, topk, extra_topk = 8, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(8, 64)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64)
    lengths = _lengths(num_tokens, topk)
    sink = torch.randn(num_heads, device="cuda")
    wrapper = _wrapper()
    first = _run(
        wrapper,
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        attn_sink=sink,
    )
    second = _run(
        wrapper,
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        attn_sink=sink,
    )
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    output = torch.empty_like(q)
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device="cuda")

    def call() -> None:
        wrapper.run(
            q,
            cache,
            indices,
            output,
            _SM_SCALE,
            topk_length=lengths,
            attn_sink=sink,
            extra_kv_cache=extra,
            extra_indices=extra_indices,
            out_lse=out_lse,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        call()  # warm-up sizes the wrapper arenas before capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    for _ in range(3):
        output.zero_()
        out_lse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, first[0]) and torch.equal(out_lse, first[1])


def test_gpu_enable_pdl_launch_is_bitwise_neutral() -> None:
    """The split merge launched with / without the Programmatic Dependent Launch attribute produces identical bits."""

    _require_family()
    torch.manual_seed(20261108)
    num_tokens, num_heads, topk, extra_topk = 8, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(8, 64)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64)
    sink = torch.randn(num_heads, device="cuda")
    wrapper = _wrapper()
    results = []
    for enable_pdl in (True, False, None):
        output = torch.empty_like(q)
        lse = route.wrapper_run(
            wrapper,
            q,
            cache,
            indices,
            output,
            _SM_SCALE,
            attn_sink=sink,
            extra_kv_cache=extra,
            extra_indices=extra_indices,
            return_lse=True,
            enable_pdl=enable_pdl,
        )
        assert lse is not None
        results.append((output, lse.clone()))
    plan = cake_sparse_mla_sm120_dsv41_mixed_plan_splits(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    assert plan[0] > 1  # the row is split, so the merge launch is exercised
    for output, lse in results[1:]:
        assert torch.equal(output, results[0][0]) and torch.equal(lse, results[0][1])


def test_gpu_workspace_reuse_across_shapes() -> None:
    """Wrapper-owned scratch grows once and is reused; results match fresh wrappers bitwise."""

    _require_family()
    torch.manual_seed(20261108)
    cache = _main_pool(16, 64)
    extra = _extra_pool(16, 64)
    shapes = [
        (32, 64, 128, 512),
        (1, 8, 128, 0),
        (32, 64, 128, 512),
        (5, 128, 512, 512),
    ]
    inputs = []
    for num_tokens, num_heads, topk, extra_topk in shapes:
        q = _query(num_tokens, num_heads)
        indices = _indices(num_tokens, topk, 16 * 64)
        extra_indices = (
            _indices(num_tokens, extra_topk, 16 * 64) if extra_topk else None
        )
        inputs.append((q, indices, extra_indices))
    shared = _wrapper()
    for q, indices, extra_indices in inputs:
        kwargs = dict(
            extra_cache=extra if extra_indices is not None else None,
            extra_indices=extra_indices,
        )
        reused = _run(shared, q, cache, indices, **kwargs)
        fresh = _run(_wrapper(), q, cache, indices, **kwargs)
        assert torch.equal(reused[0], fresh[0]) and torch.equal(reused[1], fresh[1])
        _check("bf16", reused, _reference(q, cache, indices, **kwargs))
    # Caller-owned scratch and LSE buffers are honoured.
    q, indices, extra_indices = inputs[0]
    chunks = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(
        indices.shape[1], extra_indices.shape[1]
    )
    mid_out = torch.empty(
        q.shape[0], q.shape[1], chunks, _D, dtype=torch.bfloat16, device="cuda"
    )
    mid_lse = torch.empty(
        q.shape[0], q.shape[1], chunks, dtype=torch.float32, device="cuda"
    )
    out_lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device="cuda")
    output = torch.empty_like(q)
    returned = shared.run(
        q,
        cache,
        indices,
        output,
        _SM_SCALE,
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        out_lse=out_lse,
        mid_out=mid_out,
        mid_lse=mid_lse,
        return_lse=True,
    )
    assert returned is not None and returned.data_ptr() == out_lse.data_ptr()
    base = _run(
        shared, q, cache, indices, extra_cache=extra, extra_indices=extra_indices
    )
    assert torch.equal(output, base[0]) and torch.equal(out_lse, base[1])


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_gpu_public_api_matches_wrapper_and_sparse_backend(kv_layout: str) -> None:
    _require_family()
    torch.manual_seed(20261109)
    num_tokens, num_heads, topk, extra_topk = 8, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(8, 64)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64)
    lengths = _lengths(num_tokens, topk)
    extra_lengths = _lengths(num_tokens, extra_topk)
    sink = torch.randn(num_heads, device="cuda")
    expected = _reference(
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=sink,
    )
    wrapped = _run(
        _wrapper(),
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=sink,
    )
    view = (lambda c: c) if kv_layout == "HND" else (lambda c: c.permute(0, 2, 1, 3))
    workspace = torch.empty(
        cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes(
            num_tokens, num_heads, topk, extra_topk
        ),
        dtype=torch.int8,
        device="cuda",
    )
    outputs = {}
    for backend in ("cake", "sparse"):
        outputs[backend] = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q.unsqueeze(1),
            swa_kv_cache=view(cache),
            workspace_buffer=workspace,
            sparse_indices=indices,
            compressed_kv_cache=view(extra),
            swa_topk_lens=lengths,
            extra_sparse_indices=extra_indices,
            extra_sparse_topk_lens=extra_lengths,
            bmm1_scale=_SM_SCALE,
            sinks=sink,
            kv_layout=kv_layout,
            kv_cache_format="fp8_dsv41_fp4_ca",
            backend=backend,
        ).squeeze(1)
    assert torch.equal(outputs["cake"], wrapped[0])
    torch.testing.assert_close(outputs["cake"], expected[0], **_TOL["bf16"])
    # The hand-written route is the upper bound the Cake route must not exceed.
    torch.testing.assert_close(outputs["sparse"], expected[0], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(outputs["cake"], outputs["sparse"], atol=5e-2, rtol=5e-2)


def test_gpu_fp8_route_matches_reference() -> None:
    _require_family()
    _require_precision("fp8")
    torch.manual_seed(20261110)
    num_tokens, num_heads, topk, extra_topk = 8, 64, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(8, 64)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64)
    sink = torch.randn(num_heads, device="cuda")
    actual = _run(
        _wrapper("fp8"),
        q,
        cache,
        indices,
        extra_cache=extra,
        extra_indices=extra_indices,
        attn_sink=sink,
    )
    expected = _reference(
        q,
        cache,
        indices,
        extra_cache=extra,
        extra_indices=extra_indices,
        attn_sink=sink,
    )
    _check("fp8", actual, expected)


def test_gpu_poisoned_slot_zero_is_masked() -> None:
    """Masked candidates never read slot 0: poison its data and scales in both caches."""

    _require_family()
    torch.manual_seed(20261111)
    num_tokens, num_heads, topk, extra_topk, page = 8, 64, 128, 512, 64
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, page)
    extra = _extra_pool(8, page)
    main_flat = cache.view(8, -1)
    main_flat[0, :512].fill_(0xFF)
    main_flat[0, page * 512 : page * 512 + 16].fill_(0xFF)
    extra_flat = extra.view(8, -1)
    extra_flat[0, :256].fill_(0xFF)
    extra_flat[0, page * 256 : page * 256 + 32].fill_(0xFF)
    indices = torch.randint(
        1, 8 * page, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    extra_indices = torch.randint(
        1, 8 * page, (num_tokens, extra_topk), dtype=torch.int32, device="cuda"
    )
    indices[:, topk // 2 :] = -1
    extra_indices[:, extra_topk // 2 :] = -1
    lengths = torch.full((num_tokens,), topk - 8, dtype=torch.int32, device="cuda")
    actual = _run(
        _wrapper(),
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
    )
    assert torch.isfinite(actual[0].float()).all() and torch.isfinite(actual[1]).all()
    expected = _reference(
        q,
        cache,
        indices,
        main_lengths=lengths,
        extra_cache=extra,
        extra_indices=extra_indices,
    )
    _check("bf16", actual, expected)


@pytest.mark.parametrize("num_heads", [48, 80, 96, 112])
def test_gpu_head_counts_outside_mandatory_set(num_heads: int) -> None:
    """Status table: exported head counts run against the reference, others fail loudly."""

    _require_family()
    torch.manual_seed(20261112 + num_heads)
    num_tokens, topk, extra_topk = 4, 128, 512
    q = _query(num_tokens, num_heads)
    cache = _main_pool(8, 64)
    extra = _extra_pool(8, 64)
    indices = _indices(num_tokens, topk, 8 * 64)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64)
    if num_heads not in cake_sparse_mla_sm120_dsv41_mixed_supported_heads():
        with pytest.raises(ValueError, match="query heads"):
            _run(
                _wrapper(),
                q,
                cache,
                indices,
                extra_cache=extra,
                extra_indices=extra_indices,
            )
        return
    actual = _run(
        _wrapper(), q, cache, indices, extra_cache=extra, extra_indices=extra_indices
    )
    _check(
        "bf16",
        actual,
        _reference(q, cache, indices, extra_cache=extra, extra_indices=extra_indices),
    )


def test_gpu_rejects_bad_inputs() -> None:
    _require_family()
    torch.manual_seed(20261113)
    q = _query(2, 16)
    cache = _main_pool(4, 64)
    extra = _extra_pool(4, 64)
    indices = _indices(2, 512, 4 * 64)
    output = torch.empty_like(q)
    out_lse = torch.empty(2, 16, dtype=torch.float32, device="cuda")
    decode = functools.partial(
        cake_sparse_mla_sm120_dsv41_mixed_decode, sm_scale=_SM_SCALE
    )
    with pytest.raises(ValueError, match="query heads"):
        decode(
            _query(2, 24),
            cache,
            indices,
            _query(2, 24),
            torch.empty(2, 24, device="cuda"),
        )
    with pytest.raises(ValueError, match="int32"):
        decode(q, cache, indices.to(torch.int64), output, out_lse)
    with pytest.raises(ValueError, match="528"):
        decode(
            q,
            torch.zeros(4, 1, 64, 384, dtype=torch.uint8, device="cuda"),
            indices,
            output,
            out_lse,
        )
    with pytest.raises(ValueError, match="288"):
        decode(
            q,
            cache,
            indices,
            output,
            out_lse,
            extra_kv_cache=cache,
            extra_indices=indices,
        )
    with pytest.raises(ValueError, match="multiple of 16"):
        decode(q, _padded_view(cache, 8), indices, output, out_lse)
    with pytest.raises(ValueError, match="provided together"):
        decode(q, cache, indices, output, out_lse, extra_kv_cache=extra)
    with pytest.raises(ValueError, match="mid_out"):
        decode(q, cache, indices, output, out_lse, num_splits=2)
    with pytest.raises(ValueError, match="at most"):
        decode(
            q,
            cache,
            torch.zeros(2, 2048, dtype=torch.int32, device="cuda"),
            output,
            out_lse,
            num_splits=1,
        )
    with pytest.raises(ValueError, match="compute_precision"):
        decode(q, cache, indices, output, out_lse, compute_precision="nvfp4")
