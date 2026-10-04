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

"""Cake SM120 DeepSeek-V4 NVFP4 sparse-MLA prefill (``backend="cake"`` on SM120/SM121).

The planner / crossover tests run on the CPU (they read the generated manifest
only).  Every attention case compares against the fp32 reference over the
dequantized NVFP4 operands at the SM120 NVFP4 tolerances (output
atol=rtol=5e-2, LSE atol=rtol=2e-2); every valid head-tile count is
exercised.
"""

import importlib
import itertools
import math
import os

import pytest
import torch

import flashinfer
from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from flashinfer.mla._sparse_mla_sm120._cake_dsv4_nvfp4 import (
    _cake_nvfp4_sparse_mla_decode,
    _cake_nvfp4_sparse_mla_prefill,
    cake_sparse_mla_sm120_dsv4_nvfp4_format_info,
    cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks,
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill,
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_grid_head_blocks_first,
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_q_evict_first,
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_stages,
    cake_sparse_mla_sm120_dsv4_nvfp4_prefill,
    cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles,
    cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel,
    cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads,
)
from tests.attention.test_cake_sparse_mla_sm120_dsv4_nvfp4 import (
    _BYTES,
    _D,
    _LSE_TOL,
    _OUT_TOL,
    _device_sms,
    _latent,
    _layout_view,
    _padded_view,
    _query,
    _reference,
    _require_sm120,
)

# Dotted name of the generating prefill kernel module (planner parity test); unset -> skipped.
_KERNEL_MODULE_ENV = "CAKE_SPARSE_MLA_SM120_DSV4_PREFILL_KERNEL_MODULE"
_PREFILL_HEADS = (16, 32, 48, 64, 80, 96, 112, 128)


def _valid_tiles(num_heads: int) -> tuple:
    return tuple(ht for ht in (1, 2, 4) if num_heads % (16 * ht) == 0)


# ----------------------------------------------------------------------------- CPU: facts and planners


def test_cake_prefill_format_info() -> None:
    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    assert info["prefill_heads"] == _PREFILL_HEADS
    assert set(info["prefill_heads"]) <= set(info["heads"])
    assert 8 not in info["prefill_heads"]
    assert info["prefill_max_chunks"] == 16
    assert info["prefill_kernel_commit"]
    for num_heads in info["heads"]:
        tiles = cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(num_heads)
        if num_heads in info["prefill_heads"]:
            assert tiles == _valid_tiles(num_heads), num_heads
            assert info["prefill_head_tiles"][num_heads] == tiles
        else:
            assert tiles == ()
    assert cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(128) == (1, 2, 4)
    assert cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(96) == (1, 2)
    assert cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(80) == (1,)
    assert cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(24) == ()
    # ring depths exported per head-tile count: the one-tile instance has a three-stage form
    assert info["prefill_ring_stages"] == {1: (2, 3), 2: (2,), 4: (2,)}


def test_cake_plan_prefill_rules() -> None:
    plan = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill
    # Wide head counts divisible by 32: two tiles up to 96 tokens at every candidate count on both SKUs.
    assert plan(num_tokens=64, num_heads=128, topk=128, num_sms=188) == 2
    assert plan(num_tokens=96, num_heads=64, topk=128, num_sms=170) == 2
    assert plan(num_tokens=97, num_heads=128, topk=128, num_sms=188) == 4
    assert plan(num_tokens=97, num_heads=64, topk=256, num_sms=170) == 4
    assert plan(num_tokens=96, num_heads=128, topk=512, num_sms=188) == 2
    # PRO 6000 (188 SMs) 128 heads with 512..639 candidates: two tiles through 128 tokens; the RTX 5090
    # (170 SMs), 64 heads and candidate counts outside the band keep four tiles above 96 tokens.
    assert plan(num_tokens=97, num_heads=128, topk=512, num_sms=188) == 2
    assert plan(num_tokens=128, num_heads=128, topk=512, num_sms=188) == 2
    assert (
        plan(num_tokens=100, num_heads=128, topk=256, extra_topk=256, num_sms=188) == 2
    )
    assert plan(num_tokens=129, num_heads=128, topk=512, num_sms=188) == 4
    assert (
        plan(num_tokens=128, num_heads=128, topk=512, extra_topk=128, num_sms=188) == 4
    )
    assert plan(num_tokens=128, num_heads=64, topk=512, num_sms=188) == 4
    assert plan(num_tokens=97, num_heads=128, topk=512, num_sms=170) == 4
    assert plan(num_tokens=128, num_heads=128, topk=512, num_sms=170) == 4
    assert (
        plan(num_tokens=128, num_heads=64, topk=256, extra_topk=256, num_sms=188) == 4
    )
    assert plan(num_tokens=96, num_heads=64, topk=256, extra_topk=256, num_sms=170) == 2
    assert plan(num_tokens=512, num_heads=128, topk=512, num_sms=188) == 4
    assert plan(num_tokens=512, num_heads=128, topk=512, num_sms=170) == 4
    # 96 heads: the two-tile instance is the largest one.
    assert plan(num_tokens=128, num_heads=96, topk=512, num_sms=188) == 2
    assert plan(num_tokens=2048, num_heads=96, topk=512, num_sms=170) == 2
    # Wide head counts without a two-tile instance (80, 112) stay at one tile.
    assert plan(num_tokens=64, num_heads=80, topk=512, num_sms=188) == 1
    assert plan(num_tokens=2048, num_heads=112, topk=512, num_sms=188) == 1
    # Narrow head counts: the largest instance whatever the token count (two tiles for 32 heads).
    assert plan(num_tokens=2048, num_heads=32, topk=512, num_sms=188) == 2
    assert plan(num_tokens=1, num_heads=32, topk=128, num_sms=170) == 2
    assert plan(num_tokens=8192, num_heads=16, topk=128, num_sms=170) == 1
    assert plan(num_tokens=200, num_heads=48, topk=512, num_sms=188) == 1
    with pytest.raises(ValueError, match="query heads"):
        plan(num_tokens=128, num_heads=8, topk=512, num_sms=188)
    with pytest.raises(ValueError, match="query heads"):
        plan(num_tokens=128, num_heads=24, topk=512, num_sms=188)
    with pytest.raises(ValueError, match="num_sms"):
        plan(num_tokens=128, num_heads=128, topk=512, num_sms=0)


def test_cake_plan_prefill_stages_rules() -> None:
    stages = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_stages
    # three cp.async stages only for the one-tile instance from 8 chunks (512 candidates) per token
    assert stages(head_tiles=1, topk=512, num_tokens=128) == 3
    assert stages(head_tiles=1, topk=448, num_tokens=8192) == 2
    assert stages(head_tiles=1, topk=128, num_tokens=8192) == 2
    assert stages(head_tiles=2, topk=512, num_tokens=8192) == 2
    assert stages(head_tiles=4, topk=1024, num_tokens=8192) == 2
    # dual-cache lists below 16 chunks keep two stages below 512 tokens
    assert (
        stages(head_tiles=1, topk=512, extra_topk=128, dual=True, num_tokens=128) == 2
    )
    assert (
        stages(head_tiles=1, topk=512, extra_topk=128, dual=True, num_tokens=511) == 2
    )
    assert (
        stages(head_tiles=1, topk=512, extra_topk=128, dual=True, num_tokens=512) == 3
    )
    assert (
        stages(head_tiles=1, topk=512, extra_topk=512, dual=True, num_tokens=128) == 3
    )
    assert (
        stages(head_tiles=1, topk=512, extra_topk=128, dual=False, num_tokens=128) == 3
    )


def test_cake_plan_prefill_grid_order_rules() -> None:
    grid = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_grid_head_blocks_first
    # head blocks first (the head blocks of one token are adjacent CTAs) up to 4 chunks of 64 candidates per block
    assert grid(head_tiles=4, topk=128, num_tokens=128) is True
    assert grid(head_tiles=4, topk=256, num_tokens=8192) is True
    assert grid(head_tiles=4, topk=128, extra_topk=128, num_tokens=128) is True
    assert grid(head_tiles=1, topk=256, num_tokens=1) is True
    assert grid(head_tiles=4, topk=512, num_tokens=128) is False
    assert grid(head_tiles=2, topk=512, num_tokens=8192) is False
    assert grid(head_tiles=4, topk=128, extra_topk=512, num_tokens=8192) is False
    assert grid(head_tiles=4, topk=320, num_tokens=128) is False  # 5 chunks
    # grid.y cap: head blocks first puts the token count in grid.y (CUDA limit 65535) -> tokens first beyond it
    assert grid(head_tiles=4, topk=128, num_tokens=65535) is True
    assert grid(head_tiles=4, topk=128, num_tokens=65536) is False
    assert grid(head_tiles=1, topk=64, num_tokens=100000) is False


def test_cake_plan_prefill_q_evict_first_rules() -> None:
    qef = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_q_evict_first
    assert qef(num_tokens=1) is False
    assert qef(num_tokens=128) is False
    assert qef(num_tokens=511) is False
    assert qef(num_tokens=512) is True
    assert qef(num_tokens=8192) is True


def test_cake_select_kernel_rules() -> None:
    select = cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel
    # No prefill instance (8 heads) or more than 16 chunks: always decode.
    assert select(num_tokens=4096, num_heads=8, topk=128, num_sms=188) == "decode"
    assert (
        select(num_tokens=2048, num_heads=128, topk=1024, extra_topk=64, num_sms=188)
        == "decode"
    )
    # Partial 64-wide chunks are served by the prefill's masked tail.
    assert select(num_tokens=2048, num_heads=128, topk=130, num_sms=188) == "prefill"
    assert (
        select(num_tokens=2048, num_heads=128, topk=512, extra_topk=100, num_sms=188)
        == "prefill"
    )
    # Wide head counts: decode up to 8 tokens at >= 512 candidates, up to 16 below (both SKUs).
    for sms in (188, 170):
        assert select(num_tokens=8, num_heads=128, topk=512, num_sms=sms) == "decode"
        assert select(num_tokens=9, num_heads=128, topk=512, num_sms=sms) == "prefill"
        assert select(num_tokens=16, num_heads=64, topk=128, num_sms=sms) == "decode"
        assert select(num_tokens=17, num_heads=64, topk=128, num_sms=sms) == "prefill"
        assert select(num_tokens=16, num_heads=64, topk=256, num_sms=sms) == "decode"
        assert (
            select(num_tokens=16, num_heads=96, topk=256, extra_topk=256, num_sms=sms)
            == "prefill"
        )
        assert select(num_tokens=1, num_heads=128, topk=512, num_sms=sms) == "decode"
        assert select(num_tokens=2048, num_heads=80, topk=512, num_sms=sms) == "prefill"
    # Narrow head counts at >= 512 candidates: decode up to 8 tokens with >= 180 SMs, up to 32 below.
    assert select(num_tokens=8, num_heads=16, topk=512, num_sms=188) == "decode"
    assert select(num_tokens=9, num_heads=16, topk=512, num_sms=188) == "prefill"
    assert select(num_tokens=32, num_heads=16, topk=512, num_sms=170) == "decode"
    assert select(num_tokens=33, num_heads=16, topk=512, num_sms=170) == "prefill"
    assert (
        select(num_tokens=8, num_heads=48, topk=256, extra_topk=256, num_sms=188)
        == "decode"
    )
    assert (
        select(num_tokens=32, num_heads=48, topk=256, extra_topk=256, num_sms=170)
        == "decode"
    )
    # Narrow head counts below 512 candidates: prefill at every token count, one token included.
    assert select(num_tokens=1, num_heads=16, topk=128, num_sms=188) == "prefill"
    assert select(num_tokens=1, num_heads=16, topk=128, num_sms=170) == "prefill"
    assert select(num_tokens=2, num_heads=32, topk=128, num_sms=188) == "prefill"
    assert select(num_tokens=64, num_heads=48, topk=256, num_sms=170) == "prefill"
    with pytest.raises(ValueError, match="num_sms"):
        select(num_tokens=128, num_heads=128, topk=512, num_sms=0)
    # Every prefill-selected shape has a plan.
    for num_tokens, num_heads, topk, sms in itertools.product(
        (1, 2, 33, 128, 2048), _PREFILL_HEADS, (128, 512), (188, 170)
    ):
        if (
            select(num_tokens=num_tokens, num_heads=num_heads, topk=topk, num_sms=sms)
            == "prefill"
        ):
            head_tiles = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
                num_tokens=num_tokens, num_heads=num_heads, topk=topk, num_sms=sms
            )
            assert isinstance(head_tiles, int)
            assert head_tiles in _valid_tiles(num_heads)


def test_cake_prefill_planner_parity_with_kernel_module() -> None:
    """The Python prefill facts mirror the generating kernel module exactly.

    Set ``CAKE_SPARSE_MLA_SM120_DSV4_PREFILL_KERNEL_MODULE`` to the module's dotted name to run it.
    """

    name = os.environ.get(_KERNEL_MODULE_ENV)
    if not name:
        pytest.skip(f"{_KERNEL_MODULE_ENV} not set")
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"kernel module {name} unavailable: {exc}")
    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    assert info["heads_per_block"] == mod.HPB
    assert info["prefill_max_chunks"] == mod.MAX_CPB
    assert tuple(mod.SUPPORTED_HEAD_COUNTS) == info["prefill_heads"]
    for num_heads in info["prefill_heads"]:
        tiles = cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(num_heads)
        # The module's default is the largest instance; every exported tile count traces.
        assert mod.plan_head_tiles(num_heads) == max(tiles)
        assert tiles == tuple(ht for ht in (1, 2, 4) if num_heads % (mod.HPB * ht) == 0)
        for num_tokens in (128, 2048):
            head_tiles = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
                num_tokens=num_tokens, num_heads=num_heads, topk=512, num_sms=188
            )
            assert head_tiles in tiles
    for topk, extra_topk in ((64, 0), (128, 128), (512, 512), (130, 1), (1024, 0)):
        assert cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(
            topk, extra_topk
        ) == mod.max_chunks(topk, extra_topk)
    # Ring depth and query L2 policy mirror the module's planner over the measured grid.
    assert mod.PLAN_NUM_STAGES_MIN_CHUNKS == 8
    assert mod.PLAN_NUM_STAGES_DUAL_MIN_CHUNKS == 16
    assert mod.PLAN_NUM_STAGES_DUAL_MIN_TOKENS == 512
    assert mod.PLAN_Q_EVICT_MIN_TOKENS == 512
    assert mod.PLAN_GRID_HB_FIRST_MAX_CHUNKS == 4
    assert mod.PLAN_GRID_HB_FIRST_MAX_TOKENS == 65535
    for num_tokens in (1, 128, 511, 512, 2048, 8192, 65535, 65536):
        assert cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_q_evict_first(
            num_tokens=num_tokens
        ) == mod.plan_q_evict_first(num_tokens)
        for head_tiles in (1, 2, 4):
            for topk, extra_topk, dual in (
                (128, 0, False),
                (448, 0, False),
                (512, 0, False),
                (512, 128, True),
                (512, 512, True),
                (1024, 0, False),
            ):
                assert cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_stages(
                    head_tiles=head_tiles,
                    topk=topk,
                    extra_topk=extra_topk,
                    dual=dual,
                    num_tokens=num_tokens,
                ) == mod.plan_num_stages(
                    head_tiles,
                    mod.max_chunks(topk, extra_topk),
                    dual=dual,
                    num_tokens=num_tokens,
                ), (num_tokens, head_tiles, topk, extra_topk, dual)
                assert (
                    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_grid_head_blocks_first(
                        head_tiles=head_tiles,
                        topk=topk,
                        extra_topk=extra_topk,
                        num_tokens=num_tokens,
                    )
                    == mod.plan_grid_head_blocks_first(
                        head_tiles, mod.max_chunks(topk, extra_topk), num_tokens
                    )
                ), (num_tokens, head_tiles, topk, extra_topk)
    # The decode / prefill crossover and the tile choice mirror the module's planner on both
    # sm_120a SKUs (188 / 170 SMs) over the measured token, head and candidate grid.
    for num_sms in (188, 170):
        for num_tokens in (
            1,
            4,
            8,
            9,
            16,
            17,
            24,
            32,
            33,
            48,
            64,
            96,
            97,
            128,
            129,
            512,
            2048,
        ):
            for num_heads in info["prefill_heads"]:
                for topk, extra_topk in (
                    (128, 0),
                    (256, 0),
                    (512, 0),
                    (256, 256),
                    (512, 512),
                ):
                    plan = mod.plan_prefill(
                        num_tokens, num_heads, topk, extra_topk, num_sms=num_sms
                    )
                    form = cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
                        num_tokens=num_tokens,
                        num_heads=num_heads,
                        topk=topk,
                        extra_topk=extra_topk,
                        num_sms=num_sms,
                    )
                    assert form == plan["form"], (
                        num_sms,
                        num_tokens,
                        num_heads,
                        topk,
                        extra_topk,
                    )
                    if form == "prefill":
                        assert (
                            cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
                                num_tokens=num_tokens,
                                num_heads=num_heads,
                                topk=topk,
                                extra_topk=extra_topk,
                                num_sms=num_sms,
                            )
                            == plan["head_tiles"]
                        ), (num_sms, num_tokens, num_heads, topk, extra_topk)


# ----------------------------------------------------------------------------- GPU: attention cases


def _prefill_forms(num_heads: int):
    yield from _valid_tiles(num_heads)


@pytest.mark.parametrize("topk", [128, 512])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("num_heads", [16, 64, 128])
@pytest.mark.parametrize("with_sink", [False, True])
def test_cake_prefill_matches_dequantized_reference(
    topk: int, page_size: int, num_heads: int, with_sink: bool
) -> None:
    """Every head-tile count and both CTA forms over runtime page sizes with lengths, -1 masks, sink, lse_scale."""

    _require_sm120()
    torch.manual_seed(20261101 + topk + page_size + num_heads + int(with_sink))
    num_tokens = 3
    num_pages = 16 * 64 // page_size
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    topk_len = (topk * 7) // 10
    indices[:, topk_len - 7 : topk_len] = -1
    indices[0, : topk // 8] = -1
    lengths = torch.tensor(
        [topk_len, topk, topk_len - 1], dtype=torch.int32, device="cuda"
    )
    attn_sink = (
        torch.linspace(-1.0, 1.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    sm_scale = _D**-0.5
    lse_scale = 0.5 if with_sink else 1.0
    reference, reference_lse = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        attn_sink=attn_sink,
        lse_scale=lse_scale,
    )
    for head_tiles in _prefill_forms(num_heads):
        output, lse = _cake_nvfp4_sparse_mla_prefill(
            q,
            cache,
            indices,
            sm_scale,
            topk_length=lengths,
            attn_sink=attn_sink,
            lse_scale=lse_scale,
            head_tiles=head_tiles,
        )
        torch.testing.assert_close(output, reference, **_OUT_TOL)
        torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)


@pytest.mark.parametrize("num_heads", [16, 128])
@pytest.mark.parametrize("main_page_size", [32, 64])
@pytest.mark.parametrize("extra_page_size,extra_topk", [(2, 128), (32, 512), (64, 512)])
def test_cake_prefill_dual_cache_matches_reference(
    num_heads: int, main_page_size: int, extra_page_size: int, extra_topk: int
) -> None:
    """Main and compressed cache sections share one online softmax (65 tokens: an odd multi-wave grid)."""

    _require_sm120()
    torch.manual_seed(20261102 + num_heads + main_page_size + extra_page_size)
    num_tokens, main_topk = 65, 128 if extra_topk == 128 else 512
    main_pages = 8 * 64 // main_page_size
    extra_pages = max(16, (2 * extra_topk + extra_page_size - 1) // extra_page_size)
    q = _query(num_tokens, num_heads)
    main_cache = nvfp4_quantize_pack_sparse_mla_cache(
        _latent(main_pages, main_page_size)
    )
    extra_cache = nvfp4_quantize_pack_sparse_mla_cache(
        _latent(extra_pages, extra_page_size)
    )
    main_indices = torch.randint(
        0,
        main_pages * main_page_size,
        (num_tokens, main_topk),
        dtype=torch.int32,
        device="cuda",
    )
    extra_indices = torch.randint(
        0,
        extra_pages * extra_page_size,
        (num_tokens, extra_topk),
        dtype=torch.int32,
        device="cuda",
    )
    main_lengths = torch.randint(
        1, main_topk + 1, (num_tokens,), dtype=torch.int32, device="cuda"
    )
    extra_lengths = torch.randint(
        1, extra_topk + 1, (num_tokens,), dtype=torch.int32, device="cuda"
    )
    main_indices[:, 91:96] = -1
    extra_indices[:, 37:43] = -1
    attn_sink = torch.linspace(-1.0, 1.0, num_heads, dtype=torch.float32, device="cuda")
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(
        q,
        main_cache,
        main_indices,
        sm_scale,
        main_lengths=main_lengths,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=attn_sink,
    )
    for head_tiles in _prefill_forms(num_heads):
        output, lse = _cake_nvfp4_sparse_mla_prefill(
            q,
            main_cache,
            main_indices,
            sm_scale,
            topk_length=main_lengths,
            attn_sink=attn_sink,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_lengths,
            head_tiles=head_tiles,
        )
        torch.testing.assert_close(output, reference, **_OUT_TOL)
        torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)


@pytest.mark.parametrize("num_tokens", [3, 65, 257])
def test_cake_prefill_token_counts(num_tokens: int) -> None:
    """Planner-default head tiles over a short, an odd multi-wave and a long token count."""

    _require_sm120()
    torch.manual_seed(20261103 + num_tokens)
    num_heads, topk, page_size, num_pages = 64, 512, 64, 32
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.randint(
        1, topk + 1, (num_tokens,), dtype=torch.int32, device="cuda"
    )
    reference, reference_lse = _reference(
        q, cache, indices, _D**-0.5, main_lengths=lengths
    )
    output = torch.empty_like(q)
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    plan = cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
        q,
        cache,
        indices,
        output,
        out_lse,
        _D**-0.5,
        topk_length=lengths,
    )
    expected_tiles = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        num_sms=_device_sms(),
    )
    assert plan["head_tiles"] == expected_tiles
    items = num_tokens * (num_heads // (16 * plan["head_tiles"]))
    assert plan["num_ctas"] == items
    torch.testing.assert_close(output, reference, **_OUT_TOL)
    torch.testing.assert_close(out_lse, reference_lse, **_LSE_TOL)


def test_cake_prefill_grid_y_cap_matches_split_launch() -> None:
    """Above 65535 tokens the planner and the binding keep the token-major grid.

    CUDA caps grid.y at 65535, so the head-blocks-first order of short candidate lists cannot carry the
    token count there.  The 65536-token launch (tokens first by the cap) must be bitwise equal to the two
    32768-token halves, which launch head blocks first.
    """

    _require_sm120()
    torch.manual_seed(20261103)
    num_tokens, num_heads, topk, page_size, num_pages = 65536, 16, 64, 64, 64
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    output = torch.empty_like(q)
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    plan = cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
        q, cache, indices, output, out_lse, _D**-0.5
    )
    assert plan["grid_head_blocks_first"] is False
    half = num_tokens // 2
    parts, part_lse = [], []
    for lo in (0, half):
        part = torch.empty(half, num_heads, _D, dtype=q.dtype, device="cuda")
        lse = torch.empty(half, num_heads, dtype=torch.float32, device="cuda")
        part_plan = cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
            q[lo : lo + half].contiguous(),
            cache,
            indices[lo : lo + half].contiguous(),
            part,
            lse,
            _D**-0.5,
        )
        assert part_plan["grid_head_blocks_first"] is True
        parts.append(part)
        part_lse.append(lse)
    torch.cuda.synchronize()
    assert torch.equal(output, torch.cat(parts))
    assert torch.equal(out_lse, torch.cat(part_lse))


@pytest.mark.parametrize("layout", ["NHD", "3D", "padded"])
def test_cake_prefill_cache_layouts_match_hnd(layout: str) -> None:
    """3-D, NHD and padded-page-stride views of one page-48 pool give the HND result bit for bit."""

    _require_sm120()
    torch.manual_seed(20261104)
    num_tokens, num_heads, topk, page_size, num_pages = 5, 32, 512, 48, 16
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(q, cache, indices, sm_scale)
    expected, expected_lse = _cake_nvfp4_sparse_mla_prefill(q, cache, indices, sm_scale)
    torch.testing.assert_close(expected, reference, **_OUT_TOL)
    torch.testing.assert_close(expected_lse, reference_lse, **_LSE_TOL)
    view = _layout_view(cache, layout)
    assert view.stride(0) > page_size * _BYTES or layout != "padded"
    output, lse = _cake_nvfp4_sparse_mla_prefill(q, view, indices, sm_scale)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
    # The same pool as the extra cache of a dual call.
    extra_indices = torch.randint(
        0, num_pages * page_size, (num_tokens, 128), dtype=torch.int32, device="cuda"
    )
    dual_ref, dual_ref_lse = _reference(
        q, cache, indices, sm_scale, extra_cache=cache, extra_indices=extra_indices
    )
    dual_out, dual_lse = _cake_nvfp4_sparse_mla_prefill(
        q, cache, indices, sm_scale, extra_kv_cache=view, extra_indices=extra_indices
    )
    torch.testing.assert_close(dual_out, dual_ref, **_OUT_TOL)
    torch.testing.assert_close(dual_lse, dual_ref_lse, **_LSE_TOL)


@pytest.mark.parametrize("with_sink", [False, True])
def test_cake_prefill_empty_rows(with_sink: bool) -> None:
    """Zero lengths and all -1 rows write zeros and -inf / sink-only LSE next to populated rows."""

    _require_sm120()
    torch.manual_seed(20261105 + int(with_sink))
    num_tokens, num_heads, topk = 4, 128, 128
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(4, 64))
    indices = torch.randint(
        0, 4 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    indices[1] = -1
    lengths = torch.tensor([0, topk, 1, 65], dtype=torch.int32, device="cuda")
    attn_sink = (
        torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        attn_sink=attn_sink,
        lse_scale=2.0,
    )
    for head_tiles in _prefill_forms(num_heads):
        output, lse = _cake_nvfp4_sparse_mla_prefill(
            q,
            cache,
            indices,
            sm_scale,
            topk_length=lengths,
            attn_sink=attn_sink,
            lse_scale=2.0,
            head_tiles=head_tiles,
        )
        assert torch.count_nonzero(output[:2]) == 0
        if attn_sink is None:
            assert torch.isneginf(lse[:2]).all()
        else:
            torch.testing.assert_close(
                lse[:2],
                (attn_sink * math.log2(math.e) * 2.0).unsqueeze(0).expand(2, -1),
            )
        torch.testing.assert_close(output[2:], reference[2:], **_OUT_TOL)
        torch.testing.assert_close(lse[2:], reference_lse[2:], **_LSE_TOL)


def test_cake_prefill_sink_dominated_rows() -> None:
    """Sinks above the row's log-sum-exp gate the output of the direct epilogue."""

    _require_sm120()
    torch.manual_seed(20261106)
    num_tokens, num_heads, topk = 6, 16, 256
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(8, 64))
    indices = torch.randint(
        0, 8 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor([1, 3, 2, 1, 5, 4], dtype=torch.int32, device="cuda")
    attn_sink = torch.linspace(-6.0, 6.0, num_heads, dtype=torch.float32, device="cuda")
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(
        q, cache, indices, sm_scale, main_lengths=lengths, attn_sink=attn_sink
    )
    output, lse = _cake_nvfp4_sparse_mla_prefill(
        q,
        cache,
        indices,
        sm_scale,
        topk_length=lengths,
        attn_sink=attn_sink,
    )
    torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)
    torch.testing.assert_close(output, reference, **_OUT_TOL)


def test_cake_prefill_bitwise_repeatable() -> None:
    _require_sm120()
    torch.manual_seed(20261107)
    num_tokens, num_heads, topk = 40, 64, 512
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(32, 64))
    indices = torch.randint(
        0, 32 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    first = _cake_nvfp4_sparse_mla_prefill(q, cache, indices, _D**-0.5)
    for _ in range(3):
        again = _cake_nvfp4_sparse_mla_prefill(q, cache, indices, _D**-0.5)
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


def test_cake_prefill_matches_decode() -> None:
    """Both Cake families agree within the NVFP4 tolerances on one shape (shared Q quantization and cache ABI)."""

    _require_sm120()
    torch.manual_seed(20261108)
    num_tokens, num_heads, topk = 24, 96, 512
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(16, 64))
    indices = torch.randint(
        0, 16 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    indices[3, 200:] = -1
    attn_sink = torch.linspace(-0.5, 0.5, num_heads, dtype=torch.float32, device="cuda")
    decode_out, decode_lse = _cake_nvfp4_sparse_mla_decode(
        q, cache, indices, _D**-0.5, attn_sink=attn_sink
    )
    reference, reference_lse = _reference(
        q, cache, indices, _D**-0.5, attn_sink=attn_sink
    )
    for head_tiles in _prefill_forms(num_heads):
        output, lse = _cake_nvfp4_sparse_mla_prefill(
            q,
            cache,
            indices,
            _D**-0.5,
            attn_sink=attn_sink,
            head_tiles=head_tiles,
        )
        torch.testing.assert_close(output, reference, **_OUT_TOL)
        torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)
        torch.testing.assert_close(output, decode_out, **_OUT_TOL)
        torch.testing.assert_close(lse, decode_lse, **_LSE_TOL)


def test_cake_prefill_head_count_48_and_partial_chunk() -> None:
    """A head count outside the hand-written set and a candidate count that is not a multiple of 64."""

    _require_sm120()
    torch.manual_seed(20261109)
    num_tokens, num_heads, topk = 7, 48, 130
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(8, 64))
    indices = torch.randint(
        0, 8 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    indices[2, 129] = -1
    reference, reference_lse = _reference(q, cache, indices, _D**-0.5)
    output, lse = _cake_nvfp4_sparse_mla_prefill(q, cache, indices, _D**-0.5)
    torch.testing.assert_close(output, reference, **_OUT_TOL)
    torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)


def test_cake_prefill_rejects_bad_inputs() -> None:
    _require_sm120()
    torch.manual_seed(20261110)
    q = _query(2, 32)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(4, 64))
    indices = torch.randint(0, 4 * 64, (2, 512), dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)
    out_lse = torch.empty(2, 32, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="query heads"):
        _cake_nvfp4_sparse_mla_prefill(_query(2, 24), cache, indices, 1.0)
    with pytest.raises(ValueError, match="prefill supports"):
        _cake_nvfp4_sparse_mla_prefill(_query(2, 8), cache, indices, 1.0)
    with pytest.raises(ValueError, match="int32"):
        _cake_nvfp4_sparse_mla_prefill(q, cache, indices.to(torch.int64), 1.0)
    with pytest.raises(ValueError, match="multiple of 16"):
        _cake_nvfp4_sparse_mla_prefill(q, _padded_view(cache, 8), indices, 1.0)
    with pytest.raises(ValueError, match="at most"):
        cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
            q,
            cache,
            torch.zeros(2, 1025, dtype=torch.int32, device="cuda"),
            output,
            out_lse,
            1.0,
        )
    with pytest.raises(ValueError, match="at most"):
        _cake_nvfp4_sparse_mla_prefill(
            q,
            cache,
            indices,
            1.0,
            extra_kv_cache=cache,
            extra_indices=torch.zeros(2, 576, dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(ValueError, match="provided together"):
        _cake_nvfp4_sparse_mla_prefill(q, cache, indices, 1.0, extra_kv_cache=cache)
    with pytest.raises(ValueError, match="not valid"):
        _cake_nvfp4_sparse_mla_prefill(q, cache, indices, 1.0, head_tiles=4)
    with pytest.raises(ValueError, match="not valid"):
        _cake_nvfp4_sparse_mla_prefill(_query(2, 48), cache, indices, 1.0, head_tiles=2)
    with pytest.raises(ValueError, match="out_lse"):
        cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
            q, cache, indices, output, out_lse[:1], 1.0
        )


def test_cake_prefill_cuda_graph() -> None:
    """The caller-owned prefill entry is replayable in a CUDA graph and bitwise stable across replays."""

    _require_sm120()
    torch.manual_seed(20261111)
    num_tokens, num_heads, topk, page_size, num_pages = 36, 128, 512, 32, 32
    q = _query(num_tokens, num_heads)
    replay_q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.full((num_tokens,), topk - 3, dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device="cuda")

    def run() -> None:
        cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
            q,
            cache,
            indices,
            output,
            out_lse,
            _D**-0.5,
            topk_length=lengths,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.copy_(replay_q)
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    first_lse = out_lse.clone()
    reference, reference_lse = _reference(
        replay_q, cache, indices, _D**-0.5, main_lengths=lengths
    )
    torch.testing.assert_close(first, reference, **_OUT_TOL)
    torch.testing.assert_close(first_lse, reference_lse, **_LSE_TOL)
    for _ in range(2):
        output.zero_()
        out_lse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, first) and torch.equal(out_lse, first_lse)


@pytest.mark.parametrize(
    "num_tokens,num_heads,page_size,topk,extra_page_size",
    [(257, 64, 64, 128, None), (65, 128, 32, 512, 2), (70, 16, 128, 128, 32)],
)
def test_cake_public_api_prefill_crossover(
    num_tokens: int,
    num_heads: int,
    page_size: int,
    topk: int,
    extra_page_size: int | None,
) -> None:
    """``trtllm_batch_decode_sparse_mla_dsv4(backend="cake", kv_cache_format="nvfp4")`` on prefill-selected shapes."""

    _require_sm120()
    torch.manual_seed(20261112 + num_tokens + num_heads + page_size)
    extra_topk = 128 if extra_page_size is not None else 0
    assert (
        cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=_device_sms(),
        )
        == "prefill"
    )
    num_pages = 8 * 64 // page_size
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.randint(
        topk // 2, topk + 1, (num_tokens,), dtype=torch.int32, device="cuda"
    )
    extra_cache = extra_indices = extra_lengths = None
    if extra_page_size is not None:
        extra_pages = max(16, 256 // extra_page_size)
        extra_cache = nvfp4_quantize_pack_sparse_mla_cache(
            _latent(extra_pages, extra_page_size)
        )
        extra_indices = torch.randint(
            0,
            extra_pages * extra_page_size,
            (num_tokens, extra_topk),
            dtype=torch.int32,
            device="cuda",
        )
        extra_lengths = torch.randint(
            1, extra_topk + 1, (num_tokens,), dtype=torch.int32, device="cuda"
        )
    sm_scale = _D**-0.5
    reference, _ = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
    )
    # The prefill path carves only the LSE from the workspace.
    workspace = torch.empty(
        num_tokens * num_heads * 4 + 16, dtype=torch.uint8, device="cuda"
    )
    output = torch.empty_like(q)
    returned = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=cache,
        workspace_buffer=workspace,
        sparse_indices=indices,
        compressed_kv_cache=extra_cache,
        swa_topk_lens=lengths,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lengths,
        out=output,
        bmm1_scale=sm_scale,
        backend="cake",
        kv_cache_format="nvfp4",
    )
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, reference, **_OUT_TOL)
    # 4-D query [T, 1, H, D] with the NHD cache layout gives the same bits.
    output4 = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=cache.permute(0, 2, 1, 3),
        workspace_buffer=workspace,
        sparse_indices=indices,
        compressed_kv_cache=extra_cache.permute(0, 2, 1, 3)
        if extra_cache is not None
        else None,
        swa_topk_lens=lengths,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lengths,
        bmm1_scale=sm_scale,
        kv_layout="NHD",
        backend="cake",
        kv_cache_format="nvfp4",
    )
    torch.testing.assert_close(output4.squeeze(1), output, atol=0, rtol=0)
    with pytest.raises(ValueError, match="workspace"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=cache,
            workspace_buffer=torch.empty(16, dtype=torch.uint8, device="cuda"),
            sparse_indices=indices,
            swa_topk_lens=lengths,
            bmm1_scale=sm_scale,
            backend="cake",
            kv_cache_format="nvfp4",
        )


@pytest.mark.parametrize("num_tokens", [3, 257])
def test_cake_wrapper_run_crossover(num_tokens: int) -> None:
    """``SparseMLASm120Wrapper(backend="cake").run`` routes 3 tokens to the decode and 257 tokens to the prefill."""

    _require_sm120()
    torch.manual_seed(20261113 + num_tokens)
    num_heads, topk, extra_topk = 64, 128, 128
    expected_kernel = "decode" if num_tokens == 3 else "prefill"
    assert (
        cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=_device_sms(),
        )
        == expected_kernel
    )
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(8, 64))
    extra_cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(64, 2))
    indices = torch.randint(
        0, 8 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    extra_indices = torch.randint(
        0, 64 * 2, (num_tokens, extra_topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.full((num_tokens,), topk - 5, dtype=torch.int32, device="cuda")
    sm_scale = _D**-0.5
    runner = flashinfer.mla.SparseMLASm120Wrapper(
        max_num_tokens=512,
        max_num_heads=num_heads,
        kv_cache_format="nvfp4",
        backend="cake",
        device=q.device,
    )
    output = torch.empty_like(q)
    lse = runner.run(
        q.unsqueeze(1),
        cache,
        indices.unsqueeze(1),
        output.unsqueeze(1),
        sm_scale,
        topk_length=lengths,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices.unsqueeze(1),
        return_lse=True,
        lse_scale=math.log(2),
    )
    reference, reference_lse = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        lse_scale=math.log(2),
    )
    torch.testing.assert_close(output, reference, **_OUT_TOL)
    torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)
    # Caller-owned LSE; the prefill path leaves the wrapper's split arenas untouched.
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    output2 = torch.empty_like(q)
    returned = runner.run(
        q,
        cache,
        indices,
        output2,
        sm_scale,
        topk_length=lengths,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        out_lse=out_lse,
        return_lse=True,
        lse_scale=math.log(2),
    )
    assert returned.data_ptr() == out_lse.data_ptr()
    torch.testing.assert_close(output2, output, atol=0, rtol=0)
    torch.testing.assert_close(out_lse, lse, atol=0, rtol=0)
    if expected_kernel == "prefill":
        assert "mid_out" not in runner.__dict__.get("_cake_arenas", {})
    assert runner.run(q, cache, indices, output2, sm_scale) is None
    assert 64 in cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads()
