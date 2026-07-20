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

"""Resource-footprint regression tests for PrimTS FMHA decode."""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from cutlass import BFloat16, Float16, Float8E4M3FN
from cutlass import utils as cutlass_utils
from cutlass.experimental.task_scheduling.memory import SmemAllocation

from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
    make_decode_config,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
    _build_decode_gen_schedule,
    build_decode_task_manager,
)


def test_tensor_map_reexports_pinned_public_view_builder() -> None:
    """Keep kernel imports bound to the exact 4.7 public tensor-map API."""

    from cutlass.experimental.cuda.tensor_map import (
        create_tensor_map_tiled_from_view as stock_builder,
    )
    from flashinfer.attention.prims_ts.kernels.tensor_map import (
        create_tensor_map_tiled_from_view as exported_builder,
    )

    assert exported_builder is stock_builder


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _resources_by_name(resource_dependency_graph):
    resources_by_id = {}
    for resource, dependencies in resource_dependency_graph.items():
        resources_by_id[id(resource)] = resource
        for dependency in dependencies:
            resources_by_id[id(dependency)] = dependency
    return {resource.name: resource for resource in resources_by_id.values()}


def _make_contiguous_keeps_config(*, dtype, tile_size_q: int, headdim: int = 128):
    return make_decode_config(
        headdim=headdim,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": tile_size_q,
            "groups_tokens_heads_q": False,
        },
        seq_len_q=1,
        seq_len_kv=4096,
        batch_size=8,
        num_heads_q=4 * tile_size_q,
        num_heads_kv=4,
        qkv_dtype=dtype,
        o_dtype=Float16 if dtype == Float8E4M3FN else dtype,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )


def _make_paged_paired_keeps_config(*, page_size: int = 16):
    """Build the two-tile sliding case that crosses a 32-page-ID window."""
    pages_per_tile = 128 // page_size
    page_window_tiles = 32 // pages_per_tile
    seq_len_kv = (page_window_tiles + 1) * 128
    return make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "groups_tokens_heads_q": False,
        },
        seq_len_q=1,
        seq_len_kv=seq_len_kv,
        batch_size=1,
        num_heads_q=64,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=page_size,
        split_kv_mode="disabled",
        splits_kv=1,
        sliding_window_causal=True,
        attention_window_size=256,
        mask_type="causal",
        auto_tuner=False,
    )


@pytest.mark.parametrize("page_size", (16, 32, 64, 128))
def test_paired_page_offsets_cross_window_schedule_is_deadlock_and_race_free(
    page_size: int,
) -> None:
    """Validate K0 ending at page 31 and K1 beginning at page 32."""
    cfg = _make_paged_paired_keeps_config(page_size=page_size)
    pages_per_tile = cfg.tile_size_kv // page_size
    page_window_tiles = 32 // pages_per_tile
    seq_len_kv = (page_window_tiles + 1) * cfg.tile_size_kv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            cfg,
            seq_len_kv=seq_len_kv,
            batch_size=1,
            num_heads_kv=1,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )

    assert cfg.static_num_skipped_kv_tiles == page_window_tiles - 1
    assert cfg.total_kv_tiles == 2
    assert cfg.static_num_skipped_kv_tiles * pages_per_tile == 32 - pages_per_tile
    assert (cfg.static_num_skipped_kv_tiles + 1) * pages_per_tile == 32


def test_q128_fp16_contiguous_keeps_tmem_p_reduces_smem() -> None:
    """Keep the Q128 profile comfortably below the SM100 SMEM capacity."""
    cfg = make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 128,
            "groups_tokens_heads_q": True,
        },
        seq_len_q=1,
        seq_len_kv=4096,
        batch_size=148,
        num_heads_q=128,
        num_heads_kv=1,
        qkv_dtype=Float16,
        o_dtype=Float16,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )
    assert (cfg.q_stages, cfg.kv_stages, cfg.num_insts_kv, cfg.o_stages) == (
        1,
        4,
        2,
        2,
    )
    assert cfg.uses_tmem_p
    assert cfg.uses_two_inst_tmem_p
    assert cfg.keeps_stats_via_smem

    cfg.total_kv_tiles = 32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (
            _tasks,
            resource_dependency_graph,
            _dma_consumer_release_labels,
            smem_allocator,
            _tmem_allocator,
            _correction_resources,
        ) = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=cfg.total_kv_tiles,
            # A non-None descriptor selects the DSL kernel-construction path.
            # Resource construction stores but does not dereference this sentinel.
            tma_desc_q=object(),
        )

    resources_by_name = _resources_by_name(resource_dependency_graph)
    pipeline_configs = [
        resource.pipeline_config
        for resource in resources_by_name.values()
        if getattr(resource, "pipeline_config", None) is not None
        and getattr(resource, "pipeline_group", None) is None
    ]
    assert pipeline_configs
    assert all(config.barrier_ptr is None for config in pipeline_configs)
    expected_barrier_bytes = sum(
        config.num_stages * 2 * 8 for config in pipeline_configs
    )
    assert expected_barrier_bytes == smem_allocator.barrier_smem_bytes == 208

    tmem_ptr = smem_allocator.tmem_ptr_alloc
    assert isinstance(tmem_ptr, SmemAllocation)
    assert tmem_ptr.name == "fmha_tmem_ptr_i32"
    assert tmem_ptr.size_bytes == 4
    assert tmem_ptr.alignment == 4
    assert tmem_ptr.offset + tmem_ptr.size_bytes <= smem_allocator.total_smem_bytes

    assert smem_allocator.total_smem_bytes == 165_924
    unified_smem_bytes = (
        _align_up(smem_allocator.total_smem_bytes, 8)
        + smem_allocator.barrier_smem_bytes
    )
    sm100_capacity = cutlass_utils.get_smem_capacity_in_bytes("sm_100")
    assert unified_smem_bytes == 166_136
    assert sm100_capacity == 232_448
    assert unified_smem_bytes <= sm100_capacity
    assert sm100_capacity - unified_smem_bytes == 66_312
    launch_smem_bytes = _align_up(unified_smem_bytes, cfg.stensor_align)
    assert cfg.stensor_align == 1_024
    assert launch_smem_bytes == 166_912


@pytest.mark.parametrize(
    ("dtype", "expected_p_cols"),
    ((Float16, 64), (BFloat16, 64), (Float8E4M3FN, 32)),
)
@pytest.mark.parametrize("headdim", (64, 128))
def test_q128_two_inst_keeps_p_aliases_only_its_own_consumed_s_region(
    dtype, expected_p_cols: int, headdim: int
) -> None:
    """Guard the own-instance S/P overlay selected for every Q128 dtype."""
    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=128,
        headdim=headdim,
    )
    assert cfg.uses_two_inst_tmem_p
    assert cfg.uses_tmem_p

    cfg.total_kv_tiles = 32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (
            _tasks,
            resource_dependency_graph,
            _dma_consumer_release_labels,
            smem_allocator,
            tmem_allocator,
            _correction_resources,
        ) = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=cfg.total_kv_tiles,
            tma_desc_q=object(),
        )

    resources = _resources_by_name(resource_dependency_graph)
    s0 = resources["tmemS0"]._alloc
    s1 = resources["tmemS1"]._alloc
    p0 = resources["smemP0"]._tmem_alloc
    p1 = resources["smemP1"]._tmem_alloc
    o = resources["tmemO"]._alloc

    assert p0.num_columns == p1.num_columns == expected_p_cols
    assert (s0.offset, s1.offset) == (0, cfg.tmem_s_cols)
    assert s0.offset <= p0.offset
    assert p0.offset + p0.num_columns <= s0.offset + s0.num_columns
    assert s1.offset <= p1.offset
    assert p1.offset + p1.num_columns <= s1.offset + s1.num_columns
    assert o.offset >= s1.offset + s1.num_columns
    assert tmem_allocator.total_tmem_columns == cfg.tmem_total_cols <= 512

    # P has no SMEM allocation after selecting the overlay. Keep distinct K/V
    # resources for each instruction; the exhaustive checker below validates
    # the required PV_i -> QK_i overwrite order.
    assert resources["smemP0"]._alloc is None
    assert resources["smemP1"]._alloc is None
    assert {"smemK0", "smemK1", "smemV0", "smemV1"} <= resources.keys()
    unified_smem_bytes = (
        _align_up(smem_allocator.total_smem_bytes, 8)
        + smem_allocator.barrier_smem_bytes
    )
    assert unified_smem_bytes <= cutlass_utils.get_smem_capacity_in_bytes("sm_100")


@pytest.mark.parametrize("dtype", (Float16, BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("headdim", (64, 128))
def test_q64_keeps_uses_smem_p_within_capacity(dtype, headdim: int) -> None:
    """Q64 keeps P in SMEM so S can be released before the next QK wave."""
    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=64,
        headdim=headdim,
    )
    assert not cfg.uses_two_inst_tmem_p
    assert not cfg.uses_tmem_p

    cfg.total_kv_tiles = 32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (
            _tasks,
            resource_dependency_graph,
            _dma_consumer_release_labels,
            smem_allocator,
            _tmem_allocator,
            _correction_resources,
        ) = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=cfg.total_kv_tiles,
            tma_desc_q=object(),
        )

    resources = _resources_by_name(resource_dependency_graph)
    for name in ("smemP0", "smemP1"):
        p = resources[name]
        assert isinstance(p._alloc, SmemAllocation)
        assert p._alloc.name == name
        assert p._alloc.size_bytes == cfg.smem_p_tile_bytes
        assert p._tmem_alloc is None
    assert "tmemStatsDone0" not in resources
    assert "tmemStatsDone1" not in resources

    unified_smem_bytes = (
        _align_up(smem_allocator.total_smem_bytes, 8)
        + smem_allocator.barrier_smem_bytes
    )
    sm100_capacity = cutlass_utils.get_smem_capacity_in_bytes("sm_100")
    assert sm100_capacity == 232_448
    assert unified_smem_bytes <= sm100_capacity


@pytest.mark.parametrize("dtype", (Float16, BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("tile_size_q", (64, 128))
def test_d256_keeps_staged_tmem_p_has_overwrite_credit_gate(
    dtype, tile_size_q: int
) -> None:
    """The staged D256 S/P overlay has an overwrite-credit gate."""
    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=tile_size_q,
        headdim=256,
    )
    assert cfg.uses_staged_one_inst_tmem_p
    assert not cfg.uses_two_inst_tmem_p
    assert cfg.uses_tmem_p

    cfg.total_kv_tiles = 32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (
            _tasks,
            resource_dependency_graph,
            _dma_consumer_release_labels,
            _smem_allocator,
            _tmem_allocator,
            _correction_resources,
        ) = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=cfg.total_kv_tiles,
            tma_desc_q=object(),
        )

    resources = _resources_by_name(resource_dependency_graph)
    s = resources["tmemS0"]._alloc
    p = resources["smemP0"]._tmem_alloc
    assert "tmemStatsDone0" in resources
    assert "tmemStatsDone1" not in resources
    assert resources["smemP0"]._alloc is None
    assert s.offset <= p.offset
    assert p.offset + p.num_columns <= s.offset + s.num_columns


@pytest.mark.parametrize("dtype", (BFloat16, Float8E4M3FN))
def test_q128_two_inst_tmem_p_schedule_passes_exhaustive_alias_race_check(
    dtype,
) -> None:
    """Exhaust three KV tiles so HEAD, LOOP, and odd TAIL all reuse S/P."""
    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=128,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            cfg,
            seq_len_kv=3 * cfg.tile_size_kv,
            batch_size=1,
            num_heads_kv=4,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )


@pytest.mark.parametrize("dtype", (Float16, BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("headdim", (64, 128))
def test_q64_smem_p_schedule_passes_exhaustive_race_check(dtype, headdim: int) -> None:
    """Exhaust three KV tiles across every Q64 SMEM-P resource profile."""
    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=64,
        headdim=headdim,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            cfg,
            seq_len_kv=3 * cfg.tile_size_kv,
            batch_size=1,
            num_heads_kv=4,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )
