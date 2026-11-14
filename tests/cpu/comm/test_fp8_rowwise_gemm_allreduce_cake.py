"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Behavioural tests of the Cake fused FP8 row-wise GEMM + all-reduce backend that
need no GPU: the per-architecture route tables and the frozen program records,
workspace and launch geometry of every delivered architecture, operand
validation, the symmetric-memory rendezvous guard, workspace allocation
(readiness words in the signal pad or in a dedicated symmetric buffer, the
multicast view of the nvls carrier), the capacity-bound prepared launcher, the
launch argument binding, CUDA-graph capture rules and the public dispatch. The
tests parametrize over the loader's tables, so they cover exactly what the
export delivered. The multi-GPU numerical test lives in
``tests/comm/test_fp8_rowwise_gemm_allreduce_cake_e2e.py``.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from flashinfer.jit import cake_fp8_rowwise_gemm_allreduce as loader

backend = importlib.import_module(
    "flashinfer.comm.fp8_rowwise_gemm_allreduce.cake_fp8_rowwise_gemm_allreduce"
)

K_L = 768
FP8 = torch.float8_e4m3fn
ARCHES = sorted(loader.ROUTES)
# Every delivered (architecture, world size) pair.
DELIVERED = [
    (arch, world_size)
    for arch in ARCHES
    for world_size in sorted(loader.COMM_MODES[arch])
]
PROGRAMS = sorted(loader.PROGRAMS)
CAPABILITY = {arch: capability for capability, arch in loader.CAPABILITY_ARCH.items()}
UNDELIVERED_WORLD_SIZES = [
    world_size
    for world_size in (1, 2, 3, 4, 6, 8, 16)
    if world_size not in loader.SUPPORTED_WORLD_SIZES
]
SM_COUNT_ARCHES = [
    arch
    for arch in ARCHES
    if loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_SM_COUNT
]
CLUSTER_ARCHES = [
    arch
    for arch in ARCHES
    if loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS
]


def _n(arch):
    """A column count every architecture's tile divides (and 3072 does)."""

    return 3072 if 3072 % loader.block_n(arch) == 0 else 12 * loader.block_n(arch)


def _mode(arch, world_size):
    return loader.COMM_MODES[arch][world_size]


# --------------------------------------------------------------------------- #
# Route tables and frozen records
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_route_tables_cover_every_delivered_world_size_once():
    assert ARCHES and set(loader.ARCH_FLAGS) == set(ARCHES) == set(loader.GEOMETRY)
    assert set(loader.COMM_MODES) == set(ARCHES) == set(loader.FROZEN)
    assert set(CAPABILITY) == set(ARCHES)
    assert tuple(sorted(loader.SUPPORTED_WORLD_SIZES)) == loader.SUPPORTED_WORLD_SIZES
    routed = set()
    for arch in ARCHES:
        table = loader.ROUTES[arch]
        world_sizes = sorted(loader.COMM_MODES[arch])
        assert set(world_sizes) <= set(loader.SUPPORTED_WORLD_SIZES)
        expected_keys = set()
        for world_size in world_sizes:
            program = loader.main_program(world_size, arch, 1024)
            assert program in loader.PROGRAMS
            assert arch in loader.PROGRAMS[program]["arches"]
            expected_keys.add(loader.route_key(world_size, arch, 1024))
            for _ws, capacity, suffix in loader.CAPACITY_VARIANTS.get(arch, []):
                if int(_ws) == world_size:
                    key = loader.route_key(world_size, arch, capacity)
                    assert key.endswith(suffix) and key in table
                    expected_keys.add(key)
        # Every other key extends one of those capacity routes by a lever suffix.
        levers = loader.lever_suffixes(arch)[1:]
        extended = {base + lever for base in expected_keys for lever in levers}
        assert set(table) - expected_keys <= extended
        routed |= set(table.values())
        for world_size in UNDELIVERED_WORLD_SIZES:
            with pytest.raises(ValueError, match="delivered"):
                loader.main_program(world_size, arch, 1024)
    assert routed == set(loader.PROGRAMS)
    with pytest.raises(ValueError, match="delivered"):
        loader.main_program(loader.SUPPORTED_WORLD_SIZES[0], "sm_1a", 1024)


@pytest.mark.cpu_only
@pytest.mark.parametrize("program", PROGRAMS)
def test_program_records_carry_the_fused_launcher_contract(program):
    record = loader.PROGRAMS[program]
    # The device unit and its thin tvm-ffi launcher, both directly under the package
    # or both in one family directory of it, next to the shared headers they include.
    assert len(record["sources"]) == 2
    directories = set()
    for relative in record["sources"]:
        assert relative.startswith("csrc/cake_fp8_rowwise_gemm_allreduce/")
        directory = relative.split("/", 2)[2].rpartition("/")[0]
        assert directory in ("", *loader.SOURCE_DIRECTORIES)
        assert loader._source_path(relative).is_file()
        directories.add(directory)
    assert len(directories) == 1
    assert record["arches"] and set(record["arches"]) <= set(ARCHES)
    args = loader.launcher_args(program)
    assert args[-3:] == ("grid_x", "grid_y", "grid_z")
    assert len(set(args)) == len(args)
    common = {
        "A",
        "W",
        "OUT",
        "out_ptr",
        "sa",
        "sw",
        "ctl",
        "S_mc",
        "S",
        "flags",
        "M",
        "N",
        "K",
        "grid_n",
        "K_tiles",
        "C",
        "C_cap",
        "S_bytes",
        "flag_words",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "S_peers",
        "flags_peers",
        "grid_x",
        "grid_y",
        "grid_z",
    }
    for arch in record["arches"]:
        geometry = loader.GEOMETRY[arch]
        if geometry["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS:
            assert set(args) == common | {"pair_tiles"}
        else:
            assert set(args) == common | {"tiles", "group_m"}
        assert loader.cluster_dims(program) == tuple(geometry["CLUSTER_DIMS"])
        assert (
            0 < loader.dynamic_smem_bytes(program) <= geometry["MAX_DYNAMIC_SMEM_BYTES"]
        )
        modes = {
            _mode(arch, world_size)
            for world_size, key in (
                (world_size, loader.route_key(world_size, arch, 1024))
                for world_size in loader.COMM_MODES[arch]
            )
            if loader.ROUTES[arch][key] == program
        }
        for key, name in loader.ROUTES[arch].items():
            if name == program:
                modes.add(key.split("_")[2])
        assert modes
        for mode in modes:
            assert loader.launch_block(program) == (geometry["THREADS"][mode], 1, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_frozen_trace_time_choices_are_recorded_not_exposed(arch):
    assert loader.EXCHANGE_MODE == "X16"
    for world_size, mode in loader.COMM_MODES[arch].items():
        assert mode in ("p2p", "nvls")
        frozen = loader.FROZEN[arch][mode]
        assert frozen["timeout_cycles_log2"] == 34
        assert frozen["num_stages"] > 0
        if mode == "p2p":
            assert frozen["transport"] == "pull"
            assert loader.transport(world_size, arch) == "pull"
        else:
            assert "transport" not in frozen
            assert loader.transport(world_size, arch) is None
        assert loader.comm_mode(world_size, arch) == mode
    geometry = loader.GEOMETRY[arch]
    assert geometry["BLOCK_M"] == geometry["CHUNK_ROWS"] == loader.BLOCK_M == 128
    assert geometry["BLOCK_K"] == loader.BLOCK_K == 128
    assert geometry["TILE_BYTES"] == 128 * geometry["BLOCK_N"] * geometry["OUT_BYTES"]
    assert geometry["GRID_RULE"] in (
        loader.GRID_RULE_CLUSTER_PAIRS,
        loader.GRID_RULE_SM_COUNT,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "program, arch",
    [
        (program, arch)
        for program in PROGRAMS
        for arch in loader.PROGRAMS[program]["arches"]
    ],
)
def test_spec_names_carry_the_exact_architecture(monkeypatch, program, arch):
    specs = []

    def gen_jit_spec(**kwargs):
        specs.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(loader, "gen_jit_spec", gen_jit_spec)
    monkeypatch.setattr(
        loader, "_source_path", lambda relative: f"/csrc/{relative.rsplit('/', 1)[-1]}"
    )
    monkeypatch.setattr(loader, "_source_dir", lambda: loader.Path("/csrc/pkg"))
    loader.spec.cache_clear()
    try:
        spec = loader.spec(program, arch)
        flags = list(loader.ARCH_FLAGS[arch])
        assert spec.name == f"cake_fp8_rowwise_gemm_allreduce_{arch}_{program}"
        assert spec.extra_cuda_cflags[: len(flags)] == flags
        assert spec.extra_cuda_cflags[len(flags) :] == list(loader.COMPILE_FLAGS)
        assert spec.use_fast_math == ("--use_fast_math" in loader.COMPILE_FLAGS)
        assert len(spec.sources) == 2
        other = next(
            (
                candidate
                for candidate in ("sm_90a", "sm_100a", "sm_103a")
                if candidate not in loader.PROGRAMS[program]["arches"]
            ),
            "sm_1a",
        )
        with pytest.raises(ValueError, match=other):
            loader.spec(program, other)
    finally:
        loader.spec.cache_clear()
    assert len(specs) == 1


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows, expected",
    [(1, 128), (127, 128), (128, 128), (129, 256), (1024, 1024), (4133, 4224)],
)
def test_ceil128_rounds_up_to_the_chunk(rows, expected):
    assert loader.ceil128(rows) == expected
    assert loader.chunk_count(rows) == expected // 128
    assert loader.chunk_capacity(rows) == expected // 128


@pytest.mark.cpu_only
def test_ceil128_rejects_non_positive_rows():
    for rows in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            loader.ceil128(rows)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_workspace_bytes_follow_the_slot_and_row_rules(arch, world_size):
    mode = _mode(arch, world_size)
    n = _n(arch)
    tile = loader.tile_bytes(arch)
    gn = loader.grid_n(n, arch)
    slots = loader.slots_per_chunk(world_size, mode)
    assert slots == (1 if mode == "nvls" or world_size <= 2 else world_size)
    for max_rows in (1, 129, 1024, 8192):
        c_cap = loader.chunk_capacity(max_rows)
        rows = loader.s_rows(c_cap, world_size, arch)
        if (
            loader.GEOMETRY[arch]["S_ROWS_OWNED_ONLY_UNDER_PULL"]
            and mode == "p2p"
            and loader.transport(world_size, arch) == "pull"
        ):
            assert rows == -(-c_cap // world_size)
        else:
            assert rows == c_cap
        assert (
            loader.workspace_bytes(max_rows, n, world_size, arch)
            == rows * slots * gn * tile
        )
    with pytest.raises(ValueError, match="world size"):
        loader.slots_per_chunk(UNDELIVERED_WORLD_SIZES[0], mode)


@pytest.mark.cpu_only
@pytest.mark.skipif(not CLUSTER_ARCHES, reason="no cluster-pair architecture delivered")
def test_blackwell_workspace_is_two_n_times_slots_times_ceil128_rows():
    arch = CLUSTER_ARCHES[0]
    assert loader.tile_bytes(arch) == 65536 and loader.block_n(arch) == 256
    for world_size in sorted(loader.COMM_MODES[arch]):
        slots = loader.slots_per_chunk(world_size, _mode(arch, world_size))
        # C_cap=8, 12 column tiles, 64 KiB tiles: 2 * N * slots * 1024 bytes.
        assert (
            loader.workspace_bytes(1024, 3072, world_size, arch)
            == 2 * 3072 * slots * 1024
        )
        assert loader.workspace_bytes(1, 256, world_size, arch) == slots * 65536


@pytest.mark.cpu_only
@pytest.mark.skipif(not SM_COUNT_ARCHES, reason="no SM-count architecture delivered")
def test_hopper_workspace_compacts_owned_rows_under_pull():
    arch = SM_COUNT_ARCHES[0]
    assert loader.tile_bytes(arch) == 32768 and loader.block_n(arch) == 128
    for world_size in sorted(loader.COMM_MODES[arch]):
        if _mode(arch, world_size) != "p2p":
            continue
        slots = loader.slots_per_chunk(world_size, "p2p")
        rows = -(-8 // world_size)
        assert (
            loader.workspace_bytes(1024, 3072, world_size, arch)
            == rows * slots * 24 * 32768
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "max_rows, flag_bytes", [(1024, 64), (8192, 512), (65536, 4096), (1, 8)]
)
def test_flag_words_are_two_per_chunk(arch, max_rows, flag_bytes):
    assert loader.flag_bytes(max_rows, arch) == flag_bytes
    assert loader.flag_words(max_rows) == flag_bytes // 4
    assert loader.GEOMETRY[arch]["FLAG_WORD_BYTES_PER_CHUNK"] == 8


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_ctl_words_follow_the_control_block_layout(arch):
    geometry = loader.GEOMETRY[arch]
    assert geometry["MAX_CTAS"] == 1024 and geometry["CTL_EXTRA_WORDS"] == 2
    # arr[C_cap] | red_done[C_cap] | cta_arrivals[1024] | exit | EP (| steal[C_cap]) (| sched)
    steal = 8 if geometry["CTL_STEAL_WORDS"] else 0
    sched = 1 if geometry["CTL_SCHED_WORDS"] else 0
    assert loader.ctl_words(8, arch) == 2 * 8 + 1024 + 2 + steal + sched
    assert loader.ctl_words(1, arch) == 2 + 1024 + 2 + (1 if steal else 0) + sched
    with pytest.raises(ValueError, match="positive"):
        loader.ctl_words(0, arch)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_tiles_count_the_persistent_tiles_of_a_call(arch):
    n = _n(arch)
    gn = loader.grid_n(n, arch)
    assert gn == n // loader.block_n(arch)
    if loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS:
        assert loader.tiles(1, n, arch) == gn
        assert loader.tiles(256, n, arch) == gn
        assert loader.tiles(257, n, arch) == 2 * gn
        assert loader.tiles(65536, n, arch) == 256 * gn
    else:
        assert loader.tiles(1, n, arch) == gn
        assert loader.tiles(128, n, arch) == gn
        assert loader.tiles(129, n, arch) == 2 * gn
        assert loader.tiles(65536, n, arch) == 512 * gn
    with pytest.raises(ValueError, match="M must be positive"):
        loader.tiles(0, n, arch)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", CLUSTER_ARCHES)
@pytest.mark.parametrize(
    "m, n, max_clusters, expected",
    [
        (1, 3072, 74, (24, 1, 1)),
        (257, 3072, 74, (48, 1, 1)),
        (65536, 3072, 74, (148, 1, 1)),
        (65536, 3072, 70, (140, 1, 1)),
        (37, 256, 74, (2, 1, 1)),
    ],
)
def test_cluster_pair_grid_is_two_ctas_per_co_resident_cluster_bounded_by_the_pair_tiles(
    arch, m, n, max_clusters, expected
):
    assert loader.launch_grid(m, n, arch, max_clusters=max_clusters) == expected
    with pytest.raises(ValueError, match="max_clusters"):
        loader.launch_grid(1, 256, arch, max_clusters=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", SM_COUNT_ARCHES)
@pytest.mark.parametrize(
    "m, n, sm_count, expected",
    [
        (1, 3072, 132, (24, 1, 1)),
        (129, 3072, 132, (48, 1, 1)),
        (65536, 3072, 132, (132, 1, 1)),
        (65536, 3072, 114, (114, 1, 1)),
        (37, 128, 132, (1, 1, 1)),
    ],
)
def test_sm_count_grid_is_one_cta_per_sm_bounded_by_the_tiles(
    arch, m, n, sm_count, expected
):
    assert loader.launch_grid(m, n, arch, sm_count=sm_count) == expected
    with pytest.raises(ValueError, match="sm_count"):
        loader.launch_grid(1, 128, arch, sm_count=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", SM_COUNT_ARCHES)
def test_raster_group_follows_the_l2_footprint_rule(arch):
    geometry = loader.GEOMETRY[arch]
    assert geometry["RASTER_GROUP_CHOICES"][0] == 1
    # Short K_l, few column tiles or few chunks: the lever does not apply.
    assert loader.raster_group_m(4096, 3072, 768, arch) == 1
    assert loader.raster_group_m(4096, 1024, 8192, arch) == 1
    assert loader.raster_group_m(256, 3072, 8192, arch) == 1
    sm_count = geometry["RASTER_SM_COUNT"]
    half_l2 = 0.5 * geometry["RASTER_L2_BYTES"]
    for m, n, k_l in ((4096, 3072, 6144), (16384, 8192, 16384), (131072, 8192, 4096)):
        group = loader.raster_group_m(m, n, k_l, arch)
        choices = [
            g for g in geometry["RASTER_GROUP_CHOICES"] if g <= loader.chunk_count(m)
        ]
        assert group in choices
        live = [(g, (sm_count / g + g) * 128 * k_l) for g in choices]
        fits = [g for g, bytes_live in live if bytes_live <= half_l2]
        assert group == (fits[0] if fits else min(live, key=lambda item: item[1])[0])


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", CLUSTER_ARCHES)
def test_cluster_pair_programs_carry_no_raster_group(arch):
    assert loader.raster_group_m(131072, 8192, 16384, arch) == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_validate_shape_enforces_the_operand_contract(arch):
    width = loader.block_n(arch)
    assert loader.validate_shape(1, 256, 128, arch) == (1, 256, 128)
    assert loader.validate_shape(4133, 3072, 768, arch) == (4133, 3072, 768)
    assert loader.validate_shape(1 << 20, 1024, 128, arch) == (1 << 20, 1024, 128)
    with pytest.raises(ValueError, match="M must be positive"):
        loader.validate_shape(0, 256, 128, arch)
    for n in (64, 1000, 1281, 3073):
        with pytest.raises(ValueError, match=f"multiple of {width}"):
            loader.validate_shape(1, n, 128, arch)
    # Tile multiples outside the 256 contract (they pass grid_n on sm_90a).
    for n in (128, 384, 640):
        with pytest.raises(ValueError, match="multiple of 256"):
            loader.validate_shape(1, n, 128, arch)
    if width == 128:
        assert loader.grid_n(384, arch) == 3
    for k_l in (0, 64, 769):
        with pytest.raises(ValueError, match="multiple of 128"):
            loader.validate_shape(1, 256, k_l, arch)
    with pytest.raises(ValueError, match=r"2\*\*30"):
        loader.validate_shape((1 << 20) + 1, 1024, 128, arch)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_validate_shape_bounds_n_so_vector_indices_fit_int32(arch):
    limit = loader.MAX_COLS
    assert limit == (1 << 27) - (1 << 15) and limit % loader.N_MULTIPLE == 0
    assert 16 * limit + 512 * int(loader.GEOMETRY[arch]["MAX_CTAS"]) <= 2**31
    assert loader.validate_shape(1, limit, 128, arch) == (1, limit, 128)
    for n in (limit + 256, 1 << 27, (1 << 27) + 256):
        assert n <= loader.MAX_ROWS_TIMES_COLS
        with pytest.raises(ValueError, match=str(limit)):
            loader.validate_shape(1, n, 128, arch)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_capacity_variants_select_a_program_by_exact_rows_capacity(arch):
    for world_size, capacity, suffix in loader.CAPACITY_VARIANTS.get(arch, []):
        assert capacity % 128 == 0 and suffix.startswith("_")
        assert loader.program_variant(world_size, capacity, arch) == suffix
        assert loader.program_variant(world_size, capacity - 1, arch) == suffix
        assert loader.program_variant(world_size, capacity + 1, arch) == ""
        assert loader.program_variant(world_size, capacity - 128, arch) == ""
        base = loader.main_program(world_size, arch, capacity + 128)
        variant = loader.main_program(world_size, arch, capacity)
        assert base != variant and {base, variant} <= set(loader.PROGRAMS)
    for world_size in loader.COMM_MODES[arch]:
        assert loader.program_variant(world_size, 1, arch) == ""


def _capacities(arch, world_size):
    """A plain capacity and every capacity variant of ``world_size`` on ``arch``."""

    return [1024, 131072] + [
        int(capacity)
        for ws, capacity, _suffix in loader.CAPACITY_VARIANTS.get(arch, [])
        if int(ws) == world_size
    ]


# Call shapes around every threshold of the lever rule (chunk counts, columns, K_l).
LEVER_CHUNKS = (1, 4, 32, 95, 96, 97, 191, 192, 193, 512, 1024)
LEVER_COLUMNS = (256, 3072, 8192, 12288, 16384, 49152)
LEVER_K_L = (128, 768, 1536, 1664, 2048, 4096, 4224, 8192)


def _without_drain(arch, lever):
    """``lever`` without the drainer lever's suffix (it comes last)."""

    drain = loader.LEVER_DISPATCH[arch].get("DRAIN")
    if drain and lever.endswith(drain["suffix"]):
        return lever[: -len(drain["suffix"])]
    return lever


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_lever_table_records_each_lever_once(arch):
    table = loader.LEVER_DISPATCH[arch]
    assert set(table) <= {"REDUCERS", "LOCKSTEP", "DRAIN"}
    suffixes = loader.lever_suffixes(arch)
    assert suffixes[0] == "" and len(suffixes) == 2 ** len(table)
    assert len(set(suffixes)) == len(suffixes)
    if "REDUCERS" in table:
        reducers = table["REDUCERS"]
        assert reducers["suffix"].startswith("_")
        assert set(reducers) <= {"suffix", "max_k_l", "mn_limit"}
        assert set(reducers["max_k_l"]) <= set(loader.COMM_MODES[arch])
        assert all(k > 0 and k % 128 == 0 for k in reducers["max_k_l"].values())
        limits = reducers.get("mn_limit", {})
        assert set(limits) <= set(reducers["max_k_l"])
        assert all(limit > 0 for limit in limits.values())
    if "LOCKSTEP" in table:
        lockstep = table["LOCKSTEP"]
        assert set(lockstep) == {"suffix", "min_raster_group", "min_tiles", "variants"}
        assert lockstep["suffix"].startswith("_") and lockstep["min_tiles"] > 0
        assert lockstep["min_raster_group"] >= 2
        assert loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_SM_COUNT
        capacity_suffixes = {row[2] for row in loader.CAPACITY_VARIANTS.get(arch, [])}
        assert lockstep["variants"][0] == ""
        assert set(lockstep["variants"]) <= {"", *capacity_suffixes}
    if "DRAIN" in table:
        drain = table["DRAIN"]
        assert set(drain) == {"suffix", "worlds", "max_m", "variants"}
        assert drain["suffix"].startswith("_") and drain["max_m"] > 0
        assert set(drain["worlds"]) <= set(loader.COMM_MODES[arch])
        assert all(world_size >= 2 for world_size in drain["worlds"])
        capacity_suffixes = {row[2] for row in loader.CAPACITY_VARIANTS.get(arch, [])}
        assert drain["variants"][0] == ""
        assert set(drain["variants"]) <= {"", *capacity_suffixes}
    for world_size in loader.COMM_MODES[arch]:
        for chunks in LEVER_CHUNKS:
            for n in LEVER_COLUMNS:
                for k_l in LEVER_K_L:
                    lever = loader.lever_variant(
                        world_size, chunks * 128, n, k_l, arch, sm_count=132
                    )
                    assert lever in suffixes
                    if not table:
                        assert lever == ""


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_reducer_lever_follows_its_k_l_bound_and_the_column_count(arch, world_size):
    reducers = loader.LEVER_DISPATCH[arch].get("REDUCERS")
    bound = None if reducers is None else reducers["max_k_l"].get(world_size)
    if bound is None:
        for k_l in LEVER_K_L:
            lever = loader.lever_variant(
                world_size, 128, 1 << 16, k_l, arch, sm_count=1
            )
            assert lever == ""
        return
    suffix = reducers["suffix"]
    n = -(-bound * world_size // 256) * 256
    assert loader.lever_variant(world_size, 128, n, bound, arch, sm_count=1) == suffix
    assert loader.lever_variant(world_size, 128, n, 128, arch, sm_count=1) == suffix
    assert (
        loader.lever_variant(world_size, 128, n + 256, bound + 128, arch, sm_count=1)
        == ""
    )
    # K_l * world_size above N keeps the route's program.
    small = -(-(bound * world_size) // 256) * 256 - 256
    if small >= 256 and bound * world_size > small:
        assert (
            loader.lever_variant(world_size, 128, small, bound, arch, sm_count=1) == ""
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_reducer_lever_stops_at_its_m_n_limit(arch, world_size):
    table = loader.LEVER_DISPATCH[arch]
    reducers = table.get("REDUCERS")
    if reducers is None or world_size not in reducers["max_k_l"]:
        return
    lockstep = table.get("LOCKSTEP", {}).get("suffix", "")

    def reducer_part(lever: str) -> str:
        # Large calls can also select the lockstep lever, whose suffix comes last.
        if lockstep and lever.endswith(lockstep):
            return lever[: -len(lockstep)]
        return lever

    bound = reducers["max_k_l"][world_size]
    n = -(-bound * world_size // 256) * 256
    limit = reducers.get("mn_limit", {}).get(world_size)
    if limit is None:
        # Without a limit the reducer lever holds up to the largest M.
        m = loader.MAX_ROWS_TIMES_COLS // n
        lever = loader.lever_variant(world_size, m, n, bound, arch, sm_count=1)
        assert reducer_part(lever) == reducers["suffix"]
        return
    above = -(-limit // n)
    below = loader.lever_variant(world_size, above - 1, n, bound, arch, sm_count=1)
    at = loader.lever_variant(world_size, above, n, bound, arch, sm_count=1)
    assert reducer_part(below) == reducers["suffix"] and reducer_part(at) == ""


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_lockstep_lever_needs_the_raster_group_the_tiles_and_a_bounded_grid(
    arch, world_size
):
    lockstep = loader.LEVER_DISPATCH[arch].get("LOCKSTEP")
    n, k_l = 16384, 8192
    if lockstep is None:
        # Without the lockstep lever a call selects at most the reducer lever.
        table = loader.LEVER_DISPATCH[arch]
        allowed = {"", *(entry["suffix"] for entry in table.values())}
        for chunks in LEVER_CHUNKS:
            lever = loader.lever_variant(
                world_size, chunks * 128, n, k_l, arch, sm_count=132
            )
            assert lever in allowed
        return
    suffix = lockstep["suffix"]
    edge = -(-lockstep["min_tiles"] // loader.grid_n(n, arch)) * 128
    assert loader.raster_group_m(edge, n, k_l, arch) >= lockstep["min_raster_group"]
    assert loader.tiles(edge, n, arch) >= lockstep["min_tiles"]
    assert loader.tiles(edge - 128, n, arch) < lockstep["min_tiles"]
    # The drainer lever may follow (its suffix comes last).
    lever = loader.lever_variant(world_size, edge, n, k_l, arch, sm_count=132)
    assert _without_drain(arch, lever).endswith(suffix)
    assert not loader.lever_variant(
        world_size, edge - 128, n, k_l, arch, sm_count=132
    ).endswith(suffix)
    max_ctas = int(loader.GEOMETRY[arch]["MAX_CTAS"])
    for sm_count in (0, max_ctas, max_ctas + 1):
        lever = loader.lever_variant(world_size, edge, n, k_l, arch, sm_count=sm_count)
        assert not lever.endswith(suffix)
    # A raster group of 1 (K_l below the raster threshold) keeps the throttle off.
    short = int(loader.GEOMETRY[arch]["RASTER_MIN_K"]) - 128
    assert loader.raster_group_m(edge, n, short, arch) == 1
    assert not loader.lever_variant(
        world_size, edge, n, short, arch, sm_count=132
    ).endswith(suffix)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_every_call_selects_a_delivered_program_of_its_route(arch, world_size):
    for capacity in _capacities(arch, world_size):
        base = loader.route_key(world_size, arch, capacity)
        group = set(
            loader.dispatch_programs(world_size, arch, capacity, 3072, sm_count=132)
        )
        assert loader.main_program(world_size, arch, capacity) in group
        assert group <= set(loader.ROUTES[arch].values())
        for chunks in LEVER_CHUNKS:
            m = chunks * 128
            for n in LEVER_COLUMNS:
                if m > capacity or m * n > loader.MAX_ROWS_TIMES_COLS:
                    continue
                for k_l in LEVER_K_L:
                    program = loader.call_program(
                        world_size, arch, capacity, m, n, k_l, sm_count=132
                    )
                    lever = loader.lever_variant(
                        world_size,
                        m,
                        n,
                        k_l,
                        arch,
                        sm_count=132,
                        variant=loader.program_variant(world_size, capacity, arch),
                    )
                    assert program == loader.ROUTES[arch][base + lever]
                    assert (
                        program in group and arch in loader.PROGRAMS[program]["arches"]
                    )


def _variant_classes(arch, world_size):
    """``(program variant, capacity rows)`` of ``world_size`` on ``arch``: the default
    variant (a capacity no variant names) and every capacity variant."""

    classes = [("", None)]
    for ws, capacity, suffix in loader.CAPACITY_VARIANTS.get(arch, []):
        if int(ws) == world_size:
            classes.append((str(suffix), int(capacity)))
    return classes


def _unrouted_key_certificate(arch, world_size, variant, lever):
    """Why no call of ``world_size`` in program variant ``variant`` selects ``lever``,
    or ``None``. Each reason is a contradiction between the conditions of
    ``lever_variant``: the lockstep lever is not offered to this variant; the reducer
    lever has no ``K_l`` bound at this world size; or the reducer bound is below the
    raster threshold that the lockstep lever needs (``raster_group_m`` is 1 for
    ``K_l < RASTER_MIN_K``); or the drainer lever is not offered to this world size or
    to this variant."""

    table = loader.LEVER_DISPATCH[arch]
    reducers, lockstep = table.get("REDUCERS"), table.get("LOCKSTEP")
    drain = table.get("DRAIN")
    rest = lever
    with_drain = bool(drain) and rest.endswith(drain["suffix"])
    if with_drain:
        rest = rest[: -len(drain["suffix"])]
    with_lockstep = bool(lockstep) and rest.endswith(lockstep["suffix"])
    if with_lockstep:
        rest = rest[: -len(lockstep["suffix"])]
    with_reducers = bool(reducers) and rest == reducers["suffix"]
    assert rest in ("", reducers["suffix"] if reducers else "")
    if with_drain and world_size not in drain["worlds"]:
        return "the drainer is not offered to the world size"
    if with_drain and variant not in drain["variants"]:
        return "the drainer is not offered to the variant"
    if with_lockstep and variant not in lockstep["variants"]:
        return "lockstep is not offered to the variant"
    bound = reducers["max_k_l"].get(world_size) if reducers else None
    if with_reducers and bound is None:
        return "no reducer K_l bound at this world size"
    raster_min_k = loader.GEOMETRY[arch].get("RASTER_MIN_K")
    if with_reducers and with_lockstep and bound < raster_min_k:
        return "reducer K_l bound below the raster threshold"
    return None


def _edge_columns(arch, world_size, chunk_counts):
    """Column counts on both sides of every column threshold of the lever rule, the
    smallest and the largest admissible ``N``."""

    width, step = loader.block_n(arch), loader.N_MULTIPLE
    geometry = loader.GEOMETRY[arch]
    edges = {step, loader.MAX_COLS // step * step, 65536, 131072}
    if "RASTER_MIN_GRID_N" in geometry:
        edges.add(int(geometry["RASTER_MIN_GRID_N"]) * width)
    table = loader.LEVER_DISPATCH[arch]
    if "REDUCERS" in table:
        edges |= {k * world_size for k in table["REDUCERS"]["max_k_l"].values()}
    if "LOCKSTEP" in table:
        for chunks in chunk_counts:
            edges.add(-(-table["LOCKSTEP"]["min_tiles"] // chunks) * width)
    columns = set()
    for edge in edges:
        base = -(-edge // step) * step
        columns |= {base - step, base, base + step}
    return sorted(n for n in columns if step <= n <= loader.MAX_COLS)


def _edge_k_l(arch):
    """``K_l`` on both sides of every ``K_l`` threshold, and a large ``K_l``."""

    table = loader.LEVER_DISPATCH[arch]
    edges = {128, 1 << 16}
    if "REDUCERS" in table:
        edges |= set(table["REDUCERS"]["max_k_l"].values())
    if "RASTER_MIN_K" in loader.GEOMETRY[arch]:
        edges.add(int(loader.GEOMETRY[arch]["RASTER_MIN_K"]))
    return sorted({k + d for k in edges for d in (-128, 0, 128) if k + d > 0})


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_every_reachable_lever_key_is_a_route(arch, world_size):
    # Proof: every lever key outside ROUTES carries a contradiction of the rule.
    mode = loader.comm_mode(world_size, arch)
    for variant, _capacity in _variant_classes(arch, world_size):
        base = f"main_ws{world_size}_{mode}{variant}"
        assert base in loader.ROUTES[arch]
        for lever in loader.lever_suffixes(arch):
            if base + lever not in loader.ROUTES[arch]:
                reason = _unrouted_key_certificate(arch, world_size, variant, lever)
                assert reason is not None, base + lever
    # The certificate's raster premise, on the threshold grid below.
    raster_min_k = loader.GEOMETRY[arch].get("RASTER_MIN_K")
    # Check: M over every chunk count of a capacity variant (both ends of each chunk)
    # and over the threshold chunk counts of the default variant; N and K_l on both
    # sides of every threshold up to the largest admissible N.
    tile_rule = loader.LEVER_DISPATCH[arch].get("LOCKSTEP", {}).get("min_tiles", 1)
    reached = set()
    for variant, capacity in _variant_classes(arch, world_size):
        if capacity is None:
            chunk_counts = {1, 2, 3, 4, 5, 32, 33, 1024, 8192, 32768}
            raster_chunks = loader.GEOMETRY[arch].get("RASTER_MIN_CHUNKS", 1)
            chunk_counts |= {raster_chunks - 1, raster_chunks, raster_chunks + 1}
            drain = loader.LEVER_DISPATCH[arch].get("DRAIN")
            if drain:
                # the drainer's row bound and its grid of one CTA per peer
                edge = -(-int(drain["max_m"]) // 128)
                chunk_counts |= {edge, edge + 1, world_size - 2, world_size - 1}
            for gn in (16, 64):
                chunk_counts |= {-(-tile_rule // gn) - 1, -(-tile_rule // gn)}
            capacity_rows = None
        else:
            chunk_counts = set(range(1, capacity // 128 + 1))
            capacity_rows = capacity
        chunk_counts = sorted(c for c in chunk_counts if c > 0)
        rows = sorted({m for c in chunk_counts for m in ((c - 1) * 128 + 1, c * 128)})
        for n in _edge_columns(arch, world_size, chunk_counts):
            for k_l in _edge_k_l(arch):
                for m in rows:
                    if m * n > loader.MAX_ROWS_TIMES_COLS:
                        break
                    max_rows = capacity_rows or loader.ceil128(m)
                    if loader.program_variant(world_size, max_rows, arch) != variant:
                        max_rows += 128  # the default capacity next to a variant's
                    assert loader.program_variant(world_size, max_rows, arch) == variant
                    loader.validate_shape(m, n, k_l, arch)
                    program = loader.call_program(
                        world_size, arch, max_rows, m, n, k_l, sm_count=132
                    )
                    key = loader.route_key(world_size, arch, max_rows)
                    key += loader.lever_variant(
                        world_size, m, n, k_l, arch, sm_count=132, variant=variant
                    )
                    assert loader.ROUTES[arch][key] == program
                    reached.add(key)
                    if raster_min_k is not None and k_l < raster_min_k:
                        assert loader.raster_group_m(m, n, k_l, arch) == 1
    # Not vacuous: the grid reaches every routed key of this world size.
    prefix = f"main_ws{world_size}_{mode}"
    assert reached == {key for key in loader.ROUTES[arch] if key.startswith(prefix)}


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_capacity_variants_outside_the_lockstep_variants_never_throttle(arch):
    lockstep = loader.LEVER_DISPATCH[arch].get("LOCKSTEP")
    if lockstep is None:
        return
    step = loader.N_MULTIPLE
    for world_size, capacity, suffix in loader.CAPACITY_VARIANTS.get(arch, []):
        if suffix in lockstep["variants"]:
            continue
        # A call that meets every other lockstep condition at this capacity: all
        # chunks of the capacity, K_l at the raster threshold, the tile threshold.
        chunks = capacity // 128
        n = -(-lockstep["min_tiles"] // chunks) * loader.block_n(arch)
        n = -(-n // step) * step
        k_l = int(loader.GEOMETRY[arch]["RASTER_MIN_K"])
        assert loader.tiles(capacity, n, arch) >= lockstep["min_tiles"]
        plain = loader.lever_variant(world_size, capacity, n, k_l, arch, sm_count=132)
        lever = loader.lever_variant(
            world_size, capacity, n, k_l, arch, sm_count=132, variant=suffix
        )
        # The default variant at the same call takes the throttle, this one not.
        assert _without_drain(arch, plain).endswith(lockstep["suffix"])
        assert not lever.endswith(lockstep["suffix"])
        key = loader.route_key(world_size, arch, capacity) + plain
        assert key not in loader.ROUTES[arch]
        main = loader.main_program(world_size, arch, capacity)
        program = loader.call_program(
            world_size, arch, capacity, capacity, n, k_l, sm_count=132
        )
        programs = loader.dispatch_programs(
            world_size, arch, capacity, n, k_l, sm_count=132
        )
        assert program == main and programs == (main,)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_drain_lever_needs_its_world_size_its_row_bound_and_a_cta_per_peer(
    arch, world_size
):
    drain = loader.LEVER_DISPATCH[arch].get("DRAIN")
    # N = 3072, K_l = 3072 take neither the reducer nor the lockstep lever.
    n, k_l = 3072, 3072
    if drain is None or world_size not in drain["worlds"]:
        for chunks in LEVER_CHUNKS:
            lever = loader.lever_variant(
                world_size, chunks * 128, n, k_l, arch, sm_count=132
            )
            assert lever == ""
        return
    suffix, max_m = drain["suffix"], int(drain["max_m"])
    for m in (1, 128, 4096, max_m - 1, max_m):
        assert loader.lever_variant(world_size, m, n, k_l, arch, sm_count=132) == suffix
    for m in (max_m + 1, max_m + 128, loader.MAX_ROWS_TIMES_COLS // n):
        assert loader.lever_variant(world_size, m, n, k_l, arch, sm_count=132) == ""
    # One CTA per peer: the grid is min(sm_count, tiles).
    for sm_count in (world_size - 2, world_size - 1, 132):
        lever = loader.lever_variant(world_size, 4096, n, k_l, arch, sm_count=sm_count)
        assert lever == (suffix if sm_count >= world_size - 1 else "")
    narrow = loader.N_MULTIPLE
    edge = -(-(world_size - 1) // loader.grid_n(narrow, arch)) * 128
    assert loader.tiles(edge, narrow, arch) >= world_size - 1
    lever = loader.lever_variant(world_size, edge, narrow, 128, arch, sm_count=132)
    assert lever == suffix
    if edge > 128:
        assert loader.tiles(edge - 128, narrow, arch) < world_size - 1
        lever = loader.lever_variant(
            world_size, edge - 128, narrow, 128, arch, sm_count=132
        )
        assert lever == ""


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_drain_lever_is_only_offered_to_its_program_variants(arch):
    drain = loader.LEVER_DISPATCH[arch].get("DRAIN")
    if drain is None:
        return
    n, k_l = 3072, 3072
    for world_size, capacity, suffix in loader.CAPACITY_VARIANTS.get(arch, []):
        if suffix in drain["variants"] or world_size not in drain["worlds"]:
            continue
        # A call of the whole capacity that meets every other drainer condition.
        plain = loader.lever_variant(world_size, capacity, n, k_l, arch, sm_count=132)
        lever = loader.lever_variant(
            world_size, capacity, n, k_l, arch, sm_count=132, variant=suffix
        )
        assert plain == drain["suffix"] and lever == ""
        key = loader.route_key(world_size, arch, capacity) + plain
        assert key not in loader.ROUTES[arch]
        main = loader.main_program(world_size, arch, capacity)
        program = loader.call_program(
            world_size, arch, capacity, capacity, n, k_l, sm_count=132
        )
        programs = loader.dispatch_programs(
            world_size, arch, capacity, n, k_l, sm_count=132
        )
        assert program == main and programs == (main,)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_bound_k_l_dispatch_programs_are_the_programs_of_every_row_count(
    arch, world_size
):
    for capacity in (4133, *_capacities(arch, world_size)[2:]):
        for n, k_l in ((3072, 768), (8192, 4096), (16384, 8192)):
            programs = loader.dispatch_programs(
                world_size, arch, capacity, n, k_l, sm_count=132
            )
            calls = {
                loader.call_program(world_size, arch, capacity, m, n, k_l, sm_count=132)
                for m in range(1, capacity + 1, 64)
            }
            calls.add(
                loader.call_program(
                    world_size, arch, capacity, capacity, n, k_l, sm_count=132
                )
            )
            assert programs[0] == loader.main_program(world_size, arch, capacity)
            assert set(programs) == calls | {programs[0]}


# Architectures whose calls all run the main program (no lever table entry).
NO_LEVER_ARCHES = [arch for arch in ARCHES if not loader.LEVER_DISPATCH.get(arch)]


@pytest.mark.cpu_only
@pytest.mark.skipif(
    not NO_LEVER_ARCHES, reason="every delivered architecture has a lever"
)
def test_dispatch_programs_without_a_lever_is_the_main_program_only(monkeypatch):
    def no_call(*args, **kwargs):
        raise AssertionError("dispatch_programs enumerated a call without a lever")

    monkeypatch.setattr(loader, "call_program", no_call)
    for arch, world_size in DELIVERED:
        if arch not in NO_LEVER_ARCHES:
            continue
        for capacity in _capacities(arch, world_size):
            main = loader.main_program(world_size, arch, capacity)
            for k_l in (None, K_L):
                assert loader.dispatch_programs(
                    world_size, arch, capacity, _n(arch), k_l, sm_count=132
                ) == (main,)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _fake_group(world_size, rank, name="g"):
    return SimpleNamespace(group_name=name, _world_size=world_size, _rank=rank)


def _patch_distributed(
    monkeypatch,
    *,
    rank=0,
    backend_name="nccl",
    capability=None,
    symm_backend="CUDA",
    sm_count=148,
):
    capability = CAPABILITY[ARCHES[0]] if capability is None else capability
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: backend_name)
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: group._world_size)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: group._rank)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: symm_backend)
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(
            capability=capability,
            arch=loader.CAPABILITY_ARCH.get(capability),
            sm_count=sm_count,
        ),
    )


class _CudaLike:
    """Tensor metadata stand-in: shape, dtype, contiguity and a CUDA device without a GPU."""

    def __init__(self, shape, dtype, *, contiguous=True, index=0, ptr=0x10000):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", index)
        self._contiguous = contiguous
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def is_contiguous(self):
        return self._contiguous

    def view(self, dtype):
        return ("view", dtype, self)


def _operands(m, n=3072, k_l=K_L, *, index=0):
    return (
        _CudaLike((m, k_l), FP8, index=index),
        _CudaLike((m,), torch.float32, index=index),
        _CudaLike((n, k_l), FP8, index=index),
        _CudaLike((n,), torch.float32, index=index),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
@pytest.mark.parametrize("rows", [1, 37, 1024, 4133])
@pytest.mark.parametrize("k_l", [128, 768, 1024])
def test_validation_admits_any_rows_and_the_tile_multiples(
    monkeypatch, arch, world_size, rows, k_l
):
    _patch_distributed(monkeypatch, rank=world_size - 1, capability=CAPABILITY[arch])
    n = _n(arch)
    call = backend._validate(
        *_operands(rows, n, k_l), _fake_group(world_size, world_size - 1)
    )
    assert (call.m, call.n, call.k_l) == (rows, n, k_l)
    assert (call.rank, call.world_size, call.arch) == (world_size - 1, world_size, arch)
    assert call.comm_mode == _mode(arch, world_size)
    assert (call.device_index, call.group_name) == (0, "g")


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED[:1])
def test_validation_rejects_dtype_rank_and_shape_mismatches(
    monkeypatch, arch, world_size
):
    _patch_distributed(monkeypatch, capability=CAPABILITY[arch])
    group = _fake_group(world_size, 0)
    n = _n(arch)
    a, sa, w, sw = _operands(256, n)
    with pytest.raises(ValueError, match="a must have dtype"):
        backend._validate(_CudaLike((256, K_L), torch.bfloat16), sa, w, sw, group)
    with pytest.raises(ValueError, match="w must have dtype"):
        backend._validate(a, sa, _CudaLike((n, K_L), torch.int8), sw, group)
    with pytest.raises(ValueError, match="a_scale must have dtype"):
        backend._validate(a, _CudaLike((256,), torch.bfloat16), w, sw, group)
    with pytest.raises(ValueError, match="w_scale must be 1-D"):
        backend._validate(a, sa, w, _CudaLike((n, 1), torch.float32), group)
    with pytest.raises(ValueError, match=r"w must be \[N, K_l\]"):
        backend._validate(a, sa, _CudaLike((n, 2 * K_L), FP8), sw, group)
    with pytest.raises(ValueError, match=r"a_scale must be \[M\]"):
        backend._validate(a, _CudaLike((255,), torch.float32), w, sw, group)
    with pytest.raises(ValueError, match=r"w_scale must be \[N\]"):
        backend._validate(a, sa, w, _CudaLike((n + 1,), torch.float32), group)
    with pytest.raises(ValueError, match="multiple of"):
        backend._validate(*_operands(256, 1000), group)
    with pytest.raises(ValueError, match="multiple of 128"):
        backend._validate(*_operands(256, n, 700), group)
    with pytest.raises(ValueError, match=r"2\*\*30"):
        backend._validate(*_operands((1 << 20) + 1, 1024), group)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED[:1])
def test_validation_rejects_strided_and_foreign_device_operands(
    monkeypatch, arch, world_size
):
    _patch_distributed(monkeypatch, capability=CAPABILITY[arch])
    group = _fake_group(world_size, 0)
    n = _n(arch)
    a, sa, w, sw = _operands(256, n)
    with pytest.raises(ValueError, match="a must be contiguous"):
        backend._validate(
            _CudaLike((256, K_L), FP8, contiguous=False), sa, w, sw, group
        )
    with pytest.raises(ValueError, match="w must be contiguous"):
        backend._validate(a, sa, _CudaLike((n, K_L), FP8, contiguous=False), sw, group)
    with pytest.raises(ValueError, match="share one CUDA device"):
        backend._validate(a, sa, _CudaLike((n, K_L), FP8, index=1), sw, group)
    cpu = SimpleNamespace(
        shape=(256, K_L),
        ndim=2,
        dtype=FP8,
        device=torch.device("cpu"),
        is_contiguous=lambda: True,
    )
    with pytest.raises(ValueError, match="CUDA tensor"):
        backend._validate(cpu, sa, w, sw, group)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_validation_rejects_misaligned_vector_and_tma_operands(
    monkeypatch, arch, world_size
):
    _patch_distributed(monkeypatch, capability=CAPABILITY[arch])
    group = _fake_group(world_size, 0)
    n = _n(arch)
    a, sa, w, sw = _operands(256, n)
    # A 4-byte offset (storage[1:1 + N]) breaks every vector load; an 8-byte
    # offset still breaks the 16-byte loads.
    for ptr in (0x10004, 0x10008):
        with pytest.raises(ValueError, match="w_scale must start at a 16-byte aligned"):
            backend._validate(a, sa, w, _CudaLike((n,), torch.float32, ptr=ptr), group)
    with pytest.raises(ValueError, match="a must start at a 16-byte aligned"):
        backend._validate(_CudaLike((256, K_L), FP8, ptr=0x10001), sa, w, sw, group)
    with pytest.raises(ValueError, match="w must start at a 16-byte aligned"):
        backend._validate(a, sa, _CudaLike((n, K_L), FP8, ptr=0x10001), sw, group)
    # a_scale is read with scalar loads only: a 4-byte offset stays admitted.
    backend._validate(a, _CudaLike((256,), torch.float32, ptr=0x10004), w, sw, group)
    call = _call(256, arch=arch, world_size=world_size)
    with pytest.raises(ValueError, match="out must start at a 16-byte aligned"):
        backend._output(_CudaLike((256, n), torch.bfloat16, ptr=0x10002), call, a)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_validation_rejects_other_groups_world_sizes_and_devices(monkeypatch, arch):
    world_size = sorted(loader.COMM_MODES[arch])[0]
    n = _n(arch)
    _patch_distributed(monkeypatch, backend_name="gloo", capability=CAPABILITY[arch])
    with pytest.raises(ValueError, match="NCCL"):
        backend._validate(*_operands(256, n), _fake_group(world_size, 0))
    _patch_distributed(monkeypatch, capability=CAPABILITY[arch])
    for other in UNDELIVERED_WORLD_SIZES:
        with pytest.raises(ValueError, match="world size"):
            backend._validate(*_operands(256, n), _fake_group(other, 0))
    _patch_distributed(monkeypatch, capability=(8, 9))
    with pytest.raises(ValueError, match="delivered for .*got SM89"):
        backend._validate(*_operands(256, n), _fake_group(world_size, 0))
    undelivered = [
        capability
        for capability in ((9, 0), (10, 0), (10, 3))
        if loader.CAPABILITY_ARCH.get(capability) is None
    ]
    for capability in undelivered:
        _patch_distributed(monkeypatch, capability=capability)
        with pytest.raises(ValueError, match="delivered for"):
            backend._validate(*_operands(256, n), _fake_group(world_size, 0))


@pytest.mark.cpu_only
@pytest.mark.parametrize("symm_backend", ["CUDA", "NVSHMEM", "cuda", "nvshmem"])
def test_validation_admits_the_cuda_and_nvshmem_symmetric_memory_backends(
    monkeypatch, symm_backend
):
    arch, world_size = DELIVERED[0]
    _patch_distributed(
        monkeypatch, symm_backend=symm_backend, capability=CAPABILITY[arch]
    )
    device = torch.device("cuda", 0)
    assert backend.symmetric_memory_backend(device) == symm_backend.upper()
    assert backend.supports_symmetric_memory_backend(device)
    backend._validate(*_operands(256, _n(arch)), _fake_group(world_size, 0))


@pytest.mark.cpu_only
@pytest.mark.parametrize("symm_backend", ["NCCL", None, "XPU"])
def test_validation_rejects_other_symmetric_memory_backends_without_selecting_one(
    monkeypatch, symm_backend
):
    arch, world_size = DELIVERED[0]
    _patch_distributed(
        monkeypatch, symm_backend=symm_backend, capability=CAPABILITY[arch]
    )
    selected = []
    monkeypatch.setattr(backend.symm_mem, "set_backend", selected.append)
    device = torch.device("cuda", 0)
    assert not backend.supports_symmetric_memory_backend(device)
    with pytest.raises(ValueError, match="symmetric-memory backends"):
        backend._validate(*_operands(256, _n(arch)), _fake_group(world_size, 0))
    assert selected == []


@pytest.mark.cpu_only
def test_group_name_is_required():
    with pytest.raises(ValueError, match="group_name"):
        backend._group_name(SimpleNamespace())


def _call(m=256, n=None, k_l=K_L, *, arch=None, world_size=None, rank=1, index=0):
    arch = ARCHES[0] if arch is None else arch
    world_size = (
        sorted(loader.COMM_MODES[arch])[0] if world_size is None else world_size
    )
    n = _n(arch) if n is None else n
    return backend._Call(
        index, rank, world_size, "g", arch, _mode(arch, world_size), m, n, k_l
    )


@pytest.mark.cpu_only
def test_out_must_match_the_result_contract():
    call = _call(250)
    n = call.n
    like = _CudaLike((250, K_L), FP8)
    with pytest.raises(ValueError, match="bfloat16"):
        backend._output(_CudaLike((250, n), torch.float16), call, like)
    with pytest.raises(ValueError, match=rf"\[250, {n}\]"):
        backend._output(_CudaLike((256, n), torch.bfloat16), call, like)
    with pytest.raises(ValueError, match=rf"\[250, {n}\]"):
        backend._output(
            _CudaLike((250, n), torch.bfloat16, contiguous=False), call, like
        )
    with pytest.raises(ValueError, match="device"):
        backend._output(_CudaLike((250, n), torch.bfloat16, index=1), call, like)
    out = _CudaLike((250, n), torch.bfloat16)
    assert backend._output(out, call, like) is out


# --------------------------------------------------------------------------- #
# Entry points and dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_backend_entrypoints_reject_non_cake_routes():
    arch, world_size = DELIVERED[0]
    with pytest.raises(ValueError, match="exactly 'cake'"):
        backend.fp8_rowwise_gemm_allreduce_cake(
            *_operands(256, _n(arch)), _fake_group(world_size, 0), backend="auto"
        )
    with pytest.raises(ValueError, match="exactly 'cake'"):
        backend.prepare_fp8_rowwise_gemm_allreduce_cake(
            *_operands(256, _n(arch)), _fake_group(world_size, 0), backend="auto"
        )


@pytest.mark.cpu_only
def test_public_entrypoint_routes_auto_and_cake_to_the_backend(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    calls = []
    result = object()

    def fake_backend(a, a_scale, w, w_scale, group, *, backend, out):
        calls.append((a, a_scale, w, w_scale, group, backend, out))
        return result

    monkeypatch.setattr(backend, "fp8_rowwise_gemm_allreduce_cake", fake_backend)
    a, sa, w, sw, group, out = (object() for _ in range(6))
    assert dispatcher.fp8_rowwise_gemm_allreduce(a, sa, w, sw, group) is result
    assert (
        dispatcher.fp8_rowwise_gemm_allreduce(
            a, sa, w, sw, group, backend="cake", out=out
        )
        is result
    )
    assert calls == [
        (a, sa, w, sw, group, "cake", None),
        (a, sa, w, sw, group, "cake", out),
    ]
    with pytest.raises(ValueError, match="'auto' or 'cake'"):
        dispatcher.fp8_rowwise_gemm_allreduce(a, sa, w, sw, group, backend="trtllm")


@pytest.mark.cpu_only
def test_public_prepare_forwards_the_row_capacity(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    calls = []
    launcher = object()

    def fake_prepare(a, a_scale, w, w_scale, group, *, backend, max_rows):
        calls.append((a, a_scale, w, w_scale, group, backend, max_rows))
        return launcher

    monkeypatch.setattr(
        backend, "prepare_fp8_rowwise_gemm_allreduce_cake", fake_prepare
    )
    a, sa, w, sw, group = (object() for _ in range(5))
    assert (
        dispatcher.prepare_fp8_rowwise_gemm_allreduce(a, sa, w, sw, group) is launcher
    )
    assert (
        dispatcher.prepare_fp8_rowwise_gemm_allreduce(
            a, sa, w, sw, group, backend="cake", max_rows=4133
        )
        is launcher
    )
    assert calls == [
        (a, sa, w, sw, group, "cake", None),
        (a, sa, w, sw, group, "cake", 4133),
    ]
    with pytest.raises(ValueError, match="'auto' or 'cake'"):
        dispatcher.prepare_fp8_rowwise_gemm_allreduce(a, sa, w, sw, group, backend="x")


@pytest.mark.cpu_only
def test_package_exports_the_public_names():
    package = importlib.import_module("flashinfer.comm.fp8_rowwise_gemm_allreduce")
    comm = importlib.import_module("flashinfer.comm")
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    assert package.fp8_rowwise_gemm_allreduce is dispatcher.fp8_rowwise_gemm_allreduce
    assert comm.fp8_rowwise_gemm_allreduce is dispatcher.fp8_rowwise_gemm_allreduce
    assert (
        comm.prepare_fp8_rowwise_gemm_allreduce
        is dispatcher.prepare_fp8_rowwise_gemm_allreduce
    )


@pytest.mark.cpu_only
def test_public_entrypoints_carry_the_flashinfer_api_decorator():
    dispatcher = importlib.import_module(
        "flashinfer.comm.fp8_rowwise_gemm_allreduce.fp8_rowwise_gemm_allreduce"
    )
    decorated = set()
    for node in ast.parse(inspect.getsource(dispatcher)).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = set()
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            names.add(getattr(target, "id", getattr(target, "attr", None)))
        if "flashinfer_api" in names:
            decorated.add(node.name)
    assert decorated == {
        "fp8_rowwise_gemm_allreduce",
        "prepare_fp8_rowwise_gemm_allreduce",
    }


# --------------------------------------------------------------------------- #
# Symmetric memory: rendezvous guard, multicast view and workspace allocation
# --------------------------------------------------------------------------- #


class _FakeHandle:
    """Symmetric-memory handle stand-in: topology, sizes, CPU peer buffers / pads, multicast address."""

    def __init__(
        self, *, rank, world_size, buffer_size, signal_pad_size=0, multicast_ptr=None
    ):
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.signal_pad_size = signal_pad_size
        self.pads = {}
        if multicast_ptr is not None:
            self.multicast_ptr = multicast_ptr

    def get_buffer(self, peer, shape, dtype, offset):
        assert offset == 0
        return torch.empty(shape, dtype=dtype)

    def get_signal_pad(self, peer, shape, dtype, offset):
        assert offset == 0 and dtype == torch.uint32
        if peer not in self.pads:
            self.pads[peer] = torch.full(shape, 7, dtype=dtype)
        return self.pads[peer]


@pytest.mark.cpu_only
def test_rendezvous_guard_checks_topology_and_size(monkeypatch):
    tensor = torch.empty(64, dtype=torch.uint8)
    handles = []
    monkeypatch.setattr(backend.symm_mem, "rendezvous", lambda t, group: handles.pop(0))
    call = _call()
    good = _FakeHandle(rank=1, world_size=call.world_size, buffer_size=64)
    handles.append(good)
    assert backend._rendezvous(call, tensor, "receive workspace") is good
    handles.append(_FakeHandle(rank=0, world_size=call.world_size, buffer_size=64))
    with pytest.raises(RuntimeError, match="topology"):
        backend._rendezvous(call, tensor, "receive workspace")
    handles.append(_FakeHandle(rank=1, world_size=call.world_size, buffer_size=63))
    with pytest.raises(backend.StaleSymmetricHandleError, match="pytorch#192579"):
        backend._rendezvous(call, tensor, "receive workspace")
    assert issubclass(backend.StaleSymmetricHandleError, RuntimeError)


@pytest.mark.cpu_only
def test_multicast_view_describes_the_multicast_address_as_uint32():
    view = backend._MulticastView(0x7F00_0000_1000, 4096)
    interface = view.__cuda_array_interface__
    assert interface["shape"] == (1024,) and interface["typestr"] == "<u4"
    assert interface["data"] == (0x7F00_0000_1000, False)
    assert interface["strides"] is None and interface["version"] == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", [None, 0, -1, False, "0x10", 2.0])
def test_multicast_tensor_refuses_a_handle_without_a_positive_multicast_address(value):
    handle = _FakeHandle(rank=0, world_size=2, buffer_size=4096)
    if value is not None:
        handle.multicast_ptr = value
    with pytest.raises(backend.MulticastUnavailableError, match="multicast_ptr"):
        backend._multicast_tensor(handle, 4096, torch.device("cuda", 0))
    assert issubclass(backend.MulticastUnavailableError, RuntimeError)


@pytest.mark.cpu_only
def test_multicast_tensor_aliases_the_address_through_the_array_interface(monkeypatch):
    handle = _FakeHandle(
        rank=0, world_size=2, buffer_size=4096, multicast_ptr=4096 * 17
    )
    seen = []

    def as_tensor(view, device):
        seen.append((view.__cuda_array_interface__, device))
        return SimpleNamespace(
            dtype=torch.uint32, data_ptr=lambda: 4096 * 17, numel=lambda: 1024
        )

    monkeypatch.setattr(backend.torch, "as_tensor", as_tensor)
    view, pointer = backend._multicast_tensor(handle, 4096, torch.device("cuda", 0))
    assert pointer == 4096 * 17 and view.data_ptr() == pointer
    assert seen[0][0]["data"] == (4096 * 17, False) and seen[0][1] == torch.device(
        "cuda", 0
    )
    # A ctypes-like address object is unwrapped through ``.value``.
    handle.multicast_ptr = SimpleNamespace(value=4096 * 17)
    assert (
        backend._multicast_tensor(handle, 4096, torch.device("cuda", 0))[1] == 4096 * 17
    )


def _patch_allocation(
    monkeypatch, *, handles, max_clusters=74, sm_count=148, loaded=None
):
    """CPU stand-ins for every device-side step of ``_allocate_workspace``."""

    events = []
    allocated = []

    def empty(numel, *, dtype, device):
        tensor = torch.empty(numel, dtype=dtype)
        allocated.append((numel, dtype, device))
        return tensor

    def rendezvous(call, tensor, what):
        events.append(("rendezvous", what))
        return handles.pop(0)

    def multicast_tensor(handle, nbytes, device):
        events.append(("multicast", nbytes))
        return ("multicast", handle.multicast_ptr), int(handle.multicast_ptr)

    monkeypatch.setattr(backend.symm_mem, "empty", empty)
    monkeypatch.setattr(backend, "_rendezvous", rendezvous)
    monkeypatch.setattr(backend, "_multicast_tensor", multicast_tensor)
    monkeypatch.setattr(
        backend.torch.cuda, "device", lambda index: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(
        backend.torch.cuda, "synchronize", lambda device: events.append(("sync",))
    )
    monkeypatch.setattr(
        backend.dist, "barrier", lambda group: events.append(("barrier", group))
    )
    monkeypatch.setattr(
        loader,
        "load",
        lambda program, arch: events.append(("load", program, arch)) or loaded,
    )
    monkeypatch.setattr(
        loader,
        "cluster_capacity",
        lambda index, program: events.append(("capacity", index, program))
        or max_clusters,
    )
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(
            capability=(0, 0), arch=None, sm_count=sm_count
        ),
    )
    return events, allocated


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_allocation_places_the_readiness_words_in_the_signal_pad_when_they_fit(
    monkeypatch, arch, world_size
):
    n = _n(arch)
    mode = _mode(arch, world_size)
    s_bytes = loader.workspace_bytes(1024, n, world_size, arch)
    handle = _FakeHandle(
        rank=1,
        world_size=world_size,
        buffer_size=s_bytes,
        signal_pad_size=65536,
        multicast_ptr=0x1000 if mode == "nvls" else None,
    )
    module = object()
    events, allocated = _patch_allocation(monkeypatch, handles=[handle], loaded=module)
    group = _fake_group(world_size, 1)
    call = _call(1024, arch=arch, world_size=world_size)
    workspace = backend._allocate_workspace(call, group, max_rows=1024)
    program = loader.main_program(world_size, arch, 1024)
    cluster_grid = loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS
    # Without a bound K_l the one-shot workspace loads every program of its route.
    programs = loader.dispatch_programs(
        world_size, arch, 1024, n, sm_count=0 if cluster_grid else 148
    )
    assert programs[0] == program
    expected = [("load", name, arch) for name in programs]
    expected.append(("rendezvous", "receive workspace"))
    if mode == "nvls":
        expected.append(("multicast", s_bytes))
    if cluster_grid:
        expected.append(("capacity", 0, program))
    expected += [("sync",), ("barrier", group)]
    assert events == expected
    assert allocated == [(s_bytes, torch.uint8, torch.device("cuda", 0))]
    assert workspace.module is module and workspace.program == program
    assert workspace.modules == {name: module for name in programs}
    assert (workspace.rank, workspace.world_size, workspace.n) == (1, world_size, n)
    assert (workspace.arch, workspace.comm_mode) == (arch, mode)
    assert workspace.transport == loader.transport(world_size, arch)
    assert (workspace.max_rows, workspace.rows_capacity) == (1024, 1024)
    assert workspace.chunk_capacity == 8
    assert workspace.slots == loader.slots_per_chunk(world_size, mode)
    assert workspace.grid_n == loader.grid_n(n, arch)
    assert workspace.s_bytes == s_bytes == workspace.s.numel()
    assert workspace.flags_in_signal_pad and workspace.flags_buffer is None
    assert workspace.flags_handle is None and workspace.signal_pad_size == 65536
    # This rank's pad words are zeroed; the peers' pads are not touched.
    assert workspace.flags is handle.pads[1]
    assert workspace.flags.numel() == 16 and int(workspace.flags.sum()) == 0
    assert all(
        int(handle.pads[peer].sum()) == 7 * 16
        for peer in range(world_size)
        if peer != 1
    )
    assert len(workspace.s_peers) == world_size
    assert len(workspace.flags_peers) == world_size
    assert workspace.ctl.dtype == torch.uint32
    assert workspace.ctl.numel() == loader.ctl_words(8, arch)
    assert int(workspace.ctl.sum()) == 0
    if mode == "nvls":
        assert workspace.multicast == ("multicast", 0x1000)
        assert workspace.multicast_ptr == 0x1000
    else:
        assert workspace.multicast is None and workspace.multicast_ptr == 0
    if cluster_grid:
        assert (workspace.max_clusters, workspace.sm_count) == (74, 0)
    else:
        assert (workspace.max_clusters, workspace.sm_count) == (0, 148)
    assert workspace.captured is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED[:1])
def test_allocation_uses_a_dedicated_symmetric_buffer_when_the_pad_is_too_small(
    monkeypatch, arch, world_size
):
    n = _n(arch)
    mode = _mode(arch, world_size)
    s_bytes = loader.workspace_bytes(8192, n, world_size, arch)
    words = loader.flag_words(8192)
    s_handle = _FakeHandle(
        rank=0,
        world_size=world_size,
        buffer_size=s_bytes,
        signal_pad_size=256,
        multicast_ptr=0x2000 if mode == "nvls" else None,
    )
    flag_handle = _FakeHandle(rank=0, world_size=world_size, buffer_size=4 * words)
    events, allocated = _patch_allocation(monkeypatch, handles=[s_handle, flag_handle])
    group = _fake_group(world_size, 0)
    call = _call(8192, arch=arch, world_size=world_size, rank=0)
    workspace = backend._allocate_workspace(call, group, max_rows=8192)
    assert [event for event in events if event[0] == "rendezvous"] == [
        ("rendezvous", "receive workspace"),
        ("rendezvous", "readiness words"),
    ]
    assert allocated == [
        (s_bytes, torch.uint8, torch.device("cuda", 0)),
        (words, torch.uint32, torch.device("cuda", 0)),
    ]
    assert not workspace.flags_in_signal_pad
    assert workspace.flags is workspace.flags_buffer
    assert workspace.s_handle is s_handle and workspace.flags_handle is flag_handle
    assert workspace.flags.numel() == words == 128 and int(workspace.flags.sum()) == 0
    assert workspace.ctl.numel() == loader.ctl_words(64, arch)
    assert events[-2:] == [("sync",), ("barrier", group)]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED[:1])
def test_allocation_refuses_inside_capture_and_an_oversubscribed_grid(
    monkeypatch, arch, world_size
):
    n = _n(arch)
    mode = _mode(arch, world_size)
    s_bytes = loader.workspace_bytes(128, n, world_size, arch)
    handle = _FakeHandle(
        rank=1,
        world_size=world_size,
        buffer_size=s_bytes,
        signal_pad_size=65536,
        multicast_ptr=0x1000 if mode == "nvls" else None,
    )
    events, _ = _patch_allocation(
        monkeypatch, handles=[handle], max_clusters=513, sm_count=1025
    )
    call = _call(128, arch=arch, world_size=world_size)
    with pytest.raises(RuntimeError, match="per-CTA arrival words"):
        backend._allocate_workspace(call, _fake_group(world_size, 1), max_rows=128)
    assert ("barrier", _fake_group(world_size, 1)) not in events
    monkeypatch.setattr(backend.torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA-graph capture"):
        backend._allocate_workspace(call, _fake_group(world_size, 1), max_rows=128)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "arch, world_size", [item for item in DELIVERED if _mode(*item) == "nvls"]
)
def test_allocation_on_the_nvls_carrier_requires_the_multicast_address(
    monkeypatch, arch, world_size
):
    n = _n(arch)
    s_bytes = loader.workspace_bytes(128, n, world_size, arch)
    handle = _FakeHandle(
        rank=1, world_size=world_size, buffer_size=s_bytes, signal_pad_size=65536
    )
    original = backend._multicast_tensor
    events, _ = _patch_allocation(monkeypatch, handles=[handle])
    monkeypatch.setattr(backend, "_multicast_tensor", original)
    call = _call(128, arch=arch, world_size=world_size)
    with pytest.raises(backend.MulticastUnavailableError, match="multicast_ptr=None"):
        backend._allocate_workspace(call, _fake_group(world_size, 1), max_rows=128)
    assert ("barrier", _fake_group(world_size, 1)) not in events


# --------------------------------------------------------------------------- #
# One-shot workspace cache, prepared launcher, launch
# --------------------------------------------------------------------------- #


def _workspace(
    *,
    max_rows,
    arch=None,
    world_size=None,
    n=None,
    rank=1,
    max_clusters=74,
    sm_count=148,
    program=None,
    module=None,
    modules=None,
    multicast=None,
):
    arch = ARCHES[0] if arch is None else arch
    world_size = (
        sorted(loader.COMM_MODES[arch])[0] if world_size is None else world_size
    )
    n = _n(arch) if n is None else n
    mode = _mode(arch, world_size)
    c_cap = loader.chunk_capacity(max_rows)
    cluster_grid = loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS
    if mode == "nvls" and multicast is None:
        multicast = "S_mc"
    program = program or loader.main_program(world_size, arch, max_rows)
    if modules is None:
        # Every program of the route (its lever programs included) runs ``module``.
        modules = {
            name: module
            for name in (
                program,
                *loader.dispatch_programs(
                    world_size, arch, max_rows, n, sm_count=sm_count
                ),
            )
        }
    return backend._Workspace(
        rank=rank,
        world_size=world_size,
        device_index=0,
        arch=arch,
        comm_mode=mode,
        transport=loader.transport(world_size, arch),
        n=n,
        program=program,
        module=module,
        modules=modules,
        max_rows=max_rows,
        rows_capacity=loader.ceil128(max_rows),
        chunk_capacity=c_cap,
        slots=loader.slots_per_chunk(world_size, mode),
        grid_n=loader.grid_n(n, arch),
        s_bytes=loader.workspace_bytes(max_rows, n, world_size, arch),
        s="S",
        s_handle=None,
        s_peers="S_peers",
        flags="flags",
        flags_peers="flags_peers",
        flags_in_signal_pad=True,
        signal_pad_size=65536,
        flags_buffer=None,
        flags_handle=None,
        ctl="ctl",
        multicast=multicast,
        multicast_ptr=0x1000 if multicast is not None else 0,
        max_clusters=max_clusters if cluster_grid else 0,
        sm_count=0 if cluster_grid else sm_count,
    )


@pytest.mark.cpu_only
def test_one_shot_cache_serves_a_fitting_workspace_without_a_collective(monkeypatch):
    call = _call(1000)
    group = _fake_group(call.world_size, 1)
    key = (0, id(group), "g", call.n)
    state = backend._LaunchState(rank=1, world_size=call.world_size)
    workspace = _workspace(max_rows=1024)
    monkeypatch.setitem(backend._WORKSPACES, key, workspace)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setitem(backend._LAUNCH_STATES, (0, id(group), "g"), state)

    def no_collective(*args, **kwargs):
        raise AssertionError("no allocation expected")

    monkeypatch.setattr(backend, "_allocate_workspace", no_collective)
    assert backend._prepare_one_shot(call, group) == (state, workspace)
    assert backend._workspace_of(group, 0, call.n) is workspace
    with pytest.raises(
        RuntimeError, match="no fused FP8 GEMM \\+ all-reduce workspace"
    ):
        backend._workspace_of(group, 0, 2 * call.n)


@pytest.mark.cpu_only
def test_one_shot_cache_refuses_to_grow_a_captured_workspace_before_any_collective(
    monkeypatch,
):
    call = _call(129)
    group = _fake_group(call.world_size, 1)
    key = (0, id(group), "g", call.n)
    state = backend._LaunchState(rank=1, world_size=call.world_size)
    workspace = _workspace(max_rows=128)
    workspace.captured = True
    monkeypatch.setitem(backend._WORKSPACES, key, workspace)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setitem(backend._LAUNCH_STATES, (0, id(group), "g"), state)

    def no_collective(*args, **kwargs):
        raise AssertionError("no allocation expected")

    monkeypatch.setattr(backend, "_allocate_workspace", no_collective)
    with pytest.raises(RuntimeError, match="captured into a CUDA graph"):
        backend._prepare_one_shot(call, group)
    # A refused growth is a caller error; the launch state stays usable.
    assert state.poisoned is False
    assert backend._prepare_one_shot(_call(128), group) == (state, workspace)


@pytest.mark.cpu_only
def test_one_shot_growth_retires_the_previous_workspace_and_poisons_on_failure(
    monkeypatch,
):
    call = _call(200)
    group = _fake_group(call.world_size, 1)
    key = (0, id(group), "g", call.n)
    state = backend._LaunchState(rank=1, world_size=call.world_size)
    old = _workspace(max_rows=128)
    grown = _workspace(max_rows=256)
    monkeypatch.setitem(backend._WORKSPACES, key, old)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setitem(backend._LAUNCH_STATES, (0, id(group), "g"), state)
    monkeypatch.setattr(backend, "_RETIRED", [])
    synced = []
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda index: SimpleNamespace(synchronize=lambda: synced.append(index)),
    )
    requests = []

    def allocate(call, g, *, max_rows):
        requests.append((call.m, g, max_rows))
        if max_rows > 256:
            raise RuntimeError("peer went away")
        return grown

    monkeypatch.setattr(backend, "_allocate_workspace", allocate)
    assert backend._prepare_one_shot(call, group) == (state, grown)
    assert requests == [(200, group, 256)] and synced == [0]
    assert backend._WORKSPACES[key] is grown and list(backend._RETIRED) == [old]
    with pytest.raises(RuntimeError, match="peer went away"):
        backend._prepare_one_shot(_call(300), group)
    assert state.poisoned and "peer went away" in state.poison_reason
    with pytest.raises(RuntimeError, match="poisoned"):
        backend._prepare_one_shot(_call(1), group)
    # The grown generation is still published; the failed one never was.
    assert backend._WORKSPACES[key] is grown


@pytest.mark.cpu_only
def test_launch_state_is_cached_per_device_and_group(monkeypatch):
    call = _call(rank=2)
    group = _fake_group(call.world_size, 2)
    monkeypatch.setattr(backend, "_LAUNCH_STATES", {})
    first = backend._launch_state(call, group)
    assert backend._launch_state(call, group) is first
    assert (first.rank, first.world_size, first.calls) == (2, call.world_size, 0)
    with pytest.raises(RuntimeError, match="topology changed"):
        backend._launch_state(_call(rank=3), group)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_prepared_launcher_serves_every_row_count_up_to_its_capacity(
    monkeypatch, arch, world_size
):
    rank = world_size - 1
    _patch_distributed(monkeypatch, rank=rank, capability=CAPABILITY[arch])
    group = _fake_group(world_size, rank)
    n = _n(arch)
    sample = _operands(1, n)
    call = backend._validate(*sample, group)
    launcher = backend._PreparedLauncher(
        group=group,
        group_id=id(group),
        call=call,
        max_rows=4133,
        device=sample[0].device,
        state=backend._LaunchState(rank=rank, world_size=world_size),
        workspace=_workspace(
            max_rows=4133, arch=arch, world_size=world_size, rank=rank
        ),
    )
    assert (launcher.n, launcher.k_l, launcher.max_rows) == (n, K_L, 4133)
    assert (launcher.arch, launcher.comm_mode) == (arch, _mode(arch, world_size))
    assert launcher.workspace_bytes == loader.workspace_bytes(4133, n, world_size, arch)
    assert launcher.flags_in_signal_pad
    assert launcher.program == loader.main_program(world_size, arch, 4133)
    assert launcher.programs[0] == launcher.program
    assert set(launcher.programs) == set(launcher.workspace.modules)
    for rows in (1, 129, 4133):
        bound = launcher._validate_call(*_operands(rows, n))
        assert (bound.m, bound.n, bound.k_l) == (rows, n, K_L)
    with pytest.raises(ValueError, match=r"\[capacity_exceeded\] M=4134"):
        launcher._validate_call(*_operands(4134, n))
    with pytest.raises(ValueError, match=r"\[N_mismatch\]"):
        launcher._validate_call(*_operands(1, 2 * n))
    with pytest.raises(ValueError, match=r"\[K_l_mismatch\]"):
        launcher._validate_call(*_operands(1, n, 2 * K_L))
    with pytest.raises(ValueError, match="contiguous"):
        a, sa, w, sw = _operands(1, n)
        launcher._validate_call(_CudaLike((1, K_L), FP8, contiguous=False), sa, w, sw)
    with pytest.raises(ValueError, match="device changed"):
        launcher._validate_call(*_operands(1, n, index=1))


@pytest.mark.cpu_only
def test_prepare_requires_the_capacity_to_cover_the_sample(monkeypatch):
    arch, world_size = DELIVERED[0]
    _patch_distributed(monkeypatch, capability=CAPABILITY[arch])
    with pytest.raises(ValueError, match="max_rows=100 is smaller"):
        backend.prepare_fp8_rowwise_gemm_allreduce_cake(
            *_operands(256, _n(arch)),
            _fake_group(world_size, 0),
            backend="cake",
            max_rows=100,
        )


@pytest.mark.cpu_only
def test_capacity_agreement_names_every_rank_that_differs(monkeypatch):
    table = {0: 4133, 1: 4133, 2: 4133, 3: 4160}

    def fake_all_gather(outputs, value, group):
        assert group.group_name == "g" and value.dtype == torch.int64
        for rank, output in enumerate(outputs):
            output.copy_(torch.tensor([table[rank]], dtype=torch.int64))

    monkeypatch.setattr(backend.dist, "all_gather", fake_all_gather)
    cpu = torch.device("cpu")
    with pytest.raises(ValueError, match="rank 2: 4133, rank 3: 4160"):
        backend._agree_capacity(_fake_group(4, 2), 4133, world_size=4, device=cpu)
    table[3] = 4133
    backend._agree_capacity(_fake_group(4, 2), 4133, world_size=4, device=cpu)


@pytest.mark.cpu_only
def test_prepare_agrees_the_capacity_before_any_allocation(monkeypatch):
    arch, world_size = DELIVERED[0]
    _patch_distributed(monkeypatch, capability=CAPABILITY[arch])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(backend, "_LAUNCH_STATES", {})
    agreed = []

    def disagree(group, capacity, *, world_size, device):
        agreed.append((capacity, world_size, device))
        raise ValueError("max_rows must be identical on every rank of the group")

    monkeypatch.setattr(backend, "_agree_capacity", disagree)
    monkeypatch.setattr(
        backend, "_allocate_workspace", lambda *args, **kwargs: pytest.fail("allocated")
    )
    with pytest.raises(ValueError, match="identical on every rank"):
        backend.prepare_fp8_rowwise_gemm_allreduce_cake(
            *_operands(256, _n(arch)),
            _fake_group(world_size, 0),
            backend="cake",
            max_rows=4133,
        )
    assert agreed == [(4133, world_size, torch.device("cuda", 0))]
    assert backend._LAUNCH_STATES == {}


@pytest.mark.cpu_only
def test_launch_refuses_a_workspace_smaller_than_the_call_without_poisoning():
    call = _call(129)
    state = backend._LaunchState(rank=1, world_size=call.world_size)
    with pytest.raises(RuntimeError, match="holds 128 rows, the call needs 256"):
        backend._launch(state, _workspace(max_rows=128), call, *([None] * 5))
    with pytest.raises(RuntimeError, match="another N"):
        backend._launch(
            state, _workspace(max_rows=128), _call(1, 2 * call.n), *([None] * 5)
        )
    state.calls = backend.EPOCH_MAX_CALLS
    with pytest.raises(RuntimeError, match="epoch exhausted"):
        backend._launch(state, _workspace(max_rows=128), _call(1), *([None] * 5))
    assert state.poisoned is False
    assert backend.EPOCH_MAX_CALLS == 2**32 - 2


class _Module:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def run(self, *args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error


def _launch_fixture(
    monkeypatch, *, arch=None, world_size=None, capturing=False, error=None
):
    module = _Module(error)
    workspace = _workspace(
        max_rows=1024, arch=arch, world_size=world_size, module=module
    )
    state = backend._LaunchState(rank=1, world_size=workspace.world_size)
    tails = []
    monkeypatch.setattr(backend, "_current_stream_handle", lambda index: 4242)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: capturing
    )
    monkeypatch.setattr(
        backend,
        "_remember_tail",
        lambda state, handle, index: tails.append((handle, index)),
    )
    monkeypatch.setattr(backend.tvm_ffi, "use_torch_stream", contextlib.nullcontext)
    return state, workspace, module, tails


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_launch_binds_every_argument_by_name_in_the_recorded_order(
    monkeypatch, arch, world_size
):
    state, workspace, module, tails = _launch_fixture(
        monkeypatch, arch=arch, world_size=world_size
    )
    n = workspace.n
    a, sa, w, sw = _operands(1000, n)
    out = _CudaLike((1000, n), torch.bfloat16)
    call = _call(1000, arch=arch, world_size=world_size)
    backend._launch(state, workspace, call, a, sa, w, sw, out)
    names = loader.launcher_args(workspace.program)
    assert len(module.calls) == 1 and len(module.calls[0]) == len(names)
    bound = dict(zip(names, module.calls[0], strict=True))
    assert bound["A"] == ("view", torch.uint8, a) and bound["W"] == (
        "view",
        torch.uint8,
        w,
    )
    assert bound["OUT"] is out and bound["out_ptr"] is out
    assert bound["sa"] is sa and bound["sw"] is sw
    assert bound["ctl"] == "ctl"
    assert bound["S_mc"] == ("S_mc" if workspace.comm_mode == "nvls" else "ctl")
    assert bound["S"] == "S" and bound["S_peers"] == "S_peers"
    assert bound["flags"] == "flags"
    assert bound["flags_peers"] == "flags_peers" and bound["pg_flags"] == "flags_peers"
    assert (bound["M"], bound["N"], bound["K"]) == (1000, n, K_L)
    assert bound["grid_n"] == loader.grid_n(n, arch)
    assert bound["K_tiles"] == K_L // 128
    assert (bound["C"], bound["C_cap"]) == (8, 8)
    assert bound["S_bytes"] == loader.workspace_bytes(1024, n, world_size, arch)
    assert bound["flag_words"] == 16
    assert (bound["pg_world"], bound["pg_rank"]) == (world_size, 1)
    count = loader.tiles(1000, n, arch)
    if loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS:
        assert bound["pair_tiles"] == count and "tiles" not in bound
        grid = (2 * min(74, count), 1, 1)
    else:
        assert bound["tiles"] == count and "pair_tiles" not in bound
        assert bound["group_m"] == loader.raster_group_m(1000, n, K_L, arch)
        grid = (min(148, count), 1, 1)
    assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == grid
    assert state.calls == 1 and state.captured_calls == 0
    assert tails == [(4242, 0)] and workspace.captured is False


@pytest.mark.cpu_only
def test_captured_launch_marks_the_workspace_and_skips_the_tail(monkeypatch):
    state, workspace, module, tails = _launch_fixture(monkeypatch, capturing=True)
    a, sa, w, sw = _operands(37, workspace.n)
    out = _CudaLike((37, workspace.n), torch.bfloat16)
    backend._launch(state, workspace, _call(37), a, sa, w, sw, out)
    assert len(module.calls) == 1
    assert workspace.captured is True
    assert (state.calls, state.captured_calls) == (1, 1)
    assert tails == []


@pytest.mark.cpu_only
def test_launch_failure_poisons_the_state(monkeypatch):
    state, workspace, module, _ = _launch_fixture(
        monkeypatch, error=RuntimeError("boom")
    )
    a, sa, w, sw = _operands(1, workspace.n)
    out = _CudaLike((1, workspace.n), torch.bfloat16)
    with pytest.raises(RuntimeError, match="boom"):
        backend._launch(state, workspace, _call(1), a, sa, w, sw, out)
    assert state.poisoned and "boom" in state.poison_reason
    assert state.calls == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        backend._launch(state, workspace, _call(1), a, sa, w, sw, out)
    assert len(module.calls) == 1


@pytest.mark.cpu_only
def test_launch_refuses_unknown_launcher_arguments(monkeypatch):
    state, workspace, module, _ = _launch_fixture(monkeypatch)
    monkeypatch.setattr(
        loader,
        "launcher_args",
        lambda program: ("A", "mystery", "grid_x", "grid_y", "grid_z"),
    )
    a, sa, w, sw = _operands(1, workspace.n)
    with pytest.raises(RuntimeError, match=r"unknown arguments \['mystery'\]"):
        backend._launch(
            state,
            workspace,
            _call(1),
            a,
            sa,
            w,
            sw,
            _CudaLike((1, workspace.n), torch.bfloat16),
        )
    assert module.calls == [] and state.poisoned


@pytest.mark.cpu_only
def test_launch_refuses_a_workspace_of_another_architecture():
    call = _call(1)
    state = backend._LaunchState(rank=1, world_size=call.world_size)
    workspace = _workspace(max_rows=128)
    other = backend._Call(
        0, 1, call.world_size, "g", "sm_1a", call.comm_mode, 1, call.n, K_L
    )
    with pytest.raises(RuntimeError, match="another N, world size or architecture"):
        backend._launch(state, workspace, other, *([None] * 5))
    assert state.poisoned is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_prepared_allocation_loads_every_program_its_calls_can_dispatch(
    monkeypatch, arch, world_size
):
    n = _n(arch)
    mode = _mode(arch, world_size)
    handle = _FakeHandle(
        rank=1,
        world_size=world_size,
        buffer_size=loader.workspace_bytes(4133, n, world_size, arch),
        signal_pad_size=65536,
        multicast_ptr=0x1000 if mode == "nvls" else None,
    )
    module = object()
    events, _ = _patch_allocation(monkeypatch, handles=[handle], loaded=module)
    call = _call(1, arch=arch, world_size=world_size)
    workspace = backend._allocate_workspace(
        call, _fake_group(world_size, 1), max_rows=4133, k_l=K_L
    )
    cluster_grid = loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS
    programs = loader.dispatch_programs(
        world_size, arch, 4133, n, K_L, sm_count=0 if cluster_grid else 148
    )
    loads = [event for event in events if event[0] == "load"]
    assert loads == [("load", name, arch) for name in programs]
    # Every program is built before the rendezvous.
    assert events.index(loads[-1]) < events.index(("rendezvous", "receive workspace"))
    assert tuple(workspace.modules) == programs and workspace.program == programs[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_launch_runs_the_program_each_call_dispatches(monkeypatch, arch, world_size):
    state, workspace, _module, _tails = _launch_fixture(
        monkeypatch, arch=arch, world_size=world_size
    )
    n = workspace.n
    workspace.modules = {name: _Module() for name in workspace.modules}
    expected = dict.fromkeys(workspace.modules, 0)
    for m in (1, 129, 1000, 1024):
        a, sa, w, sw = _operands(m, n)
        call = _call(m, arch=arch, world_size=world_size)
        out = _CudaLike((m, n), torch.bfloat16)
        backend._launch(state, workspace, call, a, sa, w, sw, out)
        program = loader.call_program(
            world_size, arch, 1024, m, n, K_L, sm_count=workspace.sm_count
        )
        expected[program] += 1
    assert {name: len(module.calls) for name, module in workspace.modules.items()} == (
        expected
    )
    assert state.calls == 4 and state.poisoned is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_launch_refuses_a_program_the_workspace_did_not_load(
    monkeypatch, arch, world_size
):
    state, workspace, module, _ = _launch_fixture(
        monkeypatch, arch=arch, world_size=world_size
    )
    n = workspace.n
    workspace.modules = {workspace.program: module}
    for m in (1, 129, 1000):
        a, sa, w, sw = _operands(m, n)
        call = _call(m, arch=arch, world_size=world_size)
        out = _CudaLike((m, n), torch.bfloat16)
        program = loader.call_program(
            world_size, arch, 1024, m, n, K_L, sm_count=workspace.sm_count
        )
        if program == workspace.program:
            backend._launch(state, workspace, call, a, sa, w, sw, out)
            continue
        calls = len(module.calls)
        with pytest.raises(RuntimeError, match="did not load program"):
            backend._launch(state, workspace, call, a, sa, w, sw, out)
        assert len(module.calls) == calls
    assert state.poisoned is False


@pytest.mark.cpu_only
def test_cross_stream_join_waits_on_the_previous_tail_and_skips_capture(monkeypatch):
    state = backend._LaunchState(rank=1, world_size=DELIVERED[0][1])
    waited = []
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda index: SimpleNamespace(wait_stream=waited.append, cuda_stream=4242),
    )
    assert backend._join_previous_tail(state, 4242, 0, capturing=False) is False
    previous = SimpleNamespace(cuda_stream=4242)
    state.tail_stream = previous
    assert backend._join_previous_tail(state, 4242, 0, capturing=False) is False
    other = SimpleNamespace(cuda_stream=17)
    state.tail_stream = other
    assert backend._join_previous_tail(state, 4242, 0, capturing=True) is False
    assert backend._join_previous_tail(state, 4242, 0, capturing=False) is True
    assert waited == [other]
    backend._remember_tail(state, 4242, 0)
    assert state.tail_handle == 4242 and state.tail_stream.cuda_stream == 4242


def _e2e_test_module():
    """The multi-GPU e2e test module, loaded from its file (nothing runs)."""

    path = (
        Path(__file__).resolve().parents[2]
        / "comm"
        / "test_fp8_rowwise_gemm_allreduce_cake_e2e.py"
    )
    spec = importlib.util.spec_from_file_location("_cake_fp8_ar_e2e_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch, world_size", DELIVERED)
def test_e2e_lever_shapes_select_every_delivered_lever_program(arch, world_size):
    # The e2e test runs one call per lever program a route delivers, chosen from
    # its LEVER_SHAPES by the dispatch rule. A new lever, or a new bound of one,
    # that leaves a delivered program without a call fails here, without a GPU.
    sm_count = int(loader.GEOMETRY[arch].get("RASTER_SM_COUNT", 148))
    cases = _e2e_test_module()._lever_cases(world_size, arch, sm_count)
    for capacity, key, program, shape in cases:
        assert shape is not None, f"no e2e shape selects {key} on {arch}"
        m, n, k_l = shape
        loader.validate_shape(m, n, k_l, arch)
        rows = capacity or m
        assert (
            loader.call_program(world_size, arch, rows, m, n, k_l, sm_count=sm_count)
            == program
        )
