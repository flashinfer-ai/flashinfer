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

Backend suite of the Cake SM110 XQA package (``flashinfer.experimental.sm110_xqa``,
exact SM110a): the cake-build/1 registry and manifest, the route and grid rules
of the D512 tree schedules and the D128 decode, and, on an SM110 device, the
FP32-oracle correctness of every schedule, the bitwise determinism of two direct
launches, a side-stream launch and two CUDA-graph replays, the uniform query
prefix the backend synthesizes, the decode counter reset and the retirement of
the superseded tree routes.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path runs
it unchanged after pointing that table entry at its package.  The registry and
rule tests run on CPU; every kernel test skips without an SM110 device.  The
public ``flashinfer.sm110_xqa`` surface is covered by
``tests/experimental/sm110_xqa/test_sm110_xqa.py``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("sm110_xqa")
backend = cake_package("sm110_xqa", "backend")
loader = cake_package("sm110_xqa", "cake_jit")
cake_build = cake_package("sm110_xqa", "cake_build")

PACKAGE_DIR = Path(pkg.__file__).resolve().parent
LEDGER = PACKAGE_DIR.parents[2] / "benchmarks" / "sm110_xqa_shapes.json"
PLAN_KINDS = {"buffer", "parameter", "grid"}
DECODE_RECORD = "decode_fp16_contig"

_REQUIRES_SM110 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="requires an exact SM110 GPU",
)


# ---------------------------------------------------------------------------------------------
# Registry and manifest (CPU)
# ---------------------------------------------------------------------------------------------


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding sources
    under ``csrc/`` of this package, their digests, a closure digest and the source
    build's numerics flag; the registry itself records the host plan only.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES) == {"sm_110a"}
    libraries = cake_build.libraries_by_name()
    referenced = set()
    for name, record in loader.MODULES.items():
        assert not {"sources", "sources_sha256", "compile_flags"} & set(record), name
        assert record["ffi_entry"] == "run", name
        assert record["stage"] in loader.STAGES, name
        assert tuple(record["arches"]) == loader.ARCHES, name
        plan = [tuple(item) for item in record["arg_plan"]]
        assert {kind for kind, _ in plan} <= PLAN_KINDS, name
        assert plan[-3:] == [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
        for kind, key in plan[:-3]:
            logical = record["logical"][key]
            vocabulary = (
                loader.LOGICAL_TENSORS if kind == "buffer" else loader.LOGICAL_SCALARS
            )
            assert logical in vocabulary, (name, key, logical)
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (PACKAGE_DIR / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            # The source build's numerics: fast math on every entry, the default --fmad left to nvcc.
            assert "--use_fast_math" in entry["nvcc_flags"]
            assert not any(flag.startswith("--fmad") for flag in entry["nvcc_flags"])
    assert referenced == set(libraries)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    assert "csrc" in manifest["common"]["include_dirs"]


@pytest.mark.parametrize("precision", ["fp16", "fp8"])
@pytest.mark.parametrize("cache", ["contig", "page128"])
@pytest.mark.parametrize("ratio", [2, 4, 8, 16])
@pytest.mark.parametrize("q_len", [1, 7, 20, 32, 33, 64, 65])
def test_select_record_and_launch_grid(precision, cache, ratio, q_len) -> None:
    tiles = -(-q_len * ratio // loader.TILE_ROWS)
    even = tiles % 2 == 0
    tmem = loader.select_record(
        precision=precision, cache=cache, ratio=ratio, q_len=q_len, kernel="tmem"
    )
    assert tmem == f"tree_{precision}_{cache}_g{ratio}_tmem_{'even' if even else 'odd'}"
    assert loader.launch_grid(tmem, num_kv_heads=3, batch=2, q_len=q_len) == (
        2,
        3 * tiles,
        2,
    )
    auto = loader.select_record(
        precision=precision, cache=cache, ratio=ratio, q_len=q_len
    )
    if precision == "fp8" and even:
        assert auto == f"tree_fp8_{cache}_g{ratio}_pair"
        assert loader.launch_grid(auto, num_kv_heads=3, batch=2, q_len=q_len) == (
            4,
            3 * tiles // 2,
            2,
        )
        assert (
            loader.select_record(
                precision=precision,
                cache=cache,
                ratio=ratio,
                q_len=q_len,
                kernel="pair",
            )
            == auto
        )
    else:
        assert auto == tmem
        with pytest.raises(ValueError, match="E4M3|even number"):
            loader.select_record(
                precision=precision,
                cache=cache,
                ratio=ratio,
                q_len=q_len,
                kernel="pair",
            )


def test_select_record_refuses_outside_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="precision"):
        loader.select_record(precision="bf16", cache="contig", ratio=8, q_len=20)
    with pytest.raises(ValueError, match="cache"):
        loader.select_record(precision="fp16", cache="page64", ratio=8, q_len=20)
    with pytest.raises(ValueError, match="Hq/Hkv"):
        loader.select_record(precision="fp16", cache="contig", ratio=3, q_len=20)
    with pytest.raises(ValueError, match="kernel must be one of"):
        loader.select_record(
            precision="fp16", cache="contig", ratio=8, q_len=20, kernel="register_mma"
        )
    assert set(backend.RETIRED_KERNELS).isdisjoint(loader.KERNELS)
    assert backend.SELECTABLE_KERNELS == loader.KERNELS == ("auto", "tmem", "pair")


@pytest.mark.parametrize(
    "capacity,partition_tokens,partitions",
    [
        (1, None, 1),
        (256, None, 1),
        (257, None, 2),
        (1024, None, 4),
        (513, 64, 9),
        (513, 320, 2),
        (513, 1024, 1),
    ],
)
def test_decode_record_partitions_and_grid(
    capacity, partition_tokens, partitions
) -> None:
    for ratio in loader.DECODE_RATIOS:
        assert loader.select_decode_record(ratio=ratio) == DECODE_RECORD
    count, tokens = loader.decode_partitions(capacity, partition_tokens)
    assert (count, tokens) == (
        partitions,
        partition_tokens or loader.DEFAULT_PARTITION_TOKENS,
    )
    assert loader.launch_grid(
        DECODE_RECORD, num_kv_heads=4, batch=3, partitions=count
    ) == (count, 4, 3)
    # The count selects the record: the shipped cluster program of two or four
    # partitions, else the workspace merge.
    for ratio in loader.DECODE_RATIOS:
        record = loader.select_decode_record(ratio=ratio, partitions=count)
        clustered = loader.MODULES[record]["variant"].get("cluster_partitions")
        assert loader.MODULES[record]["stage"] == "decode"
        assert clustered in (None, count)
        assert (record == DECODE_RECORD) == (clustered is None)
        assert record == (
            f"{DECODE_RECORD}_c{count}" if count in (2, 4) else DECODE_RECORD
        )
        assert loader.launch_grid(
            record, num_kv_heads=4, batch=3, partitions=count
        ) == (count, 4, 3)
    with pytest.raises(ValueError, match="Hq/Hkv"):
        loader.select_decode_record(ratio=2)
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="partitions must be None or a positive"):
            loader.select_decode_record(ratio=8, partitions=bad)
    for bad in (0, 65, 63, -64, 2**32):
        with pytest.raises(ValueError, match="positive multiple of 64"):
            loader.decode_partitions(capacity, bad)


def test_decode_record_selection_prefers_a_cluster_program_of_the_count(
    monkeypatch,
) -> None:
    """``select_decode_record`` returns the shipped cluster record whose
    ``cluster_partitions`` equals the count (two and four) and the workspace merge
    for every other count; a cluster record launches its count only; two records
    of one count are refused."""
    for count in (2, 4):
        record = loader.MODULES[f"{DECODE_RECORD}_c{count}"]
        assert record["variant"] == {
            "precision": "fp16",
            "cache": "contig",
            "cluster_partitions": count,
        }
        assert record["library"] != loader.MODULES[DECODE_RECORD]["library"]
    assert loader.select_decode_record(ratio=8, partitions=2) == f"{DECODE_RECORD}_c2"
    assert loader.select_decode_record(ratio=4, partitions=4) == f"{DECODE_RECORD}_c4"
    for count in (None, 1, 3, 8, 9):
        assert loader.select_decode_record(ratio=16, partitions=count) == DECODE_RECORD
    assert loader.launch_grid(
        f"{DECODE_RECORD}_c2", num_kv_heads=4, batch=3, partitions=2
    ) == (2, 4, 3)
    with pytest.raises(ValueError, match="merges exactly 2 partitions"):
        loader.launch_grid(f"{DECODE_RECORD}_c2", num_kv_heads=4, batch=3, partitions=4)
    assert loader.launch_grid(DECODE_RECORD, num_kv_heads=4, batch=3, partitions=4) == (
        4,
        4,
        3,
    )
    registry = dict(loader.MODULES)
    registry[f"{DECODE_RECORD}_c2_twin"] = registry[f"{DECODE_RECORD}_c2"]
    monkeypatch.setattr(loader, "MODULES", registry)
    with pytest.raises(ValueError, match="2 decode records for 2 cluster partitions"):
        loader.select_decode_record(ratio=8, partitions=2)


def test_argument_list_follows_the_plan() -> None:
    for name, record in loader.MODULES.items():
        plan = [tuple(item) for item in record["arg_plan"]]
        values = {logical: f"<{logical}>" for logical in record["logical"].values()}
        args = loader.argument_list(name, values, grid=(7, 11, 13))
        expected = [
            (7, 11, 13)["xyz".index(key[-1])]
            if kind == "grid"
            else f"<{record['logical'][key]}>"
            for kind, key in plan
        ]
        assert args == expected, name
        with pytest.raises(ValueError, match="needs logical argument"):
            loader.argument_list(name, {}, grid=(1, 1, 1))


def test_ledger_rows_are_served() -> None:
    if not LEDGER.is_file():
        pytest.skip(
            "the benchmark ledger ships with the FlashInfer tree, not the package"
        )
    ledger = json.loads(LEDGER.read_text())
    assert ledger["architecture"] == "sm_110a"
    assert ledger["shape_count"] == len(ledger["shapes"])
    for shape in ledger["shapes"]:
        ratio = shape["query_heads"] // shape["kv_heads"]
        if shape["head_dim"] == 128:
            assert shape["name"].startswith("decode_")
            assert loader.select_decode_record(ratio=ratio) == DECODE_RECORD
            continue
        kernel = shape.get("kernel", "auto")
        assert kernel in loader.KERNELS, shape["name"]
        precision = "fp8" if shape["kv_dtype"] == "float8_e4m3fn" else "fp16"
        cache = "page128" if shape["page_size"] else "contig"
        record = loader.select_record(
            precision=precision,
            cache=cache,
            ratio=ratio,
            q_len=shape["query_tokens"],
            kernel=kernel,
        )
        assert record in loader.MODULES


# ---------------------------------------------------------------------------------------------
# Kernels (SM110)
# ---------------------------------------------------------------------------------------------


def _sample(shape, seed, distribution="uniform"):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.empty(shape, device="cuda", dtype=torch.float16)
    if distribution == "uniform":
        return values.uniform_(-1, 1, generator=generator)
    return values.normal_(generator=generator)


def _causal_mask(queries, batch):
    rows = []
    for row in range(queries):
        words = []
        for word in range((queries + 31) // 32):
            bits = sum(
                1 << (token % 32) for token in range(row + 1) if token // 32 == word
            )
            words.append(bits if bits < 2**31 else bits - 2**32)
        rows.append(words)
    return (
        torch.tensor(rows, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch, 1, 1)
    )


def _cache(kv, fp8, paged):
    if fp8:
        k_scale = float(kv[:, 0].abs().max()) / 448.0
        v_scale = float(kv[:, 1].abs().max()) / 448.0
        scales = torch.tensor([k_scale, v_scale], device="cuda").reshape(1, 2, 1, 1, 1)
        stored = (kv.float() / scales).to(torch.float8_e4m3fn)
        dense = stored.float() * scales
    else:
        stored, dense, k_scale, v_scale = kv, kv.float(), 1.0, 1.0
    if not paged:
        return stored, None, dense, k_scale, v_scale
    batch, _, heads, capacity, dim = kv.shape
    pages = capacity // 128
    storage = stored.reshape(batch, 2, heads, pages, 128, dim).permute(0, 1, 3, 4, 2, 5)
    storage = storage.contiguous().reshape(batch * 2 * pages, 128, heads, dim)
    permutation = torch.randperm(
        storage.shape[0],
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(5),
    )
    page_table = permutation.argsort().to(torch.int32).reshape(batch, 2, pages)
    return storage[permutation], page_table, dense, k_scale, v_scale


def _tree_oracle(q, dense, lengths, mask):
    batch, queries, q_heads, dim = q.shape
    heads = dense.shape[2]
    ratio = q_heads // heads
    result = []
    for request, length in enumerate(lengths):
        key = dense[request, 0, :, :length].repeat_interleave(ratio, dim=0)
        value = dense[request, 1, :, :length].repeat_interleave(ratio, dim=0)
        scores = torch.einsum("qhd,hkd->qhk", q[request].float(), key) / math.sqrt(dim)
        draft = torch.arange(queries, device="cuda")
        allowed = (
            (mask[request, :, draft // 32].to(torch.int64) >> (draft % 32)) & 1
        ).bool()
        scores[:, :, length - queries :] = scores[:, :, length - queries :].masked_fill(
            ~allowed[:, None], -torch.inf
        )
        probabilities = torch.softmax(scores, dim=-1)
        result.append(torch.einsum("qhk,hkd->qhd", probabilities, value))
    return torch.stack(result).to(torch.float16)


def _decode_oracle(q, kv, lengths):
    batch, q_heads, dim = q.shape
    heads = kv.shape[2]
    ratio = q_heads // heads
    result = []
    for request, length in enumerate(lengths):
        key = kv[request, 0, :, :length].float().repeat_interleave(ratio, dim=0)
        value = kv[request, 1, :, :length].float().repeat_interleave(ratio, dim=0)
        scores = torch.einsum("hd,hkd->hk", q[request].float(), key) / math.sqrt(dim)
        result.append(torch.einsum("hk,hkd->hd", torch.softmax(scores, dim=-1), value))
    return torch.stack(result).to(torch.float16)


def _tree_plan(batch, queries, ratio, capacity, fp8, paged, kernel, seed=11):
    q = _sample((batch, queries, 4 * ratio, 512), seed)
    kv = _sample((batch, 2, 4, capacity, 512), seed + 1)
    kv[:, 0].mul_(0.5)
    kv[:, 1].mul_(2.0)
    lengths = [capacity] * batch
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    mask = _causal_mask(queries, batch)
    stored, pages, dense, k_scale, v_scale = _cache(kv, fp8, paged)
    plan = backend.prepare(
        q,
        stored,
        seq,
        mask=mask,
        page_table=pages,
        page_size=128 if paged else 0,
        k_scale=k_scale,
        v_scale=v_scale,
        kernel=kernel,
    )
    return plan, _tree_oracle(q, dense, lengths, mask)


def _replays_are_bitwise(plan, expected, atol=1e-2):
    """GOAL-grade determinism: two direct launches, one side-stream eager launch and two CUDA-graph replays
    (poisoned output in between) are byte-identical, and the first is correct."""
    plan.output.fill_(float("nan"))
    first = plan.run().clone()
    torch.testing.assert_close(first, expected, atol=atol, rtol=atol)
    plan.output.fill_(float("nan"))
    second = plan.run().clone()
    assert torch.equal(first, second), "two direct launches differ"
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        plan.output.fill_(float("nan"))
        plan.run()
        side_out = plan.output.clone()
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(first, side_out), "the side-stream launch differs"
    graph = torch.cuda.CUDAGraph()
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        with torch.cuda.graph(graph, stream=capture):
            plan.run()
        for replay in range(2):
            plan.output.fill_(float("nan"))
            graph.replay()
            assert torch.equal(first, plan.output), f"graph replay {replay + 1} differs"
    torch.cuda.current_stream().wait_stream(capture)


TREE_SHAPES = [
    # (batch, queries, ratio, capacity, fp8, paged, kernel)
    (1, 20, 8, 256, False, False, "auto"),  # fp16 even -> tmem_even
    (1, 7, 8, 256, False, True, "auto"),  # fp16 odd, paged -> tmem_odd
    (2, 20, 8, 256, True, False, "auto"),  # fp8 even -> pair
    (1, 33, 8, 384, True, True, "auto"),  # fp8 odd, paged -> tmem_odd
    (1, 20, 8, 256, True, True, "tmem"),  # explicit tmem over an fp8 even shape
    (1, 65, 2, 256, True, False, "pair"),  # explicit pair at ratio 2 (two tiles)
]


@_REQUIRES_SM110
@pytest.mark.parametrize("batch,queries,ratio,capacity,fp8,paged,kernel", TREE_SHAPES)
def test_tree_schedules_match_the_oracle_and_replay_bitwise(
    batch, queries, ratio, capacity, fp8, paged, kernel
):
    plan, expected = _tree_plan(batch, queries, ratio, capacity, fp8, paged, kernel)
    precision = "fp8" if fp8 else "fp16"
    cache = "page128" if paged else "contig"
    assert plan.route == loader.select_record(
        precision=precision, cache=cache, ratio=ratio, q_len=queries, kernel=kernel
    )
    assert plan.program == loader.MODULES[plan.route]["library"]
    assert plan.grid == loader.launch_grid(
        plan.route, num_kv_heads=4, batch=batch, q_len=queries
    )
    _replays_are_bitwise(plan, expected)


@_REQUIRES_SM110
def test_uniform_queries_launch_with_a_synthesized_prefix():
    """The kernels read the query offsets through a non-null pointer: a uniform call carries the arithmetic
    prefix 0, Q, 2Q, ... among the plan's retained inputs."""
    plan, expected = _tree_plan(3, 20, 8, 256, False, False, "auto")
    prefixes = [
        item
        for item in plan.inputs
        if item.dtype == torch.int32 and tuple(item.shape) == (4,)
    ]
    assert len(prefixes) == 1
    assert prefixes[0].tolist() == [0, 20, 40, 60]
    torch.testing.assert_close(plan.run(), expected, atol=1e-2, rtol=1e-2)


@_REQUIRES_SM110
@pytest.mark.parametrize("kernel", backend.RETIRED_KERNELS)
def test_retired_kernels_raise(kernel):
    q = _sample((1, 20, 32, 512), 3)
    kv = _sample((1, 2, 4, 256, 512), 4)
    seq = torch.full((1,), 256, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="is retired"):
        backend.prepare(q, kv, seq, mask=_causal_mask(20, 1), kernel=kernel)
    with pytest.raises(ValueError, match="kernel must be one of"):
        backend.prepare(q, kv, seq, mask=_causal_mask(20, 1), kernel="wgmma")


@_REQUIRES_SM110
@pytest.mark.parametrize(
    "batch,capacity,ratio,partition_tokens",
    [
        (1, 1024, 8, None),
        (2, 512, 8, None),
        (4, 256, 8, None),
        (3, 513, 8, 64),
        (1, 1024, 4, 320),
        (2, 513, 16, 1024),
    ],
)
def test_decode_matches_the_oracle_and_replays_bitwise(
    batch, capacity, ratio, partition_tokens
):
    q = _sample((batch, 4 * ratio, 128), 21, "normal")
    kv = _sample((batch, 2, 4, capacity, 128), 22, "normal")
    lengths = [max(1, capacity * (request + 1) // batch) for request in range(batch)]
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    plan = backend.prepare(q, kv, seq, partition_tokens=partition_tokens)
    partitions, tokens = loader.decode_partitions(capacity, partition_tokens)
    assert plan.route == loader.select_decode_record(ratio=ratio, partitions=partitions)
    assert plan.route == (
        f"{DECODE_RECORD}_c{partitions}" if partitions in (2, 4) else DECODE_RECORD
    )
    assert plan.program == loader.MODULES[plan.route]["library"]
    assert plan.grid == (partitions, 4, batch)
    partial, statistics, counters = plan.workspace
    assert tuple(partial.shape) == (batch * 4 * ratio, partitions, 128)
    assert tuple(statistics.shape) == (batch * 4 * ratio, partitions, 2)
    assert tuple(counters.shape) == (batch, 4) and counters.dtype == torch.int32
    _replays_are_bitwise(plan, _decode_oracle(q, kv, lengths))
    # The last-CTA merge wraps every counter back to zero after each launch.
    assert torch.equal(counters, torch.zeros_like(counters))


@_REQUIRES_SM110
def test_decode_rejects_partition_lengths_outside_the_contract():
    q = _sample((1, 32, 128), 31, "normal")
    kv = _sample((1, 2, 4, 256, 128), 32, "normal")
    seq = torch.full((1,), 256, device="cuda", dtype=torch.int32)
    for bad in (0, 100, 2**32):
        with pytest.raises(ValueError, match="positive multiple of 64"):
            backend.prepare(q, kv, seq, partition_tokens=bad)
    with pytest.raises(ValueError, match="tcgen05 kernel family"):
        backend.prepare(q, kv, seq, kernel="tmem")
    for name in ("auto", "tcgen05"):
        assert backend.prepare(q, kv, seq, kernel=name).route == DECODE_RECORD
