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

Backend suite of the Cake SM110 GQA decode package
(``flashinfer.experimental.sm110_gqa_decode``, exact SM110a): the cake-build/1
registry and manifest, the size-based route table and the public ``num_splits``
set as data, the grid rules and the argument order, and, on an SM110 device,
the FP32-oracle correctness of every record (the one-CTA routes, the one-split
builds and every accepted split count of the run-time-split programs), the
bitwise determinism of two direct launches, a side-stream launch and two
CUDA-graph replays, and the self-reset of the completion counters.

This suite exercises the backend package alone: it names the package once,
in ``PACKAGE``, and imports nothing else from flashinfer, so a build that
carries the package under another import path runs it unchanged after
pointing ``PACKAGE`` at its package.  The registry and
rule tests run on CPU; every kernel test skips without an SM110 device.  The
public ``flashinfer.sm110_gqa_decode`` / ``prepare_sm110_gqa_decode`` surface is
covered by ``tests/experimental/test_sm110_gqa_decode*.py``.
"""

from __future__ import annotations

import importlib
import math
import re
from pathlib import Path

import pytest
import torch

PACKAGE = "flashinfer.experimental.sm110_gqa_decode"

pkg = importlib.import_module(PACKAGE)
backend = importlib.import_module(f"{PACKAGE}.backend")
loader = importlib.import_module(f"{PACKAGE}.cake_jit")
cake_build = importlib.import_module(f"{PACKAGE}.cake_build")

PACKAGE_DIR = Path(pkg.__file__).resolve().parent
RECORDS = (
    "short",
    "long",
    "split_n32_direct",
    "split_n64_direct",
    "split_n64_r4_direct",
    "split_n32",
    "split_n64",
)
SPLIT_WORKSPACE = ["partial_O", "partial_max", "partial_sum", "completed"]

_REQUIRES_SM110 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="requires an exact SM110 GPU",
)


# ---------------------------------------------------------------------------------------------
# Registry and manifest (CPU)
# ---------------------------------------------------------------------------------------------


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES) == {"sm_110a"}
    libraries = cake_build.libraries_by_name()
    assert set(loader.MODULES) == set(RECORDS)
    referenced = set()
    for name, record in loader.MODULES.items():
        assert not {"sources", "sources_sha256", "compile_flags", "module"} & set(
            record
        ), name
        assert record["ffi_entry"] == "run", name
        assert record["stage"] == "decode" and record["variant"] == name
        assert tuple(record["arches"]) == loader.ARCHES, name
        plan = [tuple(item) for item in record["arg_plan"]]
        assert plan[-3:] == [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
        for kind, key in plan[:-3]:
            logical = record["logical"][key]
            vocabulary = (
                loader.LOGICAL_TENSORS if kind == "buffer" else loader.LOGICAL_SCALARS
            )
            assert logical in vocabulary, (name, key, logical)
        if name.startswith("split_"):
            assert record["workspace"] == SPLIT_WORKSPACE
            assert [key for _, key in plan[:-3]] == [
                "Q",
                "K",
                "V",
                "partial_O",
                "partial_max",
                "partial_sum",
                "completed",
                "O",
                "sequence_lengths",
                "softmax_scale_log2",
            ]
        else:
            assert "workspace" not in record
            assert [key for _, key in plan[:-3]] == [
                "Q",
                "K",
                "V",
                "O",
                "sequence_lengths",
                "softmax_scale_log2",
            ]
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
            assert "--use_fast_math" in entry["nvcc_flags"]
            assert not any(flag.startswith("--fmad") for flag in entry["nvcc_flags"])
    assert referenced == set(libraries)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    assert "csrc" in manifest["common"]["include_dirs"]


def test_constants_and_accepted_splits() -> None:
    assert (loader.NUM_Q_HEADS, loader.NUM_KV_HEADS, loader.HEAD_DIM) == (32, 8, 128)
    assert loader.SHORT_CAPACITY_MAX == backend.MIN_CAPACITY == 64
    assert loader.ACCEPTED_SPLITS == backend.ACCEPTED_SPLITS == (1, 2, 4, 8, 10, 16)


def test_every_row_carries_the_split_count_its_record_launches() -> None:
    """The route table rows carry the record and the split count: "runtime" (the caller's) only on an
    explicit-split row of a run-time-split record, an accepted integer -- today 1 everywhere -- on a size-based
    row, equal to a fixed record's count."""
    fallbacks = []
    for name, record in loader.MODULES.items():
        rule, grid = record["select_rule"], record["grid_rule"]
        runtime = grid["kind"] == "split_pairs" and grid["num_splits"] == "runtime"
        fixed = 1 if grid["kind"] == "batch_heads" else grid["num_splits"]
        if rule["kind"] == "otherwise":
            fallbacks.append(rule["splits"])
            rows = [rule]
        else:
            rows = rule["regions"]
            assert rows, name
        for row in rows:
            assert row["splits"] in ("one", "many"), name
            if row["splits"] == "many":
                assert row["num_splits"] == "runtime" and runtime, name
            else:
                assert row["num_splits"] in loader.ACCEPTED_SPLITS, name
                assert row["num_splits"] == 1, name
                if not runtime:
                    assert row["num_splits"] == fixed, name
    assert sorted(fallbacks) == ["many", "one"]


ROUTE_TABLE = [
    # (batch, capacity, num_splits, record, run-time splits, grid)
    (1, 1, None, "short", 1, (8, 1, 1)),
    (1, 64, None, "short", 1, (8, 1, 1)),
    (1, 64, 1, "short", 1, (8, 1, 1)),
    (1, 65, None, "long", 1, (8, 1, 1)),
    (1, 128, None, "long", 1, (8, 1, 1)),
    (1, 129, None, "split_n64_direct", 1, (1, 8, 1)),
    (1, 256, None, "split_n64_direct", 1, (1, 8, 1)),
    (2, 64, None, "split_n64_direct", 1, (1, 16, 1)),
    (2, 129, 1, "split_n64_direct", 1, (1, 16, 1)),
    (2, 512, None, "split_n64_direct", 1, (1, 16, 1)),
    (1, 257, None, "split_n64_r4_direct", 1, (1, 8, 1)),
    (1, 4096, None, "split_n64_r4_direct", 1, (1, 8, 1)),
    (2, 513, 1, "split_n64_r4_direct", 1, (1, 16, 1)),
    (2, 4096, None, "split_n64_r4_direct", 1, (1, 16, 1)),
    (3, 64, None, "split_n32_direct", 1, (1, 24, 1)),
    (3, 4096, None, "split_n32_direct", 1, (1, 24, 1)),
    (4, 256, None, "split_n32_direct", 1, (1, 32, 1)),
    (1, 64, 2, "split_n64", 2, (2, 8, 1)),
    (1, 128, 16, "split_n64", 16, (16, 8, 1)),
    (1, 129, 2, "split_n64", 2, (2, 8, 1)),
    (1, 1024, 10, "split_n64", 10, (10, 8, 1)),
    (1, 4095, 10, "split_n64", 10, (10, 8, 1)),
    (2, 4095, 16, "split_n64", 16, (16, 16, 1)),
    (1, 4096, 10, "split_n32", 10, (10, 8, 1)),
    (2, 4096, 16, "split_n32", 16, (16, 16, 1)),
    (3, 1024, 4, "split_n32", 4, (4, 24, 1)),
    (4, 256, 8, "split_n32", 8, (8, 32, 1)),
]


@pytest.mark.parametrize("batch,capacity,num_splits,record,splits,grid", ROUTE_TABLE)
def test_size_rule_and_grid(batch, capacity, num_splits, record, splits, grid) -> None:
    assert (
        loader.select_record(batch=batch, capacity=capacity, num_splits=num_splits)
        == record
    )
    assert loader.select_route(
        batch=batch, capacity=capacity, num_splits=num_splits
    ) == (record, splits)
    assert backend.route(batch, capacity, num_splits) == (record, splits, grid)


def test_select_record_refusals() -> None:
    for bad in (0, 3, 5, 32, True, 2.0):
        with pytest.raises(
            ValueError, match="num_splits must be None or an integer in"
        ):
            loader.select_record(batch=1, capacity=1024, num_splits=bad)
        with pytest.raises(
            ValueError, match="num_splits must be None or an integer in"
        ):
            loader.select_route(batch=1, capacity=1024, num_splits=bad)
    with pytest.raises(ValueError, match="positive"):
        loader.select_record(batch=0, capacity=64)
    with pytest.raises(ValueError, match="positive"):
        loader.select_record(batch=1, capacity=0)
    with pytest.raises(ValueError, match="takes no split count"):
        loader.launch_grid("short", batch=1, num_splits=2)
    with pytest.raises(ValueError, match="fixed at 1"):
        loader.launch_grid("split_n64_direct", batch=1, num_splits=2)
    with pytest.raises(ValueError, match="fixed at 1"):
        loader.launch_grid("split_n64_r4_direct", batch=1, num_splits=2)
    with pytest.raises(ValueError, match="needs a positive num_splits"):
        loader.launch_grid("split_n32", batch=1)


def test_argument_list_follows_the_plan() -> None:
    for name, record in loader.MODULES.items():
        values = {logical: f"<{logical}>" for logical in record["logical"].values()}
        args = loader.argument_list(name, values, grid=(7, 11, 13))
        expected = [
            (7, 11, 13)["xyz".index(key[-1])]
            if kind == "grid"
            else f"<{record['logical'][key]}>"
            for kind, key in record["arg_plan"]
        ]
        assert args == expected, name
        with pytest.raises(ValueError, match="needs logical argument"):
            loader.argument_list(name, {}, grid=(1, 1, 1))


# ---------------------------------------------------------------------------------------------
# Kernels (SM110)
# ---------------------------------------------------------------------------------------------


def _inputs(batch, capacity, lengths, seed, q_scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch, 32, 128, dtype=torch.float16, device="cuda", generator=generator
    )
    kv = torch.randn(
        batch,
        2,
        8,
        capacity,
        128,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    return {
        "Q": q,
        "KV": kv,
        "O": torch.empty_like(q),
        "sequence_lengths": torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        "q_scale": q_scale,
    }


def _reference(inputs):
    q, kv, lengths = (inputs[k] for k in ("Q", "KV", "sequence_lengths"))
    batch, _, head_dim = q.shape
    capacity = kv.shape[-2]
    grouped_q = q.float().view(batch, 8, 4, head_dim)
    scores = torch.einsum("bhgd,bhkd->bhgk", grouped_q, kv[:, 0].float())
    positions = torch.arange(capacity, device=q.device)
    valid = positions.view(1, 1, 1, capacity) < lengths.view(batch, 1, 1, 1)
    scores.masked_fill_(~valid, -torch.inf)
    probabilities = torch.softmax(
        scores * inputs["q_scale"] / math.sqrt(head_dim), dim=-1
    )
    return (
        torch.einsum("bhgk,bhkd->bhgd", probabilities, kv[:, 1].float())
        .reshape_as(q)
        .half()
    )


def _replays_are_bitwise(launch, output, expected):
    """Two direct launches, one side-stream eager launch and two CUDA-graph replays (poisoned output in
    between) are byte-identical, and the first is correct."""
    output.fill_(float("nan"))
    launch()
    first = output.clone()
    torch.testing.assert_close(first, expected, atol=1e-2, rtol=1e-2)
    output.fill_(float("nan"))
    launch()
    assert torch.equal(first, output), "two direct launches differ"
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        output.fill_(float("nan"))
        launch()
        side_out = output.clone()
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(first, side_out), "the side-stream launch differs"
    graph = torch.cuda.CUDAGraph()
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        with torch.cuda.graph(graph, stream=capture):
            launch()
        for replay in range(2):
            output.fill_(float("nan"))
            graph.replay()
            assert torch.equal(first, output), f"graph replay {replay + 1} differs"
    torch.cuda.current_stream().wait_stream(capture)


KERNEL_CASES = [
    # (batch, capacity, lengths, num_splits, record)
    (1, 64, [1], None, "short"),
    (1, 64, [64], None, "short"),
    (1, 128, [100], None, "long"),
    (1, 256, [200], None, "split_n64_direct"),
    (2, 300, [37, 300], None, "split_n64_direct"),
    (2, 65, [1, 65], None, "split_n64_direct"),
    (1, 1024, [1024], None, "split_n64_r4_direct"),
    (2, 513, [1, 513], None, "split_n64_r4_direct"),
    (4, 256, [64, 127, 191, 256], None, "split_n32_direct"),
    (3, 1023, [255, 257, 1023], None, "split_n32_direct"),
    (1, 1024, [1024], 10, "split_n64"),
    (2, 1023, [63, 1023], 4, "split_n64"),
    (1, 64, [1], 16, "split_n64"),
    (1, 4096, [3968], 10, "split_n32"),
    (4, 256, [64, 127, 191, 256], 8, "split_n32"),
    (8, 64, [1, 2, 3, 4, 5, 6, 7, 8], 2, "split_n32"),
]


@_REQUIRES_SM110
@pytest.mark.parametrize("batch,capacity,lengths,num_splits,record", KERNEL_CASES)
@pytest.mark.parametrize("q_scale", [1.0, 1.5])
def test_records_match_the_oracle_and_replay_bitwise(
    batch, capacity, lengths, num_splits, record, q_scale
):
    inputs = _inputs(batch, capacity, lengths, 7 * batch + capacity, q_scale)
    expected = _reference(inputs)
    state = backend.prepare_for_launch(inputs, num_splits=num_splits)
    assert state["route"] == record
    assert state["num_splits"] == (num_splits or 1)
    assert state["grid"] == backend.route(batch, capacity, num_splits)[2]
    _replays_are_bitwise(lambda: backend.launch_prepared(state), inputs["O"], expected)
    if record.startswith("split_"):
        partial_o, partial_max, partial_sum, completed = state["workspace"]
        assert tuple(partial_o.shape) == (batch, 32, state["num_splits"], 128)
        assert (
            tuple(partial_max.shape)
            == tuple(partial_sum.shape)
            == (batch, 32, state["num_splits"])
        )
        if loader.MODULES[record]["grid_rule"]["num_splits"] == "runtime":
            # The last CTA of every (request, KV head) resets its counter after the merge.
            assert torch.equal(completed, torch.zeros_like(completed))
    else:
        assert state["workspace"] == ()


@_REQUIRES_SM110
@pytest.mark.parametrize(
    "batch,capacity,lengths,num_splits,record",
    [case for case in KERNEL_CASES if case[3] is None],
)
def test_convenience_api_matches_the_oracle(
    batch, capacity, lengths, num_splits, record
):
    inputs = _inputs(batch, capacity, lengths, 11 * batch + capacity)
    expected = _reference(inputs)
    out = torch.full_like(inputs["Q"], float("nan"))
    result = backend.sm110_gqa_decode(
        inputs["Q"], inputs["KV"], inputs["sequence_lengths"], out=out
    )
    assert result is out
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    again = backend.sm110_gqa_decode(
        inputs["Q"], inputs["KV"], inputs["sequence_lengths"]
    )
    assert torch.equal(again, out)


@_REQUIRES_SM110
def test_caches_below_the_short_tma_box_are_rejected():
    """Both entry points keep the previous release's rejection of a cache under
    64 tokens (its short route's TMA box), type and message included."""
    inputs = _inputs(1, 32, [32], 7)
    out = torch.full_like(inputs["Q"], float("nan"))
    message = r"TMA box \(64, 64, 1, 1\) exceeds resolved global dims for 'K'"
    with pytest.raises(ValueError, match=message):
        backend.sm110_gqa_decode(
            inputs["Q"], inputs["KV"], inputs["sequence_lengths"], out=out
        )
    with pytest.raises(ValueError, match=message):
        backend.prepare_for_launch({**inputs, "O": out}, num_splits=2)
    assert torch.isnan(out).all().item()


@_REQUIRES_SM110
def test_dynamic_lengths_replay_from_a_graph():
    inputs = _inputs(2, 1024, [1024, 1024], 99, 1.5)
    state = backend.prepare_for_launch(inputs, num_splits=8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.launch_prepared(state)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        backend.launch_prepared(state)
    snapshots = []
    with torch.cuda.stream(stream):
        for lengths in ([1, 1024], [127, 513], [1023, 65], [1024, 1]):
            inputs["sequence_lengths"].copy_(
                torch.tensor(lengths, dtype=torch.int32, device="cuda")
            )
            for _ in range(4):
                graph.replay()
            snapshots.append((inputs["O"].clone(), _reference(inputs)))
    stream.synchronize()
    for actual, expected in snapshots:
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    completed = state["workspace"][-1]
    assert torch.equal(completed, torch.zeros_like(completed))
