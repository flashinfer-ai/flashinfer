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

Backend suite of the Cake StepFun fused MoE (``flashinfer.experimental.
cake_stepfun_moe``, exact SM100 / SM103): the cake-build/1 registry and
manifest, the keyword-only prepare surface, and every exported stage --
routing (Renormalize top-k), FC1 with the StepFun epilogue, the NVFP4
per-token requantization, FC2 and finalize -- plus the whole graph, each
against the torch-only fp32 reference of
``tests/test_helpers/cake_stepfun_moe_reference.py`` at the per-family
tolerances (bf16 1e-2 / 1e-2, fp8 and mxfp8 0.1 / 0.1 with the e4m3 one-ulp
block rule, nvfp4 1.0 / 0.1 after dequantization with the e2m1 one-code block
rule; routing tables bitwise, permutation maps canonically above 16 tokens).
The whole graph is gated hop by hop on the kernel's own upstream rows (FC1
rows, requantized rows, FC2 rows, output; ``reference.graph_hops``) because
kernel and reference round every stage independently and finalize carries a
flipped row ulp into much smaller outputs; the end-to-end error against the
graph reference is recorded, not asserted.
The limit modes 7 / 16 / mixed / 0.5 / inf and the routing regimes uniform /
ragged / randperm / empty (experts 13 and 41 receive no token) follow the
reference module.  CUDA-graph replays and fresh-process first replays must
equal eager bitwise, and a whole-graph forward launches only Cake kernels.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
registry and keyword tests run on CPU; every kernel test skips without an
SM100 / SM103 device.  The ``flashinfer.fused_moe`` facade
(``CakeStepFunConfig(backend="cake_stepfun")``) and the native comparison are
covered by ``test_cake_stepfun_moe_api.py``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from tests.test_helpers import cake_stepfun_moe_reference as ref
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_stepfun_moe")
cb = cake_package("cake_stepfun_moe", "cake_backend")
loader = cake_package("cake_stepfun_moe", "cake_jit")
cake_build = cake_package("cake_stepfun_moe", "cake_build")

pytestmark = pytest.mark.long_running

H, I, E, K = ref.HIDDEN_SIZE, ref.INTERMEDIATE_SIZE, ref.NUM_EXPERTS, ref.TOP_K
PRECISIONS = tuple(ref.PRECISION_FAMILY)
PACKAGE_DIR = Path(pkg.__file__).resolve().parent
STAGE_TABLE_SIZES = {"routing": 50, "fc1": 26, "requant": 2, "fc2": 27, "finalize": 4}
ROUTE_COUNT = 135
ARG_PLAN_COUNT = 20
# The export's measurement catalogue (``ROUTES``), delivered with the tests.
ROUTES_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_helpers"
    / "cake_stepfun_moe_routes.json"
)
CAKE_KERNEL_SYMBOL = re.compile(r"\bkernel_cake_stepfun_moe_[0-9a-f]+\b")
# The retired per-arch minor macro: its name must not appear in any delivered source.
RETIRED_MINOR_MACRO = "TARGET" + "_MINOR"


# ---------------------------------------------------------------------------------------------
# Device, warm-up, shared data
# ---------------------------------------------------------------------------------------------


def _capability():
    return torch.cuda.get_device_capability(torch.device("cuda", 0))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in cb.SUPPORTED_COMPUTE_CAPABILITIES
        and bool(loader.PROGRAMS)
    )


def _require_device(*families: str) -> tuple[torch.device, str]:
    """The SM100 / SM103 device and its arch name; skips elsewhere and for a family without
    exported FC1 / FC2 units on this registry."""
    if not torch.cuda.is_available():
        pytest.skip("Cake StepFun fused MoE requires CUDA.")
    capability = _capability()
    arch = cb.SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        pytest.skip(
            "Cake StepFun fused MoE requires exact SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}."
        )
    if not loader.PROGRAMS:
        pytest.fail("the registry of cake_stepfun_moe.cake_jit is empty")
    for family in families:
        for stage in ("fc1", "fc2"):
            if not cb.exported_tiles(stage, family):
                pytest.skip(f"no exported {stage} unit of family {family!r} for {arch}")
    return torch.device("cuda", torch.cuda.current_device()), arch


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before the first
    # kernel test, through the spec objects the package's loader itself builds on first use
    # (``cake_build.load_library``); a prebuilt or already cached library loads without
    # compiling.
    if _supported():
        arch = cb.SUPPORTED_COMPUTE_CAPABILITIES[_capability()]
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


@pytest.fixture(scope="module")
def weights_cache():
    return {}


def _weights(cache, family: str, seed: int, device) -> ref.StepFunWeights:
    key = (family, seed)
    if key not in cache:
        cache[key] = ref.build_weights(family, seed, device)
    return cache[key]


def _sync():
    torch.cuda.synchronize()


def _ragged64_routing(tile: int, device):
    """The host router's 64-expert tables of the ``empty`` regime at T=2048 (experts 13 / 41 empty)."""
    return ref.reference_routing(
        ref.routing_logits(2048, "empty", device=device), tile=tile
    )


# Explicit per-expert count fixtures: (counts, activation count the backend sees).
_COUNT_FIXTURES = {
    "fc1_counts": (ref.COUNTS_E2E_FC1, 64),
    "fc1_counts_mxfp8": (ref.COUNTS_E2E_FC1_MXFP8, 64),
    "fc2_counts": (ref.COUNTS_E2E_FC2, 256),
}


def _routing_fixture(name: str, tile: int, device):
    """(routing tables, activation count) of a stage-level fixture: explicit per-expert counts on
    64 / 256 activations, or the host router's 64-expert tables of the ``empty`` regime at T=2048.
    The tables carry the backend's extents for that activation count (``max_num_ctas`` CTA
    entries, ``max_padded + 1`` permutation slots); ``num_non_exiting_ctas`` /
    ``total_num_padded_tokens`` are the live values."""
    if name == "ragged64":
        return _ragged64_routing(tile, device), 2048
    counts, num_tokens = _COUNT_FIXTURES[name]
    tables = ref.routing_from_counts(
        counts, tile=tile, device=device, num_tokens=num_tokens
    )
    return tables, num_tokens


def _routing_kwargs(tables: dict, *, stage: str) -> dict:
    """The routing-table keywords of ``prepare_fc1`` (``stage="fc1"``: the CTA tables, the counts
    and ``permuted_idx_to_token_idx``) or ``prepare_fc2`` (``stage="fc2"``: no permutation) from
    fixture or reference tables, at their full extents; the counts are the live values."""
    nnec = ref._as_int(tables["num_non_exiting_ctas"])
    device = tables["cta_idx_xy_to_batch_idx"].device
    kwargs = dict(
        cta_idx_xy_to_batch_idx=tables["cta_idx_xy_to_batch_idx"]
        .reshape(-1)
        .contiguous(),
        cta_idx_xy_to_mn_limit=tables["cta_idx_xy_to_mn_limit"]
        .reshape(-1)
        .contiguous(),
        num_non_exiting_ctas=torch.tensor([nnec], dtype=torch.int32, device=device),
        total_num_padded_tokens=torch.tensor(
            [ref._as_int(tables["total_num_padded_tokens"])],
            dtype=torch.int32,
            device=device,
        ),
    )
    if stage == "fc1":
        kwargs["permuted_idx_to_token_idx"] = (
            tables["permuted_idx_to_token_idx"].reshape(-1).contiguous()
        )
    elif stage != "fc2":
        raise KeyError(stage)
    return kwargs


def _prepare_graph(case: ref.StepFunCase, logits: torch.Tensor, **overrides):
    kwargs = dict(
        family=case.family,
        routing_logits=logits,
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
    )
    kwargs.update(case.view)
    kwargs.update(overrides)
    return cb.prepare_stepfun_moe(**kwargs)


def _plan_kernels(launch) -> list[str]:
    """The kernel symbols one forward of ``launch`` must run, in stage order."""
    programs = list(launch.routing_programs) + [launch.plan.fc1_program]
    if launch.plan.requant_program is not None:
        programs.append(launch.plan.requant_program)
    programs.append(launch.plan.fc2_program)
    if launch.finalize_program is not None:
        programs.append(launch.finalize_program)
    return [loader.PROGRAMS[program]["kernel"] for program in programs]


def _profiled_kernel_names(fn) -> list[str]:
    """CUDA kernel names of one call of ``fn`` (memcpy / memset activities excluded).  The profiler's
    warm-up step runs one call whose records are discarded: CUDA activity collection starts
    asynchronously once the profiler is entered."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
    ) as profile:
        fn()
        _sync()
        profile.step()
        fn()
        _sync()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
    ]


def _capture(launch, outputs):
    """Capture ``launch.launch()`` in a CUDA graph (after a warm-up on the capture stream); returns the
    graph and clones of ``outputs`` taken after the eager warm-up."""
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        launch.launch()
        stream.synchronize()
        eager = [t.clone() for t in outputs]
        with torch.cuda.graph(graph, stream=stream):
            launch.launch()
    _sync()
    return graph, eager


# ---------------------------------------------------------------------------------------------
# B1 -- registry, manifest, launchers (CPU)
# ---------------------------------------------------------------------------------------------


def _defines_kernel(text: str, symbol: str) -> bool:
    """``text`` holds the ``__global__`` definition (parameter list followed by a body) of
    ``symbol``; a binding's ``extern "C" __global__ void symbol(...);`` declaration does not match."""
    pattern = rf"__global__[^;{{}}]*?\bvoid\s+{re.escape(symbol)}\s*\([^;{{}}]*\)\s*\{{"
    return re.search(pattern, text) is not None


def test_manifest_entries_are_the_registry_programs_on_both_architectures():
    """Every program is built once per architecture: ``build.json`` holds exactly the 222 entries
    ``cake_jit.library_name`` resolves, each naming a delivered kernel translation unit that defines
    the record's kernel symbol and a delivered binding translation unit exporting ``run``, digests
    keyed by exactly those sources, a 64-hex closure digest, string flags without fast math and the
    ``run`` entry."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    assert len(loader.PROGRAMS) == 111
    assert loader.ARCHES == ("sm_100a", "sm_103a")
    expected = {
        loader.library_name(program, arch)
        for program, record in loader.PROGRAMS.items()
        for arch in record["arches"]
    }
    assert set(libraries) == expected and len(libraries) == 2 * len(loader.PROGRAMS)
    texts: dict[str, str] = {}
    for name, entry in libraries.items():
        program = name.rsplit("_sm_", 1)[0]
        record = loader.PROGRAMS[program]
        assert record["library"] == program and record["ffi_entry"] == "run"
        assert entry["arch"] in record["arches"] and name.endswith("_" + entry["arch"])
        assert entry["entry"] == "run"
        kernel_tu, binding_tu = entry["sources"]
        assert kernel_tu.startswith("csrc/cake_stepfun_moe/"), name
        assert kernel_tu.endswith("_kernel.cu"), name
        assert binding_tu.startswith("csrc/cake_stepfun_moe/"), name
        assert binding_tu.endswith("_binding.cu"), name
        assert set(entry["sha256"]) == set(entry["sources"])
        for source in entry["sources"]:
            assert (PACKAGE_DIR / source).is_file(), source
            if source not in texts:
                texts[source] = (PACKAGE_DIR / source).read_text(errors="replace")
        assert _defines_kernel(texts[kernel_tu], record["kernel"]), name
        assert re.search(
            r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*run\s*,", texts[binding_tu]
        ), name
        assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
        assert all(isinstance(flag, str) for flag in entry["nvcc_flags"]), name
        assert "--use_fast_math" not in entry["nvcc_flags"]
        assert not any("fast_math" in flag for flag in entry["nvcc_flags"])
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


def test_stage_tables_and_routes_cover_every_program_once():
    """The five dispatcher tables hold program names only, in the production order and counts (50 /
    26 / 2 / 27 / 4); with the two histogram pre units of the cooperative rows they name every
    program exactly once.  Every program is reachable from a route of the export's measurement
    catalogue (``tests/test_helpers/cake_stepfun_moe_routes.json``) and every route stage names a
    registered program delivered for the route's architectures."""
    assert tuple(loader.STAGE_TABLES) == loader.STAGE_ORDER
    named = Counter()
    for stage, table in loader.STAGE_TABLES.items():
        assert len(table) == STAGE_TABLE_SIZES[stage], stage
        assert loader.programs_of(stage) == table
        for program in table:
            unit = loader.unit(program)
            assert unit["stage"] == stage, program
            named[program] += 1
            if stage == "routing" and unit.get("pre") is not None:
                assert loader.unit(unit["pre"])["leads"] == program
                named[unit["pre"]] += 1
    assert set(named) == set(loader.PROGRAMS)
    assert all(count == 1 for count in named.values()), named.most_common(3)
    with pytest.raises(KeyError):
        loader.programs_of("gemm")
    routes = json.loads(ROUTES_PATH.read_text())
    assert len(routes) == ROUTE_COUNT
    reachable = set()
    for name, route in routes.items():
        assert route["kind"] in ("fc1", "fc2", "routing", "requant", "finalize", "full")
        assert name.startswith(route["kind"] + "_")
        stages = [stage["name"] for stage in route["stages"]]
        assert stages, name
        if route["kind"] == "full":
            per_token = loader.HOST_PLAN_CONSTANTS["families"][route["family"]][
                "per_token"
            ]
            expected = (
                ["routing", "fc1"]
                + (["requant"] if per_token else [])
                + ["fc2", "finalize"]
            )
            assert [s for s in stages if s != "routing_pre"] == expected, name
        for stage in route["stages"]:
            record = loader.PROGRAMS[stage["program"]]
            assert set(route["arches"]) <= set(record["arches"]), (name, stage)
            reachable.add(stage["program"])
    assert reachable == set(loader.PROGRAMS)
    # The routing dispatch: per input kind and logits dtype, the rows cover 1..2^30 tokens in order
    # without a gap.
    for kind, dtype in (
        ("scores", "bfloat16"),
        ("scores", "float32"),
        ("topk_ids", None),
    ):
        rows = [
            loader.unit(p)
            for p in loader.STAGE_TABLES["routing"]
            if loader.unit(p)["input"] == kind
            and loader.unit(p)["logits_dtype"] == dtype
        ]
        assert rows, (kind, dtype)
        assert rows[0]["min_tokens"] == 1
        for previous, row in zip(rows, rows[1:], strict=False):
            assert row["min_tokens"] == previous["max_tokens"] + 1, (
                kind,
                dtype,
                row["variant"],
            )
        assert rows[-1]["max_tokens"] >= 2048


def test_unit_blocks_carry_the_stage_facts():
    """The ``unit`` block of every program carries its ``unit_key`` and the per-stage facts the host
    plan reads (tile and K geometry, block-scale layouts, routing windows, finalize variants) plus
    the storage dtype of every tensor argument and the rank of every tensor-map view."""
    families = loader.HOST_PLAN_CONSTANTS["families"]
    keys = set()
    for program, record in loader.PROGRAMS.items():
        assert tuple(record) == (
            "kernel",
            "library",
            "ffi_entry",
            "arg_plan",
            "launch",
            "arches",
            "unit",
        )
        assert record["kernel"] == loader._KERNEL_OVERRIDES.get(
            program, "kernel_" + program
        )
        assert set(record["launch"]) == {"block", "cluster"}
        assert all(len(record["launch"][k]) == 3 for k in ("block", "cluster"))
        kinds = [kind for kind, _ in record["arg_plan"]]
        assert set(kinds) <= {"tma_buffer", "buffer", "parameter", "launch", "grid"}
        assert record["arg_plan"][-3:] == [
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
        unit = record["unit"]
        key = tuple(unit["unit_key"])
        assert key not in keys, key
        keys.add(key)
        assert key[0] == unit["stage"] and unit["stage"] in loader.STAGE_ORDER
        tensor_keys = {
            name
            for kind, name in record["arg_plan"]
            if kind in ("tma_buffer", "buffer")
        }
        assert set(unit["arg_dtypes"]) == tensor_keys, program
        assert set(unit["tma_ranks"]) == {
            name for kind, name in record["arg_plan"] if kind == "tma_buffer"
        }, program
        if unit["stage"] in ("fc1", "fc2"):
            assert key == (unit["stage"], unit["family"], unit["tile_n"])
            assert unit["family"] in families
            assert unit["family_index"] == families[unit["family"]]["index"]
            assert unit["tile_n"] in loader.HOST_PLAN_CONSTANTS["fi_tiles"]
            assert (
                H % unit["block_k"] == 0
                if unit["stage"] == "fc1"
                else I % unit["block_k"] == 0
            )
            assert unit["per_token"] is families[unit["family"]]["per_token"]
            assert unit["weight_layout"] == families[unit["family"]]["weight_layout"]
            if unit["stage"] == "fc1":
                assert unit["output_rows_per_cta"] == ref.FC1_OUTPUT_ROWS_PER_CTA
            else:
                assert unit["output_rows_per_cta"] == ref.FC2_OUTPUT_ROWS_PER_CTA
                assert unit["split_k"] >= 1 and unit["min_k_tiles"] >= unit["split_k"]
                assert unit["k_tiles_multiple"] in (1, unit["split_k"])
                assert unit["sf_layout_a"] in ("none", "r8c4", "r128c4")
                assert (
                    cb.fc2_activation_sf_layout(unit["family"], unit["tile_n"])
                    == unit["sf_layout_a"]
                )
        elif unit["stage"] == "routing":
            assert key == ("routing", unit["variant"])
            assert unit["input"] in ("scores", "topk_ids")
            assert unit["grid_rule"] in ("fixed", "coop_sms")
            assert 1 <= unit["min_tokens"] <= unit["max_tokens"]
            assert unit["cooperative"] is (unit["kind"] == "coop")
        elif unit["stage"] == "requant":
            assert key == ("requant", unit["variant"]) and unit["variant"] in (
                "r8c4",
                "r128c4",
            )
            assert cb.requant_program(unit["variant"]) == program
            assert unit["e4m3_max"] == 448
        else:
            assert key == ("finalize", unit["variant"])
            assert unit["kernel_variant"] in ("scalar", "vector")
            assert unit["expert_weights_dtype"] in ("bfloat16", "float32")
            assert (
                cb.finalize_program(
                    unit["kernel_variant"], unit["expert_weights_dtype"]
                )
                == program
            )
    assert len(keys) == 111
    for stage, family in (("fc1", "nvfp4"), ("fc2", "bf16")):
        tiles = cb.exported_tiles(stage, family)
        assert tiles == tuple(sorted(tiles)) and tiles
        for tile in tiles:
            program = (
                cb.fc1_program(family, tile)
                if stage == "fc1"
                else cb.fc2_program(family, tile)
            )
            assert tuple(loader.unit(program)["unit_key"]) == (stage, family, tile)
    assert cb.exported_tiles("fc1", "no_such_family") == ()
    with pytest.raises(ValueError, match="no_such_family"):
        cb.fc1_program("no_such_family", 8)
    assert set(cb.routing_inputs()) == {"scores", "topk_ids"}
    assert set(cb.routing_logits_dtypes()) == {"bfloat16", "float32"}
    assert set(cb.finalize_weight_dtypes()) == {"bfloat16", "float32"}


def test_launchers_follow_the_argument_plans():
    """``cake_launch.LAUNCH`` holds one positional launcher per program, derived from the
    program's ``arg_plan``: ``run`` receives ``values[<name>]`` for every non-grid slot in
    plan order, then the grid triple (the plan's trailing three slots)."""
    cake_launch = cake_package("cake_stepfun_moe", "cake_launch")
    assert set(cake_launch.LAUNCH) == set(loader.PROGRAMS)
    for program, record in loader.PROGRAMS.items():
        kinds = [kind for kind, _name in record["arg_plan"]]
        names = [name for _kind, name in record["arg_plan"]]
        assert kinds[-3:] == ["grid"] * 3 and "grid" not in kinds[:-3], program
        assert names[-3:] == ["grid_x", "grid_y", "grid_z"]
        seen = []

        def run(*args, _seen=seen):
            _seen.extend(args)

        values = {name: f"v:{name}" for name in names[:-3]}
        cake_launch.LAUNCH[program](run, values, ("gx", "gy", "gz"))
        assert seen == [f"v:{name}" for name in names[:-3]] + ["gx", "gy", "gz"], (
            program
        )


def test_host_plan_formulas_match_the_reference():
    """The backend's host plan reproduces the reference formulas: ``max_num_ctas`` (trtllm-gen
    ``getMaxNumCtasInBatchDim``), the centre tile, the finalize dispatch and the plan ladder
    (every plan tile exported for both FC1 and FC2; the default plan is the first tile at or above
    the centre tile, else the largest)."""
    for num_tokens in (1, 3, *ref.TOKENS, 4096):
        for tile in loader.HOST_PLAN_CONSTANTS["fi_tiles"]:
            assert cb.max_num_ctas(num_tokens, K, E, tile) == ref.max_num_ctas(
                num_tokens, K, E, tile
            )
            assert cb.max_padded_tokens(
                num_tokens, K, E, tile
            ) == ref.max_padded_tokens(num_tokens, K, E, tile)
        assert cb.route_tile(num_tokens) == ref.route_tile(num_tokens)
        assert cb.finalize_variant(H, num_tokens) == ref.finalize_variant(H, num_tokens)
    for family in ref.FAMILIES:
        fc1, fc2 = cb.exported_tiles("fc1", family), cb.exported_tiles("fc2", family)
        for num_tokens in ref.TOKENS:
            plans = cb.plans_for(family, num_tokens, H, I)
            tiles = [plan.tile_n for plan in plans]
            assert tiles == sorted(tiles) and set(tiles) <= set(fc1) & set(fc2), family
            for plan in plans:
                assert plan.family == family
                assert plan.fc1_program == cb.fc1_program(family, plan.tile_n)
                assert plan.fc2_program == cb.fc2_program(family, plan.tile_n)
                assert plan.sf_layout_a == cb.fc2_activation_sf_layout(
                    family, plan.tile_n
                )
                if family == "nvfp4_bf16tok":
                    assert plan.requant_program == cb.requant_program(plan.sf_layout_a)
                else:
                    assert plan.requant_program is None
            default = cb.default_plan(family, num_tokens, H, I)
            centre = cb.route_tile(num_tokens)
            above = [t for t in tiles if t >= centre]
            assert default.tile_n == (above[0] if above else tiles[-1])
    for num_tokens in ref.TOKENS:
        for kind, dtype in (
            ("scores", "bfloat16"),
            ("scores", "float32"),
            ("topk_ids", None),
        ):
            plan = cb.routing_plan(num_tokens, kind, dtype)
            assert 1 <= len(plan) <= 2
            main = loader.unit(plan[-1])
            assert main["input"] == kind and main["logits_dtype"] == dtype
            assert main["min_tokens"] <= num_tokens <= main["max_tokens"]
            if len(plan) == 2:
                assert main["pre"] == plan[0]
    with pytest.raises(ValueError):
        cb.plans_for("no_such_family", 8, H, I)


def test_package_sources_carry_no_target_minor_and_no_fast_math():
    """No delivered source of the package names the retired per-arch minor macro or enables fast math
    (the loader's ``use_fast_math=False`` keyword is the one spelling that may appear)."""
    checked = 0
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if (
            path.suffix not in (".py", ".cu", ".cuh", ".h", ".json")
            or "aot" in path.parts
        ):
            continue
        text = path.read_text(errors="replace")
        assert RETIRED_MINOR_MACRO not in text, path
        assert "--use_fast_math" not in text and "use_fast_math=True" not in text, path
        checked += 1
    assert checked >= 2 * len(loader.PROGRAMS) + 4


def test_program_rows_expand_over_the_shared_argument_plans():
    """Every ``PROGRAMS`` record is expanded from a compact row: its argument plan is one of the
    shared ``ARG_PLANS`` (20 distinct ``[kind, name]`` sequences, each serving at least one
    program), the tensor-map operands of that plan are exactly the keys of ``unit["tma_ranks"]``,
    and the derived fields hold -- ``library`` is the program name, ``ffi_entry`` is ``run``,
    ``arches`` is ``ARCHES``, ``kernel`` is ``kernel_<program>`` unless ``_KERNEL_OVERRIDES`` names
    the kernel of another registered program, and ``unit_key`` is ``[stage, family, tile_n]`` for
    the GEMM stages and ``[stage, variant]`` otherwise."""
    plans = loader.ARG_PLANS
    assert len(plans) == ARG_PLAN_COUNT
    assert len({json.dumps(plan) for plan in plans}) == len(plans)
    kernels = {"kernel_" + program for program in loader.PROGRAMS}
    for program, kernel in loader._KERNEL_OVERRIDES.items():
        assert program in loader.PROGRAMS and kernel in kernels, program
        assert kernel != "kernel_" + program, program
    used = set()
    for program, record in loader.PROGRAMS.items():
        index = plans.index(record["arg_plan"])
        used.add(index)
        tma_operands = {
            name for kind, name in record["arg_plan"] if kind == "tma_buffer"
        }
        assert set(record["unit"]["tma_ranks"]) == tma_operands, program
        assert record["library"] == program and record["ffi_entry"] == "run"
        assert record["arches"] == list(loader.ARCHES)
        assert record["kernel"] == loader._KERNEL_OVERRIDES.get(
            program, "kernel_" + program
        )
        unit = record["unit"]
        if unit["stage"] in ("fc1", "fc2"):
            assert unit["unit_key"] == [unit["stage"], unit["family"], unit["tile_n"]]
        else:
            assert unit["unit_key"] == [unit["stage"], unit["variant"]]
    assert used == set(range(len(plans)))


def test_folded_programs_launch_the_kernel_of_their_twin():
    """A program whose ``_KERNEL_OVERRIDES`` entry names another program's kernel is a fold of
    twin kernels: both records share the stage, argument plan and launch shape; every
    ``build.json`` entry of the folded program compiles the twin's kernel source next to the
    folded program's own binding; and that binding declares and launches the twin's symbol,
    never one of its own."""
    libraries = cake_build.libraries_by_name()
    for program, kernel in loader._KERNEL_OVERRIDES.items():
        twin = kernel.removeprefix("kernel_")
        assert twin in loader.PROGRAMS and twin != program, program
        record, twin_record = loader.PROGRAMS[program], loader.PROGRAMS[twin]
        assert record["unit"]["stage"] == twin_record["unit"]["stage"], program
        assert record["arg_plan"] == twin_record["arg_plan"], program
        assert record["launch"] == twin_record["launch"], program
        binding_tu = f"csrc/cake_stepfun_moe/{program}_binding.cu"
        for arch in record["arches"]:
            entry = libraries[loader.library_name(program, arch)]
            twin_entry = libraries[loader.library_name(twin, arch)]
            assert entry["sources"][0] == twin_entry["sources"][0], (program, arch)
            assert entry["sources"][1] == binding_tu, (program, arch)
        binding = (PACKAGE_DIR / binding_tu).read_text(errors="replace")
        assert re.search(rf'extern "C" __global__ void {kernel}\(', binding), program
        assert f"reinterpret_cast<const void*>({kernel})" in binding, program
        assert "kernel_" + program not in binding, program


def _content_digest(items) -> str:
    """SHA-256 over ``items`` (``str`` or ``bytes``), each framed by its byte length as eight
    little-endian bytes: the ``closure_sha256`` definition of ``build.json``."""
    digest = hashlib.sha256()
    for item in items:
        data = item if isinstance(item, bytes) else item.encode()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def test_manifest_digests_are_recomputed_from_the_delivered_sources():
    """Every ``build.json`` entry seals its delivery: ``sha256[source]`` is the SHA-256 of the
    delivered bytes and ``closure_sha256`` -- the salt of the entry's JIT library name -- is the
    content digest over [entry name, architecture, each nvcc flag, then each source's
    package-relative path and bytes], so a changed source, flag or source set names a fresh
    build."""
    for name, entry in cake_build.libraries_by_name().items():
        items = [name, entry["arch"], *entry["nvcc_flags"]]
        for source in entry["sources"]:
            data = (PACKAGE_DIR / source).read_bytes()
            assert entry["sha256"][source] == hashlib.sha256(data).hexdigest(), (
                name,
                source,
            )
            items += [source, data]
        assert entry["closure_sha256"] == _content_digest(items), name


# ---------------------------------------------------------------------------------------------
# B2 -- keyword switches, no environment (CPU)
# ---------------------------------------------------------------------------------------------


def test_prepare_entries_take_keyword_switches_and_the_package_reads_no_environment():
    """Every switch of the prepare surface is a keyword (``do_finalize``, ``plan``, ``enable_pdl``,
    ``reserved_sms``, ``global_scale_inv``, ``poison_routing_tail``) with the documented default;
    there is no ``full_path`` keyword; the package reads no environment variable (the
    ``flashinfer.fused_moe`` facade resolves the knobs)."""
    signature = inspect.signature(cb.prepare_stepfun_moe)
    params = signature.parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    assert "full_path" not in params
    assert params["do_finalize"].default is True
    assert params["plan"].default is None
    assert params["enable_pdl"].default is True
    assert params["reserved_sms"].default == 8
    assert params["global_scale_inv"].default == pytest.approx(ref.GLOBAL_SCALE_INV)
    assert params["poison_routing_tail"].default is False
    for name in (
        "family",
        "hidden_states",
        "gemm1_weights",
        "gemm1_clamp_limit",
        "gemm2_weights",
    ):
        assert params[name].default is inspect.Parameter.empty, name
    for entry in (
        cb.prepare_routing,
        cb.prepare_fc1,
        cb.prepare_requant,
        cb.prepare_fc2,
        cb.prepare_finalize,
    ):
        assert all(
            p.kind is inspect.Parameter.KEYWORD_ONLY
            for p in inspect.signature(entry).parameters.values()
        ), entry.__name__
    assert "full_path" not in inspect.signature(cb.prepare_fc1).parameters
    for module in (pkg, cb, loader, cake_package("cake_stepfun_moe", "cake_launch")):
        text = Path(module.__file__).read_text()
        assert "os.environ" not in text, module.__name__
        assert "getenv" not in text, module.__name__
        assert not any(name.endswith("_ENV") for name in vars(module)), module.__name__
    assert "flashinfer.fused_moe" not in Path(cb.__file__).read_text()


# ---------------------------------------------------------------------------------------------
# B3 -- routing vs the host reference
# ---------------------------------------------------------------------------------------------


def _routing_tiles(num_tokens: int) -> list[int]:
    tiles = {cb.route_tile(num_tokens)}
    fc1 = cb.exported_tiles("fc1", "bf16")
    if fc1:
        tiles |= {fc1[0], fc1[-1]}
    return sorted(tiles)


def _run_routing(logits, num_tokens, tile, device, *, topk_ids=None, topk_weights=None):
    buffers = ref.routing_buffers(num_tokens, tile, device)
    kwargs = dict(buffers)
    if topk_ids is not None:
        kwargs["topk_ids"] = topk_ids
        kwargs["topk_weights"] = topk_weights
        kwargs["routing_logits"] = None
    else:
        kwargs["routing_logits"] = logits
    launch = cb.prepare_routing(
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
        tile_tokens_dim=tile,
        **kwargs,
    )
    launch.launch()
    _sync()
    return buffers, launch


@pytest.mark.parametrize("regime", ref.ROUTING_REGIMES)
@pytest.mark.parametrize("num_tokens", ref.TOKENS)
def test_routing_matches_the_host_reference(num_tokens, regime):
    """The Cake router writes the trtllm-gen tables of the host reference: the CTA tables, the padded
    extents and the expert weights (one bf16 ulp) exactly; the permutation maps bitwise for T <= 16
    and canonically above; the ``empty`` regime leaves experts 13 and 41 without a CTA."""
    device, _arch = _require_device()
    logits = ref.routing_logits(num_tokens, regime, device=device)
    for tile in _routing_tiles(num_tokens):
        host = ref.reference_routing(logits, tile=tile)
        buffers, launch = _run_routing(logits, num_tokens, tile, device)
        ours = ref.live_tables(buffers, tile=tile)
        context = f"T={num_tokens} {regime} tile {tile}"
        ref.assert_routing_tables_match(
            ours,
            host,
            num_tokens=num_tokens,
            tile=tile,
            top_k=K,
            context=context,
            weights_ulp=1.0,
        )
        assert torch.equal(
            ref.expert_ids_of(ours, num_tokens=num_tokens, top_k=K), host["topk_ids"]
        ), context
        assert int(buffers["permuted_idx_to_token_idx"][-1]) == -1, (
            f"{context}: slack entry touched"
        )
        if regime == "empty":
            ref.assert_empty_experts(ours, tile=tile, context=context)
            ref.assert_empty_experts(host, tile=tile, context=context + " (host)")
        expected = cb.routing_plan(num_tokens, "scores", "bfloat16")
        assert tuple(launch.programs) == tuple(expected), context


@pytest.mark.parametrize("num_tokens", (8, 64, 2048))
def test_routing_from_float32_logits_matches_the_host_reference(num_tokens):
    device, _arch = _require_device()
    logits = ref.routing_logits(num_tokens, "ragged", device=device).float()
    tile = cb.route_tile(num_tokens)
    host = ref.reference_routing(logits, tile=tile)
    buffers, launch = _run_routing(logits, num_tokens, tile, device)
    ours = ref.live_tables(buffers, tile=tile)
    ref.assert_routing_tables_match(
        ours,
        host,
        num_tokens=num_tokens,
        tile=tile,
        top_k=K,
        context=f"f32 T={num_tokens}",
        weights_ulp=1.0,
    )
    assert tuple(launch.programs) == tuple(
        cb.routing_plan(num_tokens, "scores", "float32")
    )


@pytest.mark.parametrize("num_tokens", ref.TOKENS)
def test_routing_from_precomputed_ids_matches_the_scores_path(num_tokens):
    """The ``topk_ids`` router reproduces the tables of the scores router on the same selection (the
    expert weights are the caller's and pass through untouched)."""
    device, _arch = _require_device()
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    tile = cb.route_tile(num_tokens)
    host = ref.reference_routing(logits, tile=tile)
    scores_buffers, _ = _run_routing(logits, num_tokens, tile, device)
    scores = ref.live_tables(scores_buffers, tile=tile)
    ids = host["topk_ids"].contiguous()
    weights = scores["topk_weights"].clone()
    buffers, launch = _run_routing(
        None, num_tokens, tile, device, topk_ids=ids, topk_weights=weights
    )
    ours = ref.live_tables(buffers, tile=tile)
    ours["expert_weights"] = weights
    ref.assert_routing_tables_match(
        ours,
        scores,
        num_tokens=num_tokens,
        tile=tile,
        top_k=K,
        context=f"ids T={num_tokens}",
    )
    assert torch.equal(ref.expert_ids_of(ours, num_tokens=num_tokens, top_k=K), ids)
    assert tuple(launch.programs) == tuple(
        cb.routing_plan(num_tokens, "topk_ids", None)
    )


# ---------------------------------------------------------------------------------------------
# B4 -- FC1 stage vs the fp32 reference
# ---------------------------------------------------------------------------------------------


def _fc1_cells():
    cells = []
    for family in ref.FAMILIES:
        tiles = cb.exported_tiles("fc1", family)
        counts = "fc1_counts_mxfp8" if family == "mxfp8" else "fc1_counts"
        for tile in tiles:
            cells.append((family, tile, counts, "mixed"))
            cells.append((family, tile, "fc2_counts", "mixed"))
        if tiles:
            for mode in ("7", "0.5", "inf"):
                cells.append((family, tiles[0], "fc2_counts", mode))
            for tile in sorted({tiles[0], tiles[-1]}):
                cells.append((family, tile, "ragged64", "mixed"))
    return cells


@pytest.mark.parametrize("family,tile,fixture,limit_mode", _fc1_cells())
def test_fc1_stage_matches_the_fp32_reference(
    family, tile, fixture, limit_mode, weights_cache
):
    """One FC1 launch on a routing fixture (explicit per-expert counts with empty experts, or the
    host router's 64-expert tables with experts 13 / 41 empty) reproduces the fp32 StepFun
    reference of every live row at the family's tolerance, with the one-ulp block rule of the
    family's output code."""
    device, _arch = _require_device(family)
    routing, num_tokens = _routing_fixture(fixture, tile, device)
    weights = _weights(weights_cache, family, 0, device)
    case = ref.build_case(weights, num_tokens, limit_mode, device=device)
    # The backend sizes ``gemm1_output`` by its geometry (the activation count, top-k, E and the
    # tile), not by the fixture's live padded count.
    max_padded = ref.max_padded_tokens(num_tokens, K, E, tile)
    sf_layout = cb.fc2_activation_sf_layout(family, tile)
    program = cb.fc1_program(family, tile)
    # ``gemm1_output`` carries the unit's declared storage dtype (``arg_dtypes["C"]``: ``uint8``
    # bytes for the fp8 / mxfp8 e4m3 rows and the nvfp4 codes), which the backend checks.
    buffers = ref.fc1_output_buffers(
        family,
        max_padded,
        sf_layout,
        device,
        output_dtype=getattr(torch, loader.unit(program)["arg_dtypes"]["C"]),
    )
    launch = cb.prepare_fc1(
        family=family,
        **case.fc1_view(),
        **_routing_kwargs(routing, stage="fc1"),
        **buffers,
        top_k=K,
        tile_tokens_dim=tile,
    )
    assert tuple(launch.programs) == (program,)
    launch.launch()
    _sync()
    expected = ref.fc1_stage_reference(case, routing, tile=tile)
    assert expected, "the fixture routes no token"
    for expert, (rows, _act, quantized) in expected.items():
        actual = ref.fc1_output_rows(family, buffers, rows, max_padded, sf_layout)
        ref.assert_fc1_rows_close(
            family,
            actual,
            quantized,
            label=f"{family} tile {tile} {fixture} L={limit_mode} expert {expert}",
        )


# ---------------------------------------------------------------------------------------------
# B5 -- requantization stage vs the fp32 reference
# ---------------------------------------------------------------------------------------------


def _requant_layouts():
    return sorted(
        {
            cb.fc2_activation_sf_layout("nvfp4_bf16tok", t)
            for t in cb.exported_tiles("fc2", "nvfp4_bf16tok")
        }
    )


def _requant_tile(sf_layout: str, num_tokens: int) -> int:
    """The plan tile whose FC2 unit reads ``sf_layout`` (the full path pairs the requantization
    with that unit): the centre tile of ``num_tokens`` when its unit does, else the smallest such
    tile.  For this export's units the padded row count of that geometry is a multiple of the
    layout's block rows (``r128c4`` is read by the tile-256 FC2 unit only; ``r8c4`` by tiles that are
    multiples of 8); the caller asserts it."""
    tiles = [
        t
        for t in cb.exported_tiles("fc2", "nvfp4_bf16tok")
        if cb.fc2_activation_sf_layout("nvfp4_bf16tok", t) == sf_layout
    ]
    centre = cb.route_tile(num_tokens)
    return centre if centre in tiles else tiles[0]


@pytest.mark.parametrize("num_tokens,regime", ref.API_REQUANT_CELLS)
@pytest.mark.parametrize("sf_layout", _requant_layouts())
def test_requant_stage_matches_the_fp32_reference(sf_layout, num_tokens, regime):
    """The NVFP4 per-token requantization of bf16 FC1 rows, in the geometry of the plan tile whose
    FC2 unit reads ``sf_layout``, equals the reference mirror of the kernel arithmetic
    (``token_scale = amax / (448 * 6)``, ``SFValue = (1 / token_scale) * (vec_max / 6)``): block
    scales and codes as the dequantized product at 1.0 / 0.1 under the e2m1 one-code rule (a
    one-ulp scale byte only on the reference's e4m3 rounding edges); the per-token scales are
    bitwise; rows no expanded index names stay untouched."""
    device, _arch = _require_device("nvfp4_bf16tok")
    tile = _requant_tile(sf_layout, num_tokens)
    logits = ref.routing_logits(num_tokens, regime, device=device)
    routing = ref.reference_routing(logits, tile=tile)
    # ``prepare_requant`` reads max_padded from ``gemm1_output``: the geometry's row count (a
    # multiple of the layout's block rows); the block scales hold ceil(max_padded / 128) * 128 rows.
    max_padded = ref.max_padded_tokens(num_tokens, K, E, tile)
    assert max_padded % ref.SF_BLOCK_ROWS[sf_layout] == 0, (sf_layout, tile, max_padded)
    expanded = routing["expanded_idx_to_permuted_idx"].reshape(-1).contiguous()
    generator = torch.Generator(device=device).manual_seed(
        ref.ROUTING_SEED + num_tokens
    )
    gemm1_output = torch.randn(
        max_padded, I, device=device, dtype=torch.float32, generator=generator
    )
    gemm1_output = (4.0 * gemm1_output).to(torch.bfloat16)
    output = torch.full((max_padded, I // 2), 0xAB, dtype=torch.uint8, device=device)
    output_scale = torch.full(
        (-(-max_padded // 128) * 128 * (I // ref.NVFP4_SF_VEC),),
        0xAB,
        dtype=torch.uint8,
        device=device,
    )
    per_token_scale = torch.full(
        (max_padded,), -1.0, dtype=torch.float32, device=device
    )
    launch = cb.prepare_requant(
        gemm1_output=gemm1_output,
        expanded_idx_to_permuted_idx=expanded,
        output=output,
        output_scale=output_scale,
        per_token_scale=per_token_scale,
        sf_layout=sf_layout,
    )
    assert tuple(launch.programs) == (cb.requant_program(sf_layout),)
    launch.launch()
    _sync()
    live = expanded[expanded >= 0].long().unique()
    expected = ref.requant_reference(gemm1_output[live])
    sf = ref.unswizzle_sf(output_scale, max_padded, I // ref.NVFP4_SF_VEC, sf_layout)
    record = ref.nvfp4_rows_compare(
        packed_a=output[live],
        sf_a=sf[live],
        packed_e=expected.packed,
        sf_e=expected.scale,
        token_scale_a=per_token_scale[live],
        token_scale_e=expected.token_scale,
        sf_edge_e=expected.sf_edge,
        label=f"requant {sf_layout} T={num_tokens} {regime}",
    )
    assert record["token_scale_bitwise"], record
    untouched = torch.ones(max_padded, dtype=torch.bool, device=device)
    untouched[live] = False
    assert bool((output[untouched] == 0xAB).all()), (
        "rows without an expanded index were written"
    )
    assert bool((per_token_scale[untouched] == -1.0).all())


# ---------------------------------------------------------------------------------------------
# B6 -- FC2 stage vs the fp32 reference
# ---------------------------------------------------------------------------------------------


def _fc2_cells():
    return [
        (family, tile, fixture)
        for family in ref.FAMILIES
        for tile in cb.exported_tiles("fc2", family)
        for fixture in ("fc2_counts", "ragged64")
    ]


@pytest.mark.parametrize("family,tile,fixture", _fc2_cells())
def test_fc2_stage_matches_the_fp32_reference(family, tile, fixture, weights_cache):
    """One FC2 launch on a routing fixture, fed the family's quantized FC1 rows (block scales in the
    unit's activation layout; the per-token family through the requantization reference),
    reproduces the fp32 reference of every live row at the family's tolerance."""
    device, _arch = _require_device(family)
    routing, num_tokens = _routing_fixture(fixture, tile, device)
    weights = _weights(weights_cache, family, 0, device)
    case = ref.build_case(weights, num_tokens, "mixed", device=device)
    # ``gemm2_input`` / ``gemm2_output`` must hold the geometry's row count (``prepare_fc2`` derives
    # it from ``num_tokens``, top-k, the expert rows and the tile), not the live padded count.
    max_padded = ref.max_padded_tokens(num_tokens, K, E, tile)
    sf_layout = cb.fc2_activation_sf_layout(family, tile)
    generator = torch.Generator(device=device).manual_seed(ref.ROUTING_SEED + tile)
    rows = 3.0 * torch.randn(
        max_padded, I, device=device, dtype=torch.float32, generator=generator
    )
    inputs, dequant = ref.fc2_input_from_rows(family, rows, sf_layout)
    gemm2_output = torch.full(
        (max_padded, H), float("nan"), dtype=torch.bfloat16, device=device
    )
    launch = cb.prepare_fc2(
        family=family,
        **inputs,
        **case.fc2_view(),
        **_routing_kwargs(routing, stage="fc2"),
        gemm2_output=gemm2_output,
        num_tokens=num_tokens,
        top_k=K,
        tile_tokens_dim=tile,
    )
    assert tuple(launch.programs) == (cb.fc2_program(family, tile),)
    launch.launch()
    _sync()
    segments = ref.expert_segments(routing, tile=tile)
    assert segments
    for expert, (start, end) in segments.items():
        token_scale = inputs.get("per_token_scale")
        expected = ref.fc2_reference(
            dequant[start:end],
            case.w2_ref(expert),
            scale_c=case.sc2(expert),
            token_scale=None if token_scale is None else token_scale[start:end],
        )
        ref.assert_family_close(
            gemm2_output[start:end],
            expected,
            family,
            label=f"{family} tile {tile} {fixture} expert {expert}",
        )


# ---------------------------------------------------------------------------------------------
# B7 -- finalize vs the host reference
# ---------------------------------------------------------------------------------------------


def _finalize_inputs(num_tokens, device):
    tile = cb.route_tile(num_tokens)
    routing = ref.reference_routing(
        ref.routing_logits(num_tokens, "ragged", device=device), tile=tile
    )
    max_padded = routing["total_num_padded_tokens"]
    generator = torch.Generator(device=device).manual_seed(
        ref.ROUTING_SEED + num_tokens
    )
    gemm2_output = torch.randn(
        max_padded, H, device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    return routing, gemm2_output


def _run_finalize(routing, gemm2_output, expert_weights, device):
    num_tokens = expert_weights.shape[0]
    output = torch.full(
        (num_tokens, H), float("nan"), dtype=torch.bfloat16, device=device
    )
    launch = cb.prepare_finalize(
        gemm2_output=gemm2_output,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=routing["expanded_idx_to_permuted_idx"]
        .reshape(-1)
        .contiguous(),
        total_num_padded_tokens=torch.tensor(
            [routing["total_num_padded_tokens"]], dtype=torch.int32, device=device
        ),
        output=output,
        num_experts=E,
    )
    launch.launch()
    _sync()
    return output, launch


@pytest.mark.parametrize("num_tokens", ref.TOKENS)
def test_finalize_matches_the_host_reference(num_tokens):
    """The Cake finalize reduces the eight routed rows of every token in fp32 like the reference
    (``TOL["bf16"]``), with bf16 and fp32 expert weights, through the variant of the native
    dispatch rule (scalar below 1184 CTAs, vector otherwise)."""
    device, _arch = _require_device()
    routing, gemm2_output = _finalize_inputs(num_tokens, device)
    expected = ref.finalize_reference(
        gemm2_output, routing["topk_weights"], routing["expanded_idx_to_permuted_idx"]
    )
    variant = ref.finalize_variant(H, num_tokens)
    for dtype in (torch.bfloat16, torch.float32):
        weights = routing["topk_weights"].to(dtype).contiguous()
        output, launch = _run_finalize(routing, gemm2_output, weights, device)
        assert not torch.isnan(output).any()
        ref.assert_family_close(
            output, expected, "bf16", label=f"finalize T={num_tokens} {dtype}"
        )
        dtype_name = {torch.bfloat16: "bfloat16", torch.float32: "float32"}[dtype]
        assert tuple(launch.programs) == (cb.finalize_program(variant, dtype_name),)


@pytest.mark.parametrize("num_tokens", ref.TOKENS)
def test_finalize_weight_dtypes_agree_bitwise(num_tokens):
    """fp32 expert weights holding the same bf16 values reduce to the same bytes (both variants
    accumulate in fp32)."""
    device, _arch = _require_device()
    routing, gemm2_output = _finalize_inputs(num_tokens, device)
    bf16, _ = _run_finalize(
        routing, gemm2_output, routing["topk_weights"].contiguous(), device
    )
    f32, _ = _run_finalize(
        routing, gemm2_output, routing["topk_weights"].float().contiguous(), device
    )
    assert torch.equal(bf16, f32)


# ---------------------------------------------------------------------------------------------
# B8 -- the whole graph vs the fp32 reference
# ---------------------------------------------------------------------------------------------


def _graph_cells():
    cells = [
        (ref.PRECISION_FAMILY[p], t, mode, regime)
        for p in PRECISIONS
        for t in ref.TOKENS
        for mode, regime in ref.BACKEND_GRAPH_CELLS
    ]
    cells += [
        ("nvfp4_bf16tok", t, mode, regime)
        for t in ref.TOKENS
        for mode, regime in (("mixed", "ragged"), ("inf", "empty"))
    ]
    return cells


@pytest.mark.parametrize("family,num_tokens,limit_mode,regime", _graph_cells())
def test_whole_graph_matches_the_fp32_reference(
    family, num_tokens, limit_mode, regime, weights_cache
):
    """``prepare_stepfun_moe(...).launch()`` on every plan of the token count reproduces the fp32
    reference of the whole graph -- routing (the kernel's tables checked against the host router),
    FC1 with the family's output quantization, (requant), FC2, finalize -- hop by hop at the
    family's tolerance: each stage's kernel rows against the reference of that stage fed with the
    kernel's own upstream rows (``ref.graph_hops``), because the stages round independently and
    an end-to-end comparison fails on carried row-ulp flips without any stage being wrong; the
    end-to-end error against the graph reference is recorded in the assertion context."""
    device, _arch = _require_device(family)
    weights = _weights(weights_cache, family, 0, device)
    case = ref.build_case(weights, num_tokens, limit_mode, device=device)
    logits = ref.routing_logits(num_tokens, regime, device=device)
    plans = cb.plans_for(family, num_tokens, H, I)
    assert plans
    default = cb.default_plan(family, num_tokens, H, I)
    assert default in plans
    chosen = plans if (limit_mode, regime) == ("mixed", "ragged") else (default,)
    for plan in chosen:
        launch = _prepare_graph(case, logits, plan=plan)
        assert launch.plan == plan and launch.do_finalize is True
        launch.launch()
        _sync()
        (output,) = launch.outputs()
        assert output.shape == (num_tokens, H) and output.dtype == torch.bfloat16
        tables = ref.live_tables(launch.buffers, tile=plan.tile_n)
        host = ref.reference_routing(logits, tile=plan.tile_n)
        context = f"{family} T={num_tokens} L={limit_mode} {regime} tile {plan.tile_n}"
        ref.assert_routing_tables_match(
            tables,
            host,
            num_tokens=num_tokens,
            tile=plan.tile_n,
            top_k=K,
            context=context,
            weights_ulp=1.0,
        )
        if regime == "empty":
            ref.assert_empty_experts(tables, tile=plan.tile_n, context=context)
        expected = ref.graph_reference(case, tables, tile=plan.tile_n)
        e2e = ref.end_to_end_record(output, expected["output"], family)
        e2e_note = (
            f"end to end vs the graph reference: {e2e['n_outside']}/{e2e['numel']} "
            f"outside, max_abs {e2e['max_abs']:.4g}, max_rel {e2e['max_rel']:.4g}"
        )
        hops = ref.graph_hops(
            case,
            launch.buffers,
            tables,
            tile=plan.tile_n,
            sf_layout=plan.sf_layout_a,
            output=output,
        )
        records = ref.assert_graph_hops_close(hops, label=f"{context} ({e2e_note})")
        print(
            f"{context}: "
            + "; ".join(
                f"hop {name} {rec['n_outside']}/{rec['numel']} outside "
                f"(max_abs {rec['max_abs']:.4g}, flips {rec['one_ulp_block_flips']})"
                for name, rec in records.items()
            )
            + f"; {e2e_note}"
        )
    # ``plan=None`` runs the default plan; an int names the tile.
    launch = _prepare_graph(case, logits)
    assert launch.plan == default
    launch = _prepare_graph(case, logits, plan=default.tile_n)
    assert launch.plan == default


@pytest.mark.parametrize("precision", PRECISIONS)
def test_unfinalized_outputs_follow_the_native_layout(precision, weights_cache):
    """``do_finalize=False`` returns ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``:
    the permuted FC2 rows (``max(max_padded, ceil(128 * 1024 * 8 / (16 H)))`` rows as the native
    runner sizes them), the routing weights and the flat ``[T * top_k]`` map; the live rows pass
    the hop-wise gate (FC1 rows, (requant), FC2 rows on the kernel's own upstream rows,
    ``ref.graph_hops``: the stages round independently, so the FC2 rows are not gated end to end)
    and the finalized output of the same inputs equals the reference finalize of the returned
    rows, weights and map at the family's tolerance."""
    family = ref.PRECISION_FAMILY[precision]
    device, _arch = _require_device(family)
    num_tokens = 64
    weights = _weights(weights_cache, family, 0, device)
    case = ref.build_case(weights, num_tokens, "mixed", device=device)
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    launch = _prepare_graph(case, logits, do_finalize=False)
    assert launch.do_finalize is False and launch.finalize_program is None
    launch.launch()
    _sync()
    gemm2_output, expert_weights, expanded = launch.outputs()
    tile = launch.plan.tile_n
    max_padded = ref.max_padded_tokens(num_tokens, K, E, tile)
    assert gemm2_output.shape == (ref.gemm2_output_rows(max_padded, H), H)
    assert gemm2_output.dtype == torch.bfloat16
    assert (
        expert_weights.shape == (num_tokens, K)
        and expert_weights.dtype == torch.bfloat16
    )
    assert expanded.shape == (num_tokens * K,) and expanded.dtype == torch.int32
    tables = ref.live_tables(launch.buffers, tile=tile)
    live = expanded[expanded >= 0].long()
    assert torch.equal(gemm2_output[live], launch.buffers["gemm2_output"][live]), (
        "the returned rows are not the launch's gemm2_output rows"
    )
    expected = ref.graph_reference(case, tables, tile=tile)
    e2e = ref.end_to_end_record(
        gemm2_output[live], expected["gemm2_output"][live], family
    )
    hops = ref.graph_hops(
        case, launch.buffers, tables, tile=tile, sf_layout=launch.plan.sf_layout_a
    )
    ref.assert_graph_hops_close(
        hops,
        label=(
            f"{family} permuted FC2 rows (end to end vs the graph reference: "
            f"{e2e['n_outside']}/{e2e['numel']} outside, max_abs {e2e['max_abs']:.4g})"
        ),
    )
    # The same inputs finalized by the package (default plan again) equal the reference finalize
    # of the returned rows with the returned weights and map (hop 3 across the two launches).
    finalizing = _prepare_graph(case, logits)
    assert finalizing.plan == launch.plan and finalizing.do_finalize is True
    finalizing.launch()
    _sync()
    (output,) = finalizing.outputs()
    same_rows = torch.equal(
        finalizing.buffers["gemm2_output"][live], gemm2_output[live]
    )
    finalized = ref.finalize_reference(
        gemm2_output[:max_padded], expert_weights, expanded
    )
    ref.assert_family_close(
        output,
        finalized,
        family,
        label=f"{family} finalize of the rows (FC2 rows of both launches bitwise: {same_rows})",
    )


# ---------------------------------------------------------------------------------------------
# B9 -- CUDA-graph replay
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", (8, 512))
@pytest.mark.parametrize("precision", PRECISIONS)
def test_cuda_graph_replay_equals_eager(precision, num_tokens, weights_cache):
    """A prepared launch captured in a CUDA graph (after one eager launch) replays the eager bytes,
    on the first and on the second replay."""
    family = ref.PRECISION_FAMILY[precision]
    device, _arch = _require_device(family)
    weights = _weights(weights_cache, family, 0, device)
    case = ref.build_case(weights, num_tokens, "7", device=device)
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    launch = _prepare_graph(case, logits)
    outputs = launch.outputs()
    graph, eager = _capture(launch, outputs)
    for replay in (1, 2):
        for tensor in outputs:
            tensor.zero_()
        graph.replay()
        _sync()
        for tensor, expected in zip(outputs, eager, strict=True):
            assert torch.equal(tensor, expected), (
                f"{precision} T={num_tokens}: replay {replay} differs"
            )


# ---------------------------------------------------------------------------------------------
# B10 -- only Cake kernels
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", ref.TOKENS)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_whole_graph_launches_only_cake_kernels(precision, num_tokens, weights_cache):
    """One ``launch()`` runs exactly the Cake kernels of its stages -- the routing kernel(s) of the
    token count, the FC1 unit of the plan's tile, (requant), the FC2 unit and the finalize variant
    -- each once, and nothing else."""
    family = ref.PRECISION_FAMILY[precision]
    device, _arch = _require_device(family)
    weights = _weights(weights_cache, family, 0, device)
    case = ref.build_case(weights, num_tokens, "mixed", device=device)
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    launch = _prepare_graph(case, logits)
    launch.launch()
    _sync()
    names = _profiled_kernel_names(launch.launch)
    assert names, "the profiler captured no kernel"
    offenders = [name for name in names if CAKE_KERNEL_SYMBOL.search(name) is None]
    assert not offenders, (
        f"non-Cake kernels in a Cake forward: {sorted(set(offenders))}"
    )
    symbols = [CAKE_KERNEL_SYMBOL.search(name).group(0) for name in names]
    assert Counter(symbols) == Counter(_plan_kernels(launch)), (
        symbols,
        _plan_kernels(launch),
    )
    assert set(launch.stage_launchers) >= {"routing", "fc1", "fc2", "finalize"}


# ---------------------------------------------------------------------------------------------
# B11 -- first graph replay in fresh processes
# ---------------------------------------------------------------------------------------------


def _first_replay_probe(family: str, num_tokens: int) -> dict:
    """Fresh-process probe: eager launch, graph capture, one replay; bitwise agreement."""
    device = torch.device("cuda", torch.cuda.current_device())
    weights = ref.build_weights(family, 0, device)
    case = ref.build_case(weights, num_tokens, "7", device=device)
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    launch = _prepare_graph(case, logits)
    outputs = launch.outputs()
    graph, eager = _capture(launch, outputs)
    for tensor in outputs:
        tensor.zero_()
    graph.replay()
    _sync()
    return {
        "plan": launch.plan.tile_n,
        "bitwise": all(torch.equal(t, e) for t, e in zip(outputs, eager, strict=True)),
        "max_abs_diff": max(
            float((t.float() - e.float()).abs().max().item())
            for t, e in zip(outputs, eager, strict=True)
        ),
    }


_PROBE_SCRIPT = """
import json, sys
sys.path.insert(0, {root!r})
import torch
from tests.experimental import test_cake_stepfun_moe as t
results = dict()
for family in {families!r}:
    results[family] = t._first_replay_probe(family, {num_tokens})
print("FIRST_REPLAY_RESULT " + json.dumps(results), flush=True)
"""


def test_first_graph_replay_in_fresh_processes():
    """The first CUDA-graph replay of a fresh process matches eager for every family.  Each
    subprocess prepares the T=8 case of every family, runs eager, captures a graph with PDL enabled
    and replays once.  ``CAKE_STEPFUN_GRAPH_STRESS_PROCESSES`` sets the number of processes
    (default 4); every process is a fresh module load."""
    _require_device(*ref.FAMILIES)
    processes = int(os.environ.get("CAKE_STEPFUN_GRAPH_STRESS_PROCESSES", "4"))
    root = str(Path(__file__).resolve().parents[2])
    script = _PROBE_SCRIPT.format(root=root, families=list(ref.FAMILIES), num_tokens=8)
    env = dict(
        os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    failures = []
    checked = 0
    for index in range(processes):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        marker = [
            line
            for line in proc.stdout.splitlines()
            if line.startswith("FIRST_REPLAY_RESULT ")
        ]
        assert proc.returncode == 0 and marker, (
            f"probe process {index} failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
        results = json.loads(marker[-1][len("FIRST_REPLAY_RESULT ") :])
        for key, result in results.items():
            checked += 1
            if not result["bitwise"]:
                failures.append((index, key, result))
    assert not failures, (
        f"{len(failures)}/{checked} first replays differ from eager: {failures}"
    )
    assert checked == processes * len(ref.FAMILIES)
