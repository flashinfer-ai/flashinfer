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
"""

import math

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

# Backend suite: the package is named once, through ``cake_package``; nothing else
# of flashinfer is imported here.  The tests of the stable flashinfer entry point
# ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` live in
# ``test_cake_moe_grouped_gemm_api.py``.
pkg = cake_package("cake_moe_grouped_gemm")
cb = cake_package("cake_moe_grouped_gemm", "cake_backend")
cake_jit = cake_package("cake_moe_grouped_gemm", "cake_jit")
grouped_gemm_dgrad = pkg.grouped_gemm_dgrad
grouped_gemm_fwd = pkg.grouped_gemm_fwd
grouped_gemm_wgrad = pkg.grouped_gemm_wgrad
SUPPORTED_COMPUTE_CAPABILITIES = cb.SUPPORTED_COMPUTE_CAPABILITIES
prepare_grouped_gemm_dgrad = cb.prepare_grouped_gemm_dgrad
prepare_grouped_gemm_fwd = cb.prepare_grouped_gemm_fwd
prepare_grouped_gemm_wgrad = cb.prepare_grouped_gemm_wgrad
HOST_PLAN_CONSTANTS = cake_jit.HOST_PLAN_CONSTANTS
PROGRAMS = cake_jit.PROGRAMS
ROUTES = cake_jit.ROUTES

ATOL = RTOL = 1e-2  # bf16 outputs
FP32_REL_L2 = 1e-4  # fp32 weight gradient vs the fp64 reference

# Group size lists: an empty group, a one-row group, sizes that are not tile
# multiples, and a long-reduction case that can select the 512 k tile.
RAGGED_E4 = [300, 0, 1, 700]
RAGGED_E6 = [1, 257, 511, 4099, 0, 33]
TINY_E5 = [1, 2, 3, 0, 4]
LONG_E3 = [6145, 0, 12289]
# Groups shorter than one 64-row k-step: every output tile is a single k-step, and
# a CTA switches group at almost every tile (the shape of flashinfer-ai/flashinfer#6570).
SMALL_E512 = [32] * 512
SMALL_GROUP_LAUNCHES = 200
# Two ragged groups of more than every architecture's ``k512_min_rows_per_group``
# rows on average: the weight gradient takes the 512-wide k tile on the
# architectures of ``k512_archs``.
K512_E2 = [8200, 8333]
# Two ragged groups of at least ``dgrad_evict_first_min_rows`` rows on average: with
# ``K >= dgrad_evict_first_min_k`` the data gradient takes the evict-first twin of its
# wide tile on the architectures of ``dgrad_evict_first_archs``.
EVICT_FIRST_E2 = [16390, 16500]
# The MoE expert rows the evict-first and k-tile thresholds were measured on:
# ``EXPERT_ROWS`` routed rows split over each group count of ``EXPERT_GROUPS``.
EXPERT_ROWS = 131072
EXPERT_GROUPS = (4, 8, 16, 32)


def _require_program(op, out_dtype=None):
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    if not cb.generated_program_available(device, op, out_dtype):
        pytest.skip(
            f"no generated ragged grouped GEMM program for {op} registered for {arch}"
        )
    return device


def _offs(sizes, device):
    return torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )


def _inputs(sizes, n, k, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    sum_m = sum(sizes)
    x = torch.randn(sum_m, k, generator=gen, device=device, dtype=torch.float32)
    w = torch.randn(len(sizes), n, k, generator=gen, device=device, dtype=torch.float32)
    g = torch.randn(sum_m, n, generator=gen, device=device, dtype=torch.float32)
    x = (x / math.sqrt(k)).to(torch.bfloat16)
    w = (w / math.sqrt(k)).to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    return x, w, g, _offs(sizes, device)


def _bounds(sizes):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return list(zip([0] + ends[:-1], ends, strict=True))


def _reference(op, a, b, sizes):
    """Per-group fp64 reference in fp32: fwd (x, w), dgrad (g, w), wgrad (g, x)."""
    if op == "fwd":
        out = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32, device=a.device)
        for e, (s, t) in enumerate(_bounds(sizes)):
            if t > s:
                out[s:t] = (a[s:t].double() @ b[e].double().T).float()
    elif op == "dgrad":
        out = torch.zeros(a.shape[0], b.shape[2], dtype=torch.float32, device=a.device)
        for e, (s, t) in enumerate(_bounds(sizes)):
            if t > s:
                out[s:t] = (a[s:t].double() @ b[e].double()).float()
    else:
        out = torch.zeros(
            len(sizes), a.shape[1], b.shape[1], dtype=torch.float32, device=a.device
        )
        for e, (s, t) in enumerate(_bounds(sizes)):
            if t > s:
                out[e] = (a[s:t].double().T @ b[s:t].double()).float()
    return out


def _assert_bf16_close(out, ref):
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Host plan (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sm_count", [148, 160, 212])
@pytest.mark.parametrize(
    "sizes", [RAGGED_E4, RAGGED_E6, LONG_E3, [131072] * 1, [4096] * 32]
)
def test_plan_invariants(sm_count, sizes):
    c = HOST_PLAN_CONSTANTS
    sum_m, e = sum(sizes), len(sizes)
    cta_group = int(c["cta_group"])
    for op, n, k in (("fwd", 4096, 6144), ("dgrad", 4096, 6144)):
        plan = cb.plan_fwd_dgrad(
            op, sum_m=sum_m, num_groups=e, n=n, k=k, sm_count=sm_count, constants=c
        )
        grid_x = plan["grid"][0]
        assert (
            grid_x % cta_group == 0
            and cta_group <= grid_x <= (sm_count // cta_group) * cta_group
        )
        assert plan["launches"] == 1 and plan["tile_k"] is None
        # The two claim queues: c0 = clusters // 2, c1 = the rest.
        assert plan["clusters"] == grid_x // cta_group
        assert plan["c0"] == plan["clusters"] // 2
        assert plan["c0"] + plan["c1"] == plan["clusters"] and plan["c1"] >= 1
        assert plan["tile_n"] == int(c["block_n"])
        # The wide tile bounds the clusters by its 512-column tiles.
        wide = cb.plan_fwd_dgrad(
            op,
            sum_m=sum_m,
            num_groups=e,
            n=n,
            k=k,
            sm_count=sm_count,
            tile_n=int(c["wide_block_n"]),
            constants=c,
        )
        tiles = (sum_m // int(c["cluster_m"]) + e) * (
            (n if op == "fwd" else k) // int(c["wide_block_n"])
        )
        assert wide["tile_n"] == int(c["wide_block_n"])
        assert wide["clusters"] == min(sm_count // cta_group, max(1, tiles))
        assert wide["grid"][0] == wide["clusters"] * cta_group
        assert wide["c0"] == wide["clusters"] // 2
        assert wide["c0"] + wide["c1"] == wide["clusters"]
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for n, k in ((4096, 6144), (6144, 2048), (256, 256)):
            tile_k = cb.select_wgrad_tile(k, arch, sum_m, e, c)
            assert tile_k in (256, 512) and k % tile_k == 0
            if tile_k == 512:
                assert (
                    arch in c["k512_archs"]
                    and sum_m / e >= c["k512_min_rows_per_group"][arch]
                )
            plan = cb.plan_wgrad(
                sum_m=sum_m,
                num_groups=e,
                n=n,
                k=k,
                tile_k=tile_k,
                sm_count=sm_count,
                reduce_threads=256,
                constants=c,
            )
            assert plan["grid"][0] == plan["clusters"] * cta_group
            assert 0 <= plan["num_tail"] < plan["clusters"]
            assert 1 <= plan["tail_splits"] <= int(c["tail_plan"]["max_splits"])
            assert plan["raster_rows"] in (0, 1)
            assert plan["launches"] == 1 + int(plan["tail_splits"] > 1)
            assert (plan["tail_reduce_grid"] is None) == (plan["tail_splits"] == 1)
            if plan["tail_reduce_grid"] is not None:
                assert plan["tail_reduce_grid"][0] > 0
            if plan["num_tail"] == 0:
                assert plan["tail_splits"] == 1
            assert plan["tail_tile_k"] == tile_k
            if tile_k != 512:
                continue
            # The k256-tail stage: the 512-wide tile's grid, n-major, and the tail of
            # the 256-wide tiling of the same clusters, split as that tile's own.
            half = cb.K256_TAIL_TILE_K
            k256t = cb.plan_wgrad(
                sum_m=sum_m,
                num_groups=e,
                n=n,
                k=k,
                tile_k=tile_k,
                sm_count=sm_count,
                reduce_threads=256,
                k256_tail=True,
                constants=c,
            )
            assert k256t["grid"] == plan["grid"] and k256t["tile_k"] == 512
            assert k256t["tail_tile_k"] == half and k256t["raster_rows"] == 0
            clusters = k256t["clusters"]
            assert k256t["num_tail"] == (
                (e * (n // int(c["block_n"])) * (k // half)) % clusters
                if int(c["wgrad_tail_split"])
                else 0
            )
            assert k256t["tail_splits"] == (
                cb.wgrad_tail_plan(
                    k256t["num_tail"],
                    clusters,
                    sum_m,
                    e,
                    half,
                    block_k=int(c["block_k"]),
                    cluster_m=int(c["cluster_m"]),
                    max_splits=int(c["tail_plan"]["max_splits"]),
                    sm_clock_ghz=float(c["tail_plan"]["sm_clock_ghz"]),
                    dram_tbps=float(c["tail_plan"]["dram_tbps"]),
                    launch_us=float(c["tail_plan"]["launch_us"]),
                    cycles_per_step=c["tail_plan"]["cycles_per_step"],
                )
            )
            assert k256t["launches"] == 1 + int(k256t["tail_splits"] > 1)
            # One reduce CTA per tail tile and row block; a thread sums epi_chunk columns.
            rows_per_cta = 256 // (half // int(c["epi_chunk"]))
            assert k256t["tail_reduce_grid"] == (
                [k256t["num_tail"] * (int(c["cluster_m"]) // rows_per_cta), 1, 1]
                if k256t["tail_splits"] > 1
                else None
            )


def test_stage_bindings_are_unique_per_architecture():
    """A route may spell two stages of one name; they serve disjoint architectures,
    and ``stage_binding`` returns the one built for the asked architecture."""
    for name, route in ROUTES.items():
        for arch in route["arches"]:
            served = {}
            for stage in route["stages"]:
                if arch in PROGRAMS[stage["program"]]["arches"]:
                    assert stage["name"] not in served, (name, arch, stage["name"])
                    served[stage["name"]] = stage
            assert "main" in served or "main_wave" in served, (name, arch)
            for stage_name, stage in served.items():
                assert cake_jit.stage_binding(name, stage_name, arch) is stage
                assert cake_jit.stage_served(name, stage_name, arch)
            assert not cake_jit.stage_served(name, "no_such_stage", arch)
            with pytest.raises(KeyError):
                cake_jit.stage_binding(name, "no_such_stage", arch)


def test_program_choice_follows_the_host_plan_constants():
    """``wide_fwd_archs`` / ``wide_dgrad_archs`` select the 512-column tile of
    ``fwd`` / ``dgrad`` when the output columns are a multiple of
    ``wide_block_n``; ``dgrad_evict_first_archs`` the evict-first twin of the
    ``dgrad`` wide tile (``main_wide_ef``) where ``dgrad_evict_first`` holds;
    ``wave_sync_archs`` the wave build of each k tile from ``wave_sync_min_k``,
    launched with the architecture's ``wave_sync_mode``; ``k512_release_archs``
    the twin releasing the 512-wide accumulator in halves; ``k256_tail_archs``
    its twin running the leftover tiles as 256-column half tiles
    (``main_halves_k256t``) where ``wgrad_k256_tail`` holds.  Each listed
    architecture's routes serve the stage the constant selects."""
    c = HOST_PLAN_CONSTANTS
    block_n, wide_n = int(c["block_n"]), int(c["wide_block_n"])
    assert wide_n == 2 * block_n
    for key in ("wide_fwd_archs", "wide_dgrad_archs", "k512_archs"):
        assert set(c[key]) <= set(cake_jit.ARCHES), key
    assert set(c["dgrad_evict_first_archs"]) <= set(c["wide_dgrad_archs"])
    assert int(c["dgrad_evict_first_min_rows"]) > 0
    assert int(c["dgrad_evict_first_min_k"]) % wide_n == 0
    assert set(c["fwd_lwh_ef_archs"]) <= set(c["wide_fwd_archs"])
    assert int(c["fwd_lwh_ef_max_k"]) > 0
    assert set(c["k512_min_rows_per_group"]) == set(c["k512_archs"])
    assert all(int(v) > 0 for v in c["k512_min_rows_per_group"].values())
    assert set(c["k512_release_archs"]) <= set(c["k512_archs"])
    assert set(c["k256_tail_archs"]) <= set(c["k512_release_archs"])
    assert int(c["k256_tail_max_k"]) % 512 == 0 and int(c["k256_tail_max_k"]) > 0
    assert int(c["k256_tail_min_rows"]) > 0
    assert set(c["wave_sync_archs"]) == {256, 512}
    for archs in c["wave_sync_archs"].values():
        assert set(archs) <= set(cake_jit.ARCHES)
        assert all(int(c["wave_sync_mode"][arch]) > 0 for arch in archs)
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        wide_archs = c[f"wide_{op}_archs"]
        for arch in ROUTES[name]["arches"]:
            served = cake_jit.stage_served(name, "main_wide", arch)
            if arch in wide_archs:
                assert served, (name, arch)
            served_ef = cake_jit.stage_served(name, "main_wide_ef", arch)
            ef_key = "dgrad_evict_first_archs" if op == "dgrad" else "fwd_lwh_ef_archs"
            if arch in c[ef_key]:
                assert served_ef, (name, arch)
            for cols in (4096, 4096 - block_n, block_n):
                n, k = (cols, 512) if op == "fwd" else (512, cols)
                for sum_m, groups in ((4096, 4), (EXPERT_ROWS, 4)):
                    stage, tile_n = cb.select_row_stage(
                        name, op, n, k, arch, c, sum_m=sum_m, num_groups=groups
                    )
                    wide = cols % wide_n == 0 and arch in wide_archs and served
                    ef = (
                        wide
                        and served_ef
                        and (
                            cb.dgrad_evict_first(k, arch, sum_m, groups, c)
                            if op == "dgrad"
                            else cb.fwd_lwh_ef(k, arch, c)
                        )
                    )
                    expected = "main_wide_ef" if ef else "main_wide"
                    assert (stage, tile_n) == (
                        (expected, wide_n) if wide else ("main", block_n)
                    ), (op, arch, cols, sum_m)
                assert cb.fwd_dgrad_tile_n(op, n, k, arch, c) == (
                    wide_n if cols % wide_n == 0 and arch in wide_archs else block_n
                )
    min_k = int(c["wave_sync_min_k"])
    for out_dtype in ("bfloat16", "float32"):
        for tile_k in (256, 512):
            name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=tile_k)
            assert name in ROUTES, name
            for arch in ROUTES[name]["arches"]:
                if tile_k == 512 and arch in c["k512_archs"]:
                    assert cake_jit.stage_served(name, "main", arch), (name, arch)
                if arch in c["wave_sync_archs"][tile_k]:
                    assert cake_jit.stage_served(name, "main_wave", arch), (name, arch)
                if tile_k == 512 and arch in c["k256_tail_archs"]:
                    assert cake_jit.stage_served(name, "main_halves_k256t", arch)
                groups = 4
                for k, sum_m in (
                    (min_k, 4096),
                    (min_k, EXPERT_ROWS),
                    (min_k // 2, 4096),
                    (min_k // 2, EXPERT_ROWS),
                ):
                    stage, wave = cb.select_wgrad_stage(
                        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=groups
                    )
                    expect_wave = (
                        k >= min_k
                        and arch in c["wave_sync_archs"][tile_k]
                        and cake_jit.stage_served(name, "main_wave", arch)
                    )
                    base = "main_wave" if expect_wave else "main"
                    halves = (
                        tile_k == 512
                        and arch in c["k512_release_archs"]
                        and cake_jit.stage_served(name, f"{base}_halves", arch)
                    )
                    if tile_k == 512 and arch in c["k512_release_archs"]:
                        assert halves, (name, arch, base)
                    k256t = (
                        halves
                        and base == "main"
                        and cb.wgrad_k256_tail(k, arch, sum_m, groups, c)
                        and cake_jit.stage_served(name, "main_halves_k256t", arch)
                    )
                    expected = f"{base}_halves" if halves else base
                    assert wave == expect_wave, (name, arch, k)
                    assert stage == ("main_halves_k256t" if k256t else expected), (
                        name,
                        arch,
                        k,
                        sum_m,
                    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_expert_rows_take_the_measured_dgrad_stage_and_k_tile(arch):
    """Over ``EXPERT_ROWS`` routed rows: the data gradient of the gate/up projection
    (``K`` 6144) takes the evict-first wide tile on sm_107a at 4 and 8 groups and the
    plain wide tile at 16 and 32; the down projection (``K`` 2048) and the other
    architectures never take it.  The weight gradient's 512-wide k tile starts at 4096
    rows per group (32 groups) on every architecture."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("dgrad")
    for groups in EXPERT_GROUPS:
        for n, k in ((4096, 6144), (6144, 2048)):
            ef = arch == "sm_107a" and k == 6144 and groups in (4, 8)
            assert cb.dgrad_evict_first(k, arch, EXPERT_ROWS, groups, c) == ef
            stage, tile_n = cb.select_row_stage(
                name, "dgrad", n, k, arch, c, sum_m=EXPERT_ROWS, num_groups=groups
            )
            assert tile_n == int(c["wide_block_n"]), (arch, groups, k)
            assert stage == ("main_wide_ef" if ef else "main_wide"), (arch, groups, k)
        k512 = EXPERT_ROWS // groups >= 4096
        for k in (2048, 6144):
            tile_k = cb.select_wgrad_tile(k, arch, EXPERT_ROWS, groups, c)
            assert tile_k == (512 if k512 else 256), (arch, groups, k)


@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_expert_rows_take_the_measured_wgrad_stage(arch, out_dtype):
    """Over ``EXPERT_ROWS`` routed rows the weight gradient ``dW[e]`` (``N`` x ``K``)
    takes the 512-wide k tile released in halves.  On sm_107a the down projection
    (``N`` 6144, ``K`` 2048) takes its k256-tail twin (``main_halves_k256t``) at 4, 8
    and 16 groups; at 32 groups, and for the gate/up projection (``N`` 4096, ``K``
    6144) at every group count, it keeps ``main_halves``.  The other architectures
    never take the twin."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=512)
    for groups in EXPERT_GROUPS:
        for k in (2048, 6144):
            assert cb.select_wgrad_tile(k, arch, EXPERT_ROWS, groups, c) == 512
            k256t = arch == "sm_107a" and k == 2048 and groups in (4, 8, 16)
            assert cb.wgrad_k256_tail(k, arch, EXPERT_ROWS, groups, c) == k256t
            stage, wave = cb.select_wgrad_stage(
                name, k, 512, arch, c, sum_m=EXPERT_ROWS, num_groups=groups
            )
            base = "main_wave" if wave else "main"
            assert stage == ("main_halves_k256t" if k256t else f"{base}_halves"), (
                arch,
                groups,
                k,
            )


def test_workspace_arguments_of_the_argument_plans():
    """``sched`` is followed by the queue split ``c0``, ``c1``; ``wave_sync`` by
    ``wave_mode``; the ``main_wave`` stages bind programs taking ``wave_sync``."""
    for program in PROGRAMS.values():
        plan = [tuple(item) for item in program["arg_plan"]]
        for buffer, parameters in (
            ("sched", [("parameter", "c0"), ("parameter", "c1")]),
            ("wave_sync", [("parameter", "wave_mode")]),
        ):
            if ("buffer", buffer) in plan:
                at = plan.index(("buffer", buffer))
                assert plan[at + 1 : at + 1 + len(parameters)] == parameters
    for route in ROUTES.values():
        for stage in route["stages"]:
            plan = [tuple(item) for item in PROGRAMS[stage["program"]]["arg_plan"]]
            if stage["name"] in ("main_wave", "main_wave_halves"):
                assert ("buffer", "wave_sync") in plan
            if route["op"] != "wgrad":
                assert ("buffer", "wave_sync") not in plan
    rest = HOST_PLAN_CONSTANTS["sched_rest"]
    assert len(rest) == 3 and all(0 <= int(v) < 2**32 for v in rest)
    assert int(HOST_PLAN_CONSTANTS["sched_words"]) >= len(rest)
    assert int(HOST_PLAN_CONSTANTS["wave_sync_words"]) >= 1


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 512, 256), (RAGGED_E6, 1024, 512), (TINY_E5, 256, 64)]
)
def test_fwd_matches_reference(sizes, n, k):
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(sizes, n, k, device)
    out = grouped_gemm_fwd(x, w, offs)
    assert out.shape == (sum(sizes), n)
    _assert_bf16_close(out, _reference("fwd", x, w, sizes))


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 256, 512), (RAGGED_E6, 1024, 512), (TINY_E5, 64, 256)]
)
def test_dgrad_matches_reference(sizes, n, k):
    device = _require_program("dgrad")
    _x, w, g, offs = _inputs(sizes, n, k, device)
    out = grouped_gemm_dgrad(g, w, offs)
    assert out.shape == (sum(sizes), k)
    _assert_bf16_close(out, _reference("dgrad", g, w, sizes))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 512, 256), (RAGGED_E6, 512, 256), (TINY_E5, 256, 512)]
)
def test_wgrad_matches_reference(sizes, n, k, out_dtype):
    device = _require_program("wgrad", out_dtype)
    x, _w, g, offs = _inputs(sizes, n, k, device)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    out = launch.launch()
    assert out.shape == (len(sizes), n, k) and out.dtype == out_dtype
    ref = _reference("wgrad", g, x, sizes)
    if out_dtype == torch.bfloat16:
        _assert_bf16_close(out, ref)
    else:
        rel_l2 = (out - ref).norm() / ref.norm()
        assert rel_l2.item() <= FP32_REL_L2, (
            f"fp32 weight gradient rel-L2 {rel_l2.item():.3e}"
        )
    for e, (s, t) in enumerate(_bounds(sizes)):
        if t == s:
            assert torch.equal(out[e], torch.zeros_like(out[e])), (
                f"empty group {e} must be exact zeros"
            )
    # Deterministic: a second launch into the same output is bitwise identical.
    first = out.clone()
    out.fill_(float("nan"))
    launch.launch()
    torch.cuda.synchronize()
    assert torch.equal(out, first)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_long_reduction_selects_a_valid_tile(out_dtype):
    device = _require_program("wgrad", out_dtype)
    n, k = 512 if out_dtype == torch.bfloat16 else 256, 1024
    x, _w, g, offs = _inputs(LONG_E3, n, k, device, seed=3)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    assert launch.plan["tile_k"] in (256, 512) and k % launch.plan["tile_k"] == 0
    out = launch.launch()
    ref = _reference("wgrad", g, x, LONG_E3)
    if out_dtype == torch.bfloat16:
        _assert_bf16_close(out, ref)
    else:
        assert ((out - ref).norm() / ref.norm()).item() <= FP32_REL_L2
    assert torch.equal(out[1], torch.zeros_like(out[1]))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_short_groups_never_include_the_next_group(out_dtype):
    """A tile of a group shorter than one k-step must not add the rows of the group
    the same CTA reduces next (#6570). Repeated launches of one prepared call,
    checked block by block against the per-group reference."""
    device = _require_program("wgrad", out_dtype)
    n, k = 512, 3072
    x, _w, g, offs = _inputs(SMALL_E512, n, k, device, seed=6)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    ref = torch.stack(
        [g[s:t].float().T @ x[s:t].float() for s, t in _bounds(SMALL_E512)]
    )
    blocks = ref.view(len(SMALL_E512), n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(2, 4), keepdim=True)
    wrong = []
    for call in range(SMALL_GROUP_LAUNCHES):
        out = launch.launch().float().view_as(blocks)
        bad = ((out - blocks).abs() > scale * 2.0**-6 + 1e-6).any(dim=4).any(dim=2)
        if bool(bad.any()):
            wrong.append((call, bad.nonzero()[:4].tolist()))
    assert not wrong, (
        f"{len(wrong)} of {SMALL_GROUP_LAUNCHES} launches have wrong 128 x 128 blocks "
        f"(call, [group, n block, k block]): {wrong[:3]}"
    )


def test_prepared_launch_reuses_buffers_without_allocation():
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(RAGGED_E6, 1024, 512, device)
    out = torch.full(
        (sum(RAGGED_E6), 1024), float("nan"), dtype=torch.bfloat16, device=device
    )
    launch = prepare_grouped_gemm_fwd(x, w, offs, out=out)
    assert launch.launch() is out
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device)
    launch.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device)
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    _assert_bf16_close(out, _reference("fwd", x, w, RAGGED_E6))


def _u32_words(tensor):
    """The words of a u32 workspace as Python ints."""
    return [v & 0xFFFFFFFF for v in tensor.view(torch.int32).tolist()]


def _workspace_kinds(launch):
    return {kind for item in launch.launches for _, kind in item.workspace_slots}


def _prepare_row_op(op, sizes, n, k, device, out=None):
    x, w, g, offs = _inputs(sizes, n, k, device)
    if op == "fwd":
        return prepare_grouped_gemm_fwd(x, w, offs, out=out), _reference(
            "fwd", x, w, sizes
        )
    return prepare_grouped_gemm_dgrad(g, w, offs, out=out), _reference(
        "dgrad", g, w, sizes
    )


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_scheduler_workspace_rests_after_every_launch(op):
    """The kernels re-arm ``sched`` at the end of every launch: repeated launches of
    one prepared call on one stream each compute every tile (the output is refilled
    with NaN before each), and the stream's workspace is at its rest words."""
    device = _require_program(op)
    launch, ref = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
    if "sched" not in _workspace_kinds(launch):
        pytest.skip(
            f"the {op} program bound on this device takes no scheduler workspace"
        )
    c = HOST_PLAN_CONSTANTS
    for _ in range(4):
        launch.out.fill_(float("nan"))
        _assert_bf16_close(launch.launch(), ref)
    torch.cuda.synchronize()
    sched = cb.stream_workspace("sched", device)
    assert sched is not None and sched.dtype == torch.uint32
    assert sched.numel() == int(c["sched_words"]) and sched.is_contiguous()
    assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_wave_sync_route_and_rest_state(out_dtype, monkeypatch):
    """From ``wave_sync_min_k`` on, an architecture ``wave_sync_archs`` lists for the
    k tile, whose route has a ``main_wave`` stage, launches the wave build with
    its ``wave_sync_mode`` as ``wave_mode``; it leaves its counters at zero and
    gives the plain build's output bits."""
    device = _require_program("wgrad", out_dtype)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 256, int(c["wave_sync_min_k"])
    x, _w, g, offs = _inputs(RAGGED_E4, n, k, device, seed=7)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    tile_k = launch.plan["tile_k"]
    wave = arch in c["wave_sync_archs"][tile_k] and cake_jit.stage_served(
        launch.record_name, "main_wave", arch
    )
    assert launch.stages[0] == ("main_wave" if wave else "main")
    assert launch.plan["wave_mode"] == (int(c["wave_sync_mode"][arch]) if wave else 0)
    assert ("wave_sync" in _workspace_kinds(launch)) == wave
    ref = _reference("wgrad", g, x, RAGGED_E4)
    for _ in range(3):
        launch.out.fill_(float("nan"))
        out = launch.launch()
        if out_dtype == torch.bfloat16:
            _assert_bf16_close(out, ref)
        else:
            assert ((out - ref).norm() / ref.norm()).item() <= FP32_REL_L2
    torch.cuda.synchronize()
    if wave:
        counters = cb.stream_workspace("wave_sync", device)
        assert counters is not None and counters.dtype == torch.uint32
        assert counters.numel() == int(c["wave_sync_words"])
        assert _u32_words(counters) == [0] * int(c["wave_sync_words"])
    # Below the threshold (or with the wave build switched off) the plain build runs.
    short = prepare_grouped_gemm_wgrad(
        g, x[:, : k // 2].contiguous(), offs, out_dtype=out_dtype
    )
    assert short.stages[0] == "main" and short.plan["wave_mode"] == 0
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wave_sync_min_k", 1 << 30)
    plain = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    assert plain.stages[0] == "main" and plain.plan["wave_mode"] == 0
    assert plain.plan["tile_k"] == launch.plan["tile_k"]
    torch.cuda.synchronize()
    assert torch.equal(plain.launch(), out)


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_wide_tile_gives_the_256_column_tile_bits(op, monkeypatch):
    """On an architecture of ``wide_fwd_archs`` / ``wide_dgrad_archs`` an output
    width that is a multiple of ``wide_block_n`` launches the 512-column tile; it
    leaves ``sched`` at rest and gives the 256-column tile's output bits (both
    accumulate each element's k steps in order and round once)."""
    device = _require_program(op)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = (1536, 512) if op == "fwd" else (512, 1536)
    launch, ref = _prepare_row_op(op, RAGGED_E6, n, k, device)
    wide = arch in c[f"wide_{op}_archs"] and cake_jit.stage_served(
        launch.record_name, "main_wide", arch
    )
    # The forward pass's short reductions take the wide tile's last-wave-halves twin.
    ef = (
        wide
        and op == "fwd"
        and cb.fwd_lwh_ef(k, arch, c)
        and cake_jit.stage_served(launch.record_name, "main_wide_ef", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef" if ef else "main_wide" if wide else "main"
    )
    assert launch.plan["tile_n"] == int(c["wide_block_n" if wide else "block_n"])
    for _ in range(3):
        launch.out.fill_(float("nan"))
        _assert_bf16_close(launch.launch(), ref)
    torch.cuda.synchronize()
    if not wide:
        pytest.skip(f"no 512-column {op} tile on {arch}")
    sched = cb.stream_workspace("sched", device)
    assert sched is not None
    assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, f"wide_{op}_archs", [])
    narrow, _ = _prepare_row_op(op, RAGGED_E6, n, k, device)
    assert narrow.stages[0] == "main"
    assert narrow.plan["tile_n"] == int(c["block_n"])
    assert narrow.plan["clusters"] >= launch.plan["clusters"]
    torch.cuda.synchronize()
    assert torch.equal(narrow.launch(), launch.out)


def test_dgrad_evict_first_gives_the_wide_tile_bits(monkeypatch):
    """Where ``dgrad_evict_first`` holds, the data gradient launches the evict-first
    twin of its wide tile (``main_wide_ef``).  The cache policy changes no arithmetic:
    the output bits are those of the wide tile and of the 256-column tile."""
    device = _require_program("dgrad")
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 64, int(c["dgrad_evict_first_min_k"])
    sizes = EVICT_FIRST_E2
    assert sum(sizes) // len(sizes) >= int(c["dgrad_evict_first_min_rows"])
    launch, ref = _prepare_row_op("dgrad", sizes, n, k, device)
    name = launch.record_name
    ef = arch in c["dgrad_evict_first_archs"] and cake_jit.stage_served(
        name, "main_wide_ef", arch
    )
    wide = arch in c["wide_dgrad_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    assert launch.stages[0] == (
        "main_wide_ef" if ef else "main_wide" if wide else "main"
    )
    for _ in range(2):
        launch.out.fill_(float("nan"))
        _assert_bf16_close(launch.launch(), ref)
    torch.cuda.synchronize()
    if not ef:
        pytest.skip(f"no evict-first data-gradient tile on {arch}")
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "dgrad_evict_first_min_rows", 1 << 30)
    plain, _ = _prepare_row_op("dgrad", sizes, n, k, device)
    assert plain.stages[0] == "main_wide"
    assert plain.plan["tile_n"] == launch.plan["tile_n"]
    torch.cuda.synchronize()
    assert torch.equal(plain.launch(), launch.out)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wide_dgrad_archs", [])
    narrow, _ = _prepare_row_op("dgrad", sizes, n, k, device)
    assert narrow.stages[0] == "main"
    torch.cuda.synchronize()
    assert torch.equal(narrow.launch(), launch.out)


def test_fwd_lwh_ef_gives_the_wide_tile_bits(monkeypatch):
    """Where ``fwd_lwh_ef`` holds (a short reduction, the down projection), the forward
    pass launches the twin of its wide tile that releases the last wave in halves and
    stores Y evict-first (``main_wide_ef``).  Neither changes the arithmetic: the
    output bits are those of the wide tile and of the 256-column tile."""
    device = _require_program("fwd")
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 2 * int(c["wide_block_n"]), int(c["fwd_lwh_ef_max_k"])
    sizes = EVICT_FIRST_E2
    launch, ref = _prepare_row_op("fwd", sizes, n, k, device)
    name = launch.record_name
    ef = arch in c["fwd_lwh_ef_archs"] and cake_jit.stage_served(
        name, "main_wide_ef", arch
    )
    wide = arch in c["wide_fwd_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    assert launch.stages[0] == (
        "main_wide_ef" if ef else "main_wide" if wide else "main"
    )
    for _ in range(2):
        launch.out.fill_(float("nan"))
        _assert_bf16_close(launch.launch(), ref)
    torch.cuda.synchronize()
    if not ef:
        pytest.skip(f"no last-wave-halves forward tile on {arch}")
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "fwd_lwh_ef_max_k", k - 1)
    plain, _ = _prepare_row_op("fwd", sizes, n, k, device)
    assert plain.stages[0] == "main_wide"
    assert plain.plan["tile_n"] == launch.plan["tile_n"]
    torch.cuda.synchronize()
    assert torch.equal(plain.launch(), launch.out)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wide_fwd_archs", [])
    narrow, _ = _prepare_row_op("fwd", sizes, n, k, device)
    assert narrow.stages[0] == "main"
    torch.cuda.synchronize()
    assert torch.equal(narrow.launch(), launch.out)


@pytest.mark.parametrize("wave", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_k512_halves_gives_the_drain_build_bits(out_dtype, wave, monkeypatch):
    """With the 512-wide k tile on an architecture of ``k512_release_archs`` the
    weight gradient launches the twin of its build that releases the accumulator in
    two halves (``main_halves``; ``main_wave_halves`` from ``wave_sync_min_k``).  The
    output bits are those of the build that drains the whole accumulator."""
    device = _require_program("wgrad", out_dtype)
    c = HOST_PLAN_CONSTANTS
    # The k256-tail twin of main_halves has its own test.
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k256_tail_archs", [])
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 256, (int(c["wave_sync_min_k"]) if wave else 1024)
    x, _w, g, offs = _inputs(K512_E2, n, k, device, seed=10)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    if launch.plan["tile_k"] != 512:
        assert arch not in c["k512_archs"]
        pytest.skip(f"the weight gradient takes no 512-wide k tile on {arch}")
    name = launch.record_name
    base = "main_wave" if launch.plan["wave_mode"] else "main"
    assert (base == "main_wave") == (
        wave
        and arch in c["wave_sync_archs"][512]
        and cake_jit.stage_served(name, "main_wave", arch)
    )
    halves = arch in c["k512_release_archs"]
    assert launch.stages[0] == (f"{base}_halves" if halves else base)
    ref = _reference("wgrad", g, x, K512_E2)
    for _ in range(2):
        launch.out.fill_(float("nan"))
        out = launch.launch()
        if out_dtype == torch.bfloat16:
            _assert_bf16_close(out, ref)
        else:
            assert ((out - ref).norm() / ref.norm()).item() <= FP32_REL_L2
    torch.cuda.synchronize()
    if base == "main_wave":
        counters = cb.stream_workspace("wave_sync", device)
        assert _u32_words(counters) == [0] * int(c["wave_sync_words"])
    if not halves:
        pytest.skip(f"no half-release twin on {arch}")
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k512_release_archs", [])
    drain = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    assert drain.stages[0] == base and drain.plan["tile_k"] == 512
    assert drain.plan["wave_mode"] == launch.plan["wave_mode"]
    torch.cuda.synchronize()
    assert torch.equal(drain.launch(), out)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_k256_tail_gives_the_k256_tile_bits(out_dtype, monkeypatch):
    """Where ``wgrad_k256_tail`` holds, the half-release 512-wide k tile launches its
    twin that runs the tiles left after its full waves as 256-column half tiles
    (``main_halves_k256t``): n-major, the last partial wave of them split as the
    256-wide k tile splits its tail and summed by that tile's tail reduce.  Each
    output element accumulates its rows in the order of the 256-wide k tile, so where
    the two plans split the same tiles the same way the output bits are that tile's."""
    device = _require_program("wgrad", out_dtype)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    # A down-projection-like output (K at most k256_tail_max_k) over two groups of more
    # than k256_tail_min_rows rows on average, with 128 512-wide tiles, more than the
    # 106 clusters of a 212-SM device: the 256-column tail is split and reduced.
    sizes = K512_E2
    n, k = 4096, int(c["k256_tail_max_k"])
    assert sum(sizes) // len(sizes) >= int(c["k256_tail_min_rows"])
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=11)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    if arch not in c["k256_tail_archs"]:
        assert launch.stages[0] != "main_halves_k256t"
        pytest.skip(f"no k256-tail weight-gradient build on {arch}")
    assert cake_jit.stage_served(launch.record_name, "main_halves_k256t", arch)
    plan = launch.plan
    assert plan["tile_k"] == 512 and plan["tail_tile_k"] == cb.K256_TAIL_TILE_K
    assert plan["raster_rows"] == 0
    tail = ("tail_reduce",) if plan["tail_splits"] > 1 else ()
    assert launch.stages == ("main_halves_k256t", *tail)
    outs = []
    for _ in range(2):
        launch.out.fill_(float("nan"))
        outs.append(launch.launch().clone())
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])
    ref = _reference("wgrad", g, x, sizes)
    if out_dtype == torch.bfloat16:
        _assert_bf16_close(outs[0], ref)
    else:
        assert ((outs[0] - ref).norm() / ref.norm()).item() <= FP32_REL_L2
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k512_archs", [])
    k256 = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    assert k256.plan["tile_k"] == cb.K256_TAIL_TILE_K and k256.stages[0] == "main"
    keys = ("clusters", "num_tail", "tail_splits", "raster_rows")
    if any(k256.plan[key] != plan[key] for key in keys):
        pytest.skip(f"the 256-wide k tile splits other tiles on this {arch} device")
    torch.cuda.synchronize()
    assert torch.equal(k256.launch(), outs[0])


def test_two_streams_take_separate_workspaces():
    """Launches on two streams in flight together never share a workspace; each
    stream keeps its own buffer at rest, and both results are exact."""
    device = _require_program("fwd")
    first, ref = _prepare_row_op("fwd", RAGGED_E6, 1024, 512, device)
    if "sched" not in _workspace_kinds(first):
        pytest.skip("the fwd program bound on this device takes no scheduler workspace")
    second, ref2 = _prepare_row_op("dgrad", RAGGED_E4, 256, 512, device)
    streams = [torch.cuda.Stream(device), torch.cuda.Stream(device)]
    main = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(main)
    for _ in range(8):
        for stream, launch in zip(streams, (first, second), strict=True):
            with torch.cuda.stream(stream):
                launch.launch()
    for stream in streams:
        main.wait_stream(stream)
    torch.cuda.synchronize()
    _assert_bf16_close(first.out, ref)
    _assert_bf16_close(second.out, ref2)
    buffers = [cb.stream_workspace("sched", device, stream) for stream in streams]
    assert all(buffer is not None for buffer in buffers)
    assert buffers[0].data_ptr() != buffers[1].data_ptr()
    rest = [int(v) for v in HOST_PLAN_CONSTANTS["sched_rest"]]
    for buffer in buffers:
        assert _u32_words(buffer[:3]) == rest
    current = cb.stream_workspace("sched", device)
    assert current is None or current.data_ptr() not in {
        buffer.data_ptr() for buffer in buffers
    }


@pytest.mark.parametrize("op", ["fwd", "wgrad"])
def test_cuda_graph_replay(op):
    """A captured launch takes workspaces of its own, set to their rest state in the
    graph; every replay computes every tile."""
    if op == "fwd":
        device = _require_program("fwd")
        launch, ref = _prepare_row_op("fwd", RAGGED_E6, 1024, 512, device)
    else:
        device = _require_program("wgrad", torch.bfloat16)
        k = int(HOST_PLAN_CONSTANTS["wave_sync_min_k"])
        x, _w, g, offs = _inputs(RAGGED_E4, 256, k, device, seed=8)
        launch = prepare_grouped_gemm_wgrad(g, x, offs)
        ref = _reference("wgrad", g, x, RAGGED_E4)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        launch.launch()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    for _ in range(3):
        launch.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bf16_close(launch.out, ref)


@pytest.mark.parametrize("op", ["dgrad", "wgrad"])
def test_prepared_launches_allocate_nothing_after_the_first(op):
    """After the first launch on a stream (which creates the stream workspaces), a
    prepared launch allocates no device memory: dgrad, and the weight gradient on
    its wave build where the device has one."""
    if op == "dgrad":
        device = _require_program("dgrad")
        launch, ref = _prepare_row_op("dgrad", RAGGED_E6, 1024, 512, device)
    else:
        device = _require_program("wgrad", torch.bfloat16)
        k = int(HOST_PLAN_CONSTANTS["wave_sync_min_k"])
        x, _w, g, offs = _inputs(RAGGED_E4, 256, k, device, seed=9)
        launch = prepare_grouped_gemm_wgrad(g, x, offs)
        ref = _reference("wgrad", g, x, RAGGED_E4)
    launch.launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device)
    for _ in range(3):
        launch.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device)
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    _assert_bf16_close(launch.out, ref)


def test_prepared_launch_follows_new_offsets():
    """The same prepared launch serves new group boundaries written into offs."""
    device = _require_program("dgrad")
    first, second = [300, 0, 1, 700], [1, 500, 250, 250]
    _x, w, g, offs = _inputs(first, 256, 512, device)
    launch = prepare_grouped_gemm_dgrad(g, w, offs)
    _assert_bf16_close(launch.launch(), _reference("dgrad", g, w, first))
    offs.copy_(_offs(second, device))
    _assert_bf16_close(launch.launch(), _reference("dgrad", g, w, second))


def test_padded_outputs_leave_the_padding_untouched():
    device = _require_program("fwd")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(RAGGED_E4, 512, 256, device)
    sum_m, n, k = sum(RAGGED_E4), 512, 256
    padded = torch.full(
        (sum_m, n + 64), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = grouped_gemm_fwd(x, w, offs, out=padded[:, :n])
    _assert_bf16_close(out, _reference("fwd", x, w, RAGGED_E4))
    assert torch.isnan(padded[:, n:]).all()
    padded = torch.full(
        (len(RAGGED_E4), n, k + 32), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = grouped_gemm_wgrad(g, x, offs, out=padded[:, :, :k])
    _assert_bf16_close(out, _reference("wgrad", g, x, RAGGED_E4))
    assert torch.isnan(padded[:, :, k:]).all()


def test_empty_batch():
    device = _require_program("fwd")
    device = _require_program("wgrad", torch.float32)
    x, w, g, offs = _inputs([0, 0], 256, 256, device)
    y = torch.full((0, 256), 0.0, dtype=torch.bfloat16, device=device)
    assert grouped_gemm_fwd(x, w, offs, out=y).shape == (0, 256)
    dw = torch.full((2, 256, 256), float("nan"), dtype=torch.float32, device=device)
    assert torch.equal(grouped_gemm_wgrad(g, x, offs, out=dw), torch.zeros_like(dw))


def test_rejects_misaligned_shapes():
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(RAGGED_E4, 384, 256, device)  # N % 256 != 0
    with pytest.raises(ValueError):
        prepare_grouped_gemm_fwd(x, w, offs)
    with pytest.raises(ValueError):
        prepare_grouped_gemm_fwd(x, w, offs.to(torch.int64))


# ---------------------------------------------------------------------------
# The m_indptr form and the autograd wrapper
# ---------------------------------------------------------------------------


def _m_indptr(sizes, device):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return torch.tensor([0] + ends, dtype=torch.int32, device=device)


def test_m_indptr_form_matches_offs_form():
    device = _require_program("fwd")
    device = _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device)
    m_indptr = _m_indptr(RAGGED_E6, device)
    assert torch.equal(grouped_gemm_fwd(x, w, m_indptr), grouped_gemm_fwd(x, w, offs))
    assert torch.equal(
        grouped_gemm_dgrad(g, w, m_indptr), grouped_gemm_dgrad(g, w, offs)
    )
    assert torch.equal(
        grouped_gemm_wgrad(g, x, m_indptr, num_groups=len(RAGGED_E6)),
        grouped_gemm_wgrad(g, x, offs),
    )
    with pytest.raises(ValueError):  # neither E nor E + 1 entries
        grouped_gemm_wgrad(g, x, m_indptr[:-2], num_groups=len(RAGGED_E6))


def test_autograd_wrapper_matches_reference_and_is_deterministic():
    device = _require_program("fwd")
    device = _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    cake_grouped_mm = pkg.cake_grouped_mm

    x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device)
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = cake_grouped_mm(x, w, offs)
    assert y.shape == (sum(RAGGED_E6), 512) and y.dtype == torch.bfloat16
    _assert_bf16_close(y.detach(), _reference("fwd", x.detach(), w.detach(), RAGGED_E6))
    y.backward(g)
    assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.bfloat16
    _assert_bf16_close(x.grad, _reference("dgrad", g, w.detach(), RAGGED_E6))
    _assert_bf16_close(w.grad, _reference("wgrad", g, x.detach(), RAGGED_E6))
    for e, (s, t) in enumerate(_bounds(RAGGED_E6)):
        if t == s:
            assert torch.equal(w.grad[e], torch.zeros_like(w.grad[e])), (
                f"empty group {e} must receive an exactly zero weight gradient"
            )
    dx, dw = x.grad.clone(), w.grad.clone()
    x.grad = None
    w.grad = None
    # Same gradients from the m_indptr form; bitwise reproducible.
    cake_grouped_mm(x, w, _m_indptr(RAGGED_E6, device), deterministic=True).backward(g)
    torch.cuda.synchronize()
    assert torch.equal(x.grad, dx) and torch.equal(w.grad, dw)
    with pytest.raises(ValueError):
        cake_grouped_mm(x.float(), w, offs)
