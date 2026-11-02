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

import collections
import contextlib
import functools
import math

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

# Backend suite: the package is named once, through ``cake_package``; nothing else
# of flashinfer is imported here.  The tests of the stable flashinfer entry point
# ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` live in
# ``test_cake_moe_grouped_det_gemm_api.py``.
pkg = cake_package("cake_moe_grouped_det_gemm")
cb = cake_package("cake_moe_grouped_det_gemm", "cake_backend")
cake_jit = cake_package("cake_moe_grouped_det_gemm", "cake_jit")
cake_autograd = cake_package("cake_moe_grouped_det_gemm", "cake_autograd")
grouped_det_gemm_dgrad = pkg.grouped_det_gemm_dgrad
grouped_det_gemm_fwd = pkg.grouped_det_gemm_fwd
grouped_det_gemm_wgrad = pkg.grouped_det_gemm_wgrad
SUPPORTED_COMPUTE_CAPABILITIES = cb.SUPPORTED_COMPUTE_CAPABILITIES
prepare_grouped_det_gemm_dgrad = cb.prepare_grouped_det_gemm_dgrad
prepare_grouped_det_gemm_fwd = cb.prepare_grouped_det_gemm_fwd
prepare_grouped_det_gemm_wgrad = cb.prepare_grouped_det_gemm_wgrad
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
# wide tile on the architectures of ``dgrad_evict_first_archs`` (and, with
# ``K >= dgrad_wide_ts33_min_k``, its 3 / 3 head / tail twin on those of
# ``dgrad_wide_ts33_archs``).
EVICT_FIRST_E2 = [16390, 16500]
# The MoE expert rows the evict-first and k-tile thresholds were measured on:
# ``EXPERT_ROWS`` routed rows split over each group count of ``EXPERT_GROUPS``.
EXPERT_ROWS = 131072
EXPERT_GROUPS = (4, 8, 16, 32)
# The MoE benchmark rows the 3 / 3 head / tail rule was measured on: (label, sum_m, E,
# row variants); the balanced, skewed and one-empty variants of an E4 row share sum_m
# and E, so the host plan gives them one stage.
BENCH_ROWS = (
    ("E4_m112576", 112576, 4, 3),
    ("E4_m136064", 136064, 4, 3),
    ("E8", EXPERT_ROWS, 8, 1),
    ("E16", EXPERT_ROWS, 16, 1),
    ("E32", EXPERT_ROWS, 32, 1),
)
# The benchmark projections, (N, K) of w ``[E, N, K]``.
BENCH_PROJ = (("gate_up", 4096, 6144), ("down", 6144, 2048))


def _wide_ts33(op, k, arch, sum_m, groups, c):
    """The 3 / 3 head / tail rule of ``op``: ``dgrad_wide_ts33`` / ``fwd_wide_ts33``."""
    rule = cb.dgrad_wide_ts33 if op == "dgrad" else cb.fwd_wide_ts33
    return rule(k, arch, sum_m, groups, c)


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


def _fresh_stream_workspaces(monkeypatch):
    """Start the test with no kept stream workspace and no queue probe (both restored
    after it): torch reuses its pooled stream handles, so a new ``torch.cuda.Stream``
    could otherwise find the workspace an earlier test kept for the same handle."""
    monkeypatch.setattr(cb, "_stream_workspaces", {})
    monkeypatch.setattr(cb, "_queue_splits", {})


def _probes_queue_split(device, op):
    """Whether ``op`` is listed in ``domain_split_ops`` and ``device``'s architecture in
    ``domain_split_archs``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    c = HOST_PLAN_CONSTANTS
    return op in c["domain_split_ops"] and arch in c["domain_split_archs"]


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
        assert plan["queue_split"] == wide["queue_split"] == "even"
    # The probed queue split: the ops of ``domain_split_ops`` on the architectures of
    # ``domain_split_archs``, weights of at most ``domain_split_max_weight_elems``
    # elements; c0 / c1 stay the even split.
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for op, n, k in (
            ("fwd", 4096, 6144),
            ("fwd", 6144, 2048),
            ("dgrad", 4096, 6144),
            ("dgrad", 6144, 2048),
        ):
            plan = cb.plan_fwd_dgrad(
                op,
                sum_m=sum_m,
                num_groups=e,
                n=n,
                k=k,
                sm_count=sm_count,
                tile_n=int(c["wide_block_n"]),
                constants=c,
                arch=arch,
            )
            probed = (
                op in c["domain_split_ops"]
                and arch in c["domain_split_archs"]
                and n * k <= int(c["domain_split_max_weight_elems"])
            )
            assert plan["queue_split"] == ("probed" if probed else "even")
            assert plan["c0"] == plan["clusters"] // 2
            assert plan["c0"] + plan["c1"] == plan["clusters"]
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
    ``dgrad_wide_ts33_archs`` / ``fwd_wide_ts33_archs`` the 3 / 3 head / tail twin of
    the ``dgrad`` / ``fwd`` wide tile (``main_wide_ts33``) where ``dgrad_wide_ts33`` /
    ``fwd_wide_ts33`` holds and the evict-first twin does not; ``wave_sync_archs``
    the wave build of each k tile from ``wave_sync_min_k``, launched with the
    architecture's ``wave_sync_mode``; ``k512_release_archs`` the twin releasing the
    512-wide accumulator in halves; ``k256_tail_archs`` its twin running the
    leftover tiles as 256-column half tiles
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
    assert set(c["dgrad_wide_ts33_archs"]) <= set(c["wide_dgrad_archs"])
    assert set(c["dgrad_wide_ts33_min_rows"]) == set(c["dgrad_wide_ts33_archs"])
    assert all(int(v) > 0 for v in c["dgrad_wide_ts33_min_rows"].values())
    assert int(c["dgrad_wide_ts33_min_k"]) % wide_n == 0
    assert set(c["fwd_wide_ts33_archs"]) <= set(c["wide_fwd_archs"])
    assert int(c["fwd_wide_ts33_max_k"]) > 0
    assert int(c["fwd_wide_ts33_max_rows"]) > 0
    assert set(c["fwd_lwh_ef_archs"]) <= set(c["wide_fwd_archs"])
    assert int(c["fwd_lwh_ef_max_k"]) > 0
    # The probed queue split serves the domain-scheduled programs only (sm_107a's are
    # static and ignore c0 / c1), and the data gradient only.
    assert set(c["domain_split_archs"]) <= {"sm_100a", "sm_103a"}
    assert list(c["domain_split_ops"]) == ["dgrad"]
    assert int(c["domain_split_max_weight_elems"]) > 0
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
            served_ts33 = cake_jit.stage_served(name, "main_wide_ts33", arch)
            if arch in c[f"{op}_wide_ts33_archs"]:
                assert served_ts33, (name, arch)
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
                    ts33 = (
                        wide
                        and not ef
                        and served_ts33
                        and _wide_ts33(op, k, arch, sum_m, groups, c)
                    )
                    expected = (
                        "main_wide_ef"
                        if ef
                        else "main_wide_ts33"
                        if ts33
                        else "main_wide"
                    )
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
    (``K`` 6144) takes the evict-first wide tile on sm_107a and the 3 / 3 head / tail
    wide tile on sm_103a at 4, 8 and 16 groups, else the plain wide tile; the down
    projection (``K`` 2048) takes the plain wide tile.
    The weight gradient's 512-wide k tile starts at 4096 rows per group (32 groups) on
    every architecture."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("dgrad")
    for groups in EXPERT_GROUPS:
        for n, k in ((4096, 6144), (6144, 2048)):
            ef = arch == "sm_107a" and k == 6144 and groups in (4, 8)
            ts33 = k == 6144 and arch == "sm_103a" and groups in (4, 8, 16)
            assert cb.dgrad_evict_first(k, arch, EXPERT_ROWS, groups, c) == ef
            assert cb.dgrad_wide_ts33(k, arch, EXPERT_ROWS, groups, c) == ts33
            stage, tile_n = cb.select_row_stage(
                name, "dgrad", n, k, arch, c, sum_m=EXPERT_ROWS, num_groups=groups
            )
            assert tile_n == int(c["wide_block_n"]), (arch, groups, k)
            expected = (
                "main_wide_ef" if ef else "main_wide_ts33" if ts33 else "main_wide"
            )
            assert stage == expected, (arch, groups, k)
        k512 = EXPERT_ROWS // groups >= 4096
        for k in (2048, 6144):
            tile_k = cb.select_wgrad_tile(k, arch, EXPERT_ROWS, groups, c)
            assert tile_k == (512 if k512 else 256), (arch, groups, k)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bench_rows_take_the_measured_wide_ts33_stage(arch):
    """Over the MoE benchmark rows (``BENCH_ROWS`` x ``BENCH_PROJ``, both ops): sm_103a
    takes ``main_wide_ts33`` for the down forward pass at E32 and the gate/up data
    gradient at E4, E8 and E16 (9 rows with the E4 variants); sm_100a and sm_107a
    for none.  Every other row keeps its stage without the rule."""
    c = HOST_PLAN_CONSTANTS
    taken = set()
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        for proj, n, k in BENCH_PROJ:
            for label, sum_m, groups, _variants in BENCH_ROWS:
                stage, tile_n = cb.select_row_stage(
                    name, op, n, k, arch, c, sum_m=sum_m, num_groups=groups
                )
                assert tile_n == int(c["wide_block_n"]), (op, proj, label)
                if stage == "main_wide_ts33":
                    taken.add((op, proj, label))
                else:
                    assert stage in ("main_wide", "main_wide_ef"), (op, proj, label)
    expected = {
        "sm_100a": set(),
        "sm_103a": {
            ("fwd", "down", "E32"),
            ("dgrad", "gate_up", "E4_m112576"),
            ("dgrad", "gate_up", "E4_m136064"),
            ("dgrad", "gate_up", "E8"),
            ("dgrad", "gate_up", "E16"),
        },
        "sm_107a": set(),
    }[arch]
    assert taken == expected, (arch, sorted(taken ^ expected))
    variants = {label: v for label, _sum_m, _groups, v in BENCH_ROWS}
    rows = sum(variants[label] for _op, _proj, label in taken)
    assert rows == {"sm_100a": 0, "sm_103a": 9, "sm_107a": 0}[arch]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bench_rows_take_the_probed_queue_split(arch):
    """Over the MoE benchmark rows (``BENCH_ROWS`` x ``BENCH_PROJ``, both ops): sm_100a
    and sm_103a pass the probed queue split for every data-gradient down-projection row
    (9 rows with the E4 variants) and the even split for every forward row and every
    gate/up row; sm_107a passes the even split everywhere."""
    c = HOST_PLAN_CONSTANTS
    taken = set()
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        for proj, n, k in BENCH_PROJ:
            for label, sum_m, groups, _variants in BENCH_ROWS:
                _stage, tile_n = cb.select_row_stage(
                    name, op, n, k, arch, c, sum_m=sum_m, num_groups=groups
                )
                plan = cb.plan_fwd_dgrad(
                    op,
                    sum_m=sum_m,
                    num_groups=groups,
                    n=n,
                    k=k,
                    sm_count=152,
                    tile_n=tile_n,
                    constants=c,
                    arch=arch,
                )
                if plan["queue_split"] == "probed":
                    taken.add((op, proj, label))
                else:
                    assert plan["queue_split"] == "even", (op, proj, label)
    expected = (
        {("dgrad", "down", label) for label, _sum_m, _groups, _variants in BENCH_ROWS}
        if arch in ("sm_100a", "sm_103a")
        else set()
    )
    assert taken == expected, (arch, sorted(taken ^ expected))
    variants = {label: v for label, _sum_m, _groups, v in BENCH_ROWS}
    rows = sum(variants[label] for _op, _proj, label in taken)
    assert rows == {"sm_100a": 9, "sm_103a": 9, "sm_107a": 0}[arch]


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
    out = grouped_det_gemm_fwd(x, w, offs)
    assert out.shape == (sum(sizes), n)
    _assert_bf16_close(out, _reference("fwd", x, w, sizes))


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 256, 512), (RAGGED_E6, 1024, 512), (TINY_E5, 64, 256)]
)
def test_dgrad_matches_reference(sizes, n, k):
    device = _require_program("dgrad")
    _x, w, g, offs = _inputs(sizes, n, k, device)
    out = grouped_det_gemm_dgrad(g, w, offs)
    assert out.shape == (sum(sizes), k)
    _assert_bf16_close(out, _reference("dgrad", g, w, sizes))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 512, 256), (RAGGED_E6, 512, 256), (TINY_E5, 256, 512)]
)
def test_wgrad_matches_reference(sizes, n, k, out_dtype):
    device = _require_program("wgrad", out_dtype)
    x, _w, g, offs = _inputs(sizes, n, k, device)
    launch = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    launch = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    launch = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    launch = prepare_grouped_det_gemm_fwd(x, w, offs, out=out)
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
        return prepare_grouped_det_gemm_fwd(x, w, offs, out=out), _reference(
            "fwd", x, w, sizes
        )
    return prepare_grouped_det_gemm_dgrad(g, w, offs, out=out), _reference(
        "dgrad", g, w, sizes
    )


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_scheduler_workspace_rests_after_every_launch(op, monkeypatch):
    """The kernels re-arm ``sched`` at the end of every launch: repeated launches of
    one prepared call on one stream each compute every tile (the output is refilled
    with NaN before each), and the stream's workspace is at its rest words.  Where the
    launch probes the queue split, the probe's add-zero atomics leave it at rest too
    and the later launches pass the probed split."""
    device = _require_program(op)
    _fresh_stream_workspaces(monkeypatch)
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
    if _probes_queue_split(device, op):
        assert launch.plan["queue_split"] == "probed"
        assert cb.stream_queue_split(device) is not None


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
    launch = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    short = prepare_grouped_det_gemm_wgrad(
        g, x[:, : k // 2].contiguous(), offs, out_dtype=out_dtype
    )
    assert short.stages[0] == "main" and short.plan["wave_mode"] == 0
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wave_sync_min_k", 1 << 30)
    plain = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    ts33 = (
        wide
        and not ef
        and _wide_ts33(op, k, arch, sum(RAGGED_E6), len(RAGGED_E6), c)
        and cake_jit.stage_served(launch.record_name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
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
    ts33 = (
        wide
        and not ef
        and cb.dgrad_wide_ts33(k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
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


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_wide_ts33_gives_the_wide_tile_bits(op, monkeypatch):
    """Where ``fwd_wide_ts33`` / ``dgrad_wide_ts33`` holds, the forward pass / data
    gradient launches the twin of its wide tile with three head and three tail k steps
    around the release of the accumulator's halves (``main_wide_ts33``).  Every element
    still accumulates its k steps in order: the output bits are those of the wide tile
    and of the 256-column tile, a CUDA-graph replay gives them too, and the launch
    leaves ``sched`` at rest."""
    device = _require_program(op)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    if op == "dgrad":  # the gate/up data gradient over few groups
        n, k, sizes = 64, int(c["dgrad_wide_ts33_min_k"]), EVICT_FIRST_E2
    else:  # the down projection over many small groups
        n, k = 2 * int(c["wide_block_n"]), int(c["fwd_wide_ts33_max_k"])
        sizes = RAGGED_E6
    launch, ref = _prepare_row_op(op, sizes, n, k, device)
    name = launch.record_name
    ef_key = "dgrad_evict_first_archs" if op == "dgrad" else "fwd_lwh_ef_archs"
    ef_rule = (
        cb.dgrad_evict_first(k, arch, sum(sizes), len(sizes), c)
        if op == "dgrad"
        else cb.fwd_lwh_ef(k, arch, c)
    )
    ef = (
        arch in c[ef_key]
        and ef_rule
        and cake_jit.stage_served(name, "main_wide_ef", arch)
    )
    wide = arch in c[f"wide_{op}_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    ts33 = (
        wide
        and not ef
        and _wide_ts33(op, k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
    )
    for _ in range(3):
        launch.out.fill_(float("nan"))
        _assert_bf16_close(launch.launch(), ref)
    torch.cuda.synchronize()
    if not ts33:
        pytest.skip(f"no 3 / 3 head / tail {op} tile on {arch}")
    sched = cb.stream_workspace("sched", device)
    assert sched is not None
    assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]
    captured = launch.out.clone()
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        launch.launch()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    assert torch.equal(launch.out, captured)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    for _ in range(2):
        launch.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(launch.out, captured)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, f"{op}_wide_ts33_archs", [])
    plain, _ = _prepare_row_op(op, sizes, n, k, device)
    assert plain.stages[0] == "main_wide"
    assert plain.plan["tile_n"] == launch.plan["tile_n"]
    torch.cuda.synchronize()
    assert torch.equal(plain.launch(), launch.out)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, f"wide_{op}_archs", [])
    narrow, _ = _prepare_row_op(op, sizes, n, k, device)
    assert narrow.stages[0] == "main"
    torch.cuda.synchronize()
    assert torch.equal(narrow.launch(), launch.out)


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_probed_queue_split_gives_the_even_split_bits(op, monkeypatch):
    """On the architectures of ``domain_split_archs`` a launch of an op in
    ``domain_split_ops`` whose weight has at most ``domain_split_max_weight_elems``
    elements probes its stream's scheduler workspace at its first launch there and
    passes the measured SM pairs of each die as ``c0`` / ``c1`` once the probe has
    completed; the other ops pass the even split and start no probe.  The split only
    divides the tiles between the claim queues: every launch gives the bits of the
    even split, the workspace stays at rest, and a second stream probes a workspace
    of its own."""
    device = _require_program(op)
    _fresh_stream_workspaces(monkeypatch)
    c = HOST_PLAN_CONSTANTS
    launch, ref = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
    if "sched" not in _workspace_kinds(launch):
        pytest.skip(
            f"the {op} program bound on this device takes no scheduler workspace"
        )
    probed = _probes_queue_split(device, op)
    assert launch.plan["queue_split"] == ("probed" if probed else "even")
    main = torch.cuda.current_stream(device)
    outs, splits = [], []
    for stream in (torch.cuda.Stream(device), torch.cuda.Stream(device)):
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch.out.fill_(float("nan"))
                launch.launch()
                outs.append(launch.out.clone())
                stream.synchronize()
                splits.append(cb.stream_queue_split(device, stream))
            sched = cb.stream_workspace("sched", device, stream)
            assert sched is not None
            assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]
        main.wait_stream(stream)
    torch.cuda.synchronize()
    _assert_bf16_close(outs[0], ref)
    assert all(torch.equal(out, outs[0]) for out in outs)
    if not probed:
        assert splits == [None] * len(splits)
        pytest.skip(f"no probed queue split for {op} on this device")
    # The first launch on each stream enqueues the probe and passes the even split;
    # the stream is synchronized after it, so the next launches pass the probe's split.
    for split in splits:
        assert split is not None
        c0, c1 = split
        assert c0 >= 1 and c1 >= 1 and c0 + c1 == cb.sm_count(device) // 2
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "domain_split_archs", [])
    even, _ = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
    assert even.plan["queue_split"] == "even"
    assert (even.plan["c0"], even.plan["c1"]) == (launch.plan["c0"], launch.plan["c1"])
    torch.cuda.synchronize()
    assert torch.equal(even.launch(), outs[0])


@pytest.mark.parametrize("probe", ["no_compiler", "no_sm_seen"])
def test_queue_split_stays_even_without_a_probe(probe, monkeypatch):
    """Where the queue probe cannot be built (no CUDA compiler) or does not see every
    SM (here: a probe that records nothing), a probed launch keeps passing the even
    split, with the same bits."""
    device = _require_program("dgrad")
    if not _probes_queue_split(device, "dgrad"):
        pytest.skip("no probed queue split on this device")
    _fresh_stream_workspaces(monkeypatch)

    class _NoRecord:
        @staticmethod
        def cake_queue_probe(sched, lat, ctas, smem):
            return None

    module = None if probe == "no_compiler" else _NoRecord()
    monkeypatch.setattr(cb, "_queue_probe_module", lambda capability: module)
    launch, ref = _prepare_row_op("dgrad", RAGGED_E6, 1024, 512, device)
    assert launch.plan["queue_split"] == "probed"
    for _ in range(3):
        launch.out.fill_(float("nan"))
        _assert_bf16_close(launch.launch(), ref)
    torch.cuda.synchronize()
    assert cb.stream_queue_split(device) is None


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
    ts33 = (
        wide
        and not ef
        and cb.fwd_wide_ts33(k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
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
    launch = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    drain = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    launch = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
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
    k256 = prepare_grouped_det_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    assert k256.plan["tile_k"] == cb.K256_TAIL_TILE_K and k256.stages[0] == "main"
    keys = ("clusters", "num_tail", "tail_splits", "raster_rows")
    if any(k256.plan[key] != plan[key] for key in keys):
        pytest.skip(f"the 256-wide k tile splits other tiles on this {arch} device")
    torch.cuda.synchronize()
    assert torch.equal(k256.launch(), outs[0])


def test_two_streams_take_separate_workspaces(monkeypatch):
    """Launches on two streams in flight together never share a workspace; each
    stream keeps its own buffer at rest, and both results are exact.  Where a launch
    probes the queue split (the data gradient), its stream's workspace has a probe of
    its own; the forward's stream has none."""
    device = _require_program("fwd")
    _fresh_stream_workspaces(monkeypatch)
    first, ref = _prepare_row_op("fwd", RAGGED_E6, 1024, 512, device)
    if "sched" not in _workspace_kinds(first):
        pytest.skip("the fwd program bound on this device takes no scheduler workspace")
    second, ref2 = _prepare_row_op("dgrad", RAGGED_E4, 256, 512, device)
    streams = [torch.cuda.Stream(device), torch.cuda.Stream(device)]
    main = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(main)
    # One launch per stream alone first: a probe sees every SM only when no other
    # kernel holds one.
    for stream, launch in zip(streams, (first, second), strict=True):
        with torch.cuda.stream(stream):
            launch.launch()
        stream.synchronize()
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
    for stream, launch, op in zip(
        streams, (first, second), ("fwd", "dgrad"), strict=True
    ):
        probed = _probes_queue_split(device, op)
        assert launch.plan["queue_split"] == ("probed" if probed else "even")
        assert (cb.stream_queue_split(device, stream) is not None) == probed


@pytest.mark.parametrize("op", ["fwd", "dgrad", "wgrad"])
def test_cuda_graph_replay(op, monkeypatch):
    """A captured launch takes workspaces of its own, set to their rest state in the
    graph; every replay computes every tile.  A captured forward or data-gradient
    launch passes the even queue split and starts no probe; its replays give the bits
    of the eager launches, which pass the probed split where the launch probes it."""
    _fresh_stream_workspaces(monkeypatch)
    if op in ("fwd", "dgrad"):
        device = _require_program(op)
        launch, ref = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
    else:
        device = _require_program("wgrad", torch.bfloat16)
        k = int(HOST_PLAN_CONSTANTS["wave_sync_min_k"])
        x, _w, g, offs = _inputs(RAGGED_E4, 256, k, device, seed=8)
        launch = prepare_grouped_det_gemm_wgrad(g, x, offs)
        ref = _reference("wgrad", g, x, RAGGED_E4)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        launch.launch()
        side.synchronize()
        launch.launch()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    eager = launch.out.clone()
    probes = len(cb._queue_splits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    assert len(cb._queue_splits) == probes
    for _ in range(3):
        launch.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bf16_close(launch.out, ref)
        if op != "wgrad":
            assert torch.equal(launch.out, eager)
    if _probes_queue_split(device, op):
        assert launch.plan["queue_split"] == "probed" and probes == 1
        assert cb.stream_queue_split(device, side) is not None


@pytest.mark.parametrize("op", ["dgrad", "wgrad"])
def test_prepared_launches_allocate_nothing_after_the_first(op, monkeypatch):
    """After the first launch on a stream (which creates the stream workspaces and
    starts the queue probe where the launch probes one), a prepared launch allocates
    no device memory, also at the launch that takes up the probed split: dgrad, and
    the weight gradient on its wave build where the device has one."""
    _fresh_stream_workspaces(monkeypatch)
    if op == "dgrad":
        device = _require_program("dgrad")
        launch, ref = _prepare_row_op("dgrad", RAGGED_E6, 1024, 512, device)
    else:
        device = _require_program("wgrad", torch.bfloat16)
        k = int(HOST_PLAN_CONSTANTS["wave_sync_min_k"])
        x, _w, g, offs = _inputs(RAGGED_E4, 256, k, device, seed=9)
        launch = prepare_grouped_det_gemm_wgrad(g, x, offs)
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
    if _probes_queue_split(device, op):
        assert launch.plan["queue_split"] == "probed"
        assert cb.stream_queue_split(device) is not None


def test_prepared_launch_follows_new_offsets():
    """The same prepared launch serves new group boundaries written into offs."""
    device = _require_program("dgrad")
    first, second = [300, 0, 1, 700], [1, 500, 250, 250]
    _x, w, g, offs = _inputs(first, 256, 512, device)
    launch = prepare_grouped_det_gemm_dgrad(g, w, offs)
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
    out = grouped_det_gemm_fwd(x, w, offs, out=padded[:, :n])
    _assert_bf16_close(out, _reference("fwd", x, w, RAGGED_E4))
    assert torch.isnan(padded[:, n:]).all()
    padded = torch.full(
        (len(RAGGED_E4), n, k + 32), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = grouped_det_gemm_wgrad(g, x, offs, out=padded[:, :, :k])
    _assert_bf16_close(out, _reference("wgrad", g, x, RAGGED_E4))
    assert torch.isnan(padded[:, :, k:]).all()


def test_empty_batch():
    device = _require_program("fwd")
    device = _require_program("wgrad", torch.float32)
    x, w, g, offs = _inputs([0, 0], 256, 256, device)
    y = torch.full((0, 256), 0.0, dtype=torch.bfloat16, device=device)
    assert grouped_det_gemm_fwd(x, w, offs, out=y).shape == (0, 256)
    dw = torch.full((2, 256, 256), float("nan"), dtype=torch.float32, device=device)
    assert torch.equal(grouped_det_gemm_wgrad(g, x, offs, out=dw), torch.zeros_like(dw))


def test_rejects_misaligned_shapes():
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(RAGGED_E4, 384, 256, device)  # N % 256 != 0
    with pytest.raises(ValueError):
        prepare_grouped_det_gemm_fwd(x, w, offs)
    with pytest.raises(ValueError):
        prepare_grouped_det_gemm_fwd(x, w, offs.to(torch.int64))


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
    assert torch.equal(
        grouped_det_gemm_fwd(x, w, m_indptr), grouped_det_gemm_fwd(x, w, offs)
    )
    assert torch.equal(
        grouped_det_gemm_dgrad(g, w, m_indptr), grouped_det_gemm_dgrad(g, w, offs)
    )
    assert torch.equal(
        grouped_det_gemm_wgrad(g, x, m_indptr, num_groups=len(RAGGED_E6)),
        grouped_det_gemm_wgrad(g, x, offs),
    )
    with pytest.raises(ValueError):  # neither E nor E + 1 entries
        grouped_det_gemm_wgrad(g, x, m_indptr[:-2], num_groups=len(RAGGED_E6))


def test_autograd_wrapper_matches_reference_and_is_deterministic():
    device = _require_program("fwd")
    device = _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    cake_grouped_det_mm = pkg.cake_grouped_det_mm

    x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device)
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = cake_grouped_det_mm(x, w, offs)
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
    cake_grouped_det_mm(
        x, w, _m_indptr(RAGGED_E6, device), deterministic=True
    ).backward(g)
    torch.cuda.synchronize()
    assert torch.equal(x.grad, dx) and torch.equal(w.grad, dw)
    with pytest.raises(ValueError):
        cake_grouped_det_mm(x.float(), w, offs)


# ---------------------------------------------------------------------------
# Autograd wrapper: the forward launch after the saves (checkpoint early stop)
# ---------------------------------------------------------------------------

# ``cake_grouped_det_mm`` launches the forward after autograd saved its inputs, so
# the early stop of a non-reentrant checkpoint skips the recomputed GEMM when nothing
# after it saves a tensor.  These tests count the forward launches (a pass-through on
# ``cake_autograd._fwd_op``) and, where the torch profiler records kernels, the CUDA
# kernels of checkpointed steps; the control arm is a direct
# ``CakeGroupedDetMm.apply``, which launches inside ``forward``.  Both reductions are
# deterministic, so every output and gradient is compared bitwise.

# The small shape of the autograd tests: ragged groups with an empty and a one-row
# group.
SMALL_SIZES = RAGGED_E4
SMALL_SHAPE = (512, 256)  # (N, K)
# (sizes, (N, K)) of the early-stop tests: the small shape, and ``K512_E2``, whose
# weight gradient takes the 512-wide k tile on the architectures of ``k512_archs``.
AUTOGRAD_CASES = {"small": (SMALL_SIZES, SMALL_SHAPE), "k512": (K512_E2, (256, 1024))}
# Forward launches of a checkpointed step of each ``_region`` (early stop on): the
# recomputation stops after the GEMM's saves unless a later op saves a tensor.
REGION_FORWARDS = {"silu": 2, "scale": 1, "pair": 2}
# The MoE expert block of ``_moe_block`` and the (gate_up, down) forward launches of
# its checkpointed step: only the down projection's recomputation is skipped, and not
# when the routing multiply after it saves tensors.
MOE_HIDDEN, MOE_INTER = 256, 256
MOE_BLOCK_FORWARDS = {"block": (2, 1), "routed": (2, 2), "selective": (2, 1)}
# Forward launches of a checkpointed step: reentrant (always recomputes); nested
# non-reentrant (the outer stop fires at the inner checkpoint's input saves, the inner
# one before the GEMM); nested with an outer op that saves the inner output.
NESTED_FORWARDS = {"reentrant": 2, "nested": 1, "nested_then_save": 2}


def _require_autograd_programs():
    """The device, when the forward, data-gradient and bf16 weight-gradient programs are
    all registered for it."""
    _require_program("fwd")
    _require_program("dgrad")
    return _require_program("wgrad", torch.bfloat16)


def _wrapper(offs):
    """``gemm(a, b)`` through ``cake_grouped_det_mm``: the forward launch after ``apply``."""
    return lambda a, b: pkg.cake_grouped_det_mm(a, b, offs)


def _in_forward(offs, deterministic=True):
    """The control arm: ``gemm(a, b)`` through a direct ``CakeGroupedDetMm.apply``, which
    launches inside ``forward``, before autograd saves the inputs."""
    return lambda a, b: cake_autograd.CakeGroupedDetMm.apply(a, b, offs, deterministic)


def _leaves(*tensors):
    """Fresh leaves requiring grad, cloned from ``tensors``."""
    return [t.clone().requires_grad_(True) for t in tensors]


def _count_launches(monkeypatch):
    """Replace ``_fwd_op``, ``_dgrad_op`` and ``_wgrad_op`` of ``cake_autograd`` by
    pass-throughs that record the projection ``(N, K)`` of every call (of ``w``, or of the
    ``[E, N, K]`` weight gradient); returns ``{"fwd": [...], "dgrad": [...], "wgrad": [...]}``."""
    calls = {"fwd": [], "dgrad": [], "wgrad": []}

    def counting(op, launch):
        def call(*args):
            shape = args[3].shape if op == "wgrad" else args[1].shape
            calls[op].append((int(shape[1]), int(shape[2])))
            launch(*args)

        return call

    for op in calls:
        name = f"_{op}_op"
        monkeypatch.setattr(
            cake_autograd, name, counting(op, getattr(cake_autograd, name))
        )
    return calls


def _step(fn, leaves, grads, ckpt=None, early_stop=True, context_fn=None):
    """One training step: ``fn(*leaves)`` plain (``ckpt`` None) or under
    ``torch.utils.checkpoint`` (``"non_reentrant"`` / ``"reentrant"``; ``context_fn`` for
    a selective checkpoint), the early stop on (torch's default) or off around the forward
    and the backward, then the backward of the output(s) with ``grads``.  Returns
    ``(outputs, leaf gradients)``, both tuples, the outputs detached."""
    kwargs = {} if context_fn is None else {"context_fn": context_fn}
    stop = torch.utils.checkpoint.set_checkpoint_early_stop(bool(early_stop))
    with stop:
        if ckpt is None:
            out = fn(*leaves)
        else:
            out = torch.utils.checkpoint.checkpoint(
                fn, *leaves, use_reentrant=ckpt == "reentrant", **kwargs
            )
        outs = out if isinstance(out, tuple) else (out,)
        torch.autograd.backward(outs, grads)
    torch.cuda.synchronize()
    return tuple(o.detach() for o in outs), tuple(t.grad for t in leaves)


def _assert_same_step(got, want):
    """The outputs and gradients of two ``_step`` results are bitwise equal (None where
    None)."""
    pairs = zip((*got[0], *got[1]), (*want[0], *want[1]), strict=True)
    for index, (a, b) in enumerate(pairs):
        assert (a is None) == (b is None), index
        assert a is None or torch.equal(a, b), index


def _region(gemm, region):
    """``fn(x, w)`` of a checkpointed region around ``gemm(x, w)``: ``"silu"`` (saves the
    GEMM's output), ``"scale"`` (``* 2.0``, saves nothing) or ``"pair"`` (also returns
    ``x.sin()``, which saves ``x`` after the GEMM)."""
    if region == "silu":
        return lambda a, b: torch.nn.functional.silu(gemm(a, b))
    if region == "scale":
        return lambda a, b: gemm(a, b) * 2.0
    return lambda a, b: (gemm(a, b), a.sin())


def _region_grads(region, g, x):
    """The output gradients of a ``_region``: ``g``, and for ``"pair"`` one for ``x.sin()``."""
    if region != "pair":
        return (g,)
    gen = torch.Generator(device=x.device).manual_seed(7)
    return (g, torch.randn(x.shape, generator=gen, device=x.device).to(x.dtype))


def _moe_block(gemm, routed):
    """``fn(x, w1, w2[, probs])``: the MoE expert block ``down(silu(gate) * up)`` with
    ``gate, up = gate_up(x).chunk(2)``, both projections ``gemm``; with ``routed`` its
    output is weighted by routing probabilities."""

    def block(x, w1, w2, *probs):
        gate, up = gemm(x, w1).chunk(2, dim=-1)
        out = gemm(torch.nn.functional.silu(gate) * up, w2)
        return out * probs[0] if routed else out

    return block


def _cuda_kernel_names(fn):
    """The multiset of the names of the CUDA kernels one call of ``fn`` launches, the
    backward's (launched from the autograd thread) included (torch profiler; memcpy /
    memset activities excluded)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return collections.Counter(
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith(("Memcpy", "Memset"))
    )


def _require_profiled_kernels(device):
    if not _cuda_kernel_names(lambda: torch.ones(8, device=device).add_(1)):
        pytest.skip("the torch profiler records no CUDA kernels in this environment")


@contextlib.contextmanager
def _torch_deterministic():
    """``torch.use_deterministic_algorithms(True)`` inside the block, the previous setting
    restored after it."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


@pytest.mark.parametrize("sizes", [RAGGED_E4, [0, 0]], ids=["ragged", "empty"])
def test_wrapper_saves_inputs_before_the_forward_launch(sizes, monkeypatch):
    """Through ``cake_grouped_det_mm`` autograd saves ``x``, ``w`` and ``offs`` (the
    inputs themselves, in this order) and then the forward launches into ``Y``, saving
    nothing; a direct ``CakeGroupedDetMm.apply`` launches first and saves after.  The
    node is ``CakeGroupedDetMmBackward`` and ``Y`` has the forward's bits either way,
    also with no rows (``sum_m == 0``)."""
    device = _require_program("fwd")
    x0, w0, _g, offs = _inputs(sizes, *SMALL_SHAPE, device, seed=40)
    want = grouped_det_gemm_fwd(x0, w0, offs)
    events = []
    launch = cake_autograd._fwd_op

    def spy(x, w, offs_, out):
        events.append(("fwd", out.data_ptr()))
        launch(x, w, offs_, out)

    def pack(t):
        events.append(("pack", t))
        return t

    monkeypatch.setattr(cake_autograd, "_fwd_op", spy)
    for gemm, order in (
        (_wrapper(offs), ["pack", "pack", "pack", "fwd"]),
        (_in_forward(offs), ["fwd", "pack", "pack", "pack"]),
    ):
        x, w = _leaves(x0, w0)
        events.clear()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = gemm(x, w)
        torch.cuda.synchronize()
        assert [kind for kind, _ in events] == order
        packed = [t for kind, t in events if kind == "pack"]
        for t, src in zip(packed, (x, w, offs), strict=True):
            assert t.data_ptr() == src.data_ptr(), (t.shape, src.shape)
            assert t.shape == src.shape and t.dtype == src.dtype
        assert [p for kind, p in events if kind == "fwd"] == [y.data_ptr()]
        assert type(y.grad_fn).__name__ == "CakeGroupedDetMmBackward"
        assert torch.equal(y.detach(), want)


@pytest.mark.parametrize("case", list(AUTOGRAD_CASES))
@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_checkpoint_early_stop_skips_the_forward_launch(early_stop, case, monkeypatch):
    """Under ``torch.utils.checkpoint`` (non-reentrant) the wrapper launches the forward
    once with the early stop on (torch's default: the recomputation stops at the third
    saved input, before the launch) and twice with it off; the control arm (direct
    ``apply``) twice either way.  ``Y``, ``dX`` and ``dW`` are bitwise the plain step's
    in both arms, which are within the budget of the fp64 reference."""
    device = _require_autograd_programs()
    sizes, (n, k) = AUTOGRAD_CASES[case]
    x0, w0, g, offs = _inputs(sizes, n, k, device, seed=13)
    calls = _count_launches(monkeypatch)
    plain = _step(_wrapper(offs), _leaves(x0, w0), (g,))
    assert calls["fwd"] == [(n, k)]
    (y,), (dx, dw) = plain
    _assert_bf16_close(y, _reference("fwd", x0, w0, sizes))
    _assert_bf16_close(dx, _reference("dgrad", g, w0, sizes))
    _assert_bf16_close(dw, _reference("wgrad", g, x0, sizes))
    for arm, gemm, forwards in (
        ("wrapper", _wrapper(offs), 1 if early_stop else 2),
        ("control", _in_forward(offs), 2),
    ):
        calls["fwd"].clear()
        got = _step(
            gemm, _leaves(x0, w0), (g,), ckpt="non_reentrant", early_stop=early_stop
        )
        assert calls["fwd"] == [(n, k)] * forwards, arm
        _assert_same_step(got, plain)


@pytest.mark.parametrize("case", list(AUTOGRAD_CASES))
@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_checkpoint_early_stop_skips_the_forward_kernels(early_stop, case):
    """The device side of ``test_checkpoint_early_stop_skips_the_forward_launch``: after a
    warm-up (programs loaded, stream workspaces and queue probe done) the CUDA kernels of
    a checkpointed step through the wrapper are those of the plain step with the early
    stop on, and those plus the kernels of one bare forward launch with it off."""
    device = _require_autograd_programs()
    _require_profiled_kernels(device)
    sizes, (n, k) = AUTOGRAD_CASES[case]
    x0, w0, g, offs = _inputs(sizes, n, k, device, seed=13)
    wrapper = _wrapper(offs)
    out = torch.empty(x0.shape[0], n, dtype=torch.bfloat16, device=device)
    bare = lambda: cake_autograd._fwd_op(x0, w0, offs, out)
    plain = lambda: _step(wrapper, _leaves(x0, w0), (g,))
    checkpointed = lambda: _step(
        wrapper, _leaves(x0, w0), (g,), ckpt="non_reentrant", early_stop=early_stop
    )
    for warm in (bare, plain, checkpointed):
        warm()
    fwd = _cuda_kernel_names(bare)
    assert fwd, "the forward launch records no kernel"
    want = _cuda_kernel_names(plain)
    if not early_stop:
        want += fwd
    assert _cuda_kernel_names(checkpointed) == want


@pytest.mark.parametrize("region", list(REGION_FORWARDS))
def test_checkpoint_recomputes_the_forward_when_a_later_op_saves(region, monkeypatch):
    """The early stop skips the recomputed GEMM only when no op after it in the region
    saves a tensor, as for a built-in op: ``silu`` of its output saves that output (2
    forward launches), ``* 2.0`` saves nothing (1), ``x.sin()`` after it saves ``x`` (2).
    The outputs and gradients are bitwise the plain step's."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=42)
    grads = _region_grads(region, g, x0)
    fn = _region(_wrapper(offs), region)
    calls = _count_launches(monkeypatch)
    plain = _step(fn, _leaves(x0, w0), grads)
    assert calls["fwd"] == [SMALL_SHAPE]
    calls["fwd"].clear()
    checkpointed = _step(fn, _leaves(x0, w0), grads, ckpt="non_reentrant")
    assert calls["fwd"] == [SMALL_SHAPE] * REGION_FORWARDS[region]
    _assert_same_step(checkpointed, plain)


@pytest.mark.parametrize("region", list(REGION_FORWARDS))
@pytest.mark.parametrize("impl", ["cake_det", "torch_grouped_mm"])
def test_checkpoint_recompute_kernels_match_torch_grouped_mm(impl, region):
    """The regions of ``test_checkpoint_recomputes_the_forward_when_a_later_op_saves``
    through this package and through ``torch._grouped_mm`` (whose autograd saves its
    inputs before its kernel): after a warm-up the CUDA kernels of a checkpointed step are
    those of the plain step plus ``REGION_FORWARDS[region] - 1`` times those of one bare
    forward GEMM, the same pattern for both."""
    device = _require_autograd_programs()
    _require_profiled_kernels(device)
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=43)
    grads = _region_grads(region, g, x0)
    if impl == "cake_det":
        gemm = _wrapper(offs)
        out = torch.empty(
            x0.shape[0], SMALL_SHAPE[0], dtype=torch.bfloat16, device=device
        )
        bare = lambda: cake_autograd._fwd_op(x0, w0, offs, out)
    else:
        if not hasattr(torch, "_grouped_mm"):
            pytest.skip("this torch has no torch._grouped_mm")
        gemm = lambda a, b: torch._grouped_mm(a, b.transpose(1, 2), offs=offs)
        bare = lambda: gemm(x0, w0)
    fn = _region(gemm, region)
    plain = lambda: _step(fn, _leaves(x0, w0), grads)
    checkpointed = lambda: _step(fn, _leaves(x0, w0), grads, ckpt="non_reentrant")
    try:
        for warm in (bare, plain, checkpointed):
            warm()
    except RuntimeError as exc:
        if impl == "cake_det":
            raise
        arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
        pytest.skip(f"torch._grouped_mm does not run this step on {arch}: {exc}")
    fwd = _cuda_kernel_names(bare)
    assert fwd, f"a bare {impl} forward records no kernel"
    want = _cuda_kernel_names(plain)
    for _ in range(REGION_FORWARDS[region] - 1):
        want += fwd
    assert _cuda_kernel_names(checkpointed) == want


@pytest.mark.parametrize("variant", list(MOE_BLOCK_FORWARDS))
def test_checkpoint_moe_block_skips_only_the_last_gemm(variant, monkeypatch):
    """A checkpointed MoE expert block (``_moe_block``, both projections through the
    wrapper): with the early stop on the gate / up projection is recomputed (SiLU saves
    its output) and the down projection is not; a routing multiply whose probabilities
    require grad saves the down projection's output, which is then recomputed too (the
    scope of the skip, as for built-in ops); a selective checkpoint that saves the
    ``aten.mul.Tensor`` outputs launches as the plain one.  The output and the gradients
    of ``x``, ``w1``, ``w2`` (and of the probabilities) are bitwise the plain step's."""
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    device = _require_autograd_programs()
    gate_up, down = (2 * MOE_INTER, MOE_HIDDEN), (MOE_HIDDEN, MOE_INTER)
    x0, w1, _g, offs = _inputs(SMALL_SIZES, *gate_up, device, seed=44)
    _x, w2, g, _offs = _inputs(SMALL_SIZES, *down, device, seed=45)
    inputs = [x0, w1, w2]
    if variant == "routed":
        gen = torch.Generator(device=device).manual_seed(46)
        probs = torch.rand(x0.shape[0], 1, generator=gen, device=device)
        inputs.append(probs.to(torch.bfloat16))

    def policy(ctx, op, *args, **kwargs):
        if op == torch.ops.aten.mul.Tensor:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    context_fn = None
    if variant == "selective":
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy)
    block = _moe_block(_wrapper(offs), variant == "routed")
    calls = _count_launches(monkeypatch)
    plain = _step(block, _leaves(*inputs), (g,))
    assert collections.Counter(calls["fwd"]) == {gate_up: 1, down: 1}
    calls["fwd"].clear()
    checkpointed = _step(
        block, _leaves(*inputs), (g,), ckpt="non_reentrant", context_fn=context_fn
    )
    want = dict(zip((gate_up, down), MOE_BLOCK_FORWARDS[variant], strict=True))
    assert collections.Counter(calls["fwd"]) == want
    _assert_same_step(checkpointed, plain)


@pytest.mark.parametrize("structure", list(NESTED_FORWARDS))
def test_reentrant_and_nested_checkpoints(structure, monkeypatch):
    """A reentrant checkpoint recomputes the forward (2 launches); a non-reentrant one
    nested in another launches it once (the outer recomputation stops at the inner
    checkpoint's input saves, the inner one before the launch), twice when an outer op
    saves the inner output (``.sin()``).  The outputs and gradients are bitwise the plain
    step's."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=46)
    wrapper = _wrapper(offs)

    def inner(a, b):
        return torch.utils.checkpoint.checkpoint(wrapper, a, b, use_reentrant=False)

    plain_fn, fn, ckpt = wrapper, inner, "non_reentrant"
    if structure == "reentrant":
        fn, ckpt = wrapper, "reentrant"
    elif structure == "nested_then_save":
        plain_fn = lambda a, b: wrapper(a, b).sin()
        fn = lambda a, b: inner(a, b).sin()
    calls = _count_launches(monkeypatch)
    plain = _step(plain_fn, _leaves(x0, w0), (g,))
    calls["fwd"].clear()
    checkpointed = _step(fn, _leaves(x0, w0), (g,), ckpt=ckpt)
    assert calls["fwd"] == [SMALL_SHAPE] * NESTED_FORWARDS[structure]
    _assert_same_step(checkpointed, plain)


def test_checkpointed_gradient_accumulation_and_retain_graph(monkeypatch):
    """Two backward passes, each step under a non-reentrant checkpoint: one forward
    launch per step, ``x.grad`` the bf16 sum of the two data gradients and ``w.grad``
    that of the two weight gradients of the package functions (bitwise), the empty
    group's exact zeros.  One checkpointed graph backpropagated twice
    (``retain_graph=True`` first): one forward launch in all through the wrapper (both
    recomputations stop before it), three through the control arm, and ``x.grad`` twice
    the data gradient (bitwise); a third backward raises."""
    device = _require_autograd_programs()
    _fresh_stream_workspaces(monkeypatch)
    sizes, (n, k) = AUTOGRAD_CASES["small"]
    x0, w0, g1, offs = _inputs(sizes, n, k, device, seed=14)
    g2 = _inputs(sizes, n, k, device, seed=15)[2]
    calls = _count_launches(monkeypatch)
    wrapper = _wrapper(offs)
    x, w = _leaves(x0, w0)
    for g in (g1, g2):
        y = torch.utils.checkpoint.checkpoint(wrapper, x, w, use_reentrant=False)
        y.backward(g)
    torch.cuda.synchronize()
    assert calls["fwd"] == [(n, k)] * 2
    dx1 = grouped_det_gemm_dgrad(g1, w0, offs)
    dx2 = grouped_det_gemm_dgrad(g2, w0, offs)
    dw1 = grouped_det_gemm_wgrad(g1, x0, offs)
    dw2 = grouped_det_gemm_wgrad(g2, x0, offs)
    assert torch.equal(x.grad, dx1 + dx2)
    assert torch.equal(w.grad, dw1 + dw2)
    assert not w.grad[sizes.index(0)].any()
    for gemm, forwards in ((wrapper, 1), (_in_forward(offs), 3)):
        calls["fwd"].clear()
        x, w = _leaves(x0, w0)
        y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
        y.backward(g1, retain_graph=True)
        y.backward(g1)
        torch.cuda.synchronize()
        assert calls["fwd"] == [(n, k)] * forwards
        assert torch.equal(x.grad, dx1 + dx1)
        with pytest.raises(RuntimeError, match="a second time"):
            y.backward(g1)


@pytest.mark.parametrize("frozen", ["w", "x"])
def test_checkpoint_with_a_frozen_operand(frozen, monkeypatch):
    """A checkpointed step with a frozen operand, under torch's deterministic mode (no
    alert: both reductions are deterministic), launches the forward once and only the
    gradient it needs: a frozen ``w`` the data gradient alone, a frozen ``x`` the weight
    gradient alone.  The gradient has the bits of the package function."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=47)
    x = x0.clone().requires_grad_(frozen != "x")
    w = w0.clone().requires_grad_(frozen != "w")
    calls = _count_launches(monkeypatch)
    with _torch_deterministic():
        _step(_wrapper(offs), [x, w], (g,), ckpt="non_reentrant")
    assert calls["fwd"] == [SMALL_SHAPE]
    if frozen == "w":
        assert calls["dgrad"] == [SMALL_SHAPE] and calls["wgrad"] == []
        assert w.grad is None
        assert torch.equal(x.grad, grouped_det_gemm_dgrad(g, w0, offs))
    else:
        assert calls["wgrad"] == [SMALL_SHAPE] and calls["dgrad"] == []
        assert x.grad is None
        assert torch.equal(w.grad, grouped_det_gemm_wgrad(g, x0, offs))


@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
def test_wrapper_forward_cuda_graph_replay(grad):
    """A CUDA graph captured around ``cake_grouped_det_mm`` (with leaves that require
    grad and grad mode on, or under ``no_grad``) replays the forward launch the wrapper
    makes after ``apply``: after new group boundaries (same ``sum_m``) are written into
    ``offs``, every replay gives the bits of the eager wrapper on those boundaries."""
    device = _require_program("fwd")
    first, second = list(SMALL_SIZES), [1, 500, 250, 250]
    x0, w0, _g, offs = _inputs(first, *SMALL_SHAPE, device, seed=48)
    x = x0.clone().requires_grad_(grad)
    w = w0.clone().requires_grad_(grad)
    wrapper = _wrapper(offs)
    grad_mode = contextlib.nullcontext if grad else torch.no_grad
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with grad_mode(), torch.cuda.stream(side):
        wrapper(x, w)  # program load before the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with grad_mode(), torch.cuda.graph(graph):
        y = wrapper(x, w)
    assert (y.grad_fn is not None) == grad
    for sizes in (first, second, first):
        offs.copy_(_offs(sizes, device))
        y.detach().fill_(float("nan"))
        graph.replay()
        eager = pkg.cake_grouped_det_mm(x0, w0, offs)
        torch.cuda.synchronize()
        assert torch.equal(y.detach(), eager), sizes
        _assert_bf16_close(y.detach(), _reference("fwd", x0, w0, sizes))


@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpoint"])
def test_wrapper_retains_the_memory_of_the_in_forward_path(checkpointed):
    """Against the control arm (direct ``apply``), plain and under a non-reentrant
    checkpoint, after a warm-up: the same allocated memory after the forward, a backward
    peak no higher, and after the backward and ``del y`` the two gradients alone (dropping
    them returns to the memory before the forward)."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=49)
    ckpt = "non_reentrant" if checkpointed else None

    def measure(gemm):
        x, w = _leaves(x0, w0)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(device)
        if checkpointed:
            y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
        else:
            y = gemm(x, w)
        torch.cuda.synchronize()
        forward = torch.cuda.memory_allocated(device) - base
        torch.cuda.reset_peak_memory_stats(device)
        y.backward(g)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) - base
        del y
        kept = torch.cuda.memory_allocated(device) - base
        grads = x.grad.untyped_storage().nbytes() + w.grad.untyped_storage().nbytes()
        x.grad = w.grad = None
        assert torch.cuda.memory_allocated(device) == base
        return forward, peak, kept, grads

    arms = (_wrapper(offs), _in_forward(offs))
    for gemm in arms:  # stream workspaces and the queue probe
        _step(gemm, _leaves(x0, w0), (g,), ckpt=ckpt)
    new, old = (measure(gemm) for gemm in arms)
    assert new[0] == old[0], (new, old)
    assert new[1] <= old[1], (new, old)
    assert new[2] == new[3] and old[2] == old[3], (new, old)


def test_wrapper_does_not_synchronize():
    """After a warm-up, a forward and a checkpointed step (the forward, then the backward
    with its recomputation) through the wrapper and through the control arm run under
    ``torch.cuda.set_sync_debug_mode("error")``, so without a host synchronisation, and
    give the same gradients."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=50)
    arms = (_wrapper(offs), _in_forward(offs))
    for gemm in arms:
        _step(gemm, _leaves(x0, w0), (g,), ckpt="non_reentrant")
    leaves = [_leaves(x0, w0) for _ in arms]
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for gemm, (x, w) in zip(arms, leaves, strict=True):
            gemm(x, w)
            y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
            y.backward(g)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    (x_new, w_new), (x_old, w_old) = leaves
    assert torch.equal(x_new.grad, x_old.grad)
    assert torch.equal(w_new.grad, w_old.grad)


def test_wrapper_under_jit_trace(monkeypatch):
    """Under ``torch.jit.trace`` the wrapper keeps the launch inside ``forward`` (a trace
    records ``apply`` only): the traced function replayed on a new ``x`` launches the
    forward once and gives the eager wrapper's bits."""
    device = _require_program("fwd")
    x0, w0, _g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=51)
    x1 = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=52)[0]
    traced = torch.jit.trace(
        lambda a, b, o: pkg.cake_grouped_det_mm(a, b, o),
        (x0, w0, offs),
        check_trace=False,
    )
    calls = _count_launches(monkeypatch)
    out = traced(x1, w0, offs)
    torch.cuda.synchronize()
    assert calls["fwd"] == [SMALL_SHAPE]
    eager = pkg.cake_grouped_det_mm(x1, w0, offs)
    assert torch.equal(out, eager)
    assert not torch.equal(eager, grouped_det_gemm_fwd(x0, w0, offs))


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_wrapper_under_torch_compile(backend):
    """``torch.compile`` (``backend``) of a non-reentrant checkpointed region through the
    wrapper gives the bits of the eager checkpointed step (``Y``, ``dX``, ``dW``); under
    Dynamo the wrapper keeps the launch inside ``forward`` (Dynamo cannot lift the
    deferred launch out of the checkpointed region).  Where the compiled step raises, the
    control arm (direct ``apply``) must raise the same exception type, and the case is
    skipped with that failure as the reason (a limit the wrapper did not introduce); a
    failure of the wrapper alone fails."""
    import torch._dynamo

    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=53)
    eager = _step(_wrapper(offs), _leaves(x0, w0), (g,), ckpt="non_reentrant")

    def compiled_step(gemm):
        def region(a, b):
            return torch.utils.checkpoint.checkpoint(gemm, a, b, use_reentrant=False)

        torch._dynamo.reset()
        try:
            step = torch.compile(region, backend=backend)
            return _step(step, _leaves(x0, w0), (g,)), None
        except Exception as exc:  # noqa: BLE001 - compared with the control arm
            return None, exc
        finally:
            torch._dynamo.reset()

    got, failure = compiled_step(_wrapper(offs))
    if failure is None:
        _assert_same_step(got, eager)
        return
    _, control_failure = compiled_step(_in_forward(offs))
    assert type(control_failure) is type(failure), (
        f"torch.compile ({backend}) fails through the wrapper only: {failure!r}; "
        f"control arm: {control_failure!r}"
    )
    pytest.skip(
        f"torch.compile ({backend}) fails the same way through CakeGroupedDetMm.apply: "
        f"{type(failure).__name__}: {str(failure)[:400]}"
    )


def test_direct_apply_keeps_the_in_forward_launch(monkeypatch):
    """A direct ``CakeGroupedDetMm.apply(x, w, offs, deterministic)`` launches the forward
    inside ``forward`` and gives the wrapper's bits (``Y``, ``dX``, ``dW``) with
    ``deterministic`` ``True`` or ``False`` (the same programs); shape and offset errors,
    which the wrapper now raises after autograd saved the inputs, are still
    ``ValueError``."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=54)
    calls = _count_launches(monkeypatch)
    wrapped = _step(_wrapper(offs), _leaves(x0, w0), (g,))
    for deterministic in (True, False):
        direct = _step(_in_forward(offs, deterministic), _leaves(x0, w0), (g,))
        _assert_same_step(direct, wrapped)
    assert calls["fwd"] == [SMALL_SHAPE] * 3
    x, _w = _leaves(x0, w0)
    n, k = SMALL_SHAPE
    for bad_w, bad_offs in (
        (w0[:, : n - 128], offs),  # N % 256 != 0
        (w0[:, :, : k // 2], offs),  # K of w differs from x's
        (w0, offs.to(torch.int64)),
    ):
        with pytest.raises(ValueError):
            pkg.cake_grouped_det_mm(x, bad_w.clone().requires_grad_(True), bad_offs)
