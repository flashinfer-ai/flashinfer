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

Backend suite of the non-deterministic ragged BF16 MoE grouped GEMM package
(``deterministic=False``).

The forward pass, the data gradient and the 256-wide k tile's weight gradient
run copies of the deterministic package's programs; the 512-wide k tile's
weight gradient adds its split tail in place with unordered fp32 reductions.
Every output is checked against a per-group fp64 reference under the
deterministic package's budgets (bf16: atol = rtol = 1e-2; fp32 weight
gradient: rel-L2 1e-4), as the worst case over repeated launches, and the
split tail's workspaces are checked at rest (zero) after the launches.  Only
this package is imported (through ``cake_package``), so the suite also runs
against a vendored copy of it.
"""

import functools
import math

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

# Backend suite: the package is named through ``cake_package`` only; nothing
# else of flashinfer is imported here.  The facade and the numerical-mode
# checks live in ``test_cake_moe_grouped_gemm_api.py``, the comparisons with
# the deterministic package (its routes and bits) in
# ``test_cake_moe_grouped_gemm_reproducibility.py``.
PACKAGE = "cake.cake_moe_grouped_gemm"
pkg = cake_package(PACKAGE)
cb = cake_package(PACKAGE, "cake_backend")
cake_jit = cake_package(PACKAGE, "cake_jit")
cake_autograd = cake_package(PACKAGE, "cake_autograd")
SUPPORTED_COMPUTE_CAPABILITIES = cb.SUPPORTED_COMPUTE_CAPABILITIES
HOST_PLAN_CONSTANTS = cake_jit.HOST_PLAN_CONSTANTS
PROGRAMS = cake_jit.PROGRAMS
ROUTES = cake_jit.ROUTES

# Every entry point of the package needs deterministic=False.
grouped_gemm_fwd = functools.partial(pkg.grouped_gemm_fwd, deterministic=False)
grouped_gemm_dgrad = functools.partial(pkg.grouped_gemm_dgrad, deterministic=False)
grouped_gemm_wgrad = functools.partial(pkg.grouped_gemm_wgrad, deterministic=False)
prepare_fwd = functools.partial(cb.prepare_grouped_gemm_fwd, deterministic=False)
prepare_dgrad = functools.partial(cb.prepare_grouped_gemm_dgrad, deterministic=False)
prepare_wgrad = functools.partial(cb.prepare_grouped_gemm_wgrad, deterministic=False)

ATOL = RTOL = 1e-2  # bf16 outputs
FP32_REL_L2 = 1e-4  # fp32 weight gradient vs the fp64 reference
REPEATS = 4  # launches per accuracy case of the unordered weight gradient (worst case)
GARBAGE_BYTE = 0x5A
SENTINEL_WORD = 0x5A5A5A5A  # int32 fill of kept workspaces a launch must not touch
DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}
# The SM counts of the parts of each architecture (B200 / B300: 148, GB300: 160,
# R200: 212 / 216); the CPU plan checks run at each.
SM_COUNTS = {"sm_100a": (148, 160), "sm_103a": (148, 160), "sm_107a": (212, 216)}
ARCH_SM_COUNTS = [(arch, sms) for arch, counts in SM_COUNTS.items() for sms in counts]
# ~1 s at 2 GHz: a kernel resident on another stream for the whole of the
# concurrent-stream test (flashinfer-ai/flashinfer#7038, #7548).
RESIDENT_SLEEP_CYCLES = 2_000_000_000

# Group size lists: an empty group, a one-row group, sizes that are not tile
# multiples, and a long-reduction case that can select the 512 k tile.
RAGGED_E4 = [300, 0, 1, 700]
RAGGED_E6 = [1, 257, 511, 4099, 0, 33]
TINY_E5 = [1, 2, 3, 0, 4]
LONG_E3 = [6145, 0, 12289]
# Groups shorter than one 64-row k-step (flashinfer-ai/flashinfer#6570).
SMALL_E512 = [32] * 512
SMALL_GROUP_LAUNCHES = 200
# Short groups beside long ones: with the 512-wide k tile forced, the split tail
# mixes the tiles of one-k-step groups and of long groups.
MIXED_E128 = [32] * 120 + [40000] * 8
# More than every architecture's k512_min_rows_per_group (and k256_tail_min_rows)
# rows per group on average, with an empty group: the 512-wide k tile.
K512_E3 = [16400, 0, 16666]
# The group counts every op is checked at (#5678: E in {1, 4, 8, 16, 32}), ragged
# with an empty and a one-row group from E = 4, averaging k512_min_rows_per_group.
# At (N, K) = GROUP_SHAPE the weight gradient of at least one group count splits
# its unordered tail at every SM count of SM_COUNTS
# (test_group_count_sweep_splits_the_tail_at_every_sm_count).
GROUP_COUNTS = (1, 4, 8, 16, 32)
GROUP_ROWS = 4096
GROUP_SHAPE = (1024, 1024)
# Two weight-gradient shapes (N, K) over K512_E3 whose unordered tails split, the
# second needing strictly more tail sums and tickets than the first at every SM
# count of SM_COUNTS (test_changed_shapes_grow_the_tail_at_every_sm_count).
CHANGED_SHAPES = ((4864, 1024), (5120, 1024))
# The 256-wide k tile's wave-synchronised weight gradient (``main_wave`` where
# ``wave_sync_archs[256]`` lists the architecture, sm_100a / sm_103a; ``main``
# elsewhere): average rows below k512_min_rows_per_group, K >= wave_sync_min_k.
K256_WAVE_SIZES = [2048] * 8
K256_WAVE_SHAPE = (4096, 4096)  # (N, K)
ROW_PAD = 256  # extra columns of the wider tensors row-padded operands are cut from
# The MoE expert rows of the deterministic package's measured thresholds.
EXPERT_ROWS = 131072
EXPERT_GROUPS = (4, 8, 16, 32)
# The #5678 weight-gradient rows (the benchmark rows of the 512-wide k tile's routes):
# (label, sizes, (N, K)).
BENCH_PROJ = {"gate_up": (4096, 6144), "down": (6144, 2048)}
BENCH_WGRAD_ROWS = (
    ("e4_m112576_balanced", [28128, 28128, 28128, 28192]),
    ("e4_m112576_skewed", [45024, 33760, 22528, 11264]),
    ("e4_m112576_one_empty", [56288, 33760, 22528, 0]),
    ("e4_m136064_balanced", [34016] * 4),
    ("e4_m136064_skewed", [54432, 40832, 27200, 13600]),
    ("e4_m136064_one_empty", [68032, 40832, 27200, 0]),
    ("e8_m131072_balanced", [16384] * 8),
    ("e16_m131072_balanced", [8192] * 16),
    ("e32_m131072_balanced", [4096] * 32),
)
K512_FORMS = (
    "main",
    "main_wave",
    "main_halves",
    "main_wave_halves",
    "main_halves_k256t",
)


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


def _arch(device):
    return SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]


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


def _assert_wgrad_close(out, ref):
    """The deterministic package's weight-gradient budget of ``out``'s dtype; returns the error."""
    if out.dtype == torch.bfloat16:
        _assert_bf16_close(out, ref)
        return float((out.float() - ref).abs().max())
    rel_l2 = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
    assert rel_l2 <= FP32_REL_L2, f"fp32 weight gradient rel-L2 {rel_l2:.3e}"
    return rel_l2


def _assert_empty_groups_zero(out, sizes):
    for e, (s, t) in enumerate(_bounds(sizes)):
        if t == s:
            assert torch.equal(out[e], torch.zeros_like(out[e])), (
                f"empty group {e} must be exact zeros"
            )


def _fresh_stream_workspaces(monkeypatch):
    """Start the test with no kept stream workspace and no queue probe (both restored
    after it): torch reuses its pooled stream handles, so a new ``torch.cuda.Stream``
    could otherwise find the workspace an earlier test kept for the same handle."""
    monkeypatch.setattr(cb, "_stream_workspaces", {})
    monkeypatch.setattr(cb, "_queue_splits", {})


def _u32_words(tensor):
    """The words of a u32 workspace as Python ints."""
    return [v & 0xFFFFFFFF for v in tensor.view(torch.int32).tolist()]


def _workspace_kinds(launch):
    return {kind for item in launch.launches for _, kind, _size in item.workspace_slots}


def _assert_tail_at_rest(device, stream=None):
    """The kept split-tail sums and tickets of ``stream`` are all zero."""
    for kind in ("tail_sums", "tail_tickets"):
        buffer = cb.stream_workspace(kind, device, stream)
        assert buffer is not None, kind
        assert int(buffer.view(torch.int32).count_nonzero()) == 0, (
            f"{kind} is not at rest: {int(buffer.view(torch.int32).count_nonzero())} nonzero words"
        )


def _force_form(monkeypatch, form, arch):
    """Constants under which a weight gradient on ``arch`` takes the 512-wide k tile (at
    any rows per group) in stage ``form`` (given a ``K`` of ``_form_ks(form)``); False
    where the arch has no such form."""
    c = HOST_PLAN_CONSTANTS
    halves = form in ("main_halves", "main_wave_halves", "main_halves_k256t")
    wave = form in ("main_wave", "main_wave_halves")
    if halves and arch not in c["k512_release_archs"]:
        return False
    if wave and arch not in c["wave_sync_archs"][512]:
        return False
    if form == "main_halves_k256t" and arch not in c["k256_tail_archs"]:
        return False
    monkeypatch.setitem(c["k512_min_rows_per_group"], arch, 1)
    if form == "main_halves_k256t":
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k256_tail_min_rows", 1)
    if not halves:
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k512_release_archs", [])
    if not wave:
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wave_sync_min_k", 1 << 30)
    if form != "main_halves_k256t":
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k256_tail_archs", [])
    return True


def _form_ks(form):
    """Reductions ``K`` a 512-wide k tile shape of ``form`` may take."""
    c = HOST_PLAN_CONSTANTS
    if form in ("main_wave", "main_wave_halves"):
        return [int(c["wave_sync_min_k"]) + 512 * i for i in range(4)]
    if form == "main_halves_k256t":
        return [k for k in (2048, 1536, 1024) if k <= int(c["k256_tail_max_k"])]
    return [1024, 1536, 2048, 3072]


def _wgrad_plan(sizes, n, k, arch, sm_count, out_dtype=torch.bfloat16):
    """``(main stage, plan)`` that ``prepare_wgrad`` takes for group sizes ``sizes`` and
    ``(N, K)`` on ``arch`` with ``sm_count`` SMs, under the constants as the test set them."""
    c = HOST_PLAN_CONSTANTS
    sum_m, e = sum(sizes), len(sizes)
    tile_k = cb.select_wgrad_tile(k, arch, sum_m, e, c)
    name = cake_jit.record_name(
        "wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=tile_k
    )
    stage, _wave = cb.select_wgrad_stage(
        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=e
    )
    plan = cb.plan_wgrad(
        sum_m=sum_m,
        num_groups=e,
        n=n,
        k=k,
        tile_k=tile_k,
        sm_count=sm_count,
        reduce_threads=256,
        k256_tail=stage == "main_halves_k256t",
        unordered_tail=cb.wgrad_unordered_tail(tile_k, arch, c),
        constants=c,
    )
    return stage, plan


def _split_tail_shape(device, sizes, ks):
    """``(N, K)`` over ``ks`` (and ``N`` up to 48 column blocks) whose weight-gradient plan
    on ``device`` takes the 512-wide k tile and splits its unordered tail
    (``tail_splits > 1``); None when no candidate does."""
    arch, sms = _arch(device), cb.sm_count(device)
    for k in ks:
        for blocks in range(1, 49):
            _stage, plan = _wgrad_plan(sizes, 256 * blocks, k, arch, sms)
            if plan["unordered_tail"] and plan["tail_splits"] > 1:
                return 256 * blocks, k
    return None


def _row_padded(t):
    """``t`` as the leading columns of a wider tensor whose other columns hold NaN: the
    same values, row stride ``t.size(-1) + ROW_PAD``."""
    cols = t.size(-1)
    wide = torch.full(
        (*t.shape[:-1], cols + ROW_PAD), float("nan"), dtype=t.dtype, device=t.device
    )
    wide[..., :cols] = t
    return wide[..., :cols]


def _k256_wave_stage(device, out_dtype=torch.bfloat16):
    """The main stage of the ``K256_WAVE_*`` weight gradient on ``device``, checked
    against the host plan constants: the 256-wide k tile, ``main_wave`` where the
    architecture is in ``wave_sync_archs[256]``, else ``main``."""
    arch = _arch(device)
    stage, plan = _wgrad_plan(
        K256_WAVE_SIZES, *K256_WAVE_SHAPE, arch, cb.sm_count(device), out_dtype
    )
    wave = arch in HOST_PLAN_CONSTANTS["wave_sync_archs"][256]
    assert plan["tile_k"] == 256, plan
    assert stage == ("main_wave" if wave else "main"), (arch, stage)
    return stage


def _ragged(e, rows, seed):
    """``e`` ragged group sizes averaging ``rows`` (sum ``e * rows``), with an empty and a
    one-row group from ``e = 4``."""
    if e == 1:
        return [rows]
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(e, generator=gen) + 0.25
    if e >= 4:
        weights[1] = 0.0
        weights[2] = 0.0
    total = e * rows - (1 if e >= 4 else 0)
    sizes = [int(v) for v in (weights / weights.sum() * total).floor()]
    sizes[0] += total - sum(sizes)
    if e >= 4:
        sizes[2] = 1
    return sizes


# ---------------------------------------------------------------------------
# Registry and host plan (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sm_count", [148, 160, 212])
@pytest.mark.parametrize(
    "sizes", [RAGGED_E4, RAGGED_E6, LONG_E3, K512_E3, [131072] * 1, [4096] * 32]
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
        assert plan["clusters"] == grid_x // cta_group
        assert plan["c0"] == plan["clusters"] // 2
        assert plan["c0"] + plan["c1"] == plan["clusters"] and plan["c1"] >= 1
    atomic = c["atomic_tail_plan"]
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for n, k in ((4096, 6144), (6144, 2048), (256, 256), (1536, 3072)):
            tile_k = cb.select_wgrad_tile(k, arch, sum_m, e, c)
            assert tile_k in (256, 512) and k % tile_k == 0
            for k256_tail in (False, True) if tile_k == 512 else (False,):
                unordered = cb.wgrad_unordered_tail(tile_k, arch, c)
                plan = cb.plan_wgrad(
                    sum_m=sum_m,
                    num_groups=e,
                    n=n,
                    k=k,
                    tile_k=tile_k,
                    sm_count=sm_count,
                    reduce_threads=256,
                    k256_tail=k256_tail,
                    unordered_tail=unordered,
                    constants=c,
                )
                tail_tile_k = cb.K256_TAIL_TILE_K if k256_tail else tile_k
                clusters = plan["clusters"]
                assert plan["grid"][0] == clusters * cta_group
                assert plan["tail_tile_k"] == tail_tile_k
                assert plan["unordered_tail"] is unordered
                assert plan["num_tail"] == (
                    (e * (n // int(c["block_n"])) * (k // tail_tile_k)) % clusters
                    if int(c["wgrad_tail_split"])
                    else 0
                )
                assert 0 <= plan["num_tail"] < clusters
                splits = plan["tail_splits"]
                if plan["num_tail"] == 0:
                    assert splits == 1
                if not unordered:
                    # The 256-wide k tile: the deterministic package's plan.
                    assert 1 <= splits <= int(c["tail_plan"]["max_splits"])
                    assert plan["launches"] == 1 + int(splits > 1)
                    assert (plan["tail_reduce_grid"] is None) == (splits == 1)
                    assert plan["tail_sum_elems"] == plan["tail_ticket_words"] == 0
                    continue
                # The unordered tail: atomic_tail_plan, one launch, no reduce grid.
                assert 1 <= splits <= int(atomic["max_splits"])
                if plan["num_tail"]:
                    assert splits == cb.wgrad_atomic_tail_plan(
                        plan["num_tail"],
                        clusters,
                        sum_m,
                        e,
                        tail_tile_k,
                        block_k=int(c["block_k"]),
                        cluster_m=int(c["cluster_m"]),
                        max_splits=int(atomic["max_splits"]),
                        sm_clock_ghz=float(atomic["sm_clock_ghz"]),
                        dram_tbps=float(atomic["dram_tbps"]),
                        launch_us=float(atomic["launch_us"]),
                        cycles_per_step=atomic["cycles_per_step"],
                    )
                assert plan["launches"] == 1 and plan["tail_reduce_grid"] is None
                split = splits > 1
                assert plan["tail_sum_elems"] == (
                    plan["num_tail"] * int(c["cluster_m"]) * tail_tile_k if split else 0
                )
                assert plan["tail_ticket_words"] == (
                    plan["num_tail"] * cta_group * cb.TAIL_TICKETS_PER_CTA
                    if split
                    else 0
                )
                assert plan["raster_rows"] in (0, 1)
                if k256_tail:
                    assert plan["raster_rows"] == 0


def test_atomic_tail_plan_constants():
    """``atomic_tail_plan`` carries the keys of the ordered ``tail_plan`` (the parent cost
    model), with a cycles-per-step entry for both k tiles."""
    c = HOST_PLAN_CONSTANTS
    atomic, ordered = c["atomic_tail_plan"], c["tail_plan"]
    assert set(atomic) == set(ordered)
    assert int(atomic["max_splits"]) >= 1
    assert float(atomic["sm_clock_ghz"]) > 0 and float(atomic["dram_tbps"]) > 0
    assert float(atomic["launch_us"]) >= 0
    assert {256, 512} <= set(atomic["cycles_per_step"])
    assert cb.TAIL_TICKETS_PER_CTA >= 8  # the half-release builds' epilogue warps


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


def test_unordered_tail_archs_bind_the_unordered_k512_builds():
    """``unordered_tail_archs`` names the architectures whose 512-wide k tile routes bind
    the unordered-tail builds (``tail_sums`` and ``tail_tickets``, no ``tail_reduce``
    stage).  The registry holds no ordered k512 build, so the list names every
    architecture of the k512 routes (leaving one out needs the deterministic k512
    builds and a ``tail_reduce`` stage on those routes); the routes agree with the list
    either way.  ``wgrad_unordered_tail`` follows the list, and the 256-wide k tile
    always binds the deterministic programs."""
    c = HOST_PLAN_CONSTANTS
    assert set(c["unordered_tail_archs"]) <= set(
        SUPPORTED_COMPUTE_CAPABILITIES.values()
    )
    for name, route in ROUTES.items():
        if route["op"] != "wgrad":
            continue
        if name.endswith("_k512"):
            assert set(route["arches"]) <= set(c["unordered_tail_archs"]), name
        tile_k = 512 if name.endswith("_k512") else 256
        for arch in route["arches"]:
            unordered = cb.wgrad_unordered_tail(tile_k, arch, c)
            assert unordered == (tile_k == 512 and arch in c["unordered_tail_archs"])
            assert cake_jit.stage_served(name, "tail_reduce", arch) == (not unordered)
            for stage in route["stages"]:
                program = PROGRAMS[stage["program"]]
                if arch not in program["arches"] or stage["name"] == "tail_reduce":
                    continue
                plan = [tuple(item) for item in program["arg_plan"]]
                assert (("buffer", "tail_sums") in plan) == unordered, (name, arch)
                assert (("buffer", "tail_tickets") in plan) == unordered, (name, arch)
                assert (("buffer", "partials") in plan) == (not unordered), (name, arch)


def test_program_choice_follows_the_host_plan_constants():
    """The deterministic package's rules select the stages: the 512-column tile of
    ``fwd`` / ``dgrad``, its evict-first and 3 / 3 head / tail twins, the wave builds,
    the half-release builds and the k256-tail twin; each listed architecture's routes
    serve the stage the constant selects (the 512-wide k tile's as unordered-tail
    builds, without a tail reduce)."""
    c = HOST_PLAN_CONSTANTS
    block_n, wide_n = int(c["block_n"]), int(c["wide_block_n"])
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        wide_archs = c[f"wide_{op}_archs"]
        for arch in ROUTES[name]["arches"]:
            if arch in wide_archs:
                assert cake_jit.stage_served(name, "main_wide", arch), (name, arch)
            ef_key = "dgrad_evict_first_archs" if op == "dgrad" else "fwd_lwh_ef_archs"
            if arch in c[ef_key]:
                assert cake_jit.stage_served(name, "main_wide_ef", arch), (name, arch)
            if arch in c[f"{op}_wide_ts33_archs"]:
                assert cake_jit.stage_served(name, "main_wide_ts33", arch)
            for cols in (4096, 4096 - block_n, block_n):
                n, k = (cols, 512) if op == "fwd" else (512, cols)
                stage, tile_n = cb.select_row_stage(
                    name, op, n, k, arch, c, sum_m=4096, num_groups=4
                )
                wide = cols % wide_n == 0 and arch in wide_archs
                assert tile_n == (wide_n if wide else block_n)
                assert stage.startswith("main_wide") == wide
    min_k = int(c["wave_sync_min_k"])
    for out_dtype in ("bfloat16", "float32"):
        for tile_k in (256, 512):
            name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=tile_k)
            assert name in ROUTES, name
            for arch in ROUTES[name]["arches"]:
                assert cake_jit.stage_served(name, "tail_reduce", arch) == (
                    not cb.wgrad_unordered_tail(tile_k, arch, c)
                ), (name, arch)
                if arch in c["wave_sync_archs"][tile_k]:
                    assert cake_jit.stage_served(name, "main_wave", arch), (name, arch)
                if tile_k == 512 and arch in c["k256_tail_archs"]:
                    assert cake_jit.stage_served(name, "main_halves_k256t", arch)
                for k, sum_m in ((min_k, 4096), (min_k // 2, EXPERT_ROWS)):
                    stage, wave = cb.select_wgrad_stage(
                        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=4
                    )
                    expect_wave = k >= min_k and arch in c["wave_sync_archs"][tile_k]
                    assert wave == expect_wave, (name, arch, k)
                    if tile_k == 512 and arch in c["k512_release_archs"]:
                        assert "halves" in stage, (name, arch, stage)


@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_expert_rows_take_the_measured_wgrad_stage(arch, out_dtype):
    """Over ``EXPERT_ROWS`` routed rows the weight gradient takes the 512-wide k tile
    released in halves (the deterministic package's measured stages); on sm_107a the
    down projection takes its k256-tail twin at 4, 8 and 16 groups."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=512)
    for groups in EXPERT_GROUPS:
        for k in (2048, 6144):
            assert cb.select_wgrad_tile(k, arch, EXPERT_ROWS, groups, c) == 512
            k256t = arch == "sm_107a" and k == 2048 and groups in (4, 8, 16)
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
    """``sched`` is followed by ``c0``, ``c1``; ``wave_sync`` by ``wave_mode``;
    ``tail_sums`` by ``tail_tickets``.  The 512-wide k tile's stages take the split
    tail's sums and tickets and no partials; every other program takes no tail sums."""
    for program in PROGRAMS.values():
        plan = [tuple(item) for item in program["arg_plan"]]
        for buffer, following in (
            ("sched", [("parameter", "c0"), ("parameter", "c1")]),
            ("wave_sync", [("parameter", "wave_mode")]),
            ("tail_sums", [("buffer", "tail_tickets")]),
        ):
            if ("buffer", buffer) in plan:
                at = plan.index(("buffer", buffer))
                assert plan[at + 1 : at + 1 + len(following)] == following
    for name, route in ROUTES.items():
        for stage in route["stages"]:
            program = PROGRAMS[stage["program"]]
            plan = [tuple(item) for item in program["arg_plan"]]
            unordered = route["op"] == "wgrad" and any(
                cb.wgrad_unordered_tail(512 if name.endswith("_k512") else 256, arch)
                for arch in program["arches"]
            )
            assert (("buffer", "tail_sums") in plan) == unordered, (name, stage)
            if unordered:
                assert ("buffer", "partials") not in plan
            if stage["name"] in ("main_wave", "main_wave_halves"):
                assert ("buffer", "wave_sync") in plan
    assert set(cb.SIZED_STREAM_WORKSPACES) <= set(cb.STREAM_WORKSPACES)


def test_every_program_argument_is_bound_by_the_host():
    """Every name in the argument plan of every program a route binds is one the host
    binding of that stage gives (``bindable_arguments``: the binding's arguments, the
    stream workspaces, the row bases of its operands and the grid), with the kind the
    binding passes; so a rendered program that takes a new argument fails here, on the
    CPU, instead of at its first launch."""
    for name, route in ROUTES.items():
        for stage in route["stages"]:
            program = PROGRAMS[stage["program"]]
            allowed = cb.bindable_arguments(route["op"], stage["name"])
            for kind, arg in program["arg_plan"]:
                where = (name, stage["name"], stage["program"], kind, arg)
                assert arg in allowed, where
                if kind == "grid" or arg in cb.GRID_ARGUMENTS:
                    assert kind == "grid" and arg in cb.GRID_ARGUMENTS, where
                if arg in cb.ROW_BASES or arg in cb.STREAM_WORKSPACES:
                    assert kind == "buffer", where
                if arg in ("A", "B"):
                    assert kind == "tma_buffer", where


@pytest.mark.parametrize(
    "op,operand,label", [("wgrad", "A", "g"), ("wgrad", "B", "x"), ("fwd", "A", "x")]
)
def test_row_base_arguments(op, operand, label):
    """Where the build never reads a row base (not ``ROW_BASE_READ_ARCHS``) the row base of
    an operand is a one-element view at its first element, whatever its row stride (no
    copy); where the build reads it, the operand itself, refused with a ``ValueError``
    naming the caller's tensor unless its rows are dense (a single row may have any
    stride)."""
    padded = torch.zeros(5, 136, dtype=torch.bfloat16)[1:, 8:72]
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        if arch in cb.ROW_BASE_READ_ARCHS:
            continue
        view = cb._row_base(padded, op, operand, "program", "main_wave", arch)
        assert view.shape == (1,) and view.is_contiguous(), arch
        assert view.data_ptr() == padded.data_ptr(), arch
    for arch in cb.ROW_BASE_READ_ARCHS:
        dense = torch.zeros(4, 64, dtype=torch.bfloat16)
        assert cb._row_base(dense, op, operand, "program", "main_wave", arch) is dense
        single = torch.zeros(1, 128, dtype=torch.bfloat16)[:, :64]
        assert cb._row_base(single, op, operand, "program", "main_wave", arch) is single
        with pytest.raises(ValueError, match=f"{label} must have dense rows"):
            cb._row_base(padded, op, operand, "program", "main_wave", arch)


class _NoModule:
    """A stand-in for a loaded program module: every entry is a no-op."""

    def __getattr__(self, name):
        return lambda *args: None


def test_bind_accepts_row_strided_operands_of_the_row_base_programs(monkeypatch):
    """``_bind`` binds every program of the registry that takes row bases for row-strided
    operands (the leading columns of wider tensors): on an architecture whose build never
    reads them the row base is a one-element view at the operand's first element; on one
    whose build reads them (``ROW_BASE_READ_ARCHS``) a row-strided operand is refused and
    a dense one bound as is.  Static: the rendered tables, no program is loaded."""
    monkeypatch.setattr(cb, "load_module", lambda *args, **kwargs: _NoModule())
    bound = 0
    for name, route in ROUTES.items():
        for stage in route["stages"]:
            program = PROGRAMS[stage["program"]]
            plan_args = [arg for _kind, arg in program["arg_plan"]]
            rows = [arg for arg in plan_args if arg in cb.ROW_BASES]
            if not rows:
                continue
            host = cb.HOST_ARGUMENTS[cb.host_binding(route["op"], stage["name"])]
            for arch in program["arches"]:
                if arch not in route["arches"]:
                    continue
                kwargs = dict.fromkeys(host, 0)
                kwargs["A"] = torch.zeros(16, 80, dtype=torch.bfloat16)[:, :64]
                kwargs["B"] = torch.zeros(16, 144, dtype=torch.bfloat16)[:, :128]
                kwargs["grid"] = [1, 1, 1]
                if arch in cb.ROW_BASE_READ_ARCHS:
                    with pytest.raises(ValueError, match="dense rows"):
                        cb._bind(name, stage["name"], kwargs, arch)
                    kwargs["A"] = kwargs["A"].contiguous()
                    kwargs["B"] = kwargs["B"].contiguous()
                launch = cb._bind(name, stage["name"], kwargs, arch)
                for arg in rows:
                    value = launch.arguments[plan_args.index(arg)]
                    operand = kwargs[cb.ROW_BASES[arg]]
                    assert value.data_ptr() == operand.data_ptr(), (name, arch, arg)
                    if arch in cb.ROW_BASE_READ_ARCHS:
                        assert value is operand, (name, arch, arg)
                    else:
                        assert value.shape == (1,) and value.is_contiguous()
                bound += 1
    if not bound:
        pytest.skip("no program of the registry takes row bases")


@pytest.mark.parametrize("arch,sm_count", ARCH_SM_COUNTS)
def test_changed_shapes_grow_the_tail_at_every_sm_count(arch, sm_count):
    """Both ``CHANGED_SHAPES`` split their unordered tail over ``K512_E3``, and the
    second needs strictly more tail sums and tickets than the first, so the
    changed-shapes test replaces the kept buffers at every SM count."""
    small, large = (
        _wgrad_plan(K512_E3, n, k, arch, sm_count)[1] for n, k in CHANGED_SHAPES
    )
    for plan in (small, large):
        assert plan["unordered_tail"] and plan["tail_splits"] > 1, plan
    assert large["tail_sum_elems"] > small["tail_sum_elems"]
    assert large["tail_ticket_words"] > small["tail_ticket_words"]


@pytest.mark.parametrize("arch,sm_count", ARCH_SM_COUNTS)
def test_group_count_sweep_splits_the_tail_at_every_sm_count(arch, sm_count):
    """At ``GROUP_SHAPE`` every group count of the sweep takes the 512-wide k tile with
    the unordered tail, and at least one splits it."""
    splits = {}
    for groups in GROUP_COUNTS:
        sizes = _ragged(groups, GROUP_ROWS, seed=groups)
        _stage, plan = _wgrad_plan(sizes, *GROUP_SHAPE, arch, sm_count)
        assert plan["tile_k"] == 512 and plan["unordered_tail"], (groups, plan)
        splits[groups] = plan["tail_splits"]
    assert max(splits.values()) > 1, splits


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
    "sizes,n,k",
    [
        (RAGGED_E4, 512, 256),
        (RAGGED_E6, 512, 256),
        (TINY_E5, 256, 512),
        (LONG_E3, 512, 1024),
    ],
)
def test_wgrad_matches_reference(sizes, n, k, out_dtype):
    """Worst case over repeated launches into an output refilled with NaN."""
    device = _require_program("wgrad", out_dtype)
    x, _w, g, offs = _inputs(sizes, n, k, device)
    launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    ref = _reference("wgrad", g, x, sizes)
    for _ in range(REPEATS):
        launch.out.fill_(float("nan"))
        out = launch.launch()
        assert out.shape == (len(sizes), n, k) and out.dtype == out_dtype
        _assert_wgrad_close(out, ref)
        _assert_empty_groups_zero(out, sizes)


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
@pytest.mark.parametrize("groups", GROUP_COUNTS)
def test_group_counts_match_reference(groups, op):
    """``fwd`` and ``dgrad`` at E in {1, 4, 8, 16, 32} over ragged groups (an empty and a
    one-row group from E = 4) averaging ``GROUP_ROWS`` rows."""
    device = _require_program(op)
    sizes = _ragged(groups, GROUP_ROWS, seed=groups)
    n, k = GROUP_SHAPE
    x, w, g, offs = _inputs(sizes, n, k, device, seed=groups)
    if op == "fwd":
        _assert_bf16_close(grouped_gemm_fwd(x, w, offs), _reference("fwd", x, w, sizes))
    else:
        _assert_bf16_close(
            grouped_gemm_dgrad(g, w, offs), _reference("dgrad", g, w, sizes)
        )


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_group_counts_match_reference(out_dtype, monkeypatch):
    """The weight gradient at E in {1, 4, 8, 16, 32} over the groups of
    ``test_group_counts_match_reference``: every group count takes the 512-wide k tile
    with the unordered tail, at least one splits it on this device, every repeat is
    within the budget, and a split tail's workspaces rest at zero."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    n, k = GROUP_SHAPE
    splits = {}
    for groups in GROUP_COUNTS:
        sizes = _ragged(groups, GROUP_ROWS, seed=groups)
        x, _w, g, offs = _inputs(sizes, n, k, device, seed=groups)
        launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
        plan = launch.plan
        assert plan["tile_k"] == 512 and plan["unordered_tail"], (groups, plan)
        ref = _reference("wgrad", g, x, sizes)
        for _ in range(REPEATS):
            launch.out.fill_(float("nan"))
            _assert_wgrad_close(launch.launch(), ref)
            _assert_empty_groups_zero(launch.out, sizes)
        splits[groups] = plan["tail_splits"]
        if plan["tail_splits"] > 1:
            torch.cuda.synchronize()
            _assert_tail_at_rest(device)
    assert max(splits.values()) > 1, (
        f"no group count splits the unordered tail on this device: {splits}"
    )


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("form", K512_FORMS)
def test_unordered_tail_forms_match_reference(form, out_dtype, monkeypatch):
    """Every 512-wide k tile form the device's architecture serves, on a shape whose plan
    splits its tail: the launch is one kernel with the unordered tail, every repeat (the
    output refilled with NaN, then with garbage bytes) is within the budget against the
    fp64 reference, the empty group is exact zeros, and the stream's tail sums and tickets
    rest at zero after the launches."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    arch = _arch(device)
    if not _force_form(monkeypatch, form, arch):
        pytest.skip(f"no {form} weight-gradient build on {arch}")
    sizes = K512_E3
    shape = _split_tail_shape(device, sizes, _form_ks(form))
    assert shape is not None, f"no candidate shape splits the {form} tail"
    n, k = shape
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=21)
    launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    plan = launch.plan
    assert plan["tile_k"] == 512 and launch.stages == (form,), (launch.stages, plan)
    assert plan["unordered_tail"] and plan["tail_splits"] > 1 and plan["launches"] == 1
    assert {"tail_sums", "tail_tickets"} <= _workspace_kinds(launch)
    ref = _reference("wgrad", g, x, sizes)
    worst = 0.0
    for fill in (None, GARBAGE_BYTE):
        for _ in range(REPEATS):
            if fill is None:
                launch.out.fill_(float("nan"))
            else:
                launch.out.view(torch.uint8).fill_(fill)
            out = launch.launch()
            worst = max(worst, _assert_wgrad_close(out, ref))
            _assert_empty_groups_zero(out, sizes)
    torch.cuda.synchronize()
    _assert_tail_at_rest(device)
    sums = cb.stream_workspace("tail_sums", device)
    tickets = cb.stream_workspace("tail_tickets", device)
    assert sums.dtype == torch.float32 and sums.numel() >= plan["tail_sum_elems"]
    assert tickets.dtype == torch.uint32
    assert tickets.numel() >= plan["tail_ticket_words"]
    print(
        f"UNORDERED form={form} arch={arch} n={n} k={k} splits={plan['tail_splits']} "
        f"num_tail={plan['num_tail']} worst={worst:.3e}"
    )


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("label,sizes", BENCH_WGRAD_ROWS)
@pytest.mark.parametrize("proj", list(BENCH_PROJ))
def test_bench_wgrad_rows_match_reference(proj, label, sizes, out_dtype):
    """The #5678 MoE projection rows (gate/up K 6144, down K 2048) of the weight
    gradient, worst case over repeats against the fp64 reference."""
    device = _require_program("wgrad", out_dtype)
    n, k = BENCH_PROJ[proj]
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=len(label))
    launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    ref = _reference("wgrad", g, x, sizes)
    worst = 0.0
    for _ in range(2):
        launch.out.fill_(float("nan"))
        worst = max(worst, _assert_wgrad_close(launch.launch(), ref))
        _assert_empty_groups_zero(launch.out, sizes)
    print(
        f"BENCHROW {proj}_{label} {out_dtype} stages={launch.stages} "
        f"unordered={launch.plan['unordered_tail']} splits={launch.plan['tail_splits']} "
        f"worst={worst:.3e}"
    )


# (groups, form): SMALL_E512 as planned (the 256-wide k tile), then SMALL_E512 and
# MIXED_E128 with the 512-wide k tile forced into each of its forms.
SHORT_GROUP_CASES = [("short", None)] + [
    (groups, form) for groups in ("short", "mixed") for form in K512_FORMS
]


def _short_group_launch(device, groups, form, out_dtype, monkeypatch, seed):
    """A prepared weight gradient of short groups (``SMALL_E512``) or of short groups beside
    long ones (``MIXED_E128``): with ``form`` None as planned (the 256-wide k tile), else
    with the 512-wide k tile forced into ``form``; a mixed launch on a shape whose plan
    splits the unordered tail.  Returns ``(launch, sizes, g, x, offs)``."""
    arch = _arch(device)
    sizes, n, k = SMALL_E512, 512, 3072
    if form is not None:
        if not _force_form(monkeypatch, form, arch):
            pytest.skip(f"no {form} weight-gradient build on {arch}")
        if groups == "short":
            k = _form_ks(form)[0]
        else:
            sizes = MIXED_E128
            shape = _split_tail_shape(device, sizes, _form_ks(form))
            assert shape is not None, f"no candidate shape splits the {form} tail"
            n, k = shape
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=seed)
    launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    plan = launch.plan
    if form is None:
        assert plan["tile_k"] == 256, plan
    else:
        assert plan["tile_k"] == 512 and plan["unordered_tail"], plan
        assert launch.stages == (form,), (launch.stages, plan)
        if groups == "mixed":
            assert plan["tail_splits"] > 1, plan
        else:
            # One k-step per group: a unit is a whole tile, the plan never splits.
            assert plan["tail_splits"] == 1, plan
    return launch, sizes, g, x, offs


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "groups,form", SHORT_GROUP_CASES, ids=[f"{g}-{f}" for g, f in SHORT_GROUP_CASES]
)
def test_wgrad_short_groups_never_include_the_next_group(
    groups, form, out_dtype, monkeypatch
):
    """A tile of a group shorter than one k-step must not add the rows of the group the
    same CTA reduces next (#6570): the shape as planned (the 256-wide k tile), then every
    512-wide k tile form this architecture serves (``launch.stages == (form,)``) over
    groups of one k-step, and over short groups beside long ones with a split unordered
    tail.  Repeated launches of one prepared call, checked block by block."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    launch, sizes, g, x, _ = _short_group_launch(
        device, groups, form, out_dtype, monkeypatch, seed=6
    )
    n, k = launch.out.shape[1:]
    ref = torch.stack([g[s:t].float().T @ x[s:t].float() for s, t in _bounds(sizes)])
    blocks = ref.view(len(sizes), n // 128, 128, k // 128, 128)
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
    if launch.plan["tail_splits"] > 1 and launch.plan["unordered_tail"]:
        torch.cuda.synchronize()
        _assert_tail_at_rest(device)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("form", K512_FORMS)
def test_short_groups_on_two_concurrent_streams(form, out_dtype, monkeypatch):
    """Two prepared weight gradients of the mixed short / long groups, the 512-wide k tile
    forced into ``form`` with a split unordered tail, launched interleaved on two streams
    in flight together while a long-running kernel stays resident on a third stream and
    copies and GEMMs run on a fourth (flashinfer-ai/flashinfer#7038, #7548: correctness
    must not depend on other streams being idle).  Each stream keeps its own tail sums and
    tickets, every result is within the budget, and both streams' workspaces rest at
    zero."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    launch, sizes, g, x, offs = _short_group_launch(
        device, "mixed", form, out_dtype, monkeypatch, seed=12
    )
    launches = [launch, prepare_wgrad(g, x, offs, out_dtype=out_dtype)]
    assert launches[1].stages == (form,) and launches[1].plan == launch.plan
    ref = _reference("wgrad", g, x, sizes)
    streams = [torch.cuda.Stream(device) for _ in range(4)]
    main = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(main)
    src = torch.randn(1 << 24, device=device)
    dst = torch.empty_like(src)
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    c = torch.empty_like(a)
    with torch.cuda.stream(streams[2]):
        torch.cuda._sleep(RESIDENT_SLEEP_CYCLES)
    outs = [[], []]
    for _ in range(16):
        for index in range(2):
            with torch.cuda.stream(streams[index]):
                launches[index].out.fill_(float("nan"))
                outs[index].append(launches[index].launch().clone())
        with torch.cuda.stream(streams[3]):
            dst.copy_(src)
            torch.mm(a, a, out=c)
    for stream in streams[:2]:
        stream.synchronize()
    # Whether the weight gradients finished while the resident kernel still ran (a
    # timing fact, reported only: correctness is what the test asserts).
    overlap = not streams[2].query()
    for stream in streams:
        main.wait_stream(stream)
    torch.cuda.synchronize()
    print(f"RESIDENT form={form} dtype={out_dtype} overlap={overlap}")
    for index in range(2):
        for out in outs[index]:
            _assert_wgrad_close(out, ref)
            _assert_empty_groups_zero(out, sizes)
    for kind in cb.SIZED_STREAM_WORKSPACES:
        buffers = [cb.stream_workspace(kind, device, s) for s in streams]
        assert buffers[0] is not None and buffers[1] is not None, kind
        assert buffers[0].data_ptr() != buffers[1].data_ptr(), kind
        assert buffers[2] is None and buffers[3] is None, kind
    for stream in streams[:2]:
        _assert_tail_at_rest(device, stream)


@pytest.mark.parametrize("op", ["fwd", "dgrad", "wgrad"])
def test_cuda_graph_replay_with_changed_offsets(op, monkeypatch):
    """A captured launch takes zeroed workspaces of its own, set in the graph, and never
    the buffers kept for a stream: the kept buffers (the split tail's filled with a
    sentinel) keep their bytes through the capture and every replay.  Every replay
    computes every tile, also after new group boundaries (same ``sum_m``) are written
    into ``offs`` between replays.  The weight gradient is a 512-wide k tile shape whose
    plan splits its unordered tail."""
    _fresh_stream_workspaces(monkeypatch)
    first = list(K512_E3)
    second = [first[0] // 2, first[2], first[0] - first[0] // 2]
    if op in ("fwd", "dgrad"):
        device = _require_program(op)
        n, k = 1024, 512
    else:
        device = _require_program("wgrad", torch.bfloat16)
        shape = _split_tail_shape(device, first, _form_ks("main_halves"))
        assert shape is not None, "no candidate shape splits the tail"
        n, k = shape
    x, w, g, offs = _inputs(first, n, k, device, seed=8)
    if op == "fwd":
        launch, refs = (
            prepare_fwd(x, w, offs),
            [_reference("fwd", x, w, s) for s in (first, second)],
        )
    elif op == "dgrad":
        launch, refs = (
            prepare_dgrad(g, w, offs),
            [_reference("dgrad", g, w, s) for s in (first, second)],
        )
    else:
        launch, refs = (
            prepare_wgrad(g, x, offs),
            [_reference("wgrad", g, x, s) for s in (first, second)],
        )
        assert launch.plan["unordered_tail"] and launch.plan["tail_splits"] > 1
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        launch.launch()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    kept = dict(cb._stream_workspaces)
    if op == "wgrad":
        assert set(cb.SIZED_STREAM_WORKSPACES) <= {key[0] for key in kept}
    for key, buffer in kept.items():
        if key[0] in cb.SIZED_STREAM_WORKSPACES:
            # A graph that used them would read the sentinel into its sums.
            buffer.view(torch.int32).fill_(SENTINEL_WORD)
    snapshot = {key: buffer.view(torch.int32).clone() for key, buffer in kept.items()}
    captured = []
    launch_workspace = cb._launch_workspace

    def recording(kind, device_, device_index, keep, size=0):
        buffer = launch_workspace(kind, device_, device_index, keep, size)
        captured.append((kind, keep, buffer.data_ptr()))
        return buffer

    monkeypatch.setattr(cb, "_launch_workspace", recording)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    monkeypatch.setattr(cb, "_launch_workspace", launch_workspace)
    assert captured and not any(keep for _kind, keep, _ptr in captured), captured
    if op == "wgrad":
        assert set(cb.SIZED_STREAM_WORKSPACES) <= {kind for kind, _, _ in captured}
    kept_ptrs = {buffer.data_ptr() for buffer in kept.values()}
    assert not kept_ptrs & {ptr for _kind, _keep, ptr in captured}
    # The capture kept no buffer of its own and replaced none.
    assert cb._stream_workspaces.keys() == kept.keys()
    assert all(cb._stream_workspaces[key] is buffer for key, buffer in kept.items())
    for sizes, ref in zip((first, second, first), (*refs, refs[0]), strict=True):
        offs.copy_(_offs(sizes, device))
        for _ in range(2):
            launch.out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            _assert_bf16_close(launch.out, ref)
            if op == "wgrad":
                _assert_empty_groups_zero(launch.out, sizes)
    for key, buffer in kept.items():
        assert torch.equal(buffer.view(torch.int32), snapshot[key]), key


def test_non_reentrant_checkpoint_replays_the_forward(monkeypatch):
    """Under ``torch.utils.checkpoint`` (non-reentrant) the forward is recomputed in
    the backward: the same ``Y`` and ``dX`` bits as without the checkpoint, and a
    weight gradient within the budget of the fp64 reference (512-wide k tile with a
    split unordered tail)."""
    device = _require_program("wgrad", torch.bfloat16)
    _require_program("fwd")
    _require_program("dgrad")
    sizes = K512_E3
    shape = _split_tail_shape(device, sizes, _form_ks("main_halves"))
    assert shape is not None, "no candidate shape splits the tail"
    n, k = shape
    x0, w0, g, offs = _inputs(sizes, n, k, device, seed=13)

    def run(checkpointed):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)

        def f(a, b):
            return pkg.cake_grouped_mm(a, b, offs, deterministic=False)

        if checkpointed:
            y = torch.utils.checkpoint.checkpoint(f, x, w, use_reentrant=False)
        else:
            y = f(x, w)
        y.backward(g)
        torch.cuda.synchronize()
        return y.detach(), x.grad, w.grad

    y_plain, dx_plain, dw_plain = run(False)
    y_ckpt, dx_ckpt, dw_ckpt = run(True)
    assert torch.equal(y_ckpt, y_plain)
    assert torch.equal(dx_ckpt, dx_plain)
    ref = _reference("wgrad", g, x0, sizes)
    for dw in (dw_plain, dw_ckpt):
        assert dw.dtype == torch.bfloat16
        _assert_bf16_close(dw, ref)
        _assert_empty_groups_zero(dw, sizes)


def test_gradient_accumulation(monkeypatch):
    """Two backward passes accumulate into ``.grad`` on a shape whose weight gradient
    splits its unordered tail: ``x.grad`` is the bf16 sum of the two data gradients
    (bitwise); each backward's weight gradient (captured by a hook before the
    accumulation) is within the deterministic package's budget of its fp64 reference,
    ``w.grad`` is the bf16 sum of the two (bitwise), and an empty group's weight
    gradient stays exact zeros."""
    device = _require_program("wgrad", torch.bfloat16)
    _require_program("fwd")
    _require_program("dgrad")
    _fresh_stream_workspaces(monkeypatch)
    sizes = K512_E3
    shape = _split_tail_shape(device, sizes, _form_ks("main_halves"))
    assert shape is not None, "no candidate shape splits the tail"
    n, k = shape
    x0, w0, g1, offs = _inputs(sizes, n, k, device, seed=14)
    g2 = _inputs(sizes, n, k, device, seed=15)[2]
    plan = prepare_wgrad(g1, x0, offs).plan
    assert plan["unordered_tail"] and plan["tail_splits"] > 1, plan
    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    dws = []
    w.register_hook(lambda grad: dws.append(grad.detach().clone()))
    for g in (g1, g2):
        pkg.cake_grouped_mm(x, w, offs, deterministic=False).backward(g)
    torch.cuda.synchronize()
    dx1 = grouped_gemm_dgrad(g1, w0, offs)
    dx2 = grouped_gemm_dgrad(g2, w0, offs)
    assert torch.equal(x.grad, dx1 + dx2)
    assert len(dws) == 2
    for dw, g in zip(dws, (g1, g2), strict=True):
        assert dw.dtype == torch.bfloat16
        _assert_wgrad_close(dw, _reference("wgrad", g, x0, sizes))
    assert torch.equal(w.grad, dws[0] + dws[1])
    _assert_empty_groups_zero(w.grad, sizes)


def test_changed_shapes_reuse_and_garbage_filled_scratch(monkeypatch):
    """Prepared launches of changing shapes on one stream (``CHANGED_SHAPES``: small,
    large, small), each with its per-CTA descriptor slots and its output filled with
    garbage bytes before the launch: the results are within the budget; the large
    launch needs strictly more tail sums and tickets and replaces both kept buffers with
    larger zeroed ones, the later small launch reuses them, and the kept sums and
    tickets rest at zero after every launch."""
    device = _require_program("wgrad", torch.bfloat16)
    _fresh_stream_workspaces(monkeypatch)
    sizes = K512_E3
    cases = []
    for index, (n, k) in enumerate(CHANGED_SHAPES):
        x, _w, g, offs = _inputs(sizes, n, k, device, seed=16 + index)
        launch = prepare_wgrad(g, x, offs)
        assert launch.plan["unordered_tail"] and launch.plan["tail_splits"] > 1
        cases.append((launch, _reference("wgrad", g, x, sizes)))
    small, large = cases
    need = {
        kind: (small[0].plan[field], large[0].plan[field])
        for kind, field in (
            ("tail_sums", "tail_sum_elems"),
            ("tail_tickets", "tail_ticket_words"),
        )
    }
    for kind, (small_size, large_size) in need.items():
        assert 0 < small_size < large_size, (kind, small_size, large_size)
    kept = []
    for launch, ref in (small, large, small):
        launch.workspaces[0].fill_(0xFF)  # the per-CTA descriptor slots
        launch.out.view(torch.uint8).fill_(GARBAGE_BYTE)
        _assert_bf16_close(launch.launch(), ref)
        _assert_empty_groups_zero(launch.out, sizes)
        torch.cuda.synchronize()
        _assert_tail_at_rest(device)
        kept.append(
            {
                kind: (buffer.numel(), buffer.data_ptr())
                for kind in need
                for buffer in [cb.stream_workspace(kind, device)]
            }
        )
    for kind, (small_size, large_size) in need.items():
        assert kept[0][kind][0] == small_size, kind
        # The large launch replaced the kept buffer with a larger one ...
        assert kept[1][kind][0] == large_size, kind
        assert kept[1][kind][1] != kept[0][kind][1], kind
        # ... and the small launch after it reuses that buffer.
        assert kept[2][kind] == kept[1][kind], kind


@pytest.mark.parametrize("op", ["dgrad", "wgrad"])
def test_prepared_launches_allocate_nothing_after_the_first(op, monkeypatch):
    """After the first launch on a stream (which creates its stream workspaces), a
    prepared launch allocates no device memory: dgrad, and the weight gradient with a
    split unordered tail."""
    _fresh_stream_workspaces(monkeypatch)
    if op == "dgrad":
        device = _require_program("dgrad")
        x, w, g, offs = _inputs(RAGGED_E6, 1024, 512, device)
        launch, ref = prepare_dgrad(g, w, offs), _reference("dgrad", g, w, RAGGED_E6)
    else:
        device = _require_program("wgrad", torch.bfloat16)
        shape = _split_tail_shape(device, K512_E3, _form_ks("main_halves"))
        assert shape is not None, "no candidate shape splits the tail"
        x, _w, g, offs = _inputs(K512_E3, *shape, device, seed=9)
        launch = prepare_wgrad(g, x, offs)
        ref = _reference("wgrad", g, x, K512_E3)
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
    launch = prepare_dgrad(g, w, offs)
    _assert_bf16_close(launch.launch(), _reference("dgrad", g, w, first))
    offs.copy_(_offs(second, device))
    _assert_bf16_close(launch.launch(), _reference("dgrad", g, w, second))


def test_padded_outputs_leave_the_padding_untouched():
    _require_program("fwd")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(K512_E3, 512, 1024, device)
    sum_m, n, k = sum(K512_E3), 512, 1024
    padded = torch.full(
        (sum_m, n + 64), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = grouped_gemm_fwd(x, w, offs, out=padded[:, :n])
    _assert_bf16_close(out, _reference("fwd", x, w, K512_E3))
    assert torch.isnan(padded[:, n:]).all()
    padded = torch.full(
        (len(K512_E3), n, k + 32), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = grouped_gemm_wgrad(g, x, offs, out=padded[:, :, :k])
    _assert_bf16_close(out, _reference("wgrad", g, x, K512_E3))
    assert torch.isnan(padded[:, :, k:]).all()


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_accepts_row_padded_operands(out_dtype):
    """The weight gradient of row-padded ``g`` and ``x`` (the leading columns of wider
    tensors whose other columns hold NaN) on the shape of the 256-wide k tile's
    ``main_wave`` program (sm_100a / sm_103a, whose row-base arguments take any row
    stride; ``main`` on sm_107a): the result is bitwise the result of the same call on
    dense copies and within the budget of the fp64 reference."""
    device = _require_program("wgrad", out_dtype)
    stage = _k256_wave_stage(device, out_dtype)
    sizes = K256_WAVE_SIZES
    n, k = K256_WAVE_SHAPE
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=31)
    g_pad, x_pad = _row_padded(g), _row_padded(x)
    assert g_pad.stride(0) == n + ROW_PAD and x_pad.stride(0) == k + ROW_PAD
    launch = prepare_wgrad(g_pad, x_pad, offs, out_dtype=out_dtype)
    assert launch.stages[0] == stage, launch.stages
    out = launch.launch().clone()
    dense = grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    assert torch.equal(out, dense)
    _assert_wgrad_close(out, _reference("wgrad", g, x, sizes))


def test_autograd_backward_takes_row_strided_operands():
    """``cake_grouped_mm(deterministic=False)`` on a row-strided leaf ``x`` whose output is
    concatenated with another tensor, so that the backward receives a row-strided
    ``grad_y`` (a narrow of the concatenation's gradient); on the shape of the 256-wide k
    tile's ``main_wave`` weight gradient, ``Y``, ``dX`` and ``dW`` are within the
    deterministic package's budgets of the fp64 references."""
    device = _require_program("wgrad", torch.bfloat16)
    _require_program("fwd")
    _require_program("dgrad")
    _k256_wave_stage(device)
    sizes = K256_WAVE_SIZES
    n, k = K256_WAVE_SHAPE
    x0, w0, g, offs = _inputs(sizes, n, k, device, seed=32)
    x = _row_padded(x0).detach().requires_grad_(True)
    assert x.is_leaf and x.stride(0) == k + ROW_PAD
    w = w0.clone().requires_grad_(True)
    y = pkg.cake_grouped_mm(x, w, offs, deterministic=False)
    other = torch.zeros(y.shape[0], ROW_PAD, dtype=y.dtype, device=device)
    z = torch.cat([y, other], dim=1)
    grad_strides = []
    y.register_hook(lambda grad: grad_strides.append(grad.stride()))
    z.backward(torch.cat([g, torch.ones_like(other)], dim=1))
    torch.cuda.synchronize()
    # The backward received the row-strided narrow of the concatenation's gradient.
    assert grad_strides == [(n + ROW_PAD, 1)], grad_strides
    _assert_bf16_close(y.detach(), _reference("fwd", x0, w0, sizes))
    _assert_bf16_close(x.grad, _reference("dgrad", g, w0, sizes))
    assert w.grad.dtype == torch.bfloat16
    _assert_wgrad_close(w.grad, _reference("wgrad", g, x0, sizes))


def test_empty_batch():
    _require_program("fwd")
    device = _require_program("wgrad", torch.float32)
    x, w, g, offs = _inputs([0, 0], 256, 512, device)
    y = torch.full((0, 256), 0.0, dtype=torch.bfloat16, device=device)
    assert grouped_gemm_fwd(x, w, offs, out=y).shape == (0, 256)
    dw = torch.full((2, 256, 512), float("nan"), dtype=torch.float32, device=device)
    assert torch.equal(grouped_gemm_wgrad(g, x, offs, out=dw), torch.zeros_like(dw))


def test_rejects_misaligned_shapes():
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(RAGGED_E4, 384, 256, device)  # N % 256 != 0
    with pytest.raises(ValueError):
        prepare_fwd(x, w, offs)
    with pytest.raises(ValueError):
        prepare_fwd(x, w, offs.to(torch.int64))


def _m_indptr(sizes, device):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return torch.tensor([0] + ends, dtype=torch.int32, device=device)


def test_m_indptr_form_matches_offs_form():
    _require_program("fwd")
    _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device)
    m_indptr = _m_indptr(RAGGED_E6, device)
    assert torch.equal(grouped_gemm_fwd(x, w, m_indptr), grouped_gemm_fwd(x, w, offs))
    assert torch.equal(
        grouped_gemm_dgrad(g, w, m_indptr), grouped_gemm_dgrad(g, w, offs)
    )
    ref = _reference("wgrad", g, x, RAGGED_E6)
    _assert_bf16_close(
        grouped_gemm_wgrad(g, x, m_indptr, num_groups=len(RAGGED_E6)), ref
    )
    with pytest.raises(ValueError):  # neither E nor E + 1 entries
        grouped_gemm_wgrad(g, x, m_indptr[:-2], num_groups=len(RAGGED_E6))


def test_autograd_wrapper_matches_reference():
    _require_program("fwd")
    _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(K512_E3, 512, 1024, device)
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = pkg.cake_grouped_mm(x, w, offs, deterministic=False)
    assert y.shape == (sum(K512_E3), 512) and y.dtype == torch.bfloat16
    _assert_bf16_close(y.detach(), _reference("fwd", x.detach(), w.detach(), K512_E3))
    y.backward(g)
    assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.bfloat16
    _assert_bf16_close(x.grad, _reference("dgrad", g, w.detach(), K512_E3))
    _assert_bf16_close(w.grad, _reference("wgrad", g, x.detach(), K512_E3))
    _assert_empty_groups_zero(w.grad, K512_E3)
    with pytest.raises(ValueError):
        pkg.cake_grouped_mm(x.float(), w, offs, deterministic=False)
    with pytest.raises(ValueError):
        cake_autograd.CakeGroupedMm.apply(x, w, offs, True)
