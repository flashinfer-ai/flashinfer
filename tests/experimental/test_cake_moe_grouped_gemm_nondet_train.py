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
(``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train``).

The forward pass, the data gradient and the 256-wide k tile's weight gradient
run copies of the deterministic package's programs; the 512-wide k tile's weight
gradient finishes its split tail in the same launch, summing the units' fp32 slots
in slot order.
Every output is checked against a per-group fp64 reference under the
deterministic package's budgets (bf16: atol = rtol = 1e-2; fp32 weight
gradient: rel-L2 1e-4, but on the one-group E1 rows, which the reproducibility
file holds to the deterministic package's error), and every repeated launch
(the output refilled, the split-tail slots poisoned with garbage bytes) must
give the first launch's bits; the split tail's tickets are checked at rest
(zero) after the launches.
Only this package is imported (through ``cake_package``), so the suite also runs
against a vendored copy of it.

``cake_moe_grouped_gemm_nondet`` launches the forward after autograd saved its inputs, so the
early stop of a non-reentrant checkpoint skips the recomputed GEMM when nothing
after it saves a tensor.  The autograd tests count the forward launches (a
pass-through on ``cake_autograd._fwd_op``) and, where the torch profiler records
kernels, the CUDA kernels of checkpointed steps; the control arm is a direct
``CakeMoEGroupedGemmNondetFunction.apply``, which launches inside ``forward``.
"""

import collections
import contextlib
import functools
import math

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package

# Backend suite: the package is named through ``cake_package`` only; nothing
# else of flashinfer is imported here.  The facade and the numerical-mode
# checks live in ``test_cake_moe_grouped_gemm_nondet_train_api.py``, the comparisons with
# the deterministic package (its routes and bits) in
# ``test_cake_moe_grouped_gemm_nondet_train_reproducibility.py``.
PACKAGE = "cake_moe_grouped_gemm_nondet_train"
pkg = cake_package(PACKAGE)
cb = cake_package(PACKAGE, "cake_backend")
cake_jit = cake_package(PACKAGE, "cake_jit")
cake_autograd = cake_package(PACKAGE, "cake_autograd")
SUPPORTED_COMPUTE_CAPABILITIES = cb.SUPPORTED_COMPUTE_CAPABILITIES
HOST_PLAN_CONSTANTS = cake_jit.HOST_PLAN_CONSTANTS
PROGRAMS = cake_jit.PROGRAMS
ROUTES = cake_jit.ROUTES

# Short names of the package's entry points.
prepare_fwd = cb.cake_moe_grouped_gemm_nondet_train_prepare_forward
prepare_dgrad = cb.cake_moe_grouped_gemm_nondet_train_prepare_dgrad
prepare_wgrad = cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad

ATOL = RTOL = 1e-2  # bf16 outputs
FP32_REL_L2 = 1e-4  # fp32 weight gradient vs the fp64 reference
# Launches per accuracy case of the weight gradient: the first is checked against the
# fp64 reference, and every later one must give its bits.
REPEATS = 4
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
# its ordered one-launch tail at every SM count of SM_COUNTS
# (test_group_count_sweep_splits_the_tail_at_every_sm_count).
GROUP_COUNTS = (1, 4, 8, 16, 32)
GROUP_ROWS = 4096
GROUP_SHAPE = (1024, 1024)
# Two weight-gradient shapes (N, K) over K512_E3 whose ordered one-launch tails split,
# the second needing strictly more tail slots and tickets than the first at every SM
# count of SM_COUNTS (test_changed_shapes_grow_the_tail_at_every_sm_count).
CHANGED_SHAPES = ((7168, 1024), (4864, 1536))
# The 256-wide k tile's wave-synchronised weight gradient (``main_wave`` where
# ``wave_sync_archs[256]`` lists the architecture, sm_100a / sm_103a; ``main``
# elsewhere): average rows below k512_min_rows_per_group, K >= wave_sync_min_k.
K256_WAVE_SIZES = [2048] * 8
K256_WAVE_SHAPE = (4096, 4096)  # (N, K)
ROW_PAD = 256  # extra columns of the wider tensors row-padded operands are cut from
# Two ragged groups of at least every architecture's ``dgrad_evict_first_min_rows``
# rows on average: with ``K >= dgrad_evict_first_min_k`` the data gradient takes the
# evict-first twin of its wide tile on the architectures of ``dgrad_evict_first_archs``
# (sm_100a: the domain-scheduled twin; sm_107a: the A-collector twin).
EVICT_FIRST_E2 = [16390, 16500]
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


def _aligned_balanced(sum_m, e):
    """The #5678 harness's balanced split of ``sum_m`` rows over ``e`` groups: ``e - 1``
    groups of ``(sum_m // e) // 32 * 32`` rows and the remainder in the last."""
    each = (sum_m // e) // 32 * 32
    return [each] * (e - 1) + [sum_m - each * (e - 1)]


# The #5678 customer rows the benchmark rows above do not cover: E1, and E8 / E16 / E32
# balanced, at both row sums of the E4 rows.
CUSTOMER_ROWS = tuple(
    (f"e{e}_m{sum_m}_balanced", _aligned_balanced(sum_m, e))
    for sum_m in (112576, 136064)
    for e in (1, 8, 16, 32)
)
# The E1 rows reduce 112,576 / 136,064 rows in one group.  There the deterministic
# package's own fp32 weight gradient has rel-L2 1.17e-4 / 1.41e-4 against the fp64
# reference on GB300, above FP32_REL_L2, so the fp32 outputs of these rows are held to
# the deterministic package's error at the same inputs instead (rel-L2 and worst / max
# |ref| each at most 1.1x, test_e1_fp32_wgrad_error_within_det in the reproducibility
# file).  Here they keep the bf16 budget, the bitwise repeats and the plan checks.
DET_RELATIVE_FP32_ROWS = frozenset(
    label for label, sizes in CUSTOMER_ROWS if len(sizes) == 1
)
# Every #5678 row the accuracy tests check: the benchmark rows and the customer rows.
ACCURACY_ROWS = BENCH_WGRAD_ROWS + CUSTOMER_ROWS
K512_FORMS = (
    "main",
    "main_wave",
    "main_halves",
    "main_wave_halves",
    "main_halves_k256t",
    cb.QUEUE_STAGE,
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


def _evict_first(op, k, arch, sum_m, groups, c):
    """The evict-first rule of ``op``: ``dgrad_evict_first``; ``fwd_lwh_ef`` or
    ``fwd_evict_first``."""
    if op == "dgrad":
        return cb.dgrad_evict_first(k, arch, sum_m, groups, c)
    return cb.fwd_lwh_ef(k, arch, c) or cb.fwd_evict_first(k, arch, c)


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


_BIT_VIEWS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _assert_same_bits(got, want, what=""):
    """``got`` holds exactly the bits of ``want`` (raw words, so NaN payloads and signed
    zeros count)."""
    assert got.dtype == want.dtype and got.shape == want.shape, (what, got.shape)
    words = _BIT_VIEWS[got.dtype]
    differ = got.view(words) != want.view(words)
    assert not bool(differ.any()), f"{what}: {int(differ.sum())} elements differ"


def _repeat_bits(launch, device, repeats, fills=(None,), check=None):
    """Launch the prepared ``launch`` ``repeats`` times per output fill (``None``: NaN,
    else every byte ``fill``) with the stream's split-tail slots poisoned before every
    launch; ``check(out)`` (if given) judges the first output, and every later one must
    give its bits.  Returns that first output."""
    first = None
    for fill in fills:
        for call in range(repeats):
            _poison_tail_sums(device)
            if fill is None:
                launch.out.fill_(float("nan"))
            else:
                launch.out.view(torch.uint8).fill_(fill)
            out = launch.launch()
            if first is None:
                if check is not None:
                    check(out)
                first = out.clone()
            else:
                _assert_same_bits(out, first, f"fill {fill} launch {call}")
    return first


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
    """The kept split-tail workspaces of ``stream`` that have a rest state are at it: the
    tickets are all zero.  The slots (``tail_sums``, ``cb.UNRESTED_STREAM_WORKSPACES``) have
    none; they must be kept, but their values are not checked.  The queue walk's counters
    (``cb.QUEUE_SCHED``), where a queue launch kept them, are zero too."""
    assert cb.UNRESTED_STREAM_WORKSPACES == ("tail_sums",)
    for kind in ("tail_sums", "tail_tickets"):
        buffer = cb.stream_workspace(kind, device, stream)
        assert buffer is not None, kind
        if kind in cb.UNRESTED_STREAM_WORKSPACES:
            continue
        assert int(buffer.view(torch.int32).count_nonzero()) == 0, (
            f"{kind} is not at rest: {int(buffer.view(torch.int32).count_nonzero())} nonzero words"
        )
    _assert_queue_at_rest(device, stream)


def _assert_queue_at_rest(device, stream=None):
    """The queue walk's counters kept for ``stream`` (if a queue launch kept them) are
    ``queue_sched_words`` zero words."""
    words = cb.stream_workspace(cb.QUEUE_SCHED, device, stream)
    if words is None:
        return
    assert words.numel() == int(HOST_PLAN_CONSTANTS["queue_sched_words"])
    assert _u32_words(words) == [0] * words.numel(), _u32_words(words)


def _poison_tail_sums(device, stream=None):
    """Fill the kept split-tail slots of ``stream`` (the current stream by default) with
    ``GARBAGE_BYTE`` bytes, on that stream: the ordered builds overwrite every slot they
    read, so a launch after this gives the bits of a launch on fresh slots.  Returns
    whether a kept slot buffer existed."""
    buffer = cb.stream_workspace("tail_sums", device, stream)
    if buffer is None:
        return False
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        buffer.view(torch.uint8).fill_(GARBAGE_BYTE)
    return True


def _force_form(monkeypatch, form, arch):
    """Constants under which a weight gradient on ``arch`` takes the 512-wide k tile (at
    any rows per group) in stage ``form`` (given a ``K`` of ``_form_ks(form)``); False
    where the arch has no such form.  The queue walk (``cb.QUEUE_STAGE``) is forced as
    the stand-in of ``main_halves`` for every output, SM count and group count; every
    other form is forced with no ``queue_walk`` rule."""
    c = HOST_PLAN_CONSTANTS
    queue = form == cb.QUEUE_STAGE
    if queue:
        served = any(
            cake_jit.stage_served(
                cake_jit.record_name("wgrad", out_dtype=name, tile_k=512),
                cb.QUEUE_STAGE,
                arch,
            )
            for name in DTYPE_NAMES.values()
        )
        if not served:
            return False
        form = "main_halves"
    rules = []
    if queue:
        rules = [
            dict(
                arch=arch,
                sm_counts=list(range(1, 1025)),
                stage="main_halves",
                outputs=["bfloat16", "float32", "accumulate"],
                groups=[1, 1 << 30],
            )
        ]
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "queue_walk", rules)
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
    if form == cb.QUEUE_STAGE:
        form = "main_halves"
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
        ordered_tail=cb.wgrad_ordered_tail(tile_k, arch, c),
        constants=c,
    )
    return stage, plan


def _split_tail_shape(device, sizes, ks):
    """``(N, K)`` over ``ks`` (and ``N`` up to 48 column blocks) whose weight-gradient plan
    on ``device`` takes the 512-wide k tile and splits its ordered one-launch tail
    (``tail_splits > 1``), with the ``[E, N, K]`` output within the kernel's 32-bit
    element addressing (``int32_elems``: which candidates split depends on the SM count);
    None when no candidate does."""
    arch, sms = _arch(device), cb.sm_count(device)
    span = int(HOST_PLAN_CONSTANTS["int32_elems"])
    for k in ks:
        for blocks in range(1, 49):
            n = 256 * blocks
            if len(sizes) * n * k > span:
                break
            _stage, plan = _wgrad_plan(sizes, n, k, arch, sms)
            if plan["ordered_tail"] and plan["tail_splits"] > 1:
                assert not plan["unordered_tail"], plan
                return n, k
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
                ordered = cb.wgrad_ordered_tail(tile_k, arch, c)
                # The 512-wide k tile takes the ordered one-launch tail everywhere.
                assert not unordered and ordered == (tile_k == 512), (arch, tile_k)
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
                    ordered_tail=ordered,
                    constants=c,
                )
                tail_tile_k = cb.K256_TAIL_TILE_K if k256_tail else tile_k
                clusters = plan["clusters"]
                assert plan["grid"][0] == clusters * cta_group
                assert plan["tail_tile_k"] == tail_tile_k
                assert plan["unordered_tail"] is unordered
                assert plan["ordered_tail"] is ordered
                assert plan["num_tail"] == (
                    (e * (n // int(c["block_n"])) * (k // tail_tile_k)) % clusters
                    if int(c["wgrad_tail_split"])
                    else 0
                )
                assert 0 <= plan["num_tail"] < clusters
                splits = plan["tail_splits"]
                if plan["num_tail"] == 0:
                    assert splits == 1
                if not ordered:
                    # The 256-wide k tile: the deterministic package's plan.
                    assert 1 <= splits <= int(c["tail_plan"]["max_splits"])
                    assert plan["launches"] == 1 + int(splits > 1)
                    assert (plan["tail_reduce_grid"] is None) == (splits == 1)
                    assert plan["tail_sum_elems"] == plan["tail_ticket_words"] == 0
                    continue
                # The ordered one-launch tail: atomic_tail_plan, one launch, no reduce grid,
                # one slot per (tail tile, split).
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
                    plan["num_tail"] * splits * int(c["cluster_m"]) * tail_tile_k
                    if split
                    else 0
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
    """``atomic_tail_plan`` (the cost model of the one-launch split tail) carries the keys
    of the two-launch ``tail_plan`` (the parent cost model), with a cycles-per-step entry
    for both k tiles."""
    c = HOST_PLAN_CONSTANTS
    atomic, two_launch = c["atomic_tail_plan"], c["tail_plan"]
    assert set(atomic) == set(two_launch)
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


def test_ordered_tail_archs_bind_the_ordered_k512_builds():
    """``ordered_tail_archs`` names the architectures whose 512-wide k tile routes bind
    the ordered one-launch tail builds (``tail_sums`` slots and ``tail_tickets``, no
    ``tail_reduce`` stage): every architecture of the k512 routes.
    ``unordered_tail_archs`` is empty (the registry holds no unordered build), and no
    architecture is in both lists.  ``wgrad_ordered_tail`` and ``wgrad_unordered_tail``
    follow the lists, and the 256-wide k tile always binds the deterministic programs."""
    c = HOST_PLAN_CONSTANTS
    supported = set(SUPPORTED_COMPUTE_CAPABILITIES.values())
    assert c["unordered_tail_archs"] == []
    assert set(c["ordered_tail_archs"]) <= supported
    assert not set(c["ordered_tail_archs"]) & set(c["unordered_tail_archs"])
    k512_arches = set()
    for name, route in ROUTES.items():
        if route["op"] != "wgrad":
            continue
        if name.endswith("_k512"):
            k512_arches |= set(route["arches"])
            assert set(route["arches"]) <= set(c["ordered_tail_archs"]), name
        tile_k = 512 if name.endswith("_k512") else 256
        for arch in route["arches"]:
            ordered = cb.wgrad_ordered_tail(tile_k, arch, c)
            assert ordered == (tile_k == 512 and arch in c["ordered_tail_archs"])
            assert not cb.wgrad_unordered_tail(tile_k, arch, c), (name, arch)
            assert cake_jit.stage_served(name, "tail_reduce", arch) == (not ordered)
            for stage in route["stages"]:
                program = PROGRAMS[stage["program"]]
                if arch not in program["arches"] or stage["name"] == "tail_reduce":
                    continue
                plan = [tuple(item) for item in program["arg_plan"]]
                assert (("buffer", "tail_sums") in plan) == ordered, (name, arch)
                assert (("buffer", "tail_tickets") in plan) == ordered, (name, arch)
                assert (("buffer", "partials") in plan) == (not ordered), (name, arch)
    assert k512_arches == set(c["ordered_tail_archs"])


def test_program_choice_follows_the_host_plan_constants():
    """The deterministic package's rules select the stages: the 512-column tile of
    ``fwd`` / ``dgrad``, its evict-first and 3 / 3 head / tail twins, the wave builds,
    the half-release builds and the k256-tail twin; each listed architecture's routes
    serve the stage the constant selects (the 512-wide k tile's as ordered one-launch
    tail builds, without a tail reduce).  ``dgrad_evict_first_archs`` selects the
    evict-first twin of the ``dgrad`` wide tile where ``dgrad_evict_first`` holds, and
    ``fwd_lwh_ef_archs`` / ``fwd_evict_first_archs`` that of the ``fwd`` wide tile
    where ``fwd_lwh_ef`` / ``fwd_evict_first`` holds."""
    c = HOST_PLAN_CONSTANTS
    block_n, wide_n = int(c["block_n"]), int(c["wide_block_n"])
    assert set(c["dgrad_evict_first_archs"]) <= set(c["wide_dgrad_archs"])
    assert set(c["dgrad_evict_first_archs"]) <= set(c["dgrad_evict_first_min_rows"])
    assert all(int(v) > 0 for v in c["dgrad_evict_first_min_rows"].values())
    assert set(c["fwd_lwh_ef_archs"]) <= set(c["wide_fwd_archs"])
    assert set(c["fwd_evict_first_archs"]) <= set(c["wide_fwd_archs"])
    assert int(c["fwd_evict_first_min_k"]) > 0
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        wide_archs = c[f"wide_{op}_archs"]
        for arch in ROUTES[name]["arches"]:
            if arch in wide_archs:
                assert cake_jit.stage_served(name, "main_wide", arch), (name, arch)
            served_ef = cake_jit.stage_served(name, "main_wide_ef", arch)
            ef_keys = (
                ("dgrad_evict_first_archs",)
                if op == "dgrad"
                else ("fwd_lwh_ef_archs", "fwd_evict_first_archs")
            )
            if any(arch in c[key] for key in ef_keys):
                assert served_ef, (name, arch)
            if arch in c[f"{op}_wide_ts33_archs"]:
                assert cake_jit.stage_served(name, "main_wide_ts33", arch)
            for cols, red in (
                (4096, 512),
                (4096 - block_n, 512),
                (block_n, 512),
                (4096, 2048),
                (4096, 6144),
            ):
                n, k = (cols, red) if op == "fwd" else (red, cols)
                stage, tile_n = cb.select_row_stage(
                    name, op, n, k, arch, c, sum_m=4096, num_groups=4
                )
                wide = cols % wide_n == 0 and arch in wide_archs
                assert tile_n == (wide_n if wide else block_n)
                assert stage.startswith("main_wide") == wide
                ef = wide and served_ef and _evict_first(op, k, arch, 4096, 4, c)
                assert (stage == "main_wide_ef") == ef, (name, arch, n, k, stage)
    min_k = int(c["wave_sync_min_k"])
    for out_dtype in ("bfloat16", "float32"):
        for tile_k in (256, 512):
            name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=tile_k)
            assert name in ROUTES, name
            for arch in ROUTES[name]["arches"]:
                one_launch = cb.wgrad_ordered_tail(
                    tile_k, arch, c
                ) or cb.wgrad_unordered_tail(tile_k, arch, c)
                assert one_launch == (tile_k == 512), (name, arch)
                assert cake_jit.stage_served(name, "tail_reduce", arch) == (
                    not one_launch
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


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bench_and_expert_rows_take_the_measured_dgrad_stage(arch):
    """The data gradient over the MoE benchmark rows (``BENCH_WGRAD_ROWS``) and over
    ``EXPERT_ROWS`` routed rows split into each of ``EXPERT_GROUPS``, for both
    projections of ``BENCH_PROJ``: the gate/up projection (``K`` 6144) takes the
    evict-first wide tile on sm_100a at every group count (at least 4096 rows per
    group) and on sm_107a at 4 and 8 groups, the 3 / 3 head / tail wide tile at 4, 8
    and 16 groups on sm_103a, else the plain wide tile; the down projection (``K``
    2048) takes the plain wide tile."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("dgrad")
    rows = [(label, sum(sizes), len(sizes)) for label, sizes in BENCH_WGRAD_ROWS]
    rows += [(f"expert_e{groups}", EXPERT_ROWS, groups) for groups in EXPERT_GROUPS]
    taken = collections.Counter()
    for proj, (n, k) in BENCH_PROJ.items():
        for label, sum_m, groups in rows:
            gate_up = proj == "gate_up"
            ef = gate_up and (
                arch == "sm_100a" or (arch == "sm_107a" and groups in (4, 8))
            )
            ts33 = arch == "sm_103a" and gate_up and groups in (4, 8, 16)
            assert cb.dgrad_evict_first(k, arch, sum_m, groups, c) == ef, (proj, label)
            stage, tile_n = cb.select_row_stage(
                name, "dgrad", n, k, arch, c, sum_m=sum_m, num_groups=groups
            )
            assert tile_n == int(c["wide_block_n"]), (proj, label)
            expected = (
                "main_wide_ef" if ef else "main_wide_ts33" if ts33 else "main_wide"
            )
            assert stage == expected, (proj, label, stage)
            taken[stage] += 1
    # sm_100a: every gate/up row; sm_107a: the six E4 rows and E8 of the benchmark,
    # and E4 and E8 of the expert rows.
    assert taken["main_wide_ef"] == {"sm_100a": 13, "sm_103a": 0, "sm_107a": 9}[arch]
    assert taken["main_wide_ts33"] == {"sm_100a": 0, "sm_103a": 11, "sm_107a": 0}[arch]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bench_and_expert_rows_take_the_measured_fwd_stage(arch):
    """The forward pass over the rows of the data-gradient test above: the gate/up
    projection (``K`` 6144) takes the wide tile's X / Y evict-first twin on sm_100a and
    sm_103a at every group count (``fwd_evict_first``); the down projection (``K``
    2048) takes the last-wave-halves evict-first twin on sm_107a (``fwd_lwh_ef``) and
    the 3 / 3 head / tail wide tile on sm_103a at 4096 rows per group (32 groups); every
    other row the plain wide tile."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("fwd")
    rows = [(label, sum(sizes), len(sizes)) for label, sizes in BENCH_WGRAD_ROWS]
    rows += [(f"expert_e{groups}", EXPERT_ROWS, groups) for groups in EXPERT_GROUPS]
    taken = collections.Counter()
    for proj, (n, k) in BENCH_PROJ.items():
        for label, sum_m, groups in rows:
            gate_up = proj == "gate_up"
            ef = arch in (("sm_100a", "sm_103a") if gate_up else ("sm_107a",))
            ts33 = arch == "sm_103a" and not gate_up and sum_m // groups <= 4096
            assert _evict_first("fwd", k, arch, sum_m, groups, c) == ef, (proj, label)
            stage, tile_n = cb.select_row_stage(
                name, "fwd", n, k, arch, c, sum_m=sum_m, num_groups=groups
            )
            assert tile_n == int(c["wide_block_n"]), (proj, label)
            expected = (
                "main_wide_ef" if ef else "main_wide_ts33" if ts33 else "main_wide"
            )
            assert stage == expected, (proj, label, stage)
            taken[stage] += 1
    # E32 of the benchmark and of the expert rows take the 3 / 3 twin on sm_103a.
    assert taken["main_wide_ef"] == 13
    assert taken["main_wide_ts33"] == {"sm_100a": 0, "sm_103a": 2, "sm_107a": 0}[arch]


def test_workspace_arguments_of_the_argument_plans():
    """``sched`` is followed by ``c0``, ``c1`` (a queue-walk program's ``sched``, its tile
    counters, is its last argument before the grid); ``wave_sync`` by ``wave_mode``; ``tail_sums`` by
    ``tail_tickets``.  The 512-wide k tile's stages take the split tail's slots and
    tickets and no partials; every other program takes no tail slots."""
    queue_programs = {
        stage["program"]
        for route in ROUTES.values()
        for stage in route["stages"]
        if stage["name"].removesuffix(cb.ACC_SUFFIX) == cb.QUEUE_STAGE
    }
    for program_name, program in PROGRAMS.items():
        plan = [tuple(item) for item in program["arg_plan"]]
        if program_name in queue_programs:
            at = plan.index(("buffer", "sched"))
            assert all(kind == "grid" for kind, _ in plan[at + 1 :]), program_name
            assert ("parameter", "c0") not in plan, program_name
            plan = plan[:at] + plan[at + 1 :]
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
            tile_k = 512 if name.endswith("_k512") else 256
            one_launch = route["op"] == "wgrad" and any(
                cb.wgrad_ordered_tail(tile_k, arch)
                or cb.wgrad_unordered_tail(tile_k, arch)
                for arch in program["arches"]
            )
            assert (("buffer", "tail_sums") in plan) == one_launch, (name, stage)
            if one_launch:
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
    """Both ``CHANGED_SHAPES`` split their ordered one-launch tail over ``K512_E3``,
    and the second needs strictly more tail slots and tickets than the first, so the
    changed-shapes test replaces the kept buffers at every SM count."""
    small, large = (
        _wgrad_plan(K512_E3, n, k, arch, sm_count)[1] for n, k in CHANGED_SHAPES
    )
    for plan in (small, large):
        assert plan["ordered_tail"] and plan["tail_splits"] > 1, plan
        assert not plan["unordered_tail"], plan
    assert large["tail_sum_elems"] > small["tail_sum_elems"]
    assert large["tail_ticket_words"] > small["tail_ticket_words"]


@pytest.mark.parametrize("arch,sm_count", ARCH_SM_COUNTS)
def test_group_count_sweep_splits_the_tail_at_every_sm_count(arch, sm_count):
    """At ``GROUP_SHAPE`` every group count of the sweep takes the 512-wide k tile with
    the ordered one-launch tail, and at least one splits it."""
    splits = {}
    for groups in GROUP_COUNTS:
        sizes = _ragged(groups, GROUP_ROWS, seed=groups)
        _stage, plan = _wgrad_plan(sizes, *GROUP_SHAPE, arch, sm_count)
        assert plan["tile_k"] == 512 and plan["ordered_tail"], (groups, plan)
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
    out = pkg.cake_moe_grouped_gemm_nondet_train_forward(x, w, offs)
    assert out.shape == (sum(sizes), n)
    _assert_bf16_close(out, _reference("fwd", x, w, sizes))


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 256, 512), (RAGGED_E6, 1024, 512), (TINY_E5, 64, 256)]
)
def test_dgrad_matches_reference(sizes, n, k):
    device = _require_program("dgrad")
    _x, w, g, offs = _inputs(sizes, n, k, device)
    out = pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g, w, offs)
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
    """The first launch is within the budget against the fp64 reference; every repeat
    (the output refilled with NaN, the split-tail slots poisoned) gives its bits."""
    device = _require_program("wgrad", out_dtype)
    x, _w, g, offs = _inputs(sizes, n, k, device)
    launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    ref = _reference("wgrad", g, x, sizes)

    def check(out):
        assert out.shape == (len(sizes), n, k) and out.dtype == out_dtype
        _assert_wgrad_close(out, ref)
        _assert_empty_groups_zero(out, sizes)

    _repeat_bits(launch, device, REPEATS, check=check)


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
        _assert_bf16_close(
            pkg.cake_moe_grouped_gemm_nondet_train_forward(x, w, offs),
            _reference("fwd", x, w, sizes),
        )
    else:
        _assert_bf16_close(
            pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g, w, offs),
            _reference("dgrad", g, w, sizes),
        )


def test_dgrad_evict_first_gives_the_wide_tile_bits(monkeypatch):
    """Where ``dgrad_evict_first`` holds, the data gradient launches the evict-first
    twin of its wide tile (``main_wide_ef``: on sm_100a the domain-scheduled twin, on
    sm_107a the A-collector twin).  The cache policy changes no arithmetic: the
    output is within the budget of the fp64 reference, and its bits are those of the
    wide tile and of the 256-column tile."""
    device = _require_program("dgrad")
    c = HOST_PLAN_CONSTANTS
    arch = _arch(device)
    n, k = 64, int(c["dgrad_evict_first_min_k"])
    sizes = EVICT_FIRST_E2
    assert sum(sizes) // len(sizes) >= max(
        int(v) for v in c["dgrad_evict_first_min_rows"].values()
    )
    _x, w, g, offs = _inputs(sizes, n, k, device)
    ref = _reference("dgrad", g, w, sizes)
    launch = prepare_dgrad(g, w, offs)
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
    assert ef == (arch in c["dgrad_evict_first_archs"]), arch
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
    monkeypatch.setitem(
        HOST_PLAN_CONSTANTS,
        "dgrad_evict_first_min_rows",
        dict.fromkeys(c["dgrad_evict_first_min_rows"], 1 << 30),
    )
    plain = prepare_dgrad(g, w, offs)
    assert plain.stages[0] == "main_wide"
    assert plain.plan["tile_n"] == launch.plan["tile_n"]
    torch.cuda.synchronize()
    assert torch.equal(plain.launch(), launch.out)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wide_dgrad_archs", [])
    narrow = prepare_dgrad(g, w, offs)
    assert narrow.stages[0] == "main"
    torch.cuda.synchronize()
    assert torch.equal(narrow.launch(), launch.out)


@pytest.mark.parametrize("sizes", [RAGGED_E6, EVICT_FIRST_E2], ids=["ragged", "long"])
def test_fwd_evict_first_gives_the_wide_tile_bits(sizes, monkeypatch):
    """Where ``fwd_evict_first`` holds (a long reduction, the gate/up projection), the
    forward pass launches the twin of its wide tile that loads X and stores Y with the
    L2 evict-first policy (``main_wide_ef``), whatever the group sizes (an empty group
    and a one-row group among them).  The cache policy changes no arithmetic: five
    launches and a CUDA-graph replay give one set of output bits, those of the wide
    tile and of the 256-column tile."""
    device = _require_program("fwd")
    c = HOST_PLAN_CONSTANTS
    arch = _arch(device)
    n, k = 2 * int(c["wide_block_n"]), int(c["fwd_evict_first_min_k"])
    x, w, _g, offs = _inputs(sizes, n, k, device)
    ref = _reference("fwd", x, w, sizes)
    launch = prepare_fwd(x, w, offs)
    name = launch.record_name
    wide = arch in c["wide_fwd_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    ef = (
        wide
        and _evict_first("fwd", k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ef", arch)
    )
    if arch in c["fwd_evict_first_archs"]:
        assert ef, (name, arch)
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
    first = None
    for _ in range(5):
        launch.out.fill_(float("nan"))
        _assert_bf16_close(launch.launch(), ref)
        bits = launch.out.view(torch.int16).clone()
        if first is None:
            first = bits
        assert torch.equal(bits, first)
    torch.cuda.synchronize()
    if not ef:
        pytest.skip(f"no X / Y evict-first forward tile on {arch}")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    launch.out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(launch.out.view(torch.int16), first)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "fwd_evict_first_archs", [])
    plain = prepare_fwd(x, w, offs)
    assert plain.stages[0] == "main_wide"
    assert plain.plan["tile_n"] == launch.plan["tile_n"]
    torch.cuda.synchronize()
    assert torch.equal(plain.launch().view(torch.int16), first)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wide_fwd_archs", [])
    narrow = prepare_fwd(x, w, offs)
    assert narrow.stages[0] == "main"
    torch.cuda.synchronize()
    assert torch.equal(narrow.launch().view(torch.int16), first)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_group_counts_match_reference(out_dtype, monkeypatch):
    """The weight gradient at E in {1, 4, 8, 16, 32} over the groups of
    ``test_group_counts_match_reference``: every group count takes the 512-wide k tile
    with the ordered one-launch tail, at least one splits it on this device, the first
    launch is within the budget, every repeat (output refilled, slots poisoned) gives its
    bits, and a split tail's tickets rest at zero."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    n, k = GROUP_SHAPE
    splits = {}
    for groups in GROUP_COUNTS:
        sizes = _ragged(groups, GROUP_ROWS, seed=groups)
        x, _w, g, offs = _inputs(sizes, n, k, device, seed=groups)
        launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
        plan = launch.plan
        assert plan["tile_k"] == 512 and plan["ordered_tail"], (groups, plan)
        assert not launch.order_varying, plan
        ref = _reference("wgrad", g, x, sizes)

        def check(out, sizes=sizes, ref=ref):
            _assert_wgrad_close(out, ref)
            _assert_empty_groups_zero(out, sizes)

        _repeat_bits(launch, device, REPEATS, check=check)
        splits[groups] = plan["tail_splits"]
        if plan["tail_splits"] > 1:
            torch.cuda.synchronize()
            _assert_tail_at_rest(device)
    assert max(splits.values()) > 1, (
        f"no group count splits the ordered tail on this device: {splits}"
    )


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("form", K512_FORMS)
def test_ordered_tail_forms_match_reference(form, out_dtype, monkeypatch):
    """Every 512-wide k tile form the device's architecture serves, on a shape whose plan
    splits its tail: the launch is one kernel with the ordered one-launch tail, the first
    launch is within the budget against the fp64 reference with the empty group exact
    zeros, every repeat (the output refilled with NaN, then with garbage bytes; the
    stream's tail slots poisoned before every launch) gives its bits, and the stream's
    tail tickets rest at zero after the launches."""
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
    assert plan["ordered_tail"] and not plan["unordered_tail"], plan
    assert plan["tail_splits"] > 1 and plan["launches"] == 1, plan
    assert not launch.order_varying
    assert {"tail_sums", "tail_tickets"} <= _workspace_kinds(launch)
    ref = _reference("wgrad", g, x, sizes)
    worst = []

    def check(out):
        worst.append(_assert_wgrad_close(out, ref))
        _assert_empty_groups_zero(out, sizes)

    _repeat_bits(launch, device, REPEATS, fills=(None, GARBAGE_BYTE), check=check)
    torch.cuda.synchronize()
    _assert_tail_at_rest(device)
    sums = cb.stream_workspace("tail_sums", device)
    tickets = cb.stream_workspace("tail_tickets", device)
    assert sums.dtype == torch.float32 and sums.numel() >= plan["tail_sum_elems"]
    assert plan["tail_sum_elems"] == (
        plan["num_tail"]
        * plan["tail_splits"]
        * int(HOST_PLAN_CONSTANTS["cluster_m"])
        * plan["tail_tile_k"]
    )
    assert tickets.dtype == torch.uint32
    assert tickets.numel() >= plan["tail_ticket_words"]
    print(
        f"ORDERED form={form} arch={arch} n={n} k={k} splits={plan['tail_splits']} "
        f"num_tail={plan['num_tail']} worst={worst[0]:.3e}"
    )


@pytest.mark.parametrize("label,sizes", ACCURACY_ROWS)
@pytest.mark.parametrize("proj", list(BENCH_PROJ))
@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_bench_row_ops_match_reference(op, proj, label, sizes):
    """``fwd`` and ``dgrad`` over the #5678 MoE projection rows (gate/up K 6144, down
    K 2048; the benchmark rows and the E1 / E8-32 customer rows) against the fp64
    reference within the bf16 budget; a repeat into a NaN-refilled output gives the same
    bits."""
    device = _require_program(op)
    n, k = BENCH_PROJ[proj]
    x, w, g, offs = _inputs(sizes, n, k, device, seed=len(label))
    if op == "fwd":
        launch = prepare_fwd(x, w, offs)
        ref = _reference("fwd", x, w, sizes)
    else:
        launch = prepare_dgrad(g, w, offs)
        ref = _reference("dgrad", g, w, sizes)

    def check(out):
        _assert_bf16_close(out, ref)

    _repeat_bits(launch, device, 2, check=check)
    print(f"BENCHROW {op} {proj}_{label} stages={launch.stages}")


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("label,sizes", ACCURACY_ROWS)
@pytest.mark.parametrize("proj", list(BENCH_PROJ))
def test_bench_wgrad_rows_match_reference(proj, label, sizes, out_dtype):
    """The #5678 MoE projection rows (gate/up K 6144, down K 2048; the benchmark rows
    and the E1 / E8-32 customer rows) of the weight gradient: the first launch within
    the budget against the fp64 reference (but the fp32 outputs of
    ``DET_RELATIVE_FP32_ROWS``) with empty groups exact zeros, and a repeat (output
    refilled, split-tail slots poisoned) with the same bits."""
    device = _require_program("wgrad", out_dtype)
    n, k = BENCH_PROJ[proj]
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=len(label))
    launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    plan = launch.plan
    assert plan["ordered_tail"] == (plan["tile_k"] == 512), plan
    assert not plan["unordered_tail"] and not launch.order_varying, plan
    ref = _reference("wgrad", g, x, sizes)
    worst = []

    def check(out):
        if out.dtype == torch.float32 and label in DET_RELATIVE_FP32_ROWS:
            # Judged against the deterministic package in the reproducibility file.
            worst.append(float((out - ref).norm() / ref.norm().clamp_min(1e-30)))
        else:
            worst.append(_assert_wgrad_close(out, ref))
        _assert_empty_groups_zero(out, sizes)

    _repeat_bits(launch, device, 2, check=check)
    print(
        f"BENCHROW wgrad {proj}_{label} {out_dtype} stages={launch.stages} "
        f"tile_k={plan['tile_k']} ordered={plan['ordered_tail']} "
        f"splits={plan['tail_splits']} num_tail={plan['num_tail']} worst={worst[0]:.3e}"
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
    splits the ordered one-launch tail.  Returns ``(launch, sizes, g, x, offs)``."""
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
        assert plan["tile_k"] == 512 and plan["ordered_tail"], plan
        assert not plan["unordered_tail"], plan
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
    groups of one k-step, and over short groups beside long ones with a split ordered
    tail.  Repeated launches of one prepared call: the first checked block by block, and
    every later one (the split-tail slots poisoned before it) with the first's bits."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    launch, sizes, g, x, _ = _short_group_launch(
        device, groups, form, out_dtype, monkeypatch, seed=6
    )
    n, k = launch.out.shape[1:]
    ref = torch.stack([g[s:t].float().T @ x[s:t].float() for s, t in _bounds(sizes)])
    blocks = ref.view(len(sizes), n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(2, 4), keepdim=True)
    first = launch.launch().clone()
    bad = (
        ((first.float().view_as(blocks) - blocks).abs() > scale * 2.0**-6 + 1e-6)
        .any(dim=4)
        .any(dim=2)
    )
    assert not bool(bad.any()), (
        f"wrong 128 x 128 blocks [group, n block, k block]: {bad.nonzero()[:4].tolist()}"
    )
    for call in range(1, SMALL_GROUP_LAUNCHES):
        _poison_tail_sums(device)
        _assert_same_bits(launch.launch(), first, f"launch {call}")
    if launch.plan["tail_splits"] > 1:
        assert launch.plan["ordered_tail"], launch.plan
        torch.cuda.synchronize()
        _assert_tail_at_rest(device)


# The flashinfer-ai/flashinfer#6570 repro shape: 256 groups of 32 rows at (N, K).
RACE6570_SIZES = [32] * 256
RACE6570_SHAPE = (1024, 6144)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("form", (*K512_FORMS, "race6570"))
def test_short_groups_on_two_concurrent_streams(form, out_dtype, monkeypatch):
    """Two prepared weight gradients launched interleaved on two streams in flight
    together while a long-running kernel stays resident on a third stream and copies and
    GEMMs run on a fourth (flashinfer-ai/flashinfer#7038, #7548: correctness must not
    depend on other streams being idle): the mixed short / long groups with the 512-wide
    k tile forced into ``form`` and a split ordered tail, or (``"race6570"``) the #6570
    repro shape as planned.  Every result has the bits of an isolated launch on a quiet
    stream (within the budget of the fp64 reference), with each stream's split-tail
    slots poisoned before each of its launches; each stream keeps its own tail slots and
    tickets, and both streams' tickets rest at zero."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    if form == "race6570":
        sizes, (n, k) = RACE6570_SIZES, RACE6570_SHAPE
        x, _w, g, offs = _inputs(sizes, n, k, device, seed=12)
        launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    else:
        launch, sizes, g, x, offs = _short_group_launch(
            device, "mixed", form, out_dtype, monkeypatch, seed=12
        )
    split = launch.plan["tail_splits"] > 1 and launch.plan["ordered_tail"]
    launches = [launch, prepare_wgrad(g, x, offs, out_dtype=out_dtype)]
    assert launches[1].stages == launch.stages and launches[1].plan == launch.plan
    assert not launch.order_varying
    # The isolated launch: alone on the (quiet) current stream.
    torch.cuda.synchronize()
    launch.out.fill_(float("nan"))
    want = launch.launch().clone()
    torch.cuda.synchronize()
    _assert_wgrad_close(want, _reference("wgrad", g, x, sizes))
    _assert_empty_groups_zero(want, sizes)
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
    # Each launch's output words are compared with the isolated launch's on the launch's
    # own stream, before its next launch writes the output again.  A copy per launch would
    # keep 32 copies of the output: 206 GB for the #6570 fp32 output, more than a B200 has.
    words = _BIT_VIEWS[want.dtype]
    want_words = want.view(words)
    differ = [[], []]
    for _ in range(16):
        for index in range(2):
            with torch.cuda.stream(streams[index]):
                _poison_tail_sums(device, streams[index])
                launches[index].out.fill_(float("nan"))
                got = launches[index].launch().view(words)
                differ[index].append(torch.ne(got, want_words).sum())
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
    print(
        f"RESIDENT form={form} dtype={out_dtype} stages={launch.stages} "
        f"splits={launch.plan['tail_splits']} overlap={overlap}"
    )
    for index in range(2):
        for call, count in enumerate(differ[index]):
            assert int(count) == 0, (
                f"stream {index} launch {call}: {int(count)} elements differ"
            )
    if form != "race6570":
        assert split, launch.plan
    if not split:
        return
    for kind in cb.SIZED_STREAM_WORKSPACES:
        buffers = [cb.stream_workspace(kind, device, s) for s in streams]
        assert buffers[0] is not None and buffers[1] is not None, kind
        assert buffers[0].data_ptr() != buffers[1].data_ptr(), kind
        assert buffers[2] is None and buffers[3] is None, kind
    for stream in streams[:2]:
        _assert_tail_at_rest(device, stream)


@pytest.mark.parametrize("op", ["fwd", "dgrad", "wgrad"])
def test_cuda_graph_replay_with_changed_offsets(op, monkeypatch):
    """A captured launch takes workspaces of its own (those with a rest state set in the
    graph, the split-tail slots allocated without a fill) and never the buffers kept for
    a stream: the kept buffers (the split tail's filled with a sentinel) keep their bytes
    through the capture and every replay.  Every replay gives the bits of an eager launch
    on the same group boundaries, also after new group boundaries (same ``sum_m``) are
    written into ``offs`` between replays (first, second, first again).  The weight
    gradient is a 512-wide k tile shape whose plan splits its ordered one-launch tail."""
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
        assert launch.plan["ordered_tail"] and launch.plan["tail_splits"] > 1
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    # The eager bits of each layout (on the side stream, whose kept buffers the capture
    # must not touch); ``offs`` ends at the first layout, which the capture records.
    eager = {}
    with torch.cuda.stream(side):
        for sizes in (second, first):
            offs.copy_(_offs(sizes, device))
            _poison_tail_sums(device, side)
            launch.out.fill_(float("nan"))
            eager[tuple(sizes)] = launch.launch().clone()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    for sizes, ref in zip((first, second), refs, strict=True):
        _assert_bf16_close(eager[tuple(sizes)], ref)
        if op == "wgrad":
            _assert_empty_groups_zero(eager[tuple(sizes)], sizes)
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
    for sizes in (first, second, first):
        offs.copy_(_offs(sizes, device))
        for replay in range(2):
            launch.out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            _assert_same_bits(
                launch.out, eager[tuple(sizes)], f"replay {replay} of {sizes}"
            )
    for key, buffer in kept.items():
        assert torch.equal(buffer.view(torch.int32), snapshot[key]), key


def test_gradient_accumulation(monkeypatch):
    """Two backward passes accumulate into ``.grad`` on a shape whose weight gradient
    splits its ordered one-launch tail: ``x.grad`` is the bf16 sum of the two data
    gradients (bitwise); each backward's weight gradient (captured by a hook before the
    accumulation) has the bits of the direct weight-gradient call and is within the
    deterministic package's budget of its fp64 reference, ``w.grad`` is the bf16 sum of
    the two (bitwise), and an empty group's weight gradient stays exact zeros."""
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
    assert plan["ordered_tail"] and plan["tail_splits"] > 1, plan
    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    dws = []
    w.register_hook(lambda grad: dws.append(grad.detach().clone()))
    for g in (g1, g2):
        pkg.cake_moe_grouped_gemm_nondet(x, w, offs).backward(g)
    torch.cuda.synchronize()
    dx1 = pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g1, w0, offs)
    dx2 = pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g2, w0, offs)
    assert torch.equal(x.grad, dx1 + dx2)
    assert len(dws) == 2
    for dw, g in zip(dws, (g1, g2), strict=True):
        assert dw.dtype == torch.bfloat16
        _poison_tail_sums(device)
        direct = pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x0, offs)
        _assert_same_bits(dw, direct, "backward dW vs the direct weight gradient")
        _assert_wgrad_close(dw, _reference("wgrad", g, x0, sizes))
    assert torch.equal(w.grad, dws[0] + dws[1])
    _assert_empty_groups_zero(w.grad, sizes)


def test_changed_shapes_reuse_and_garbage_filled_scratch(monkeypatch):
    """Prepared launches of changing shapes on one stream (``CHANGED_SHAPES``: small,
    large, small), each with its per-CTA descriptor slots and its output filled with
    garbage bytes before the launch: the results are within the budget; the large
    launch needs strictly more tail slots and tickets and replaces both kept buffers
    with larger ones (the tickets zeroed), the later small launch reuses them (their
    slots still holding the large launch's partials) and gives the bits of the first
    small launch, and the kept tickets rest at zero after every launch."""
    device = _require_program("wgrad", torch.bfloat16)
    _fresh_stream_workspaces(monkeypatch)
    sizes = K512_E3
    cases = []
    for index, (n, k) in enumerate(CHANGED_SHAPES):
        x, _w, g, offs = _inputs(sizes, n, k, device, seed=16 + index)
        launch = prepare_wgrad(g, x, offs)
        assert launch.plan["ordered_tail"] and launch.plan["tail_splits"] > 1
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
    kept, outs = [], []
    for launch, ref in (small, large, small):
        launch.workspaces[0].fill_(0xFF)  # the per-CTA descriptor slots
        launch.out.view(torch.uint8).fill_(GARBAGE_BYTE)
        outs.append(launch.launch().clone())
        _assert_bf16_close(outs[-1], ref)
        _assert_empty_groups_zero(outs[-1], sizes)
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
    _assert_same_bits(outs[2], outs[0], "small launch after the large one")


@pytest.mark.parametrize("op", ["dgrad", "wgrad"])
def test_prepared_launches_allocate_nothing_after_the_first(op, monkeypatch):
    """After the first launch on a stream (which creates its stream workspaces), a
    prepared launch allocates no device memory: dgrad, and the weight gradient with a
    split ordered one-launch tail."""
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
        assert launch.plan["ordered_tail"] and launch.plan["tail_splits"] > 1
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


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_prepared_wgrad_follows_new_offsets(out_dtype, monkeypatch):
    """An eager prepared weight gradient of a 512-wide k tile shape whose plan splits its
    ordered one-launch tail serves new group boundaries written into ``offs`` (same
    ``sum_m``): each layout's launch is within the budget of its fp64 reference with
    exact-zero empty groups and has the bits of a fresh preparation over that layout,
    and the first layout again gives its first bits (the slots poisoned before every
    launch)."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    first = list(K512_E3)
    second = [first[0] // 2, first[2], first[0] - first[0] // 2]
    shape = _split_tail_shape(device, first, _form_ks("main_halves"))
    assert shape is not None, "no candidate shape splits the tail"
    n, k = shape
    x, _w, g, offs = _inputs(first, n, k, device, seed=23)
    launch = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    assert launch.plan["ordered_tail"] and launch.plan["tail_splits"] > 1
    bits = {}
    for sizes in (first, second, first):
        offs.copy_(_offs(sizes, device))
        _poison_tail_sums(device)
        launch.out.fill_(float("nan"))
        out = launch.launch().clone()
        _assert_wgrad_close(out, _reference("wgrad", g, x, sizes))
        _assert_empty_groups_zero(out, sizes)
        fresh_offs = _offs(sizes, device)
        fresh = prepare_wgrad(g, x, fresh_offs, out_dtype=out_dtype)
        assert fresh.plan == launch.plan
        _poison_tail_sums(device)
        _assert_same_bits(out, fresh.launch(), f"layout {sizes} vs a fresh preparation")
        if tuple(sizes) in bits:
            _assert_same_bits(out, bits[tuple(sizes)], f"layout {sizes} again")
        bits[tuple(sizes)] = out


def test_padded_outputs_leave_the_padding_untouched():
    _require_program("fwd")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(K512_E3, 512, 1024, device)
    sum_m, n, k = sum(K512_E3), 512, 1024
    padded = torch.full(
        (sum_m, n + 64), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = pkg.cake_moe_grouped_gemm_nondet_train_forward(x, w, offs, out=padded[:, :n])
    _assert_bf16_close(out, _reference("fwd", x, w, K512_E3))
    assert torch.isnan(padded[:, n:]).all()
    padded = torch.full(
        (len(K512_E3), n, k + 32), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out=padded[:, :, :k])
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
    dense = pkg.cake_moe_grouped_gemm_nondet_train_wgrad(
        g, x, offs, out_dtype=out_dtype
    )
    assert torch.equal(out, dense)
    _assert_wgrad_close(out, _reference("wgrad", g, x, sizes))


def test_autograd_backward_takes_row_strided_operands():
    """``cake_moe_grouped_gemm_nondet`` on a row-strided leaf ``x`` whose output is
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
    y = pkg.cake_moe_grouped_gemm_nondet(x, w, offs)
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
    assert pkg.cake_moe_grouped_gemm_nondet_train_forward(x, w, offs, out=y).shape == (
        0,
        256,
    )
    dw = torch.full((2, 256, 512), float("nan"), dtype=torch.float32, device=device)
    assert torch.equal(
        pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out=dw),
        torch.zeros_like(dw),
    )


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
    assert torch.equal(
        pkg.cake_moe_grouped_gemm_nondet_train_forward(x, w, m_indptr),
        pkg.cake_moe_grouped_gemm_nondet_train_forward(x, w, offs),
    )
    assert torch.equal(
        pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g, w, m_indptr),
        pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g, w, offs),
    )
    ref = _reference("wgrad", g, x, RAGGED_E6)
    _assert_bf16_close(
        pkg.cake_moe_grouped_gemm_nondet_train_wgrad(
            g, x, m_indptr, num_groups=len(RAGGED_E6)
        ),
        ref,
    )
    with pytest.raises(ValueError):  # neither E nor E + 1 entries
        pkg.cake_moe_grouped_gemm_nondet_train_wgrad(
            g, x, m_indptr[:-2], num_groups=len(RAGGED_E6)
        )


def test_autograd_wrapper_matches_reference():
    _require_program("fwd")
    _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(K512_E3, 512, 1024, device)
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = pkg.cake_moe_grouped_gemm_nondet(x, w, offs)
    assert y.shape == (sum(K512_E3), 512) and y.dtype == torch.bfloat16
    _assert_bf16_close(y.detach(), _reference("fwd", x.detach(), w.detach(), K512_E3))
    y.backward(g)
    assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.bfloat16
    _assert_bf16_close(x.grad, _reference("dgrad", g, w.detach(), K512_E3))
    _assert_bf16_close(w.grad, _reference("wgrad", g, x.detach(), K512_E3))
    _assert_empty_groups_zero(w.grad, K512_E3)
    with pytest.raises(ValueError):
        pkg.cake_moe_grouped_gemm_nondet(x.float(), w, offs)
    with pytest.raises(TypeError, match="takes no fourth argument"):
        cake_autograd.CakeMoEGroupedGemmNondetFunction.apply(x, w, offs, True)


# ---------------------------------------------------------------------------
# Autograd wrapper: the forward launch after the saves (checkpoint early stop)
# ---------------------------------------------------------------------------

# The small shape of the autograd tests: ragged groups with an empty and a one-row
# group, whose weight gradient takes the 256-wide k tile (ordered: the same bits at
# every launch, so these tests compare dW bitwise).
SMALL_SIZES = RAGGED_E4
SMALL_SHAPE = (512, 256)  # (N, K)
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


def _ordered_inputs(device, n, k, seed, sizes=SMALL_SIZES):
    """``_inputs`` of a shape whose weight gradient takes the 256-wide k tile with its
    two-launch tail reduce (checked on the host plan)."""
    _stage, plan = _wgrad_plan(sizes, n, k, _arch(device), cb.sm_count(device))
    assert plan["tile_k"] == 256 and not plan["unordered_tail"], plan
    return _inputs(sizes, n, k, device, seed=seed)


def _tile_case(device, tile):
    """``(sizes, (N, K))``: for ``"k512"`` a 512-wide k tile shape whose weight gradient
    splits its ordered one-launch tail, for ``"k256"`` the ``K256_WAVE_*`` shape of the
    256-wide k tile (two-launch tail reduce)."""
    if tile == "k256":
        _k256_wave_stage(device)
        return K256_WAVE_SIZES, K256_WAVE_SHAPE
    shape = _split_tail_shape(device, K512_E3, _form_ks("main_halves"))
    assert shape is not None, "no candidate shape splits the tail"
    return K512_E3, shape


def _wrapper(offs):
    """``gemm(a, b)`` through ``cake_moe_grouped_gemm_nondet``: the forward launch after ``apply``."""
    return lambda a, b: pkg.cake_moe_grouped_gemm_nondet(a, b, offs)


def _in_forward(offs):
    """The control arm: ``gemm(a, b)`` through a direct ``CakeMoEGroupedGemmNondetFunction.apply``, which
    launches inside ``forward``, before autograd saves the inputs."""
    return lambda a, b: cake_autograd.CakeMoEGroupedGemmNondetFunction.apply(a, b, offs)


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
    """Through ``cake_moe_grouped_gemm_nondet`` autograd saves ``x``, ``w`` and ``offs`` (the inputs
    themselves, in this order) and then the forward launches into ``Y``, saving nothing;
    a direct ``CakeMoEGroupedGemmNondetFunction.apply`` launches first and saves after.  The node is
    ``CakeMoEGroupedGemmNondetFunctionBackward`` and ``Y`` has the forward's bits either way, also with no
    rows (``sum_m == 0``)."""
    device = _require_program("fwd")
    x0, w0, _g, offs = _inputs(sizes, *SMALL_SHAPE, device, seed=40)
    want = pkg.cake_moe_grouped_gemm_nondet_train_forward(x0, w0, offs)
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
        assert type(y.grad_fn).__name__ == "CakeMoEGroupedGemmNondetFunctionBackward"
        assert torch.equal(y.detach(), want)


@pytest.mark.parametrize("tile", ["k512", "k256"])
@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_checkpoint_early_stop_skips_the_forward_launch(early_stop, tile, monkeypatch):
    """Under ``torch.utils.checkpoint`` (non-reentrant) the wrapper launches the forward
    once with the early stop on (torch's default: the recomputation stops at the third
    saved input, before the launch) and twice with it off; the control arm (direct
    ``apply``) twice either way.  ``Y``, ``dX`` and ``dW`` are bitwise the plain step's in
    both arms, on the 256-wide k tile and on the 512-wide one (split ordered one-launch
    tail); the 512-wide one's ``dW`` is also within the budget of the fp64 reference with
    exact-zero empty groups."""
    device = _require_autograd_programs()
    sizes, (n, k) = _tile_case(device, tile)
    x0, w0, g, offs = _inputs(sizes, n, k, device, seed=13)
    calls = _count_launches(monkeypatch)
    plain = _step(_wrapper(offs), _leaves(x0, w0), (g,))
    assert calls["fwd"] == [(n, k)]
    runs = {}
    for arm, gemm, forwards in (
        ("wrapper", _wrapper(offs), 1 if early_stop else 2),
        ("control", _in_forward(offs), 2),
    ):
        calls["fwd"].clear()
        runs[arm] = _step(
            gemm, _leaves(x0, w0), (g,), ckpt="non_reentrant", early_stop=early_stop
        )
        assert calls["fwd"] == [(n, k)] * forwards, arm
    for arm, ((y,), (dx, dw)) in runs.items():
        assert torch.equal(y, plain[0][0]), arm
        assert torch.equal(dx, plain[1][0]), arm
        _assert_same_bits(dw, plain[1][1], arm)
    if tile == "k512":
        ref = _reference("wgrad", g, x0, sizes)
        dw = plain[1][1]
        assert dw.dtype == torch.bfloat16
        _assert_wgrad_close(dw, ref)
        _assert_empty_groups_zero(dw, sizes)


@pytest.mark.parametrize("tile", ["k512", "k256"])
@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_checkpoint_early_stop_skips_the_forward_kernels(early_stop, tile):
    """The device side of ``test_checkpoint_early_stop_skips_the_forward_launch``: after a
    warm-up (programs loaded, stream workspaces and queue probe done) the CUDA kernels of
    a checkpointed step through the wrapper are those of the plain step with the early
    stop on, and those plus the kernels of one bare forward launch with it off."""
    device = _require_autograd_programs()
    _require_profiled_kernels(device)
    sizes, (n, k) = _tile_case(device, tile)
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
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=42)
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
@pytest.mark.parametrize("impl", ["cake", "torch_grouped_mm"])
def test_checkpoint_recompute_kernels_match_torch_grouped_mm(impl, region):
    """The regions of ``test_checkpoint_recomputes_the_forward_when_a_later_op_saves``
    through this package and through ``torch._grouped_mm`` (whose autograd saves its
    inputs before its kernel): after a warm-up the CUDA kernels of a checkpointed step are
    those of the plain step plus ``REGION_FORWARDS[region] - 1`` times those of one bare
    forward GEMM, the same pattern for both."""
    device = _require_autograd_programs()
    _require_profiled_kernels(device)
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=43)
    grads = _region_grads(region, g, x0)
    if impl == "cake":
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
        if impl == "cake":
            raise
        pytest.skip(
            f"torch._grouped_mm does not run this step on {_arch(device)}: {exc}"
        )
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
    x0, w1, _g, offs = _ordered_inputs(device, *gate_up, seed=44)
    _x, w2, g, _offs = _ordered_inputs(device, *down, seed=45)
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
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=46)
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
    """``test_gradient_accumulation`` with each step under a non-reentrant checkpoint: one
    forward launch per step, ``x.grad`` the bf16 sum of the two data gradients and
    ``w.grad`` that of the two weight gradients (bitwise), each weight gradient with the
    bits of the direct weight-gradient call and within the budget, and the empty group
    exact zeros.  One checkpointed graph backpropagated twice
    (``retain_graph=True`` first): one forward launch in all through the wrapper (both
    recomputations stop before it), three through the control arm, and ``x.grad`` twice
    the data gradient (bitwise); a third backward raises."""
    device = _require_autograd_programs()
    _fresh_stream_workspaces(monkeypatch)
    sizes, (n, k) = _tile_case(device, "k512")
    x0, w0, g1, offs = _inputs(sizes, n, k, device, seed=14)
    g2 = _inputs(sizes, n, k, device, seed=15)[2]
    calls = _count_launches(monkeypatch)
    wrapper = _wrapper(offs)
    x, w = _leaves(x0, w0)
    dws = []
    w.register_hook(lambda grad: dws.append(grad.detach().clone()))
    for g in (g1, g2):
        y = torch.utils.checkpoint.checkpoint(wrapper, x, w, use_reentrant=False)
        y.backward(g)
    torch.cuda.synchronize()
    assert calls["fwd"] == [(n, k)] * 2
    dx1 = pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g1, w0, offs)
    dx2 = pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g2, w0, offs)
    assert torch.equal(x.grad, dx1 + dx2)
    assert len(dws) == 2
    for dw, g in zip(dws, (g1, g2), strict=True):
        assert dw.dtype == torch.bfloat16
        _poison_tail_sums(device)
        direct = pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x0, offs)
        _assert_same_bits(dw, direct, "backward dW vs the direct weight gradient")
        _assert_wgrad_close(dw, _reference("wgrad", g, x0, sizes))
    assert torch.equal(w.grad, dws[0] + dws[1])
    _assert_empty_groups_zero(w.grad, sizes)
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
    """A checkpointed step with a frozen operand launches the forward once and only the
    gradient it needs: a frozen ``w`` the data gradient alone, also under torch's
    deterministic mode (no alert, which a weight gradient would raise); a frozen ``x`` the
    weight gradient alone.  The gradient has the bits of the package function."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=47)
    x = x0.clone().requires_grad_(frozen != "x")
    w = w0.clone().requires_grad_(frozen != "w")
    calls = _count_launches(monkeypatch)
    mode = _torch_deterministic() if frozen == "w" else contextlib.nullcontext()
    with mode:
        _step(_wrapper(offs), [x, w], (g,), ckpt="non_reentrant")
    assert calls["fwd"] == [SMALL_SHAPE]
    if frozen == "w":
        assert calls["dgrad"] == [SMALL_SHAPE] and calls["wgrad"] == []
        assert w.grad is None
        assert torch.equal(
            x.grad, pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g, w0, offs)
        )
    else:
        assert calls["wgrad"] == [SMALL_SHAPE] and calls["dgrad"] == []
        assert x.grad is None
        assert torch.equal(
            w.grad, pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x0, offs)
        )


@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
def test_wrapper_forward_cuda_graph_replay(grad):
    """A CUDA graph captured around ``cake_moe_grouped_gemm_nondet`` (with leaves that require grad and
    grad mode on, or under ``no_grad``) replays the forward launch the wrapper makes after
    ``apply``: after new group boundaries (same ``sum_m``) are written into ``offs``,
    every replay gives the bits of the eager wrapper on those boundaries."""
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
        eager = pkg.cake_moe_grouped_gemm_nondet(x0, w0, offs)
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
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=49)
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
    give the same gradients (``dW`` of the 256-wide k tile)."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=50)
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
        lambda a, b, o: pkg.cake_moe_grouped_gemm_nondet(a, b, o),
        (x0, w0, offs),
        check_trace=False,
    )
    calls = _count_launches(monkeypatch)
    out = traced(x1, w0, offs)
    torch.cuda.synchronize()
    assert calls["fwd"] == [SMALL_SHAPE]
    eager = pkg.cake_moe_grouped_gemm_nondet(x1, w0, offs)
    assert torch.equal(out, eager)
    assert not torch.equal(
        eager, pkg.cake_moe_grouped_gemm_nondet_train_forward(x0, w0, offs)
    )


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_wrapper_under_torch_compile(backend):
    """``torch.compile`` (``backend``) of a non-reentrant checkpointed region through the
    wrapper gives the bits of the eager checkpointed step (``Y``, ``dX``, ``dW`` of the
    256-wide k tile); under Dynamo the wrapper keeps the launch inside ``forward``
    (Dynamo cannot lift the deferred launch out of the checkpointed region).  Where the
    compiled step raises, the control arm (direct ``apply``)
    must raise the same exception type, and the case is skipped with that failure as the
    reason (a limit the wrapper did not introduce); a failure of the wrapper alone fails."""
    import torch._dynamo

    device = _require_autograd_programs()
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=53)
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
        f"torch.compile ({backend}) fails the same way through CakeMoEGroupedGemmNondetFunction.apply: "
        f"{type(failure).__name__}: {str(failure)[:400]}"
    )


def test_direct_apply_keeps_the_in_forward_launch(monkeypatch):
    """A direct ``CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)`` launches the forward inside
    ``forward`` and gives the wrapper's bits (``Y``, ``dX``, ``dW`` of the 256-wide k
    tile); a stale fourth argument (``apply(..., True)``) raises ``TypeError``; shape and
    offset errors, which the wrapper now raises after autograd saved the inputs, are
    still ``ValueError``."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _ordered_inputs(device, *SMALL_SHAPE, seed=54)
    calls = _count_launches(monkeypatch)
    wrapped = _step(_wrapper(offs), _leaves(x0, w0), (g,))
    direct = _step(_in_forward(offs), _leaves(x0, w0), (g,))
    assert calls["fwd"] == [SMALL_SHAPE] * 2
    _assert_same_step(direct, wrapped)
    x, w = _leaves(x0, w0)
    with pytest.raises(TypeError, match="takes no fourth argument"):
        cake_autograd.CakeMoEGroupedGemmNondetFunction.apply(x, w, offs, True)
    n, k = SMALL_SHAPE
    for bad_w, bad_offs in (
        (w0[:, : n - 128], offs),  # N % 256 != 0
        (w0[:, :, : k // 2], offs),  # K of w differs from x's
        (w0, offs.to(torch.int64)),
    ):
        with pytest.raises(ValueError):
            pkg.cake_moe_grouped_gemm_nondet(
                x, bad_w.clone().requires_grad_(True), bad_offs
            )
