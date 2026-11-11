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

# Acceptance checks of the deterministic ragged BF16 MoE grouped GEMM at the
# shapes of flashinfer-ai/flashinfer#5678, on top of the determinism file:
#
# * ``test_five_repeats_under_copy_streams``: every output (``Y``, ``dX``, bf16 and
#   fp32 ``dW``, fp32 ``main_grad`` accumulate on the fused route and on the composed
#   route) repeats bit for bit over 5 launches into a zero-byte output and 5 into a
#   ``FILL_BYTE`` output (the accumulate: 10 from its starting ``main_grad``, with
#   ``-0.0`` and subnormals planted), while three other streams run device-to-device
#   copies, pinned host copies and a GEMM.  The bits must equal those of a launch with
#   no other stream running, and the fused accumulate's those of the composed one.
#   Group patterns span 1 to 32 experts: highly imbalanced, empty, one-row,
#   non-tile-aligned and < 64-row groups.
# * ``test_changing_offsets_cycle``: one prepared launch serves changing group
#   offsets (the plan never reads them); cycling four layouts five times under the same
#   traffic gives, for each layout, the bits of a fresh solo launch of that layout.
# * ``test_error_vs_fp64_per_output``: the error of every output against an fp64
#   reference on a sample of each group, within the package's existing budgets
#   (bf16 ``1e-2 + 1e-2 |ref|`` per element, fp32 rel-L2 ``1e-4``).  The measured
#   errors are recorded as test properties (``max_abs``, ``max_rel``, ``rel_l2``).
# * ``test_checkpoint_replay_equals_saved_set``: a non-reentrant checkpoint's
#   recompute gives the bits of the saved-set step, with early stop on and off, with
#   and without ``fuse_wgrad_accumulation``.

import math
import zlib

import pytest
import torch
import torch.utils.checkpoint

from tests.experimental import test_cake_moe_grouped_gemm_det_train as base
from tests.experimental import test_cake_moe_grouped_gemm_det_train_determinism as det
from tests.cake.test_helpers.cake_package import cake_package

PACKAGE = "cake_moe_grouped_gemm_det_train"
cb = cake_package(PACKAGE, "cake_backend")
cake_jit = cake_package(PACKAGE, "cake_jit")
cake_autograd = cake_package(PACKAGE, "cake_autograd")
INT32_ELEMS = int(det.HOST_PLAN_CONSTANTS["int32_elems"])

REPEATS = 5  # launches per output pre-state (zero bytes, then FILL_BYTE)
FILL_BYTE = det.FILL_BYTE
PROJ = {"gate_up": (4096, 6144), "down": (6144, 2048)}  # (N, K)
SUM_M = 131072

_SMALL = [1, 2, 3, 5, 8, 13, 21, 31, 32, 33, 63, 64, 65, 127, 129]
_E32_HOT_REST = SUM_M - sum(_SMALL)
PATTERNS = {
    "e1": [16384],
    "e4_skewed": [45024, 33760, 22528, 11264],
    "e4_hot": [112000, 511, 64, 1],
    "e8_edges": [1, 63, 65, 0, 4097, 255, 257, 126334],
    "e16": [SUM_M // 16] * 16,
    "e32": [SUM_M // 32] * 32,
    "e32_hot": [0] * 15
    + _SMALL
    + [_E32_HOT_REST // 2, _E32_HOT_REST - _E32_HOT_REST // 2],
}
assert all(len(v) <= 32 for v in PATTERNS.values())
assert sum(PATTERNS["e32_hot"]) == SUM_M and len(PATTERNS["e32_hot"]) == 32
# (op, output dtype); "acc" is the fp32 main_grad accumulate (accumulate=True) on the
# route the package takes (fused where the device's accumulate builds load), "acc_composed"
# the same call forced onto the composed route (overwrite into a scratch, then add_).
OUTPUTS = (
    ("fwd", torch.bfloat16),
    ("dgrad", torch.bfloat16),
    ("wgrad", torch.bfloat16),
    ("wgrad", torch.float32),
    ("acc", torch.float32),
    ("acc_composed", torch.float32),
)
ACC_OPS = ("acc", "acc_composed")
FP32_SUBNORMAL = 1e-40

# Changing offsets: four layouts of one sum_m per expert count.
LAYOUT_EXPERTS = (4, 8, 32)


def _layouts(e):
    total = 112576 if e == 4 else SUM_M
    balanced = [total // e] * e
    balanced[-1] += total - sum(balanced)
    hot = [1] * (e - 1)
    hot = [total - sum(hot)] + hot
    edges = ([0, 1, 63, 65, 255, 257] * e)[: e - 1]
    edges = edges + [total - sum(edges)]
    return {
        "balanced": balanced,
        "hot": hot,
        "edges": edges,
        "hot_reversed": hot[::-1],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label(*parts):
    return "_".join(str(p) for p in parts)


def _gen(label, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(label.encode()))
    return gen


def _operands(op, sizes, n, k, device, label):
    """Seeded operands with the scaling of the package's tests (x and w ~ N(0,1)/sqrt(K),
    g ~ N(0,1)): (a, b, offs) for fwd (x, w), dgrad (g, w), wgrad / acc (g, x); the two
    accumulate routes of one case get the same operands."""
    gen = _gen(label.replace("_acc_composed_", "_acc_"), device)
    sum_m, e = sum(sizes), len(sizes)
    scale = 1.0 / math.sqrt(k)

    def randn(*shape, s=1.0):
        t = torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
        return t.mul_(s).to(torch.bfloat16)

    offs = base._offs(sizes, device)
    if op == "fwd":
        return randn(sum_m, k, s=scale), randn(e, n, k, s=scale), offs
    if op == "dgrad":
        return randn(sum_m, n), randn(e, n, k, s=scale), offs
    return randn(sum_m, n), randn(sum_m, k, s=scale), offs


def _out(op, dtype, a, b, num_groups):
    if op == "fwd":
        return torch.empty(a.shape[0], b.shape[1], dtype=dtype, device=a.device)
    if op == "dgrad":
        return torch.empty(a.shape[0], b.shape[2], dtype=dtype, device=a.device)
    return torch.empty(num_groups, a.shape[1], b.shape[1], dtype=dtype, device=a.device)


def _prepare(op, dtype, a, b, offs, out):
    if op == "fwd":
        return cb.cake_moe_grouped_gemm_det_train_prepare_forward(
            x=a, w=b, offs=offs, out=out
        )
    if op == "dgrad":
        return cb.cake_moe_grouped_gemm_det_train_prepare_dgrad(
            g=a, w=b, offs=offs, out=out
        )
    if op in ACC_OPS:
        return cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g=a, x=b, offs=offs, out=out, accumulate=True
        )
    return cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g=a, x=b, offs=offs, out=out, out_dtype=dtype
    )


def _force_route(op, monkeypatch):
    """``acc_composed``: every accumulate build counts as not loadable, so the call composes
    (``"min_cuda"``) without the once-per-process warning."""
    if op == "acc_composed":
        monkeypatch.setattr(cb, "acc_library_loadable", lambda *args: False)
        monkeypatch.setattr(cb, "_WARNED", {"accumulate_min_cuda"})


def _assert_route(op, launch, device):
    """``acc``: fused wherever the device's fp32 accumulate builds load (every planned shape
    here takes a half-release stage, which ships one on every architecture);
    ``acc_composed``: composed.  Returns the route."""
    route = launch.plan.get("accumulate")
    if op == "acc_composed":
        # "min_cuda" where the stage has an accumulate build (the builds count as not
        # loadable), else the first reason that fails before it.
        reason = launch.plan.get("accumulate_reason")
        stage = launch.stages[0] if launch.stages else None
        served = stage is not None and cake_jit.stage_served(
            launch.record_name, stage + cb.ACC_SUFFIX, cb.arch_for(device)
        )
        assert route == "composed" and reason == ("min_cuda" if served else reason)
    elif op == "acc" and cb.generated_program_available(
        device, "wgrad", torch.float32, accumulate=True
    ):
        assert route == "fused", launch.plan
    return route


def _require(op, dtype, device, sizes, n, k):
    prog_op = "wgrad" if op in ACC_OPS else op
    out_dtype = dtype if prog_op == "wgrad" else None
    if not cb.generated_program_available(device, op=prog_op, out_dtype=out_dtype):
        pytest.skip(f"no generated program for {prog_op} on {cb.arch_for(device)}")
    if prog_op == "wgrad" and len(sizes) * n * k > INT32_ELEMS:
        pytest.skip("the weight gradient addresses dW with 32-bit element offsets")


def _main_grad0(shape, device, label, sizes=None):
    """The starting fp32 main_grad of an accumulate case (seeded by ``label`` without its
    route, so ``acc`` and ``acc_composed`` start alike).  With ``sizes``: every empty group
    all ``-0.0`` but for a positive and a negative subnormal (it must end ``+0.0`` and keep
    them), and a subnormal of each sign in every other group."""
    label = label.replace("_acc_composed_", "_acc_")
    gen = _gen(label + "_main_grad", device)
    mg = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    for e, (s, t) in enumerate(base._bounds(sizes) if sizes else ()):
        if t == s:
            mg[e].fill_(-0.0)
        mg[e, 0, 1] = FP32_SUBNORMAL
        mg[e, 1, 0] = -FP32_SUBNORMAL
    return mg


class _Traffic:
    """Copies and a GEMM on three side streams: device-to-device copies of 256 MiB,
    pinned host-to-device / device-to-host copies of 64 MiB, and an 8192^2 bf16 mm.
    ``kick`` enqueues one round on each stream; the GEMM under test runs on its own
    stream meanwhile."""

    def __init__(self, device):
        self.device = device
        self.streams = [torch.cuda.Stream(device) for _ in range(3)]
        self.d_src = torch.empty(64 << 20, dtype=torch.float32, device=device)
        self.d_dst = torch.empty_like(self.d_src)
        self.h_src = torch.empty(16 << 20, dtype=torch.float32, pin_memory=True)
        self.h_dst = torch.empty_like(self.h_src)
        self.h_dev = torch.empty(16 << 20, dtype=torch.float32, device=device)
        self.a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
        self.c = torch.empty_like(self.a)

    def kick(self, rounds=2):
        for _ in range(rounds):
            with torch.cuda.stream(self.streams[0]):
                self.d_dst.copy_(self.d_src)
                self.d_src.copy_(self.d_dst)
            with torch.cuda.stream(self.streams[1]):
                self.h_dev.copy_(self.h_src, non_blocking=True)
                self.h_dst.copy_(self.h_dev, non_blocking=True)
            with torch.cuda.stream(self.streams[2]):
                torch.mm(self.a, self.a, out=self.c)

    def join(self, stream):
        for s in self.streams:
            stream.wait_stream(s)


def _bits(t):
    return det._bits(t)


def _launch_into(launch, out, fill, acc0):
    """Set ``out``'s pre-state (``fill`` bytes, or the accumulate's starting main_grad)
    and launch."""
    if acc0 is not None:
        out.copy_(acc0)
    else:
        out.view(torch.uint8).fill_(fill)
    launch.launch()


def _assert_empty_groups_zero(op, sizes, out, acc0):
    if op not in ("wgrad",) + ACC_OPS:
        return
    for e, (s, t) in enumerate(base._bounds(sizes)):
        if t > s:
            continue
        if op == "wgrad":
            assert int(_bits(out[e]).count_nonzero()) == 0, (
                f"group {e} is empty but its weight gradient is not exact zeros"
            )
        else:
            # An empty group adds +0.0: main_grad keeps its value (-0.0 becomes +0.0).
            want = acc0[e] + 0.0
            assert torch.equal(_bits(out[e]), _bits(want)), (
                f"group {e} is empty but its main_grad changed"
            )


# ---------------------------------------------------------------------------
# 5 repeats under concurrent copy streams
# ---------------------------------------------------------------------------

T1_CASES = [
    (pattern, proj, op, dtype)
    for pattern in PATTERNS
    for proj in PROJ
    for op, dtype in OUTPUTS
]


def _t1_id(case):
    pattern, proj, op, dtype = case
    return _label(pattern, proj, op, "f32" if dtype == torch.float32 else "bf16")


@pytest.mark.parametrize("case", T1_CASES, ids=[_t1_id(c) for c in T1_CASES])
def test_five_repeats_under_copy_streams(case, monkeypatch, record_property):
    pattern, proj, op, dtype = case
    device = det._device()
    base._fresh_stream_workspaces(monkeypatch)
    sizes = PATTERNS[pattern]
    n, k = PROJ[proj]
    _require(op, dtype, device, sizes, n, k)
    _force_route(op, monkeypatch)
    label = _t1_id(case)
    a, b, offs = _operands(op, sizes, n, k, device, label)
    out = _out(op, dtype, a, b, len(sizes))
    # The accumulate's main_grad: -0.0 in the empty groups (the contract turns it into
    # +0.0) and subnormals in every group (kept: no flush to zero).
    acc0 = _main_grad0(out.shape, device, label, sizes) if op in ACC_OPS else None

    gemm = torch.cuda.Stream(device)
    gemm.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(gemm):
        launch = _prepare(op, dtype, a, b, offs, out)
        det._settle_probe(launch, device)
        # Solo reference: no other stream runs.
        _launch_into(launch, out, 0, acc0)
    torch.cuda.synchronize(device)
    ref = _bits(out).clone()
    if op in ("fwd", "dgrad", "wgrad"):
        det._spot_check(det.Case(label, op, n, k, tuple(sizes), dtype), a, b, out)
    _assert_empty_groups_zero(op, sizes, out, acc0)

    traffic = _Traffic(device)
    mismatches = torch.zeros((), dtype=torch.int64, device=device)
    # Order the reference copy, the traffic buffers and the counter (current stream) before
    # the gemm and traffic streams use them.
    torch.cuda.synchronize(device)
    launches = 0
    for fill in (0, FILL_BYTE):
        for _ in range(REPEATS):
            traffic.kick()
            with torch.cuda.stream(gemm):
                _launch_into(launch, out, fill, acc0)
                mismatches += (_bits(out) != ref).sum()
            launches += 1
    traffic.join(gemm)
    torch.cuda.synchronize(device)
    record_property("launches", launches)
    record_property("plan", det._plan_record(launch))
    if op in ACC_OPS:
        route = _assert_route(op, launch, device)
        record_property("accumulate", route)
        record_property("stages", ",".join(launch.stages))
    bad = int(mismatches)
    assert bad == 0, (
        f"{label}: {bad} element mismatches over {launches} launches under traffic"
    )
    if op == "acc" and route == "fused":
        # The same call on the composed route gives the fused bits (int32 view).
        with monkeypatch.context() as mp:
            _force_route("acc_composed", mp)
            composed = _prepare(op, dtype, a, b, offs, out)
        assert _assert_route("acc_composed", composed, device) == "composed"
        _launch_into(composed, out, 0, acc0)
        torch.cuda.synchronize(device)
        differ = int((_bits(out) != ref).sum())
        assert differ == 0, (
            f"{label}: the composed route differs from the fused one in {differ} elements"
        )
        print(
            f"FUSEDEQ {label} stages={','.join(launch.stages)} composed={','.join(composed.stages)}"
        )


# ---------------------------------------------------------------------------
# Changing offsets
# ---------------------------------------------------------------------------

T2_CASES = [
    (e, proj, op, dtype)
    for e in LAYOUT_EXPERTS
    for proj in PROJ
    for op, dtype in OUTPUTS
]


def _t2_id(case):
    e, proj, op, dtype = case
    return _label(f"e{e}", proj, op, "f32" if dtype == torch.float32 else "bf16")


@pytest.mark.parametrize("case", T2_CASES, ids=[_t2_id(c) for c in T2_CASES])
def test_changing_offsets_cycle(case, monkeypatch, record_property):
    e, proj, op, dtype = case
    device = det._device()
    base._fresh_stream_workspaces(monkeypatch)
    n, k = PROJ[proj]
    layouts = _layouts(e)
    sizes0 = layouts["balanced"]
    _require(op, dtype, device, sizes0, n, k)
    _force_route(op, monkeypatch)
    label = _t2_id(case)
    a, b, offs = _operands(op, sizes0, n, k, device, label)
    layout_offs = {name: base._offs(sizes, device) for name, sizes in layouts.items()}
    acc0 = None
    if op in ACC_OPS:
        acc0 = _main_grad0((e, n, k), device, label)

    # Expected bits of each layout: a fresh solo prepared launch with its own offs.
    expected = {}
    for name, sizes in layouts.items():
        out = _out(op, dtype, a, b, e)
        solo = _prepare(op, dtype, a, b, layout_offs[name].clone(), out)
        det._settle_probe(solo, device)
        _launch_into(solo, out, 0, acc0)
        torch.cuda.synchronize(device)
        _assert_empty_groups_zero(op, sizes, out, acc0)
        expected[name] = _bits(out).clone()
        del solo, out

    gemm = torch.cuda.Stream(device)
    gemm.wait_stream(torch.cuda.current_stream(device))
    out = _out(op, dtype, a, b, e)
    with torch.cuda.stream(gemm):
        launch = _prepare(op, dtype, a, b, offs, out)
        det._settle_probe(launch, device)
    torch.cuda.synchronize(device)
    if op in ACC_OPS:
        record_property("accumulate", _assert_route(op, launch, device))
    traffic = _Traffic(device)
    mismatches = {
        name: torch.zeros((), dtype=torch.int64, device=device) for name in layouts
    }
    # Order the traffic buffers and the counters (current stream) before the gemm and
    # traffic streams use them.
    torch.cuda.synchronize(device)
    launches = 0
    for cycle in range(REPEATS):
        for name in layouts:
            traffic.kick(1)
            with torch.cuda.stream(gemm):
                offs.copy_(layout_offs[name])
                _launch_into(launch, out, FILL_BYTE if cycle % 2 else 0, acc0)
                mismatches[name] += (_bits(out) != expected[name]).sum()
            launches += 1
    traffic.join(gemm)
    torch.cuda.synchronize(device)
    record_property("launches", launches)
    bad = {name: int(v) for name, v in mismatches.items() if int(v)}
    assert not bad, f"{label}: element mismatches per layout {bad}"


# ---------------------------------------------------------------------------
# Error against fp64, per output
# ---------------------------------------------------------------------------

T3_PATTERNS = {
    "e4_m112576": [28128, 28128, 28128, 28192],
    "e4_m136064_skewed": [54432, 40832, 27200, 13600],
    "e8": [SUM_M // 8] * 8,
    "e32": [SUM_M // 32] * 32,
    "e32_hot": PATTERNS["e32_hot"],
}
T3_CASES = [
    (pattern, proj, op, dtype)
    for pattern in T3_PATTERNS
    for proj in PROJ
    for op, dtype in OUTPUTS
]
SAMPLE_ROWS = 16  # first and last rows of each group for Y / dX
SAMPLE_COLS = 512
SAMPLE_DW = (64, 256)  # (N rows, K columns) of each group's dW


def _t3_id(case):
    pattern, proj, op, dtype = case
    return _label(pattern, proj, op, "f32" if dtype == torch.float32 else "bf16")


def _sample_ref(op, a, b, sizes, out, acc0):
    """(got, ref64) over the sample of every non-empty group."""
    got, ref = [], []
    for e, (s, t) in enumerate(base._bounds(sizes)):
        if t == s:
            continue
        if op in ("fwd", "dgrad"):
            rows = list(range(s, min(t, s + SAMPLE_ROWS)))
            rows += [r for r in range(max(s, t - SAMPLE_ROWS), t) if r not in rows]
            idx = torch.tensor(rows, device=a.device)
            if op == "fwd":
                c = min(b.shape[1], SAMPLE_COLS)
                r64 = a[idx].double() @ b[e, :c].double().T
            else:
                c = min(b.shape[2], SAMPLE_COLS)
                r64 = a[idx].double() @ b[e][:, :c].double()
            got.append(out[idx, :c].double())
            ref.append(r64)
        else:
            rn, ck = min(a.shape[1], SAMPLE_DW[0]), min(b.shape[1], SAMPLE_DW[1])
            r64 = a[s:t, :rn].double().T @ b[s:t, :ck].double()
            if op in ACC_OPS:
                r64 = r64 + acc0[e, :rn, :ck].double()
            got.append(out[e, :rn, :ck].double())
            ref.append(r64)
    return got, ref


@pytest.mark.parametrize("case", T3_CASES, ids=[_t3_id(c) for c in T3_CASES])
def test_error_vs_fp64_per_output(case, monkeypatch, record_property):
    pattern, proj, op, dtype = case
    device = det._device()
    base._fresh_stream_workspaces(monkeypatch)
    sizes = T3_PATTERNS[pattern]
    n, k = PROJ[proj]
    _require(op, dtype, device, sizes, n, k)
    _force_route(op, monkeypatch)
    label = _t3_id(case)
    a, b, offs = _operands(op, sizes, n, k, device, label)
    out = _out(op, dtype, a, b, len(sizes))
    acc0 = _main_grad0(out.shape, device, label) if op in ACC_OPS else None
    launch = _prepare(op, dtype, a, b, offs, out)
    if op in ACC_OPS:
        record_property("accumulate", _assert_route(op, launch, device))
    _launch_into(launch, out, 0, acc0)
    torch.cuda.synchronize(device)
    got, ref = _sample_ref(op, a, b, sizes, out, acc0)
    got, ref = (
        torch.cat([g.flatten() for g in got]),
        torch.cat([r.flatten() for r in ref]),
    )
    assert bool(torch.isfinite(got).all()), f"{label}: non-finite sampled outputs"
    err = (got - ref).abs()
    max_abs = float(err.max())
    max_rel = float((err / ref.abs().clamp_min(1e-30)).max())
    rel_l2 = float(err.norm() / ref.norm().clamp_min(1e-30))
    record_property("max_abs", max_abs)
    record_property("max_rel", max_rel)
    record_property("rel_l2", rel_l2)
    record_property("samples", int(got.numel()))
    print(
        f"ERR {label} max_abs={max_abs:.4e} max_rel={max_rel:.4e} rel_l2={rel_l2:.4e} n={got.numel()}"
    )
    if dtype == torch.bfloat16:
        bound = base.ATOL + base.RTOL * ref.abs()
        bad = int((~(err <= bound)).sum())  # a NaN counts as beyond the budget
        assert bad == 0, f"{label}: {bad} bf16 elements beyond 1e-2 + 1e-2 |ref64|"
    else:
        assert rel_l2 <= base.FP32_REL_L2, f"{label}: fp32 rel-L2 {rel_l2:.3e}"


# ---------------------------------------------------------------------------
# Checkpoint replay equals the saved-set step
# ---------------------------------------------------------------------------

T5_CASES = [
    (sizes_name, proj, fuse)
    for sizes_name in ("e8", "e32_hot")
    for proj in PROJ
    for fuse in (False, True)
]


def _t5_id(case):
    sizes_name, proj, fuse = case
    return _label(sizes_name, proj, "main_grad" if fuse else "grad")


def _step(x0, w0, g, offs, fuse, checkpoint, early_stop):
    x = x0.detach().requires_grad_(True)
    w = w0.detach().requires_grad_(True)
    if fuse:
        w.main_grad = torch.zeros(w.shape, dtype=torch.float32, device=w.device)
        w.grad_added_to_main_grad = False

    def region(x_, w_, offs_):
        return cake_autograd.cake_moe_grouped_gemm_det(
            x_, w_, offs_, fuse_wgrad_accumulation=fuse
        )

    if checkpoint:
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            y = torch.utils.checkpoint.checkpoint(
                region, x, w, offs, use_reentrant=False
            )
            y.backward(g)
    else:
        y = region(x, w, offs)
        y.backward(g)
    wg = w.main_grad if fuse else w.grad
    return y.detach(), x.grad, wg


@pytest.mark.parametrize("case", T5_CASES, ids=[_t5_id(c) for c in T5_CASES])
def test_checkpoint_replay_equals_saved_set(case, monkeypatch):
    sizes_name, proj, fuse = case
    device = det._device()
    base._fresh_stream_workspaces(monkeypatch)
    sizes = {"e8": [SUM_M // 8] * 8, "e32_hot": PATTERNS["e32_hot"]}[sizes_name]
    n, k = PROJ[proj]
    for op in ("fwd", "dgrad"):
        _require(op, torch.bfloat16, device, sizes, n, k)
    _require(
        "acc" if fuse else "wgrad",
        torch.float32 if fuse else torch.bfloat16,
        device,
        sizes,
        n,
        k,
    )
    label = _t5_id(case)
    gen = _gen(label, device)
    scale = 1.0 / math.sqrt(k)
    sum_m = sum(sizes)
    x0 = (torch.randn(sum_m, k, generator=gen, device=device) * scale).to(
        torch.bfloat16
    )
    w0 = (torch.randn(len(sizes), n, k, generator=gen, device=device) * scale).to(
        torch.bfloat16
    )
    g = torch.randn(sum_m, n, generator=gen, device=device).to(torch.bfloat16)
    offs = base._offs(sizes, device)
    saved = _step(x0, w0, g, offs, fuse, checkpoint=False, early_stop=False)
    for early_stop in (False, True):
        replay = _step(x0, w0, g, offs, fuse, checkpoint=True, early_stop=early_stop)
        for name, want, got in zip(
            ("y", "x.grad", "w grad"), saved, replay, strict=True
        ):
            assert torch.equal(_bits(got), _bits(want)), (
                f"{label} early_stop={early_stop}: {name} of the checkpoint replay differs "
                f"from the saved-set step in {int((_bits(got) != _bits(want)).sum())} elements"
            )
