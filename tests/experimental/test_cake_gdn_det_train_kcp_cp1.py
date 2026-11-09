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

The CP = 1 local path of the Cake packed GDN training package
(``flashinfer.cake_gdn_det_train_kcp`` with a one-rank context) against the exact FP64
reference of ``tests/experimental/test_helpers/cake_gdn_det_train_kcp_reference.py``.

Covered here: the six outputs ``out, dq, dk, dv, dbeta, dg`` on ragged packs
of the normal gate regime (natural-log gates down to the -42 class; zero
elementwise mismatches), the adversarial weak-erasure regimes (contract,
finiteness and a relative-RMS sanity bound; mismatch counts recorded as
evidence), the six head pairs (16, 48), (8, 24), (4, 12),
(2, 6), (1, 3), (4, 4), the degenerate one-token / zero-token packs, the packed
QKV strided view against the contiguous input, and the explicit forward /
backward entries against autograd (output order ``dq, dk, dv, dbeta, dg``,
saved-context against replay), the fused output walk against the split forward
on the same build (test knob ``fused_o`` 1 / 0, bitwise), and ``save_mode="full"`` without ``save_transfer`` at
``P = 2`` under NaN-filled allocations against the light form (bitwise).  The multi-rank rows, the bitwise contracts and
the runtime contracts live in ``test_cake_gdn_det_train_kcp_dist.py``; the CPU plan,
table and byte-query tests in ``test_cake_gdn_det_train_kcp.py``.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from tests.experimental.test_helpers.cake_gdn_det_train_kcp_reference import (
    BLOCK,
    D,
    GRAD_NAMES,
    STORAGE_OF,
    TENSORS,
    elementwise_mismatch,
    error_metrics,
    head_params,
    make_inputs,
    reference_pack,
)
from tests.cake.test_helpers.cake_package import PACKAGES, cake_package

gdn_api = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")

# Accuracy gate, declared before the first run: every output elementwise within
# ``|cand - ref| <= ATOL + RTOL * |ref|`` of the exact FP64 reference of the same BF16 / FP32 inputs with ZERO
# mismatches on all six tensors; ``out`` / ``dq`` / ``dk`` / ``dv`` are compared as stored BF16, ``dg`` / ``dbeta`` as
# FP32.  The rel-RMS / max-abs of every tensor is recorded beside the verdict (the budget tables of the multi-rank
# rows are derived from the same metrics).
ATOL = RTOL = 1e-2
# The zero-mismatch gate is the contract of the normal gate regime (the correctness rows).  The adversarial weak-erasure
# regimes (slow isotropic / anisotropic decay over one long document) are judged by the multi-rank budget tables with
# their negative control; here they keep the contract, finiteness and a sanity bound on the relative RMS error, and
# their mismatch counts are recorded as evidence.  Bound declared before the first run of these shapes.
GATED_REGIMES = ("normal", "normal+decay4")
ADVERSARIAL_REL_RMS_MAX = 1e-2
OPERANDS = ("q", "k", "v", "g", "beta")
EXPLICIT_GRAD_ORDER = (
    "dq",
    "dk",
    "dv",
    "dbeta",
    "dg",
)  # the backward entries' output order
STORAGE_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}

# The pinned-decay GEN row: the normal regime's per-head decay rate e^A is raised to at least PINNED_DECAY_MIN, so every
# full 64-token chunk's gate sum |G_L| exceeds 100: the class where e^{G_L} overflows FP32 unless the kernel rescales
# within the chunk.
PINNED_DECAY_MIN = 4.0
PINNED_CHUNK_GATE_SUM_MIN = 100.0

# id, document lengths (none a multiple of 64 unless the case says so), (H_qk, HV), operand regime
SHAPES = (
    ("c1-2x6-ragged", (100, 640, 77, 1000, 129), (2, 6), "normal"),
    ("c1-2x6-gen-decay-pinned", (100, 640, 77, 1000, 129), (2, 6), "normal+decay4"),
    ("c2-4x12-one-doc", (3000,), (4, 12), "normal"),
    ("c3-1x3-tiny-docs", (2048, 5, 3000, 1), (1, 3), "normal"),
    ("c4-8x24-chunk-interior", (1500, 1500), (8, 24), "normal"),
    ("c5-4x4-63-64-65", (63 + 64 + 65 + 500,), (4, 4), "normal"),
    ("c8s-16x48", (2048 + 37, 1000), (16, 48), "normal"),
    ("a1s-2x6-slow-decay", (1201,), (2, 6), "A1"),
    ("a2s-2x6-anisotropic", (1301,), (2, 6), "A2"),
    ("one-token", (1,), (2, 6), "normal"),
    ("zero-tokens", (0,), (2, 6), "normal"),
)


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def _inputs(cu: list[int], heads: tuple[int, int], regime: str) -> dict:
    """The row's operands; ``"normal+decay4"`` is the normal regime with every value head's decay rate e^A raised
    to at least PINNED_DECAY_MIN (g = -max(e^A, 4) * softplus(x), i.e. g scaled by max(1, 4 / e^A) per head)."""
    if regime != "normal+decay4":
        return make_inputs(cu, *heads, regime=regime)
    inputs = make_inputs(cu, *heads, regime="normal")
    rate = torch.exp(
        head_params(*heads)["a_log"]
    )  # [HV] decay rates e^A of the normal regime
    inputs["g"] = (
        (inputs["g"] * torch.clamp(PINNED_DECAY_MIN / rate, min=1.0))
        .float()
        .contiguous()
    )
    for d in range(len(cu) - 1):
        for c0 in range(cu[d], cu[d + 1] - BLOCK + 1, BLOCK):
            gate_sum = inputs["g"][0, c0 : c0 + BLOCK].sum(
                0
            )  # [HV] per full in-document chunk
            assert bool((gate_sum.abs() > PINNED_CHUNK_GATE_SUM_MIN).all()), (
                d,
                c0,
                gate_sum,
            )
    return inputs


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    if not gdn_api.is_cake_gdn_det_train_kcp_supported(index):
        pytest.skip("the package does not support this device")
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _local_context(cu: list[int], device: torch.device):
    """The one-rank context of the local path (the public entry names; group None = single process)."""
    part = gdn_api.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, layout="contiguous"
    )
    return gdn_api.cake_gdn_det_train_kcp_build_context(part, None, 0, device)


def _run_fwd_bwd(inputs: dict, ctx, device: torch.device) -> dict:
    """Autograd forward + backward of the local path; the results keyed by tensor name (no batch axis)."""
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    out = gdn_api.cake_gdn_det_kcp(*leaves, ctx)
    grads = torch.autograd.grad(out, leaves, inputs["dout"].to(device))
    res = {"out": out.detach()[0]}
    res.update({n: gr.detach()[0] for n, gr in zip(GRAD_NAMES, grads, strict=True)})
    return res


def _check_contract(res: dict, inputs: dict) -> None:
    T, H, HV = inputs["q"].shape[1], inputs["q"].shape[2], inputs["v"].shape[2]
    expect = {
        "out": (T, HV, D),
        "dq": (T, H, D),
        "dk": (T, H, D),
        "dv": (T, HV, D),
        "dg": (T, HV),
        "dbeta": (T, HV),
    }
    for n in TENSORS:
        assert tuple(res[n].shape) == expect[n], (n, tuple(res[n].shape))
        assert res[n].dtype == STORAGE_DTYPE[STORAGE_OF[n]], (n, res[n].dtype)
        assert bool(torch.isfinite(res[n]).all()), f"{n} has non-finite values"


@pytest.mark.parametrize(
    ("lengths", "heads", "regime"),
    [s[1:] for s in SHAPES],
    ids=[s[0] for s in SHAPES],
)
def test_local_path_matches_the_exact_reference(
    device, record_property, lengths, heads, regime
) -> None:
    cu = _cu(lengths)
    inputs = _inputs(cu, heads, regime)
    ref, _ = reference_pack(inputs, device)
    res = _run_fwd_bwd(inputs, _local_context(cu, device), device)
    _check_contract(res, inputs)
    rows, failures = [], []
    for n in TENSORS:
        r = ref[n].to(device)
        m = error_metrics(res[n], r)
        e = elementwise_mismatch(res[n], r, ATOL, RTOL)
        rows.append(
            f"| {heads} | {regime} | {cu[-1]} | {n} | {STORAGE_OF[n]} | {m['rel_rms']:.3e} | {m['max_abs']:.3e} | {e['mismatch']} / {e['numel']} |"
        )
        record_property(f"{n}_rel_rms", m["rel_rms"])
        record_property(f"{n}_max_abs", m["max_abs"])
        record_property(f"{n}_mismatch", e["mismatch"])
        if regime in GATED_REGIMES and not e["pass"]:
            failures.append(
                f"{n}: {e['mismatch']} of {e['numel']} elements outside |cand - ref| <= {ATOL} + {RTOL} |ref| (worst excess {e['worst_excess']:.3e}), rel-RMS {m['rel_rms']:.3e}, max-abs {m['max_abs']:.3e}, worst {m['worst']}"
            )
        if regime not in GATED_REGIMES and not m["rel_rms"] <= ADVERSARIAL_REL_RMS_MAX:
            failures.append(
                f"{n}: adversarial regime {regime} rel-RMS {m['rel_rms']:.3e} > {ADVERSARIAL_REL_RMS_MAX} (max-abs {m['max_abs']:.3e}, {e['mismatch']} of {e['numel']} elements outside the elementwise band, worst {m['worst']})"
            )
    print(
        "\n| heads | regime | tokens | tensor | storage | rel-RMS | max-abs | mismatches |\n|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
    )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    ("lengths", "heads"),
    [((300, 777), (2, 6)), ((1000, 37), (16, 48))],
    ids=["2x6", "16x48"],
)
def test_packed_qkv_strided_view_equals_the_contiguous_input(
    device, lengths, heads
) -> None:
    H, HV = heads
    cu = _cu(lengths)
    inputs = make_inputs(cu, H, HV)
    T = cu[-1]
    # the packed QKV row: q at head 0, k at head H, v at head 2H; token stride (2H + HV) * 128 elements
    packed = torch.empty(1, T, 2 * H + HV, D, dtype=torch.bfloat16, device=device)
    packed[:, :, :H] = inputs["q"].to(device)
    packed[:, :, H : 2 * H] = inputs["k"].to(device)
    packed[:, :, 2 * H :] = inputs["v"].to(device)
    views = {
        "q": packed[:, :, :H],
        "k": packed[:, :, H : 2 * H],
        "v": packed[:, :, 2 * H :],
    }
    for n, t in views.items():
        assert not t.is_contiguous(), n
        assert t.stride(1) == (2 * H + HV) * D, (n, t.stride())
        assert torch.equal(t.cpu(), inputs[n]), n
    ctx = _local_context(cu, device)
    contiguous = _run_fwd_bwd(inputs, ctx, device)
    strided = _run_fwd_bwd({**inputs, **views}, ctx, device)
    for n in TENSORS:
        assert torch.equal(strided[n], contiguous[n]), (
            f"{n}: strided view differs from the contiguous input"
        )


def _split_view_pass(form, row0, g0, b0, dout, ctx, heads, *, ckpt, **o) -> dict:
    """One autograd pass with q / k / v taken from the packed ``[1, T, (2 H_qk + HV) * 128]`` row ``row0``:
    ``"plain"`` = ``split`` + ``view`` head views, ``"packed"`` = ``cake_gdn_det_train_kcp_packed_qkv_views``, ``"contiguous"`` =
    separate contiguous leaves holding the same values.  A tensor hook (recording the gradient) and ``retain_grad``
    are set on every view (and on the split outputs) of the original forward."""
    H, HV = heads
    T = row0.shape[1]
    sizes = (H * D, H * D, HV * D)
    row = row0.detach().clone().requires_grad_(True)
    g, beta = (x.detach().clone().requires_grad_(True) for x in (g0, b0))
    leaves = []
    if form == "contiguous":
        leaves = [
            p.detach()
            .clone(memory_format=torch.contiguous_format)
            .view(1, T, n, D)
            .requires_grad_(True)
            for p, n in zip(row0.split(sizes, dim=-1), (H, H, HV), strict=True)
        ]
    fired: dict = {}
    views: dict = {}

    def fn(row, g, beta, *leaves):
        parts = ()
        if form == "contiguous":
            q, k, v = leaves
        elif form == "plain":
            parts = row.split(sizes, dim=-1)
            q, k, v = (
                p.view(1, T, n, D) for p, n in zip(parts, (H, H, HV), strict=True)
            )
        else:
            q, k, v = gdn_api.cake_gdn_det_train_kcp_packed_qkv_views(
                row, H_qk=H, HV=HV
            )
        if not views:  # the original forward (a checkpoint recompute builds a graph no backward runs)
            views.update(q=q, k=k, v=v, **{f"part{i}": p for i, p in enumerate(parts)})
            for name, t in views.items():
                t.register_hook(
                    lambda gr, name=name: fired.setdefault(name, []).append(gr.clone())
                )
                t.retain_grad()
        return gdn_api.cake_gdn_det_kcp(q, k, v, g, beta, ctx, **o)

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, row, g, beta, *leaves, use_reentrant=False)
    else:
        out = fn(row, g, beta, *leaves)
    out.backward(dout)
    drow = (
        torch.cat([x.grad.reshape(1, T, -1) for x in leaves], -1)
        if leaves
        else row.grad
    )
    return {
        "out": out.detach(),
        "row": drow,
        "dg": g.grad,
        "dbeta": beta.grad,
        "fired": fired,
        "views": views,
    }


@pytest.mark.parametrize("ckpt", [False, True], ids=["direct", "checkpoint"])
@pytest.mark.parametrize("norm", [False, True], ids=["l2n-off", "l2n-on"])
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [((300, 777), (2, 6)), ((2048 + 37, 1000), (16, 48))],
    ids=["2x6", "16x48"],
)
def test_plain_split_views_keep_hooks_and_bitwise_gradients(
    device, lengths, heads, norm, ckpt
) -> None:
    """Issue #8367 on the device programs: plain ``split`` / ``view`` head views of one packed QKV row, the
    ``cake_gdn_det_train_kcp_packed_qkv_views`` views of it and contiguous operands give bitwise the same ``out``, row gradient
    (``dq | dk | dv``), ``dg`` and ``dbeta``, with the in-kernel norm off and on, direct and checkpointed.  Plain views
    keep torch's autograd path: their tensor hooks and ``retain_grad`` (and those on the split outputs) fire once
    with the gradient, and only the packed views' split hands the packed dQKV row on without a concatenation."""
    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    H, HV = heads
    cu = _cu(lengths)
    T = cu[-1]
    inputs = make_inputs(cu, H, HV)
    ctx = _local_context(cu, device)
    row0 = torch.cat(
        [inputs[n].to(device).reshape(1, T, -1) for n in ("q", "k", "v")], -1
    ).contiguous()
    g0, b0, dout = (inputs[n].to(device) for n in ("g", "beta", "dout"))
    o = {"use_qk_l2norm_in_kernel": norm}
    cols = {"q": (0, H * D), "k": (H * D, 2 * H * D), "v": (2 * H * D, None)}
    expect_split = {
        "contiguous": {"aliased": 0, "concatenated": 0},
        "plain": {"aliased": 0, "concatenated": 0},
        "packed": {"aliased": 1, "concatenated": 0},
    }
    ref = None
    for form, split in expect_split.items():
        before = dict(kcp.PACKED_SPLIT_STATS)
        got = _split_view_pass(form, row0, g0, b0, dout, ctx, heads, ckpt=ckpt, **o)
        delta = {n: kcp.PACKED_SPLIT_STATS[n] - before[n] for n in before}
        assert delta == split, (form, delta)
        assert kcp.LAST_CALL["bwd"]["grad_layout"] == (
            "separate" if form == "contiguous" else "packed"
        ), form
        if ref is None:
            ref = got
        for n in ("out", "row", "dg", "dbeta"):
            assert torch.equal(got[n], ref[n]), f"{form}: {n} differs from contiguous"
        assert sorted(got["fired"]) == sorted(got["views"]), form
        for i, n in enumerate(("q", "k", "v")):
            a, b = cols[n]
            want = ref["row"][..., a:b]
            assert len(got["fired"][n]) == 1, (form, n)
            assert torch.equal(got["fired"][n][0].reshape(want.shape), want), (form, n)
            assert torch.equal(got["views"][n].grad.reshape(want.shape), want), (
                form,
                n,
            )
            if form == "plain":
                part = got["fired"][f"part{i}"]
                assert len(part) == 1 and torch.equal(part[0], want), (form, n)


def test_explicit_entries_match_autograd_and_keep_the_output_order(device) -> None:
    cu = _cu((100, 640, 77))
    inputs = make_inputs(cu, 2, 6)
    ctx = _local_context(cu, device)
    ops = [inputs[n].to(device) for n in OPERANDS]
    dout = inputs["dout"].to(device)
    out, saved = gdn_api.cake_gdn_det_train_kcp_forward(
        *ops, ctx, save_for_backward=True
    )
    grads = gdn_api.cake_gdn_det_train_kcp_backward(*ops, ctx, dout, saved=saved)
    assert len(grads) == len(EXPLICIT_GRAD_ORDER)
    auto = _run_fwd_bwd(inputs, ctx, device)
    assert torch.equal(out[0], auto["out"])
    for n, t in zip(EXPLICIT_GRAD_ORDER, grads, strict=True):
        assert t.dtype == STORAGE_DTYPE[STORAGE_OF[n]], (n, t.dtype)
        assert torch.equal(t[0], auto[n]), (
            f"{n} differs between the explicit entry and autograd"
        )
    replay = gdn_api.cake_gdn_det_train_kcp_backward(*ops, ctx, dout, saved=None)
    for n, t_saved, t_replay in zip(EXPLICIT_GRAD_ORDER, grads, replay, strict=True):
        assert torch.equal(t_saved, t_replay), (
            f"{n}: saved-context and replay backward differ"
        )


@pytest.mark.skipif(
    "cake_gdn_nondet_train_kcp" not in PACKAGES,
    reason="pending integration: the non-deterministic package is not in this tree's PACKAGES table",
)
def test_backward_refuses_the_non_deterministic_forwards_saved_set_and_workspace(
    device,
) -> None:
    """A saved set and a caller-held workspace of the non-deterministic package's forward (kernels, same tokens) are
    refused by the deterministic package backward with ``ValueError`` naming the deterministic mode and the expected
    type; the deterministic forward's own saved set and workspace still run."""
    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    # the non-deterministic package under its FlashInfer #8928 path and names; it ships with this package, so a
    # package that does not import fails here, it is not skipped
    nd = cake_package("cake_gdn_nondet_train_kcp")
    cu = _cu((100, 640, 77))
    inputs = make_inputs(cu, 2, 6)
    ops = [inputs[n].to(device) for n in OPERANDS]
    dout = inputs["dout"].to(device)
    nd_part = nd.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, layout="contiguous"
    )
    nd_ws = nd.CakeGDNNondetKcpWorkspace()
    _, nd_saved = nd.cake_gdn_nondet_train_kcp_forward(
        *ops,
        nd.cake_gdn_nondet_train_kcp_build_context(nd_part, None, 0, device),
        workspace=nd_ws,
    )
    assert type(nd_saved) is nd.CakeGDNNondetKcpSaved
    ctx = _local_context(cu, device)
    ws = kcp.CakeGDNDetKcpWorkspace()
    _, saved = kcp.cake_gdn_det_train_kcp_forward(*ops, ctx, workspace=ws)
    with pytest.raises(
        ValueError, match="saved must be a CakeGDNDetKcpSaved of the deterministic mode"
    ):
        kcp.cake_gdn_det_train_kcp_backward(
            *ops, ctx, dout, saved=nd_saved, workspace=ws
        )
    with pytest.raises(
        ValueError,
        match="workspace must be a CakeGDNDetKcpWorkspace of the deterministic mode",
    ):
        kcp.cake_gdn_det_train_kcp_backward(
            *ops, ctx, dout, saved=saved, workspace=nd_ws
        )
    grads = kcp.cake_gdn_det_train_kcp_backward(
        *ops, ctx, dout, saved=saved, workspace=ws
    )
    assert all(bool(torch.isfinite(t).all()) for t in grads)


@pytest.mark.parametrize("ckpt", [False, True], ids=["direct", "checkpoint"])
def test_every_launch_binds_the_caller_stream_once(device, monkeypatch, ckpt) -> None:
    """The kernels backend's launch path on a caller (side) stream: an autograd forward + backward, direct and under a
    non-reentrant checkpoint, binds the caller's stream for tvm-ffi (``tvm_ffi.use_raw_stream`` on the stream's device
    and raw handle) exactly once per launch, every binding is the caller stream, and the results are bitwise the same
    call on the default stream."""
    from torch.utils.checkpoint import checkpoint

    be = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_backend"
    )
    cu = _cu((100, 640, 77, 1000, 129))
    inputs = make_inputs(cu, 2, 6)
    ctx = _local_context(cu, device)
    dout = inputs["dout"].to(device)

    def run() -> dict:
        leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]

        def fn(*xs):
            return gdn_api.cake_gdn_det_kcp(*xs, ctx)

        out = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
        grads = torch.autograd.grad(out, leaves, dout)
        res = {"out": out.detach()[0]}
        res.update({n: gr.detach()[0] for n, gr in zip(GRAD_NAMES, grads, strict=True)})
        return res

    want = run()  # the default stream, the glue as shipped
    torch.cuda.synchronize(device)
    real_ffi, real_run = be.tvm_ffi, be.run_module
    bound, launches = [], [0]

    class _Counting:
        def __getattr__(self, name):
            return getattr(real_ffi, name)

        @staticmethod
        def use_raw_stream(dev, raw):
            bound.append((dev.index, int(raw)))
            return real_ffi.use_raw_stream(dev, raw)

    def counting_run(*a, **kw):
        launches[0] += 1
        return real_run(*a, **kw)

    monkeypatch.setattr(be, "tvm_ffi", _Counting())
    monkeypatch.setattr(be, "run_module", counting_run)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        got = run()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    assert launches[0] > 0 and len(bound) == launches[0], (len(bound), launches[0])
    assert set(bound) == {(device.index, side.cuda_stream)}, set(bound)
    for n in TENSORS:
        assert torch.equal(got[n], want[n]), n


# --------------------------------------------------------------------------- issue #7074 (1): sync-free context preparation (one GPU)
ISSUE_CU = [
    0,
    131071,
    218602,
    261987,
]  # the issue's layout: 261,987 tokens in 3 documents, (H_qk, HV) = (2, 6)


def test_context_preparation_does_not_synchronize(device) -> None:
    """``cake_gdn_det_train_kcp_partition`` + ``cake_gdn_det_train_kcp_build_context`` + ``prepare`` of a new layout make no synchronizing CUDA call (the
    tables travel in pinned, asynchronous copies), and every device table is bitwise the table built on the CPU."""
    plan = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp_plan"
    )
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        part = gdn_api.cake_gdn_det_train_kcp_partition(
            torch.tensor(ISSUE_CU, dtype=torch.int64), 1, layout="contiguous"
        )
        ctx = gdn_api.cake_gdn_det_train_kcp_build_context(part, None, 0, device)
        prepared = ctx.prepare(2, 6)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(device)
    ref = plan.cake_gdn_det_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=ctx.sm_count
    )
    ref_prepared = ref.prepare(2, 6)
    pairs = (
        [(f"tables.{k}", ctx.tables[k], ref.tables[k]) for k in ctx.tables]
        + [
            (f"halo.{k}", ctx.halo_tables[k], ref.halo_tables[k])
            for k in ctx.halo_tables
        ]
        + [
            ("frag_table", ctx.frag_table, ref.frag_table),
            ("cu_seqlens_local", ctx.cu_seqlens_local, ref.cu_seqlens_local),
        ]
        + [
            (f"prepared.{k}", prepared.device[k], ref_prepared.device[k])
            for k in prepared.device
        ]
    )
    assert len(pairs) > 20
    for name, dev_t, cpu_t in pairs:
        assert dev_t.device.type == "cuda", name
        assert dev_t.dtype == cpu_t.dtype and tuple(dev_t.shape) == tuple(
            cpu_t.shape
        ), name
        assert dev_t.data_ptr() % plan.UPLOAD_ALIGN_BYTES == 0, name
        assert torch.equal(dev_t.cpu(), cpu_t), name
    assert prepared.P == ref_prepared.P


# --------------------------------------------------------------------------- issue #7074 (4): the workspace bytes the backward binds (one GPU)


class _PeakWorkspace:
    """A ``CakeGDNDetKcpWorkspace`` recording the most bytes it binds (every allocation / release path)."""

    def __new__(cls, kcp):
        class Tracked(kcp.CakeGDNDetKcpWorkspace):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.peak = 0

            def _upd(self):
                self.peak = max(self.peak, self.bytes)

            def tape(self, name):
                t = super().tape(name)
                self._upd()
                return t

            def pool(self, name, n=None):
                t = super().pool(name, n)
                self._upd()
                return t

            def ensure(self, *a, **k):
                tb = super().ensure(*a, **k)
                self._upd()
                return tb

            def adopt_tape(self, name, t):
                r = super().adopt_tape(name, t)
                self._upd()
                return r

        return Tracked()


@pytest.mark.parametrize(
    ("cu", "heads"),
    [(ISSUE_CU, (2, 6)), ([0, 11000, 23000, 32768], (16, 48))],
    ids=["issue-2x6", "32k-16x48"],
)
def test_workspace_bytes_match_what_the_backward_binds(device, cu, heads) -> None:
    """``cake_gdn_det_train_kcp_workspace_layout`` is what a caller-held workspace binds (tolerance 0): the forward's peak (the
    fused forward's #1; with test knob ``fused_o=0`` the split forward's tape walk, ``forward_split``), and the
    backward's phase maximum of each tape form -- the kept-tape form (the ten light tapes are the recompute's hand-off
    tensors, the tapes the forward leaves re-adopted after #8) and the replay form (the ten light tapes are
    workspace)."""
    import gc

    from torch.utils.checkpoint import checkpoint

    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    gb = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_bytes"
    )
    H, HV = heads
    inputs = make_inputs(cu, H, HV)
    ctx = _local_context(cu, device)
    ctx.prepare(H, HV)
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    dout = inputs["dout"].to(device)
    for form in (True, False):
        layout = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H, HV=HV, reuse_recompute_tapes=form
        )
        ws = _PeakWorkspace(kcp)

        def fn(*a, _ws=ws, _form=form):
            return gdn_api.cake_gdn_det_kcp(
                *a, ctx, workspace=_ws, reuse_recompute_tapes=_form
            )

        out = checkpoint(fn, *leaves, use_reentrant=False)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["forward"], (form, ws.peak, layout["forward"])
        ws.peak = 0
        out.backward(dout)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["backward"], (form, ws.peak, layout["backward"])
        assert ws.bytes == layout["bwd.phase.end"], (
            form,
            ws.bytes,
            layout["bwd.phase.end"],
        )
        assert gdn_api.cake_gdn_det_train_kcp_workspace_bytes(
            ctx, H_qk=H, HV=HV, reuse_recompute_tapes=form
        ) == max(layout["forward"], layout["backward"])
        for lf in leaves:
            lf.grad = None
        del out, ws
        gc.collect()
        torch.cuda.empty_cache()
        # the split forward (fused_o = 0) binds the h / v_new tapes in its tape walk: forward_split
        ws = _PeakWorkspace(kcp)

        def split(*a, _ws=ws, _form=form):
            return gdn_api.cake_gdn_det_kcp(
                *a,
                ctx,
                workspace=_ws,
                reuse_recompute_tapes=_form,
                test_knobs={"fused_o": 0},
            )

        out = checkpoint(split, *leaves, use_reentrant=False)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["forward_split"], (
            form,
            ws.peak,
            layout["forward_split"],
        )
        out.backward(dout)
        torch.cuda.synchronize(device)
        for lf in leaves:
            lf.grad = None
        del out, ws
        gc.collect()
        torch.cuda.empty_cache()
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- issue #7074 (2): reuse_recompute_tapes="auto" (one GPU)


@pytest.mark.parametrize(
    "later_saver", [False, True], ids=["gdn-closes-the-region", "later-saver"]
)
def test_reuse_recompute_tapes_auto_follows_the_region_shape(
    device, later_saver
) -> None:
    """``"auto"`` hands the recompute's tapes to the backward when the GDN node is the region's last saver (frame
    proof ``dead``, kept form) and takes the replay form when a later operation of the region saves a tensor
    (``holders_remain``); ``True`` always hands off (``always``), ``False`` never (``off``).  Every form is bitwise the
    direct autograd result, the registry is empty afterwards, and the ``"auto"`` backward -- the recompute forward,
    the frame proof over the autograd graph, the take -- makes no synchronizing CUDA call."""
    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    H, HV = 2, 6
    cu = [0, 1500, 4200, 6144]
    inputs = make_inputs(cu, H, HV)
    ctx = _local_context(cu, device)
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    proj = None
    if later_saver:
        gen = torch.Generator(device="cpu").manual_seed(7074)
        proj = (
            (torch.randn(HV * D, HV * D, generator=gen) * 0.02)
            .to(torch.bfloat16)
            .to(device)
        )
        dout = (
            inputs["dout"].to(device).reshape(1, -1, HV * D)
            if proj is not None
            else None
        )
        proj_in = proj
        proj = None  # the projection acts on the flattened heads: wrap below

        def flat_proj(out, w=proj_in):
            return torch.nn.functional.linear(
                out.reshape(out.shape[0], out.shape[1], -1), w
            )

        proj = flat_proj
    else:
        dout = inputs["dout"].to(device)

    def region(ckpt, form, sync_error=False):
        from torch.utils.checkpoint import checkpoint

        def fn(*a):
            out = gdn_api.cake_gdn_det_kcp(
                *a, ctx, **({} if form is None else {"reuse_recompute_tapes": form})
            )
            return out if proj is None else proj(out)

        for lf in leaves:
            lf.grad = None
        out = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
        torch.cuda.synchronize(device)
        if sync_error:
            torch.cuda.set_sync_debug_mode("error")
        try:
            out.backward(dout)
        finally:
            if sync_error:
                torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize(device)
        return {
            "out": out.detach(),
            **{
                n: lf.grad.detach().clone()
                for n, lf in zip(OPERANDS, leaves, strict=True)
            },
        }

    direct = region(False, None)
    verified = kcp.torch_release_verified()
    for form, reason, source in (
        (True, "always", "kept"),
        (False, "off", "replay"),
        (
            "auto",
            ("holders_remain" if later_saver else "dead") if verified else None,
            "replay" if (later_saver or not verified) else "kept",
        ),
    ):
        res = region(True, form, sync_error=(form == "auto"))
        for n in direct:
            assert torch.equal(res[n], direct[n]), (later_saver, form, n)
        got_reason = kcp.LAST_CALL["fwd"]["hand_off_reason"]
        if reason is not None:
            assert got_reason == reason, (later_saver, form, got_reason)
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == source, (
            later_saver,
            form,
            kcp.LAST_CALL["bwd"]["tapes_source"],
        )
        assert kcp.kept_registry_size() == 0, (later_saver, form)


# --------------------------------------------------------------------------- the output walk kcp_fwdh_out (one GPU)
# A light, saved-boundary forward that launches #5 and hands no tape to a backward runs the output walk in place of the
# tape walk and #5 (test knob fused_o = 1, the shipped form); fused_o = 0 forces the split form.  Each form below runs
# both on the same build: ``out`` and every gradient are bitwise the split form's (and two split runs are bitwise
# equal, the package's contract).  The launch logs show which form each forward ran.

FUSED_SHAPES = (
    ("2x6", (1500, 2700, 1944), (2, 6)),
    ("16x48", (2048 + 37, 1000), (16, 48)),
)
FUSED_FORMS = (
    # id, layers, checkpoint, later saver, options (``explicit`` / ``handle`` select the entry form), test knobs
    ("direct", 1, False, False, {}, {}),
    ("explicit", 1, False, False, {"explicit": True}, {}),
    ("ckpt-closes", 1, True, False, {}, {}),
    ("ckpt-later-saver", 1, True, True, {}, {}),
    ("ckpt-replay-form", 1, True, False, {"reuse_recompute_tapes": False}, {}),
    ("ckpt-3-layers-later-saver", 3, True, True, {}, {}),
    ("ckpt-handle", 1, True, False, {"handle": True}, {}),
    ("segments-2", 1, True, True, {"segments": 2}, {}),
    ("vh64", 1, True, True, {}, {"fwdh_vh": 64}),
    ("vh128", 1, True, True, {}, {"fwdh_vh": 128}),
    ("l2n", 1, True, True, {"use_qk_l2norm_in_kernel": True}, {}),
    ("graph", 1, False, False, {"graph": "replay"}, {}),
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(torch.uint8)


def _fused_arm(device, cu, heads, form, fused_o: int, log: list) -> dict:
    """One run of ``form`` with test knob ``fused_o``: ``out``, the five gradients (and ``dw`` of the later saver);
    ``log`` receives ``(kind, sequence, fwd_o decision)`` of every recorded call."""
    from torch.utils.checkpoint import checkpoint

    _fid, layers, ckpt, later, opts, knobs = form
    opts = dict(opts)
    explicit, handle = opts.pop("explicit", False), opts.pop("handle", False)
    graph = opts.get("graph") == "replay"
    H, HV = heads
    inputs = make_inputs(cu, H, HV)
    part = gdn_api.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, layout="contiguous"
    )
    ctx = gdn_api.cake_gdn_det_train_kcp_build_context(part, None, 0, device)
    kw = {"test_knobs": {"fused_o": fused_o, **knobs}, **opts}
    if graph:
        kw["workspace"] = gdn_api.CakeGDNDetKcpWorkspace()
    dout = inputs["dout"].to(device)
    del log[:]
    if explicit:
        q, k, v, g, beta = (inputs[n].to(device) for n in OPERANDS)
        out, saved = gdn_api.cake_gdn_det_train_kcp_forward(
            q, k, v, g, beta, ctx, save_for_backward=True, **kw
        )
        dq, dk, dv, dbeta, dg = gdn_api.cake_gdn_det_train_kcp_backward(
            q, k, v, g, beta, ctx, dout, saved=saved, **kw
        )
        torch.cuda.synchronize(device)
        return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}
    w = None
    if later:
        gen = torch.Generator(device="cpu").manual_seed(6098)
        w = (torch.rand(HV, D, generator=gen) + 0.5).to(torch.bfloat16).to(device)
        w.requires_grad_(True)

    def fn(q, k, v, g, beta, *handles):
        x = v
        for i in range(layers):
            hk = {"recompute_handle": handles[i]} if handles else {}
            x = gdn_api.cake_gdn_det_kcp(q, k, x, g, beta, ctx, **hk, **kw)
        # a later saver of the region: w's gradient reads the output
        return x if w is None else x * w

    # graph: capture, then replays (a replay records no call)
    for _rep in range(3 if graph else 1):
        leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
        handles = (
            [gdn_api.CakeGDNDetKcpRecomputeHandle() for _ in range(layers)]
            if handle
            else []
        )
        out = (
            checkpoint(fn, *leaves, *handles, use_reentrant=False)
            if ckpt
            else fn(*leaves, *handles)
        )
        grads = torch.autograd.grad(out, leaves + ([w] if w is not None else []), dout)
        torch.cuda.synchronize(device)
    res = {"out": out.detach()}
    res.update({n: gr for n, gr in zip(GRAD_NAMES, grads, strict=False)})
    if w is not None:
        res["dw"] = grads[-1]
    return res


@pytest.mark.parametrize("form", FUSED_FORMS, ids=[f[0] for f in FUSED_FORMS])
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [s[1:] for s in FUSED_SHAPES],
    ids=[s[0] for s in FUSED_SHAPES],
)
def test_fused_output_walk_is_bitwise_the_split_forward(
    device, monkeypatch, lengths, heads, form
) -> None:
    """The same-build matrix of the output walk: direct, explicit, checkpointed (the region closed by the node, a later
    saver, the replay form, three layers with a later saver, a recompute handle), two segments, both value widths,
    the in-kernel L2 norm and the graph replay, each with test knob ``fused_o`` 1 and 0 (split twice): every tensor
    bitwise equal."""
    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_backend"
    )
    if not kcp.fused_o_shipped(backend.device_arch(device.index)):
        pytest.skip("the registry ships no output walk for this architecture")
    if form[4].get("use_qk_l2norm_in_kernel") and not kcp.l2n_records_shipped():
        pytest.skip("the registry ships no in-kernel L2 norm records")
    cu = _cu(lengths)
    log: list = []
    record = kcp._record

    def recording(kind, call):
        record(kind, call)
        entry = kcp.LAST_CALL[kind]
        log.append((kind, tuple(entry.get("sequence") or ()), entry.get("fwd_o")))

    monkeypatch.setattr(kcp, "_record", recording)
    split = [_fused_arm(device, cu, heads, form, 0, log) for _ in range(2)]
    split_log = list(log)
    fused = _fused_arm(device, cu, heads, form, 1, log)
    fused_log = list(log)
    # the launch logs: the same calls; the fused arm's forwards that write the output run the output walk (the
    # first forward always: it hands no tape over), the split arm never does
    assert [k for k, _s, _o in fused_log] == [k for k, _s, _o in split_log]
    fwd = [(s, o) for k, s, o in fused_log if k == "fwd"]
    assert fwd and fwd[0][1] == "fused", fwd[:1]
    for seq, decision in fwd:
        assert ("kcp_fwdh_out" in seq) == (decision == "fused"), (decision, seq)
        if decision == "fused":
            assert "kcp_fwdh_tape" not in seq and "fwd_o" not in seq, seq
    assert not any("kcp_fwdh_out" in s or o == "fused" for _k, s, o in split_log)
    assert [s for k, s, _o in fused_log if k == "bwd"] == [
        s for k, s, _o in split_log if k == "bwd"
    ]  # every backward is the split form's
    assert set(fused) == set(split[0])
    for n in fused:
        assert torch.equal(_bits(split[0][n]), _bits(split[1][n])), (form[0], n)
        assert torch.equal(_bits(fused[n]), _bits(split[0][n])), (form[0], n)
        assert bool(torch.isfinite(fused[n].float()).all()), n
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- save_mode="full" without save_transfer at P > 1


@pytest.mark.parametrize("norm", [False, True], ids=["l2n-off", "l2n-on"])
def test_full_backward_without_save_transfer_matches_light_on_poisoned_memory(
    device, monkeypatch, norm
) -> None:
    """``save_mode="full"`` with ``save_transfer=False`` (the only form of ``exchange="ring"``) at ``P = 2``: every
    tensor the package allocates is filled with NaN bytes, so a read of a plane no program of the call wrote (the
    ``M_seg^T`` staging plane the backward's merges read) shows up as a non-finite gradient.  The saved set carries
    ``M_seg``; the full form gives the light form's bits (out and the five gradients) on a per-call, a fresh and a
    shared workspace (twice on the shared one) and under a non-reentrant checkpoint."""
    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    real_alloc = kcp._alloc

    def nan_alloc(shape, dtype, dev):
        t = real_alloc(shape, dtype, dev)
        if t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    monkeypatch.setattr(kcp, "_alloc", nan_alloc)
    cu = [0, 11000, 23000, 32768]
    H, HV = 16, 48
    inputs = make_inputs(cu, H, HV)
    ctx = _local_context(cu, device)
    dout = inputs["dout"].to(device)
    o = {"segments": 2, "save_transfer": False, "use_qk_l2norm_in_kernel": norm}

    def run(workspace, *, ckpt: bool = False, **kw) -> dict:
        leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]

        def fn(*a):
            return gdn_api.cake_gdn_det_kcp(*a, ctx, workspace=workspace, **o, **kw)

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, use_reentrant=False)
        else:
            out = fn(*leaves)
        out.backward(dout)
        torch.cuda.synchronize(device)
        return {
            "out": out.detach(),
            **{n: lf.grad for n, lf in zip(OPERANDS, leaves, strict=True)},
        }

    ops = [inputs[n].to(device) for n in OPERANDS]
    _, saved = gdn_api.cake_gdn_det_train_kcp_forward(
        *ops, ctx, save_for_backward=True, save_mode="full", **o
    )
    assert saved.P == 2 and saved.c1_recv is None and saved.M_seg is not None
    del saved, ops
    light = run(None)
    assert kcp.LAST_CALL["fwd"]["P"] == 2
    for n, t in light.items():
        assert bool(torch.isfinite(t.float()).all()), ("light", n)
    shared = kcp.CakeGDNDetKcpWorkspace()
    arms = (
        (None, False),
        (kcp.CakeGDNDetKcpWorkspace(), False),
        (shared, False),
        (shared, False),
        (None, True),
    )
    for i, (ws, ckpt) in enumerate(arms):
        full = run(ws, ckpt=ckpt, save_mode="full")
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == "saved", i
        for n in light:
            assert torch.equal(full[n], light[n]), (i, n)


# --------------------------------------------------------------------------- eager pool / ticket sizing after launch #1 (one GPU)


@pytest.mark.parametrize(
    "knobs",
    [{}, {"save_mode": "full"}, {"use_qk_l2norm_in_kernel": True}, {"segments": 2}],
    ids=("light", "full", "l2n", "P2"),
)
@pytest.mark.parametrize("ckpt", (False, True), ids=("direct", "checkpoint"))
def test_eager_pools_and_tickets_are_sized_after_the_first_launch(
    device, monkeypatch, ckpt, knobs
) -> None:
    """The kernels backend, one rank, an autograd forward + backward directly and under a non-reentrant checkpoint:
    at the first launch of every eager call the call's (fresh) workspace holds no pool (and no ticket slot unless the
    first program is persistent: #6 of a backward on saved or kept tapes) and its sizing is pending; out and the five gradients, the launch records (program, grid) and sequences of the last
    forward and backward, their workspace bytes and the CUDA peak above the inputs are those of the package with
    the sizing done in ``ensure`` (``defer=False``), the peak within the allocator's block rounding; every run
    repeats bitwise."""
    from torch.utils.checkpoint import checkpoint

    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    cu = _cu((2048 + 37, 1000, 129))
    inputs = make_inputs(cu, 16, 48)
    ctx = _local_context(cu, device)
    ctx.prepare(16, 48, P=knobs.get("segments"))
    dout = inputs["dout"].to(device)
    leaves0 = [inputs[n].to(device) for n in OPERANDS]
    first: list = []
    real_run = kcp.KernelBackend.run

    def run(self, name, args):
        call = args["_call"]
        if all(c is not call for c, *_ in first):
            ws = call.ws
            first.append(
                (
                    call,
                    name,
                    sorted(ws._pools),
                    ws._tickets is not None,
                    ws._pending is not None,
                )
            )
        return real_run(self, name, args)

    monkeypatch.setattr(kcp.KernelBackend, "run", run)
    real_ensure = kcp.CakeGDNDetKcpWorkspace.ensure

    def step(defer: bool) -> tuple[dict, dict, int]:
        def ensure(self, *a, defer=False, **k):  # noqa: ARG001 - the undeferred sizing
            return real_ensure(self, *a, defer=False, **k)

        if not defer:
            monkeypatch.setattr(kcp.CakeGDNDetKcpWorkspace, "ensure", ensure)
        try:
            leaves = [x.detach().clone().requires_grad_(True) for x in leaves0]

            def fn(*xs):
                return gdn_api.cake_gdn_det_kcp(*xs, ctx, **knobs)

            torch.cuda.synchronize(device)
            base = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            out = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
            grads = torch.autograd.grad(out, leaves, dout)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) - base
        finally:
            monkeypatch.setattr(kcp.CakeGDNDetKcpWorkspace, "ensure", real_ensure)
        res = {"out": out.detach()}
        res.update({n: gr.detach() for n, gr in zip(GRAD_NAMES, grads, strict=True)})
        log = {
            kind: {
                key: kcp.LAST_CALL[kind][key]
                for key in (
                    "records",
                    "sequence",
                    "workspace_bytes",
                    "pool_bytes",
                    "tape_bytes",
                )
            }
            for kind in ("fwd", "bwd")
        }
        return res, log, peak

    ref, ref_log, ref_peak = step(False)
    n_ref = len(first)
    assert n_ref >= 2 and all(
        tk and not pend and pools for _, _, pools, tk, pend in first
    )
    got, got_log, got_peak = step(True)
    deferred = first[n_ref:]
    assert len(deferred) == n_ref
    for call, name, pools, tk, pend in deferred:
        assert call.T_r > 0 and pend and pools == [], (call.backward, name, pools)
        # the ticket slots exist at the first launch only when it is a persistent program (#6 on saved / kept tapes)
        assert tk == (name in kcp.TICKET_SITES), (call.backward, name, tk)
        assert call.ws._pending is None
    for n in TENSORS:
        assert torch.equal(got[n], ref[n]), n
    assert got_log == ref_log
    # the same tensors are live at every moment; the caching allocator may round a reused block differently when
    # the allocation order changes (a large block keeps an unsplit remainder below 1 MiB)
    assert abs(got_peak - ref_peak) <= 4 << 20, (got_peak, ref_peak)
    again, again_log, _ = step(True)
    for n in TENSORS:
        assert torch.equal(again[n], got[n]), n
    assert again_log == got_log


# --------------------------------------------------------------------------- the forward form of #1_l2n (one GPU)
# With the in-kernel L2 norm a forward whose akk / g_cum have no reader (not a replay-mode or hand-off forward, no saved
# set that keeps them) launches the record fwd_prep_l2n_fwd (no akk / g_cum stores; test knob prep_form = 1, the shipped
# form); prep_form = 0 runs the full fwd_prep_l2n everywhere.  Every form of the output-walk matrix with the norm on:
# out and every gradient bitwise the full form's, two full runs bitwise equal, and every backward replays the full #1.

PREP_FORMS = tuple(
    (
        f"l2n-{fid}",
        layers,
        ckpt,
        later,
        {**opts, "use_qk_l2norm_in_kernel": True},
        knobs,
    )
    for fid, layers, ckpt, later, opts, knobs in FUSED_FORMS
)


@pytest.mark.parametrize("form", PREP_FORMS, ids=[f[0] for f in PREP_FORMS])
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [s[1:] for s in FUSED_SHAPES],
    ids=[s[0] for s in FUSED_SHAPES],
)
def test_prep_forward_form_is_bitwise_the_full_prep(
    device, monkeypatch, lengths, heads, form
) -> None:
    """The forward form of #1_l2n (record ``fwd_prep_l2n_fwd``) against the full ``fwd_prep_l2n`` (test knob
    ``prep_form`` 1 / 0) on every form of the output-walk matrix with the norm on: every tensor bitwise equal; the
    first forward of every arm with ``prep_form`` 1 runs the forward form, no backward and no ``prep_form`` 0 call does."""
    kcp = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp"
    )
    backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_backend"
    )
    if not kcp.l2n_records_shipped():
        pytest.skip("the registry ships no in-kernel L2 norm records")
    if not kcp.prep_form_shipped(backend.device_arch(device.index)):
        pytest.skip(
            "the registry ships no forward form of fwd_prep_l2n for this architecture"
        )
    cu = _cu(lengths)
    log: list = []
    record = kcp._record

    def recording(kind, call):
        record(kind, call)
        entry = kcp.LAST_CALL[kind]
        preps = tuple(
            r for r, _g in (entry.get("records") or ()) if str(r).startswith("fwd_prep")
        )
        log.append((kind, preps))

    monkeypatch.setattr(kcp, "_record", recording)

    def arm(prep_form: int) -> dict:
        fid, layers, ckpt, later, opts, knobs = form
        return _fused_arm(
            device,
            cu,
            heads,
            (fid, layers, ckpt, later, opts, {**knobs, "prep_form": prep_form}),
            1,
            log,
        )

    full = [arm(0) for _ in range(2)]
    full_log = list(log)
    fwd = arm(1)
    fwd_log = list(log)
    assert [k for k, _p in fwd_log] == [k for k, _p in full_log]
    assert not any("fwd_prep_l2n_fwd" in p for _k, p in full_log), full_log
    if fwd_log:  # a graph replay records no call
        firsts = [p for k, p in fwd_log if k == "fwd"]
        assert firsts and firsts[0] == ("fwd_prep_l2n_fwd",), fwd_log
        assert all("fwd_prep_l2n_fwd" not in p for k, p in fwd_log if k == "bwd"), (
            fwd_log
        )
    assert set(fwd) == set(full[0])
    for n in fwd:
        assert torch.equal(_bits(full[0][n]), _bits(full[1][n])), (form[0], n)
        assert torch.equal(_bits(fwd[n]), _bits(full[0][n])), (form[0], n)
        assert bool(torch.isfinite(fwd[n].float()).all()), n
    assert kcp.kept_registry_size() == 0
