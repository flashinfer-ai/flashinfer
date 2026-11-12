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

Backend suite of the Cake deterministic GDN chunked-training package
(``flashinfer.experimental.cake.cake_gdn_det_train``).

Without a device: the package's names (the det names only, the former names removed) and its mode contract (no
``deterministic`` keyword; the low-level backward refuses a tape without the det tag), the registry (one record per stage and architecture, the argument plans of the call sites, the
scan constants, the ``dhu_tc2`` selection constants and the ``prep_tma`` window), the build manifest (every record's
library, the recorded source digests), the launch entries (one resolution per stage and device, then the cached FFI
entry; the first launch's errors), the plan pins, the finalize rule, the ``dhu_tc2`` selection rule, the split pass-2 rule on the campaign rows and
the #7074 packs, the serial-scan rule on those rows (and its cached decision in the chunk tables) and the two-stream
groups on the campaign rows, the ``prep_tma`` dispatch window and the host chunk / segment tables.  On SM100 / SM103 / SM107: the six outputs ``o, dq, dk, dv, dg, dbeta`` against an FP64 token recurrence on
packed layouts (host and device table routes, FP32 and BF16 ``beta``, every register-scan configuration, the serial
and segment-parallel scans, both adjoint scans, both preps, the fused tails of a value-head group of two and of three
and the unfused tail with both finalize stages, the recompute and uncached forms, the un-normalized ``q`` / ``k`` form), the
initial / final state with its gradients (also through empty sequences), the pre-scan capacity, the bitwise
contracts: two runs, PyTorch's deterministic mode against the default, the two table routes on one plan (a split
plan's tcgen05 pass 2 with ``DVL32_ARCHS`` off; on ``dhu_tc2_dvl32`` ``o`` bitwise and the gradients within the
budgets), a split plan's pass 2 on ``dhu_tc2`` after the fused pre-scan
against ``dv_local`` and the register one, the fused tail of a value-head group of three against the unfused one with
either finalize, the longest-first dispatch order against the caller's (and its
tables without a device), two streams against one (the adjoint scan of serial plans beside the fused tail of a
value-head group of two and the ``dhu_tc2`` pass 2 of split plans beside that of three, the other two pairings on one
stream; and the groups' rule without a device, caller-stream allocations between and during two-stream calls, a CUDA
graph capture and one side stream per caller stream), the launches on the
caller's stream (one and two streams) and from cached entries, the two preps, ``q`` / ``k`` off the tensor maps' 16-byte
alignment (the plain prep), and an equal-length batch against the same sequences packed; and the input contract: every
rejected input raises before the first launch, a host tensor at the low-level entries reaches the binding's device
check, and the device route does not synchronize.  The wavefront pair (``dhu_tc2_wavefront`` / ``tail_g2_bf16_wavefront``,
the split pass 2 of more CTAs than SMs on ``dhu_tc2_wavefront_split``): its chunk order and its rule per architecture
(and the scan it picks) without a device; on a device the pair against the separate launches with the adjoint
workspaces poisoned (serial, with an empty sequence, a split pass 2, and one of more CTAs than SMs on
``dhu_tc2_wavefront_split``), repeated calls, the shipped rule on
the device's architecture (serial plans, and split pass 2s below and above the split side's work floor), the rows the
rule leaves alone launching exactly what the rule turned off launches, a CUDA graph capture (the separate launches),
accumulated gradients, strided views and a checkpointed call.  The serial-scan rule on its campaign rows: one adjoint
scan over whole sequences, the forward's launches and ``o`` / ``dq`` bitwise those of the split plan, two runs bitwise.  The
training-loop contracts: the head pairs (16, 48), (8, 24),
(4, 12), (16, 32) and (8, 16) on their fused tails (the groups of three also with the split pass 2 on two streams
and the serial adjoint scan on one),
gradients accumulated into a nonzero ``.grad`` (one and two streams), strided views of packed and head-major operands
(and a strided output gradient) against the contiguous call, a non-reentrant ``torch.utils.checkpoint`` that retains no
bytes (one and two streams), all bitwise, and the forward and backward beside a resident side-stream kernel (a time
ratio declared before the run; also the split pass 2 on the device's own SM count: two streams beside ``tail_g3``,
one beside ``tail_g2``).  The forwards on the WY stage's A (FP32 ``beta``, normalized q / k, a value-head group of at
most three, no plan pin, a host-built table): without a device their route rules (the pins and the three switches
the non-deterministic package's tests turn off by name), the split forward's lists and shape class at 148 / 152 SMs,
the serial forward table kept on R200 at 208 / 212 / 216 SMs and the switches in the chunk tables' key; on a device
the routes' FP64 accuracy (the budget above), run-to-run bits (also uncached, and one stream against two), the
caller's order, PyTorch's deterministic mode, a CUDA graph capture, the tape a call that records no backward does not
write, the saved list's arity, a checkpoint that writes the tape once (non-reentrant and reentrant, with the initial
and final state), the low-level entries without a tape, accumulated gradients, the cached split tables across streams
and captures, the resident side-stream kernel, and the calls they do not serve launching what the switched-off
package launches.  The round-once tail of a value-head group of three (``tail_g3_f32x`` / ``tail_g3_f32x_bf16``: the
group's FP32 dq / dk partials summed before one rounding) and the FP32 ``dv_local`` adjoint (``dhu_tc2_dvl32``):
without a device their records (token-row strides of v / dq / dk / dv), their rule and its two switches
(``F32X_TAIL_ARCHS`` / ``DVL32_ARCHS``, off ``()``, which the non-deterministic package's tests turn off by name), the
token-row stride of a head view, the rule that dq and dk share one stride, and the packed row's layout; on a device the
route on every architecture of the switches (G = 3 only; G = 2 keeps tail_g2 on dhu_tc2), the strides it launches with,
each switch alone and both off (tail_g3 on dhu_tc2: the unfused tail's bits again), run-to-run bits (host and device
routes, FP32 and BF16 beta, contiguous operands and the packed row), the unfused tail and the register pass 2 within
the budgets with every output but the ones the change rounds differently the same bits, the low-level backward
writing into the caller's dq / dk / dv (and refusing dq / dk at two strides before the tail), and the packed row: one
gradient row the tail writes (CakeGDNDetPackedFunction), the bits of the head views and of the contiguous call, a
non-leaf row with another consumer, a row without a gradient, and a checkpoint (non-reentrant and reentrant) with the
eager call's bits.

The suite names its package once, through the shared alias helper, and imports nothing else from flashinfer: a build
that vendors the package under another import path runs it unchanged after pointing the ``PACKAGES`` entry at its
own package.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import re
import time
import types
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_mode_names import assert_mode_names
from tests.cake.test_helpers.cake_package import PACKAGES, cake_package

cake_backend = cake_package("cake_gdn_det_train", "cake_backend")
cake_chain = cake_package("cake_gdn_det_train", "cake_chain")
cake_jit = cake_package("cake_gdn_det_train", "cake_jit")
cake_plan = cake_package("cake_gdn_det_train", "cake_plan")
cake_gdn_det_train = cake_package("cake_gdn_det_train")
cake_gdn_det = cake_gdn_det_train.cake_gdn_det

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
ARCHES = tuple(cake_jit.ARCHES)
HEAD_DIM = 128
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
OUTPUT_NAMES = ("o",) + GRAD_NAMES

# Accuracy gate against the FP64 recurrence of the same BF16 / FP32 inputs, declared before the first run: per output
# tensor, the relative RMS error at most REL_RMS_MAX and the largest absolute error at most MAX_ABS_REL times the
# largest reference magnitude.  The chain stores o / dq / dk / dv in BF16 (a 2^-8 relative rounding step); its
# campaign rows measured at most 8.3e-3 of the largest reference magnitude on every output.
REL_RMS_MAX = 1e-2
MAX_ABS_REL = 2e-2
L2NORM_EPS = 1e-6  # the prep stage's q / k normalization: x / sqrt(sum x^2 + 1e-6)


# --------------------------------------------------------------------------- helpers


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {ARCHES}"
        )
    return torch.device("cuda", index)


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _cu(seq_lens) -> list[int]:
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + int(n))
    return cu


def _inputs(
    seq_lens, heads, beta_dtype, device, *, seed=0, batch=1, normalized_qk=False
):
    """Packed ``[1, T, ...]`` operands (or ``[batch, T, ...]`` for an equal-length batch) drawn on the host."""
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, total)
    q = torch.randn(*shape, num_heads, HEAD_DIM, generator=gen)
    k = torch.randn(*shape, num_heads, HEAD_DIM, generator=gen)
    if normalized_qk:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(*shape, num_v_heads, HEAD_DIM, generator=gen)
    g = torch.rand(*shape, num_v_heads, generator=gen) * -0.5
    beta = torch.rand(*shape, num_v_heads, generator=gen)
    do = torch.randn(*shape, num_v_heads, HEAD_DIM, generator=gen)
    out = {
        "q": q.to(torch.bfloat16),
        "k": k.to(torch.bfloat16),
        "v": v.to(torch.bfloat16),
        "g": g.float(),
        "beta": beta.to(beta_dtype),
        "do": do.to(torch.bfloat16),
    }
    return {name: t.to(device).contiguous() for name, t in out.items()}


def _run(inputs, seq_lens, *, route="host", initial_state=None, output_final_state=False, dht=None,
         use_qk_l2norm_in_kernel=True, leaves=None, **pins):  # fmt: skip
    """Forward and backward through the package; returns the outputs and gradients (and the final state / its
    initial-state gradient when asked).  ``leaves`` replaces the fresh leaf copies of the inputs it names."""
    leaves = {
        **{
            name: inputs[name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        },
        **(leaves or {}),
    }
    h0 = None
    if initial_state is not None:
        h0 = initial_state.detach().clone().requires_grad_(True)
    kwargs = {}
    if seq_lens is not None:
        cu = _cu(seq_lens)
        kwargs["cu_seqlens"] = torch.tensor(
            cu, dtype=torch.int64, device=inputs["q"].device
        )
        kwargs["cu_seqlens_cpu"] = (
            torch.tensor(cu, dtype=torch.int64) if route == "host" else None
        )
    o, final = cake_gdn_det(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=1.0 / math.sqrt(HEAD_DIM),
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **kwargs,
        **pins,
    )
    loss = (o.float() * inputs["do"].float()).sum()
    if output_final_state and dht is not None:
        loss = loss + (final * dht).sum()
    loss.backward()
    torch.cuda.synchronize()
    res = {
        "o": o.detach(),
        "dq": leaves["q"].grad,
        "dk": leaves["k"].grad,
        "dv": leaves["v"].grad,
        "dg": leaves["g"].grad,
        "dbeta": leaves["beta"].grad,
    }
    if output_final_state:
        res["final_state"] = final.detach()
    if h0 is not None:
        res["dh0"] = h0.grad
    return res


def _reference(inputs, seq_lens, *, initial_state=None, dht=None, normalize=True):
    """The FP64 token recurrence (FLA's gated delta rule) and its autograd gradients, per sequence and value head
    ``hv`` (key head ``hv // (HV / H)``) with the state ``S`` ``[K, V]``::

        S = S * exp(g_t);  u_t = beta_t (v_t - S^T k_t);  S = S + k_t u_t^T;  o_t = scale S^T q_t
    """
    dt = torch.float64
    leaves = {
        name: inputs[name].detach().to(dt).requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    h0 = None
    if initial_state is not None:
        h0 = initial_state.detach().to(dt).requires_grad_(True)
    q, k, v, g, beta = (leaves[n].flatten(0, 1) for n in ("q", "k", "v", "g", "beta"))
    num_heads, num_v_heads = q.shape[1], v.shape[1]
    group = num_v_heads // num_heads
    if normalize:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2NORM_EPS)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2NORM_EPS)
    q = q.repeat_interleave(group, dim=1) * (1.0 / math.sqrt(HEAD_DIM))
    k = k.repeat_interleave(group, dim=1)
    outs, finals = [], []
    start = 0
    for i, n in enumerate(seq_lens):
        S = (
            h0[i]
            if h0 is not None
            else torch.zeros(num_v_heads, HEAD_DIM, HEAD_DIM, dtype=dt, device=q.device)
        )
        for t in range(start, start + int(n)):
            S = S * torch.exp(g[t])[:, None, None]
            u = beta[t][:, None] * (v[t] - torch.einsum("hk,hkv->hv", k[t], S))
            S = S + k[t][:, :, None] * u[:, None, :]
            outs.append(torch.einsum("hk,hkv->hv", q[t], S))
        finals.append(S)
        start += int(n)
    o = torch.stack(outs, 0).reshape(inputs["v"].shape)
    final = torch.stack(finals, 0)
    loss = (o * inputs["do"].to(dt)).sum()
    if dht is not None:
        loss = loss + (final * dht.to(dt)).sum()
    loss.backward()
    res = {
        "o": o.detach(),
        "dq": leaves["q"].grad,
        "dk": leaves["k"].grad,
        "dv": leaves["v"].grad,
        "dg": leaves["g"].grad,
        "dbeta": leaves["beta"].grad,
        "final_state": final.detach(),
    }
    if h0 is not None:
        res["dh0"] = h0.grad
    return res


def _metrics(a: torch.Tensor, ref: torch.Tensor) -> dict:
    a64 = a.detach().to(torch.float64)
    r64 = ref.detach().to(torch.float64)
    diff = (a64 - r64).abs()
    max_ref = float(r64.abs().max())
    rms_ref = float(r64.square().mean().sqrt())
    return {
        "max_abs": float(diff.max()),
        "max_ref": max_ref,
        "rel_rms": float(diff.square().mean().sqrt()) / rms_ref if rms_ref > 0 else 0.0,
        "finite": bool(torch.isfinite(a64).all()),
    }


def _assert_close_to_reference(got: dict, ref: dict, names) -> None:
    report = {}
    failed = []
    for name in names:
        assert got[name].shape == ref[name].shape, (
            name,
            got[name].shape,
            ref[name].shape,
        )
        m = _metrics(got[name], ref[name])
        report[name] = m
        if not (
            m["finite"]
            and m["rel_rms"] <= REL_RMS_MAX
            and m["max_abs"] <= MAX_ABS_REL * m["max_ref"]
        ):
            failed.append(name)
    assert not failed, f"outside the accuracy gate: {failed}; metrics {report}"


def _assert_bitwise(a: dict, b: dict, names) -> None:
    differing = [n for n in names if not torch.equal(a[n], b[n])]
    assert not differing, f"not bitwise equal: {differing}"


def _record_launches(monkeypatch) -> list[str]:
    """The stages the package launches from here on, in order."""
    stages: list[str] = []
    launch = cake_backend.launch_stage

    def recording(stage, *args, **kwargs):
        stages.append(stage)
        return launch(stage, *args, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return stages


def _preps(stages) -> set[str]:
    return {stage for stage in stages if stage.startswith("prep")}


# --------------------------------------------------------------------------- the forwards on the WY stage's A
# Layouts whose forward segment table is serial on every device of the package (more sequences x value heads than
# SMs, 192 CTAs): partial chunks, a one-token sequence, value-head groups of three, two and one.  With FP32 beta and
# normalized q / k they run wy_a and fwd_ho_a (fwd_a_route) on every architecture of FWD_A_ARCHS.
FWD_A_CASES = {
    "g3": ((16, 48), (300, 77, 1, 130)),
    "g2": ((16, 32), (300, 77, 1, 130, 64, 65)),
    "g1": ((16, 16), (300, 77, 1, 130, 64, 65, 200, 17, 129, 5, 63, 250)),
}
# Layouts whose forward segment table splits a sequence on every device of the package: split sequences beside
# sequences of one piece (one of them a single token), value-head groups of three and two, and ("g3-all") a layout
# where every sequence splits (ForwardSplit.first_seq is None: the copy of wave 1's end states on wy_a's A indexes the
# final states by first_index alone, as on the production rows; at 148 / 152 SMs pieces of 37 / 35 chunks, 17 + 9 /
# 18 + 9 pieces).  Their pieces (30..42 chunks) are shorter than the split's shape class
# (cake_plan.FORWARD_SPLIT_MIN_PIECE): the tests of the split forward's tables, bits and accuracy run them with the
# class lowered to FWD_SPLIT_TEST_MIN_PIECE (the same kernels and tables on fewer tokens than a row of the class);
# test_fwd_split_shape_class pins the class itself.
FWD_SPLIT_CASES = {
    "g3": ((2, 6), (40000, 20000, 1000, 64, 1)),
    "g2": ((4, 8), (30000, 9000, 100)),
    "g3-all": ((2, 6), (40000, 20000)),
}
FWD_SPLIT_TEST_MIN_PIECE = 16
# the host launch order of the split forward: the middle pieces' summary (side stream) is issued before wave 1; on
# wy_a's A (FWD_SPLIT_A_ARCHS) both waves are fwd_ho_a and the copy of wave 1's end states into the segment state is a
# torch op
FWD_SPLIT_STAGES = ["wy_a", "wy_tc2", "fwd_summary", "fwd_fused", "stitch", "fwd_fused"]
FWD_SPLIT_A_STAGES = ["wy_a", "fwd_summary_a", "fwd_ho_a", "stitch", "fwd_ho_a"]
# The route switches the non-deterministic package's tests set (by name, where this package has them) to hold that
# package's prep forward to this package's bits, with their off values: with all three off this package runs the prep
# forward (the prep, the WY stage, fwd_ho) on every call, as FI main does.
PREP_FORWARD_OFF = (
    ("FWD_A_ARCHS", ()),
    ("FWD_SPLIT_ARCHS", ()),
    ("FWD_A_SERIAL_KEEP", {}),
)
# the forwards on wy_a's A that the device tests force: "a" (wy_a and fwd_ho_a on a serial table), "split_a" (the
# one-wave split forward on wy_a's A: fwd_summary_a and fwd_ho_a) and "split_w" (the split forward on the WY stage's
# w / u: fwd_summary and fwd_fused).  Each split form is forced on every architecture of FWD_SPLIT_ARCHS
# (_split_form), whichever form FWD_SPLIT_A_MIN_PIECE gives the test rows' pieces there (sm_103a: the w / u form
# below 96 chunks), so both forms run on sm_100a and on sm_103a.
FWD_SPLIT_ROUTES = ("split_a", "split_w")
FWD_ROUTES = ("a",) + FWD_SPLIT_ROUTES
# (route, layout) of the device tests that also run the split forms on the layout where every sequence splits
FWD_ROUTE_CASES = (("a", "g3"),) + tuple(
    (route, case) for route in FWD_SPLIT_ROUTES for case in ("g3", "g3-all")
)
FWD_ROUTE_CASE_IDS = [f"{route}-{case}" for route, case in FWD_ROUTE_CASES]


def _current_arch() -> str:
    return cake_backend.device_arch(torch.cuda.current_device())


def _fwd_a_or_skip(device) -> str:
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if arch not in cake_chain.FWD_A_ARCHS:
        pytest.skip(f"the forward on the WY stage's A is not routed on {arch}")
    return arch


def _fwd_split_or_skip(device) -> str:
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if arch not in cake_chain.FWD_SPLIT_ARCHS:
        pytest.skip(f"the split forward is not routed on {arch}")
    return arch


def _fwd_split_test_class(monkeypatch) -> None:
    """The split forward's shape class lowered to FWD_SPLIT_TEST_MIN_PIECE (the layouts of FWD_SPLIT_CASES)."""
    monkeypatch.setattr(
        cake_plan,
        "FORWARD_SPLIT_MIN_PIECE",
        types.MappingProxyType(
            {arch: FWD_SPLIT_TEST_MIN_PIECE for arch in cake_chain.FWD_SPLIT_ARCHS}
        ),
    )


def _split_a() -> bool:
    """Whether the split forward of the current device runs on wy_a's A on the test rows: the form _split_form forces,
    else the device's (FWD_SPLIT_A_ARCHS; the rows' pieces are shorter than every FWD_SPLIT_A_MIN_PIECE entry,
    test_fwd_split_tables_follow_the_plan)."""
    return cake_chain._fwd_split_a(_current_arch(), FWD_SPLIT_TEST_MIN_PIECE)


def _split_form(route: str, device, monkeypatch) -> str:
    """Force the split forward of ``route`` (FWD_SPLIT_ROUTES) on the device, with the shape class lowered to
    FWD_SPLIT_TEST_MIN_PIECE: "split_a" on wy_a's A (FWD_SPLIT_A_ARCHS the device's architecture, no
    FWD_SPLIT_A_MIN_PIECE), "split_w" on the WY stage's w / u (FWD_SPLIT_A_ARCHS empty); skips where the split forward
    or the form's stages are not delivered.  Returns the architecture."""
    assert route in FWD_SPLIT_ROUTES, route
    arch = _fwd_split_or_skip(device)
    _fwd_split_test_class(monkeypatch)
    a_form = route == "split_a"
    stages = cake_chain.FWD_SPLIT_A_STAGES if a_form else cake_chain.FWD_SPLIT_W_STAGES
    if not all(cake_chain._available(stage, arch) for stage in stages):
        pytest.skip(
            f"the {route} form of the split forward is not delivered for {arch}"
        )
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_A_ARCHS", (arch,) if a_form else ())
    if a_form:
        monkeypatch.setattr(
            cake_chain, "FWD_SPLIT_A_MIN_PIECE", types.MappingProxyType({})
        )
    assert _split_a() is a_form, (route, arch)
    return arch


def _split_stages() -> list[str]:
    return FWD_SPLIT_A_STAGES if _split_a() else FWD_SPLIT_STAGES


def _split_wave() -> str:
    """The stage of the split forward's two waves on the current device."""
    return "fwd_ho_a" if _split_a() else "fwd_fused"


def _route_wave(route: str) -> str:
    """The stage that carries the tape word of ``route``'s forward on the current device."""
    return "fwd_ho_a" if route == "a" else _split_wave()


def _taped_word(route: str) -> int:
    """The tape word of a taped launch of ``route``'s forward on the current device: the state of the device's tape
    level (cake_plan.tape_state) in the table of the stage that writes it."""
    words = (
        cake_chain.FWD_A_TAPE
        if _route_wave(route) == "fwd_ho_a"
        else cake_chain.FWD_FUSED_TAPE
    )
    return sum(
        words[name]
        for name in cake_plan.tape_state(_current_arch(), True)
        if name in words
    )


def _forward_stages(stages) -> list[str]:
    """The launches of a call's forward (up to its last forward scan-and-output stage)."""
    last = max(
        i
        for i, s in enumerate(stages)
        if s in ("fwd_fused", "fwd_ho", "fwd_ho_vh2", "fwd_ho_a")
    )
    return list(stages[: last + 1])


def _assert_fwd_route(stages, route: str, calls: int = 1) -> None:
    """``calls`` calls whose launches are ``stages`` each ran the forward of ``route`` (and no prep forward)."""
    assert not _preps(stages), (route, stages)
    assert not {"fwd_ho", "fwd_ho_vh2"} & set(stages), (route, stages)
    if route == "a":
        assert stages.count("wy_a") == stages.count("fwd_ho_a") == calls, stages
        assert not {"fwd_fused", "fwd_summary", "fwd_summary_a"} & set(stages), stages
    else:
        if route in FWD_SPLIT_ROUTES:  # a forced form
            assert _split_a() is (route == "split_a"), route
        summary = "fwd_summary_a" if _split_a() else "fwd_summary"
        assert stages.count("wy_a") == stages.count(summary) == calls, stages
        assert stages.count(_split_wave()) == 2 * calls, stages


def _fwd_route(
    route: str, device, monkeypatch, case: str = "g3"
) -> tuple[tuple[int, int], tuple[int, ...]]:
    """The layout ``((H, HV), sequence lengths)`` of ``case`` (FWD_A_CASES for "a", else FWD_SPLIT_CASES) that runs
    ``route``'s forward with FP32 beta on the device (fresh chunk tables; a split route's form forced by _split_form);
    skips where the device does not route it."""
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    if route == "a":
        _fwd_a_or_skip(device)
        return FWD_A_CASES[case]
    _split_form(route, device, monkeypatch)
    return FWD_SPLIT_CASES[case]


def _prep_forward_off(monkeypatch) -> None:
    """Every switch of PREP_FORWARD_OFF off, with fresh chunk tables."""
    for name, off in PREP_FORWARD_OFF:
        monkeypatch.setattr(cake_chain, name, off)
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})


def _record_launch_values(monkeypatch) -> list[tuple[str, tuple]]:
    """The (stage, values in the record's parameter order) of every launch from here on, in order."""
    launches: list[tuple[str, tuple]] = []
    launch = cake_backend.launch_stage

    def recording(stage, grid, values, **kwargs):
        launches.append((stage, tuple(values)))
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return launches


def _record_launch_scalars(monkeypatch) -> list[tuple[str, tuple]]:
    """:func:`_record_launch_values` with every tensor value recorded as ``None``: the record keeps no device memory
    alive, so a test that measures the caching allocator around the launches can take it."""
    launches: list[tuple[str, tuple]] = []
    launch = cake_backend.launch_stage

    def recording(stage, grid, values, **kwargs):
        launches.append(
            (stage, tuple(None if isinstance(x, torch.Tensor) else x for x in values))
        )
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return launches


def _tape_words(launches, stage: str) -> list[int]:
    """The tape word of every launch of ``stage`` (its ``tape`` parameter, wherever the record puts it)."""
    names = cake_backend.parameters(cake_backend.stage_record(stage, _current_arch()))
    return [int(values[names.index("tape")]) for s, values in launches if s == stage]


def _state_pair(
    seq_lens, heads, device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """An initial state and a final-state gradient of the layout, drawn on the host."""
    gen = torch.Generator().manual_seed(seed)
    shape = (len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM)
    h0 = (0.1 * torch.randn(*shape, generator=gen)).to(device)
    dht = torch.randn(*shape, generator=gen).to(device)
    return h0, dht


# --------------------------------------------------------------------------- registry and manifest (no device)


# the fused tails' cluster sizes: one CTA per value head of a key head
FUSED_TAIL_STAGES = {
    "tail_g2": 2,
    "tail_g2_bf16": 2,
    "tail_g2_bf16_wavefront": 2,
    "tail_g3": 3,
    "tail_g3_bf16": 3,
    "tail_g3_f32x": 3,
    "tail_g3_f32x_bf16": 3,
}
# The switches of the round-once tail and the FP32 dv_local adjoint with their off values: the non-deterministic
# package's tests set them by name (where this package has them) to hold its tail_g3 on dhu_tc2 to this package's
# bits; with both off a value-head group of three runs tail_g3 on dhu_tc2 again.
ROUND_ONCE_OFF = (
    ("F32X_TAIL_ARCHS", ()),
    ("DVL32_ARCHS", ()),
)


def _round_once_off(monkeypatch, names=("F32X_TAIL_ARCHS", "DVL32_ARCHS")) -> None:
    """The switches of ROUND_ONCE_OFF that ``names`` lists, off."""
    for name, off in ROUND_ONCE_OFF:
        if name in names:
            monkeypatch.setattr(cake_chain, name, off)


def _device_tail(heads, beta_dtype, finalize=None) -> str | None:
    """The fused tail of ``heads`` on this device (:func:`cake_chain.fused_tail_stage`): the round-once tail of a
    value-head group of three on the architectures of ``F32X_TAIL_ARCHS``, else tail_g3; tail_g2 for two."""
    return cake_chain.fused_tail_stage(
        _current_arch(),
        *heads,
        dbeta_bf16=beta_dtype == torch.bfloat16,
        finalize=finalize,
    )


def _tc2(heads) -> str:
    """The tcgen05 adjoint scan of ``heads`` on this device (:func:`cake_chain.dhu_tc2_stage`): dhu_tc2_dvl32 for a
    value-head group of three on the architectures of ``DVL32_ARCHS``, else dhu_tc2."""
    return cake_chain.dhu_tc2_stage(_current_arch(), *heads)


# The package exports the mode-word names of FlashInfer #8928 only; the former names (with and without det) are
# removed (no alias), and the planner and table helpers stay public in ``cake_chain`` only.
DET_NAMES = (
    "CakeGDNDetFunction",
    "cake_gdn_det",
    "cake_gdn_det_train_backward",
    "cake_gdn_det_train_forward",
)
REMOVED_NAMES = (
    "ChunkGatedDeltaRuleDetFunction",
    "ChunkGatedDeltaRuleFunction",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_det",
    "gdn_chunk_backward",
    "gdn_chunk_det_backward",
    "gdn_chunk_det_forward",
    "gdn_chunk_forward",
)
CHAIN_ONLY_NAMES = (
    "CHUNK",
    "FINALIZE_STAGES",
    "FULL",
    "HEAD_DIM",
    "PRESCAN_CONFIGS",
    "REG_SCAN_CONFIGS",
    "ChainOptions",
    "ChunkMeta",
    "ScanPlan",
    "SegmentTable",
    "StageSet",
    "StreamGroups",
    "build_chunk_meta",
    "build_chunk_meta_from_cu_seqlens",
    "choose_scan_plan",
    "dhu_tc_pricing",
    "finalize_stage",
    "lpt_sequence_order",
    "lpt_walk",
    "make_options",
    "scan_plans",
    "sdt_pricing",
    "sequence_groups",
    "stage_set",
    "two_stream_split",
    "use_prep_tma",
)


def test_former_names_are_removed() -> None:
    for module in (cake_gdn_det_train, cake_chain):
        for name in REMOVED_NAMES:
            assert not hasattr(module, name), (module.__name__, name)
            assert name not in module.__all__, (module.__name__, name)
        for name in DET_NAMES:
            assert name in module.__all__, (module.__name__, name)
            assert getattr(module, name) is getattr(cake_chain, name)
    for name in DET_NAMES:
        assert getattr(cake_chain, name).__name__ == name


def test_the_package_exports_the_det_names_only() -> None:
    assert sorted(cake_gdn_det_train.__all__) == sorted(DET_NAMES)
    # the mode word as a token (the shared check; no order check, so a vendored import path passes too)
    assert_mode_names(cake_gdn_det_train, "det", stem=False)
    for name in CHAIN_ONLY_NAMES:
        assert name in cake_chain.__all__ and hasattr(cake_chain, name), name
        assert not hasattr(cake_gdn_det_train, name), name


@pytest.mark.parametrize("value", [True, False, None])
def test_the_det_entries_take_no_deterministic_keyword(value) -> None:
    # the det entries have no mode: the keyword, of any value, fails the call's argument binding (no launch)
    calls = (
        (cake_gdn_det, (None,) * 5, {}),
        (
            cake_chain.cake_gdn_det_train_forward,
            (None,) * 5,
            {"meta": None, "arch": None},
        ),
        (cake_chain.cake_gdn_det_train_backward, (None,) * 2, {}),
    )
    for entry, args, kwargs in calls:
        with pytest.raises(TypeError, match="deterministic"):
            entry(*args, **kwargs, deterministic=value)
    with pytest.raises(TypeError):
        cake_chain.CakeGDNDetFunction.apply(*(None,) * 12, deterministic=value)


@pytest.mark.parametrize(
    "tape",
    [{"mode": "nondet"}, {}, {"mode": None}, {"deterministic": True}],
    ids=["nondet-tape", "untagged", "none", "former-key"],
)
def test_det_backward_refuses_a_tape_without_the_det_tag(tape) -> None:
    # the det tape carries "mode": "det" (the low-level forward's tape and the autograd chain's own tape); the
    # non-deterministic package's tape carries "mode": "nondet"; the check runs before the tape is read
    with pytest.raises(ValueError, match="this tape does not carry that tag"):
        cake_chain.cake_gdn_det_train_backward({"arch": None, **tape}, None)
    # a tape with the det tag passes the check and is read
    with pytest.raises(KeyError, match="meta"):
        cake_chain.cake_gdn_det_train_backward({"arch": None, "mode": "det"}, None)


# the records of the forwards on the WY stage's A: one record each, on every architecture of the package, one CTA per
# cluster, no PDL, and a program of its own
FWD_A_RECORDS = ("wy_a", "fwd_ho_a", "fwd_fused", "fwd_summary", "fwd_summary_a")


def test_registry_records_are_well_formed() -> None:
    assert len(cake_jit.STAGES) == 45 and len(set(cake_jit.STAGES)) == 45
    assert set(FWD_A_RECORDS) <= set(cake_jit.STAGES)
    assert set(cake_chain.LAUNCH_ARGS) == set(cake_jit.STAGES)
    for arch in ARCHES:
        for stage in cake_jit.STAGES:
            serving = [
                name
                for name, rec in cake_jit.MODULES.items()
                if rec["stage"] == stage and arch in rec["arches"]
            ]
            assert len(serving) == 1, (stage, arch, serving)
    libraries = set()
    for name, rec in cake_jit.MODULES.items():
        assert set(rec["arches"]) <= set(ARCHES), name
        assert rec["ffi_entry"] == "run", name
        assert re.fullmatch(r"cake_gdn_det_train_[0-9a-f]{20}", rec["library"]), name
        assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        launch = rec["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert launch["cluster"] == (FUSED_TAIL_STAGES.get(rec["stage"], 1), 1, 1), name
        assert len(launch["block"]) == 3 and launch["block"][0] % 32 == 0, name
        assert launch["dynamic_smem_bytes"] >= 0, name
        # the record's kernel parameters are the call site's, possibly reordered (the chain reorders them)
        params = cake_backend.parameters(name)
        assert sorted(params) == sorted(cake_chain.LAUNCH_ARGS[rec["stage"]]), name
        libraries.add(rec["library"])
    # one program per record: a kernel swap renames exactly the records it touches; the three shared libraries are
    # the sm_107a scan of the split pass 2's pair, which is the sm_100a / sm_103a dhu_tc2_wavefront program, the
    # sm_107a wy_tc2, which is the sm_100a / sm_103a wy_tc2 program, and the sm_100a / sm_103a fwd_ho_vh2, which is
    # the fwd_ho program
    wavefront = cake_jit.MODULES["dhu_tc2_wavefront"]
    assert wavefront["arches"] == ["sm_100a", "sm_103a"]
    shared = cake_jit.MODULES["dhu_tc2_wavefront_split_sm_107a"]
    assert shared["arches"] == ["sm_107a"]
    assert shared["library"] == wavefront["library"]
    split = cake_jit.MODULES["dhu_tc2_wavefront_split"]
    assert split["arches"] == ["sm_100a", "sm_103a"]
    # the sm_107a scan of the pair: a program of its own (the release in batches of eight chunks)
    eight = cake_jit.MODULES["dhu_tc2_wavefront_sm_107a"]
    assert eight["stage"] == "dhu_tc2_wavefront" and eight["arches"] == ["sm_107a"]
    assert len({wavefront["library"], split["library"], eight["library"]}) == 3
    wy, wy2, wy2_107 = (
        cake_jit.MODULES[name] for name in ("wy_tc", "wy_tc2", "wy_tc2_sm_107a")
    )
    assert wy2["arches"] == ["sm_100a", "sm_103a"] and wy2_107["arches"] == ["sm_107a"]
    assert (
        wy2_107["library"] == wy2["library"] != wy["library"]
        and "sm_107a" in wy["arches"]
    )
    assert (
        cake_jit.MODULES["fwd_ho_vh2"]["library"]
        == cake_jit.MODULES["fwd_ho"]["library"]
    )
    for stage in FWD_A_RECORDS:
        serving = [
            name for name, rec in cake_jit.MODULES.items() if rec["stage"] == stage
        ]
        assert serving == [stage], (stage, serving)
        rec = cake_jit.MODULES[stage]
        assert sorted(rec["arches"]) == sorted(ARCHES), (stage, rec["arches"])
        assert rec["launch"]["cluster"] == (1, 1, 1), stage
        assert rec["launch"]["use_pdl"] is False, stage
        sharing = [
            name
            for name, other in cake_jit.MODULES.items()
            if other["library"] == rec["library"]
        ]
        assert sharing == [stage], (stage, sharing)
    # fwd_fused is fwd_ho's call site with the tape word and the split forward's wave-1 outputs
    assert cake_chain.LAUNCH_ARGS["fwd_fused"] == (
        *cake_chain.LAUNCH_ARGS["fwd_ho"],
        "tape",
        "seg_state",
        "seg_flag",
        "split_out",
    )
    assert len(libraries) == len(cake_jit.MODULES) - 3


def test_the_wy_stage_follows_the_architecture() -> None:
    assert cake_chain.WY_TC2_ARCHS == ("sm_100a", "sm_103a", "sm_107a")
    assert {arch: cake_chain.wy_stage(arch) for arch in ARCHES} == {
        "sm_100a": "wy_tc2",
        "sm_103a": "wy_tc2",
        "sm_107a": "wy_tc2",
    }
    # wy_tc2 reads v in place: wy_tc's arguments and v's token pitch in heads
    assert cake_chain.LAUNCH_ARGS["wy_tc2"] == (
        *cake_chain.LAUNCH_ARGS["wy_tc"],
        "v_row_heads",
    )
    plan, plan2 = (cake_jit.MODULES[name]["arg_plan"] for name in ("wy_tc", "wy_tc2"))
    assert [arg for arg in plan2 if arg != ["parameter", "v_row_heads"]] == plan


def test_v_rows_reads_whole_head_rows_in_place() -> None:
    # the stages that read v take its token pitch in 128-element heads: the value heads of a packed
    # [T, (2 H + HV) * 128] row, contiguous rows, and leading axes that fold into the token axis; other layouts
    # are copied (None)
    tokens, heads, v_heads = 8, 2, 6
    width = (2 * heads + v_heads) * HEAD_DIM
    row = torch.empty(2, tokens, width, dtype=torch.bfloat16)
    packed = row[..., 2 * heads * HEAD_DIM :].unflatten(-1, (v_heads, HEAD_DIM))
    assert cake_chain._v_rows(packed) == 2 * heads + v_heads
    assert cake_chain._v_rows(packed[:1, :5]) == 2 * heads + v_heads
    dense = torch.empty(2, tokens, v_heads, HEAD_DIM, dtype=torch.bfloat16)
    assert cake_chain._v_rows(dense) == v_heads
    head_major = torch.empty(1, v_heads, tokens, HEAD_DIM, dtype=torch.bfloat16)
    assert cake_chain._v_rows(head_major.transpose(1, 2)) is None
    half_head = torch.empty(1, tokens, v_heads * HEAD_DIM + 64, dtype=torch.bfloat16)
    half_head = half_head[..., : v_heads * HEAD_DIM].unflatten(-1, (v_heads, HEAD_DIM))
    assert cake_chain._v_rows(half_head) is None
    batch_minor = torch.empty(tokens, 2, v_heads, HEAD_DIM, dtype=torch.bfloat16)
    assert cake_chain._v_rows(batch_minor.transpose(0, 1)) is None
    flat = torch.empty(tokens * v_heads * HEAD_DIM + 1, dtype=torch.bfloat16)
    assert cake_chain._v_rows(flat[1:].view(1, tokens, v_heads, HEAD_DIM)) is None


# --------------------------------------------------------------------------- the round-once tail (no device)

ROUND_ONCE_STAGES = ("dhu_tc2_dvl32", "tail_g3_f32x", "tail_g3_f32x_bf16")


def _views(row, widths):
    """The q / k / v head views of a packed ``[..., T, (2 H + HV) * 128]`` row (the #7074 layout)."""
    return tuple(part.unflatten(-1, (-1, HEAD_DIM)) for part in row.split(widths, -1))


def test_round_once_records_are_well_formed() -> None:
    # tail_g3_f32x / tail_g3_f32x_bf16: tail_g3's call site with the token-row strides (elements) of v, dq, dk and dv in
    # place of v's pitch in heads, a cluster of three CTAs; dhu_tc2_dvl32: dhu_tc2's call site.  One record each on
    # every architecture of the package, a program of its own, no PDL
    tail_args = cake_chain.LAUNCH_ARGS["tail_g3"]
    assert tail_args[-1] == "v_row_heads"
    strides = ("v_row_stride", "dq_row_stride", "dk_row_stride", "dv_row_stride")
    assert cake_chain.F32X_TAILS == ("tail_g3_f32x", "tail_g3_f32x_bf16")
    for stage in cake_chain.F32X_TAILS:
        assert cake_chain.LAUNCH_ARGS[stage] == (*tail_args[:-1], *strides), stage
    assert cake_chain.LAUNCH_ARGS["dhu_tc2_dvl32"] == cake_chain.LAUNCH_ARGS["dhu_tc2"]
    assert set(ROUND_ONCE_STAGES) <= set(cake_jit.STAGES)
    libraries = set()
    for stage in ROUND_ONCE_STAGES:
        serving = [
            name for name, rec in cake_jit.MODULES.items() if rec["stage"] == stage
        ]
        assert serving == [stage], (stage, serving)
        rec = cake_jit.MODULES[stage]
        assert sorted(rec["arches"]) == sorted(ARCHES), (stage, rec["arches"])
        launch = rec["launch"]
        assert launch["use_pdl"] is False and launch["cooperative"] is False, stage
        cluster = 3 if stage in cake_chain.F32X_TAILS else 1
        assert launch["cluster"] == (cluster, 1, 1), stage
        sharing = [
            name
            for name, other in cake_jit.MODULES.items()
            if other["library"] == rec["library"]
        ]
        assert sharing == [stage], (stage, sharing)
        libraries.add(rec["library"])
    assert len(libraries) == len(ROUND_ONCE_STAGES)


def test_the_round_once_rule(monkeypatch) -> None:
    # a value-head group of three runs the round-once tail on the architectures of F32X_TAIL_ARCHS and dhu_tc2_dvl32 on
    # those of DVL32_ARCHS (every architecture of the package); a finalize pin keeps the unfused tail (its adjoint
    # still dhu_tc2_dvl32); a group of two keeps tail_g2 and dhu_tc2, other groups the unfused tail and dhu_tc2.  Each
    # switch off, or a registry without the round-once records, gives tail_g3 / dhu_tc2 back, the other unchanged
    assert set(cake_chain.F32X_TAIL_ARCHS) == set(ARCHES)
    assert set(cake_chain.DVL32_ARCHS) == set(ARCHES)
    rule, scan = cake_chain.fused_tail_stage, cake_chain.dhu_tc2_stage
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    for arch in ARCHES:
        for bf16, suffix in ((False, ""), (True, "_bf16")):
            for heads in ((2, 6), (1, 3), (16, 48)):
                assert rule(arch, *heads, dbeta_bf16=bf16) == "tail_g3_f32x" + suffix
                for fin in ("finalize", "finalize_mlp"):
                    assert rule(arch, *heads, dbeta_bf16=bf16, finalize=fin) is None
            assert rule(arch, 2, 4, dbeta_bf16=bf16) == "tail_g2" + suffix
            assert rule(arch, 2, 2, dbeta_bf16=bf16) is None
            assert rule(arch, 1, 4, dbeta_bf16=bf16) is None
        assert scan(arch, 2, 6) == scan(arch, 1, 3) == "dhu_tc2_dvl32"
        assert scan(arch, 2, 4) == scan(arch, 2, 2) == scan(arch, 1, 4) == "dhu_tc2"
        with monkeypatch.context() as m:
            _round_once_off(m, ("F32X_TAIL_ARCHS",))
            assert rule(arch, 2, 6, dbeta_bf16=False) == "tail_g3"
            assert rule(arch, 2, 6, dbeta_bf16=True) == "tail_g3_bf16"
            assert scan(arch, 2, 6) == "dhu_tc2_dvl32"
        with monkeypatch.context() as m:
            _round_once_off(m, ("DVL32_ARCHS",))
            assert rule(arch, 2, 6, dbeta_bf16=False) == "tail_g3_f32x"
            assert scan(arch, 2, 6) == "dhu_tc2"
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "_available", lambda stage, _arch: stage not in ROUND_ONCE_STAGES)  # fmt: skip
            assert rule(arch, 2, 6, dbeta_bf16=True) == "tail_g3_bf16"
            assert scan(arch, 2, 6) == "dhu_tc2"
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "_available", lambda _stage, _arch: False)
            assert rule(arch, 2, 6, dbeta_bf16=False) is None


def test_the_round_once_switches_are_set_by_name() -> None:
    # the non-deterministic package's tests turn this package's round-once tail and FP32 dv_local adjoint off through
    # the names of ROUND_ONCE_OFF alone, each where this package has it (getattr not None): each is set, of its off
    # value's type, and on
    for name, off in ROUND_ONCE_OFF:
        value = getattr(cake_chain, name, None)
        assert value is not None and type(value) is type(off), (name, value)
        assert value != off, (name, value)
    for name in (
        "CakeGDNDetPackedFunction",
        "DVL32_ARCHS",
        "F32X_TAILS",
        "F32X_TAIL_ARCHS",
        "PACKED_ROW",
        "dhu_tc2_stage",
        "fused_tail_stage",
    ):
        assert name in cake_chain.__all__ and hasattr(cake_chain, name), name
    assert cake_chain.PACKED_ROW is True


def test_row_stride_of_head_views() -> None:
    # the token-row stride the round-once tail takes: heads x 128 for contiguous rows, the packed row's width for its
    # head views (with leading axes that fold into the token axis); a view whose heads or channels are not dense, or
    # whose leading axes do not fold, is refused
    h, hv, t = 2, 6, 7
    widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
    row = torch.zeros(3, t, sum(widths), dtype=torch.bfloat16)
    q, _k, v = _views(row, widths)
    assert cake_chain._row_stride(q) == cake_chain._row_stride(v) == sum(widths)
    assert cake_chain._row_stride(v[:1, :5]) == sum(widths)
    assert cake_chain._row_stride(torch.zeros(1, t, hv, HEAD_DIM)) == hv * HEAD_DIM
    assert cake_chain._row_stride(torch.zeros(t, h, HEAD_DIM)) == h * HEAD_DIM
    with pytest.raises(ValueError):
        cake_chain._row_stride(v.transpose(-1, -2))
    with pytest.raises(ValueError):
        cake_chain._row_stride(v[::2])
    with pytest.raises(ValueError):
        cake_chain._row_stride(torch.zeros(1, hv, t, HEAD_DIM).transpose(1, 2))


def test_the_round_once_tail_takes_one_dq_dk_stride() -> None:
    # the round-once tail's launcher writes dq and dk at one token-row stride: contiguous dq / dk (H x 128) and the head
    # views of one packed gradient row ((2 H + HV) x 128) pass, with v / dv at their own strides; dq and dk at two
    # strides raise (ValueError, before any launch); a caller's common stride is taken for dq / dk / dv
    h, hv, t = 2, 6, 7
    widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
    width = sum(widths)
    v = torch.zeros(1, t, hv, HEAD_DIM, dtype=torch.bfloat16)
    dq, dk = (torch.zeros(1, t, h, HEAD_DIM, dtype=torch.bfloat16) for _ in range(2))
    dv = torch.zeros_like(v)
    strides = cake_chain._tail_row_strides
    assert strides(v, dq, dk, dv) == (hv * HEAD_DIM, h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)  # fmt: skip
    pq, pk, pv = _views(torch.zeros(1, t, width, dtype=torch.bfloat16), widths)
    assert strides(v, pq, pk, pv) == (hv * HEAD_DIM, width, width, width)
    assert strides(pv, pq, pk, dv) == (width, width, width, hv * HEAD_DIM)
    assert strides(v, pq, pk, pv, width) == (hv * HEAD_DIM, width, width, width)
    for got_q, got_k in ((pq, dk), (dq, pk)):
        with pytest.raises(ValueError, match="one token-row stride"):
            strides(v, got_q, got_k, dv)


def test_packed_row_recognizes_only_its_layout() -> None:
    # the head views of one contiguous BF16 row that requires grad, at offsets 0, H x 128 and 2 H x 128 with its row
    # stride; not copies, other offsets, other widths, another dtype, a row without grad or views of a larger tensor
    h, hv, t = 2, 6, 5
    widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
    row = torch.zeros(1, t, sum(widths), dtype=torch.bfloat16, requires_grad=True)
    q, k, v = _views(row, widths)
    assert cake_chain._packed_row(q, k, v) is row
    batched = torch.zeros(3, t, sum(widths), dtype=torch.bfloat16, requires_grad=True)
    assert cake_chain._packed_row(*_views(batched, widths)) is batched
    assert cake_chain._packed_row(q.detach().clone().requires_grad_(), k, v) is None
    assert cake_chain._packed_row(k, q, v) is None  # q / k swapped: wrong offsets
    wide = torch.zeros(
        1, t, sum(widths) + HEAD_DIM, dtype=torch.bfloat16, requires_grad=True
    )
    assert cake_chain._packed_row(*_views(wide[..., : sum(widths)], widths)) is None
    assert (
        cake_chain._packed_row(*_views(row.detach().float().requires_grad_(), widths))
        is None
    )
    assert cake_chain._packed_row(*_views(row.detach(), widths)) is None
    larger = torch.zeros(2, t, sum(widths), dtype=torch.bfloat16, requires_grad=True)
    assert cake_chain._packed_row(*_views(larger[:1], widths)) is None


def test_the_value_half_split_rule() -> None:
    assert cake_chain.FWD_HO_VH2_ARCHS == ("sm_107a",)
    assert cake_chain.LAUNCH_ARGS["fwd_ho_vh2"] == cake_chain.LAUNCH_ARGS["fwd_ho"]
    split, fwd_ho = cake_jit.MODULES["fwd_ho_vh2_sm_107a"], cake_jit.MODULES["fwd_ho"]
    assert split["stage"] == "fwd_ho_vh2" and split["arches"] == ["sm_107a"]
    assert split["arg_plan"] == fwd_ho["arg_plan"]
    assert split["library"] != fwd_ho["library"]
    assert split["launch"] == fwd_ho["launch"]
    # sm_100a / sm_103a never run the split: the stage's record there is the fwd_ho program
    other = cake_jit.MODULES["fwd_ho_vh2"]
    assert other["stage"] == "fwd_ho_vh2" and other["arches"] == ["sm_100a", "sm_103a"]
    assert {
        key: value for key, value in other.items() if key not in ("arches", "stage")
    } == {key: value for key, value in fwd_ho.items() if key not in ("arches", "stage")}
    assert [cake_jit.stage_record("fwd_ho_vh2", arch) for arch in ARCHES] == [
        "fwd_ho_vh2",
        "fwd_ho_vh2",
        "fwd_ho_vh2_sm_107a",
    ]
    # the doubled grid only where the record is the split; elsewhere fwd_ho's grid (the program's own)
    assert [cake_chain.fwd_ho_vh2_splits(arch) for arch in ARCHES] == [
        False,
        False,
        True,
    ]
    engaged = cake_chain.fwd_ho_vh2_engaged
    # twice the CTAs in one round of one CTA per SM
    assert engaged(6, 16, 192) and not engaged(6, 16, 191)
    assert engaged(1, 1, 2) and not engaged(1, 1, 1)
    assert not engaged(6, 16, 148) and not engaged(6, 16, 152)


# The forward scans of the campaign rows that run fwd_ho_vh2 per device (SMs, architecture): on the R200 boards the
# one-launch forward scan of r07 (one sequence, a split plan of 96 CTAs); no row on sm_100a / sm_103a.
VH2_DEVICES = {
    (148, "sm_100a"): set(),
    (152, "sm_103a"): set(),
    (208, "sm_107a"): {"r07"},
    (212, "sm_107a"): {"r07"},
    (216, "sm_107a"): {"r07"},
}


@pytest.mark.parametrize("sms,arch", sorted(VH2_DEVICES))
def test_value_half_split_on_the_campaign_rows(sms, arch, monkeypatch) -> None:
    props = (sms, 232448, 10)
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    rows = {row: (seq_lens, hv) for row, seq_lens, hv, _sel in RULE_ROWS}
    rows.update({row: (seq_lens, hv) for row, seq_lens, hv in SDT_ROWS})
    engaged = set()
    for row, (seq_lens, num_v_heads) in rows.items():
        plans = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans, lpt=True)
        fwd, _bwd = cake_chain.sequence_groups(meta, num_v_heads, sms)
        if (
            fwd is None
            and arch in cake_chain.FWD_HO_VH2_ARCHS
            and cake_chain.fwd_ho_vh2_engaged(
                meta.fwd_segments.num_segments, num_v_heads, sms
            )
        ):
            engaged.add(row)
    assert engaged == VH2_DEVICES[(sms, arch)]


def test_scan_records_carry_the_planner_constants() -> None:
    for kind in cake_chain.SCAN_KINDS:
        for vb, ks in cake_chain.REG_SCAN_CONFIGS:
            rec = cake_jit.MODULES[f"{kind}_r{vb}k{ks}"]
            assert (rec["scan"]["v_block"], rec["scan"]["k_split"]) == (vb, ks)
            assert rec["scan"]["select_smem_bytes"] > 0
        for vb, ks in cake_chain.PRESCAN_CONFIGS:
            rec = cake_jit.MODULES[f"{kind}_pre_r{vb}k{ks}"]
            assert (rec["scan"]["v_block"], rec["scan"]["k_split"]) == (vb, ks)
            assert rec["scan"]["max_segment_chunks"] == 128
    for arch in ARCHES:
        sset = cake_chain.stage_set(arch)
        for kind in cake_chain.SCAN_KINDS:
            assert sset.reg_of(kind) == cake_chain.REG_SCAN_CONFIGS, (arch, kind)
            assert sset.split_of(kind) == cake_chain.PRESCAN_CONFIGS, (arch, kind)
        const = dict(cake_chain._dhu_tc_constants(arch))
        assert set(const) == set(cake_chain.DHU_TC_CONSTANTS), arch
        assert (
            0 < const["t_lat"] <= const["t_full"]
            and const["t_unit"] > 0
            and const["c_dvl"] > 0
        ), arch
        gain, min_saving = cake_chain._sdt_constants(arch)
        assert 0 < gain <= 1 and min_saving >= 0, arch
        ctas_per_sm, window = cake_chain._prep_tma_constants(arch)
        if arch == "sm_107a":  # window 1.0: no plain-prep band
            assert ctas_per_sm > 0 and window == 1.0, arch
            continue
        assert ctas_per_sm > 0 and window > 1, arch


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {
        f"{rec['library']}_{arch}"
        for rec in cake_jit.MODULES.values()
        for arch in rec["arches"]
    }
    assert set(entries) == wanted
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert entry["path"] == f"aot/{name}.so", name
        for rel, digest in entry["sha256"].items():
            data = (PACKAGE_DIR / rel).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


# --------------------------------------------------------------------------- launch entries (no device)


def _fake_libraries(monkeypatch, arches) -> tuple[list, list]:
    """``cake_backend`` on fake devices: device ``i`` of architecture ``arches[i]``, every library a namespace whose
    FFI entry records ``(record, arch, args)``, and no resolved entry yet.  Returns the calls and the loads."""
    calls: list = []
    loads: list = []

    def load(name, arch):
        loads.append((name, arch))

        def entry(*args):
            calls.append((name, arch, args))

        return types.SimpleNamespace(**{cake_backend.record(name)["ffi_entry"]: entry})

    monkeypatch.setattr(cake_backend, "device_arch", lambda index: arches[index])
    monkeypatch.setattr(cake_backend, "load", load)
    monkeypatch.setattr(cake_backend, "_ENTRIES", {})
    return calls, loads


def test_launch_stage_resolves_each_stage_once_per_device(monkeypatch) -> None:
    # three devices, two of one architecture: each (stage, device) resolves on its first launch, then launches from
    # its cached entry with the values and the grid as ints
    arches = (ARCHES[0], ARCHES[-1], ARCHES[0])
    calls, loads = _fake_libraries(monkeypatch, arches)
    names = [cake_backend.stage_record("wy_tc", arch) for arch in arches]
    values = tuple(range(len(cake_backend.parameters(names[0]))))
    for _ in range(3):
        for index in range(len(arches)):
            cake_backend.launch_stage("wy_tc", (5, 2, 1), values, device_index=index)
    assert loads == list(zip(names, arches, strict=True))
    assert sorted(cake_backend._ENTRIES) == [("wy_tc", i) for i in range(len(arches))]
    per_pass = [
        (n, arch, values + (5, 2, 1)) for n, arch in zip(names, arches, strict=True)
    ]
    assert calls == per_pass * 3


def test_launch_stage_keeps_its_errors(monkeypatch) -> None:
    calls, loads = _fake_libraries(monkeypatch, (ARCHES[0],))
    name = cake_backend.stage_record("wy_tc", ARCHES[0])
    expected = cake_backend.parameters(name)
    count = len(expected)
    # a stage the registry does not serve
    with pytest.raises(KeyError):
        cake_backend.launch_stage("no_such_stage", (1, 1, 1), (), device_index=0)
    # the wrong argument count raises before the library loads (first launch) and on a cached entry
    for got in (count - 1, count + 1):
        with pytest.raises(ValueError) as error:
            cake_backend.launch_stage(
                "wy_tc", (1, 1, 1), tuple(range(got)), device_index=0
            )
        assert str(error.value) == (
            f"stage wy_tc ({name}) takes {count} arguments {expected}; got {got}"
        )
        assert loads == [] and cake_backend._ENTRIES == {}
    cake_backend.launch_stage("wy_tc", (1, 1, 1), tuple(range(count)), device_index=0)
    with pytest.raises(ValueError) as error:
        cake_backend.launch_stage(
            "wy_tc", (1, 1, 1), tuple(range(count + 1)), device_index=0
        )
    assert str(error.value) == (
        f"stage wy_tc ({name}) takes {count} arguments {expected}; got {count + 1}"
    )
    assert len(loads) == 1 and len(calls) == 1

    # a device that is not an exported target: device_arch's error, nothing cached for it
    def unexported(index):
        raise RuntimeError(f"device {index} is not an exported target")

    monkeypatch.setattr(cake_backend, "device_arch", unexported)
    with pytest.raises(RuntimeError, match="not an exported target"):
        cake_backend.launch_stage(
            "wy_tc", (1, 1, 1), tuple(range(count)), device_index=1
        )
    assert sorted(cake_backend._ENTRIES) == [("wy_tc", 0)]


# --------------------------------------------------------------------------- plan pins and host tables (no device)


def test_options_validate_their_pins() -> None:
    opts = cake_chain.make_options(
        scan_config=[64, 1], segment_chunks=0, finalize="finalize"
    )
    assert opts.scan_config == (64, 1) and opts.segment_chunks == 0
    assert opts.save_state is True and opts.host_cache is True and opts.dhu_scan is None
    assert opts.prep is None
    assert cake_chain.make_options(dhu_scan="tc").dhu_scan == "tc"
    assert cake_chain.make_options(prep="tma").prep == "tma"
    assert opts.lpt and opts.streams == 2
    assert cake_chain.make_options(lpt=False).lpt is False
    assert cake_chain.make_options(streams=1).streams == 1
    for bad in (
        {"scan_config": (32,)},
        {"scan_config": (32, 0)},
        {"scan_config": (32.0, 2)},
        {"segment_chunks": -1},
        {"segment_chunks": True},
        {"finalize": "finalize_g2"},
        {"warmup_log2": float("nan")},
        {"dhu_scan": "auto"},
        {"prep": "auto"},
        {"streams": 0},
        {"streams": 3},
        {"streams": True},
        {"streams": 2.0},
        {"streams": "2"},
    ):
        with pytest.raises(ValueError):
            cake_chain.make_options(**bad)


def test_finalize_rule() -> None:
    rule = cake_chain.finalize_stage
    # the three-head MLP finalize while its grid stays within 4 CTAs per SM, else the generic one
    assert rule(100, 4, 12, 148) == "finalize_mlp"
    assert rule(4 * 148 // 4 + 1, 4, 12, 148) == "finalize"
    assert rule(10, 4, 8, 148) == "finalize"
    assert rule(10, 4, 12, 148, forced="finalize") == "finalize"
    assert rule(10_000, 4, 12, 148, forced="finalize_mlp") == "finalize_mlp"
    with pytest.raises(ValueError):
        rule(10, 4, 8, 148, forced="finalize_mlp")
    with pytest.raises(ValueError):
        rule(10, 4, 10, 148)


# The campaign rows the selection rule decides on (serial adjoint plans on SM100 / SM103; on the 216 SMs of R200 the
# planner splits the adjoint scans of r01, r03 and r08 into segments), and the decision recorded with the rule's
# constants: dhu_tc2 on r01, r03, r05, r06, r08, r09 (r01: T_tc = 64 steps x t(64 CTAs), B200 154.9 us, GB300
# 141.9 us, R200 141.6 us).
_REAL_15 = (
    2756,
    4479,
    4053,
    4052,
    5896,
    4229,
    5238,
    3013,
    4429,
    4175,
    3743,
    4029,
    5404,
    3129,
    5875,
)
RULE_ROWS = (
    ("r01", (4096,) * 4, 16, True),
    ("r03", (4096,) * 8, 16, True),
    ("r05", (4096,) * 16, 16, True),
    ("r06", _REAL_15, 16, True),
    ("r08", (4096,) * 4, 32, True),
    ("r09", (4096,) * 16, 32, True),
)
RULE_DEVICES = {
    "sm_100a": ((148, 232448, 10), 154.9),
    "sm_103a": ((152, 232448, 10), 141.9),
    "sm_107a": ((216, 232448, 10), 113.3),
}
# The rows whose adjoint plan the planner splits into segments on a device (the others run serial plans).
SPLIT_ADJOINT_ROWS = {"sm_107a": ("r01", "r03", "r08")}


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_dhu_tc_rule_on_the_campaign_rows(arch, monkeypatch) -> None:
    props, r01_tc = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    for row, seq_lens, num_v_heads, selected in RULE_ROWS:
        _fwd, bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        chunks = tuple(-(-n // 64) for n in seq_lens)
        serial = bwd.segment_len == 0 or max(chunks) <= bwd.segment_len
        assert serial is (row not in SPLIT_ADJOINT_ROWS.get(arch, ())), (row, bwd)
        rec = cake_chain.dhu_tc_pricing(chunks, num_v_heads, 0)
        assert rec["select"] is selected, (row, rec)
        if row == "r01":
            assert abs(rec["T_tc"] - r01_tc) < 0.1, rec


# The split (64, 1) adjoint plans of the campaign rows and the #7074 packs on each device, and the ones whose pass 2
# the split rule runs on dhu_tc2 (measured faster there; the other split plans keep the register pass 2).
_REAL_3 = (5406, 3448, 3896)
_REAL_7 = (4199, 2781, 4282, 4448, 4723, 5804, 3863)
SDT_ROWS = (
    ("r01", (4096,) * 4, 16),
    ("r02", _REAL_3, 16),
    ("r03", (4096,) * 8, 16),
    ("r04", _REAL_7, 16),
    ("r07", (16384,), 16),
    ("r08", (4096,) * 4, 32),
    ("r10", (16384,), 32),
    ("issue", (131071, 87531, 43385), 6),
    ("P3", (128147, 131027), 6),
    ("P1", (126565, 32714, 33696, 15081, 22979, 10782), 6),
    ("A16", (11000, 12000, 9768), 48),
)
_SDT_SPLIT_SM100 = {"r02", "r04", "r07", "r10", "issue", "P3", "P1"}
SDT_SPLIT = {
    "sm_100a": _SDT_SPLIT_SM100,
    "sm_103a": _SDT_SPLIT_SM100,
    "sm_107a": _SDT_SPLIT_SM100 | {"r01", "r03", "r08", "A16"},
}
# with the dv_local stage the register form needs priced in, every split (64, 1) row of these layouts runs pass 2 on
# dhu_tc2 (sm_100a r07: a 41 us priced saving, above the 40 us floor)
SDT_SELECTED = SDT_SPLIT


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_sdt_rule_on_the_split_rows(arch, monkeypatch) -> None:
    props, _r01_tc = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    split, selected = set(), set()
    for row, seq_lens, num_v_heads in SDT_ROWS:
        _fwd, bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        chunks = tuple(-(-n // 64) for n in seq_lens)
        if bwd.segment_len == 0 or max(chunks) <= bwd.segment_len:
            continue
        if tuple(bwd.config) != cake_chain.SDT_CONFIG:
            continue
        split.add(row)
        rec = cake_chain.sdt_pricing(chunks, bwd.segment_len, num_v_heads, 0)
        assert rec["select"] is (
            rec["T_tc2"] <= rec["gain"] * rec["T_reg2"]
            and rec["saving"] >= rec["min_saving"]
        ), (row, rec)
        if rec["select"]:
            selected.add(row)
    assert split == SDT_SPLIT[arch]
    assert selected == SDT_SELECTED[arch]


# The split adjoint plans the serial-scan rule makes serial (dhu_tc2 over whole sequences), on every SM count of the
# architecture: on R200 the 8 x 4096 and the 4 x 4096 16-head layouts; none on B200 and GB300 (not the rule's
# architectures).  The other split plans keep the planner's (the four sequences of 8 heads price slower serial; the
# 16384-token sequences and the #7074 packs far slower; A16, a value-head group of three, is not the rule's).
SERIAL_SCAN_DEVICES = {
    "sm_100a": (148,),
    "sm_103a": (152,),
    "sm_107a": (208, 212, 216),
}
SERIAL_SCAN_SELECTED = {
    "sm_100a": set(),
    "sm_103a": set(),
    "sm_107a": {"r03", "r08"},
}
_GROUP_OF_THREE = ("issue", "P3", "P1", "A16")


@pytest.mark.parametrize("arch", sorted(SERIAL_SCAN_DEVICES))
def test_serial_scan_rule_on_the_split_rows(arch, monkeypatch) -> None:
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    for sm_count in SERIAL_SCAN_DEVICES[arch]:
        monkeypatch.setattr(
            cake_chain, "_device_props", lambda _index, n=sm_count: (n, 232448, 10)
        )
        selected = set()
        for row, seq_lens, num_v_heads in SDT_ROWS:
            num_heads = num_v_heads // (3 if row in _GROUP_OF_THREE else 2)
            plans = cake_chain.scan_plans(
                sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
            )
            options = cake_chain.make_options()
            got = cake_chain._serial_adjoint_plan(
                tuple(seq_lens), plans, num_heads, num_v_heads, 0, options
            )
            assert got[0] == plans[0], (sm_count, row)
            if got != plans:
                assert got[1] == cake_chain.ScanPlan(plans[1].config, 0), (
                    sm_count,
                    row,
                    got,
                )
                chunks = tuple(-(-n // 64) for n in seq_lens)
                rec = cake_chain.serial_scan_pricing(chunks, plans[1], num_v_heads, 0)
                assert rec["select"] and rec["serial_tc"], (sm_count, row, rec)
                assert rec["T_ser"] <= rec["T_cur"], (
                    sm_count,
                    row,
                    rec,
                )
                selected.add(row)
            # a plan pin and an unknown head group keep the planner's plans
            for pins in (
                {"segment_chunks": plans[1].segment_len},
                {"scan_config": (64, 1)},
                {"dhu_scan": "tc"},
            ):
                pinned = cake_chain.make_options(**pins)
                assert cake_chain._serial_adjoint_plan(tuple(seq_lens), plans, num_heads, num_v_heads, 0,
                                                       pinned) == plans, (row, pins)  # fmt: skip
            assert (
                cake_chain._serial_adjoint_plan(
                    tuple(seq_lens), plans, None, num_v_heads, 0, options
                )
                == plans
            )
        assert selected == SERIAL_SCAN_SELECTED[arch], (sm_count, selected)


# the two layouts the rule makes serial on R200, (sequence lengths, value heads, key heads): r03 and r08
SERIAL_SCAN_CACHE_ROWS = (((4096,) * 8, 16, 8), ((4096,) * 4, 32, 16))


def _serial_scan_tables(monkeypatch) -> list:
    """sm_107a at 216 SMs with fresh chunk tables built on the host; returns the list ``serial_scan_pricing`` calls
    append to."""
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: "sm_107a")
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: (216, 232448, 10))
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    monkeypatch.setattr(cake_chain, "_host_chunk_meta", lambda seq_lens, plans, _index, lpt: cake_chain.build_chunk_meta(
        seq_lens, "cpu", scan_plans=plans, lpt=lpt))  # fmt: skip
    # the serial-scan rule's tables alone (no tables priced for the tcgen05 chains)
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    priced: list = []
    pricing = cake_chain.serial_scan_pricing
    monkeypatch.setattr(
        cake_chain,
        "serial_scan_pricing",
        lambda *a, **k: priced.append(a) or pricing(*a, **k),
    )
    return priced


@pytest.mark.parametrize("seq_lens, num_v_heads, num_heads", SERIAL_SCAN_CACHE_ROWS)
def test_serial_scan_rule_is_cached_with_the_chunk_tables(
    seq_lens, num_v_heads, num_heads, monkeypatch
) -> None:
    # the decision is made once per layout with the tables (a hit prices nothing) and keyed by the key heads too
    priced = _serial_scan_tables(monkeypatch)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    options = cake_chain.make_options()
    meta = cake_chain._chunk_meta(cu, cu, 1, sum(seq_lens), 0, num_v_heads, options, num_heads=num_heads)  # fmt: skip
    assert meta.bwd_segments.num_segments == meta.num_seqs == len(seq_lens)
    assert meta.bwd_segments.segment_len == 0
    assert len(priced) == 1
    again = cake_chain._chunk_meta(cu, cu, 1, sum(seq_lens), 0, num_v_heads, options, num_heads=num_heads)  # fmt: skip
    assert again is meta
    assert len(priced) == 1
    split = cake_chain._chunk_meta(cu, cu, 1, sum(seq_lens), 0, num_v_heads, options)
    assert split.bwd_segments.num_segments > split.num_seqs
    assert split.fwd_segments.segment_len == meta.fwd_segments.segment_len
    assert len(priced) == 1


@pytest.mark.parametrize("seq_lens, num_v_heads, num_heads", SERIAL_SCAN_CACHE_ROWS)
@pytest.mark.parametrize("dhu_scan", ["tc", "reg"])
@pytest.mark.parametrize("pinned_first", [False, True])
def test_dhu_scan_pin_keys_the_chunk_tables(
    seq_lens, num_v_heads, num_heads, dhu_scan, pinned_first, monkeypatch
) -> None:
    # a dhu_scan pin keeps the planner's split adjoint plan and the rule makes it serial: on one layout the two calls
    # take their own tables in either order (a cached table of the other call would make the bits depend on it)
    _serial_scan_tables(monkeypatch)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    plans = cake_chain.scan_plans(
        sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
    )
    assert plans[1].segment_len > 0
    calls = [cake_chain.make_options(), cake_chain.make_options(dhu_scan=dhu_scan)]
    metas = {}
    for options in reversed(calls) if pinned_first else calls:
        metas[options.dhu_scan] = cake_chain._chunk_meta(cu, cu, 1, sum(seq_lens), 0, num_v_heads, options,
                                                         num_heads=num_heads)  # fmt: skip
    rule, pinned = metas[None], metas[dhu_scan]
    assert rule.bwd_segments.segment_len == 0
    assert rule.bwd_segments.num_segments == rule.num_seqs
    assert pinned.bwd_segments.segment_len == plans[1].segment_len
    assert tuple(pinned.bwd_segments.scan_config) == tuple(plans[1].config)
    assert pinned.bwd_segments.num_segments > pinned.num_seqs
    assert pinned.fwd_segments.segment_len == rule.fwd_segments.segment_len
    # each call hits its own table afterwards
    for options in calls:
        assert cake_chain._chunk_meta(cu, cu, 1, sum(seq_lens), 0, num_v_heads, options,
                                      num_heads=num_heads) is metas[options.dhu_scan]  # fmt: skip


# --------------------------------------------------------------------------- tables priced for the tcgen05 chains

_PAIR_LAYOUT = (11000, 12000, 9768)
# (row, seq_lens, H, HV): the campaign rows, the #7074 packs, head pairs and context-parallel packs, and the
# value-head groups of two
PRICED_ROWS = (
    ("r01", (4096,) * 4, 8, 16),
    ("r02", _REAL_3, 8, 16),
    ("r03", (4096,) * 8, 8, 16),
    ("r04", _REAL_7, 8, 16),
    ("r07", (16384,), 8, 16),
    ("r08", (4096,) * 4, 16, 32),
    ("r10", (16384,), 16, 32),
    ("issue", (131071, 87531, 43385), 2, 6),
    ("P3", (128147, 131027), 2, 6),
    ("P1", (126565, 32714, 33696, 15081, 22979, 10782), 2, 6),
    ("64x4096", (4096,) * 64, 2, 6),
    ("A16", _PAIR_LAYOUT, 16, 48),
    ("h8_hv24", _PAIR_LAYOUT, 8, 24),
    ("h4_hv12", _PAIR_LAYOUT, 4, 12),
    ("h2_hv6", _PAIR_LAYOUT, 2, 6),
    ("h1_hv3", _PAIR_LAYOUT, 1, 3),
    ("cp2_h8", (30001, 20000, 15535), 8, 24),
    ("cp4_h4", (70001, 40000, 21071), 4, 12),
    ("cp16_h1", (262143, 131071, 99999, 31075), 1, 3),
    ("h4_hv8", _PAIR_LAYOUT, 4, 8),
    ("h2_hv4", _PAIR_LAYOUT, 2, 4),
)
PRICED_DEVICES = (
    ("sm_100a", 148),
    ("sm_103a", 148),
    ("sm_103a", 152),
    ("sm_107a", 212),
    ("sm_107a", 216),
)


def _priced_device(monkeypatch, arch: str, sm_count: int) -> None:
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)


def _segments(chunks, plan) -> int:
    if plan.segment_len <= 0 or max(chunks) <= plan.segment_len:
        return len(chunks)
    return sum(-(-n // plan.segment_len) for n in chunks)


@pytest.mark.parametrize("arch, sm_count", PRICED_DEVICES)
def test_tc_plan_rule(arch, sm_count, monkeypatch) -> None:
    # the tc-priced adjoint table replaces the planner's (after the serial-scan rule) only where it saves at least the
    # floor and a fraction of the planner's price, within the pre-scan capacity at SDT_CONFIG; a table whose scan runs
    # as two groups beside the fused tail keeps its form
    _priced_device(monkeypatch, arch, sm_count)
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", {})
    cap = cake_chain._prescan_capacity("dhu", cake_chain.SDT_CONFIG)
    gain, min_saving = cake_chain.tc_plan_thresholds(arch)
    options = cake_chain.make_options()
    for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
        chunks = tuple(-(-n // 64) for n in seq_lens)
        plans = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        plans = cake_chain._serial_adjoint_plan(
            seq_lens, plans, num_heads, num_v_heads, 0, options
        )
        got, bwd_tc = cake_chain.host_plans(
            seq_lens, num_v_heads, 0, options, num_heads=num_heads
        )
        assert got[0] == plans[0], row
        kept = cake_chain._two_stream_kept(
            chunks, plans[1], num_heads, num_v_heads, 0, options
        )
        rec = cake_chain.tc_plan_pricing(chunks, num_v_heads, 0, plans[1])
        if kept or not rec["select"]:
            assert (got, bwd_tc) == (plans, False), (row, rec)
            continue
        assert bwd_tc and got[1] == cake_chain.ScanPlan(
            cake_chain.SDT_CONFIG, rec["segment_len"]
        ), (row, rec)
        assert tuple(plans[1].config) == cake_chain.SDT_CONFIG, (row, rec)
        assert 0 <= rec["segment_len"] <= cap, (row, rec)
        assert (
            rec["saving"] >= min_saving and rec["best"]["T"] <= gain * rec["plan_T"]
        ), (row, rec)


@pytest.mark.parametrize(
    "arch, sm_count", [("sm_100a", 148), ("sm_103a", 148), ("sm_103a", 152)]
)
def test_tc_plan_prices_the_whole_segment_prescan(arch, sm_count, monkeypatch) -> None:
    # on an architecture of TC_PLAN_PRESCAN a split candidate's pre-scan is priced for a value head that scans its whole
    # segment: the head pairs keep the planner's serial adjoint table over 24 value heads (the 6-segment table measured
    # +8 % backward on B200) and take a 12-segment table over 12 (-24 %); the warm-up price took the 6-segment table
    _priced_device(monkeypatch, arch, sm_count)
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", {})
    options = cake_chain.make_options()
    chunks = tuple(-(-n // 64) for n in _PAIR_LAYOUT)
    for num_heads, segments in ((8, len(chunks)), (4, 12)):
        num_v_heads = 3 * num_heads
        (_fwd, bwd), _bwd_tc = cake_chain.host_plans(
            _PAIR_LAYOUT, num_v_heads, 0, options, num_heads=num_heads
        )
        assert _segments(chunks, bwd) == segments, (num_heads, bwd)
    planner = cake_chain.scan_plans(
        sum(_PAIR_LAYOUT), len(_PAIR_LAYOUT), 24, 0, _PAIR_LAYOUT
    )
    planner = cake_chain._serial_adjoint_plan(_PAIR_LAYOUT, planner, 8, 24, 0, options)
    assert _segments(chunks, planner[1]) == len(chunks)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "TC_PLAN_PRESCAN", {})
        warm = cake_chain.tc_plan_pricing(chunks, 24, 0, planner[1])
        assert warm["select"] and warm["segment_len"] > 0, warm
    # the pre-scan term covers the longest pre-scanned segment
    fixed, step, _co = cake_chain.TC_PLAN_PRESCAN[arch]
    const = dict(cake_chain._dhu_tc_constants(arch))
    const["t_lat"], const["t_full"] = cake_chain.TC_PLAN_STEP[arch]
    res = cake_chain._scan_residency("dhu", *cake_chain.SDT_CONFIG, 0)
    for seg_len in (16, 32, 48, 94):
        rec = cake_chain._tc_table_price(
            chunks, 24, seg_len, sm_count, const, res, cake_chain.TC_PLAN_PRESCAN[arch]
        )
        longest = max(-(-c // -(-c // seg_len)) for c in chunks)
        assert rec["T_extra"] >= fixed + longest * step, (seg_len, rec)
    # a selected split table runs its dhu_tc2 CTAs in one round
    for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
        plans = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        plans = cake_chain._serial_adjoint_plan(
            seq_lens, plans, num_heads, num_v_heads, 0, options
        )
        rows = tuple(-(-n // 64) for n in seq_lens)
        rec = cake_chain.tc_plan_pricing(rows, num_v_heads, 0, plans[1])
        if rec["select"] and rec["segment_len"] > 0:
            assert rec["best"]["ctas"] <= sm_count, (row, rec)


def test_tc_plan_prescan_leaves_r200_unchanged(monkeypatch) -> None:
    # an architecture outside TC_PLAN_PRESCAN keeps the warm-up price (R200 prices its split tables with CHAIN_PLAN)
    assert "sm_107a" not in cake_chain.TC_PLAN_PRESCAN
    for sm_count in (212, 216):
        _priced_device(monkeypatch, "sm_107a", sm_count)
        options = cake_chain.make_options()
        for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
            got = cake_chain.host_plans(
                seq_lens, num_v_heads, 0, options, num_heads=num_heads
            )
            with monkeypatch.context() as m:
                m.setattr(cake_chain, "TC_PLAN_PRESCAN", {"sm_100a": (0.0, 99.0, 9.0)})
                assert (
                    cake_chain.host_plans(
                        seq_lens, num_v_heads, 0, options, num_heads=num_heads
                    )
                    == got
                ), row


@pytest.mark.parametrize("arch, sm_count", PRICED_DEVICES)
def test_two_stream_forms_keep_their_tables(arch, sm_count, monkeypatch) -> None:
    # a serial adjoint table that the two-stream rule groups, and a split one of a value-head group of three whose
    # pass 2 runs on dhu_tc2 as two groups of segments beside tail_g3, keep the planner's table: the same tables with
    # one or two streams, and with a finalize pin (the unfused tail) a group of three's split table is priced
    _priced_device(monkeypatch, arch, sm_count)
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", {})
    for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
        chunks = tuple(-(-n // 64) for n in seq_lens)
        tables = {
            streams: cake_chain.host_plans(
                seq_lens,
                num_v_heads,
                0,
                cake_chain.make_options(streams=streams),
                num_heads=num_heads,
            )
            for streams in (1, 2)
        }
        assert tables[1] == tables[2], row
        bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )[1]
        serial = _segments(chunks, bwd) == len(chunks)
        grouped = cake_chain.two_stream_split(
            len(chunks) if serial else _segments(chunks, bwd), num_v_heads, sm_count
        )
        r_form = (
            not serial
            and num_v_heads == 3 * num_heads
            and tuple(bwd.config) == cake_chain.SDT_CONFIG
            and cake_chain.sdt_pricing(chunks, bwd.segment_len, num_v_heads, 0)[
                "select"
            ]
        )
        kept = cake_chain._two_stream_kept(
            chunks, bwd, num_heads, num_v_heads, 0, cake_chain.make_options()
        )
        if grouped is not None and (serial or r_form):
            assert kept, row
        if kept and not (arch == "sm_107a" and num_v_heads == 2 * num_heads):
            assert tables[2][0][1] == bwd and tables[2][1] is False, row
        if r_form and grouped is not None:
            pinned = cake_chain.make_options(finalize="finalize")
            assert not cake_chain._two_stream_kept(
                chunks, bwd, num_heads, num_v_heads, 0, pinned
            ), row


@pytest.mark.parametrize(
    "pins",
    [
        {"segment_chunks": 0},
        {"scan_config": (64, 1)},
        {"dhu_scan": "tc"},
        {"dhu_scan": "reg"},
    ],
    ids=["serial", "config", "tc", "reg"],
)
@pytest.mark.parametrize("arch, sm_count", PRICED_DEVICES)
def test_plan_pins_keep_the_planner_tables(arch, sm_count, pins, monkeypatch) -> None:
    _priced_device(monkeypatch, arch, sm_count)
    options = cake_chain.make_options(**pins)
    for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
        plans = cake_chain.scan_plans(sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens,
                                      scan_config=options.scan_config, segment_chunks=options.segment_chunks)  # fmt: skip
        got = cake_chain.host_plans(
            seq_lens, num_v_heads, 0, options, num_heads=num_heads
        )
        assert got == (plans, False), (row, pins)


@pytest.mark.parametrize(
    "arch, sm_count", [d for d in PRICED_DEVICES if d[0] != "sm_107a"]
)
def test_chain_plan_keeps_the_b200_gb300_tables(arch, sm_count, monkeypatch) -> None:
    # CHAIN_PLAN lists sm_107a only: on B200 and GB300 every row keeps the tables it plans without CHAIN_PLAN, also
    # with a saving floor of 0
    _priced_device(monkeypatch, arch, sm_count)
    assert arch not in cake_chain.CHAIN_PLAN
    options = cake_chain.make_options()
    for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
        chunks = tuple(-(-n // 64) for n in seq_lens)
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "CHAIN_PLAN", {})
            want = cake_chain.host_plans(
                seq_lens, num_v_heads, 0, options, num_heads=num_heads
            )
        floor0 = {
            name: {**const, "min_saving": 0.0}
            for name, const in cake_chain.CHAIN_PLAN.items()
        }
        for table in (cake_chain.CHAIN_PLAN, floor0):
            with monkeypatch.context() as m:
                m.setattr(cake_chain, "CHAIN_PLAN", table)
                assert (
                    cake_chain.host_plans(
                        seq_lens, num_v_heads, 0, options, num_heads=num_heads
                    )
                    == want
                ), row
                for kind, plan in zip(("fwd", "bwd"), want[0], strict=True):
                    rec = cake_chain.chain_plan_pricing(
                        kind, chunks, num_v_heads, 0, plan
                    )
                    assert rec["select"] is False, (row, kind, rec)


@pytest.mark.parametrize("sm_count", [212, 216])
def test_chain_plan_rule_on_r200(sm_count, monkeypatch) -> None:
    # a deeper table replaces the chosen one exactly where its pricing selects it and the table's scan does not run as
    # two groups: a deeper SDT_CONFIG split table within the pre-scan capacity, priced cheaper by the floor and gain
    _priced_device(monkeypatch, "sm_107a", sm_count)
    const = cake_chain.CHAIN_PLAN["sm_107a"]
    cap = {
        "fwd": cake_chain._prescan_capacity("fwd_h", cake_chain.SDT_CONFIG),
        "bwd": cake_chain._prescan_capacity("dhu", cake_chain.SDT_CONFIG),
    }
    options = cake_chain.make_options()
    for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
        chunks = tuple(-(-n // 64) for n in seq_lens)
        got, bwd_tc = cake_chain.host_plans(
            seq_lens, num_v_heads, 0, options, num_heads=num_heads
        )
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "CHAIN_PLAN", {})
            want, want_tc = cake_chain.host_plans(
                seq_lens, num_v_heads, 0, options, num_heads=num_heads
            )
        planner = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        planner = cake_chain._serial_adjoint_plan(
            seq_lens, planner, num_heads, num_v_heads, 0, options
        )
        kept = cake_chain._two_stream_kept(
            chunks, planner[1], num_heads, num_v_heads, 0, options
        )
        split = cake_chain.two_stream_split(len(chunks), num_v_heads, sm_count)
        for kind, new, old in zip(("fwd", "bwd"), got, want, strict=True):
            grouped = (_segments(chunks, old) == len(chunks) and split is not None) or (
                kind == "bwd" and kept
            )
            rec = cake_chain.chain_plan_pricing(kind, chunks, num_v_heads, 0, old)
            assert (new != old) is (rec["select"] and not grouped), (row, kind, rec)
            if new != old:
                assert tuple(new.config) == cake_chain.SDT_CONFIG, (row, kind, new)
                assert 0 < new.segment_len <= cap[kind], (row, kind, new)
                assert _segments(chunks, new) > _segments(chunks, old), (
                    row,
                    kind,
                    new,
                    old,
                )
                assert rec["saving"] >= const["min_saving"], (row, kind, rec)
                assert rec["best"]["T"] <= const["gain"] * rec["plan_T"], (
                    row,
                    kind,
                    rec,
                )
        assert bwd_tc is (want_tc or got[1] != want[1]), row


def test_priced_tables_are_keyed_with_the_chunk_tables(monkeypatch) -> None:
    # TC_PLAN and CHAIN_PLAN enter the table cache's key: a table planned with them is not served without them
    _priced_device(monkeypatch, "sm_107a", 212)
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    monkeypatch.setattr(cake_chain, "_host_chunk_meta", lambda seq_lens, plans, _index, lpt: cake_chain.build_chunk_meta(
        seq_lens, "cpu", scan_plans=plans, lpt=lpt))  # fmt: skip
    seq_lens, num_heads, num_v_heads = (16384,), 8, 16
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    options = cake_chain.make_options()
    priced = cake_chain._chunk_meta(
        cu, cu, 1, sum(seq_lens), 0, num_v_heads, options, num_heads=num_heads
    )
    want, bwd_tc = cake_chain.host_plans(
        seq_lens, num_v_heads, 0, options, num_heads=num_heads
    )
    assert priced.bwd_tc is bwd_tc
    assert priced.bwd_segments.segment_len == want[1].segment_len
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    planner = cake_chain._chunk_meta(
        cu, cu, 1, sum(seq_lens), 0, num_v_heads, options, num_heads=num_heads
    )
    assert planner is not priced and planner.bwd_tc is False
    plans = cake_chain.scan_plans(
        sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
    )
    assert planner.bwd_segments.segment_len == plans[1].segment_len


def test_two_stream_split_rule() -> None:
    split = cake_chain.two_stream_split
    assert split(1, 32, 148) is None  # one sequence
    assert split(4, 32, 148) is None  # one round of 128 CTAs
    assert split(74, 4, 148) is None  # two full rounds: no idle slot
    assert split(4, 6, 5) is None  # 24 CTAs leave 1 idle slot; every group leaves more
    # 288 of the 512 CTAs leave 8 idle slots instead of 80
    assert split(16, 32, 148) == 9
    assert split(16, 32, 152) == 14  # 448 CTAs leave 8 idle slots instead of 96
    assert split(5, 4, 8) == 4  # 8 and 16 CTAs both fill their rounds: the larger group


# The two-stream groups of the campaign rows, (forward group A, adjoint group A) in sequences (``None``: one stream):
# the forward's group A is the partial round, the adjoint's the full rounds.  On the 216 SMs of R200 the split adjoint
# plans of r03 and r08 group their segments (adjoint group A in segments; their forward runs one stream).  The other
# rows run one stream (one sequence or one round).
STREAM_ROWS = {
    "sm_100a": {"r05": (7, 9), "r06": (6, 9), "r09": (7, 9)},
    "sm_103a": {"r05": (7, 9), "r06": (6, 9), "r09": (2, 14)},
    "sm_107a": {
        "r03": (None, 13),
        "r05": (3, 13),
        "r06": (2, 13),
        "r08": (None, 6),
        "r09": (3, 13),
    },
}


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_two_stream_groups_on_the_campaign_rows(arch, monkeypatch) -> None:
    props, _r01 = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    for row, seq_lens, num_v_heads, _form in RULE_ROWS:
        plans = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        expected = STREAM_ROWS[arch].get(row)
        for lpt in (True, False):
            meta = cake_chain.build_chunk_meta(
                seq_lens, "cpu", scan_plans=plans, lpt=lpt
            )
            fwd, bwd = cake_chain.sequence_groups(meta, num_v_heads, props[0])
            got = (
                None if fwd is None else fwd.split,
                None if bwd is None else bwd.split,
            )
            assert got == (expected or (None, None)), (row, lpt, fwd, bwd, plans)
            starts = meta.seq_chunk_start.tolist()
            seg_starts = meta.bwd_segments.seg_chunk_start.tolist()
            assert fwd is None or fwd.split_chunk == starts[fwd.split], (row, lpt)
            assert bwd is None or bwd.split_chunk == seg_starts[bwd.split], (row, lpt)


# The adjoint groups of the split (64, 1) rows, adjoint group A in segments (``None``: one stream, one round or whole
# rounds); every such row runs its pass 2 on dhu_tc2 (SDT_SELECTED), so the rows of a value-head group of three
# (issue, P3, P1: tail_g3) run the two-stream form; beside tail_g2 the adjoint keeps one stream.
SPLIT_STREAM_ROWS = {
    "sm_100a": {"r02": 9, "r04": 9, "issue": 24, "P3": 24, "P1": 24},
    "sm_103a": {"r02": 9, "r10": 4, "issue": 25, "P3": 25, "P1": 25},
    "sm_107a": {"r03": 13, "r08": 6},
}


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_split_pass2_groups_on_the_split_rows(arch, monkeypatch) -> None:
    props, _r01_tc = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    grouped = set()
    for row, seq_lens, num_v_heads in SDT_ROWS:
        plans = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        for lpt in (True, False):
            meta = cake_chain.build_chunk_meta(
                seq_lens, "cpu", scan_plans=plans, lpt=lpt
            )
            segs = meta.bwd_segments
            if segs.num_segments <= meta.num_seqs:
                continue  # a serial adjoint plan (the campaign-row test above)
            fwd, bwd = cake_chain.sequence_groups(meta, num_v_heads, props[0])
            assert fwd is None, (row, fwd)
            k = cake_chain.two_stream_split(segs.num_segments, num_v_heads, props[0])
            assert (None if bwd is None else bwd.split) == k, (row, lpt, bwd, k)
            if bwd is not None:
                seg_starts = segs.seg_chunk_start.tolist()
                assert bwd.split_chunk == seg_starts[k], (row, lpt, bwd)
                assert 0 < bwd.split_chunk < meta.num_chunks
                grouped.add(row)
    assert grouped == set(SPLIT_STREAM_ROWS[arch]), grouped
    for row, seq_lens, num_v_heads in SDT_ROWS:
        if row in SPLIT_STREAM_ROWS[arch]:
            plans = cake_chain.scan_plans(
                sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
            )
            meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans)
            _fwd, bwd = cake_chain.sequence_groups(meta, num_v_heads, props[0])
            assert bwd is not None and bwd.split == SPLIT_STREAM_ROWS[arch][row], (
                row,
                bwd,
            )


def test_sequence_groups_partition_a_serial_table() -> None:
    seq_lens, num_v_heads = (300, 77, 640, 1000, 0, 129), 4
    n = len(seq_lens)
    serial = (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 0))
    for lpt in (False, True):
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=serial, lpt=lpt)
        starts = meta.seq_chunk_start.tolist()
        for sm_count in range(2, n * num_v_heads + 1):
            k = cake_chain.two_stream_split(n, num_v_heads, sm_count)
            fwd, bwd = cake_chain.sequence_groups(meta, num_v_heads, sm_count)
            if k is None:
                assert fwd is None and bwd is None, sm_count
                continue
            # group A: the table's first sequences, at a chunk boundary; group B: the rest (neither empty of chunks)
            for groups, split in ((fwd, n - k), (bwd, k)):
                if groups is None:
                    assert starts[split] in (0, meta.num_chunks), (sm_count, split)
                    continue
                assert groups.split == split and groups.split_chunk == starts[split], (
                    sm_count,
                    groups,
                )
                assert 0 < groups.split_chunk < meta.num_chunks
    # a split plan: its forward (a walk) runs one stream; its adjoint groups the table's segments by the same rule,
    # group A the first k segments (the chunks before segment k)
    split_plans = (cake_chain.ScanPlan((32, 2), 4), cake_chain.ScanPlan((64, 1), 4))
    for lpt in (True, False):
        meta = cake_chain.build_chunk_meta(
            (2000, 300, 77), "cpu", scan_plans=split_plans, lpt=lpt
        )
        assert (meta.fwd_segments.walk is not None) is lpt
        segs = meta.bwd_segments
        seg_starts = segs.seg_chunk_start.tolist()
        assert segs.num_segments == 11 > meta.num_seqs
        grouped = 0
        for sm_count in range(2, segs.num_segments * num_v_heads + 1):
            k = cake_chain.two_stream_split(segs.num_segments, num_v_heads, sm_count)
            fwd, bwd = cake_chain.sequence_groups(meta, num_v_heads, sm_count)
            assert fwd is None, (sm_count, fwd)
            if k is None:
                assert bwd is None, (sm_count, bwd)
                continue
            grouped += 1
            assert bwd is not None and (bwd.split, bwd.split_chunk) == (
                k,
                seg_starts[k],
            ), (sm_count, bwd)
            assert 0 < bwd.split_chunk < meta.num_chunks
        assert grouped > 0
    # 40 CTAs fill 8 rounds of 5 SMs: one stream
    meta = cake_chain.build_chunk_meta(
        (2000, 300), "cpu", scan_plans=split_plans, lpt=True
    )
    assert cake_chain.sequence_groups(meta, num_v_heads, 5) == (None, None)
    serial_meta = cake_chain.build_chunk_meta((2000, 300), "cpu", scan_plans=serial)
    # 8 CTAs on 5 SMs split; without a layout on the host they do not
    assert cake_chain.sequence_groups(serial_meta, num_v_heads, 5) != (None, None)
    device_meta = dataclasses.replace(serial_meta, seq_lens=None)
    assert cake_chain.sequence_groups(device_meta, num_v_heads, 5) == (None, None)


# The prep dispatch window on each architecture's SM count (slots = ctas_per_sm x SMs, the record's select
# constants): the plain prep inside (slots, window x slots], prep_tma outside; 4 x 4096 tokens at H = 8 (2048 CTAs)
# lie inside on sm_100a and sm_103a.  sm_107a has window 1.0, no band: prep_tma for every grid.
@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_prep_rule(arch, monkeypatch) -> None:
    props, _r01 = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    rule = cake_chain.use_prep_tma
    ctas_per_sm, window = cake_chain._prep_tma_constants(arch)
    slots = ctas_per_sm * props[0]
    if arch == "sm_107a":
        assert window == 1.0
        assert all(rule(n, 1, 0) for n in (1, slots, slots + 1, 4 * slots, 32 * slots))
        assert rule(4 * 4096 // 64, 8, 0) and rule(16 * 4096 // 64, 8, 0)
        assert rule(2, 2, 0)
        return
    assert rule(slots, 1, 0) and not rule(slots + 1, 1, 0)
    assert not rule(int(window * slots), 1, 0) and rule(int(window * slots) + 1, 1, 0)
    assert not rule(4 * 4096 // 64, 8, 0)
    assert rule(16 * 4096 // 64, 8, 0) and rule(2, 2, 0)


def test_host_tables_follow_the_layout() -> None:
    plans = (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 4))
    seq_lens = (100, 640, 1, 64)
    meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans)
    chunks = [2, 10, 1, 1]
    assert meta.num_chunks == sum(chunks) and meta.num_seqs == 4
    assert meta.seq_chunk_start.tolist() == [0, 2, 12, 13, 14]
    assert meta.chunk_start.tolist()[:3] == [0, 64, 100]
    assert meta.chunk_len.tolist()[:2] == [64, 36]
    assert meta.chunk_len.tolist()[-2:] == [1, 64]
    fwd, bwd = meta.fwd_segments, meta.bwd_segments
    # the serial plan: one segment per sequence
    assert fwd.num_segments == 4 and fwd.seq_seg_start.tolist() == [0, 1, 2, 3, 4]
    # segments of 4 chunks: the 10-chunk sequence splits into 3 balanced segments
    assert bwd.seg_chunk_start.tolist() == [0, 2, 5, 8, 12, 13, 14]
    assert bwd.seg_seq.tolist() == [0, 1, 1, 1, 2, 3]
    assert bwd.seq_seg_start.tolist() == [0, 1, 4, 5, 6]
    # the caller's order: no sequence order and the identity walk
    assert meta.order is None and meta.order_index is None
    assert fwd.walk is None and bwd.walk is None


def _segment_lens(seg_chunk_start) -> list[int]:
    starts = seg_chunk_start.tolist()
    return [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)]


@pytest.mark.parametrize("fwd_segment_len", [0, 4], ids=["serial", "split"])
def test_lpt_tables_reorder_the_sequences_and_the_walk(fwd_segment_len) -> None:
    seq_lens = (100, 640, 1, 64, 2000, 640)
    chunks = [-(-n // 64) for n in seq_lens]
    offsets = _cu(seq_lens)
    plans = (
        cake_chain.ScanPlan((32, 2), fwd_segment_len),
        cake_chain.ScanPlan((64, 1), 0),
    )
    meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans, lpt=True)
    plain = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans)
    order = meta.order
    # the sequences longest first by chunk count, ties in the caller's order: a permutation
    assert sorted(order) == list(range(len(seq_lens)))
    assert order == cake_chain.lpt_sequence_order(seq_lens) == (4, 1, 5, 0, 2, 3)
    assert [chunks[s] for s in order] == sorted(chunks, reverse=True)
    assert meta.order_index.tolist() == [
        list(order),
        [order.index(s) for s in range(len(order))],
    ]
    # every chunk keeps its token offset and length; sequence vs of the table is the caller's order[vs]
    seq_chunk_start = meta.seq_chunk_start.tolist()
    for vs, seq in enumerate(order):
        c0, c1 = seq_chunk_start[vs], seq_chunk_start[vs + 1]
        assert c1 - c0 == chunks[seq]
        if c1 > c0:
            assert meta.chunk_start.tolist()[c0] == offsets[seq]
    pairs = sorted(zip(meta.chunk_start.tolist(), meta.chunk_len.tolist(), strict=True))
    assert pairs == sorted(
        zip(plain.chunk_start.tolist(), plain.chunk_len.tolist(), strict=True)
    )
    assert (
        meta.num_chunks == plain.num_chunks
        and meta.seq_lens == plain.seq_lens == seq_lens
    )
    # fwd_ho's walk: the forward segments longest first, ties in table order (None: already in that order)
    fwd = meta.fwd_segments
    lens = _segment_lens(fwd.seg_chunk_start)
    walk = fwd.walk.tolist() if fwd.walk is not None else list(range(fwd.num_segments))
    assert sorted(walk) == list(range(fwd.num_segments))
    assert walk == sorted(range(len(lens)), key=lambda seg: (-lens[seg], seg))
    if fwd_segment_len == 0:
        # the table order of a serial plan is already longest first
        assert fwd.walk is None
    else:
        assert fwd.walk is not None
    assert meta.bwd_segments.walk is None
    # an already longest-first layout keeps the caller's tables
    equal = cake_chain.build_chunk_meta((4096,) * 4, "cpu", scan_plans=plans, lpt=True)
    assert equal.order is None and equal.order_index is None


# --------------------------------------------------------------------------- against the FP64 recurrence


# id, sequence lengths, (H, HV), beta dtype, table route, keyword pins, l2norm in kernel
CASES = (
    ("pack-2x6-host-fp32beta", (100, 640, 77, 300), (2, 6), torch.float32, "host", {}, True),
    ("pack-2x6-device-bf16beta", (100, 640, 77, 300), (2, 6), torch.bfloat16, "device", {}, True),
    ("pack-4x8-host-bf16beta", (63, 64, 65, 500), (4, 8), torch.bfloat16, "host", {}, True),
    ("one-token", (1,), (1, 3), torch.float32, "host", {}, True),
    ("scan-32x2-serial", (300, 1000), (2, 4), torch.bfloat16, "host", {"scan_config": (32, 2), "segment_chunks": 0}, True),
    ("scan-32x1-serial", (300, 1000), (2, 4), torch.bfloat16, "host", {"scan_config": (32, 1), "segment_chunks": 0}, True),
    ("scan-64x1-serial", (300, 1000), (2, 4), torch.float32, "device", {"scan_config": (64, 1), "segment_chunks": 0}, True),
    ("split-32x2-host", (2000, 300), (1, 2), torch.bfloat16, "host", {"scan_config": (32, 2), "segment_chunks": 4}, True),
    ("split-64x1-device", (2000, 300), (1, 2), torch.float32, "device", {"scan_config": (64, 1), "segment_chunks": 6}, True),
    ("split-warmup", (2000,), (1, 3), torch.bfloat16, "host", {"scan_config": (32, 2), "segment_chunks": 8, "warmup_log2": -30.0}, True),
    ("finalize-forced", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"finalize": "finalize"}, True),
    ("finalize-mlp-forced", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"finalize": "finalize_mlp"}, True),
    ("recompute", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"save_state": False}, True),
    ("uncached", (100, 640, 77, 300), (2, 6), torch.float32, "device", {"host_cache": False}, True),
    ("no-l2norm", (200, 513), (2, 6), torch.bfloat16, "host", {}, False),
    ("recompute-split", (2000, 300), (1, 2), torch.bfloat16, "host", {"save_state": False, "scan_config": (32, 2), "segment_chunks": 4}, True),
    ("dhu-tc-g2-host", (100, 640, 77, 300), (2, 4), torch.bfloat16, "host", {"dhu_scan": "tc"}, True),
    ("dhu-tc-g3-device", (100, 640, 77, 300), (2, 6), torch.float32, "device", {"dhu_scan": "tc"}, True),
    ("dhu-tc-g1-recompute", (300, 1000), (2, 2), torch.bfloat16, "host", {"dhu_scan": "tc", "save_state": False}, True),
    ("dhu-reg-g2", (300, 1000), (4, 8), torch.float32, "host", {"dhu_scan": "reg", "scan_config": (64, 1)}, True),
    ("prep-tma-g3-device", (100, 640, 77, 300), (2, 6), torch.bfloat16, "device", {"prep": "tma"}, True),
    ("prep-plain-g2-host", (63, 64, 65, 500), (4, 8), torch.float32, "host", {"prep": "plain"}, True),
)  # fmt: skip


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_matches_the_fp64_recurrence(case, device) -> None:
    _id, seq_lens, heads, beta_dtype, route, pins, l2norm = case
    inputs = _inputs(seq_lens, heads, beta_dtype, device, normalized_qk=not l2norm)
    got = _run(inputs, seq_lens, route=route, use_qk_l2norm_in_kernel=l2norm, **pins)
    assert got["dbeta"].dtype == beta_dtype and got["dg"].dtype == torch.float32
    assert got["o"].dtype == torch.bfloat16
    ref = _reference(inputs, seq_lens, normalize=l2norm)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES)


@pytest.mark.parametrize(
    "pins",
    [{}, {"dhu_scan": "tc"}, {"scan_config": (32, 2), "segment_chunks": 4}],
    ids=["rule", "dhu-tc", "split"],
)
def test_initial_and_final_state(pins, device) -> None:
    seq_lens, heads = (200, 333), (2, 4)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=3)
    gen = torch.Generator().manual_seed(7)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    got = _run(
        inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, **pins
    )
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


@pytest.mark.parametrize("route", ["host", "device"])
def test_empty_sequences_carry_their_state(route, device) -> None:
    # the offsets may repeat (FLA's contract): an empty sequence passes its initial state through and its final-state
    # gradient back to dh0
    seq_lens, heads = (200, 0, 333, 0), (2, 4)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=37)
    gen = torch.Generator().manual_seed(41)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    got = _run(
        inputs,
        seq_lens,
        route=route,
        initial_state=h0,
        output_final_state=True,
        dht=dht,
    )
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


# --------------------------------------------------------------------------- long layouts and the pre-scan capacity

# Layouts whose device-route bound layout (one sequence of ``T // 64 + N - (N - 1)`` chunks) prices a split longer
# than one pre-scan CTA covers unless the search rejects it.  The two routes plan different layouts, so their outputs
# are compared with tolerance; each one carries the error of its own plan (twice the FP64 gate's max-abs bound).
LONG_LAYOUTS = (
    ("16x4096-h8", (4096,) * 16, (8, 16)),
    ("15-real-h8", _REAL_15, (8, 16)),
    ("1x16384-h16", (16384,), (16, 32)),
)  # fmt: skip


@pytest.mark.parametrize("layout", LONG_LAYOUTS, ids=[c[0] for c in LONG_LAYOUTS])
def test_split_plans_fit_the_prescan_capacity(layout, device) -> None:
    _id, seq_lens, heads = layout
    index = device.index
    for lens in (
        seq_lens,
        None,
    ):  # host route: real lengths; device route: the bound layout
        fwd, bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), heads[1], index, lens
        )
        for kind, plan in (("fwd_h", fwd), ("dhu", bwd)):
            if plan.segment_len:
                capacity = cake_chain._prescan_capacity(kind, tuple(plan.config))
                assert 0 < plan.segment_len <= capacity, (
                    kind,
                    plan,
                    capacity,
                    lens is None,
                )


@pytest.mark.parametrize("layout", LONG_LAYOUTS, ids=[c[0] for c in LONG_LAYOUTS])
def test_long_device_route_runs_and_is_deterministic(layout, device) -> None:
    _id, seq_lens, heads = layout
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=13)
    first = _run(inputs, seq_lens, route="device")
    _assert_bitwise(first, _run(inputs, seq_lens, route="device"), OUTPUT_NAMES)
    host = _run(inputs, seq_lens, route="host")
    report = {name: _metrics(first[name], host[name]) for name in OUTPUT_NAMES}
    bad = [
        name
        for name, m in report.items()
        if not (
            m["finite"]
            and m["rel_rms"] <= REL_RMS_MAX
            and m["max_abs"] <= 2 * MAX_ABS_REL * m["max_ref"]
        )
    ]
    assert not bad, f"device vs host route outside the cross-plan gate: {bad}; {report}"


def test_segments_beyond_the_prescan_capacity_raise(device) -> None:
    seq_lens = (
        140 * 64,
    )  # one sequence of 140 chunks: a forced 129-chunk segment length splits it
    inputs = _inputs(seq_lens, (1, 2), torch.bfloat16, device, seed=17)
    with pytest.raises(ValueError, match="exceeds"):
        _run(inputs, seq_lens, scan_config=(64, 1), segment_chunks=129)


# --------------------------------------------------------------------------- bitwise contracts


@pytest.mark.parametrize(
    "pins",
    [
        {"scan_config": (32, 2), "segment_chunks": 0, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 0, "dhu_scan": "tc"},
        {"scan_config": (32, 2), "segment_chunks": 4, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 6, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 6, "dhu_scan": "tc"},
    ],
    ids=["serial-32x2", "serial-dhu-tc", "split-32x2", "split-64x1", "split-64x1-tc"],
)
def test_table_routes_agree_bitwise(pins, device, monkeypatch) -> None:
    # the host- and device-built tables of one plan: the same bits on every output.  A split (64, 1) plan runs its
    # tcgen05 pass 2 on a host-built table only (_sdt_selected); the device-built one keeps dv_local + the register pass
    # 2, whose bits dhu_tc2 gives.  dhu_tc2_dvl32 (this value-head group of three on DVL32_ARCHS) rounds dv2 once, so
    # the split tc pin runs here with DVL32_ARCHS off (on: test_table_routes_split_tc_on_dhu_tc2_dvl32_within_the_budgets)
    if pins["dhu_scan"] == "tc" and pins["segment_chunks"] > 0:
        _round_once_off(monkeypatch, ("DVL32_ARCHS",))
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=11)
    host = _run(inputs, seq_lens, route="host", **pins)
    dev = _run(inputs, seq_lens, route="device", **pins)
    _assert_bitwise(host, dev, OUTPUT_NAMES)


def test_table_routes_split_tc_on_dhu_tc2_dvl32_within_the_budgets(
    device, monkeypatch
) -> None:
    # the split-64x1-tc pins with DVL32_ARCHS on: the host-built table runs pass 2 on dhu_tc2_dvl32 (after the fused
    # pre-scan, no dv_local stage), the device-built one dv_local + the register pass 2.  o keeps one set of bits; the
    # gradients agree within the budgets of each other and of the FP64 recurrence; each route repeats bitwise
    arch = _current_arch()
    if arch not in cake_chain.DVL32_ARCHS:
        pytest.skip(f"{arch} runs the adjoint without the FP32 dv_local values")
    heads = (2, 6)
    assert _tc2(heads) == "dhu_tc2_dvl32"
    pins = {"scan_config": (64, 1), "segment_chunks": 6, "dhu_scan": "tc"}
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=11)
    stages = _record_launches(monkeypatch)
    runs = {}
    for route, want, other in (
        ("host", ("dhu_pre_dvl_r64k1", "dhu_tc2_dvl32"), ("dv_local", "dhu_pre_r64k1", "dhu_r64k1", "dhu_tc2")),
        ("device", ("dv_local", "dhu_pre_r64k1", "dhu_r64k1"), ("dhu_pre_dvl_r64k1", "dhu_tc2_dvl32", "dhu_tc2")),
    ):  # fmt: skip
        start = len(stages)
        runs[route] = _run(inputs, seq_lens, route=route, **pins)
        got = stages[start:]
        assert "stitch" in got and all(x in got for x in want) and not any(x in got for x in other), (route, got)  # fmt: skip
        _assert_bitwise(_run(inputs, seq_lens, route=route, **pins), runs[route], OUTPUT_NAMES)  # fmt: skip
    _assert_bitwise(runs["host"], runs["device"], ("o",))
    _assert_close_to_reference(runs["host"], runs["device"], OUTPUT_NAMES)
    ref = _reference(inputs, seq_lens)
    for route in runs:
        _assert_close_to_reference(runs[route], ref, OUTPUT_NAMES)


@pytest.mark.parametrize("route", ["host", "device"])
def test_run_to_run_bitwise(route, device) -> None:
    seq_lens = (2048, 100, 513)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=5)
    first = _run(inputs, seq_lens, route=route)
    for _ in range(2):
        _assert_bitwise(first, _run(inputs, seq_lens, route=route), OUTPUT_NAMES)
    uncached = _run(inputs, seq_lens, route=route, host_cache=False)
    _assert_bitwise(first, uncached, OUTPUT_NAMES)


@pytest.mark.parametrize("route", ["host", "device"])
def test_torch_deterministic_algorithms_change_nothing(route, device) -> None:
    # every launch is fixed-order: PyTorch's deterministic mode (strict, so an alert would raise) gives the same bits
    seq_lens = (2048, 100, 513)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=5)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        strict = _run(inputs, seq_lens, route=route)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    _assert_bitwise(strict, _run(inputs, seq_lens, route=route), OUTPUT_NAMES)


def test_low_level_entries_record_and_check_the_mode(device) -> None:
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=263)
    index = cake_chain._device_index(device)
    plans = cake_chain.scan_plans(sum(seq_lens), len(seq_lens), 6, index, seq_lens)
    meta = cake_chain.build_chunk_meta(seq_lens, device, scan_plans=plans)
    q, k, v, g, beta, do = (inputs[n] for n in ("q", "k", "v", "g", "beta", "do"))
    tape = cake_chain.cake_gdn_det_train_forward(
        q, k, v, g, beta, meta=meta, arch=cake_backend.device_arch(index)
    )
    assert tape["mode"] == "det" and "deterministic" not in tape
    grads = cake_chain.cake_gdn_det_train_backward(tape, do)
    assert all(bool(torch.isfinite(grads[n]).all()) for n in GRAD_NAMES)
    untagged = {key: value for key, value in tape.items() if key != "mode"}
    for other in ({**tape, "mode": "nondet"}, untagged):
        with pytest.raises(ValueError, match="this tape does not carry that tag"):
            cake_chain.cake_gdn_det_train_backward(other, do)


@pytest.mark.skipif(
    "cake_gdn_nondet_train" not in PACKAGES,
    reason="pending integration: the non-deterministic package is not in this tree's PACKAGES table",
)
def test_det_backward_refuses_a_real_nondet_tape(device) -> None:
    # the tape of the non-deterministic package's low-level forward (the same keys as the det tape but its own tag):
    # the det backward refuses it; a non-deterministic package that does not import fails here, it is not skipped
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=263)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    nd_chain = cake_package("cake_gdn_nondet_train", "cake_chain")
    q, k, v, g, beta, do = (inputs[n] for n in ("q", "k", "v", "g", "beta", "do"))

    def meta_of(chain):
        plans = chain.scan_plans(
            sum(seq_lens), len(seq_lens), heads[1], index, seq_lens
        )
        return chain.build_chunk_meta(seq_lens, device, scan_plans=plans)

    nd_tape = nd_chain.cake_gdn_nondet_train_forward(
        q, k, v, g, beta, meta=meta_of(nd_chain), arch=arch
    )
    assert nd_tape["mode"] == "nondet"
    with pytest.raises(ValueError, match="this tape does not carry that tag"):
        cake_chain.cake_gdn_det_train_backward(nd_tape, do)
    tape = cake_chain.cake_gdn_det_train_forward(
        q, k, v, g, beta, meta=meta_of(cake_chain), arch=arch
    )
    grads = cake_chain.cake_gdn_det_train_backward(tape, do)
    assert all(bool(torch.isfinite(grads[n]).all()) for n in GRAD_NAMES)


@pytest.mark.parametrize(
    "pins",
    [{}, {"save_state": False}, {"scan_config": (32, 2), "segment_chunks": 4}],
    ids=["rule", "recompute", "fwd-split"],
)
def test_lpt_order_matches_the_caller_order(pins, device) -> None:
    # the host route's longest-first sequence order and fwd_ho walk against the caller's order, with an initial state
    # and a final-state gradient (state rows gathered and scattered): the same bits on every output
    seq_lens, heads = (300, 1000, 77, 640), (2, 4)
    assert cake_chain.lpt_sequence_order(seq_lens) == (1, 3, 0, 2)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=53)
    gen = torch.Generator().manual_seed(59)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    runs = {
        lpt: _run(
            inputs,
            seq_lens,
            initial_state=h0,
            output_final_state=True,
            dht=dht,
            lpt=lpt,
            **pins,
        )
        for lpt in (True, False)
    }
    _assert_bitwise(runs[True], runs[False], OUTPUT_NAMES + ("final_state", "dh0"))


def test_device_route_keeps_the_caller_order(device) -> None:
    seq_lens = (300, 1000, 77, 640)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int32, device=device)
    options = cake_chain.make_options(scan_config=(32, 2), segment_chunks=4)
    index = cake_chain._device_index(device)
    meta = cake_chain._chunk_meta(cu, None, 1, sum(seq_lens), index, 4, options)
    assert meta.seq_lens is None and meta.order is None and meta.order_index is None
    assert meta.fwd_segments.walk is None and meta.bwd_segments.walk is None
    walk = cake_chain._walk(device, meta.fwd_segments, options)
    assert walk.tolist() == list(range(meta.fwd_segments.num_segments))


def _small_device(monkeypatch, device, sm_count: int) -> None:
    """Let the planner and the two-stream rule see ``sm_count`` SMs (small layouts then group), with fresh layout
    caches so no table planned here outlives the test."""
    _sm, smem, major = cake_chain._device_props(cake_chain._device_index(device))
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, smem, major)
    )
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    monkeypatch.setattr(cake_chain, "_DHU_AUTO", {})
    monkeypatch.setattr(cake_chain, "_SDT_AUTO", {})


def _record_streams(monkeypatch) -> list[tuple[str, int]]:
    """The (stage, stream) of every launch from here on, in order."""
    launches: list[tuple[str, int]] = []
    launch = cake_backend.launch_stage

    def recording(stage, grid, values, **kwargs):
        launches.append(
            (stage, torch.cuda.current_stream(kwargs["device_index"]).cuda_stream)
        )
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return launches


# (H, HV) -> the SMs the two-stream tests pretend: 4 sequences then split (HV 4: 16 CTAs on 6 SMs, adjoint A = 3;
# HV 6: 24 CTAs on 7 SMs, adjoint A = 1), and the forward's group A is the rest
STREAM_CASES = {(2, 4): 6, (2, 6): 7}
STREAM_LAYOUT = (300, 77, 640, 1000)  # every group boundary falls after a partial chunk
# serial plans on dhu_tc2: both directions can group
STREAM_PINS = {"segment_chunks": 0, "dhu_scan": "tc"}
# split (64, 1) plans with the dhu_tc2 pass 2: the adjoint groups its segments (24 + 2 + 10 chunks in segments of 4 =
# 10 segments; HV 4: 40 CTAs on 6 SMs, adjoint A = 9 segments; HV 6: 60 CTAs on 7 SMs, A = 7); the forward runs one
# stream
SPLIT_STREAM_CASES = {(2, 4): 6, (2, 6): 7}
SPLIT_STREAM_LAYOUT = (1500, 77, 640)
SPLIT_STREAM_PINS = {"scan_config": (64, 1), "segment_chunks": 4, "dhu_scan": "tc"}
# (layout, heads, SMs the planner sees, pins, the side stream's launches per call; dhu_tc2 the heads' tcgen05 adjoint
# scan on the device, _on_device)
TWO_STREAM_FORMS = {
    "serial-g2": (
        STREAM_LAYOUT,
        (2, 4),
        6,
        STREAM_PINS,
        ["wy_tc", "fwd_ho", "dhu_tc2"],
    ),
    "split-g3": (SPLIT_STREAM_LAYOUT, (2, 6), 7, SPLIT_STREAM_PINS, ["dhu_tc2"]),
}


def _on_device(stages: list[str], heads=None) -> list[str]:
    """``stages`` with ``wy_tc`` as this device's forward WY stage (:func:`cake_chain.wy_stage`) and, given
    ``heads``, ``dhu_tc2`` as their tcgen05 adjoint scan there (:func:`_tc2`)."""
    wy = cake_chain.wy_stage(cake_backend.device_arch(torch.cuda.current_device()))
    names = {"wy_tc": wy, "dhu_tc2": "dhu_tc2" if heads is None else _tc2(heads)}
    return [names.get(stage, stage) for stage in stages]


def _adjoint_two_streams(form: str, heads: tuple[int, int]) -> bool:
    """Whether the adjoint ``dhu_tc2`` runs as two groups on a grouped table on this device: a serial plan beside
    ``tail_g2`` (beside ``tail_g3`` too on the architectures of ``TWO_STREAM_SERIAL_G3_ARCHS``), the pass 2 of a split
    plan beside ``tail_g3``; the other pairings keep the adjoint on the caller's stream."""
    arch = cake_backend.device_arch(torch.cuda.current_device())
    serial = (2, 3) if arch in cake_chain.TWO_STREAM_SERIAL_G3_ARCHS else (2,)
    group = heads[1] // heads[0]
    return (form == "serial" and group in serial) or (form, group) == ("split", 3)


@pytest.mark.parametrize("save_state", [True, False], ids=["saved", "recompute"])
@pytest.mark.parametrize("lpt", [True, False], ids=["lpt", "caller-order"])
@pytest.mark.parametrize("heads", sorted(STREAM_CASES), ids=["g2", "g3"])
def test_two_streams_match_one_stream(
    heads, lpt, save_state, device, monkeypatch
) -> None:
    seq_lens = STREAM_LAYOUT
    _small_device(monkeypatch, device, STREAM_CASES[heads])
    assert (
        cake_chain.two_stream_split(len(seq_lens), heads[1], STREAM_CASES[heads])
        is not None
    )
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=61)
    gen = torch.Generator().manual_seed(67)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    runs = []
    for streams in (2, 1, 2):
        start = len(launches)
        runs.append(_run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, streams=streams,
                         lpt=lpt, save_state=save_state, **STREAM_PINS))  # fmt: skip
        side = [stage for stage, stream in launches[start:] if stream != main]
        # the side stream: wy_tc and fwd_ho of the forward's group A, dhu_tc2 (dhu_tc2_dvl32 for G = 3 on
        # DVL32_ARCHS) of the adjoint's group B beside the fused tail of group A (beside the tail of G = 3 off
        # TWO_STREAM_SERIAL_G3_ARCHS one launch on the caller's stream)
        expected = _on_device(
            ["wy_tc", "fwd_ho"]
            + (["dhu_tc2"] if _adjoint_two_streams("serial", heads) else []),
            heads,
        )
        assert side == (expected if streams == 2 else []), (streams, launches[start:])
    _assert_bitwise(runs[0], runs[1], names)
    _assert_bitwise(runs[0], runs[2], names)
    if lpt and save_state:
        ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
        _assert_close_to_reference(runs[0], ref, names)


@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("lpt", [True, False], ids=["lpt", "caller-order"])
@pytest.mark.parametrize("heads", sorted(SPLIT_STREAM_CASES), ids=["g2", "g3"])
def test_split_pass2_two_streams_match_one_stream(
    heads, lpt, beta_dtype, device, monkeypatch
) -> None:
    # a split (64, 1) plan's dhu_tc2 pass 2 beside the tail of G = 3 as two groups of segments (the caller's stream:
    # group A, then the fused tail of its chunks; the side stream: group B, joined before the tail of B), beside
    # tail_g2 the one-stream launches of streams=1, against one stream: the same bits on every output, with an initial
    # state and a final-state gradient.  Against the register pass 2 the same bits on every output where the pass 2
    # runs on dhu_tc2; on dhu_tc2_dvl32 (G = 3 on DVL32_ARCHS, its R form on tc_stage too), which rounds dv2 once, o
    # and the final state keep its bits and every output stays within the budgets of it
    seq_lens = SPLIT_STREAM_LAYOUT
    _small_device(monkeypatch, device, SPLIT_STREAM_CASES[heads])
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=131)
    gen = torch.Generator().manual_seed(137)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    tail, tc2 = _device_tail(heads, beta_dtype), _tc2(heads)
    grouped = _adjoint_two_streams("split", heads)
    runs, stage_lists = {}, {}
    for form, pins in (
        ("two", {"streams": 2}),
        ("one", {"streams": 1}),
        ("reg", {"streams": 2, "dhu_scan": "reg"}),
    ):
        start = len(launches)
        runs[form] = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, lpt=lpt,
                          **{**SPLIT_STREAM_PINS, **pins})  # fmt: skip
        # beside tail_g2_bf16 the tcgen05 pass 2 of more CTAs than the SMs the planner sees runs as the wavefront pair
        # on the architectures of the rule's split side on a call of enough tokens x value heads (the same launches
        # in another form, the tail on the side stream); this layout stays below the work floor
        pair = (
            form != "reg"
            and tail == "tail_g2_bf16"
            and arch in cake_chain.WAVEFRONT_SPLIT_ARCHS
            and sum(seq_lens) * heads[1] >= cake_chain.WAVEFRONT_SPLIT_MIN_TOKEN_HEADS
        )
        stages = stage_lists[form] = [
            stage.removesuffix("_wavefront") for stage, _stream in launches[start:]
        ]
        side = [stage for stage, stream in launches[start:] if stream != main]
        assert "stitch" in stages, stages
        if form == "two" and grouped:
            assert side == [tc2], launches[start:]
            assert stages.count(tc2) == 2 and stages.count(tail) == 2, stages
        else:
            want_side = ["tail_g2_bf16_wavefront"] if pair else []
            assert side == want_side, launches[start:]
            assert stages.count(tail) == 1, stages
            assert stages.count(tc2) == (0 if form == "reg" else 1), stages
    if not grouped:
        assert stage_lists["two"] == stage_lists["one"], stage_lists
    _assert_bitwise(runs["two"], runs["one"], names)
    if tc2 == "dhu_tc2":
        _assert_bitwise(runs["two"], runs["reg"], names)
    else:
        _assert_bitwise(runs["two"], runs["reg"], ("o", "final_state"))
        _assert_close_to_reference(runs["two"], runs["reg"], names)
    if lpt and beta_dtype == torch.bfloat16:
        ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
        _assert_close_to_reference(runs["two"], ref, names)


@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_split_pass2_on_dhu_tc2_matches_the_register_pass2(
    heads, beta_dtype, device, monkeypatch
) -> None:
    # a split (64, 1) plan's pass 2 on dhu_tc2 from the stitched segment states against the register pass 2 at the
    # same segment table: the same bits on every output, with an initial state and a final-state gradient.  The
    # adjoint rounds dv2 as the register pass 2 does (DVL32_ARCHS empty; the FP32 dv_local adjoint:
    # test_split_pass2_on_dhu_tc2_dvl32_within_the_budgets_of_the_register_pass2)
    _round_once_off(monkeypatch, ("DVL32_ARCHS",))
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=83)
    gen = torch.Generator().manual_seed(89)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    runs = {}
    for scan in ("tc", "reg"):
        start = len(launches)
        runs[scan] = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, scan_config=(64, 1),
                          segment_chunks=6, dhu_scan=scan)  # fmt: skip
        # the wavefront pair (g2, BF16 beta) is the pass 2 and the tail in another form
        stages = [
            stage.removesuffix("_wavefront") for stage, _stream in launches[start:]
        ]
        assert "stitch" in stages, stages
        # the tc form: the fused pre-scan computes the dv_local tiles it reads, no dv_local stage
        want = (
            ("dhu_pre_dvl_r64k1", "dhu_tc2")
            if scan == "tc"
            else ("dv_local", "dhu_pre_r64k1", "dhu_r64k1")
        )
        other = (
            ("dv_local", "dhu_pre_r64k1", "dhu_r64k1")
            if scan == "tc"
            else ("dhu_pre_dvl_r64k1", "dhu_tc2")
        )
        assert all(x in stages for x in want) and not any(x in stages for x in other), (
            scan,
            stages,
        )
    _assert_bitwise(runs["tc"], runs["reg"], names)


@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("heads", [(2, 6), (1, 3)], ids=["g3", "g3-h1"])
def test_split_pass2_on_dhu_tc2_dvl32_within_the_budgets_of_the_register_pass2(
    heads, beta_dtype, device, monkeypatch
) -> None:
    # the same plan with the FP32 dv_local adjoint (dhu_tc2_dvl32 rounds dv2 once; a value-head group of three): o and
    # the final state keep the register pass 2's bits; every output within the budgets of the register pass 2 and of
    # the FP64 recurrence.  The pre-scan keeps its BF16 dv_local tile (dhu_pre_dvl_r64k1); both runs take the
    # round-once tail
    arch = _current_arch()
    if arch not in cake_chain.DVL32_ARCHS:
        pytest.skip(f"{arch} runs the adjoint without the FP32 dv_local values")
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=83)
    gen = torch.Generator().manual_seed(89)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    assert _tc2(heads) == "dhu_tc2_dvl32"
    tail = _device_tail(heads, beta_dtype)
    tc_stages = ("dhu_pre_dvl_r64k1", "dhu_tc2_dvl32")
    reg_stages = ("dv_local", "dhu_pre_r64k1", "dhu_r64k1")
    runs = {}
    for scan in ("tc", "reg"):
        start = len(launches)
        runs[scan] = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, scan_config=(64, 1),
                          segment_chunks=6, dhu_scan=scan)  # fmt: skip
        stages = [stage for stage, _stream in launches[start:]]
        assert "stitch" in stages and tail in stages and "dhu_tc2" not in stages, stages
        want, other = (
            (tc_stages, reg_stages) if scan == "tc" else (reg_stages, tc_stages)
        )
        assert all(x in stages for x in want) and not any(x in stages for x in other), (scan, stages)  # fmt: skip
    _assert_bitwise(runs["tc"], runs["reg"], ("o", "final_state"))
    _assert_close_to_reference(runs["tc"], runs["reg"], names)
    _assert_close_to_reference(
        runs["tc"], _reference(inputs, seq_lens, initial_state=h0, dht=dht), names
    )


@pytest.mark.parametrize("heads", [(8, 16), (8, 24)], ids=["g2", "g3"])
def test_tc_priced_table_changes_only_the_adjoint_table(
    heads, device, monkeypatch
) -> None:
    # one 16384-token sequence (gates kept above the floors with TC_PLAN_GAIN = 1, no saving floor; no CHAIN_PLAN
    # tables): the tc-priced adjoint table leaves the forward's bits, runs on dhu_tc2 (no dv_local stage), repeats
    # bitwise and stays within the accuracy gate of the planner's table; a plan pin keeps the planner's table
    seq_lens = (16384,)
    index = cake_chain._device_index(device)
    monkeypatch.setattr(cake_chain, "TC_PLAN_MIN_SAVING", 0.0)
    monkeypatch.setattr(cake_chain, "TC_PLAN_GAIN", 1.0)
    monkeypatch.setattr(cake_chain, "TC_PLAN_SELECT", {})
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", {})
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    plans = cake_chain.scan_plans(16384, 1, heads[1], index, seq_lens)
    rec = cake_chain.tc_plan_pricing((256,), heads[1], index, plans[1])
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=307)
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    base = _run(inputs, seq_lens)
    monkeypatch.setattr(cake_chain, "TC_PLAN", True)
    stages = _record_launches(monkeypatch)
    runs = [_run(inputs, seq_lens) for _ in range(2)]
    assert torch.equal(runs[0]["o"], base["o"])
    _assert_bitwise(runs[1], runs[0], OUTPUT_NAMES)
    _assert_close_to_reference(runs[0], base, OUTPUT_NAMES)
    if rec["select"]:
        tc2 = _tc2(heads)
        want = ("dhu_pre_dvl_r64k1", tc2) if rec["segment_len"] else (tc2,)
        assert all(x in stages for x in want) and "dv_local" not in stages, (
            rec,
            stages,
        )
    else:  # the pricing keeps the planner's table: the same bits
        _assert_bitwise(runs[0], base, OUTPUT_NAMES)
    pins = {
        "scan_config": tuple(plans[1].config),
        "segment_chunks": plans[1].segment_len,
    }
    on = _run(inputs, seq_lens, **pins)
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    _assert_bitwise(on, _run(inputs, seq_lens, **pins), OUTPUT_NAMES)


@pytest.mark.parametrize("switches", ["on", "off"])
@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("route", ["host", "device"])
def test_group_of_three_tail_matches_the_unfused_tail(
    route, beta_dtype, switches, device, monkeypatch
) -> None:
    # HV = 3 H: the fused tail against dqkwg + the WY pullback + each finalize (a finalize pin keeps the three stages),
    # with an initial state and a final-state gradient.  The switches of ROUND_ONCE_OFF off: tail_g3 (on dhu_tc2), the
    # same bits on every output.  On: the round-once tail, the same bits on every output but dq / dk, which round once
    # after the group's FP32 sum where the unfused tail rounds each value head's partials first (both on the same
    # adjoint scan); all within the budgets of the FP64 recurrence
    if switches == "off":
        _round_once_off(monkeypatch)
    seq_lens, heads = (1500, 77, 640, 1), (2, 6)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=97)
    gen = torch.Generator().manual_seed(101)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    stages = _record_launches(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    tail = _device_tail(heads, beta_dtype)
    round_once = tail in cake_chain.F32X_TAILS
    assert round_once is (switches == "on"), tail
    runs = {}
    for fin in ("", "finalize", "finalize_mlp"):
        start = len(stages)
        pins = {"finalize": fin} if fin else {}
        runs[fin] = _run(inputs, seq_lens, route=route, initial_state=h0, output_final_state=True, dht=dht,
                         **pins)  # fmt: skip
        got = stages[start:]
        if fin:
            assert fin in got and "dqkwg" in got and tail not in got, (fin, got)
        else:
            assert tail in got and "dqkwg" not in got, got
            assert not any(stage.startswith(("wy_bwd", "finalize")) for stage in got), (
                got
            )
        assert not set(ROUND_ONCE_STAGES) & set(got) or round_once, got
    same = tuple(n for n in names if n not in ("dq", "dk")) if round_once else names
    for fin in ("finalize", "finalize_mlp"):
        _assert_bitwise(runs[""], runs[fin], same)
    if round_once:
        assert not (
            torch.equal(runs[""]["dq"], runs["finalize"]["dq"])
            and torch.equal(runs[""]["dk"], runs["finalize"]["dk"])
        )
        ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
        _assert_close_to_reference(runs[""], ref, names)
        _assert_close_to_reference(runs["finalize"], ref, names)


@pytest.mark.parametrize("form", sorted(TWO_STREAM_FORMS))
def test_two_streams_survive_interleaved_allocations(form, device, monkeypatch) -> None:
    # the caller's stream allocates, writes and frees between two-stream calls (pooled and fresh workspaces) and,
    # after every side-stream launch, while the side stream works: memory released before the join would be
    # overwritten with NaN on the caller's stream
    seq_lens, heads, sm_count, pins, side_per_call = TWO_STREAM_FORMS[form]
    _small_device(monkeypatch, device, sm_count)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=73)
    one = _run(inputs, seq_lens, streams=1, **pins)
    gen = torch.Generator().manual_seed(79)
    main = torch.cuda.current_stream(device)
    launch = cake_backend.launch_stage
    side_launches = []

    def churning(stage, grid, values, **kwargs):
        out = launch(stage, grid, values, **kwargs)
        if torch.cuda.current_stream(kwargs["device_index"]) != main:
            side_launches.append(stage)
            with torch.cuda.stream(main):
                for size in (1 << 12, 1 << 16, 1 << 20):
                    torch.full((size,), float("nan"), device=device)
        return out

    monkeypatch.setattr(cake_backend, "launch_stage", churning)
    for i in range(16):
        junk = [
            torch.full(
                (int(torch.randint(1, 1 << 20, (1,), generator=gen)),),
                float("nan"),
                device=device,
            )
            for _ in range(3)
        ]
        got = _run(inputs, seq_lens, streams=2, host_cache=i % 2 == 0, **pins)
        del junk
        _assert_bitwise(got, one, OUTPUT_NAMES)
    assert side_launches == _on_device(side_per_call, heads) * 16, side_launches


@pytest.mark.parametrize("form", sorted(TWO_STREAM_FORMS))
def test_two_streams_under_cuda_graph_capture(form, device, monkeypatch) -> None:
    seq_lens, heads, sm_count, pins, side_per_call = TWO_STREAM_FORMS[form]
    _small_device(monkeypatch, device, sm_count)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=83)
    eager = _run(inputs, seq_lens, streams=1, **pins)
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = cake_gdn_det(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                      scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                      cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True, streams=2,
                                      **pins)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
        return o

    # warm up on a side stream (the tables, the workspaces and the libraries), then capture forward and backward
    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream(device).wait_stream(warmup)
    launches = _record_streams(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = step()
    captured = list(launches)
    # the capture forked the side stream: two streams, the side one for the form's side launches
    streams = {stream for _stage, stream in captured}
    assert len(streams) == 2, captured
    # the capture stream: the call's first launch (the prep) runs on the caller's stream
    main = captured[0][1]
    side = [stage for stage, stream in captured if stream != main]
    assert side == _on_device(side_per_call, heads), captured
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    got = {
        "o": o,
        **{"d" + name: leaves[name].grad for name in ("q", "k", "v", "g", "beta")},
    }
    _assert_bitwise(got, eager, OUTPUT_NAMES)
    assert len(launches) == len(captured)  # replays launch nothing from the host


def test_launches_resolve_once_and_reuse_their_entries(device, monkeypatch) -> None:
    # the first call resolves every stage it launches once; the second resolves nothing and launches the same stages
    # from the cached entries, with the same bits
    monkeypatch.setattr(cake_backend, "_ENTRIES", {})
    resolved: list[str] = []
    resolve = cake_backend._stage_entry

    def counting(stage, device_index, count):
        resolved.append(stage)
        return resolve(stage, device_index, count)

    monkeypatch.setattr(cake_backend, "_stage_entry", counting)
    stages = _record_launches(monkeypatch)
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=101)
    first = _run(inputs, seq_lens)
    n = len(stages)
    assert sorted(resolved) == sorted(set(stages))
    second = _run(inputs, seq_lens)
    assert len(resolved) == len(set(stages)) and stages[n:] == stages[:n]
    _assert_bitwise(first, second, OUTPUT_NAMES)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    for stage in set(stages):
        entry, arity, name = cake_backend._ENTRIES[(stage, index)]
        assert name == cake_backend.stage_record(stage, arch)
        assert arity == len(cake_backend.parameters(name))
        assert entry is getattr(
            cake_backend.load(name, arch), cake_backend.record(name)["ffi_entry"]
        )


@pytest.mark.parametrize("form", sorted(TWO_STREAM_FORMS))
@pytest.mark.parametrize("streams", [1, 2])
def test_launches_run_on_the_callers_stream(streams, form, device, monkeypatch) -> None:
    # The caller's stream sleeps, then rewrites q (before the forward) and do (before the backward).  A launch on any
    # other stream would run during the sleep and read the old q / do; the outputs must be those of the new ones.
    seq_lens, heads, sm_count, pins, _side = TWO_STREAM_FORMS[form]
    _small_device(monkeypatch, device, sm_count)
    old = _inputs(seq_lens, heads, torch.bfloat16, device, seed=103)
    new = _inputs(seq_lens, heads, torch.bfloat16, device, seed=107)
    want = _run(
        {**old, "q": new["q"], "do": new["do"]},
        seq_lens,
        streams=streams,
        **pins,
    )
    cu = _cu(seq_lens)
    caller = torch.cuda.Stream(device=device)
    caller.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(caller):
        leaves = {
            name: old[name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        }
        do = old["do"].clone()
        torch.cuda._sleep(200_000_000)
        with torch.no_grad():
            leaves["q"].copy_(new["q"])
        o, _ = cake_gdn_det(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                      scale=1.0 / math.sqrt(HEAD_DIM),
                                      cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                                      cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64), use_qk_l2norm_in_kernel=True,
                                      streams=streams, **pins)  # fmt: skip
        torch.cuda._sleep(200_000_000)
        do.copy_(new["do"])
        o.backward(do)
    torch.cuda.synchronize()
    got = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_bitwise(got, want, OUTPUT_NAMES)


def test_low_level_entries_reject_host_tensors(device) -> None:
    # the low-level entries skip the API checks: a host tensor's launch runs on the current device (as
    # torch.device("cpu").index maps there) and the binding rejects it with its ValueError
    cu = torch.tensor([0, 100, 164], dtype=torch.int32)
    assert cake_chain._tensor_index(cu) == torch.cuda.current_device()
    assert cake_chain._tensor_index(cu.to("meta")) == torch.cuda.current_device()
    index = cake_chain._device_index(device)
    assert cake_chain._tensor_index(cu.to(device)) == index
    plans = cake_chain.scan_plans(164, 2, 4, index, (100, 64))
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        cake_chain.build_chunk_meta_from_cu_seqlens(cu, 164, scan_plans=plans)


def test_each_caller_stream_forks_its_own_side_stream(device) -> None:
    index = cake_chain._device_index(device)
    a, b = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    side = cake_chain._side_stream(index, a)
    assert cake_chain._side_stream(index, a) is side
    assert cake_chain._side_stream(index, b) not in (side, a, b)
    assert side not in (a, b)
    assert side.priority == torch.cuda.Stream.priority_range()[1]


@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
def test_prep_forms_agree_bitwise(route, beta_dtype, device, monkeypatch) -> None:
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), beta_dtype, device, seed=19)
    stages = _record_launches(monkeypatch)
    tma = _run(inputs, seq_lens, route=route, prep="tma")
    tma_preps = _preps(stages)
    stages.clear()
    plain = _run(inputs, seq_lens, route=route, prep="plain")
    suffix = "_bf16beta" if beta_dtype == torch.bfloat16 else ""
    assert tma_preps == {"prep_tma" + suffix} and _preps(stages) == {"prep" + suffix}
    _assert_bitwise(tma, plain, OUTPUT_NAMES)


@pytest.mark.parametrize("save_state", [True, False], ids=["saved", "recompute"])
def test_the_wy_stage_of_the_device_gives_the_wy_tc_bits(
    save_state, device, monkeypatch
) -> None:
    # every forward WY launch (and the backward's recompute) runs the device's stage, wy_tc2; wy_tc in its place
    # gives the same bits
    seq_lens = (300, 77, 640, 17)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=29)
    wy = cake_chain.wy_stage(cake_backend.device_arch(torch.cuda.current_device()))
    stages = _record_launches(monkeypatch)
    ruled = _run(inputs, seq_lens, save_state=save_state)
    launched = [stage for stage in stages if stage.startswith("wy_tc")]
    assert launched and set(launched) == {wy}, stages
    stages.clear()
    monkeypatch.setattr(cake_chain, "wy_stage", lambda arch: "wy_tc")
    plain = _run(inputs, seq_lens, save_state=save_state)
    replaced = [stage for stage in stages if stage.startswith("wy_tc")]
    assert replaced == ["wy_tc"] * len(launched), stages
    _assert_bitwise(ruled, plain, OUTPUT_NAMES)


# (layout, heads, route, pins): a serial plan with grouped value heads and partial chunks, a split plan (segment start
# states from the pre-scan and the stitch) and the device route (the table's segment bound, padding segments
# included); each forward scan is one launch of at most half the SMs of any of the devices
VH2_FORMS = {
    "serial": ((300, 77, 640, 17), (2, 4), "host", {}),
    "split": (
        (2000, 300),
        (1, 2),
        "host",
        {"scan_config": (32, 2), "segment_chunks": 4},
    ),
    "device": ((100, 640, 77, 300), (2, 6), "device", {}),
}


@pytest.mark.parametrize("form", sorted(VH2_FORMS))
def test_the_value_half_split_gives_the_fwd_ho_bits(form, device, monkeypatch) -> None:
    # with the rule on for the device's architecture (fwd_ho_vh2 is delivered for every architecture) the forward scan
    # runs fwd_ho_vh2; fwd_ho in its place gives the same bits, the final state and the gradients included
    seq_lens, heads, route, pins = VH2_FORMS[form]
    arch = cake_backend.device_arch(torch.cuda.current_device())
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=83)
    gen = torch.Generator().manual_seed(89)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    stages = _record_launches(monkeypatch)
    runs = []
    for archs, stage in (((arch,), "fwd_ho_vh2"), ((), "fwd_ho")):
        monkeypatch.setattr(cake_chain, "FWD_HO_VH2_ARCHS", archs)
        stages.clear()
        runs.append(_run(inputs, seq_lens, route=route, initial_state=h0, output_final_state=True, **pins))  # fmt: skip
        assert [name for name in stages if name.startswith("fwd_ho")] == [stage], stages
    _assert_bitwise(runs[0], runs[1], OUTPUT_NAMES + ("final_state", "dh0"))


def test_unaligned_qk_take_the_plain_prep(device, monkeypatch) -> None:
    # q / k as contiguous views 8 bytes past a 16-byte boundary: the tensor maps of prep_tma need 16-byte aligned
    # addresses, the plain prep's 4-element loads 8; a prep_tma pin still runs the plain prep, with the same bits.
    seq_lens = (300, 77)
    inputs = _inputs(seq_lens, (2, 4), torch.float32, device, seed=23)
    leaves = {}
    for name in ("q", "k"):
        t = inputs[name]
        buf = torch.empty(t.numel() + 4, dtype=t.dtype, device=device)
        leaves[name] = buf[4:].view(t.shape).copy_(t).requires_grad_(True)
        assert leaves[name].is_contiguous() and leaves[name].data_ptr() % 16 == 8
    stages = _record_launches(monkeypatch)
    shifted = _run(inputs, seq_lens, leaves=leaves, prep="tma")
    assert _preps(stages) == {"prep"}
    aligned = _run(inputs, seq_lens, prep="plain")
    _assert_bitwise(shifted, aligned, OUTPUT_NAMES)


# The operands of the unaligned-operand test: row -> (operand, offset in elements past a 16-byte boundary).  FI main's
# rows (v, q / k 2 bytes, do) run on every route, the prep forward included; q / k 8 bytes (which the plain prep reads
# in place), g, beta and the initial state run on the forwards on wy_a's A.
UNALIGNED_ROWS = {
    "v-2B": ("v", 1),
    "v-8B": ("v", 4),
    "q-2B": ("q", 1),
    "k-2B": ("k", 1),
    "do-2B": ("do", 1),
    "do-8B": ("do", 4),
}
UNALIGNED_A_ROWS = {
    "q-8B": ("q", 4),
    "k-8B": ("k", 4),
    "g-4B": ("g", 1),
    "g-8B": ("g", 2),
    "beta-4B": ("beta", 1),
    "beta-8B": ("beta", 2),
    "h0-4B": ("h0", 1),
    "h0-8B": ("h0", 2),
}
UNALIGNED_CASES = [
    (route, row) for route in ("prep",) + FWD_ROUTES for row in UNALIGNED_ROWS
] + [(route, row) for route in FWD_ROUTES for row in UNALIGNED_A_ROWS]


@pytest.mark.parametrize(
    "route, row", UNALIGNED_CASES, ids=[f"{r}-{w}" for r, w in UNALIGNED_CASES]
)
def test_unaligned_operands_take_an_aligned_copy(
    route, row, device, monkeypatch
) -> None:
    # a contiguous view 2, 4 or 8 bytes past a 16-byte boundary: .contiguous() keeps it where it is, and the 16-byte
    # rows of v / g / beta / do and of the initial state (the split forward's stitch), the 16-byte tensor-map rows of
    # wy_a's q / k and the plain prep's 8-byte q / k loads would fault there (misaligned address); the package copies
    # such an operand to an aligned buffer, so the run gives the bits of the aligned one (the final state and the
    # initial state's gradient included).  "prep": the switches of PREP_FORWARD_OFF off (FI main's forward: the prep,
    # the WY stage, fwd_ho) with FP32 beta, as FI main's test ran; the forwards on wy_a's A ("a" and both split forms)
    # decide their route before any copy (an address never moves a call off it: q / k 8 bytes past the boundary, which
    # the plain prep reads in place, take an aligned copy and stay on the route)
    name, offset = {**UNALIGNED_ROWS, **UNALIGNED_A_ROWS}[row]
    if route == "prep":
        _prep_forward_off(monkeypatch)
        heads, seq_lens = (2, 4), (300, 77)
    else:
        heads, seq_lens = _fwd_route(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=29)
    h0, dht = _state_pair(seq_lens, heads, device, 30)
    ops = {**inputs, "h0": h0}
    t = ops[name]
    buf = torch.empty(t.numel() + 16, dtype=t.dtype, device=device)
    lead = (-buf.data_ptr() // t.element_size()) % (16 // t.element_size())
    shifted = buf[lead + offset : lead + offset + t.numel()].view(t.shape).copy_(t)
    assert (
        shifted.is_contiguous() and shifted.data_ptr() % 16 == offset * t.element_size()
    )
    if name not in ("q", "k"):
        align = cake_chain.ROW_ALIGN_BYTES
    elif route == "prep":
        align = cake_chain.PREP_ALIGN_BYTES
    else:
        align = cake_chain.TMA_ALIGN_BYTES
    assert (
        cake_chain._aligned(t, align) is t
        and cake_chain._aligned(shifted, align) is not shifted
    )
    stages = _record_launches(monkeypatch)

    def run(ops):
        leaves = {
            n: ops[n].detach().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta", "h0")
        }
        cu = _cu(seq_lens)
        o, final = cake_gdn_det(
            leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], scale=1.0 / math.sqrt(HEAD_DIM),
            initial_state=leaves["h0"], output_final_state=True,
            cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
            cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64), use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        torch.autograd.backward((o, final), (ops["do"], dht))
        torch.cuda.synchronize()
        return {
            "o": o.detach(),
            "final_state": final.detach(),
            **{f"d{n}": leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
            "dh0": leaves["h0"].grad,
        }

    def check_route() -> None:
        if route == "prep":
            assert _preps(stages) and "wy_a" not in stages, stages
        else:
            _assert_fwd_route(stages, route)
        stages.clear()

    got = run({**ops, name: shifted})
    check_route()
    want = run(ops)
    check_route()
    _assert_bitwise(got, want, OUTPUT_NAMES + ("final_state", "dh0"))


@pytest.mark.parametrize(
    "preps", [(None, None), ("tma", "plain")], ids=["rule", "tma-vs-plain"]
)
def test_equal_length_batch_matches_the_packed_layout(preps, device) -> None:
    batch, seq_len, heads = 3, 150, (2, 6)
    inputs = _inputs((seq_len,), heads, torch.float32, device, seed=9, batch=batch)
    batched = _run(inputs, None, prep=preps[0])
    packed_inputs = {
        name: t.reshape(1, batch * seq_len, *t.shape[2:]).contiguous()
        for name, t in inputs.items()
    }
    packed = _run(packed_inputs, (seq_len,) * batch, route="host", prep=preps[1])
    for name in OUTPUT_NAMES:
        packed[name] = packed[name].reshape(batched[name].shape)
    _assert_bitwise(batched, packed, OUTPUT_NAMES)


# --------------------------------------------------------------------------- the wavefront pair

WAVEFRONT_PAIR = ("dhu_tc2_wavefront", "tail_g2_bf16_wavefront")
# the split pass 2 of more CTAs than SMs: the pair's scan with the elected release
WAVEFRONT_SPLIT_PAIR = ("dhu_tc2_wavefront_split", "tail_g2_bf16_wavefront")
WAVEFRONT_STAGES = set(WAVEFRONT_PAIR) | set(WAVEFRONT_SPLIT_PAIR)
# a serial plan with the adjoint scan on dhu_tc2
WAVEFRONT_PINS = {"segment_chunks": 0, "dhu_scan": "tc"}


def _wavefront_rule(monkeypatch, *, off: bool = False) -> None:
    """The pair's rule for every value-head count on every architecture (``off``: on none), with fresh layout caches
    (the table's side of the rule is decided when a table is planned)."""
    monkeypatch.setattr(cake_chain, "WAVEFRONT_ARCHS", () if off else ARCHES)
    monkeypatch.setattr(cake_chain, "WAVEFRONT_MIN_V_HEADS", {})
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})


def _record_calls(monkeypatch) -> list[tuple]:
    """(stage, grid, arguments) of every launch from here on; a tensor argument as (shape, stride, dtype)."""
    calls: list[tuple] = []
    launch = cake_backend.launch_stage

    def recording(stage, grid, values, **kwargs):
        calls.append((stage, tuple(grid), tuple((tuple(v.shape), v.stride(), v.dtype) if isinstance(v, torch.Tensor)
                                               else v for v in values)))  # fmt: skip
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return calls


def test_wavefront_order_follows_the_scan() -> None:
    # step k: the k-th chunk from the end of every segment that has one, segments in table order
    assert cake_chain.wavefront_order([0, 3, 4, 7]) == [2, 3, 6, 1, 5, 0, 4]
    assert cake_chain.wavefront_order([0, 0, 2, 2]) == [1, 0]
    assert cake_chain.wavefront_order([0]) == []
    # the host-built order equals the order of the table's own adjoint segments (longest first, split, empty sequences)
    for seq_lens, plans, lpt in (
        ((300, 77, 640, 1000), (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 0)), True),
        ((300, 0, 1000, 77), (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 0)), True),
        ((1500, 77, 640), (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 4)), True),
        ((1500, 77, 640), (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 4)), False),
    ):  # fmt: skip
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans, lpt=lpt)
        segs = meta.bwd_segments
        got = cake_chain._wavefront_walk(meta.seq_lens, meta.order, segs.segment_len)
        assert got == cake_chain.wavefront_order(segs.seg_chunk_start.tolist()), (
            seq_lens,
            lpt,
        )
        assert sorted(got) == list(range(meta.num_chunks))


# (sequences, HV, adjoint segment chunks) -> whether the table's side of the rule holds on (sm_100a 148 SMs, sm_103a
# 152, sm_107a 212 / 216): 2 x segments x HV <= SMs, and on sm_100a HV >= 32 (scan dhu_tc2_wavefront); on sm_100a and
# sm_103a also a split adjoint plan of more than SMs pass-2 CTAs (segments x HV) at any HV with tokens x HV >= 3 x 2^17
# (scan dhu_tc2_wavefront_split); on sm_107a only the first side (not in WAVEFRONT_SPLIT_ARCHS)
WAVEFRONT_TABLE_ROWS = {
    "r01_4x4096_hv16": ((4096,) * 4, 16, 0, (False, True, True, True)),
    "r07_1x16384_hv16": ((16384,), 16, 0, (False, True, True, True)),
    "r10_1x16384_hv32": ((16384,), 32, 0, (True, True, True, True)),
    "r03_8x4096_hv16": ((4096,) * 8, 16, 0, (False, False, False, False)),
    "3x4096_hv32": ((4096,) * 3, 32, 0, (False, False, True, True)),
    # split: 13 segments x HV 16 = 208 pass-2 CTAs, 30100 tokens x 16 = 481600 (above the work floor)
    "r04_7x4.3k_hv16_split46": ((4199, 2781, 4282, 4448, 4723, 5804, 3863), 16, 46, (True, True, False, False)),
    # split: 19 x 16 = 304 CTAs (the sm_103a plan of the same layout)
    "r04_7x4.3k_hv16_split31": ((4199, 2781, 4282, 4448, 4723, 5804, 3863), 16, 31, (True, True, False, False)),
    # split: 13 x 16 = 208 CTAs, 12750 tokens x 16 = 204000 (below the work floor: host-bound)
    "r02_3x4.25k_hv16_split17": ((5406, 3448, 3896), 16, 17, (False, False, False, False)),
    # split: 12 x 32 = 384 CTAs at the work floor (12288 x 32) and one chunk below it (12224 x 32)
    "1x12288_hv32_split16": ((12288,), 32, 16, (True, True, False, False)),
    "1x12224_hv32_split16": ((12224,), 32, 16, (False, False, False, False)),
    # split: 4 x 16 = 64 CTAs (sm_100a: below the SMs and HV < 32)
    "1x16384_hv16_split64": ((16384,), 16, 64, (False, True, True, True)),
    # split: 8 x 32 = 256 CTAs
    "1x16384_hv32_split32": ((16384,), 32, 32, (True, True, False, False)),
    # split: 7 x 32 = 224 CTAs (the sm_103a plan of r10)
    "r10_1x16384_hv32_split37": ((16384,), 32, 37, (True, True, False, False)),
    # split: 2 x 32 = 64 CTAs, at most half of every device's SMs
    "1x16384_hv32_split128": ((16384,), 32, 128, (True, True, True, True)),
}  # fmt: skip
WAVEFRONT_DEVICES = (
    ("sm_100a", 148),
    ("sm_103a", 152),
    ("sm_107a", 212),
    ("sm_107a", 216),
)


@pytest.mark.parametrize("row", sorted(WAVEFRONT_TABLE_ROWS))
def test_wavefront_rule_per_architecture(row, monkeypatch) -> None:
    assert cake_chain.WAVEFRONT_ARCHS == ("sm_100a", "sm_103a", "sm_107a")
    assert cake_chain.WAVEFRONT_MIN_V_HEADS == {"sm_100a": 32}
    assert cake_chain.WAVEFRONT_SPLIT_ARCHS == ("sm_100a", "sm_103a")
    assert cake_chain.WAVEFRONT_SPLIT_MIN_TOKEN_HEADS == 393216
    assert cake_chain.wavefront_engaged(4, 8, 148)
    assert not cake_chain.wavefront_engaged(10, 8, 148)
    assert not cake_chain.wavefront_engaged(38, 2, 148)
    assert cake_chain.wavefront_engaged(3, 32, 212)
    assert not cake_chain.wavefront_engaged(4, 32, 216)
    assert cake_chain.wavefront_split_engaged(13, 16, 148, 30100)
    assert cake_chain.wavefront_split_engaged(12, 32, 148, 12288)
    assert not cake_chain.wavefront_split_engaged(12, 32, 148, 12287)
    assert not cake_chain.wavefront_split_engaged(13, 16, 148, 12750)
    assert cake_chain.wavefront_split_engaged(7, 32, 152, 16384)
    assert cake_chain.wavefront_split_engaged(19, 16, 152, 30100)
    assert not cake_chain.wavefront_split_engaged(4, 32, 152, 16384)
    assert not cake_chain.wavefront_split_engaged(37, 4, 148, 1 << 20)
    assert not cake_chain.wavefront_split_engaged(4, 16, 148, 1 << 20)
    seq_lens, num_v_heads, segment_chunks, want = WAVEFRONT_TABLE_ROWS[row]
    plans = (
        cake_chain.ScanPlan((32, 2), 0),
        cake_chain.ScanPlan((64, 1), segment_chunks),
    )
    meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans, lpt=True)
    got = []
    for arch, sm_count in WAVEFRONT_DEVICES:
        monkeypatch.setattr(cake_backend, "device_arch", lambda _index, arch=arch: arch)
        monkeypatch.setattr(
            cake_chain, "_device_props", lambda _index, n=sm_count: (n, 232448, 10)
        )
        scan = cake_chain._wavefront_table(meta, num_v_heads, 0)
        got.append(scan is not None)
        # the split side (more pass-2 CTAs than SMs) runs the elected-release scan, the occupancy side the other
        rounds = meta.bwd_segments.num_segments * num_v_heads > sm_count
        assert scan in (
            None,
            "dhu_tc2_wavefront_split" if rounds else "dhu_tc2_wavefront",
        ), (row, arch, scan)
        # the device route has no host layout: never
        device_meta = dataclasses.replace(meta, seq_lens=None)
        assert cake_chain._wavefront_table(device_meta, num_v_heads, 0) is None
    assert tuple(got) == want, (row, got)


def _poison_adjoint_workspaces(device) -> int:
    """NaN into every pooled ``dh`` / ``dv2`` workspace of the caller's stream (a tail tile read before the scan
    stores it would carry it into the gradients); returns the buffers poisoned."""
    table = cake_chain._POOLED.table(cake_chain._device_index(device))
    count = 0
    for role in ("b_dh", "b_dv2"):
        for buffer in table.get(role, {}).values():
            buffer.fill_(float("nan"))
            count += 1
    return count


# (layout, pins, heads, scan): a serial plan on dhu_tc2 (also with an empty sequence) and a split (64, 1) plan with its
# pass 2 on dhu_tc2, value-head group of two, BF16 beta: 4 / 4 / 10 segments x HV 4 take at most half of any exported
# device (dhu_tc2_wavefront); a split pass 2 of 50 segments x HV 32 = 1600 CTAs over 12477 tokens (more CTAs than
# SMs, above the work floor: dhu_tc2_wavefront_split, on the architectures of WAVEFRONT_SPLIT_ARCHS)
WAVEFRONT_ROWS = {
    "serial": ((300, 77, 640, 1000), WAVEFRONT_PINS, (2, 4), "dhu_tc2_wavefront"),
    "serial-empty": ((300, 0, 1000, 77), WAVEFRONT_PINS, (2, 4), "dhu_tc2_wavefront"),
    "split": ((1500, 77, 640), {"scan_config": (64, 1), "segment_chunks": 4, "dhu_scan": "tc"}, (2, 4),
              "dhu_tc2_wavefront"),
    "split-rounds": ((6000, 77, 6400), {"scan_config": (64, 1), "segment_chunks": 4, "dhu_scan": "tc"}, (16, 32),
                     "dhu_tc2_wavefront_split"),
}  # fmt: skip


@pytest.mark.parametrize("row", sorted(WAVEFRONT_ROWS))
def test_wavefront_pair_matches_the_plain_launches(row, device, monkeypatch) -> None:
    # the pair against the separate dhu_tc2 / tail_g2_bf16 launches, with the initial state, its gradient, the final
    # state and its gradient: bitwise, also with dh / dv2 poisoned between calls; the tail on the side stream
    seq_lens, pins, heads, scan = WAVEFRONT_ROWS[row]
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if (
        scan == "dhu_tc2_wavefront_split"
        and arch not in cake_chain.WAVEFRONT_SPLIT_ARCHS
    ):
        pytest.skip(f"the split side of the rule does not run on {arch}")
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=401)
    gen = torch.Generator().manual_seed(409)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    names = OUTPUT_NAMES + ("final_state", "dh0")
    kwargs = dict(initial_state=h0, output_final_state=True, dht=dht, **pins)
    _wavefront_rule(monkeypatch, off=True)
    stages = _record_launches(monkeypatch)
    plain = _run(inputs, seq_lens, **kwargs)
    assert not WAVEFRONT_STAGES & set(stages) and "dhu_tc2" in stages, stages
    _wavefront_rule(monkeypatch)
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    for poison in (False, True):
        if poison:
            assert _poison_adjoint_workspaces(device) >= 2
        start = len(launches)
        got = _run(inputs, seq_lens, **kwargs)
        mine = [s for s, _ in launches[start:]]
        assert mine.count(scan) == 1 and mine.count("tail_g2_bf16_wavefront") == 1, mine
        assert not (
            {"dhu_tc2", "tail_g2_bf16"}
            | WAVEFRONT_STAGES - {scan, "tail_g2_bf16_wavefront"}
        ) & set(mine), mine
        side = [s for s, stream in launches[start:] if stream != main]
        assert side == ["tail_g2_bf16_wavefront"], launches[start:]
        _assert_bitwise(got, plain, names)


def test_wavefront_pair_repeats_bitwise(device, monkeypatch) -> None:
    seq_lens, pins, heads, _scan = WAVEFRONT_ROWS["serial"]
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=419)
    _wavefront_rule(monkeypatch, off=True)
    want = _run(inputs, seq_lens, **pins)
    _wavefront_rule(monkeypatch)
    stages = _record_launches(monkeypatch)
    for _ in range(20):
        _assert_bitwise(_run(inputs, seq_lens, **pins), want, OUTPUT_NAMES)
    assert stages.count("tail_g2_bf16_wavefront") == 20, stages


# (layout, heads, pins, architectures that engage): the shipped rule on the device's architecture.  A serial plan on
# dhu_tc2: HV 4 engages on sm_103a, not on sm_100a (HV < 32); HV 32 over two sequences (128 scan CTAs) engages on
# sm_100a and sm_103a.  A split (64, 1) plan with its pass 2 on dhu_tc2 (24 + 2 + 10 chunks in segments of 4 = 10
# segments): HV 4 (40 CTAs, below the SMs) engages on sm_103a only; HV 32 (320 CTAs, more than the SMs) nowhere, as
# 2217 tokens x 32 stay below the split side's work floor; over 94 + 2 + 100 chunks (50 segments, 1600 CTAs; 12477
# tokens x 32, above the floor) on sm_100a and sm_103a, on dhu_tc2_wavefront_split.  sm_107a (208 to 216 SMs) engages
# every scan of at most half its SMs (HV 4 and HV 32 serial, HV 4 split) and no split pass 2 of more CTAs than SMs
WAVEFRONT_SPLIT_PINS = {"scan_config": (64, 1), "segment_chunks": 4, "dhu_scan": "tc"}
WAVEFRONT_DEFAULT_ROWS = {
    "hv4": ((300, 77, 640), (2, 4), WAVEFRONT_PINS, ("sm_103a", "sm_107a"), WAVEFRONT_PAIR),
    "hv32": ((300, 640), (16, 32), WAVEFRONT_PINS, ("sm_100a", "sm_103a", "sm_107a"), WAVEFRONT_PAIR),
    "split-hv4": ((1500, 77, 640), (2, 4), WAVEFRONT_SPLIT_PINS, ("sm_103a", "sm_107a"), WAVEFRONT_PAIR),
    "split-hv32-small": ((1500, 77, 640), (16, 32), WAVEFRONT_SPLIT_PINS, (), WAVEFRONT_PAIR),
    "split-hv32": ((6000, 77, 6400), (16, 32), WAVEFRONT_SPLIT_PINS, ("sm_100a", "sm_103a"), WAVEFRONT_SPLIT_PAIR),
}  # fmt: skip


@pytest.mark.parametrize("row", sorted(WAVEFRONT_DEFAULT_ROWS))
def test_wavefront_shipped_rule_on_the_device(row, device, monkeypatch) -> None:
    # engaged: the pair replaces the separate launches with the same bits; not engaged: exactly the launches (stages,
    # grids, argument shapes, strides and scalars) of the rule turned off
    seq_lens, heads, pins, arches, pair = WAVEFRONT_DEFAULT_ROWS[row]
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    engaged = arch in arches
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=433)
    calls = _record_calls(monkeypatch)
    got = _run(inputs, seq_lens, **pins)
    n = len(calls)
    stages = {c[0] for c in calls[:n]}
    assert (set(pair) <= stages) is engaged, (arch, row)
    assert not (WAVEFRONT_STAGES - set(pair)) & stages, (arch, row)
    _wavefront_rule(monkeypatch, off=True)
    off = _run(inputs, seq_lens, **pins)
    assert not WAVEFRONT_STAGES & {c[0] for c in calls[n:]}
    if not engaged:
        assert calls[:n] == calls[n:], (arch, row)
    _assert_bitwise(got, off, OUTPUT_NAMES)


# The serial-scan rule on its campaign rows, the planner told the row's SM count: the adjoint scan is one launch over
# the sequences (dhu_tc2, or the wavefront pair's scan where it takes at most half the SMs) and no backward pre-scan;
# against the rule turned off (the planner's split plan) the forward launches and o / dq are the same, the other
# gradients agree within the accuracy gate, and two runs are bitwise.
SERIAL_SCAN_DEVICE_ROWS = {
    "sm_107a": (216, (((4096,) * 8, (8, 16)), ((4096,) * 4, (16, 32)))),
}
_BACKWARD_STAGES = ("dhu", "dv_local", "tail", "dqkwg", "wy_bwd", "finalize", "stitch")


@pytest.mark.parametrize("case", (0, 1))
def test_serial_scan_rule_on_the_device(case, device, monkeypatch) -> None:
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if arch not in SERIAL_SCAN_DEVICE_ROWS or case >= len(
        SERIAL_SCAN_DEVICE_ROWS[arch][1]
    ):
        pytest.skip(f"no serial-scan row {case} on {arch}")
    sm_count, rows = SERIAL_SCAN_DEVICE_ROWS[arch]
    seq_lens, heads = rows[case]
    _small_device(monkeypatch, device, sm_count)
    # the serial-scan rule's tables alone: the tc-priced table (TC_PLAN) also makes these layouts serial
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=581)
    calls = _record_calls(monkeypatch)
    got = _run(inputs, seq_lens)
    n = len(calls)
    _assert_bitwise(got, _run(inputs, seq_lens), OUTPUT_NAMES)
    on = calls[:n]
    assert not any(c[0].startswith("dhu_pre") for c in on), on
    scans = [c for c in on if c[0].startswith("dhu_tc2")]
    assert len(scans) == 1 and scans[0][0] in ("dhu_tc2", "dhu_tc2_wavefront"), scans
    assert scans[0][1][:2] == (len(seq_lens), heads[1]), scans
    monkeypatch.setattr(
        cake_chain, "_serial_adjoint_plan", lambda _seq_lens, plans, *_args: plans
    )
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    m = len(calls)
    off = _run(inputs, seq_lens)
    assert any(c[0].startswith("dhu_pre") for c in calls[m:]), calls[m:]

    def forward(cs):
        return [c for c in cs if not c[0].startswith(_BACKWARD_STAGES)]

    assert forward(on) == forward(calls[m:])
    _assert_bitwise(got, off, ("o", "dq"))
    for name in ("dk", "dv", "dg", "dbeta"):
        metrics = _metrics(got[name], off[name])
        assert metrics["finite"] and metrics["rel_rms"] <= REL_RMS_MAX, (name, metrics)
        assert metrics["max_abs"] <= MAX_ABS_REL * metrics["max_ref"], (name, metrics)


# rows the wavefront rule keeps on the separate launches although every architecture and value-head count is allowed:
# FP32 beta (tail_g2), a value-head group of three (tail_g3), the register adjoint scan, the device route, and a scan
# above half the SMs (the planner on 10 SMs)
WAVEFRONT_OFF_ROWS = {
    "fp32beta": ((300, 77, 640, 1000), (2, 4), torch.float32, "host", WAVEFRONT_PINS, None),
    "g3": ((300, 77, 640, 1000), (2, 6), torch.bfloat16, "host", WAVEFRONT_PINS, None),
    "reg": ((300, 77, 640, 1000), (2, 4), torch.bfloat16, "host", {"segment_chunks": 0, "dhu_scan": "reg"}, None),
    "device": ((300, 77, 640, 1000), (2, 4), torch.bfloat16, "device", WAVEFRONT_PINS, None),
    "half-sms": ((300, 77, 640, 1000), (2, 4), torch.bfloat16, "host", {**WAVEFRONT_PINS, "streams": 1}, 10),
}  # fmt: skip


@pytest.mark.parametrize("row", sorted(WAVEFRONT_OFF_ROWS))
def test_wavefront_rule_keeps_the_other_launches(row, device, monkeypatch) -> None:
    # where the rule does not engage, the call launches exactly what it launches with the rule turned off: the same
    # stages, grids and argument shapes, strides and scalars, and the same bits
    seq_lens, heads, beta_dtype, route, pins, sm_count = WAVEFRONT_OFF_ROWS[row]
    if sm_count is not None:
        _small_device(monkeypatch, device, sm_count)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=421)
    _wavefront_rule(monkeypatch)
    calls = _record_calls(monkeypatch)
    default = _run(inputs, seq_lens, route=route, **pins)
    n = len(calls)
    _wavefront_rule(monkeypatch, off=True)
    off = _run(inputs, seq_lens, route=route, **pins)
    assert calls[:n] == calls[n:], row
    assert not WAVEFRONT_STAGES & {c[0] for c in calls}, row
    _assert_bitwise(default, off, OUTPUT_NAMES)


def test_wavefront_pair_under_cuda_graph_capture(device, monkeypatch) -> None:
    # a graph does not order its branches: the capture records the separate launches; its replays equal the eager
    # calls (which run the pair) bitwise
    seq_lens, pins, heads, _scan = WAVEFRONT_ROWS["serial"]
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=431)
    _wavefront_rule(monkeypatch)
    stages = _record_launches(monkeypatch)
    eager = _run(inputs, seq_lens, **pins)
    assert set(WAVEFRONT_PAIR) <= set(stages), stages
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = cake_gdn_det(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                          scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                          cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True,
                                          **pins)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
        return o

    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream(device).wait_stream(warmup)
    start = len(stages)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = step()
    captured = stages[start:]
    assert not WAVEFRONT_STAGES & set(captured), captured
    assert {"dhu_tc2", "tail_g2_bf16"} <= set(captured), captured
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        got = {
            "o": o,
            **{"d" + name: leaves[name].grad for name in ("q", "k", "v", "g", "beta")},
        }
        _assert_bitwise(got, eager, OUTPUT_NAMES)


# --------------------------------------------------------------------------- #
# cached device tables and workspaces across streams and CUDA graph captures
# --------------------------------------------------------------------------- #
# row: (cache, sequence lengths, heads, pins, setup) of a call that reads that cached table; cache "chunk" is the chunk
# tables (through _META_FRONT, or _HOST_META alone under host_cache=False), "walk" the identity walk, "wavefront" the
# pair's tile order, "forward-split" the split forward's tables (_FORWARD_SPLITS: through _META_FRONT, or the cache
# alone under host_cache=False); setup "wavefront" turns the pair's rule on, "forward-split" routes the layout through
# the split forward (FP32 beta, the shape class lowered to FWD_SPLIT_TEST_MIN_PIECE; sm_100a / sm_103a, skipped
# elsewhere), "forward-split-a" through its form on wy_a's A on both (_split_form), an int is the SMs the planner sees
# (two-stream groups)
TABLE_CACHE_ROWS = {
    "chunk": ("chunk", (300, 77, 640), (2, 4), {}, None),
    "chunk-uncached-front": ("chunk", (300, 77, 640), (2, 4), {"host_cache": False}, None),
    "chunk-two-streams": ("chunk", STREAM_LAYOUT, (2, 4), STREAM_PINS, 6),
    "walk": ("walk", (300, 77, 640), (2, 4), {"lpt": False}, None),
    "wavefront": ("wavefront", WAVEFRONT_ROWS["serial"][0], WAVEFRONT_ROWS["serial"][2], WAVEFRONT_PINS, "wavefront"),
    "forward-split": ("forward-split", FWD_SPLIT_CASES["g2"][1], FWD_SPLIT_CASES["g2"][0], {}, "forward-split"),
    "forward-split-uncached-front": ("forward-split", FWD_SPLIT_CASES["g2"][1], FWD_SPLIT_CASES["g2"][0],
                                     {"host_cache": False}, "forward-split"),
}  # fmt: skip
# layout -> (sequence lengths, heads, setup) of the eviction and capture tests; their calls take lpt=False, so the
# prep forward's fwd_ho reads the cached identity walk ("chunk-walk") and the split forward its cached split tables
# (int64 indices of first pieces and their sequences; "forward-split-all": every sequence splits, first pieces only)
TABLE_LAYOUTS = {
    "chunk-walk": ((300, 77, 640), (2, 4), None),
    "forward-split": (FWD_SPLIT_CASES["g2"][1], FWD_SPLIT_CASES["g2"][0], "forward-split"),
    "forward-split-all": (FWD_SPLIT_CASES["g3-all"][1], FWD_SPLIT_CASES["g3-all"][0], "forward-split-a"),
}  # fmt: skip
# the setups that route a cached-table layout through the split forward
FORWARD_SPLIT_SETUPS = ("forward-split", "forward-split-a")


def _table_setup(monkeypatch, device, setup) -> None:
    """The device view of a :data:`TABLE_CACHE_ROWS` / :data:`TABLE_LAYOUTS` setup."""
    if setup == "wavefront":
        _wavefront_rule(monkeypatch)
    elif setup == "forward-split":
        _fwd_split_or_skip(device)
        _fwd_split_test_class(monkeypatch)
    elif setup == "forward-split-a":
        _split_form("split_a", device, monkeypatch)
    elif setup is not None:
        _small_device(monkeypatch, device, setup)


def _table_inputs(seq_lens, heads, setup, device, seed) -> dict:
    """The operands of a cached-table call: FP32 beta under a setup of FORWARD_SPLIT_SETUPS (a condition of the split
    forward), else BF16 (the prep forward)."""
    beta_dtype = torch.float32 if setup in FORWARD_SPLIT_SETUPS else torch.bfloat16
    return _inputs(seq_lens, heads, beta_dtype, device, seed=seed)


def _check_table_route(stages, setup) -> None:
    """Under a setup of FORWARD_SPLIT_SETUPS, the call whose launches are ``stages`` ran the split forward (on wy_a's
    A under "forward-split-a"); else the prep forward."""
    if setup in FORWARD_SPLIT_SETUPS:
        assert _forward_stages(stages) == _split_stages(), stages
        assert setup != "forward-split-a" or _split_a(), stages
    else:
        assert _preps(stages) and "wy_a" not in stages, stages


def _fresh_table_caches(monkeypatch) -> None:
    """Empty device-table caches (the chunk tables, the split forward's tables, the identity walks, the wavefront
    orders, the captured tables)."""
    for name in ("_META_FRONT", "_HOST_META", "_FORWARD_SPLITS", "_WALKS", "_GRAPH_TABLES"):  # fmt: skip
        monkeypatch.setattr(cake_chain, name, {})
    cake_chain._wavefront_order_tensor.cache_clear()


def _table_cache_size(cache: str) -> int:
    if cache == "chunk":
        return len(cake_chain._HOST_META)
    if cache == "walk":
        return len(cake_chain._WALKS)
    if cache == "forward-split":
        return len(cake_chain._FORWARD_SPLITS)
    return cake_chain._wavefront_order_tensor.cache_info().currsize


def _drop_table_cache(monkeypatch, cache: str) -> None:
    """Empty ``cache`` (and the layout front, which holds the chunk tables and the split forward's tables), so the next
    call builds that table again."""
    if cache == "chunk":
        monkeypatch.setattr(cake_chain, "_META_FRONT", {})
        monkeypatch.setattr(cake_chain, "_HOST_META", {})
    elif cache == "walk":
        monkeypatch.setattr(cake_chain, "_WALKS", {})
    elif cache == "forward-split":
        monkeypatch.setattr(cake_chain, "_META_FRONT", {})
        monkeypatch.setattr(cake_chain, "_FORWARD_SPLITS", {})
    else:
        cake_chain._wavefront_order_tensor.cache_clear()


def _cached_tables(kinds=("chunk", "forward-split", "walk")) -> list[torch.Tensor]:
    """One view of every storage of the cached device tables of ``kinds`` (none of zero bytes): the chunk tables, the
    split forward's two allocations (from _FORWARD_SPLITS and from the ChunkMetas of the caches, which carry them) and
    the identity walks."""
    metas = [*cake_chain._HOST_META.values(), *cake_chain._META_FRONT.values()]
    tables: list[torch.Tensor] = []
    if "chunk" in kinds:
        tables += [m.chunk_start for m in metas]
    if "forward-split" in kinds:
        splits = [*cake_chain._FORWARD_SPLITS.values()]
        splits += [m.fwd_split for m in metas if m.fwd_split is not None]
        for split in splits:
            tables += [split.seg_chunk_start, split.first_index]
    if "walk" in kinds:
        tables += [walk for walk, _ready in cake_chain._WALKS.values()]
    unique = {t.untyped_storage().data_ptr(): t for t in tables}
    return [t for t in unique.values() if t.untyped_storage().nbytes()]


def _storage_addresses(tables) -> list[int]:
    return sorted({t.untyped_storage().data_ptr() for t in tables})


def _drop_cached_tables() -> None:
    """Drop every reference the package keeps to a cached device table."""
    for name in ("_META_FRONT", "_HOST_META", "_FORWARD_SPLITS", "_WALKS"):
        getattr(cake_chain, name).clear()


def _issue(inputs, cu_seqlens, cu_seqlens_cpu, *, backward=True, **pins) -> dict:
    """One call on the current stream without a synchronization (``cu_seqlens`` already on the device): the output
    and, with ``backward``, the gradients."""
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(backward)
        for name in ("q", "k", "v", "g", "beta")
    }
    o, _ = cake_gdn_det(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                        scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                        use_qk_l2norm_in_kernel=True, **pins)  # fmt: skip
    if not backward:
        return {"o": o}
    (o.float() * inputs["do"].float()).sum().backward()
    return {
        "o": o.detach(),
        **{"d" + name: leaves[name].grad for name in ("q", "k", "v", "g", "beta")},
    }


def _cu_tensors(seq_lens, device):
    cu = _cu(seq_lens)
    return torch.tensor(cu, dtype=torch.int64, device=device), torch.tensor(
        cu, dtype=torch.int64
    )


def _overlapping_streams(device):
    """Two streams whose work overlaps: a launch on the second completes while the first sleeps.  Two streams can share
    one hardware queue, which runs their work in issue order and hides the ordering under test; such a pair is replaced
    by the next streams of the pool."""
    current = torch.cuda.current_stream(device)
    for _ in range(16):
        s1, s2 = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
        s1.wait_stream(current)
        s2.wait_stream(current)
        woke, probe = torch.cuda.Event(), torch.cuda.Event()
        with torch.cuda.stream(s1):
            torch.cuda._sleep(100_000_000)
            woke.record(s1)
        with torch.cuda.stream(s2):
            torch.ones(1, device=device).add_(1)
            probe.record(s2)
        probe.synchronize()
        overlap = not woke.query()
        torch.cuda.synchronize()
        if overlap:
            return s1, s2
    pytest.skip("no two streams of the pool run concurrently on this device")


@pytest.mark.parametrize("row", sorted(TABLE_CACHE_ROWS))
def test_a_cached_table_is_read_after_its_write_on_another_stream(
    row, device, monkeypatch
) -> None:
    # Stream s1 sleeps, then its call rebuilds one cached table (its write queued behind the sleep); a call on stream
    # s2 then hits that table.  s2 can run while s1 sleeps (a probe launch on s2 completes), but the s2 call must wait
    # for the write: its work is still queued, and both calls give the bits of a call on the written table.
    cache, seq_lens, heads, pins, setup = TABLE_CACHE_ROWS[row]
    _table_setup(monkeypatch, device, setup)
    _fresh_table_caches(monkeypatch)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=113)
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, **pins)
    _check_table_route(stages, setup)
    assert _table_cache_size(cache) == 1
    if isinstance(setup, int):
        assert any(
            m.fwd_groups is not None or m.bwd_groups is not None
            for m in cake_chain._META_FRONT.values()
        )
    _drop_table_cache(monkeypatch, cache)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    s1, s2 = _overlapping_streams(device)
    woke, probe, done = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    with torch.cuda.stream(s1):
        torch.cuda._sleep(2_000_000_000)
        woke.record(s1)
        first = _issue(inputs, cu_seqlens, cu_seqlens_cpu, **pins)
    assert _table_cache_size(cache) == 1  # s1 rebuilt the table
    with torch.cuda.stream(s2):
        torch.ones(1, device=device).add_(1)
        probe.record(s2)
        second = _issue(inputs, cu_seqlens, cu_seqlens_cpu, **pins)
        done.record(s2)
    probe.synchronize()
    time.sleep(0.1)  # time for the s2 call to finish if it does not wait for the write
    asleep, queued = not woke.query(), not done.query()
    torch.cuda.synchronize()
    assert asleep, "s1 woke before the check (s2 did not run while s1 slept)"
    assert queued, "the call on s2 ran before the table write on s1"
    _assert_bitwise(first, want, OUTPUT_NAMES)
    _assert_bitwise(second, want, OUTPUT_NAMES)


def _block_states(addresses) -> dict:
    """The caching allocator's state of the block holding each address (``active_allocated``, ``active_pending_free``
    while a stream recorded on a freed tensor may still use it, ``inactive`` once free)."""
    states = {}
    for segment in torch.cuda.memory_snapshot():
        for block in segment["blocks"]:
            for address in addresses:
                if block["address"] <= address < block["address"] + block["size"]:
                    states[address] = block["state"]
    return states


@pytest.mark.parametrize("layout", sorted(TABLE_LAYOUTS))
def test_an_evicted_table_waits_for_reads_on_another_stream(
    layout, device, monkeypatch
) -> None:
    # A call on stream s2, queued behind a sleep, hits tables built on stream s1; the caches then drop every table.
    # The caching allocator must keep each table's block until s2's launches have read it (so no later allocation can
    # overwrite it), and free it after them: on the prep forward the chunk tables and the identity walk, on the split
    # forward the chunk tables and the two allocations of the split forward's tables.
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("reads the state of the native caching allocator's blocks")
    seq_lens, heads, setup = TABLE_LAYOUTS[layout]
    _table_setup(monkeypatch, device, setup)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=137)
    _fresh_table_caches(monkeypatch)
    s1, s2 = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    s1.wait_stream(torch.cuda.current_stream(device))
    stages = _record_launches(monkeypatch)
    with torch.cuda.stream(s1):
        want = _run(inputs, seq_lens, lpt=False)
    _check_table_route(stages, setup)
    tables = _cached_tables()
    # the chunk tables and the identity walk; the chunk tables and the split forward's int32 tables and int64 indices
    assert len(tables) == (3 if setup in FORWARD_SPLIT_SETUPS else 2), len(tables)
    addresses = _storage_addresses(tables)
    del tables
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    s2.wait_stream(s1)
    with torch.cuda.stream(s2):
        torch.cuda._sleep(2_000_000_000)
        got = _issue(inputs, cu_seqlens, cu_seqlens_cpu, lpt=False)
    _drop_cached_tables()
    gc.collect()
    dropped = _block_states(addresses)
    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        torch.empty(
            1, device=device
        )  # an allocation frees the blocks whose recorded uses are done
    read = _block_states(addresses)
    assert dropped == dict.fromkeys(addresses, "active_pending_free"), dropped
    assert read == dict.fromkeys(addresses, "inactive"), read
    _assert_bitwise(got, want, OUTPUT_NAMES)


@pytest.mark.parametrize("layout", sorted(TABLE_LAYOUTS))
def test_tables_built_inside_a_capture_are_not_cached(
    layout, device, monkeypatch
) -> None:
    # A layout's first call inside a CUDA graph capture writes its tables (the chunk tables, the identity walk, the
    # split forward's tables) only when the graph replays: the capture caches none of them, so an eager call of that
    # layout before any replay builds its own; the replays match too.
    seq_lens, heads, setup = TABLE_LAYOUTS[layout]
    _table_setup(monkeypatch, device, setup)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=127)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    _fresh_table_caches(monkeypatch)
    stages = _record_launches(monkeypatch)
    want = _issue(inputs, cu_seqlens, cu_seqlens_cpu, backward=False, lpt=False)["o"]
    torch.cuda.synchronize()
    _check_table_route(stages, setup)
    # the stages stay loaded; every table of the layout is now built inside the capture
    _fresh_table_caches(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = _issue(inputs, cu_seqlens, cu_seqlens_cpu, backward=False, lpt=False)["o"]
    assert not (
        cake_chain._META_FRONT
        or cake_chain._HOST_META
        or cake_chain._FORWARD_SPLITS
        or cake_chain._WALKS
    )
    eager = _issue(inputs, cu_seqlens, cu_seqlens_cpu, backward=False, lpt=False)["o"]
    torch.cuda.synchronize()
    assert torch.equal(eager, want)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(o, want)


@pytest.mark.parametrize("layout", sorted(TABLE_LAYOUTS))
def test_a_capture_keeps_the_cached_tables_it_reads(
    layout, device, monkeypatch
) -> None:
    # A capture of a warmed-up layout reads its cached tables (the chunk tables and the identity walk of the prep
    # forward; the chunk tables and both allocations of the split forward's tables).  Evicting every cached table
    # afterwards and reusing memory of their sizes (zero-filled) must leave the replays unchanged: the capture keeps
    # the tables it reads.
    seq_lens, heads, setup = TABLE_LAYOUTS[layout]
    _table_setup(monkeypatch, device, setup)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=131)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    _fresh_table_caches(monkeypatch)
    stages = _record_launches(monkeypatch)
    want = _issue(inputs, cu_seqlens, cu_seqlens_cpu, backward=False, lpt=False)["o"]
    torch.cuda.synchronize()
    _check_table_route(stages, setup)
    tables = _cached_tables()
    addresses = _storage_addresses(tables)
    own = _storage_addresses(
        _cached_tables(
            ("chunk", "forward-split")
            if setup in FORWARD_SPLIT_SETUPS
            else ("chunk", "walk")
        )
    )
    assert len(own) == (3 if setup in FORWARD_SPLIT_SETUPS else 2), own
    sizes = [t.untyped_storage().nbytes() for t in tables]
    del tables
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = _issue(inputs, cu_seqlens, cu_seqlens_cpu, backward=False, lpt=False)["o"]
    # the tables the capture reads are kept for the process, nothing but cached tables is
    assert set(own) <= set(cake_chain._GRAPH_TABLES) <= set(addresses), (
        own,
        sorted(cake_chain._GRAPH_TABLES),
    )
    _drop_cached_tables()
    gc.collect()
    junk = [
        torch.zeros(n // 4, dtype=torch.int32, device=device)
        for n in sizes
        for _ in range(4)
    ]
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(o, want)
    del junk


def test_a_capture_takes_its_own_workspaces(device, monkeypatch) -> None:
    # Eager calls on stream s fill s's pooled workspaces; a capture on s must not record them: once the pool drops them
    # and s reuses their memory (zero-filled), the replays write only the graph's own memory, with the eager bits.
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=139)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    _fresh_table_caches(monkeypatch)
    monkeypatch.setattr(cake_chain, "_POOLED", cake_chain.Workspaces())
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = cake_gdn_det(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                            scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                            use_qk_l2norm_in_kernel=True)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
        return o

    with torch.cuda.stream(s):
        for _ in range(2):
            o = step()
        want = {
            "o": o.detach().clone(),
            **{"d" + n: leaves[n].grad.clone() for n in ("q", "k", "v", "g", "beta")},
        }
    torch.cuda.synchronize()
    pooled = [
        b
        for table in cake_chain._POOLED.tables.values()
        for role in table.values()
        for b in role.values()
    ]
    assert pooled
    shapes = [(tuple(b.shape), b.dtype) for b in pooled]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        o = step()
    del pooled
    cake_chain._POOLED.tables.clear()
    gc.collect()
    with torch.cuda.stream(s):
        junk = [
            torch.zeros(shape, dtype=dtype, device=device) for shape, dtype in shapes
        ]
    torch.cuda.synchronize()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        got = {
            "o": o,
            **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
        }
        _assert_bitwise(got, want, OUTPUT_NAMES)
    assert not any(j.any() for j in junk)


def test_wavefront_gradients_accumulate_into_a_nonzero_grad(
    device, monkeypatch
) -> None:
    _wavefront_rule(monkeypatch)
    stages = _record_launches(monkeypatch)
    _check_accumulation(
        (300, 77, 640), (2, 4), device, pins=WAVEFRONT_PINS, fresh_pins=WAVEFRONT_PINS
    )
    assert stages.count("tail_g2_bf16_wavefront") == 3, stages


def test_wavefront_strided_views_match_the_contiguous_call(device, monkeypatch) -> None:
    # head-major q / k / v / g / beta views (BF16 beta) and a strided output gradient through the pair: the contiguous
    # call's bits
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=437)
    _wavefront_rule(monkeypatch)
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, **WAVEFRONT_PINS)
    views, grads = _strided_leaves(inputs, "head-major")
    o, _ = _call(views, seq_lens, device, **WAVEFRONT_PINS)
    do = torch.cat([inputs["do"], torch.zeros_like(inputs["do"])], -1)[..., :HEAD_DIM]
    torch.autograd.backward(o, do)
    torch.cuda.synchronize()
    got = {"o": o.detach(), **grads()}
    _assert_bitwise(got, want, OUTPUT_NAMES)
    assert stages.count("tail_g2_bf16_wavefront") == 2, stages


# --------------------------------------------------------------------------- input contract


def test_rejects_unsupported_inputs(device) -> None:
    inputs = _inputs((64,), (1, 2), torch.float32, device)
    with pytest.raises(NotImplementedError):
        cake_gdn_det(
            inputs["q"].half(), inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        )
    cu = torch.tensor([0, 64], dtype=torch.int64, device=device)
    two = {name: torch.cat([t, t], 0) for name, t in inputs.items()}
    with pytest.raises(ValueError):
        cake_gdn_det(two["q"], two["k"], two["v"], two["g"], two["beta"], cu_seqlens=cu)
    with pytest.raises(ValueError):
        cake_gdn_det(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
            segment_chunks=-1,
        )  # fmt: skip


# One packed call of three sequences (100, 30, 34 tokens; H = 2, HV = 4) with an initial state, and the argument each
# rejected case replaces; every one raises ValueError before the first launch.
def _invalid_calls(device):
    inputs = _inputs((100, 30, 34), (2, 4), torch.float32, device, seed=29)
    q, k, v, g, beta = (inputs[n] for n in ("q", "k", "v", "g", "beta"))
    cu = torch.tensor([0, 100, 130, 164], dtype=torch.int64)
    h0 = torch.zeros(3, 4, HEAD_DIM, HEAD_DIM, device=device)
    base = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": h0, "cu_seqlens": cu.to(device),
            "cu_seqlens_cpu": cu}  # fmt: skip
    cases = {
        "q-on-host": {"q": q.cpu()},
        "g-on-host": {"g": g.cpu()},
        "initial-state-on-host": {"initial_state": h0.cpu()},
        "cu-seqlens-on-host": {"cu_seqlens": cu},
        "k-heads": {"k": k[:, :, :1]},
        "head-dim-64": {"q": q[..., :64], "k": k[..., :64]},
        "v-tokens": {"v": v[:, :100]},
        "value-heads-not-a-multiple": {
            "v": v[:, :, :3],
            "g": g[:, :, :3],
            "beta": beta[:, :, :3],
        },
        "no-value-heads": {
            "v": v[:, :, :0],
            "g": g[:, :, :0],
            "beta": beta[:, :, :0],
            "initial_state": h0[:, :0],
        },
        "g-shape": {"g": g[:, :, :2]},
        "beta-shape": {"beta": beta[:, :100]},
        "initial-state-rows": {"initial_state": h0[:2]},
        "initial-state-dims": {
            "initial_state": torch.zeros(3, 4, 64, HEAD_DIM, device=device)
        },
        "cu-seqlens-2d": {"cu_seqlens": cu.to(device).view(1, -1)},
        "cu-seqlens-float": {"cu_seqlens": cu.to(device).float()},
        "cu-seqlens-one-offset": {
            "cu_seqlens": cu[:1].to(device),
            "cu_seqlens_cpu": cu[:1],
        },
        "host-offsets-count": {"cu_seqlens_cpu": torch.tensor([0, 130, 164])},
        "host-offsets-float": {"cu_seqlens_cpu": cu.double()},
        "host-offsets-start": {"cu_seqlens_cpu": torch.tensor([4, 100, 130, 164])},
        "host-offsets-decreasing": {"cu_seqlens_cpu": torch.tensor([0, 130, 100, 164])},
        "host-offsets-end": {"cu_seqlens_cpu": torch.tensor([0, 100, 130, 160])},
        "host-offsets-list-end": {"cu_seqlens_cpu": [0, 100, 130, 200]},
        "g-not-a-tensor": {"g": None},
        "beta-not-a-tensor": {"beta": 0.5},
        "k-on-meta": {"k": k.to("meta")},
        "cu-seqlens-list": {"cu_seqlens": [0, 100, 130, 164]},
    }
    return base, cases


_INVALID_IDS = (
    "q-on-host", "g-on-host", "initial-state-on-host", "cu-seqlens-on-host", "k-heads", "head-dim-64", "v-tokens",
    "value-heads-not-a-multiple", "no-value-heads", "g-shape", "beta-shape", "initial-state-rows",
    "initial-state-dims", "cu-seqlens-2d", "cu-seqlens-float", "cu-seqlens-one-offset", "host-offsets-count",
    "host-offsets-float", "host-offsets-start", "host-offsets-decreasing", "host-offsets-end", "host-offsets-list-end",
    "g-not-a-tensor", "beta-not-a-tensor", "k-on-meta", "cu-seqlens-list",
)  # fmt: skip

# The exact message of each rejected case (the checks' contract: type and text).
_OFFSETS = "cu_seqlens must be a 1-D int32 / int64 tensor of at least two offsets; got "
_HOST_OFFSETS = (
    "cu_seqlens_cpu must start at 0, not decrease and end at the 164 tokens of q; got "
)
_INVALID_MESSAGES = {
    "q-on-host": "q must be a CUDA tensor",
    "g-on-host": "g must be a CUDA tensor",
    "initial-state-on-host": "initial_state must be a CUDA tensor",
    "cu-seqlens-on-host": "cu_seqlens must be a CUDA tensor",
    "k-heads": "k must have q's shape (1, 164, 2, 128); got (1, 164, 1, 128)",
    "head-dim-64": "q must be [B, T, H, 128]; got (1, 164, 2, 64)",
    "v-tokens": "v must be [1, 164, HV, 128]; got (1, 100, 4, 128)",
    "value-heads-not-a-multiple": "the value heads (HV=3) must be a positive multiple of the q / k heads (H=2)",
    "no-value-heads": "the value heads (HV=0) must be a positive multiple of the q / k heads (H=2)",
    "g-shape": "g must be (1, 164, 4); got (1, 164, 2)",
    "beta-shape": "beta must be (1, 164, 4); got (1, 100, 4)",
    "initial-state-rows": "initial_state must be (3, 4, 128, 128) (one state per sequence); got (2, 4, 128, 128)",
    "initial-state-dims": "initial_state must be (3, 4, 128, 128) (one state per sequence); got (3, 4, 64, 128)",
    "cu-seqlens-2d": _OFFSETS + "torch.int64 (1, 4)",
    "cu-seqlens-float": _OFFSETS + "torch.float32 (4,)",
    "cu-seqlens-one-offset": _OFFSETS + "torch.int64 (1,)",
    "host-offsets-count": "cu_seqlens_cpu has 3 offsets, cu_seqlens 4",
    "host-offsets-float": "cu_seqlens_cpu must be a 1-D int32 / int64 tensor; got torch.float64 (4,)",
    "host-offsets-start": _HOST_OFFSETS + "[4, 100, 130, 164] ending at 164",
    "host-offsets-decreasing": _HOST_OFFSETS + "[0, 130, 100, 164] ending at 164",
    "host-offsets-end": _HOST_OFFSETS + "[0, 100, 130, 160] ending at 160",
    "host-offsets-list-end": _HOST_OFFSETS + "[0, 100, 130, 200] ending at 200",
    "g-not-a-tensor": "g must be a CUDA tensor",
    "beta-not-a-tensor": "beta must be a CUDA tensor",
    "k-on-meta": "k must be a CUDA tensor",
    "cu-seqlens-list": "cu_seqlens must be a CUDA tensor",
}  # fmt: skip


def _call_rejected(call) -> ValueError:
    with pytest.raises(ValueError) as error:
        cake_gdn_det(
            call["q"], call["k"], call["v"], call["g"], call["beta"], initial_state=call["initial_state"],
            output_final_state=True, cu_seqlens=call["cu_seqlens"], cu_seqlens_cpu=call["cu_seqlens_cpu"],
            use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
    return error.value


@pytest.mark.parametrize("case", _INVALID_IDS)
def test_rejects_invalid_inputs_before_any_launch(case, device, monkeypatch) -> None:
    base, cases = _invalid_calls(device)
    assert set(cases) == set(_INVALID_IDS) == set(_INVALID_MESSAGES)
    call = {**base, **cases[case]}
    stages = _record_launches(monkeypatch)
    assert str(_call_rejected(call)) == _INVALID_MESSAGES[case]
    assert stages == [], stages
    # the unmodified call is accepted
    o, final = cake_gdn_det(
        base["q"], base["k"], base["v"], base["g"], base["beta"], initial_state=base["initial_state"],
        output_final_state=True, cu_seqlens=base["cu_seqlens"], cu_seqlens_cpu=base["cu_seqlens_cpu"],
        use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    assert o.shape == base["v"].shape and final.shape == base["initial_state"].shape


@pytest.mark.parametrize("name", ["k", "beta", "initial_state", "cu_seqlens"])
def test_rejects_inputs_on_another_device(name, device, monkeypatch) -> None:
    # every tensor must be on q's device: one input moved to a second device raises before the first launch
    if torch.cuda.device_count() < 2:
        pytest.skip("a second CUDA device required")
    other = torch.device("cuda", (device.index + 1) % torch.cuda.device_count())
    base, _ = _invalid_calls(device)
    call = {**base, name: base[name].to(other)}
    stages = _record_launches(monkeypatch)
    assert str(_call_rejected(call)) == f"{name} is on {other}, q on {device}"
    assert stages == [], stages


def test_device_route_does_not_synchronize(device) -> None:
    # the checks read tensor metadata only: a device-route forward and backward runs under the error sync-debug mode
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=31)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int32, device=device)
    h0 = torch.zeros(
        len(seq_lens), 4, HEAD_DIM, HEAD_DIM, device=device, requires_grad=True
    )

    def call():
        leaves = {
            n: inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        }
        o, final = cake_gdn_det(
            leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], initial_state=h0,
            output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        ((o.float() * inputs["do"].float()).sum() + final.sum()).backward()

    call()  # loads the libraries and fills the caches
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        call()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


# --------------------------------------------------------------------------- head pairs and training-loop contracts

# The head pairs of the source issue (#7074) beyond (2, 6) / (1, 3) and two value-head groups of two with more key
# heads, on one layout of partial, full and multi-chunk sequences (both beta dtypes across the rows).
HEAD_PAIRS = (
    ((16, 48), torch.bfloat16),
    ((8, 24), torch.float32),
    ((4, 12), torch.bfloat16),
    ((16, 32), torch.float32),
    ((8, 16), torch.bfloat16),
)
PAIR_LAYOUT = (63, 130, 257)


@pytest.mark.parametrize(
    "pair", HEAD_PAIRS, ids=[f"h{h}-hv{hv}" for (h, hv), _dtype in HEAD_PAIRS]
)
def test_head_pairs_dispatch_the_fused_tail(pair, device, monkeypatch) -> None:
    # HV = 3 H dispatches the round-once tail (tail_g3 where F32X_TAIL_ARCHS leaves the architecture out) and HV = 2 H
    # tail_g2 (no dqkwg, WY pullback or finalize stage); the outputs pass the FP64 gate, repeat bitwise, and for
    # HV = 3 H equal the unfused tail under either finalize pin bitwise (on the round-once tail every output but
    # dq / dk, which it rounds once after the group's FP32 sum)
    heads, beta_dtype = pair
    group = heads[1] // heads[0]
    inputs = _inputs(PAIR_LAYOUT, heads, beta_dtype, device, seed=211)
    stages = _record_launches(monkeypatch)
    got = _run(inputs, PAIR_LAYOUT)
    tail = _device_tail(heads, beta_dtype)
    # HV = 2 H with BF16 beta may run the tail as the wavefront pair's
    assert (
        tail in stages or tail + "_wavefront" in stages
    ) and "dqkwg" not in stages, stages
    assert not any(stage.startswith(("wy_bwd", "finalize")) for stage in stages), stages
    ref = _reference(inputs, PAIR_LAYOUT)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES)
    _assert_bitwise(got, _run(inputs, PAIR_LAYOUT), OUTPUT_NAMES)
    if group == 3:
        round_once = tail in cake_chain.F32X_TAILS
        same = OUTPUT_NAMES
        if round_once:
            same = tuple(n for n in OUTPUT_NAMES if n not in ("dq", "dk"))
        for fin in ("finalize", "finalize_mlp"):
            start = len(stages)
            unfused = _run(inputs, PAIR_LAYOUT, finalize=fin)
            assert fin in stages[start:] and tail not in stages[start:], stages[start:]
            _assert_bitwise(got, unfused, same)
            if round_once:
                _assert_close_to_reference(unfused, ref, OUTPUT_NAMES)


@pytest.mark.parametrize("form", ["serial", "split"])
@pytest.mark.parametrize(
    "pair", HEAD_PAIRS[:3], ids=[f"h{h}-hv{hv}" for (h, hv), _dtype in HEAD_PAIRS[:3]]
)
def test_head_pairs_run_the_two_stream_tail(pair, form, device, monkeypatch) -> None:
    # HV = 3 H with the planner on few SMs: the pass 2 of a split plan on the tcgen05 scan (dhu_tc2_dvl32 on
    # DVL32_ARCHS) runs as two groups, the side stream scanning group B while the caller's stream runs the fused tail
    # (the round-once tail on F32X_TAIL_ARCHS) of group A; a serial tcgen05 adjoint scan, grouped on the table, does
    # the same on the architectures of TWO_STREAM_SERIAL_G3_ARCHS and elsewhere keeps one launch on the caller's
    # stream; the outputs pass the FP64 gate and equal the one-stream call bitwise
    heads, beta_dtype = pair
    num_v_heads = heads[1]
    if form == "serial":
        # 3 sequences: 2 HV CTAs fill a round of 2 HV + 2 SMs, so adjoint group A = 2 sequences
        pins, sm_count = dict(STREAM_PINS), 2 * num_v_heads + 2
    else:
        # 1 + 3 + 5 chunks in segments of 2 = 6 segments: 4 HV CTAs leave 1 of 4 HV + 1 SMs idle, so A = 4 segments
        pins = {"scan_config": (64, 1), "segment_chunks": 2, "dhu_scan": "tc"}
        sm_count = 4 * num_v_heads + 1
    _small_device(monkeypatch, device, sm_count)
    inputs = _inputs(PAIR_LAYOUT, heads, beta_dtype, device, seed=239)
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    tail, tc2 = _device_tail(heads, beta_dtype), _tc2(heads)
    two = _run(inputs, PAIR_LAYOUT, streams=2, **pins)
    stages = [stage for stage, _stream in launches]
    side = [stage for stage, stream in launches if stream != main]
    if _adjoint_two_streams(form, heads):
        assert side.count(tc2) == 1 and tail not in side, launches
        assert stages.count(tc2) == 2 and stages.count(tail) == 2, stages
    else:
        assert tc2 not in side and tail not in side, launches
        assert stages.count(tc2) == 1 and stages.count(tail) == 1, stages
    assert "dqkwg" not in stages, stages
    start = len(launches)
    one = _run(inputs, PAIR_LAYOUT, streams=1, **pins)
    assert all(stream == main for _stage, stream in launches[start:]), launches[start:]
    _assert_bitwise(two, one, OUTPUT_NAMES)
    _assert_close_to_reference(two, _reference(inputs, PAIR_LAYOUT), OUTPUT_NAMES)


def _call(leaves, seq_lens, device, *, route="host", initial_state=None, output_final_state=False, **pins):  # fmt: skip
    """One call of the package on caller-made leaves (or views of them)."""
    cu = _cu(seq_lens)
    return cake_gdn_det(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                      scale=1.0 / math.sqrt(HEAD_DIM), initial_state=initial_state,
                                      output_final_state=output_final_state,
                                      cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                                      cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64) if route == "host" else None,
                                      use_qk_l2norm_in_kernel=True, **pins)  # fmt: skip


@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize("heads", [(2, 4), (2, 6), (2, 2)], ids=["g2", "g3", "g1"])
def test_gradients_accumulate_into_a_nonzero_grad(
    heads, route, beta_dtype, device
) -> None:
    # autograd adds the package's gradients into an existing .grad (prior + fresh in the leaf's dtype, the
    # initial-state gradient included): after one call every leaf holds exactly prior + fresh, after a second
    # prior + fresh + fresh, where fresh is the gradient of the same call into empty .grad.  Both beta dtypes: both
    # forms of the round-once tail (G = 3), and FP32 beta on the host route the forward on wy_a's A
    _check_accumulation((300, 77, 640), heads, device, route=route, beta_dtype=beta_dtype)  # fmt: skip


@pytest.mark.parametrize("form", ["serial", "split"])
@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_two_stream_gradients_accumulate_into_a_nonzero_grad(
    heads, form, device, monkeypatch
) -> None:
    # grouped tables under autograd accumulation (the adjoint on two streams: serial beside tail_g2, split pass 2
    # beside tail_g3; the other two pairings on one): prior + fresh, then + fresh again, where fresh is the one-stream
    # gradient of the same call (the side stream joins the caller's stream before the gradients reach .grad)
    if form == "serial":
        seq_lens, sm_count, pins = STREAM_LAYOUT, STREAM_CASES[heads], STREAM_PINS
    else:
        seq_lens, sm_count, pins = (
            SPLIT_STREAM_LAYOUT,
            SPLIT_STREAM_CASES[heads],
            SPLIT_STREAM_PINS,
        )
    _small_device(monkeypatch, device, sm_count)
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    _check_accumulation(
        seq_lens,
        heads,
        device,
        fresh_pins={**pins, "streams": 1},
        pins={**pins, "streams": 2},
    )
    side = [stage for stage, stream in launches if stream != main]
    # one per accumulated call where the adjoint runs two streams
    assert side.count(_tc2(heads)) == (2 if _adjoint_two_streams(form, heads) else 0), (
        launches
    )


def _check_accumulation(seq_lens, heads, device, *, route="host", fresh_pins=None, pins=None,
                        beta_dtype=torch.bfloat16) -> None:  # fmt: skip
    fresh_pins, pins = fresh_pins or {}, pins or {}
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=231)
    gen = torch.Generator().manual_seed(233)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    fresh = _run(inputs, seq_lens, route=route, initial_state=h0, output_final_state=True, dht=dht, **fresh_pins)  # fmt: skip
    names = GRAD_NAMES + ("dh0",)
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    leaves["h0"] = h0.detach().clone().requires_grad_(True)
    grad_name = {name: "dh0" if name == "h0" else "d" + name for name in leaves}
    prior = {}
    for name, leaf in leaves.items():
        prior[grad_name[name]] = torch.randn(leaf.shape, generator=gen).to(
            device=device, dtype=leaf.dtype
        )
        leaf.grad = prior[grad_name[name]].clone()

    def step() -> dict:
        o, final = _call(leaves, seq_lens, device, route=route, initial_state=leaves["h0"], output_final_state=True,
                         **pins)  # fmt: skip
        ((o.float() * inputs["do"].float()).sum() + (final * dht).sum()).backward()
        torch.cuda.synchronize()
        return {grad_name[name]: leaf.grad.clone() for name, leaf in leaves.items()}

    want = {name: prior[name] + fresh[name] for name in names}
    _assert_bitwise(step(), want, names)
    want = {name: want[name] + fresh[name] for name in names}
    _assert_bitwise(step(), want, names)


def _strided_leaves(inputs, form):
    """Leaves holding the inputs' values and q / k / v / g / beta as non-contiguous views of them.

    ``packed``: q / k / v as views of one ``[1, T, (2 H + HV) * 128]`` projection and g / beta of one ``[1, T, HV, 2]``
    FP32 gate projection (token stride ``(2 H + HV) * 128``, element stride 2); ``head-major``: every operand a
    transposed view of a contiguous ``[1, heads, T, ...]`` leaf."""
    names = ("q", "k", "v", "g", "beta")
    if form == "packed":
        total = inputs["q"].shape[1]
        widths = tuple(inputs[n].shape[2] * HEAD_DIM for n in ("q", "k", "v"))
        qkv = torch.cat([inputs[n].reshape(1, total, -1) for n in ("q", "k", "v")], -1)
        qkv = qkv.detach().clone().requires_grad_(True)
        gb = torch.stack([inputs["g"], inputs["beta"].float()], -1)
        gb = gb.detach().clone().requires_grad_(True)
        views = dict(
            zip(("q", "k", "v"), (p.unflatten(-1, (-1, HEAD_DIM)) for p in qkv.split(widths, -1)), strict=True)
        )  # fmt: skip
        views["g"], views["beta"] = gb[..., 0], gb[..., 1]

        def grads() -> dict:
            dq, dk, dv = (p.unflatten(-1, (-1, HEAD_DIM)) for p in qkv.grad.split(widths, -1))  # fmt: skip
            return {"dq": dq, "dk": dk, "dv": dv, "dg": gb.grad[..., 0], "dbeta": gb.grad[..., 1]}  # fmt: skip

    else:
        leaves = {
            n: inputs[n].transpose(1, 2).contiguous().requires_grad_(True)
            for n in names
        }
        views = {n: leaves[n].transpose(1, 2) for n in names}

        def grads() -> dict:
            return {"d" + n: leaves[n].grad.transpose(1, 2) for n in names}

    for n in names:
        assert views[n].shape == inputs[n].shape and views[n].dtype == inputs[n].dtype
        assert not views[n].is_contiguous(), (form, n, views[n].stride())
    return views, grads


@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize(
    "form, beta_dtype",
    [("packed", torch.float32), ("head-major", torch.bfloat16)],
    ids=["packed", "head-major"],
)
def test_strided_views_match_the_contiguous_call(
    form, beta_dtype, route, device, monkeypatch
) -> None:
    # q / k / v / g / beta as strided views and a strided output gradient: the outputs and the gradients that reach
    # the viewed leaves equal those of the contiguous call bitwise.  The packed row (FP32 beta) on the host route takes
    # the forward on wy_a's A (a serial table: 5 + 2 + 10 chunks), whose wy_a reads q / k and fwd_ho_a v in place (no
    # copy: the head views of the packed row are 16-byte aligned); the device route and the head-major operands (BF16
    # beta) keep the prep forward.  Under the round-once tail the packed row is the autograd input on both routes
    # (CakeGDNDetPackedFunction: the tail writes one packed gradient row)
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=227)
    want = _run(inputs, seq_lens, route=route)
    views, grads = _strided_leaves(inputs, form)
    launches = _record_launch_values(monkeypatch)
    o, _ = _call(views, seq_lens, device, route=route)
    packed = (
        form == "packed" and _device_tail(heads, beta_dtype) in cake_chain.F32X_TAILS
    )
    assert (type(o.grad_fn).__name__ == "CakeGDNDetPackedFunctionBackward") is packed, o.grad_fn  # fmt: skip
    do = torch.cat([inputs["do"], torch.zeros_like(inputs["do"])], -1)[..., :HEAD_DIM]
    assert not do.is_contiguous() and torch.equal(do, inputs["do"])
    torch.autograd.backward(o, do)
    torch.cuda.synchronize()
    got = {"o": o.detach(), **grads()}
    _assert_bitwise(got, want, OUTPUT_NAMES)
    stages = [stage for stage, _values in launches]
    if form == "packed" and route == "host":
        _assert_fwd_route(stages, "a")
        for stage, operands in (("wy_a", "q / k"), ("fwd_ho_a", "v")):
            values = next(values for s, values in launches if s == stage)
            assert any(
                isinstance(t, torch.Tensor) and not t.is_contiguous() for t in values
            ), f"{operands} were copied"
    else:
        assert "wy_a" not in stages and _preps(stages), stages


def _allocator_bytes(device) -> tuple[int, int]:
    """(requested, block) bytes of the caching allocator's live allocations on ``device``."""
    stats = torch.cuda.memory_stats(device)
    return int(stats["requested_bytes.all.current"]), int(
        stats["allocated_bytes.all.current"]
    )


@pytest.mark.parametrize("form", ["one", "serial", "split", "wavefront"])
@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_checkpoint_matches_and_retains_no_bytes(
    heads, form, device, monkeypatch
) -> None:
    # non-reentrant torch.utils.checkpoint (host table route): the outputs and gradients equal the plain (one-stream)
    # call's bitwise, and a checkpointed forward keeps nothing on the device beyond its output (#7039); "serial" /
    # "split" run grouped tables (the planner on few SMs; the adjoint on two streams where _adjoint_two_streams)
    seq_lens, pins = (300, 77, 640), {}
    if form == "serial":
        seq_lens, pins = STREAM_LAYOUT, STREAM_PINS
        _small_device(monkeypatch, device, STREAM_CASES[heads])
    elif form == "split":
        seq_lens, pins = SPLIT_STREAM_LAYOUT, SPLIT_STREAM_PINS
        _small_device(monkeypatch, device, SPLIT_STREAM_CASES[heads])
    elif form == "wavefront":
        # the wavefront pair on the side stream (g2; g3 keeps the separate launches)
        _wavefront_rule(monkeypatch)
        pins = dict(WAVEFRONT_PINS)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=229)
    eager = _run(inputs, seq_lens, streams=1, **pins)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }

    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def layer(q, k, v, g, beta):
        o, _ = cake_gdn_det(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                          cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True, **pins)  # fmt: skip
        return o

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        return checkpoint(layer, *leaves.values(), use_reentrant=False)

    for _ in range(2):  # the libraries, the tables and the workspaces of this layout
        (step().float() * inputs["do"].float()).sum().backward()
    for leaf in leaves.values():
        leaf.grad = None
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    gc.collect()
    torch.cuda.synchronize()
    requested, blocks = _allocator_bytes(device)
    o = step()
    torch.cuda.synchronize()
    requested_after, blocks_after = _allocator_bytes(device)
    # requested bytes: the block count may exceed the output without a retained tensor, since the caching allocator
    # hands out an unsplit cached block when its remainder is below the split threshold (1 MiB)
    retained = requested_after - requested - o.untyped_storage().nbytes()
    unsplit = blocks_after - blocks - (o.untyped_storage().nbytes() + 511) // 512 * 512
    assert retained == 0, (
        f"a checkpointed forward retains {retained} requested bytes beyond its output (blocks +{unsplit})"
    )
    assert unsplit < 2**20, unsplit
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    got = {"o": o.detach(), **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")}}  # fmt: skip
    _assert_bitwise(got, eager, OUTPUT_NAMES)
    side = [stage for stage, stream in launches if stream != main]
    assert (_tc2(heads) in side) is _adjoint_two_streams(form, heads), launches
    if form == "wavefront":
        assert ("tail_g2_bf16_wavefront" in side) is (heads == (2, 4)), launches


# --------------------------------------------------------------------------- the round-once tail and the packed row

# the value-head groups the round-once route tests run: groups of three (two key heads, one) and a group of two, which
# keeps tail_g2 and dhu_tc2
ROUND_ONCE_HEADS = {"g3": (2, 6), "g3-h1": (1, 3), "g2": (2, 4)}
# a serial tcgen05 adjoint scan, and a split (64, 1) plan's pass 2 on it (after the fused pre-scan): both run on
# dhu_tc2_dvl32 for a group of three, the split one on a host-built table only (a device-built table keeps the register
# pass 2, _sdt_selected); "planner": no pin
ROUND_ONCE_PINS = {
    "planner": {},
    "serial-tc": {"segment_chunks": 0, "dhu_scan": "tc"},
    "split-tc": {"scan_config": (64, 1), "segment_chunks": 6, "dhu_scan": "tc"},
}
ROUND_ONCE_LAYOUT = (1500, 77, 640, 1)
# (table route, pins) of the switch tests: the forms whose adjoint runs on the tcgen05 scan
ROUND_ONCE_SWITCH_CASES = (
    ("host", "serial-tc"),
    ("device", "serial-tc"),
    ("host", "split-tc"),
)
# the outputs the round-once tail and the FP32 dv_local adjoint leave to the other switch: the tail rounds dq / dk
# differently, every other output has one writer of the same value
NOT_DQ_DK = ("o", "dv", "dg", "dbeta", "final_state", "dh0")


def _round_once_or_skip(heads, beta_dtype) -> str:
    """The round-once tail of ``heads`` on this device; skip where the rule gives another tail."""
    tail = _device_tail(heads, beta_dtype)
    if tail not in cake_chain.F32X_TAILS:
        pytest.skip(f"{_current_arch()} runs {tail} for {heads}")
    return tail


def _launch_parameter(launches, stage: str, name: str) -> list:
    """The value of parameter ``name`` in every launch of ``stage`` (:func:`_record_launch_values`)."""
    names = cake_backend.parameters(cake_backend.stage_record(stage, _current_arch()))
    return [values[names.index(name)] for s, values in launches if s == stage]


def _packed_inputs(inputs, heads):
    """One contiguous ``[1, T, (2 H + HV) * 128]`` BF16 tensor of q / k / v (no grad) and the row widths."""
    h, hv = heads
    total = inputs["q"].shape[1]
    flat = [inputs[n].reshape(1, total, -1) for n in ("q", "k", "v")]
    return torch.cat(flat, -1).detach(), (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)


def _packed_call(views, g, beta, seq_lens, *, route="host", **pins):
    """One call of the package on q / k / v ``views`` (the head views of a packed row); returns ``o``."""
    cu = _cu(seq_lens)
    o, _ = cake_gdn_det(*views, g, beta, scale=1.0 / math.sqrt(HEAD_DIM),
                        cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=g.device),
                        cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64) if route == "host" else None,
                        use_qk_l2norm_in_kernel=True, **pins)  # fmt: skip
    return o


def _row_results(o, row_grad, widths, g, beta) -> dict:
    dq, dk, dv = _views(row_grad, widths)
    return {
        "o": o.detach(),
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": g.grad,
        "dbeta": beta.grad,
    }


def _run_packed(inputs, seq_lens, heads, *, route="host", **pins) -> dict:
    """:func:`_run` with q / k / v the head views of one packed row that requires grad: the outputs and gradients,
    dq / dk / dv the head views of the row's gradient."""
    data, widths = _packed_inputs(inputs, heads)
    row = data.clone().requires_grad_(True)
    g, beta = (inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    o = _packed_call(_views(row, widths), g, beta, seq_lens, route=route, **pins)
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    return _row_results(o, row.grad, widths, g, beta)


@pytest.mark.parametrize("pins", sorted(ROUND_ONCE_PINS))
@pytest.mark.parametrize(
    "beta_dtype", [torch.float32, torch.bfloat16], ids=["fp32beta", "bf16beta"]
)
@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize(
    "heads", list(ROUND_ONCE_HEADS.values()), ids=list(ROUND_ONCE_HEADS)
)
def test_round_once_route_on_the_device(
    heads, route, beta_dtype, pins, device, monkeypatch
) -> None:
    # every architecture of the package is in F32X_TAIL_ARCHS and DVL32_ARCHS: a value-head group of three launches the
    # round-once tail of its beta dtype (no tail_g3, dqkwg, WY pullback or finalize) and runs its tcgen05 adjoint scan
    # as dhu_tc2_dvl32 (no dhu_tc2), the tail with the token-row strides of contiguous operands (v / dv HV x 128, dq /
    # dk one stride, H x 128); a group of two keeps tail_g2 and dhu_tc2 (or the wavefront pair).  With an initial state
    # and a final-state gradient the outputs repeat bitwise and (planner) pass the FP64 gate
    arch = _current_arch()
    assert arch in cake_chain.F32X_TAIL_ARCHS and arch in cake_chain.DVL32_ARCHS, arch
    seq_lens = ROUND_ONCE_LAYOUT
    pin = ROUND_ONCE_PINS[pins]
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=601)
    h0, dht = _state_pair(seq_lens, heads, device, 602)
    kwargs = {"route": route, "initial_state": h0, "output_final_state": True, "dht": dht, **pin}  # fmt: skip
    names = OUTPUT_NAMES + ("final_state", "dh0")
    launches = _record_launch_values(monkeypatch)
    got = _run(inputs, seq_lens, **kwargs)
    stages = [stage for stage, _values in launches]
    tail = _device_tail(heads, beta_dtype)
    if heads[1] == 3 * heads[0]:
        suffix = "_bf16" if beta_dtype == torch.bfloat16 else ""
        assert tail == "tail_g3_f32x" + suffix, tail
        assert stages.count(tail) >= 1 and "dqkwg" not in stages, stages
        assert not {"tail_g3", "tail_g3_bf16", "dhu_tc2"} & set(stages), stages
        assert not any(stage.startswith(("wy_bwd", "finalize")) for stage in stages), stages  # fmt: skip
        if pins == "serial-tc" or (pins, route) == ("split-tc", "host"):
            assert "dhu_tc2_dvl32" in stages, stages
        h, hv = heads
        for name, want in (
            ("v_row_stride", hv * HEAD_DIM),
            ("dq_row_stride", h * HEAD_DIM),
            ("dk_row_stride", h * HEAD_DIM),
            ("dv_row_stride", hv * HEAD_DIM),
        ):
            values = _launch_parameter(launches, tail, name)
            assert values and set(values) == {want}, (name, values)
    else:
        assert tail.startswith("tail_g2"), tail
        assert tail in stages or tail + "_wavefront" in stages, stages
        assert not set(ROUND_ONCE_STAGES) & set(stages), stages
    _assert_bitwise(_run(inputs, seq_lens, **kwargs), got, names)
    if not pin:
        ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
        _assert_close_to_reference(got, ref, names)


@pytest.mark.parametrize(
    "beta_dtype", [torch.float32, torch.bfloat16], ids=["fp32beta", "bf16beta"]
)
@pytest.mark.parametrize(
    "route, pins",
    ROUND_ONCE_SWITCH_CASES,
    ids=[f"{route}-{pins}" for route, pins in ROUND_ONCE_SWITCH_CASES],
)
def test_round_once_switches_restore_the_tail_g3_bits(
    route, beta_dtype, pins, device, monkeypatch
) -> None:
    # the switches of ROUND_ONCE_OFF, read at each call (the chunk tables do not depend on them).  Both off: tail_g3 on
    # dhu_tc2, with the unfused tail's bits on every output (a finalize pin: dqkwg, the WY pullback and the finalize),
    # as before the change.  The tail's switch alone off: tail_g3 on dhu_tc2_dvl32, every output but dq / dk the
    # round-once route's bits.  The adjoint's switch alone off: the round-once tail on dhu_tc2, every output but dq /
    # dk the bits with both off.  Each form repeats bitwise and stays within the budgets of the FP64 recurrence
    heads = (2, 6)
    _round_once_or_skip(heads, beta_dtype)
    seq_lens = ROUND_ONCE_LAYOUT
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=607)
    h0, dht = _state_pair(seq_lens, heads, device, 608)
    kwargs = {"route": route, "initial_state": h0, "output_final_state": True, "dht": dht, **ROUND_ONCE_PINS[pins]}  # fmt: skip
    names = OUTPUT_NAMES + ("final_state", "dh0")
    suffix = "_bf16" if beta_dtype == torch.bfloat16 else ""
    forms = {
        # form: (the switches off, the tail and the adjoint scan it launches)
        "on": ((), ("tail_g3_f32x" + suffix, "dhu_tc2_dvl32")),
        "tail-off": (("F32X_TAIL_ARCHS",), ("tail_g3" + suffix, "dhu_tc2_dvl32")),
        "adjoint-off": (("DVL32_ARCHS",), ("tail_g3_f32x" + suffix, "dhu_tc2")),
        "off": (("F32X_TAIL_ARCHS", "DVL32_ARCHS"), ("tail_g3" + suffix, "dhu_tc2")),
    }
    every = {stage for _off, pair in forms.values() for stage in pair}
    stages = _record_launches(monkeypatch)
    runs = {}
    for form, (off, pair) in forms.items():
        with monkeypatch.context() as m:
            _round_once_off(m, off)
            start = len(stages)
            runs[form] = _run(inputs, seq_lens, **kwargs)
            got = set(stages[start:])
            assert set(pair) <= got and not (every - set(pair)) & got, (
                form,
                stages[start:],
            )
            _assert_bitwise(_run(inputs, seq_lens, **kwargs), runs[form], names)
            if form == "off":
                start = len(stages)
                runs["unfused"] = _run(inputs, seq_lens, finalize="finalize", **kwargs)
                got = set(stages[start:])
                assert "finalize" in got and not (every - {"dhu_tc2"}) & got, stages[start:]  # fmt: skip
    _assert_bitwise(runs["off"], runs["unfused"], names)
    _assert_bitwise(runs["tail-off"], runs["on"], NOT_DQ_DK)
    _assert_bitwise(runs["adjoint-off"], runs["off"], NOT_DQ_DK)
    for a, b in (("on", "tail-off"), ("adjoint-off", "off")):
        assert not (
            torch.equal(runs[a]["dq"], runs[b]["dq"])
            and torch.equal(runs[a]["dk"], runs[b]["dk"])
        ), (a, b)
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    for form in forms:
        _assert_close_to_reference(runs[form], ref, names)


@pytest.mark.parametrize(
    "beta_dtype", [torch.float32, torch.bfloat16], ids=["fp32beta", "bf16beta"]
)
@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize("heads", [(2, 6), (1, 3)], ids=["g3", "g3-h1"])
def test_round_once_tail_repeats_its_bits(heads, route, beta_dtype, device) -> None:
    # five runs of the round-once route on contiguous operands and five on the packed row: the same bits on every
    # output, the packed row's those of the contiguous call
    _round_once_or_skip(heads, beta_dtype)
    seq_lens = (2000, 300, 77)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=611)
    first = _run(inputs, seq_lens, route=route)
    for _ in range(4):
        _assert_bitwise(_run(inputs, seq_lens, route=route), first, OUTPUT_NAMES)
    for _ in range(5):
        _assert_bitwise(_run_packed(inputs, seq_lens, heads, route=route), first, OUTPUT_NAMES)  # fmt: skip


@pytest.mark.parametrize("finalize", [None, "finalize"], ids=["rule", "finalize-pin"])
@pytest.mark.parametrize(
    "heads", [(2, 6), (1, 3), (2, 4), (1, 4)], ids=["g3", "g3-h1", "g2", "g4"]
)
def test_packed_row_gives_the_view_bits(heads, finalize, device, monkeypatch) -> None:
    # q / k / v as head views of one packed row that requires grad: under the round-once tail the row is the autograd
    # input (CakeGDNDetPackedFunction) and the backward returns one packed gradient row the tail wrote through its
    # token-row stride (dq / dk / dv at the row's width), without the split's concatenation; the same bits as the
    # views route (PACKED_ROW off) and as the contiguous call on every output.  Another tail (a finalize pin, a group
    # of two or four) takes the views route
    seq_lens = (300, 77, 640, 1)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=613)
    data, widths = _packed_inputs(inputs, heads)
    pins = {"finalize": finalize} if finalize else {}
    tail = _device_tail(heads, torch.float32, finalize)
    packed_route = tail in cake_chain.F32X_TAILS
    launches = _record_launch_values(monkeypatch)
    results, nodes = {}, {}
    for packed in (True, False):
        monkeypatch.setattr(cake_chain, "PACKED_ROW", packed)
        launches.clear()
        row = data.clone().requires_grad_(True)
        g, beta = (
            inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta")
        )
        o = _packed_call(_views(row, widths), g, beta, seq_lens, **pins)
        nodes[packed] = type(o.grad_fn).__name__
        (o.float() * inputs["do"].float()).sum().backward()
        torch.cuda.synchronize()
        results[packed] = _row_results(o, row.grad, widths, g, beta)
        if packed and packed_route:
            for name in ("dq_row_stride", "dk_row_stride", "dv_row_stride"):
                values = _launch_parameter(launches, tail, name)
                assert values and set(values) == {sum(widths)}, (name, values)
    want_node = (
        "CakeGDNDetPackedFunctionBackward"
        if packed_route
        else "CakeGDNDetFunctionBackward"
    )
    assert nodes == {True: want_node, False: "CakeGDNDetFunctionBackward"}, nodes
    _assert_bitwise(results[True], results[False], OUTPUT_NAMES)
    _assert_bitwise(results[True], _run(inputs, seq_lens, **pins), OUTPUT_NAMES)


def test_packed_row_of_a_non_leaf_with_other_consumers(device, monkeypatch) -> None:
    # the row as a non-leaf (a projection's output) that another consumer also reads, two backward passes into the
    # projection's leaf: autograd sums the packed gradient row with the other consumer's gradient and accumulates both
    # passes; the same bits as the views route
    seq_lens, heads = (300, 77, 640), (2, 6)
    _round_once_or_skip(heads, torch.bfloat16)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=617)
    data, widths = _packed_inputs(inputs, heads)
    weight = torch.randn(data.shape, generator=torch.Generator().manual_seed(619)).to(
        device
    )
    grads = {}
    for packed in (True, False):
        monkeypatch.setattr(cake_chain, "PACKED_ROW", packed)
        leaf = data.float().clone().requires_grad_(True)
        g, beta = (
            inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta")
        )
        for _ in range(2):
            row = leaf.to(torch.bfloat16)  # a non-leaf packed row
            o = _packed_call(_views(row, widths), g, beta, seq_lens)
            assert (
                type(o.grad_fn).__name__ == "CakeGDNDetPackedFunctionBackward"
            ) is packed
            loss = (o.float() * inputs["do"].float()).sum() + (
                row.float() * weight
            ).sum()
            loss.backward()
        torch.cuda.synchronize()
        grads[packed] = {"row": leaf.grad, "dg": g.grad, "dbeta": beta.grad}
    _assert_bitwise(grads[True], grads[False], ("row", "dg", "dbeta"))


def test_packed_row_without_a_row_gradient(device) -> None:
    # only g and beta require grad: no packed route (the row records no gradient); their gradients have the bits of
    # the full backward's
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=623)
    data, widths = _packed_inputs(inputs, heads)
    full = _run_packed(inputs, seq_lens, heads)
    g, beta = (inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    o = _packed_call(_views(data.clone(), widths), g, beta, seq_lens)
    assert type(o.grad_fn).__name__ == "CakeGDNDetFunctionBackward"
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    _assert_bitwise(
        {"o": o.detach(), "dg": g.grad, "dbeta": beta.grad}, full, ("o", "dg", "dbeta")
    )


@pytest.mark.parametrize("reentrant", [False, True], ids=["non-reentrant", "reentrant"])
def test_packed_row_under_checkpoint_replay(reentrant, device) -> None:
    # torch.utils.checkpoint over the packed row: the replayed forward takes the packed route again and the gradients
    # have the bits of the eager call (the saved-set path); the non-reentrant checkpointed forward keeps nothing beyond
    # its output.  The reentrant form's first pass runs without grad (no node, no packed route there)
    seq_lens, heads = (300, 77, 640), (2, 6)
    _round_once_or_skip(heads, torch.bfloat16)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=627)
    data, widths = _packed_inputs(inputs, heads)
    row = data.clone().requires_grad_(True)
    g, beta = (inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    nodes = []

    def layer(row, g, beta):
        o = _packed_call(_views(row, widths), g, beta, seq_lens)
        if o.grad_fn is not None:
            nodes.append(type(o.grad_fn).__name__)
        return o

    def step(checkpointed):
        for leaf in (row, g, beta):
            leaf.grad = None
        o = (
            checkpoint(layer, row, g, beta, use_reentrant=reentrant)
            if checkpointed
            else layer(row, g, beta)
        )
        (o.float() * inputs["do"].float()).sum().backward()
        torch.cuda.synchronize()
        return _row_results(o, row.grad, widths, g, beta)

    eager = step(False)
    for _ in range(2):  # the libraries, the tables and the workspaces of this layout
        _assert_bitwise(step(True), eager, OUTPUT_NAMES)
    assert set(nodes) == {"CakeGDNDetPackedFunctionBackward"}, nodes
    if reentrant:
        return
    for leaf in (row, g, beta):
        leaf.grad = None
    gc.collect()
    torch.cuda.synchronize()
    requested, _blocks = _allocator_bytes(device)
    o = checkpoint(layer, row, g, beta, use_reentrant=False)
    torch.cuda.synchronize()
    retained = _allocator_bytes(device)[0] - requested - o.untyped_storage().nbytes()
    assert retained == 0, (
        f"a checkpointed forward retains {retained} requested bytes beyond its output"
    )
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    _assert_bitwise(_row_results(o, row.grad, widths, g, beta), eager, OUTPUT_NAMES)


def test_low_level_backward_writes_the_callers_rows(device, monkeypatch) -> None:
    # cake_gdn_det_train_backward(grad_qkv=...) under the round-once tail: the tail writes dq / dk / dv into the
    # caller's head views of one packed gradient row (three tensors, or a callable that returns them with their common
    # stride, called once, after the adjoint scan, right before the tail) and returns them, every element written (the
    # rows start as NaN), the bits of the call without them; dq / dk at two token-row strides raise ValueError before
    # the tail; another tail (a finalize pin) neither calls the callable nor returns the caller's views
    seq_lens, heads = (300, 77, 640), (2, 6)
    tail = _round_once_or_skip(heads, torch.float32)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=631)
    index = cake_chain._device_index(device)
    plans = cake_chain.scan_plans(sum(seq_lens), len(seq_lens), heads[1], index, seq_lens)  # fmt: skip
    meta = cake_chain.build_chunk_meta(seq_lens, device, scan_plans=plans)
    q, k, v, g, beta, do = (inputs[n] for n in ("q", "k", "v", "g", "beta", "do"))
    arch = _current_arch()
    tape = cake_chain.cake_gdn_det_train_forward(q, k, v, g, beta, meta=meta, arch=arch)
    want = cake_chain.cake_gdn_det_train_backward(tape, do)
    widths = (heads[0] * HEAD_DIM, heads[0] * HEAD_DIM, heads[1] * HEAD_DIM)
    width = sum(widths)

    def rows():
        nan = torch.full((1, sum(seq_lens), width), float("nan"), dtype=torch.bfloat16, device=device)  # fmt: skip
        return _views(nan, widths)

    def same(got) -> None:
        _assert_bitwise({n: got[n].reshape(want[n].shape) for n in GRAD_NAMES}, want, GRAD_NAMES)  # fmt: skip

    stages = _record_launches(monkeypatch)
    views = rows()
    got = cake_chain.cake_gdn_det_train_backward(tape, do, grad_qkv=views)
    assert tail in stages, stages
    assert all(got[n] is view for n, view in zip(("dq", "dk", "dv"), views, strict=True))  # fmt: skip
    same(got)
    calls = []

    def made():
        calls.append(len(stages))
        return (*rows(), width)

    start = len(stages)
    got = cake_chain.cake_gdn_det_train_backward(tape, do, grad_qkv=made)
    assert len(calls) == 1 and calls[0] > start, (calls, start)
    assert tail not in stages[start : calls[0]] and tail in stages[calls[0] :], stages[start:]  # fmt: skip
    same(got)
    q_row, _k_row, v_row = rows()
    dk = torch.empty(q_row.shape, dtype=torch.bfloat16, device=device)
    start = len(stages)
    with pytest.raises(ValueError, match="one token-row stride"):
        cake_chain.cake_gdn_det_train_backward(tape, do, grad_qkv=(q_row, dk, v_row))
    assert tail not in stages[start:], stages[start:]
    pinned = cake_chain.cake_gdn_det_train_forward(q, k, v, g, beta, meta=meta, arch=arch,
                                                   options=cake_chain.make_options(finalize="finalize"))  # fmt: skip
    calls.clear()
    start = len(stages)
    got = cake_chain.cake_gdn_det_train_backward(pinned, do, grad_qkv=made)
    assert not calls and "finalize" in stages[start:] and tail not in stages[start:], stages[start:]  # fmt: skip
    assert all(not torch.isnan(got[n]).any() for n in ("dq", "dk", "dv"))


# #7038 / #7548 form: a one-thread sleep kernel resident on a side stream for the whole call; the gate is declared
# before the run (a stage that waited for every SM, or a whole-device launch, would stretch the call to the sleep).
# The caller's stream sleeps first, so the host has enqueued the whole call before its first launch runs: the events
# time the device work, not the launch overhead.  The g2 / g3 rows run the package's own two-stream groups; the
# split row a segment-parallel plan (pre-scan, stitch and pass 2); the "-2s" rows a split plan with the dhu_tc2 pass 2
# grouped on the device's own SM count (64 segments: HV 4 or 6 CTAs per segment leave the last round of every SM count
# from 148 to 216 partly idle): two groups beside tail_g3, one launch on the caller's stream beside tail_g2.
SIDE_STREAM_RATIO_MAX = 1.3
SIDE_STREAM_ITERS = 9
SPLIT_2S_PINS = {"scan_config": (64, 1), "segment_chunks": 4, "dhu_scan": "tc"}
# The "fwd-" rows take FP32 beta and run a forward on wy_a's A (route asserted on the reference call): "fwd-a" wy_a (one
# persistent CTA per SM over 2048 (chunk, key head) items) and fwd_ho_a on a serial table (64 sequences x 6 value heads:
# more CTAs than SMs on every device of the package), "fwd-split-a" / "fwd-split-w" the split forward on wy_a's A
# (fwd_summary_a, fwd_ho_a) / on the WY stage's w / u (fwd_summary, fwd_fused), each form forced (its summary on a side
# stream of its own; the shape class lowered to FWD_SPLIT_TEST_MIN_PIECE; sm_100a / sm_103a, skipped elsewhere).
SIDE_STREAM_ROWS = (
    ("g2", (4096,) * 4, (2, 4), {}),
    ("g3", (4096,) * 4, (2, 6), {}),
    ("split-g2", (16384,), (2, 4), {"scan_config": (64, 1), "segment_chunks": 16}),
    ("split-g2-2s", (16384,), (2, 4), SPLIT_2S_PINS),
    ("split-g3-2s", (16384,), (2, 6), SPLIT_2S_PINS),
    ("fwd-a", (1024,) * 64, (2, 6), {}),
    ("fwd-split-a", FWD_SPLIT_CASES["g3"][1], FWD_SPLIT_CASES["g3"][0], {}),
    ("fwd-split-w", FWD_SPLIT_CASES["g3"][1], FWD_SPLIT_CASES["g3"][0], {}),
)  # fmt: skip


@pytest.mark.parametrize("row", SIDE_STREAM_ROWS, ids=[r[0] for r in SIDE_STREAM_ROWS])
def test_forward_backward_beside_a_resident_side_stream_kernel(row, device, monkeypatch) -> None:  # fmt: skip
    row_id, seq_lens, heads, pins = row
    route = {"fwd-a": "a", "fwd-split-a": "split_a", "fwd-split-w": "split_w"}.get(row_id)  # fmt: skip
    if route == "a":
        _fwd_a_or_skip(device)
    elif route is not None:
        _split_form(route, device, monkeypatch)
    beta_dtype = torch.bfloat16 if route is None else torch.float32
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=287)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    with monkeypatch.context() as m:
        stages = _record_launches(m)
        want = _run(inputs, seq_lens, streams=1, **pins)
    if route is not None:
        _assert_fwd_route(stages, route)
    if row_id.endswith("-2s"):
        main = torch.cuda.current_stream(device).cuda_stream
        launches = _record_streams(monkeypatch)
        _assert_bitwise(_run(inputs, seq_lens, **pins), want, OUTPUT_NAMES)
        side_stages = [stage for stage, stream in launches if stream != main]
        grouped = _adjoint_two_streams("split", heads)
        # beside tail_g2_bf16 the pass 2 of more CTAs than SMs (64 x 4 = 256) may run as the wavefront pair (the
        # rule's split side, on dhu_tc2_wavefront_split), its tail on the side stream
        pair = set(WAVEFRONT_SPLIT_PAIR) <= {stage for stage, _stream in launches}
        assert not (grouped and pair), launches
        assert side_stages == (
            [_tc2(heads)] if grouped else ["tail_g2_bf16_wavefront"] if pair else []
        ), launches
        monkeypatch.undo()

    def step() -> None:
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = _call(leaves, seq_lens, device, **pins)
        torch.autograd.backward(o, inputs["do"])

    clock_khz = getattr(torch.cuda.get_device_properties(device), "clock_rate", 1_900_000)  # fmt: skip
    side = torch.cuda.Stream(device=device)

    def timed(lead_cycles: int, side_cycles: int = 0) -> float:
        torch.cuda.synchronize()
        if side_cycles:
            with torch.cuda.stream(side):
                torch.cuda._sleep(side_cycles)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        if lead_cycles:
            torch.cuda._sleep(lead_cycles)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(2):
        step()
    wall = sorted(timed(0) for _ in range(3))[1]
    # four times the call's wall time, in SM cycles (ms x kHz)
    lead = int(4.0 * wall * clock_khz)
    alone = sorted(timed(lead) for _ in range(SIDE_STREAM_ITERS))[
        SIDE_STREAM_ITERS // 2
    ]
    # the lead and about three times the call
    side_cycles = lead + int(3.0 * alone * clock_khz)
    beside = [timed(lead, side_cycles) for _ in range(SIDE_STREAM_ITERS)]
    # A stage that waits for every SM is stretched to the end of the side kernel (about three calls past the start
    # event) on every sample, so the fastest sample beside it carries the verdict; a median would also count the
    # host and clock noise of a shared node.
    ratio = min(beside) / alone
    assert ratio <= SIDE_STREAM_RATIO_MAX, (ratio, alone, beside)
    # the last call beside the side-stream kernel has the bits of the plain call
    got = {"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")}
    _assert_bitwise(got, want, GRAD_NAMES)


# --------------------------------------------------------------------------- #
# the forwards on the WY stage's A: route rules (no device)
# --------------------------------------------------------------------------- #


def _route_meta(seq_lens, segment_len=0, num_segments=None, fwd_split=None):
    """A stand-in for the ChunkMeta of a host-built table (``seq_lens``): the forward segment table and the split
    forward's tables the route rules read."""
    segs = types.SimpleNamespace(
        segment_len=segment_len,
        num_segments=len(seq_lens) if num_segments is None else num_segments,
    )
    return types.SimpleNamespace(
        seq_lens=seq_lens,
        num_seqs=len(seq_lens),
        fwd_segments=segs,
        fwd_split=fwd_split,
    )


def _route_tensors(h, hv, beta_dtype=torch.float32, *, offset=0):
    """q, k, v and beta of a call with ``h`` key and ``hv`` value heads: host q / k ``offset`` elements past the base
    of their allocation, meta v / beta (the route rules read shapes and dtypes only)."""
    tokens = 64
    flat = torch.empty(tokens * h * HEAD_DIM + offset, dtype=torch.bfloat16)
    q = flat[offset:].view(1, tokens, h, HEAD_DIM)
    v = torch.empty(1, tokens, hv, HEAD_DIM, dtype=torch.bfloat16, device="meta")
    return q, q, v, torch.empty(1, tokens, hv, dtype=beta_dtype, device="meta")


# the pins that keep a call off the forwards on wy_a's A: a prep pin (the prep forward), and the scan pins (the prep
# forward's scans run as pinned)
ROUTE_PINS = (
    {"prep": "tma"},
    {"prep": "plain"},
    {"scan_config": (64, 1)},
    {"segment_chunks": 0},
    {"segment_chunks": 4},
)
# the options that keep a call on them
ROUTE_KEEPING_OPTIONS = (
    {"dhu_scan": "tc"},
    {"lpt": False},
    {"streams": 1},
    {"save_state": False},
    {"host_cache": False},
)


def test_fwd_a_route_rule(monkeypatch) -> None:
    # the forward on the WY stage's A takes a host-built table whose forward segment table is serial, FP32 beta,
    # normalized q / k, a value-head group of at most three and no prep / scan_config / segment_chunks pin, on an
    # architecture of FWD_A_ARCHS whose records are delivered; it reads shapes, dtypes and pins, never an address
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    route = cake_chain.fwd_a_route
    opts = cake_chain.make_options()
    whole = _route_meta((300, 77, 1, 130))
    assert set(cake_chain.FWD_A_ARCHS) == set(ARCHES)
    for arch in ARCHES:
        assert route(whole, arch, *_route_tensors(16, 48), True, opts), arch
    assert not route(whole, "sm_120a", *_route_tensors(16, 48), True, opts)
    for heads in ((16, 48), (16, 32), (16, 16)):
        assert route(whole, "sm_100a", *_route_tensors(*heads), True, opts), heads
    assert cake_chain.WY_A_MAX_GROUP == 3
    # a group of four
    assert not route(whole, "sm_100a", *_route_tensors(12, 48), True, opts)
    assert not route(whole, "sm_100a", *_route_tensors(16, 48, torch.bfloat16), True, opts)  # fmt: skip
    assert not route(whole, "sm_100a", *_route_tensors(16, 48), False, opts)
    for pins in ROUTE_PINS:
        options = cake_chain.make_options(**pins)
        assert not route(whole, "sm_100a", *_route_tensors(16, 48), True, options), pins
    for pins in ROUTE_KEEPING_OPTIONS:
        options = cake_chain.make_options(**pins)
        assert route(whole, "sm_100a", *_route_tensors(16, 48), True, options), pins
    # q / k off the 16-byte boundary keep the route (the forward copies them to aligned rows)
    unaligned = _route_tensors(16, 48, offset=1)
    assert unaligned[0].data_ptr() % 16
    assert route(whole, "sm_100a", *unaligned, True, opts)
    # long sequences on a serial forward segment table route (a segment as long as the longest sequence is serial);
    # the same lengths on a split table do not
    pair = (11000, 12000, 9768)
    assert route(_route_meta(pair), "sm_107a", *_route_tensors(16, 48), True, opts)
    assert route(_route_meta(pair, 188, 3), "sm_107a", *_route_tensors(16, 48), True, opts)  # fmt: skip
    assert not route(_route_meta(pair, 64, 9), "sm_100a", *_route_tensors(16, 48), True, opts)  # fmt: skip
    assert route(_route_meta((4096,) * 64), "sm_100a", *_route_tensors(2, 6), True, opts)  # fmt: skip
    # the device-built table (no lengths) keeps the prep forward
    device_built = types.SimpleNamespace(
        seq_lens=None,
        num_seqs=2,
        fwd_segments=types.SimpleNamespace(segment_len=0, num_segments=2),
    )
    assert not route(device_built, "sm_100a", *_route_tensors(2, 6), True, opts)
    # a record that is not delivered, and the switch off
    monkeypatch.setattr(cake_chain, "_available", lambda stage, _arch: stage != "fwd_ho_a")  # fmt: skip
    assert not route(whole, "sm_100a", *_route_tensors(16, 48), True, opts)
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    monkeypatch.setattr(cake_chain, "FWD_A_ARCHS", ())
    assert not route(whole, "sm_100a", *_route_tensors(16, 48), True, opts)


def test_fwd_split_route_rule(monkeypatch) -> None:
    # the split forward takes a host-built table that carries its split tables (ChunkMeta.fwd_split) and the inputs of
    # the forward on wy_a's A, on an architecture of FWD_SPLIT_ARCHS; its form (wy_a's A, or the waves on the WY stage's
    # w / u) follows from the piece length alone (FWD_SPLIT_A_ARCHS, FWD_SPLIT_A_MIN_PIECE), within FWD_SPLIT_ARCHS
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    split = types.SimpleNamespace(length=63)
    meta = _route_meta((40000, 20000), 63, 16, fwd_split=split)
    route = cake_chain.fwd_split_route
    opts = cake_chain.make_options()
    assert set(cake_chain.FWD_SPLIT_ARCHS) == {"sm_100a", "sm_103a"}
    for arch in ARCHES:
        got = route(meta, arch, *_route_tensors(2, 6), True, opts)
        assert got is (arch in cake_chain.FWD_SPLIT_ARCHS), arch
    no_tables = _route_meta((40000, 20000), 63, 16)
    assert not route(no_tables, "sm_100a", *_route_tensors(2, 6), True, opts)
    assert not route(meta, "sm_100a", *_route_tensors(2, 6, torch.bfloat16), True, opts)  # fmt: skip
    assert not route(meta, "sm_100a", *_route_tensors(2, 6), False, opts)
    # a group of four
    assert not route(meta, "sm_100a", *_route_tensors(2, 8), True, opts)
    for pins in ROUTE_PINS:
        options = cake_chain.make_options(**pins)
        assert not route(meta, "sm_100a", *_route_tensors(2, 6), True, options), pins
    for pins in ROUTE_KEEPING_OPTIONS:
        options = cake_chain.make_options(**pins)
        assert route(meta, "sm_100a", *_route_tensors(2, 6), True, options), pins
    assert route(meta, "sm_100a", *_route_tensors(2, 6, offset=1), True, opts)
    # the form: B200 runs every split on wy_a's A, GB300 from pieces of 96 chunks, R200 none
    form = cake_chain._fwd_split_a
    assert form("sm_100a", 16) and form("sm_100a", 63) and form("sm_100a", 200)
    assert not form("sm_103a", 63) and not form("sm_103a", 95)
    assert form("sm_103a", 96) and form("sm_103a", 200)
    assert not form("sm_107a", 200)
    # each form needs its own records: without fwd_fused only the form on wy_a's A runs
    monkeypatch.setattr(cake_chain, "_available", lambda stage, _arch: stage != "fwd_fused")  # fmt: skip
    assert route(meta, "sm_100a", *_route_tensors(2, 6), True, opts)
    assert not route(meta, "sm_103a", *_route_tensors(2, 6), True, opts)
    # the form on wy_a's A is subordinate to FWD_SPLIT_ARCHS, the switch the non-deterministic package's tests set
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_ARCHS", ())
    assert not form("sm_100a", 200)
    assert not route(meta, "sm_100a", *_route_tensors(2, 6), True, opts)


def test_the_route_switches_are_set_by_name() -> None:
    # the non-deterministic package's tests turn this package's forwards on wy_a's A off through the names of
    # PREP_FORWARD_OFF alone, each where this package has it (getattr not None): each is set, of its off value's type,
    # and on; the other route constants are subsets of them, so the three off values turn every route off
    for name, off in PREP_FORWARD_OFF:
        value = getattr(cake_chain, name, None)
        assert value is not None and type(value) is type(off), (name, value)
        assert value != off, (name, value)
    assert set(cake_chain.FWD_SPLIT_A_ARCHS) <= set(cake_chain.FWD_SPLIT_ARCHS)
    assert set(cake_chain.FWD_SPLIT_ARCHS) <= set(cake_chain.FWD_A_ARCHS)
    assert set(cake_chain.FWD_A_SERIAL_KEEP) <= set(cake_chain.FWD_A_ARCHS)
    assert set(cake_chain.FWD_SPLIT_A_MIN_PIECE) <= set(cake_chain.FWD_SPLIT_A_ARCHS)
    assert set(cake_plan.FORWARD_SPLIT_MIN_PIECE) == set(cake_chain.FWD_SPLIT_ARCHS)
    for name in (
        "FWD_A_ARCHS",
        "FWD_A_SERIAL_KEEP",
        "FWD_SPLIT_ARCHS",
        "WY_A_MAX_GROUP",
    ):
        assert name in cake_chain.__all__, name


def test_the_tape_levels_and_placeholders() -> None:
    # every architecture keeps the T_full tape (w, h, v_new; u never) under save_state and none without it; a
    # placeholder has the shape and dtype of the entry it stands for, every stride 0 and the one element of a base
    # shared by every placeholder of its (device, dtype), and is told from a state tensor by its strides alone
    assert cake_plan.STATE_NAMES == ("w", "u", "h", "v_new")
    for arch in ARCHES:
        assert cake_plan.tape_state(arch, True) == ("w", "h", "v_new"), arch
        assert cake_plan.tape_state(arch, False) == (), arch
    p = cake_plan.placeholder((5, 3, HEAD_DIM), torch.bfloat16, "cpu")
    assert p.shape == (5, 3, HEAD_DIM) and p.dtype == torch.bfloat16
    assert not any(p.stride()) and p.untyped_storage().nbytes() == p.element_size()
    other = cake_plan.placeholder((7, 1, 2, HEAD_DIM), torch.bfloat16, "cpu")
    assert other.untyped_storage().data_ptr() == p.untyped_storage().data_ptr()
    assert cake_plan.is_placeholder(p) and cake_plan.is_placeholder(other)
    for real in (torch.empty(5, 3, HEAD_DIM), torch.empty(1, 1, HEAD_DIM), torch.empty(0, 3, HEAD_DIM)):  # fmt: skip
        assert not cake_plan.is_placeholder(real), real.shape


def test_fwd_split_lists() -> None:
    # pieces by segment_bounds; wave 1 walks each split sequence's first piece (its end chunk in first_end), the
    # summary the middle pieces, wave 2 every other piece and each sequence of one piece (an empty one too)
    scs, seq, sss, first_end, first, middle, rest = cake_chain.forward_split_lists(
        (10, 3, 0, 7), 4
    )
    assert scs == [0, 3, 6, 10, 13, 13, 16, 20]
    assert seq == [0, 0, 0, 1, 2, 3, 3]
    assert sss == [0, 3, 4, 5, 7]
    assert first_end == [0, 3, 13, 13, 16]
    assert first == [0, 5] and middle == [1] and rest == [1, 2, 3, 4, 6]
    lists = cake_chain.forward_split_lists((5, 2), 8)
    assert lists == ([0, 5, 7], [0, 1], [0, 1, 2], [0, 5, 7], [], [], [0, 1])


# (H, HV), sequence lengths and whether the forward runs the split, for the rows that set the split's shape class:
# long sequences at HV 6 (pieces of 132..158 chunks) and the pair rows (HV 24: 63; HV 12: 38; HV 6: 20; HV 3: 11)
FWD_SPLIT_CLASS_ROWS = {
    "long3_hv6": ((2, 6), (131071, 87531, 43385), True),
    "long2_hv6": ((2, 6), (128147, 131027), True),
    "long6_hv6": ((2, 6), (126565, 32714, 33696, 15081, 22979, 10782), True),
    "pair_hv24": ((8, 24), _PAIR_LAYOUT, True),
    "pair_hv12": ((4, 12), _PAIR_LAYOUT, False),
    "pair_hv6": ((2, 6), _PAIR_LAYOUT, False),
    "pair_hv3": ((1, 3), _PAIR_LAYOUT, False),
}
# the split rows whose split forward keeps the waves on the WY stage's w / u, per architecture (FWD_SPLIT_A_MIN_PIECE):
# on sm_103a the pair row of HV 24 (63-chunk pieces); every other split row runs on wy_a's A
FWD_SPLIT_W_FORM_ROWS = {"sm_103a": frozenset({"pair_hv24"})}


@pytest.mark.parametrize("sm_count", [148, 152])
@pytest.mark.parametrize("row", sorted(FWD_SPLIT_CLASS_ROWS))
def test_fwd_split_shape_class(row, sm_count, monkeypatch) -> None:
    # the forward runs the split on the long rows and on the pair row of HV 24 at the SM counts of B200 (148) and
    # GB200 / GB300 (152); the pair rows of HV 12, 6 and 3 keep the prep forward's split scan.  The class and the
    # split's form are read from the host piece table (lengths, HV, SM count) alone
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    heads, seq_lens, split = FWD_SPLIT_CLASS_ROWS[row]
    table = cake_plan.pieces(seq_lens, heads[1], sm_count)
    assert table.split, table
    for arch in cake_chain.FWD_SPLIT_ARCHS:
        plan = cake_plan.host_plan(tuple(seq_lens), heads[1], arch, sm_count)
        assert plan.pieces == table and plan.sm_count == sm_count, (arch, plan)
        got = cake_plan.forward_split_length(arch, table)
        assert (got is not None) is split, (arch, table.length)
        assert got in (None, table.length)
        if got is not None:
            a_form = row not in FWD_SPLIT_W_FORM_ROWS.get(arch, frozenset())
            assert cake_chain._fwd_split_a(arch, got) is a_form, (arch, got)
    # the first wave fits one CTA per SM: (pieces - 1, at least 1, per sequence) x HV
    assert table.first_wave <= sm_count, table


# The forward table rule of FWD_A_SERIAL_KEEP on the priced rows at 208, 212 and 216 SMs (R200, sm_107a): row ->
# whether its split forward table gives way to the planner's cheapest serial table.  At 212 SMs the 3 sequences of
# 11000 / 12000 / 9768 tokens with HV 48 (144 serial CTAs, fill 0.68, the serial table 1.06 x the split table's price)
# take it; with HV 24 (72 CTAs, 1.51 x) and fewer value heads, and the long rows, the split table stays (64 x 4096 is
# serial without the rule).  At 208 SMs the same rows take it.  At 216 SMs every row keeps its split table: A16 fills
# 0.67 of the SMs but its serial table prices 1.42 x the split table's, above the 1.10 limit (the FWD_A_SERIAL lines,
# pytest -s, report the ratios of every SM count).
_FWD_A_SERIAL_KEPT = {
    "A16": True,
    "h8_hv24": False,
    "h4_hv12": False,
    "h2_hv6": False,
    "h1_hv3": False,
    "issue": False,
    "P3": False,
    "P1": False,
    "64x4096": False,
}
FWD_A_SERIAL_ROWS = {
    208: _FWD_A_SERIAL_KEPT,
    212: _FWD_A_SERIAL_KEPT,
    216: dict.fromkeys(_FWD_A_SERIAL_KEPT, False),
}


@pytest.mark.parametrize(
    "arch, sm_count",
    [("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 208), ("sm_107a", 212), ("sm_107a", 216)],
    ids=["B200", "GB300", "R200-208", "R200-212", "R200-216"],
)  # fmt: skip
def test_fwd_a_serial_keep_rule(arch, sm_count, monkeypatch) -> None:
    # a split forward table priced within FWD_A_SERIAL_KEEP's factor of the cheapest serial table (whose CTAs fill
    # enough of the SMs) becomes that serial table for a call the forward on wy_a's A serves (a_inputs), whatever
    # CHAIN_PLAN prices, so fwd_a_route's table condition holds; the adjoint table does not change; a call it does not
    # serve (a_inputs False: BF16 beta, ...) and the rule off keep the tables of the package without it; plan pins keep
    # the planner's tables
    _priced_device(monkeypatch, arch, sm_count)
    options = cake_chain.make_options()
    kept, ratios, tables = set(), {}, {}
    for row, seq_lens, num_heads, num_v_heads in PRICED_ROWS:
        chunks = tuple(-(-n // 64) for n in seq_lens)
        a_inputs = num_v_heads <= cake_chain.WY_A_MAX_GROUP * num_heads
        (fwd, bwd), bwd_tc = cake_chain.host_plans(seq_lens, num_v_heads, 0, options, num_heads=num_heads,
                                                   a_inputs=a_inputs)  # fmt: skip
        plain = cake_chain.host_plans(
            seq_lens, num_v_heads, 0, options, num_heads=num_heads
        )
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "FWD_A_SERIAL_KEEP", {})
            off = cake_chain.host_plans(seq_lens, num_v_heads, 0, options, num_heads=num_heads,
                                        a_inputs=a_inputs)  # fmt: skip
        assert plain == off, row
        (fwd0, bwd0), tc0 = off
        assert (bwd, bwd_tc) == (bwd0, tc0), row
        planner = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )[0]
        rec = cake_chain.fwd_serial_keep_pricing(seq_lens, num_v_heads, 0, planner)
        ratios[row], tables[row] = rec["ratio"], fwd
        if fwd == fwd0:
            assert not (a_inputs and rec["select"]), (row, rec)
            continue
        kept.add(row)
        assert a_inputs and rec["select"] and rec["ratio"] <= rec["limit"], (row, rec)
        assert rec["fill"] >= rec["min_fill"], (row, rec)
        assert fwd == rec["serial"] and fwd.segment_len == 0, (row, fwd)
        assert _segments(chunks, fwd0) > len(chunks), (row, fwd0)
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=(fwd, bwd))
        assert meta.fwd_segments.num_segments == len(seq_lens), row
    print(f"FWD_A_SERIAL {arch} sms={sm_count} kept={sorted(kept)} ratios={ {r: x for r, x in ratios.items() if x} }")  # fmt: skip
    if arch not in cake_chain.FWD_A_SERIAL_KEEP:
        assert kept == set()
    for row, want in FWD_A_SERIAL_ROWS.get(sm_count, {}).items():
        assert (row in kept) is want, (row, sorted(kept), ratios)
    if sm_count in FWD_A_SERIAL_ROWS:
        if FWD_A_SERIAL_ROWS[sm_count]["A16"]:
            assert 1.0 < ratios["A16"] <= 1.10 < ratios["h8_hv24"], ratios
        else:
            assert ratios["A16"] > 1.10, ratios
        chunks = tuple(-(-n // 64) for n in _PAIR_LAYOUT)
        assert _segments(chunks, tables["h8_hv24"]) > len(chunks), tables["h8_hv24"]
    pinned = cake_chain.make_options(scan_config=(64, 1), segment_chunks=64)
    planner = cake_chain.scan_plans(sum(_PAIR_LAYOUT), 3, 48, 0, _PAIR_LAYOUT, scan_config=(64, 1),
                                    segment_chunks=64)  # fmt: skip
    assert planner[0].segment_len == 64, planner
    got = cake_chain.host_plans(_PAIR_LAYOUT, 48, 0, pinned, num_heads=16, a_inputs=True)  # fmt: skip
    assert got == (tuple(planner), False), got


def test_the_route_switches_key_the_chunk_tables(monkeypatch) -> None:
    # the chunk tables of a layout differ with the forward's serial-keep rule (R200 at 212 SMs, the pair row of HV
    # 48): whether the call meets the inputs of the forwards on wy_a's A, and each switch of PREP_FORWARD_OFF, enter
    # the table cache's key, so a table planned under one is not served under the other (the non-deterministic
    # package's tests switch the routes off after this package's unpinned calls, with its tables cached)
    _priced_device(monkeypatch, "sm_107a", 212)
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    monkeypatch.setattr(cake_chain, "_host_chunk_meta", lambda seq_lens, plans, _index, lpt: cake_chain.build_chunk_meta(
        seq_lens, "cpu", scan_plans=plans, lpt=lpt))  # fmt: skip
    seq_lens, num_heads, num_v_heads = _PAIR_LAYOUT, 16, 48
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    options = cake_chain.make_options()

    def meta(a_inputs):
        return cake_chain._chunk_meta(cu, cu, 1, sum(seq_lens), 0, num_v_heads, options, num_heads=num_heads,
                                      a_inputs=a_inputs)  # fmt: skip

    kept = meta(True)
    assert kept.fwd_segments.segment_len == 0
    assert kept.fwd_segments.num_segments == len(seq_lens)
    assert kept.fwd_split is None  # no split forward on R200
    plain = meta(False)
    assert plain is not kept and plain.fwd_segments.num_segments > len(seq_lens)
    for name, off in PREP_FORWARD_OFF:
        with monkeypatch.context() as m:
            m.setattr(cake_chain, name, off)
            got = meta(True)
            assert got is not kept, name
            # the serial table needs the forward on wy_a's A and the rule
            serial = name == "FWD_SPLIT_ARCHS"
            assert (got.fwd_segments.num_segments == len(seq_lens)) is serial, name
    # each call hits its own table afterwards
    assert meta(True) is kept and meta(False) is plain


# --------------------------------------------------------------------------- #
# the forwards on the WY stage's A: on a device
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _deterministic_algorithms(mode: bool):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(mode)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def _saved_list(o) -> tuple:
    """The saved tensors of the package's autograd node behind ``o``."""
    node = o.grad_fn
    while node is not None and not type(node).__name__.startswith("CakeGDNDetFunction"):
        node = node.next_functions[0][0]
    assert node is not None, o.grad_fn
    return node.saved_tensors


@pytest.mark.parametrize("case", sorted(FWD_A_CASES))
def test_fwd_a_matches_the_fp64_recurrence(case, device, monkeypatch) -> None:
    # wy_a and fwd_ho_a replace the prep, the WY stage and fwd_ho; o, the final state and every gradient (the initial
    # state's too) within the accuracy gate of the FP64 recurrence (REL_RMS_MAX, MAX_ABS_REL)
    _fwd_a_or_skip(device)
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    heads, seq_lens = FWD_A_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=401)
    h0, dht = _state_pair(seq_lens, heads, device, 402)
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    assert _forward_stages(stages) == ["wy_a", "fwd_ho_a"], stages
    _assert_fwd_route(stages, "a")
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
@pytest.mark.parametrize("route", FWD_SPLIT_ROUTES)
def test_fwd_split_matches_the_fp64_recurrence(route, case, device, monkeypatch) -> None:  # fmt: skip
    # the split forward (on wy_a's A and on the WY stage's w / u) replaces the prep, the WY stage, the pre-scan and
    # fwd_ho; o, the final state and every gradient (the initial state's too) within the accuracy gate of the FP64
    # recurrence (REL_RMS_MAX, MAX_ABS_REL)
    heads, seq_lens = _fwd_route(route, device, monkeypatch, case)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=421)
    h0, dht = _state_pair(seq_lens, heads, device, 422)
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    assert _forward_stages(stages) == _split_stages(), stages
    _assert_fwd_route(stages, route)
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


# (route, layout) of the accuracy tests beside the prep forward: every layout of the route's cases
FWD_ACCURACY_CASES = tuple(("a", case) for case in sorted(FWD_A_CASES)) + tuple(
    (route, case) for route in FWD_SPLIT_ROUTES for case in sorted(FWD_SPLIT_CASES)
)


@pytest.mark.parametrize(
    "route, case", FWD_ACCURACY_CASES, ids=[f"{r}-{c}" for r, c in FWD_ACCURACY_CASES]
)
def test_fwd_routes_within_the_budgets_of_the_prep_forward(
    route, case, device, monkeypatch
) -> None:
    # the forwards on wy_a's A against the prep forward (the switches of PREP_FORWARD_OFF off: the prep, the WY stage,
    # fwd_ho, with the pre-scan and the stitch on a split table) on the same inputs: o, the final state and every
    # gradient (the initial state's too) within the accuracy budgets (REL_RMS_MAX, MAX_ABS_REL), the prep forward's
    # results as the reference
    heads, seq_lens = _fwd_route(route, device, monkeypatch, case)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=403)
    h0, dht = _state_pair(seq_lens, heads, device, 404)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    _assert_fwd_route(stages, route)
    _prep_forward_off(monkeypatch)
    stages.clear()
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    assert "wy_a" not in stages and _preps(stages), stages
    _assert_close_to_reference(got, want, names)


def _explicit_norm_run(inputs, seq_lens, device) -> dict:
    """The explicit normalization of #7817 on the same operands: the caller normalizes q / k in FP32 as the FP64
    reference does (``x / sqrt(sum x^2 + L2NORM_EPS)``), rounds them to BF16 and calls with
    ``use_qk_l2norm_in_kernel=False``; dq / dk flow back through the caller's normalization to the raw q / k."""
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }

    def normalized(x):
        x32 = x.float()
        return (x32 / torch.sqrt((x32 * x32).sum(-1, keepdim=True) + L2NORM_EPS)).to(x.dtype)  # fmt: skip

    cu = _cu(seq_lens)
    o, _ = cake_gdn_det(normalized(leaves["q"]), normalized(leaves["k"]), leaves["v"], leaves["g"], leaves["beta"],
                        scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                        cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),
                        use_qk_l2norm_in_kernel=False)  # fmt: skip
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    return {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }


@pytest.mark.parametrize("route", FWD_ROUTES)
def test_fwd_routes_in_kernel_norm_beside_the_explicit_norm(
    route, device, monkeypatch
) -> None:
    # #7817 on the routes (in-kernel normalization no less accurate against FP64 than the explicit path), on the same
    # FP32-beta inputs: the in-kernel normalization on the route's forward (route asserted) and the explicit one (the
    # caller's FP32 normalization with use_qk_l2norm_in_kernel=False, which keeps the prep forward), each against the
    # FP64 recurrence on the raw q / k with FP64 normalization.  Both arms hold the FP64 gate (REL_RMS_MAX,
    # MAX_ABS_REL); the ACC7817 lines (pytest -s) report both arms per output for the #7817 comparison, whose statistic
    # (seeds, RMS or worst element) the issue does not fix: this test sets no bar of its own between the arms
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=481)
    stages = _record_launches(monkeypatch)
    in_kernel = _run(inputs, seq_lens)
    _assert_fwd_route(stages, route)
    stages.clear()
    explicit = _explicit_norm_run(inputs, seq_lens, device)
    assert "wy_a" not in stages and _preps(stages), stages
    ref = _reference(inputs, seq_lens)
    index = cake_chain._device_index(device)
    tag = f"ACC7817 {_current_arch()} sms={cake_chain._device_props(index)[0]} route={route}"
    for name in OUTPUT_NAMES:
        arms = {"in_kernel": in_kernel, "explicit": explicit}
        fields = []
        for arm, res in arms.items():
            m = _metrics(res[name], ref[name])
            fields.append(f"{arm} rel_rms={m['rel_rms']:.4e} max_rel={m['max_abs'] / max(m['max_ref'], 1e-30):.4e}")  # fmt: skip
        print(f"{tag} {name} " + " ".join(fields))
    _assert_close_to_reference(in_kernel, ref, OUTPUT_NAMES)
    _assert_close_to_reference(explicit, ref, OUTPUT_NAMES)


def test_fwd_split_a_tape_matches_the_w_form(device, monkeypatch) -> None:
    # the two forms of the split forward on the same inputs: on wy_a's A the waves write the T_full tape themselves
    # (fwd_ho_a word 11), w the bits of the WY stage's w on the same A, h, v_new and o within a relative L2 of 1e-2 of
    # the waves on the WY stage's w / u (fwd_fused), u a placeholder in both
    heads, seq_lens = _fwd_route("split_a", device, monkeypatch)
    arch = _current_arch()
    assert set(cake_plan.tape_state(arch, True)) == {"w", "h", "v_new"}, arch
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=437)

    def saved_state() -> dict:
        leaves = {
            n: inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        }
        o, _ = _call(leaves, seq_lens, device)
        saved = _saved_list(o)
        return {
            "o": o.detach(),
            **dict(zip(cake_plan.STATE_NAMES, saved[9:], strict=True)),
        }

    stages = _record_launches(monkeypatch)
    got = saved_state()
    assert _forward_stages(stages) == FWD_SPLIT_A_STAGES, stages
    _split_form("split_w", device, monkeypatch)
    stages.clear()
    want = saved_state()
    assert _forward_stages(stages) == FWD_SPLIT_STAGES, stages
    assert cake_plan.is_placeholder(got["u"]) and cake_plan.is_placeholder(want["u"])
    assert torch.equal(got["w"], want["w"])
    for name in ("o", "h", "v_new"):
        a, b = got[name].float(), want[name].float()
        assert bool(torch.isfinite(a).all()), name
        rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
        assert rel <= 1e-2, (name, rel)


@pytest.mark.parametrize("route, case", FWD_ROUTE_CASES, ids=FWD_ROUTE_CASE_IDS)
def test_fwd_routes_run_to_run_bitwise(route, case, device, monkeypatch) -> None:
    # every launch of the forwards on wy_a's A has one writer per output element and a fixed order (wy_a's persistent
    # CTAs take their items in a fixed order; the split forward's wave 1 and summary write disjoint segments): run to
    # run, with tables built for the call, and on one stream as on two (the split forward's summary then on the
    # caller's stream), the same bits on every output, the final state and the initial state's gradient
    heads, seq_lens = _fwd_route(route, device, monkeypatch, case)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=425)
    h0, dht = _state_pair(seq_lens, heads, device, 426)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    _assert_fwd_route(stages, route)
    for pins in ({}, {}, {"host_cache": False}, {"streams": 1}):
        stages.clear()
        got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, **pins)  # fmt: skip
        _assert_fwd_route(stages, route)
        _assert_bitwise(got, want, names)


@pytest.mark.parametrize("route", FWD_ROUTES)
def test_fwd_routes_lpt_matches_the_caller_order(route, device, monkeypatch) -> None:
    # the host route's longest-first order (the chunk tables, the walks and the split forward's pieces in that order,
    # the state rows gathered and scattered) against the caller's order: the same bits on every output, the final
    # state and the initial state's gradient (the layout reversed, so the two orders differ)
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    seq_lens = tuple(reversed(seq_lens))
    assert cake_chain.lpt_sequence_order(seq_lens) is not None
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=427)
    h0, dht = _state_pair(seq_lens, heads, device, 428)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    stages = _record_launches(monkeypatch)
    runs = {}
    for lpt in (True, False):
        stages.clear()
        runs[lpt] = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, lpt=lpt)  # fmt: skip
        _assert_fwd_route(stages, route)
    _assert_bitwise(runs[True], runs[False], names)


@pytest.mark.parametrize("route", FWD_ROUTES)
def test_fwd_routes_under_deterministic_algorithms(route, device, monkeypatch) -> None:
    # every launch is fixed-order and the host ops of the forwards on wy_a's A (the aligned copies, index_fill_,
    # index_select, index_copy_) have deterministic implementations: under torch.use_deterministic_algorithms(True)
    # (uninitialized memory filled, warnings as errors) the forward, the backward and a non-reentrant checkpointed step
    # run without an alert, with the bits of the default mode
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=455)
    h0, dht = _state_pair(seq_lens, heads, device, 456)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    plain = _run(inputs, seq_lens)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)

    def layer(q, k, v, g, beta):
        o, _ = cake_gdn_det(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                            cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        return o

    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    with _deterministic_algorithms(True), warnings.catch_warnings():
        warnings.simplefilter("error")
        stages = _record_launches(monkeypatch)
        got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
        o = checkpoint(layer, *leaves.values(), use_reentrant=False)
        (o.float() * inputs["do"].float()).sum().backward()
        torch.cuda.synchronize()
    # the call, the checkpoint's first pass and its recompute
    _assert_fwd_route(stages, route, calls=3)
    _assert_bitwise(got, want, names)
    ckpt = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_bitwise(ckpt, plain, OUTPUT_NAMES)


@pytest.mark.parametrize("route", FWD_ROUTES)
def test_fwd_routes_under_cuda_graph_capture(route, device, monkeypatch) -> None:
    # forward and backward of a warmed-up call captured into a CUDA graph (fresh workspaces, the cached tables kept for
    # the process, the split forward's side stream forked and joined inside the capture): the replays give the eager
    # bits and launch nothing from the host
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=441)
    eager = _run(inputs, seq_lens)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = cake_gdn_det(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                            scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                            use_qk_l2norm_in_kernel=True)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
        return o

    # warm up on a side stream (the tables, the workspaces and the libraries), then capture forward and backward
    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream(device).wait_stream(warmup)
    stages = _record_launches(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = step()
    captured = len(stages)
    _assert_fwd_route(stages, route)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    got = {
        "o": o,
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_bitwise(got, eager, OUTPUT_NAMES)
    assert len(stages) == captured  # replays launch nothing from the host


@pytest.mark.parametrize("route", FWD_ROUTES)
def test_fwd_routes_tape_off_keeps_the_output_bits(route, device, monkeypatch) -> None:
    # a call that records no backward (no_grad, or no input that requires grad) writes no tape (tape word 0 on every
    # launch of its scan); o and the final state are the bits of the taped call (fwd_ho_a word 11: h, v_new and w;
    # fwd_fused word 3: h and v_new, w being the WY stage's)
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=405)
    h0, _dht = _state_pair(seq_lens, heads, device, 406)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    kwargs = {
        "scale": 1.0 / math.sqrt(HEAD_DIM),
        "initial_state": h0,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_cpu": cu_seqlens_cpu,
    }
    names = ("q", "k", "v", "g", "beta")
    launches = _record_launch_values(monkeypatch)
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]
    taped = cake_gdn_det(*leaves, **kwargs)
    with torch.no_grad():
        no_grad = cake_gdn_det(*leaves, **kwargs)
    plain = cake_gdn_det(*(inputs[n] for n in names), **kwargs)
    torch.cuda.synchronize()
    _assert_fwd_route([stage for stage, _values in launches], route, calls=3)
    word, wave = _taped_word(route), _route_wave(route)
    assert word == (11 if wave == "fwd_ho_a" else 3), (wave, word)
    per_call = 1 if route == "a" else 2
    words = _tape_words(launches, wave)
    assert words == [word] * per_call + [0] * (2 * per_call), words
    for got in (no_grad, plain):
        _assert_bitwise(
            {"o": got[0], "final_state": got[1]},
            {"o": taped[0].detach(), "final_state": taped[1].detach()},
            ("o", "final_state"),
        )


@pytest.mark.parametrize("save_state", [True, False], ids=["saved", "recompute"])
@pytest.mark.parametrize("route", ("prep",) + FWD_ROUTES)
def test_fwd_routes_saved_list_keeps_its_arity(
    route, save_state, device, monkeypatch
) -> None:
    # every forward saves 13 entries: qn, kn, rstd_q, rstd_k, v, g_cs, beta32, A, the initial state, then w, u, h and
    # v_new with the shapes, dtypes and device of the prep forward's state: the state of the device's tape level
    # (T_full: w, h and v_new) under save_state, a storage-free placeholder (every stride 0) for the rest (u always),
    # so a checkpoint's two passes save lists of the same metadata; the backward reads only the kept state, with the
    # bits of the same call through _run.  "prep": the switches of PREP_FORWARD_OFF off, on the layout of "a"
    if route == "prep":
        _fwd_a_or_skip(device)
        heads, seq_lens = FWD_A_CASES["g3"]
        _prep_forward_off(monkeypatch)
    else:
        heads, seq_lens = _fwd_route(route, device, monkeypatch)
    arch = _current_arch()
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=411)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    stages = _record_launches(monkeypatch)
    o, _ = _call(leaves, seq_lens, device, save_state=save_state)
    if route == "prep":
        assert "wy_a" not in stages and _preps(stages), stages
    else:
        _assert_fwd_route(stages, route)
    saved = _saved_list(o)
    assert len(saved) == 9 + len(cake_plan.STATE_NAMES) == 13, len(saved)
    assert saved[8] is None  # no initial state
    kept = cake_plan.tape_state(arch, save_state)
    total, hv = sum(seq_lens), heads[1]
    chunks = sum(-(-n // 64) for n in seq_lens)
    shapes = {"w": (total, hv, HEAD_DIM), "u": (1, total, hv, HEAD_DIM), "h": (chunks, hv, HEAD_DIM, HEAD_DIM),
              "v_new": (1, total, hv, HEAD_DIM)}  # fmt: skip
    for name, t in zip(cake_plan.STATE_NAMES, saved[9:], strict=True):
        placeholder = (
            all(s == 0 for s in t.stride())
            and t.untyped_storage().nbytes() == t.element_size()
        )
        assert placeholder is (name not in kept), (name, t.stride())
        assert cake_plan.is_placeholder(t) is placeholder, name
        assert tuple(t.shape) == shapes[name] and t.dtype == torch.bfloat16, (
            name,
            t.shape,
            t.dtype,
        )
        assert t.device == o.device, name
    if route != "prep":  # wy_a's A
        assert saved[7].shape == (total, hv, 64) and saved[7].dtype == torch.bfloat16
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    got = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_bitwise(got, _run(inputs, seq_lens, save_state=save_state), OUTPUT_NAMES)


@pytest.mark.parametrize("reentrant", [False, True], ids=["non-reentrant", "reentrant"])
@pytest.mark.parametrize("route, case", FWD_ROUTE_CASES, ids=FWD_ROUTE_CASE_IDS)
def test_fwd_routes_checkpoint_writes_the_tape_once(
    route, case, reentrant, device, monkeypatch
) -> None:
    # under torch.utils.checkpoint (non-reentrant and reentrant) the first pass keeps no tape (tape word 0: its saved
    # tensors are dropped either way) and retains nothing beyond its outputs (#7039), and the recompute writes the
    # tape: o, the final state and every gradient (the initial state's too) are the bits of the plain call
    heads, seq_lens = _fwd_route(route, device, monkeypatch, case)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=413)
    h0, dht = _state_pair(seq_lens, heads, device, 414)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    leaves["h0"] = h0.detach().clone().requires_grad_(True)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)

    def layer(q, k, v, g, beta, h):
        return cake_gdn_det(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), initial_state=h,
                            output_final_state=True, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                            use_qk_l2norm_in_kernel=True)  # fmt: skip

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        return checkpoint(layer, *leaves.values(), use_reentrant=reentrant)

    def backward(o, final) -> None:
        torch.autograd.backward((o, final), (inputs["do"], dht))
        torch.cuda.synchronize()

    for _ in range(2):  # the libraries, the tables and the workspaces of this layout
        backward(*step())
    for leaf in leaves.values():
        leaf.grad = None
    launches = _record_launch_scalars(monkeypatch)
    gc.collect()
    torch.cuda.synchronize()
    requested, _blocks = _allocator_bytes(device)
    o, final = step()
    torch.cuda.synchronize()
    requested_after, _blocks_after = _allocator_bytes(device)
    outputs = o.untyped_storage().nbytes() + final.untyped_storage().nbytes()
    retained = requested_after - requested - outputs
    assert retained == 0, (
        f"a checkpointed forward retains {retained} requested bytes beyond its outputs"
    )
    backward(o, final)
    _assert_fwd_route([stage for stage, _values in launches], route, calls=2)
    per_call = 1 if route == "a" else 2
    words = _tape_words(launches, _route_wave(route))
    assert words == [0] * per_call + [_taped_word(route)] * per_call, words
    got = {
        "o": o.detach(),
        "final_state": final.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
        "dh0": leaves["h0"].grad,
    }
    _assert_bitwise(got, want, names)


@pytest.mark.parametrize(
    "tape, save_state",
    [(True, True), (True, False), (False, True)],
    ids=["taped", "unsaved", "tape-off"],
)
@pytest.mark.parametrize("route", FWD_ROUTES)
def test_low_level_fwd_routes_round_trip(
    route, tape, save_state, device, monkeypatch
) -> None:
    # the low-level forward on a host-built table that carries the routes' tables (a_inputs) runs the route's forward;
    # its tape holds the state of the device's tape level under save_state (T_full: w, h and v_new) when the call
    # keeps a tape and None for the rest (u always), never a placeholder; the low-level backward rebuilds what the tape
    # lacks (the WY stage on the tape's A, then the forward scan) and nothing else: the outputs and gradients are the
    # bits of the autograd call that keeps the same state
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=415)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    options = cake_chain.make_options(save_state=save_state)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    meta = cake_chain._chunk_meta(cu.to(device), cu, 1, sum(seq_lens), index, heads[1], options, num_heads=heads[0],
                                  a_inputs=True)  # fmt: skip
    assert (meta.fwd_split is not None) is (route != "a")
    q, k, v, g, beta, do = (inputs[n] for n in ("q", "k", "v", "g", "beta", "do"))
    stages = _record_launches(monkeypatch)
    fwd = cake_chain.cake_gdn_det_train_forward(
        q, k, v, g, beta, meta=meta, arch=arch, options=options, tape=tape
    )
    assert fwd["mode"] == "det"
    assert fwd["forward"] == ("fwd_ho_a" if route == "a" else "fwd_split")
    _assert_fwd_route(stages, route)
    kept = cake_plan.tape_state(arch, save_state) if tape else ()
    for name in cake_plan.STATE_NAMES:
        if name in kept:
            assert fwd[name] is not None and not cake_plan.is_placeholder(fwd[name])
        else:
            assert fwd[name] is None, name
    start = len(stages)
    grads = cake_chain.cake_gdn_det_train_backward(fwd, do, dbeta_dtype=beta.dtype)
    torch.cuda.synchronize()
    backward = stages[start:]
    rebuilt = cake_chain.wy_stage(arch) in backward and any(
        stage.startswith("fwd_h_") for stage in backward
    )
    assert rebuilt == (len(kept) == 0), backward
    want = _run(inputs, seq_lens, save_state=bool(kept))
    got = {"o": fwd["output"], **{name: grads[name] for name in GRAD_NAMES}}
    for name in OUTPUT_NAMES:
        assert torch.equal(got[name].reshape(want[name].shape), want[name]), name


@pytest.mark.parametrize("streams", [1, 2])
@pytest.mark.parametrize("route", FWD_ROUTES)
def test_fwd_routes_accumulate_into_a_nonzero_grad(
    route, streams, device, monkeypatch
) -> None:
    # autograd adds the gradients of a call on a forward on wy_a's A into an existing .grad (the initial state's
    # included): prior + fresh, then + fresh again, where fresh is the one-stream gradient of the same call; with two
    # streams the split forward's summary runs on a side stream, joined before the gradients reach .grad
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    _check_accumulation(seq_lens, heads, device, fresh_pins={"streams": 1}, pins={"streams": streams},
                        beta_dtype=torch.float32)  # fmt: skip
    stages = [stage for stage, _stream in launches]
    # the fresh call and the two accumulating ones
    _assert_fwd_route(stages, route, calls=3)
    if route != "a":
        summary = "fwd_summary_a" if _split_a() else "fwd_summary"
        side = [stage for stage, stream in launches if stream != main]
        assert side.count(summary) == (2 if streams == 2 else 0), launches


# the calls on a layout of FWD_ROUTES that miss one condition of the forwards on wy_a's A: (beta dtype, in-kernel q / k
# normalization, table route, a value-head group of four, pins)
MAIN_ROUTE_ROWS = {
    "bf16-beta": (torch.bfloat16, True, "host", False, {}),
    "unnormalized": (torch.float32, False, "host", False, {}),
    "device-route": (torch.float32, True, "device", False, {}),
    "group-of-four": (torch.float32, True, "host", True, {}),
    "prep-tma": (torch.float32, True, "host", False, {"prep": "tma"}),
    "prep-plain": (torch.float32, True, "host", False, {"prep": "plain"}),
    "scan-config": (torch.float32, True, "host", False, {"scan_config": (64, 1)}),
    "segment-chunks": (torch.float32, True, "host", False, {"segment_chunks": 0}),
}


@pytest.mark.parametrize("row", sorted(MAIN_ROUTE_ROWS))
@pytest.mark.parametrize("route", FWD_ROUTES)
def test_fwd_routes_leave_the_other_calls_on_the_main_launches(
    route, row, device, monkeypatch
) -> None:
    # on a layout whose FP32-beta calls take a forward on wy_a's A, a call that misses one of its conditions (BF16
    # beta, q / k the caller normalized, the device-built table, a value-head group of four, a prep / scan_config /
    # segment_chunks pin) launches exactly what the package launches with the switches of PREP_FORWARD_OFF off (FI
    # main's forward: the prep, the WY stage, fwd_ho), with the same bits: a route check within this version
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    beta_dtype, l2norm, table_route, group_of_four, pins = MAIN_ROUTE_ROWS[row]
    if group_of_four:
        heads = (heads[0], 4 * heads[0])
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=461, normalized_qk=not l2norm)  # fmt: skip
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, route=table_route, use_qk_l2norm_in_kernel=l2norm, **pins)  # fmt: skip
    routed = list(stages)
    assert "wy_a" not in routed and _preps(routed), routed
    _prep_forward_off(monkeypatch)
    stages.clear()
    want = _run(inputs, seq_lens, route=table_route, use_qk_l2norm_in_kernel=l2norm, **pins)  # fmt: skip
    assert routed == stages, (routed, stages)
    _assert_bitwise(got, want, OUTPUT_NAMES)


@pytest.mark.parametrize("route", FWD_ROUTES)
def test_the_route_switches_give_the_prep_forward(route, device, monkeypatch) -> None:
    # PREP_FORWARD_OFF as the non-deterministic package's tests apply it: after an unpinned call has cached the
    # layout's tables (the switches are in their key), the switches off make an FP32-beta call of the route's layout
    # run the prep forward (FI main's forward), with the bits of the same call under a prep pin
    heads, seq_lens = _fwd_route(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=463)
    stages = _record_launches(monkeypatch)
    _run(inputs, seq_lens)
    _assert_fwd_route(stages, route)
    for name, off in PREP_FORWARD_OFF:
        monkeypatch.setattr(cake_chain, name, off)
    stages.clear()
    got = _run(inputs, seq_lens)
    assert "wy_a" not in stages and _preps(stages), stages
    assert {"fwd_ho", "fwd_ho_vh2"} & set(stages), stages
    want = _run(inputs, seq_lens, prep="plain")
    _assert_bitwise(got, want, OUTPUT_NAMES)


def test_fwd_split_tables_follow_the_plan(device, monkeypatch) -> None:
    # the split tables ride on a host-built table whose forward scan splits, at the host plan's piece length (the
    # planner's SM count); a call off the forward on wy_a's A (a_inputs False), a pinned scan table, a short piece and
    # the device-built table carry none
    arch = _fwd_split_or_skip(device)
    _fwd_split_test_class(monkeypatch)
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    index = cake_chain._device_index(device)
    sm_count = cake_chain._device_props(index)[0]

    def meta(heads, seq_lens, route="host", a_inputs=True, **pins):
        cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
        return cake_chain._chunk_meta(cu.to(device), cu if route == "host" else None, 1, sum(seq_lens), index,
                                      heads[1], cake_chain.make_options(**pins), num_heads=heads[0],
                                      a_inputs=a_inputs)  # fmt: skip

    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    plan = cake_plan.host_plan(tuple(seq_lens), heads[1], arch, sm_count)
    got = meta(heads, seq_lens).fwd_split
    assert got is not None and got.length == plan.pieces.length
    assert got.length >= cake_plan.FORWARD_SPLIT_MIN_PIECE[arch]
    assert got.ready is not None  # cached with its write
    # the test rows cut pieces shorter than every FWD_SPLIT_A_MIN_PIECE entry (_split_a); every sequence of "g3-all"
    # splits (no first_seq index: wave 1's end states are the final states in table order), the other rows keep a
    # sequence of one piece
    for case, (case_heads, case_lens) in FWD_SPLIT_CASES.items():
        split = meta(case_heads, case_lens).fwd_split
        length = split.length
        assert length < min(
            cake_chain.FWD_SPLIT_A_MIN_PIECE.values(), default=length + 1
        )
        assert cake_chain._fwd_split_a(arch, length) is _split_a(), (arch, length)
        every = split.num_first == len(case_lens)
        assert every is (case == "g3-all") and (split.first_seq is None) is every, (case, split.num_first)  # fmt: skip
    assert got.num_segments == sum(plan.pieces.counts)
    assert got.num_first + got.num_middle + len(seq_lens) == got.num_segments
    assert got.num_middle >= 1
    assert meta(heads, seq_lens, a_inputs=False).fwd_split is None
    assert meta(heads, seq_lens, segment_chunks=0).fwd_split is None
    assert meta(heads, seq_lens, scan_config=(64, 1)).fwd_split is None
    assert meta(heads, seq_lens, route="device").fwd_split is None
    assert meta((1, 3), _PAIR_LAYOUT).fwd_split is None  # pieces of about ten chunks
    # one stream: the same pieces, the summary on the caller's stream
    one = meta(heads, seq_lens, streams=1).fwd_split
    assert one.one_stream and not got.one_stream and one.length == got.length


@pytest.mark.parametrize(
    "row",
    [row for row, (*_, split) in sorted(FWD_SPLIT_CLASS_ROWS.items()) if not split],
)
def test_fwd_split_short_pieces_keep_the_prep_forward(row, device, monkeypatch) -> None:
    # the pair rows outside the split's shape class (pieces shorter than FORWARD_SPLIT_MIN_PIECE): FP32 beta and no
    # launch of the split forward
    _fwd_split_or_skip(device)
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    heads, seq_lens, _split = FWD_SPLIT_CLASS_ROWS[row]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=427)
    stages = _record_launches(monkeypatch)
    _run(inputs, seq_lens)
    assert not {"fwd_fused", "fwd_summary", "fwd_summary_a"} & set(stages), stages


def test_fwd_a_serial_keep_on_r200(device, monkeypatch) -> None:
    # on R200 (sm_107a) at 212 SMs the pair row of HV 48 keeps a serial forward table (FWD_A_SERIAL_KEEP) and runs the
    # forward on wy_a's A: the bits repeat run to run and on one stream as on two, and the outputs and gradients stay
    # within the accuracy budgets of the prep forward on the planner's split table (its FP64 recurrence is too long to
    # run here)
    arch = _fwd_a_or_skip(device)
    if arch not in cake_chain.FWD_A_SERIAL_KEEP:
        pytest.skip(f"no serial forward table is kept on {arch}")
    _small_device(monkeypatch, device, 212)
    heads, seq_lens = (16, 48), _PAIR_LAYOUT
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=471)
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens)
    _assert_fwd_route(stages, "a")
    metas = list(cake_chain._META_FRONT.values())
    assert metas and all(m.fwd_segments.num_segments == len(seq_lens) for m in metas), (
        metas
    )
    for streams in (2, 1):
        stages.clear()
        got = _run(inputs, seq_lens, streams=streams)
        _assert_fwd_route(stages, "a")
        _assert_bitwise(got, want, OUTPUT_NAMES)
    monkeypatch.setattr(cake_chain, "FWD_A_SERIAL_KEEP", {})
    stages.clear()
    prep = _run(inputs, seq_lens)
    assert "wy_a" not in stages and _preps(stages), stages
    _assert_close_to_reference(want, prep, OUTPUT_NAMES)
