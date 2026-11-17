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

Host-side contract of the Cake non-deterministic GDN chunked-training package
(``flashinfer.experimental.cake.cake_gdn_nondet_train``) on its four call routes: the host-table route
(``cu_seqlens_cpu`` given), the device route (``cu_seqlens`` only), the equal-length batch (no offsets) and the
packed row (q / k / v as head views of one row that requires grad).

* Every output repeats bit for bit: the output, the final state and every gradient of three identical calls, plain
  and under non-reentrant activation checkpointing, and a second backward into the same leaves adds exactly the
  first one's gradients (gradient accumulation).
* Reused plans on recycled scratch: a call with new inputs on a layout whose plans and tables are cached, after a
  call of another layout reused the workspaces, has the bits of the same call in a fresh process (plain and
  checkpointed; a row the adjoint walk splits and a row it does not).
* No steady-state host synchronization: once a layout's libraries, tables and workspaces exist, a forward and
  backward (plain or checkpointed) runs under the error sync-debug mode.
* A checkpointed forward keeps nothing on the device beyond its output (no hidden retention) on every route.
* The host time of a steady-state call: each forward call and each backward call returns to the caller within
  ``HOST_CALL_MAX_US``.  The package takes tensors and keywords only (no context object), so this is the whole
  per-call host cost a wrapper such as ``fully_shard`` adds around the module.
* PyTorch's deterministic algorithms (strict and ``warn_only``) change nothing on the packed row and under
  checkpointing either: no alert, no warning, the same bits.

The suite names its package through the shared alias helper and imports nothing else from flashinfer.
"""

from __future__ import annotations

import contextlib
import gc
import math
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake_gdn_nondet_train", "cake_backend")
cake_chain = cake_package("cake_gdn_nondet_train", "cake_chain")
cake_gdn_nondet = cake_package("cake_gdn_nondet_train").cake_gdn_nondet

HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)
ROUTES = ("host", "device", "equal", "packed")
# The layout of the packed routes (partial, full and multi-chunk sequences) and of the equal-length batch.
SEQ_LENS = (300, 77, 640)
EQUAL = (2, 512)  # (batch, tokens per sequence)
HEADS = (2, 6)  # the value-head group of three of the source issue (#7074)
# The host-time bound of one steady-state forward call and one backward call (FlashInfer #7914: a module's
# per-call host cost under fully_shard of about 1 ms), on the issue's longest pack.
HOST_CALL_MAX_US = 1000.0
HOST_LAYOUT = (131071, 87531, 43385)
HOST_SAMPLES = 9
# The rows of the reuse test, (lengths, heads): one the one-wave split of the adjoint walk takes on every exported
# architecture's SM count and one it never takes (the (16, 48) group of three stays on the serial walk).
REUSE_LAYOUTS = {
    "split": ((11000, 12000, 9768), (4, 12)),
    "unsplit": ((11000, 12000, 9768), (16, 48)),
}
_ALERT = "does not have a deterministic implementation"


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"compute capability {capability} is not an exported target")
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


class _Problem:
    """The operands of one route, drawn on the host, and the call: q / k / v leaves (or the packed row leaf), g,
    beta and the initial state as leaves, the output gradient and the final-state gradient."""

    def __init__(
        self,
        route: str,
        device,
        *,
        seq_lens=SEQ_LENS,
        heads=HEADS,
        seed: int = 0,
        initial: bool = True,
    ):
        self.route, self.heads = route, heads
        h, hv = heads
        if route == "equal":
            batch, tokens = EQUAL
            shape, self.seq_lens, num_seqs = (batch, tokens), (tokens,) * batch, batch
        else:
            shape, self.seq_lens, num_seqs = (
                (1, sum(seq_lens)),
                tuple(seq_lens),
                len(seq_lens),
            )
        gen = torch.Generator().manual_seed(seed)

        def randn(*s):
            return torch.randn(*s, generator=gen)

        q, k = randn(*shape, h, HEAD_DIM), randn(*shape, h, HEAD_DIM)
        v = randn(*shape, hv, HEAD_DIM)
        self.qkv = [t.to(torch.bfloat16).to(device) for t in (q, k, v)]
        self.g = (torch.rand(*shape, hv, generator=gen) * -0.5).to(device)
        self.beta = torch.rand(*shape, hv, generator=gen).to(device)
        self.do = randn(*shape, hv, HEAD_DIM).to(torch.bfloat16).to(device)
        self.h0 = (
            (0.1 * randn(num_seqs, hv, HEAD_DIM, HEAD_DIM)).to(device)
            if initial
            else None
        )
        self.dht = (
            randn(num_seqs, hv, HEAD_DIM, HEAD_DIM).to(device) if initial else None
        )
        if route == "equal":
            self.cu = self.cu_cpu = None
        else:
            offsets = [0]
            for n in self.seq_lens:
                offsets.append(offsets[-1] + n)
            self.cu = torch.tensor(offsets, dtype=torch.int32, device=device)
            self.cu_cpu = (
                torch.tensor(offsets, dtype=torch.int32) if route != "device" else None
            )
        self.widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
        self.row = (
            torch.cat([t.flatten(-2) for t in self.qkv], -1)
            if route == "packed"
            else None
        )

    def leaves(self) -> dict:
        out = {
            "g": self.g.detach().clone().requires_grad_(True),
            "beta": self.beta.detach().clone().requires_grad_(True),
        }
        if self.h0 is not None:
            out["h0"] = self.h0.detach().clone().requires_grad_(True)
        if self.route == "packed":
            out["row"] = self.row.detach().clone().requires_grad_(True)
        else:
            for name, t in zip(("q", "k", "v"), self.qkv, strict=True):
                out[name] = t.detach().clone().requires_grad_(True)
        return out

    def layer(self, leaves: dict, checkpointed: bool):
        """The forward of one call (non-reentrant checkpointing when ``checkpointed``): ``(o, final_state)``."""

        def body(*args):
            if self.route == "packed":
                row, g, beta, h0 = args
                q, k, v = (
                    part.unflatten(-1, (-1, HEAD_DIM))
                    for part in row.split(self.widths, -1)
                )
            else:
                q, k, v, g, beta, h0 = args
            return cake_gdn_nondet(q, k, v, g, beta, scale=SCALE, initial_state=h0,
                                   output_final_state=h0 is not None, cu_seqlens=self.cu,
                                   cu_seqlens_cpu=self.cu_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip

        names = (
            ("row", "g", "beta")
            if self.route == "packed"
            else ("q", "k", "v", "g", "beta")
        )
        args = [leaves[n] for n in names] + [leaves.get("h0")]
        if checkpointed:
            return checkpoint(
                body, *args, use_reentrant=False, preserve_rng_state=False
            )
        return body(*args)

    def step(self, leaves: dict, checkpointed: bool = False):
        """One forward and backward into ``leaves``; returns the output and the final state."""
        o, final = self.layer(leaves, checkpointed)
        loss = (o.float() * self.do.float()).sum()
        if final is not None:
            loss = loss + (final * self.dht).sum()
        loss.backward()
        return o, final

    def results(self, leaves: dict, o, final) -> dict:
        out = {"o": o.detach(), "dg": leaves["g"].grad, "dbeta": leaves["beta"].grad}
        if final is not None:
            out["final_state"] = final.detach()
        if "h0" in leaves:
            out["dh0"] = leaves["h0"].grad
        if self.route == "packed":
            dq, dk, dv = (
                p.unflatten(-1, (-1, HEAD_DIM))
                for p in leaves["row"].grad.split(self.widths, -1)
            )
        else:
            dq, dk, dv = (leaves[n].grad for n in ("q", "k", "v"))
        out.update(dq=dq, dk=dk, dv=dv)
        return {name: t.detach().clone() for name, t in out.items()}


def _assert_same_bits(got: dict, want: dict) -> None:
    assert set(got) == set(want), (sorted(got), sorted(want))
    for name in want:
        a, b = got[name], want[name]
        assert a.dtype == b.dtype and a.shape == b.shape, name
        a, b = (t.contiguous().view(torch.uint8) for t in (a, b))
        assert torch.equal(a, b), f"{name} differs bit for bit"


def _allocator_bytes(device) -> int:
    """Requested bytes of the caching allocator's live allocations on ``device``."""
    return int(torch.cuda.memory_stats(device)["requested_bytes.all.current"])


@contextlib.contextmanager
def _sync_debug_error():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@contextlib.contextmanager
def _deterministic_algorithms(mode: bool, *, warn_only: bool = False):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


# ---------------------------------------------------------------------------------------------------------------------
# Every output repeats bit for bit


@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpoint"])
@pytest.mark.parametrize("route", ROUTES)
def test_every_output_repeats_bit_for_bit(route, checkpointed, device) -> None:
    # three identical calls: the output, the final state and the gradients of q, k, v (or the packed row), g, beta
    # and the initial state have the same bits every time
    problem = _Problem(route, device, seed=11)
    runs = []
    for _ in range(3):
        leaves = problem.leaves()
        o, final = problem.step(leaves, checkpointed)
        torch.cuda.synchronize()
        runs.append(problem.results(leaves, o, final))
    assert set(runs[0]) == {"o", "final_state", "dq", "dk", "dv", "dg", "dbeta", "dh0"}
    for later in runs[1:]:
        _assert_same_bits(later, runs[0])


@pytest.mark.parametrize("route", ROUTES)
def test_gradients_accumulate_bit_for_bit(route, device) -> None:
    # a second backward into the same leaves adds the same gradients again: every accumulated gradient is exactly
    # twice the first one (a lost, duplicated or reordered update would show)
    problem = _Problem(route, device, seed=13)
    leaves = problem.leaves()
    o, final = problem.step(leaves)
    torch.cuda.synchronize()
    once = problem.results(leaves, o, final)
    o, final = problem.step(leaves)
    torch.cuda.synchronize()
    twice = problem.results(leaves, o, final)
    _assert_same_bits(
        {n: twice[n] for n in ("o", "final_state")},
        {n: once[n] for n in ("o", "final_state")},
    )
    grads = ("dq", "dk", "dv", "dg", "dbeta", "dh0")
    _assert_same_bits({n: twice[n] for n in grads}, {n: once[n] * 2 for n in grads})


# ---------------------------------------------------------------------------------------------------------------------
# Reused plans on recycled scratch


def _reuse_call(layout: str, seed: int, checkpointed: bool, device) -> dict:
    """One host-route forward and backward (under non-reentrant checkpointing when ``checkpointed``) of the
    :data:`REUSE_LAYOUTS` row ``layout`` on the operands of ``seed``: its outputs and gradients, on the host."""
    seq_lens, heads = REUSE_LAYOUTS[layout]
    problem = _Problem("host", device, seq_lens=seq_lens, heads=heads, seed=seed)
    leaves = problem.leaves()
    o, final = problem.step(leaves, checkpointed)
    torch.cuda.synchronize()
    return {name: t.cpu() for name, t in problem.results(leaves, o, final).items()}


@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpoint"])
@pytest.mark.parametrize("layout", sorted(REUSE_LAYOUTS))
def test_reused_plans_on_recycled_scratch_give_the_fresh_process_bits(
    layout, checkpointed, device, tmp_path
) -> None:
    # a layout's call with new inputs, after a call of the same layout (its plans and tables cached) and a call of
    # another layout in between (the workspaces taken and released at other shapes), has the bits of the same call
    # as the only call of a fresh process: outputs, final state and every gradient
    seq_lens, (h, hv) = REUSE_LAYOUTS[layout]
    index = cake_chain._device_index(device)
    split = cake_chain._adjoint_split_length(
        seq_lens, hv, index, cake_chain.make_options(), hv // h
    )
    assert (split is not None) is (layout == "split"), (layout, split)
    other = next(name for name in REUSE_LAYOUTS if name != layout)
    _reuse_call(layout, 41, checkpointed, device)
    _reuse_call(other, 43, checkpointed, device)
    got = _reuse_call(layout, 47, checkpointed, device)
    path = tmp_path / "fresh.pt"
    code = (
        "import importlib.util, sys, torch\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
        f"spec = importlib.util.spec_from_file_location('reuse_host', {str(Path(__file__).resolve())!r})\n"
        "suite = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(suite)\n"
        "device = torch.device('cuda', torch.cuda.current_device())\n"
        f"torch.save(suite._reuse_call({layout!r}, 47, {checkpointed!r}, device), {str(path)!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stderr[-4000:]
    _assert_same_bits(got, torch.load(path))


# ---------------------------------------------------------------------------------------------------------------------
# No steady-state host synchronization, no hidden retention


@pytest.mark.parametrize("initial", [False, True], ids=["zero-state", "state"])
@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpoint"])
@pytest.mark.parametrize("route", ROUTES)
def test_steady_state_calls_do_not_synchronize(
    route, checkpointed, initial, device
) -> None:
    # once the layout's libraries, tables and workspaces exist, a forward and backward runs under the error
    # sync-debug mode (a device synchronization, a blocking copy or a read of device memory would raise); without an
    # initial or final state (the backward's recorded plan replays) and with both (the general path)
    problem = _Problem(route, device, seed=17, initial=initial)
    for _ in range(2):
        problem.step(problem.leaves(), checkpointed)
    torch.cuda.synchronize()
    leaves = problem.leaves()
    with _sync_debug_error():
        problem.step(leaves, checkpointed)
    torch.cuda.synchronize()


@pytest.mark.parametrize("initial", [False, True], ids=["zero-state", "state"])
@pytest.mark.parametrize("route", ROUTES)
def test_checkpointed_forward_retains_no_bytes(route, initial, device) -> None:
    # non-reentrant checkpointing drops the saved tensors of the forward: on every route the checkpointed forward
    # keeps nothing on the device beyond its output (and final state) (FlashInfer #7039), and its backward replays it
    problem = _Problem(route, device, seed=19, initial=initial)
    for _ in range(2):  # the libraries, the tables and the workspaces of this layout
        problem.step(problem.leaves(), checkpointed=True)
    leaves = problem.leaves()
    gc.collect()
    torch.cuda.synchronize()
    before = _allocator_bytes(device)
    o, final = problem.layer(leaves, checkpointed=True)
    torch.cuda.synchronize()
    outputs = o.untyped_storage().nbytes() + (
        0 if final is None else final.untyped_storage().nbytes()
    )
    retained = _allocator_bytes(device) - before - outputs
    assert retained == 0, (
        f"a checkpointed forward retains {retained} requested bytes beyond its outputs"
    )
    loss = (o.float() * problem.do.float()).sum()
    if final is not None:
        loss = loss + (final * problem.dht).sum()
    loss.backward()
    torch.cuda.synchronize()
    assert all(
        bool(torch.isfinite(t.float()).all())
        for t in problem.results(leaves, o, final).values()
    )


# ---------------------------------------------------------------------------------------------------------------------
# Host time of a steady-state call


@pytest.mark.parametrize("route", ["host", "device", "packed"])
def test_steady_state_host_time_per_call(route, device) -> None:
    # the forward call and the backward call each return to the caller within HOST_CALL_MAX_US once the layout is
    # planned (median of HOST_SAMPLES calls; the device is idle when each timer starts)
    problem = _Problem(route, device, seq_lens=HOST_LAYOUT, seed=23, initial=False)
    for _ in range(3):
        problem.step(problem.leaves())
    torch.cuda.synchronize()
    fwd_us, bwd_us = [], []
    for _ in range(HOST_SAMPLES):
        leaves = problem.leaves()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o, _ = problem.layer(leaves, checkpointed=False)
        t1 = time.perf_counter()
        loss = (o.float() * problem.do.float()).sum()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        t3 = time.perf_counter()
        torch.cuda.synchronize()
        fwd_us.append((t1 - t0) * 1e6)
        bwd_us.append((t3 - t2) * 1e6)
        del o, loss, leaves
    fwd, bwd = statistics.median(fwd_us), statistics.median(bwd_us)
    assert fwd <= HOST_CALL_MAX_US and bwd <= HOST_CALL_MAX_US, (
        route,
        fwd,
        bwd,
        fwd_us,
        bwd_us,
    )


# ---------------------------------------------------------------------------------------------------------------------
# PyTorch's deterministic algorithms


@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpoint"])
def test_deterministic_algorithms_change_nothing_on_the_packed_row(
    checkpointed, device
) -> None:
    # every launch is fixed-order: under torch.use_deterministic_algorithms(True), strict and warn_only, the packed
    # row (and its checkpoint replay) runs without an alert or a warning, to the bits of the default mode
    problem = _Problem("packed", device, seed=29)
    leaves = problem.leaves()
    o, final = problem.step(leaves, checkpointed)
    torch.cuda.synchronize()
    want = problem.results(leaves, o, final)
    for warn_only in (False, True):
        with (
            _deterministic_algorithms(True, warn_only=warn_only),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            leaves = problem.leaves()
            o, final = problem.step(leaves, checkpointed)
            torch.cuda.synchronize()
        assert not [w for w in caught if _ALERT in str(w.message)], warn_only
        assert type(o.grad_fn).__name__ == "CakeGDNNondetPackedFunctionBackward", type(
            o.grad_fn
        ).__name__
        _assert_same_bits(problem.results(leaves, o, final), want)
