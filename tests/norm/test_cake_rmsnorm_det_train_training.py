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

Training-loop tests of the Cake fused BF16 RMSNorm training package (the
deterministic ``dw`` reduction): what a training step does around the kernels.

* Non-reentrant activation checkpointing replays the forward in the backward:
  outputs and every gradient bitwise equal to plain autograd and to the
  functional calls.
* Gradient accumulation into an existing ``.grad`` over microbatches of
  different token counts: bitwise the sum of the per-microbatch gradients in
  the parameter's dtype.
* CUDA graph capture and replay of the functional forward + backward (with a
  caller-owned workspace) and of an autograd step: every replay bitwise equal to
  the eager calls, after new data is copied into the static inputs too, and the
  workspace counters back at zero.
* Retained memory: a checkpointed forward keeps exactly its outputs alive, plain
  autograd keeps the outputs plus the FP32 ``rstd``; after the backward only the
  gradients remain.
* A resident NCCL all-reduce on another stream (two GPUs, the file's one
  ``node_exclusive`` test): the forward and backward finish while the rank's
  all-reduce kernel is resident and waiting for its delayed peer, with bitwise
  the results of the run without NCCL.
* The package imports under another name with ``flashinfer`` blocked.

The cases are the per-head ``[T, 24, 256]`` view of a ``[T, 24, 512]`` buffer
(hidden 256), ``[T, 5120]`` rows (hidden 5120) and the fused residual add on
``[T, 6144]`` rows (the width with a residual route), each with a BF16 and an
FP32 weight.  A case whose route the device does not export is skipped with
the reason.

Like ``test_cake_rmsnorm_det_train.py`` this suite names its package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so the vendored-package gate runs it against a vendored copy.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_det_train")
loader = cake_package("cake_rmsnorm_det_train", "cake_jit")
cake_rmsnorm_det = pkg.cake_rmsnorm_det
cake_rmsnorm_det_train_backward = pkg.cake_rmsnorm_det_train_backward
cake_rmsnorm_det_train_backward_workspace = (
    pkg.cake_rmsnorm_det_train_backward_workspace
)
cake_rmsnorm_det_train_forward = pkg.cake_rmsnorm_det_train_forward
is_cake_rmsnorm_det_train_supported = pkg.is_cake_rmsnorm_det_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
SEED = 5284
HEADS = 24
EPS = {256: 1e-6, 5120: 1e-5, 6144: 1e-5}
# Token counts per hidden size: a small one and a training-size one (24 x 683 = 16392 rows of 256).
TOKENS = {256: (7, 683), 5120: (65, 16231), 6144: (65, 16231)}
# The caching allocator serves a request below 1 MiB from its small pool with a block of exactly the request rounded
# up to 512 bytes, so the retained-memory test reads exact byte counts at the small token count of each case.
BLOCK_BYTES = 512
SMALL_POOL_BYTES = 1 << 20
NCCL_WORLD = 2
NCCL_TIMEOUT_S = 900
NCCL_MAIN_DEADLINE_S = 120
# Cycles the delayed rank's side stream sleeps before its all-reduce (~1 s at 2 GHz): the waiting rank's
# NCCL kernel stays resident for that long while its forward and backward run.
NCCL_PEER_SLEEP_CYCLES = 2_000_000_000
NCCL_ITERS = 3
NCCL_ELEMENTS = 16 << 20


@dataclass(frozen=True)
class Case:
    hidden: int
    heads: Optional[
        int
    ]  # None: [T, H] rows; else the [T, heads, H] view of a [T, heads, 2 H] buffer
    residual: bool
    w_dtype: torch.dtype

    @property
    def label(self) -> str:
        shape = f"h{self.hidden}" + (f"x{self.heads}heads" if self.heads else "")
        res = "-res" if self.residual else ""
        w = "-w32" if self.w_dtype == torch.float32 else "-w16"
        return f"{shape}{res}{w}"

    def rows(self, tokens: int) -> int:
        return tokens * (self.heads or 1)


# The per-head view (hidden 256) and [T, 5120] rows with a BF16 and an FP32 weight, and the fused residual add at
# hidden 6144 (the width whose residual route the package exports) with both weights.
CASES = tuple(
    Case(hidden, heads, residual, w_dtype)
    for hidden, heads, residual in (
        (256, HEADS, False),
        (5120, None, False),
        (6144, None, True),
    )
    for w_dtype in (torch.bfloat16, torch.float32)
)
CASE_IDS = [case.label for case in CASES]


def _device(index: int = 0) -> torch.device:
    return torch.device(f"cuda:{index}")


def _capability(index: int = 0) -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device(index))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_det_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build or load the local architecture's libraries before the first test (the package's own specs).
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _skip_reason(case: Case, index: int = 0) -> Optional[str]:
    """Why ``case`` has no exported route on device ``index`` (None: it has one)."""

    device = _device(index)
    if not is_cake_rmsnorm_det_train_supported(device, case.hidden):
        return f"hidden={case.hidden} has no exported route on this device"
    token = "f32" if case.w_dtype == torch.float32 else "bf16"
    for kind in ("fwd", "bwd"):
        if not loader.route_applies(
            device_capability=_capability(index),
            hidden=case.hidden,
            kind=kind,
            residual=case.residual,
            w_dtype=token,
        ):
            return (
                f"hidden={case.hidden} residual={case.residual} w={token}: "
                f"no exported {kind} route on this device"
            )
    return None


def _skip_unless_case(case: Case) -> None:
    reason = _skip_reason(case)
    if reason is not None:
        pytest.skip(reason)


def _generator(device: torch.device, *keys: int) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=device).manual_seed(seed)


@dataclass
class Inputs:
    case: Case
    tokens: int
    x_base: torch.Tensor  # the buffer x is a view of (x itself for [T, H] rows)
    u_base: Optional[torch.Tensor]
    g: torch.Tensor
    g_h: Optional[torch.Tensor]
    w: torch.Tensor

    def view(self, base: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if base is None or self.case.heads is None:
            return base
        return base[..., : self.case.hidden]

    @property
    def x(self) -> torch.Tensor:
        return self.view(self.x_base)

    @property
    def u(self) -> Optional[torch.Tensor]:
        return self.view(self.u_base)

    @property
    def eps(self) -> float:
        return EPS[self.case.hidden]


def _activation(case: Case, tokens: int, generator, device) -> torch.Tensor:
    if case.heads is None:
        shape = (tokens, case.hidden)
    else:
        shape = (tokens, case.heads, 2 * case.hidden)
    return torch.randn(
        shape, dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)


def _gradient(case: Case, tokens: int, generator, device) -> torch.Tensor:
    lead = (tokens,) if case.heads is None else (tokens, case.heads)
    return torch.randn(
        (*lead, case.hidden), dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)


def _make(case: Case, tokens: int, *keys: int, device_index: int = 0) -> Inputs:
    device = _device(device_index)
    generator = _generator(
        device, case.hidden, case.heads or 0, case.residual, tokens, *keys
    )
    x_base = _activation(case, tokens, generator, device)
    u_base = _activation(case, tokens, generator, device) if case.residual else None
    g = _gradient(case, tokens, generator, device)
    g_h = _gradient(case, tokens, generator, device) if case.residual else None
    noise = torch.randn(
        case.hidden, dtype=torch.float32, device=device, generator=generator
    )
    if case.w_dtype == torch.float32:
        w = 1.0 + 1e-3 * noise  # an effective weight BF16 does not represent
    else:
        w = (1.0 + 0.1 * noise).to(torch.bfloat16)
    inputs = Inputs(case, tokens, x_base, u_base, g, g_h, w)
    if case.heads is not None:
        assert tuple(inputs.x.stride()) == (
            case.heads * 2 * case.hidden,
            2 * case.hidden,
            1,
        )
    return inputs


@dataclass
class Result:
    y: torch.Tensor
    h_new: Optional[torch.Tensor]
    rstd: Optional[torch.Tensor]
    dx: torch.Tensor  # gradient of x (and of the residual: the same values)
    du: Optional[torch.Tensor]
    dw: torch.Tensor  # in w.dtype


def _functional(inputs: Inputs, *, workspace=None) -> Result:
    """The functional forward + backward (``dw`` cast to ``w.dtype`` as autograd returns it)."""

    outputs = cake_rmsnorm_det_train_forward(
        inputs.x, inputs.w, inputs.eps, residual=inputs.u
    )
    y, rstd = outputs[0], outputs[1]
    h_new = outputs[2] if inputs.case.residual else None
    normalized = h_new if inputs.case.residual else inputs.x
    kwargs = {"workspace": workspace}
    if inputs.case.residual:
        kwargs["g_residual"] = inputs.g_h
    dx, dw = cake_rmsnorm_det_train_backward(
        inputs.g, normalized, inputs.w, rstd, **kwargs
    )
    return Result(
        y,
        h_new,
        rstd,
        dx,
        dx if inputs.case.residual else None,
        dw.to(inputs.w.dtype),
    )


@dataclass
class Leaves:
    x_base: torch.Tensor
    u_base: Optional[torch.Tensor]
    w: torch.Tensor


def _leaves(inputs: Inputs) -> Leaves:
    def leaf(t):
        return None if t is None else t.detach().clone().requires_grad_(True)

    return Leaves(leaf(inputs.x_base), leaf(inputs.u_base), leaf(inputs.w))


def _forward(inputs: Inputs, leaves: Leaves, *, checkpoint: bool, calls=None):
    """``cake_rmsnorm_det`` on the leaves' views, plainly or under non-reentrant checkpointing."""

    def fn(x, w, u):
        if calls is not None:
            calls.append(1)
        return cake_rmsnorm_det(x, w, inputs.eps, u)

    args = (inputs.view(leaves.x_base), leaves.w, inputs.view(leaves.u_base))
    if checkpoint:
        out = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    else:
        out = fn(*args)
    return out if isinstance(out, tuple) else (out,)


def _backward(inputs: Inputs, outputs: tuple[torch.Tensor, ...]) -> None:
    grads = (inputs.g, inputs.g_h) if inputs.case.residual else (inputs.g,)
    torch.autograd.backward(outputs, grads)


def _view_grad(inputs: Inputs, base_grad: Optional[torch.Tensor]):
    """The gradient of the view (and zero outside it: the pad columns of the per-head buffer)."""

    if base_grad is None or inputs.case.heads is None:
        return base_grad
    pad = base_grad[..., inputs.case.hidden :]
    assert torch.count_nonzero(pad) == 0, "gradient leaked into the buffer's pad"
    return base_grad[..., : inputs.case.hidden]


def _assert_bitwise(label: str, **pairs) -> None:
    for name, (actual, expected) in pairs.items():
        if expected is None:
            assert actual is None, f"{label}: {name} expected None"
            continue
        assert actual is not None, f"{label}: {name} missing"
        assert actual.dtype == expected.dtype, (
            f"{label}: {name} {actual.dtype} != {expected.dtype}"
        )
        assert tuple(actual.shape) == tuple(expected.shape), (
            f"{label}: {name} {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
        assert torch.equal(actual, expected), f"{label}: {name} is not bitwise equal"


# --------------------------------------------------------------------------- activation checkpointing


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("which", (0, 1), ids=("small", "large"))
def test_checkpoint_matches_plain_autograd_bitwise(case: Case, which: int) -> None:
    """``torch.utils.checkpoint`` (``use_reentrant=False``) runs the forward twice -- once in the step, once
    replayed in the backward -- and gives bitwise the outputs and gradients of plain autograd, which equal the
    functional calls."""

    _skip_unless_case(case)
    tokens = TOKENS[case.hidden][which]
    inputs = _make(case, tokens, 11)
    want = _functional(inputs)
    label = f"{case.label} tokens={tokens}"

    plain = _leaves(inputs)
    outputs = _forward(inputs, plain, checkpoint=False)
    _backward(inputs, outputs)
    _assert_bitwise(
        f"plain {label}",
        y=(outputs[0], want.y),
        h_new=(outputs[1] if case.residual else None, want.h_new),
        dx=(_view_grad(inputs, plain.x_base.grad), want.dx),
        du=(_view_grad(inputs, plain.u_base.grad if case.residual else None), want.du),
        dw=(plain.w.grad, want.dw),
    )

    calls: list[int] = []
    ckpt = _leaves(inputs)
    outputs = _forward(inputs, ckpt, checkpoint=True, calls=calls)
    assert len(calls) == 1
    _backward(inputs, outputs)
    assert len(calls) == 2, "the checkpointed forward was not replayed in the backward"
    _assert_bitwise(
        f"checkpoint {label}",
        y=(outputs[0], want.y),
        h_new=(outputs[1] if case.residual else None, want.h_new),
        dx=(_view_grad(inputs, ckpt.x_base.grad), want.dx),
        du=(_view_grad(inputs, ckpt.u_base.grad if case.residual else None), want.du),
        dw=(ckpt.w.grad, want.dw),
    )


# --------------------------------------------------------------------------- gradient accumulation


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("checkpoint", (False, True), ids=("plain", "checkpoint"))
def test_gradient_accumulation_into_an_existing_grad(
    case: Case, checkpoint: bool
) -> None:
    """Two microbatches of different token counts accumulate into a ``.grad`` that already holds a value: the
    weight's ``.grad`` is bitwise ``(g0 + dw_1) + dw_2`` in ``w.dtype``; a leaf used twice gets ``dx + dx``."""

    _skip_unless_case(case)
    small, large = TOKENS[case.hidden]
    first = _make(case, large, 21)
    second = _make(case, small, 22)
    second.w = first.w  # one parameter across the microbatches
    want1 = _functional(first)
    want2 = _functional(second)
    g0 = 0.01 * torch.randn(
        case.hidden,
        dtype=torch.float32,
        device=_device(),
        generator=_generator(_device(), case.hidden, 23),
    ).to(case.w_dtype)

    w_leaf = first.w.detach().clone().requires_grad_(True)
    w_leaf.grad = g0.clone()
    leaves1 = _leaves(first)
    leaves1.w = w_leaf
    for _ in range(2):  # the same microbatch twice: x's .grad accumulates too
        _backward(first, _forward(first, leaves1, checkpoint=checkpoint))
    leaves2 = _leaves(second)
    leaves2.w = w_leaf
    _backward(second, _forward(second, leaves2, checkpoint=checkpoint))

    want_w = ((g0 + want1.dw) + want1.dw) + want2.dw
    label = f"{case.label} checkpoint={checkpoint}"
    _assert_bitwise(
        label,
        dw=(w_leaf.grad, want_w),
        dx=(_view_grad(first, leaves1.x_base.grad), want1.dx + want1.dx),
        du=(
            _view_grad(first, leaves1.u_base.grad if case.residual else None),
            (want1.du + want1.du) if case.residual else None,
        ),
        dx2=(_view_grad(second, leaves2.x_base.grad), want2.dx),
    )


# --------------------------------------------------------------------------- CUDA graphs


def _static_outputs(inputs: Inputs):
    case, tokens = inputs.case, inputs.tokens
    lead = (tokens,) if case.heads is None else (tokens, case.heads)
    device = _device()
    y = torch.empty((*lead, case.hidden), dtype=torch.bfloat16, device=device)
    rstd = torch.empty(lead, dtype=torch.float32, device=device)
    h_new = torch.empty_like(y) if case.residual else None
    dx = torch.empty_like(y)
    dw = torch.empty(case.hidden, dtype=torch.float32, device=device)
    return y, rstd, h_new, dx, dw


def _graph_body(inputs: Inputs, static, workspace) -> None:
    y, rstd, h_new, dx, dw = static
    kwargs = {"out": y, "rstd": rstd}
    if inputs.case.residual:
        kwargs.update(residual=inputs.u, residual_out=h_new)
    outputs = cake_rmsnorm_det_train_forward(inputs.x, inputs.w, inputs.eps, **kwargs)
    assert outputs[0] is y and outputs[1] is rstd
    normalized = h_new if inputs.case.residual else inputs.x
    kwargs = {"workspace": workspace, "dx": dx, "dw": dw}
    if inputs.case.residual:
        kwargs["g_residual"] = inputs.g_h
    got = cake_rmsnorm_det_train_backward(
        inputs.g, normalized, inputs.w, rstd, **kwargs
    )
    assert got[0] is dx and got[1] is dw


def _assert_static(label: str, inputs: Inputs, static, workspace) -> None:
    y, rstd, h_new, dx, dw = static
    want = _functional(inputs)
    _assert_bitwise(
        label,
        y=(y, want.y),
        rstd=(rstd, want.rstd),
        h_new=(h_new, want.h_new),
        dx=(dx, want.dx),
        dw=(dw if inputs.w.dtype == torch.float32 else dw.to(inputs.w.dtype), want.dw),
    )
    assert torch.count_nonzero(workspace[: loader.COUNTERS_BYTES]) == 0, (
        f"{label}: the workspace counters are not reset"
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("which", (0, 1), ids=("small", "large"))
def test_cuda_graph_replays_the_functional_step_bitwise(case: Case, which: int) -> None:
    """Forward + backward captured in one CUDA graph with preallocated outputs and a caller-owned workspace:
    each replay gives bitwise the eager results, also after new values are copied into the static inputs."""

    _skip_unless_case(case)
    tokens = TOKENS[case.hidden][which]
    inputs = _make(case, tokens, 31)
    rows = case.rows(tokens)
    workspace = cake_rmsnorm_det_train_backward_workspace(
        rows, case.hidden, _device(), residual=case.residual
    )
    static = _static_outputs(inputs)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _graph_body(inputs, static, workspace)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _graph_body(inputs, static, workspace)
    label = f"graph {case.label} tokens={tokens}"
    for replay in range(3):
        for t in static:
            if t is not None:
                t.fill_(float("nan") if t.is_floating_point() else 0)
        graph.replay()
        torch.cuda.synchronize()
        _assert_static(f"{label} replay={replay}", inputs, static, workspace)
    fresh = _make(case, tokens, 32)
    for name in ("x_base", "u_base", "g", "g_h", "w"):
        target = getattr(inputs, name)
        if target is not None:
            target.copy_(getattr(fresh, name))
    graph.replay()
    torch.cuda.synchronize()
    _assert_static(f"{label} new data", inputs, static, workspace)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_cuda_graph_replays_the_autograd_step_bitwise(case: Case) -> None:
    """A whole autograd step (``cake_rmsnorm_det`` forward and ``backward``) captured in a CUDA graph with the
    gradients set to None before the capture: each replay leaves bitwise the eager gradients in ``.grad``."""

    _skip_unless_case(case)
    tokens = TOKENS[case.hidden][1]
    inputs = _make(case, tokens, 41)
    leaves = _leaves(inputs)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _backward(inputs, _forward(inputs, leaves, checkpoint=False))
            leaves.x_base.grad = leaves.w.grad = None
            if case.residual:
                leaves.u_base.grad = None
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = _forward(inputs, leaves, checkpoint=False)
        _backward(inputs, static_out)
    label = f"autograd graph {case.label} tokens={tokens}"
    for step in range(2):
        if step:
            fresh = _make(case, tokens, 42)
            for name in ("g", "g_h"):
                if getattr(inputs, name) is not None:
                    getattr(inputs, name).copy_(getattr(fresh, name))
            with torch.no_grad():
                leaves.x_base.copy_(fresh.x_base)
                leaves.w.copy_(fresh.w)
                if case.residual:
                    leaves.u_base.copy_(fresh.u_base)
        graph.replay()
        torch.cuda.synchronize()
        current = Inputs(
            case,
            tokens,
            leaves.x_base.detach(),
            None if leaves.u_base is None else leaves.u_base.detach(),
            inputs.g,
            inputs.g_h,
            leaves.w.detach(),
        )
        want = _functional(current)
        _assert_bitwise(
            f"{label} step={step}",
            y=(static_out[0], want.y),
            h_new=(static_out[1] if case.residual else None, want.h_new),
            dx=(_view_grad(current, leaves.x_base.grad), want.dx),
            du=(
                _view_grad(current, leaves.u_base.grad if case.residual else None),
                want.du,
            ),
            dw=(leaves.w.grad, want.dw),
        )


# --------------------------------------------------------------------------- retained memory


def _blocks(*tensors: Optional[torch.Tensor]) -> int:
    """Bytes the caching allocator holds for the distinct storages of ``tensors`` (512-byte blocks)."""

    seen: dict[int, int] = {}
    for t in tensors:
        if t is None:
            continue
        storage = t.untyped_storage()
        nbytes = storage.nbytes()
        assert nbytes < SMALL_POOL_BYTES, "the exact reading needs small-pool blocks"
        seen[storage.data_ptr()] = -(-nbytes // BLOCK_BYTES) * BLOCK_BYTES
    return sum(seen.values())


def _allocated() -> int:
    torch.cuda.synchronize()
    gc.collect()
    return torch.cuda.memory_allocated(_device())


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_retained_memory_of_a_checkpointed_forward(case: Case) -> None:
    """Between forward and backward a checkpointed call keeps exactly its outputs (``y``, ``h_new``) alive and
    plain autograd keeps the outputs plus the FP32 ``rstd`` it saves (x, the residual and w are the caller's);
    after the backward and dropping the outputs only the gradients remain allocated."""

    _skip_unless_case(case)
    tokens = TOKENS[case.hidden][0]
    inputs = _make(case, tokens, 51)
    rows = case.rows(tokens)
    # Warm-up step: libraries loaded, lazy host state built.
    warm = _leaves(inputs)
    _backward(inputs, _forward(inputs, warm, checkpoint=True))
    del warm
    readings = {}
    for checkpoint in (True, False):
        leaves = _leaves(inputs)
        base = _allocated()
        outputs = _forward(inputs, leaves, checkpoint=checkpoint)
        retained = _allocated() - base
        expected = _blocks(*outputs)
        if not checkpoint:
            expected += -(-rows * 4 // BLOCK_BYTES) * BLOCK_BYTES  # the saved FP32 rstd
        readings[checkpoint] = retained
        assert retained == expected, (
            f"{case.label} checkpoint={checkpoint}: {retained} bytes retained after the forward, "
            f"expected {expected} (outputs{'' if checkpoint else ' + rstd'})"
        )
        _backward(inputs, outputs)
        del outputs
        left = _allocated() - base
        grads = _blocks(
            leaves.x_base.grad,
            leaves.w.grad,
            leaves.u_base.grad if case.residual else None,
        )
        assert left == grads, (
            f"{case.label} checkpoint={checkpoint}: {left} bytes allocated after the backward, "
            f"expected the gradients' {grads}"
        )
        del leaves
    print(
        f"RETAINED {case.label} tokens={tokens} rows={rows} checkpoint={readings[True]} "
        f"plain={readings[False]} rstd_saved={readings[False] - readings[True]}"
    )


# --------------------------------------------------------------------------- resident NCCL on another stream


def _wait_event(event: torch.cuda.Event, deadline_s: float) -> bool:
    end = time.monotonic() + deadline_s
    while not event.query():
        if time.monotonic() > end:
            return False
        time.sleep(0.001)
    return True


def _nccl_worker(rank: int, world: int, port: int, out_path: str) -> None:
    """One rank of the resident-NCCL test (run in its own process on ``cuda:<rank>``)."""

    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = _device(rank)
    report: dict = {"rank": rank, "cases": []}
    cases = []
    for case in CASES:
        reason = _skip_reason(case, rank)
        if reason is not None:
            report["cases"].append({"case": case.label, "skipped": reason})
            continue
        tokens = TOKENS[case.hidden][1]
        inputs = _make(case, tokens, 61, rank, device_index=rank)
        rows = case.rows(tokens)
        workspace = cake_rmsnorm_det_train_backward_workspace(
            rows, case.hidden, device, residual=case.residual
        )
        want = _functional(inputs, workspace=workspace)
        again = _functional(inputs, workspace=workspace)
        _assert_bitwise(f"rank {rank} {case.label} repeat", dw=(again.dw, want.dw))
        cases.append((case, inputs, workspace, want))
    torch.cuda.synchronize(device)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    warm = torch.ones(1024, device=device)
    dist.all_reduce(warm)
    torch.cuda.synchronize(device)
    side = torch.cuda.Stream(device)
    for sleeper in range(world):
        buf = torch.full((NCCL_ELEMENTS,), float(rank + 1), device=device)
        torch.cuda.synchronize(device)
        side_done = torch.cuda.Event()
        main_done = torch.cuda.Event()
        with torch.cuda.stream(side):
            if rank == sleeper:
                torch.cuda._sleep(NCCL_PEER_SLEEP_CYCLES)
            work = dist.all_reduce(buf, async_op=True)
            work.wait()
            side_done.record(side)
        results = []
        for case, inputs, workspace, want in cases:
            for _ in range(NCCL_ITERS):
                results.append((case, want, _functional(inputs, workspace=workspace)))
        main_done.record()
        finished = _wait_event(main_done, NCCL_MAIN_DEADLINE_S)
        overlapped = finished and not side_done.query()
        if not finished:
            report["hang"] = (
                f"forward/backward did not finish in {NCCL_MAIN_DEADLINE_S} s"
            )
            Path(out_path).write_text(json.dumps(report))
            os._exit(3)
        torch.cuda.synchronize(device)
        mismatches = []
        for case, want, got in results:
            for name in ("y", "h_new", "rstd", "dx", "dw"):
                a, b = getattr(got, name), getattr(want, name)
                if (a is None) != (b is None) or (
                    a is not None and not torch.equal(a, b)
                ):
                    mismatches.append(f"{case.label}:{name}")
        reduced = bool(torch.all(buf == float(world * (world + 1) // 2)).item())
        report.setdefault("rounds", []).append(
            {
                "sleeper": sleeper,
                "role": "sleeper" if rank == sleeper else "waiter",
                "overlapped": overlapped,
                "calls": len(results),
                "mismatches": sorted(set(mismatches)),
                "all_reduce_ok": reduced,
            }
        )
    report["cases"] += [
        {"case": case.label, "rows": case.rows(i.tokens)} for case, i, _w, _r in cases
    ]
    dist.barrier()
    dist.destroy_process_group()
    Path(out_path).write_text(json.dumps(report))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_CHILD = textwrap.dedent(
    """
    import importlib.util, sys
    path, rank, world, port, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    spec = importlib.util.spec_from_file_location("_cake_rmsnorm_det_train_training_child", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._nccl_worker(rank, world, port, out)
    """
)


@pytest.mark.node_exclusive  # two GPUs of the allocation visible at once
def test_forward_backward_beside_a_resident_nccl_all_reduce(tmp_path: Path) -> None:
    """Two ranks, one process and GPU each.  Each round one rank sleeps on its side stream before its
    all-reduce, so the other rank's NCCL kernel is resident and waiting while that rank runs every case's
    forward and backward on its main stream: they finish before the collective (the waiting rank overlapped),
    with bitwise the results of the run without NCCL, and the all-reduce is correct.  Both ranks wait once."""

    import torch.distributed as dist

    if torch.cuda.device_count() < NCCL_WORLD:
        pytest.skip(
            f"the resident-NCCL test needs {NCCL_WORLD} GPUs, {torch.cuda.device_count()} visible"
        )
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("torch.distributed with NCCL is not available")
    for index in range(NCCL_WORLD):
        if _capability(index) != _capability():
            pytest.skip(f"cuda:{index} is a different architecture than cuda:0")
    if all(_skip_reason(case) is not None for case in CASES):
        pytest.skip("no case has an exported route on this device")
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]
    )
    procs, outs = [], []
    with contextlib.ExitStack() as stack:
        for rank in range(NCCL_WORLD):
            out = tmp_path / f"rank{rank}.json"
            log = stack.enter_context(open(tmp_path / f"rank{rank}.log", "w"))
            outs.append(out)
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _CHILD,
                        __file__,
                        str(rank),
                        str(NCCL_WORLD),
                        str(port),
                        str(out),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + NCCL_TIMEOUT_S
        codes = []
        for proc in procs:
            try:
                codes.append(proc.wait(timeout=max(1.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                for p in procs:
                    p.kill()
                codes.append("timeout")
    tails = {
        rank: (tmp_path / f"rank{rank}.log").read_text()[-3000:]
        for rank in range(NCCL_WORLD)
    }
    assert codes == [0] * NCCL_WORLD, f"rank exit codes {codes}; log tails: {tails}"
    reports = [json.loads(out.read_text()) for out in outs]
    for report in reports:
        rank = report["rank"]
        ran = [c for c in report["cases"] if "rows" in c]
        assert ran, f"rank {rank}: no case ran: {report['cases']}"
        assert len(report["rounds"]) == NCCL_WORLD
        for rnd in report["rounds"]:
            print(
                f"NCCL_RESIDENT rank={rank} sleeper={rnd['sleeper']} role={rnd['role']} "
                f"overlapped={rnd['overlapped']} calls={rnd['calls']} all_reduce_ok={rnd['all_reduce_ok']} "
                f"mismatches={len(rnd['mismatches'])} cases={len(ran)}"
            )
            assert rnd["calls"] == len(ran) * NCCL_ITERS
            assert rnd["mismatches"] == [], f"rank {rank} round {rnd}: results differ"
            assert rnd["all_reduce_ok"], f"rank {rank} round {rnd}: wrong all-reduce"
            if rnd["role"] == "waiter":
                assert rnd["overlapped"], (
                    f"rank {rank} round {rnd}: the forward/backward did not finish while the "
                    "peer's all-reduce was pending (no overlap was exercised)"
                )


# --------------------------------------------------------------------------- standalone import


_STANDALONE = textwrap.dedent(
    """
    import importlib, importlib.abc, json, sys
    root, name, paths, subs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "flashinfer" or fullname.startswith("flashinfer."):
                raise ImportError("flashinfer is blocked: " + fullname)
            return None
    sys.meta_path.insert(0, Block())
    sys.path[:0] = [root] + paths
    package = importlib.import_module(name)
    for sub in subs:
        importlib.import_module(name + "." + sub)
    jit = sys.modules[name + ".cake_jit"]
    build = sys.modules[name + ".cake_build"]
    assert jit.MODULES and set(package.__all__) <= set(dir(package))
    library = next(iter(build.libraries_by_name()))
    try:
        build.load_library(library)
    except RuntimeError as exc:
        assert "is not prebuilt" in str(exc) and "JIT is not available" in str(exc), exc
    else:
        raise AssertionError("a library without a prebuilt file loaded with flashinfer blocked")
    leaked = sorted(m for m in sys.modules if m == "flashinfer" or m.startswith("flashinfer."))
    assert not leaked, leaked
    print("STANDALONE", name, "modules", len(jit.MODULES), "libraries", len(build.libraries_by_name()))
    """
)


def test_the_package_imports_standalone_under_another_name(tmp_path: Path) -> None:
    """A copy of the package (without ``aot/``) under another name imports in a fresh interpreter with
    ``flashinfer`` blocked: every submodule loads, nothing of flashinfer is imported, and a library with no
    prebuilt file raises the error naming the missing prebuilt and JIT instead of falling back silently."""

    source = Path(loader.__file__).parent
    name = "vendored_cake_rmsnorm_det_train_copy"
    shutil.copytree(
        source,
        tmp_path / name,
        ignore=shutil.ignore_patterns("aot", "__pycache__"),
    )
    # The interpreter's dependency paths (torch, tvm_ffi and their site directories); flashinfer is blocked by name.
    deps = [
        Path(torch.__file__).parent.parent,
        Path(sys.modules["tvm_ffi"].__file__).parent.parent,
    ]
    paths = [str(p) for p in deps] + [p for p in sys.path if p]
    subs = sorted(p.stem for p in source.glob("*.py") if p.stem != "__init__")
    assert {"cake_build", "cake_jit"} <= set(subs), subs
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _STANDALONE,
            str(tmp_path),
            name,
            json.dumps(paths),
            json.dumps(subs),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    print(done.stdout.strip())
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    assert "STANDALONE" in done.stdout
