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

CUDA-graph capture of the Cake four-stream mHC training entry points (hidden
4096, 4 streams; SM100 / SM103 / SM107).

One training iteration of a layer -- prepare, an external branch ``B = U @ W``
(BF16, ``W`` fixed), finish, the loss and its backward -- is captured whole
with ``torch.cuda.graph`` after a warm-up on a side stream (the JIT modules and
the per-device default workspaces exist before the capture), in three forms:

* the direct entry points ``forward_prepare`` -> ``forward_finish`` ->
  ``backward_prepare`` -> ``backward_finish`` (the branch backward
  ``dU = dB @ W^T`` in between);
* the autograd entry points ``cake_mhc_train_prepare`` /
  ``cake_mhc_train_finish`` and ``loss.backward()`` (``save_stats`` both ways);
* the same layer inside a non-reentrant ``torch.utils.checkpoint``
  (``preserve_rng_state=False``: the layer draws no random numbers);

and the autograd / checkpointed layer through
``torch.cuda.make_graphed_callables``.  The captured forms run with the
default workspaces and with caller-owned workspaces.  Three replays, each after
the static outputs and gradients were overwritten with 0xFF bytes, give every
output and gradient bitwise equal to the eager iteration on the same inputs
(the units' ticket words reset themselves at the end of every launch, so a
replay needs no host-side reset).  The graphed callables get new input values
on each of three calls, each compared bitwise with an eager call on the same
values.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_mhc_train")
loader = cake_package("cake_mhc_train", "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
BF16, F32 = torch.bfloat16, torch.float32
REPLAYS = 3


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(torch.cuda.get_device_capability(_device()))
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _inputs(T: int, seed: int) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, HIDDEN, dtype=BF16),
        "fn": (randn(MAPPING_ROWS, N) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(MAPPING_ROWS),
        "W": (randn(HIDDEN, HIDDEN) / 64.0).to(BF16),
        "dY": randn(T, STREAMS, HIDDEN, dtype=BF16),
    }


def _workspaces(T: int, caller_owned: bool):
    if not caller_owned:
        return None, None
    return (
        pkg.cake_mhc_train_forward_workspace(T, _device()),
        pkg.cake_mhc_train_backward_workspace(T, _device()),
    )


def _leaves(i: dict) -> list:
    return [
        i[k].detach().clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")
    ]


def _layer(i: dict, wsf, wsb, save_stats: bool):
    def layer(X, fn, scale, base):
        U, handle = pkg.cake_mhc_train_prepare(
            X,
            fn,
            scale,
            base,
            save_stats=save_stats,
            workspace=wsf,
            backward_workspace=wsb,
        )
        return pkg.cake_mhc_train_finish(X, U @ i["W"], handle)

    return layer


def _direct_step(i: dict, wsf, wsb):
    X, fn, scale, base, W, dY = (i[k] for k in ("X", "fn", "scale", "base", "W", "dY"))

    def step() -> dict:
        U, stats = pkg.cake_mhc_train_forward_prepare(X, fn, scale, base, workspace=wsf)
        B = U @ W
        Y = pkg.cake_mhc_train_forward_finish(X, B, stats)
        dB, gstats = pkg.cake_mhc_train_backward_prepare(dY, B, X, stats)
        dX, dfn, dscale, dbase = pkg.cake_mhc_train_backward_finish(
            X, dY, dB @ W.t(), B, fn, scale, base, stats, gstats, workspace=wsb
        )
        return {
            "U": U,
            "stats": stats,
            "Y": Y,
            "dB": dB,
            "grad_stats": gstats,
            "dX": dX,
            "dfn": dfn,
            "dscale": dscale,
            "dbase": dbase,
        }

    return step, ()


def _autograd_step(i: dict, wsf, wsb, *, save_stats: bool, ckpt: bool):
    layer = _layer(i, wsf, wsb, save_stats)
    leaves = _leaves(i)
    dY = i["dY"].float()

    def step() -> dict:
        if ckpt:
            Y = checkpoint(
                layer, *leaves, use_reentrant=False, preserve_rng_state=False
            )
        else:
            Y = layer(*leaves)
        (Y.float() * dY).sum().backward()
        return {"Y": Y, **_grads(leaves)}

    return step, leaves


def _grads(leaves) -> dict:
    return {
        f"grad_{n}": t.grad
        for n, t in zip(("X", "fn", "scale", "base"), leaves, strict=True)
    }


def _reset(leaves) -> None:
    for t in leaves:
        t.grad = None


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _poison(out: dict) -> None:
    for v in out.values():
        v.detach().reshape(-1).view(torch.uint8).fill_(0xFF)


def _assert_bitwise(ref: dict, got: dict, what: str) -> None:
    for k, v in ref.items():
        assert torch.equal(
            v.reshape(-1).view(torch.uint8), got[k].reshape(-1).view(torch.uint8)
        ), f"{what}: {k} differs from the eager iteration"


def _eager(make) -> dict:
    step, leaves = make()
    _reset(leaves)
    return _snapshot(step())


def _capture_and_replay(make) -> list:
    step, leaves = make()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _reset(leaves)
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    _reset(leaves)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    replays = []
    for _ in range(REPLAYS):
        _poison(out)
        graph.replay()
        torch.cuda.synchronize()
        replays.append(_snapshot(out))
    return replays


@pytest.mark.parametrize("T", (1, 129, 2049))
@pytest.mark.parametrize("caller_owned", (False, True))
def test_direct_entry_points_capture(T: int, caller_owned: bool) -> None:
    i = _inputs(T, seed=7046 + T)
    wsf, wsb = _workspaces(T, caller_owned)
    ref = _eager(lambda: _direct_step(i, wsf, wsb))
    for r, got in enumerate(_capture_and_replay(lambda: _direct_step(i, wsf, wsb))):
        _assert_bitwise(ref, got, f"direct T={T} replay {r}")


@pytest.mark.parametrize("T", (1, 129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("ckpt", (False, True))
@pytest.mark.parametrize("caller_owned", (False, True))
def test_autograd_iteration_capture(
    T: int, save_stats: bool, ckpt: bool, caller_owned: bool
) -> None:
    i = _inputs(T, seed=7146 + T)
    wsf, wsb = _workspaces(T, caller_owned)

    def make():
        return _autograd_step(i, wsf, wsb, save_stats=save_stats, ckpt=ckpt)

    ref = _eager(make)
    for r, got in enumerate(_capture_and_replay(make)):
        _assert_bitwise(ref, got, f"autograd ckpt={ckpt} T={T} replay {r}")


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("ckpt", (False, True))
def test_make_graphed_callables(T: int, save_stats: bool, ckpt: bool) -> None:
    i = _inputs(T, seed=7246 + T)
    layer = _layer(i, None, None, save_stats)
    if ckpt:

        def fwd(X, fn, scale, base):
            return checkpoint(
                layer, X, fn, scale, base, use_reentrant=False, preserve_rng_state=False
            )

    else:
        fwd = layer

    def run(f, inp: dict) -> dict:
        leaves = _leaves(inp)
        Y = f(*leaves)
        (Y.float() * inp["dY"].float()).sum().backward()
        return _snapshot({"Y": Y, **_grads(leaves)})

    graphed = torch.cuda.make_graphed_callables(
        fwd, tuple(_leaves(i)), num_warmup_iters=3
    )
    for r in range(REPLAYS):
        # New values of the same shapes on every call: a call whose graphs did
        # not replay would return the previous call's outputs and fail.
        ir = _inputs(T, seed=7246 + T + 1000 * (r + 1))
        ref = run(fwd, ir)
        got = run(graphed, ir)
        _assert_bitwise(ref, got, f"make_graphed_callables ckpt={ckpt} T={T} call {r}")
