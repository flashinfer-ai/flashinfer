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

The deterministic causal conv1d + SiLU training entry points under torch's deterministic mode.

``torch.use_deterministic_algorithms(True)`` (``warn_only=False``) turns every nondeterministic-op alert into an
error.  The deterministic entries -- the single-rank forward / backward and autograd entry of
``flashinfer.cake_causal_conv1d_det_train`` (default, bias + FP32 SiLU', BF16 taps, a finer backward partition) and
the KCP single and multi autograd entries of ``flashinfer.cake_causal_conv1d_det_train_kcp`` (both backward launch
forms) on a one-rank context whose fragments continue documents -- run under it without an error or a warning, and
return the bits they return outside it.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from flashinfer.cake_causal_conv1d_det_train import (
    cake_causal_conv1d_det,
    cake_causal_conv1d_det_train_backward,
    cake_causal_conv1d_det_train_forward,
    is_cake_causal_conv1d_det_train_supported,
)
from flashinfer.cake_causal_conv1d_det_train_kcp import (
    cake_causal_conv1d_det_kcp,
    cake_causal_conv1d_det_kcp_multi,
)

WIDTH = 4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_causal_conv1d_det_train_supported(torch.device("cuda")),
    reason="cake_causal_conv1d_det_train needs a supported CUDA device (SM100 / SM103 / SM107)",
)


def _under_deterministic_mode(fn):
    """``fn()`` under ``torch.use_deterministic_algorithms(True)`` (alerts are errors); every warning is recorded and
    returned with the result.  The previous mode is restored."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn()
            torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return out, [f"{w.category.__name__}: {w.message}" for w in caught]


def _same(a, b) -> bool:
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    )


def _check(fn) -> None:
    reference = fn()
    torch.cuda.synchronize()
    got, caught = _under_deterministic_mode(fn)
    assert not caught, caught
    assert _same(got, reference), "the result differs under torch's deterministic mode"


def _inputs(S, C, device, *, wdtype=torch.float32, bias=False, seed=2033):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(S, C, device=device, generator=g, dtype=torch.bfloat16)
    w = (torch.randn(C, WIDTH, device=device, generator=g) * 0.5).to(wdtype)
    b = torch.randn(C, device=device, generator=g) if bias else None
    dy = torch.randn(S, C, device=device, generator=g, dtype=torch.bfloat16)
    return x, w, b, dy


ROUTES = [
    ("default", {}, {}),
    ("bias_fp32_silu", {"bias": True}, {"silu_grad_preactivation": "fp32"}),
    ("bf16_taps", {"wdtype": torch.bfloat16}, {}),
    ("grid_mult_2", {}, {"bwd_grid_mult": 2}),
]


@pytest.mark.parametrize("name,make,kw", ROUTES, ids=[r[0] for r in ROUTES])
def test_single_rank_entries_raise_no_deterministic_alert(name, make, kw) -> None:
    device = torch.device("cuda")
    lens = (700, 1, 300)
    cu = torch.tensor([0, 700, 701, 1001], dtype=torch.int32, device=device)
    x, w, b, dy = _inputs(sum(lens), 512, device, **make)

    def direct():
        y = cake_causal_conv1d_det_train_forward(x, w, cu, bias=b)
        grads = cake_causal_conv1d_det_train_backward(x, w, cu, dy, bias=b, **kw)
        return (y, *grads)

    def autograd():
        xl = x.clone().requires_grad_(True)
        wl = w.clone().requires_grad_(True)
        bl = None if b is None else b.clone().requires_grad_(True)
        y = cake_causal_conv1d_det(xl, wl, cu, bl, **kw)
        y.backward(dy)
        return (y.detach(), xl.grad, wl.grad, *(() if bl is None else (bl.grad,)))

    _check(direct)
    _check(autograd)


def _one_rank_context(lens, cuts, device):
    """A one-rank KCP context (the KDA KCP plan) over documents ``lens`` cut into fragments ``cuts``."""
    from flashinfer.experimental.cake.cake_kda_det_train_kcp.cake_kcp_plan import (
        cake_kda_det_train_kcp_build_context,
        cake_kda_det_train_kcp_partition,
    )

    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    frags, pos = [], 0
    for n in cuts:
        p, end = pos, pos + n
        while p < end:
            d = max(i for i in range(len(lens)) if cu[i] <= p)
            e = min(end, cu[d + 1])
            frags.append((d, p - cu[d], e - p))
            p = e
        pos = end
    part = cake_kda_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, fragments=[frags]
    )
    return cake_kda_det_train_kcp_build_context(
        part, None, 0, device, conv_kernel_size=WIDTH
    )


@pytest.mark.parametrize("cooperative", [True, False], ids=["cooperative", "split"])
def test_kcp_entries_raise_no_deterministic_alert(cooperative: bool) -> None:
    device = torch.device("cuda")
    # fragments of one and two tokens continue the first document: the history and head rows cross fragments
    lens, cuts = (2000, 503), (1, 2, 1997, 503)
    ctx = _one_rank_context(lens, cuts, device)
    S = sum(lens)
    inputs = [_inputs(S, C, device, seed=2033 + C) for C in (512, 256)]

    def single():
        x, w, _, dy = inputs[0]
        xl = x.clone().requires_grad_(True)
        wl = w.clone().requires_grad_(True)
        y = cake_causal_conv1d_det_kcp(xl, wl, ctx, cooperative=cooperative)
        y.backward(dy)
        return (y.detach(), xl.grad, wl.grad)

    def multi():
        xs = [x.clone().requires_grad_(True) for x, _, _, _ in inputs]
        ws = [w.clone().requires_grad_(True) for _, w, _, _ in inputs]
        ys = cake_causal_conv1d_det_kcp_multi(xs, ws, ctx, cooperative=cooperative)
        torch.autograd.backward(ys, [dy for _, _, _, dy in inputs])
        return (*(y.detach() for y in ys), *(x.grad for x in xs), *(w.grad for w in ws))

    _check(single)
    _check(multi)
