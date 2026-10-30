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

Stall regression of the Cake SiLU-gated RMSNorm training forward and backward
beside a kernel resident on another stream (the concurrent-launch contract of
issue #7038).

The forward and the backward's main kernel launch plain grids of independent
CTAs and the weight-gradient reduction a fixed grid of 128 CTAs; no kernel
waits for another CTA or grid, so none needs the whole device.  Beside a
long-resident one-thread kernel on another stream each call must keep its
speed: per token count, the median of seven event-timed calls with the side
kernel resident is at most ``MAX_RATIO`` times the median of seven calls
alone, or at most ``MAX_EXTRA_MS`` above it.  The alone and busy calls
alternate in pairs, so host launch gaps and clock drift hit both arms alike.
The bounds sit above the slowdown a resident kernel causes (the CTAs sharing
its SM run a little slower) and below the stall this test guards against, a
wait of milliseconds behind the resident kernel.  ``y``, ``dx``, ``dz`` and
``dw`` are bitwise equal across the alone and busy calls.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_train")
cake_rmsnorm_silu_gate_train_backward = pkg.cake_rmsnorm_silu_gate_train_backward
cake_rmsnorm_silu_gate_train_backward_workspace = (
    pkg.cake_rmsnorm_silu_gate_train_backward_workspace
)
cake_rmsnorm_silu_gate_train_forward = pkg.cake_rmsnorm_silu_gate_train_forward
is_cake_rmsnorm_silu_gate_train_supported = (
    pkg.is_cake_rmsnorm_silu_gate_train_supported
)

D = 128
H = 48
EPS = 1e-6
# The issue's token counts and a small and a large one.
TOKENS = (256, 4096, 30193, 32737, 32388, 65536)
# Side kernel: one thread resident for about 5 ms; the main stream sleeps first
# so the side kernel is resident when the call starts.
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
WARMUP = 3
RUNS = 7
# A row passes on either bound: the busy median over the alone median, or the
# busy median minus the alone median in ms.
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020
SEED = 7075

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_train_supported(torch.device("cuda")),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


def _inputs(tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(SEED + tokens)
    kwargs = dict(dtype=torch.float32, device="cuda", generator=generator)
    x = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    z = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    dy = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    w = (1 + 0.5 * torch.randn(D, **kwargs)).to(torch.bfloat16)
    return x, z, w, dy


def _timed(call, *, side: torch.cuda.Stream | None):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if side is not None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), tuple(t.clone() for t in out)


def _keeps_its_speed(alone_ms: float, busy_ms: float) -> bool:
    return busy_ms / alone_ms <= MAX_RATIO or busy_ms - alone_ms <= MAX_EXTRA_MS


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("stage", ["forward", "backward"])
def test_keeps_its_speed_beside_a_resident_kernel(stage: str, tokens: int) -> None:
    x, z, w, dy = _inputs(tokens)
    if stage == "forward":
        y = torch.empty_like(x)

        def call():
            return (cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS, out=y),)

    else:
        workspace = cake_rmsnorm_silu_gate_train_backward_workspace(
            tokens * H, x.device
        )

        def call():
            return cake_rmsnorm_silu_gate_train_backward(
                dy, x, z, w, EPS, workspace=workspace
            )

    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call, side=None)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call, side=None))
        busy.append(_timed(call, side=side))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    print(
        f"STALL stage={stage} tokens={tokens} heads={H} alone_ms={alone_ms:.4f} "
        f"busy_ms={busy_ms:.4f} ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    reference = alone[0][1]
    for _t, out in alone[1:] + busy:
        for got, want in zip(out, reference, strict=True):
            assert torch.equal(got, want), (
                f"{stage} tokens={tokens}: results differ across calls"
            )
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"{stage} tokens={tokens}: {busy_ms:.4f} ms beside a resident kernel vs "
        f"{alone_ms:.4f} ms alone"
    )
