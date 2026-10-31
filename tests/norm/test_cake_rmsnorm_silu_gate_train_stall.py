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

Stall regression of the Cake SiLU-gated RMSNorm training forward and the
non-deterministic backward (``deterministic=False``) beside a kernel resident
on another stream (the concurrent-launch contract of issue #7038).

The forward launches a plain grid of independent CTAs.  The backward launches
one plain grid whose CTAs add their weight-gradient sums into FP64
accumulators and take a ticket on an arrival counter; whichever CTA arrives
last finishes the sum, and no CTA waits for another, so the kernel never needs
the whole device.  Beside a long-resident one-thread kernel on another stream
each call must keep its speed: per token count, the median of seven
event-timed calls with the side kernel resident is at most ``MAX_RATIO`` times
the median of seven calls alone, or at most ``MAX_EXTRA_MS`` above it.  The
alone and busy calls alternate in pairs, so host launch gaps and clock drift
hit both arms alike.  ``y``, ``dx`` and ``dz`` are bitwise equal across the
alone and busy calls; ``dw``, whose FP64 cross-CTA sum depends on scheduling,
stays within one BF16 unit in the last place of the first call's; the
workspace is zero after every call.
"""

from __future__ import annotations

import functools
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_train")
forward = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_forward, deterministic=False
)
backward = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_backward, deterministic=False
)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_backward_workspace, deterministic=False
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
SEED = 8067

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_rmsnorm_silu_gate_train_supported(
        torch.device("cuda"), deterministic=False
    ),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


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


def _ordered(bf16: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns as integers in the order of their values (+0 = -0)."""
    bits = bf16.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("stage", ["forward", "backward"])
def test_keeps_its_speed_beside_a_resident_kernel(stage: str, tokens: int) -> None:
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)
    if stage == "forward":
        y = torch.empty_like(x)

        def call():
            return (forward(x, z, w, EPS, out=y),)

    else:

        def call():
            return backward(dy, x, z, w, EPS, workspace=workspace)

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
        if stage == "forward":
            assert torch.equal(out[0], reference[0]), f"y tokens={tokens}"
            continue
        assert torch.equal(out[0], reference[0]), f"dx tokens={tokens}"
        assert torch.equal(out[1], reference[1]), f"dz tokens={tokens}"
        ulps = (_ordered(out[2]) - _ordered(reference[2])).abs().max().item()
        assert ulps <= 1, f"dw tokens={tokens}: {ulps} BF16 units apart across calls"
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"{stage} tokens={tokens}: {busy_ms:.4f} ms beside a resident kernel vs "
        f"{alone_ms:.4f} ms alone"
    )
