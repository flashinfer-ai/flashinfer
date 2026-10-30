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

Benchmark of the Cake SiLU-gated RMSNorm training kernels
(``RMSNorm(x) * w * SiLU(z)`` over ``[T, H, 128]``) against the training-side
comparators: the flash-linear-attention Triton function ``rms_norm_gated``
(the kernel of its ``FusedRMSNormGated`` module, ``activation="swish"``, when
installed), ``torch.compile`` (inductor) of the FP32 formula and eager autograd
of the same formula.

Every row times, on identical generated inputs:

- ``fwd``: one forward call as training runs it (autograd recording);
- ``bwd``: one backward call (the weight-gradient reduction and its final cast
  included; for the autograd comparators the time of forward plus backward
  less the forward);
- ``2fwd+bwd``: the activation-checkpointed step, ``2 * fwd + bwd``;
- ``ckpt``: the same step measured: ``torch.utils.checkpoint(...,
  use_reentrant=False)`` of the forward, then the backward (which replays it).

Timing uses CUPTI kernel activity with a cold L2
(``flashinfer.testing.bench_gpu_time(enable_cupti=True)``); a comparator of
several kernels reports the sum of their GPU time, copies and initializations
included.  Memory columns report the bytes autograd saves for the backward
(``saved``) and the peak allocation of one forward plus backward beyond the
inputs and gradients (``peak``); the Cake rows also report the backward's
workspace (``workspace``), the FP32 weight-gradient partials it allocates per
call unless the caller passes one.  Each comparator's outputs are checked
against the Cake outputs (largest absolute difference of ``y``, ``dx``, ``dz``,
``dw``).

``--side-stream`` adds, for the Cake kernels, the event-timed median of seven
forward and seven backward calls beside a one-thread kernel resident on
another stream over the median of seven calls alone (alternating), the
concurrent-launch measurement of issue #7038.

Example::

    python benchmarks/bench_cake_rmsnorm_silu_gate_train.py --tokens 30193 32737 32388 --heads 48
    python benchmarks/bench_cake_rmsnorm_silu_gate_train.py --side-stream --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable, Optional

import torch
import torch.utils.checkpoint

from flashinfer.cake_rmsnorm_silu_gate_train import (
    cake_rmsnorm_silu_gate,
    cake_rmsnorm_silu_gate_train_backward,
    cake_rmsnorm_silu_gate_train_backward_workspace,
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_train_forward,
)
from flashinfer.testing import bench_gpu_time

D = 128
# The side-stream measurement: a one-thread kernel resident for about 5 ms,
# started after a 1M-cycle sleep on the main stream (issue #7038's repro).
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
SIDE_RUNS = 7


def _time(fn: Callable[[], object]) -> float:
    """Median GPU time (ms) of ``fn`` over its kernels, cold L2, CUPTI."""

    return float(
        statistics.median(bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True))
    )


def _event_ms(call: Callable[[], object], side: Optional[torch.cuda.Stream]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if side is not None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _beside_resident(call: Callable[[], object]) -> dict:
    """Median ms of ``call`` alone and beside a resident side-stream kernel."""

    side = torch.cuda.Stream()
    for _ in range(3):
        _event_ms(call, None)
    alone, busy = [], []
    for _ in range(SIDE_RUNS):
        alone.append(_event_ms(call, None))
        busy.append(_event_ms(call, side))
    alone_ms, busy_ms = statistics.median(alone), statistics.median(busy)
    return {"alone_ms": alone_ms, "busy_ms": busy_ms, "ratio": busy_ms / alone_ms}


def _formula(x, z, w, eps):
    xf, zf, wf = x.float(), z.float(), w.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (((xf * rstd) * wf) * torch.nn.functional.silu(zf)).to(x.dtype)


def _comparators(eps: float) -> dict[str, Callable]:
    """name -> ``f(x, z, w)``, a differentiable forward."""

    found: dict[str, Callable] = {
        "cake": lambda x, z, w: cake_rmsnorm_silu_gate(x, z, w, eps),
    }
    try:
        from fla.modules.fused_norm_gate import rms_norm_gated

        # The functional form takes ``w`` itself, so the weight gradient flows
        # back to the caller's tensor (a module would hold its own parameter).
        found["fla_triton"] = lambda x, z, w: rms_norm_gated(
            x, z, w, None, "swish", eps=eps
        )
    except Exception as exc:  # noqa: BLE001
        print(f"# flash-linear-attention unavailable: {type(exc).__name__}: {exc}")
    compiled = torch.compile(lambda x, z, w: _formula(x, z, w, eps), dynamic=False)
    found["torch_compile"] = compiled
    found["eager"] = lambda x, z, w: _formula(x, z, w, eps)
    return found


def _inputs(T: int, H: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, D, device="cuda", generator=g).bfloat16()
    z = torch.randn(T, H, D, device="cuda", generator=g).bfloat16()
    dy = torch.randn(T, H, D, device="cuda", generator=g).bfloat16()
    w = (1.0 + 0.5 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    return x, z, w, dy


def _saved_bytes(fn, x, z, w) -> int:
    total = 0

    def pack(t):
        nonlocal total
        total += t.numel() * t.element_size()
        return t

    xa, za, wa = (t.detach().requires_grad_(True) for t in (x, z, w))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(xa, za, wa)
    return total


def bench_row(
    T: int,
    H: int,
    eps: float,
    comparators: dict[str, Callable],
    seed: int,
    side_stream: bool = False,
) -> dict:
    x, z, w, dy = _inputs(T, H, seed)
    result: dict = {"T": T, "H": H, "eps": eps, "rows": T * H}
    reference = None
    for name, fn in comparators.items():
        xa, za, wa = (t.detach().requires_grad_(True) for t in (x, z, w))

        def train_fwd():
            return fn(xa, za, wa)

        def train_fwd_bwd():
            y = fn(xa, za, wa)
            return torch.autograd.grad(y, (xa, za, wa), dy)

        def train_ckpt():
            y = torch.utils.checkpoint.checkpoint(fn, xa, za, wa, use_reentrant=False)
            return torch.autograd.grad(y, (xa, za, wa), dy)

        # Outputs (and warm-up / compilation) before timing.
        y = train_fwd()
        grads = torch.autograd.grad(y, (xa, za, wa), dy)
        outs = (y.detach(), *grads)
        if reference is None:
            reference = outs
        diff = [
            float((a.float() - b.float()).abs().max())
            for a, b in zip(outs, reference, strict=True)
        ]
        row: dict = {
            "max_abs_diff_vs_cake": dict(
                zip(("y", "dx", "dz", "dw"), diff, strict=True)
            )
        }
        if name == "cake":
            fwd_ms = _time(lambda: cake_rmsnorm_silu_gate_train_forward(x, z, w, eps))
            bwd_ms = _time(
                lambda: cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, eps)
            )
        else:
            fwd_ms = _time(train_fwd)
            bwd_ms = _time(train_fwd_bwd) - fwd_ms
        row.update(fwd_ms=fwd_ms, bwd_ms=bwd_ms, total_ms=2 * fwd_ms + bwd_ms)
        train_ckpt()
        row["ckpt_ms"] = _time(train_ckpt)
        row["saved_bytes"] = _saved_bytes(fn, x, z, w)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        y = fn(xa, za, wa)
        grads = torch.autograd.grad(y, (xa, za, wa), dy)
        torch.cuda.synchronize()
        row["peak_bytes"] = torch.cuda.max_memory_allocated() - base
        del y, grads
        result[name] = row
    cake = result["cake"]
    cake["workspace_bytes"] = cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
        T * H, x.device
    )
    if side_stream:
        y = torch.empty_like(x)
        ws = cake_rmsnorm_silu_gate_train_backward_workspace(T * H, x.device)
        cake["side_stream"] = {
            "fwd": _beside_resident(
                lambda: cake_rmsnorm_silu_gate_train_forward(x, z, w, eps, out=y)
            ),
            "bwd": _beside_resident(
                lambda: cake_rmsnorm_silu_gate_train_backward(
                    dy, x, z, w, eps, workspace=ws
                )
            ),
        }
    nbytes_fwd = 3 * T * H * D * 2
    nbytes_bwd = 5 * T * H * D * 2
    result["cake"]["fwd_gbs"] = nbytes_fwd / result["cake"]["fwd_ms"] / 1e6
    result["cake"]["bwd_gbs"] = nbytes_bwd / result["cake"]["bwd_ms"] / 1e6
    return result


def _format(result: dict, names) -> list[str]:
    lines = []
    cake = result["cake"]["total_ms"]
    for name in names:
        row = result[name]
        lines.append(
            f"T={result['T']:>6} H={result['H']:>3} {name:>13}: fwd {row['fwd_ms'] * 1e3:9.1f} us  "
            f"bwd {row['bwd_ms'] * 1e3:9.1f} us  2fwd+bwd {row['total_ms'] * 1e3:9.1f} us  "
            f"ckpt {row['ckpt_ms'] * 1e3:9.1f} us  x{row['total_ms'] / cake:5.2f} of cake  "
            f"saved {row['saved_bytes'] / 2**20:8.1f} MiB  peak {row['peak_bytes'] / 2**20:8.1f} MiB"
        )
        if "workspace_bytes" in row:
            lines[-1] += f"  workspace {row['workspace_bytes'] / 2**10:8.1f} KiB"
        for stage, side in row.get("side_stream", {}).items():
            lines.append(
                f"T={result['T']:>6} H={result['H']:>3} {name:>13}: {stage} beside a resident kernel "
                f"{side['busy_ms'] * 1e3:9.1f} us vs {side['alone_ms'] * 1e3:9.1f} us alone (x{side['ratio']:5.3f})"
            )
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[30193, 32737, 32388, 1024, 8192, 65536],
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[48])
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only", nargs="*", default=None, help="comparators to run (cake always runs)"
    )
    parser.add_argument(
        "--side-stream",
        action="store_true",
        help="also time the Cake kernels beside a resident kernel",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    comparators = _comparators(args.eps)
    if args.only:
        comparators = {
            k: v for k, v in comparators.items() if k == "cake" or k in args.only
        }
    device = torch.cuda.get_device_properties(0)
    print(
        f"# {device.name}, sm_{device.major}{device.minor}, CUDA {torch.version.cuda}, torch {torch.__version__}"
    )
    results = []
    for H in args.heads:
        for T in args.tokens:
            result = bench_row(T, H, args.eps, comparators, args.seed, args.side_stream)
            results.append(result)
            for line in _format(result, comparators):
                print(line, flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "device": device.name,
                    "cuda": torch.version.cuda,
                    "torch": torch.__version__,
                    "rows": results,
                },
                fh,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
