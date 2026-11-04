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
(``RMSNorm(x) * w * SiLU(z)`` over ``[T, H, 128]``) with the opt-in
non-deterministic weight gradient (``deterministic=False``: one backward kernel
whose last CTA finishes the FP64 weight-gradient sum) against the same entry
points with the deterministic default (``deterministic=True``: the
deterministic package's backward kernel and its fixed-order weight-gradient
reduction).

Every row times, for both modes, on identical generated inputs:

- ``fwd``: one forward call (preallocated output);
- ``bwd``: one backward call into a BF16 weight gradient, the final weight
  gradient included (the deterministic reduction kernel, or the last CTA of
  the non-deterministic kernel), with a preallocated caller workspace;
- ``ckpt``: the activation-checkpointed step through autograd,
  ``torch.utils.checkpoint(cake_rmsnorm_silu_gate, ..., use_reentrant=False)``
  then the backward (which replays the forward): ``2 * fwd + bwd`` measured.

Timing uses CUPTI kernel activity with a cold L2
(``flashinfer.testing.bench_gpu_time(enable_cupti=True)``); each stage reports
the median and, as its variation, the samples' minimum, maximum and coefficient
of variation; ``speedup`` is the deterministic median over the
non-deterministic one.  Memory columns report the bytes autograd keeps for the backward
(``saved``, retained between forward and backward), the peak allocation of one
autograd forward plus backward beyond the inputs (``peak``, outputs included),
the backward's temporary allocation with preallocated outputs and no caller
workspace (``temp``) and the caller workspace of each mode (``workspace``).
Output columns check the non-deterministic results against the deterministic
ones: ``dx`` and ``dz`` bitwise, ``dw`` as the largest distance in BF16 units
in the last place, and the number of distinct ``dw`` bit patterns over
``--repeats`` backward calls.

Example::

    python benchmarks/bench_cake_rmsnorm_silu_gate_train.py --tokens 30193 32737 32388 --heads 48
    python benchmarks/bench_cake_rmsnorm_silu_gate_train.py --json out.json
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
    is_cake_rmsnorm_silu_gate_train_supported,
)
from flashinfer.testing import bench_gpu_time

D = 128
MODES = (("deterministic", True), ("nondeterministic", False))
STAGES = ("fwd", "bwd", "ckpt")
# The issue's token counts and a spread around them, then small and irregular ones.
TOKENS = (30193, 32737, 32388, 30000, 33000, 1, 7, 129, 1031, 4097)


def _stats(fn: Callable[[], object]) -> dict:
    """Median, minimum, maximum (ms) and coefficient of variation of the GPU
    time of ``fn``'s kernels, CUPTI, cold L2."""

    samples = [
        float(t) for t in bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True)
    ]
    mean = statistics.fmean(samples)
    return {
        "ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "cv": statistics.pstdev(samples) / mean if mean > 0 else 0.0,
        "n": len(samples),
    }


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


def _peak_bytes(call: Callable[[], object]) -> int:
    """Peak allocation of ``call`` beyond what is allocated before it."""

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak


def _ordered(bf16: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns as integers in the order of their values (+0 = -0)."""

    bits = bf16.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def bench_row(T: int, H: int, eps: float, seed: int, repeats: int) -> dict:
    x, z, w, dy = _inputs(T, H, seed)
    rows = T * H
    result: dict = {"T": T, "H": H, "rows": rows, "eps": eps, "modes": {}}
    outputs = {}
    for name, deterministic in MODES:
        y = torch.empty_like(x)
        dx, dz = torch.empty_like(x), torch.empty_like(x)
        dw = torch.empty(D, dtype=torch.bfloat16, device=x.device)
        ws = cake_rmsnorm_silu_gate_train_backward_workspace(
            rows, x.device, deterministic=deterministic
        )

        def fwd(y=y, deterministic=deterministic):
            return cake_rmsnorm_silu_gate_train_forward(
                x, z, w, eps, out=y, deterministic=deterministic
            )

        def bwd(dx=dx, dz=dz, dw=dw, ws=ws, deterministic=deterministic):
            return cake_rmsnorm_silu_gate_train_backward(
                dy,
                x,
                z,
                w,
                eps,
                grad_x=dx,
                grad_z=dz,
                grad_weight=dw,
                workspace=ws,
                deterministic=deterministic,
            )

        def bwd_no_workspace(dx=dx, dz=dz, dw=dw, deterministic=deterministic):
            return cake_rmsnorm_silu_gate_train_backward(
                dy,
                x,
                z,
                w,
                eps,
                grad_x=dx,
                grad_z=dz,
                grad_weight=dw,
                deterministic=deterministic,
            )

        def layer(xa, za, wa, deterministic=deterministic):
            return cake_rmsnorm_silu_gate(xa, za, wa, eps, deterministic=deterministic)

        xa, za, wa = (t.detach().requires_grad_(True) for t in (x, z, w))

        def step(xa=xa, za=za, wa=wa, layer=layer):
            yc = torch.utils.checkpoint.checkpoint(
                layer, xa, za, wa, use_reentrant=False
            )
            return torch.autograd.grad(yc, (xa, za, wa), dy)

        def fwd_bwd(xa=xa, za=za, wa=wa, layer=layer):
            return torch.autograd.grad(layer(xa, za, wa), (xa, za, wa), dy)

        # Outputs (and module loading) before timing.
        fwd()
        bwd()
        step()
        torch.cuda.synchronize()
        outputs[name] = (y.clone(), dx.clone(), dz.clone(), dw.clone())
        patterns = set()
        for _ in range(repeats):
            bwd()
            patterns.add(bytes(dw.view(torch.uint8).cpu().numpy()))
        entry: dict = {
            "fwd": _stats(fwd),
            "bwd": _stats(bwd),
            "ckpt": _stats(step),
            "saved_bytes": _saved_bytes(layer, x, z, w),
            "peak_bytes": _peak_bytes(fwd_bwd),
            "temp_bytes": _peak_bytes(bwd_no_workspace),
            "workspace_bytes": cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
                rows, x.device, deterministic=deterministic
            ),
            "dw_patterns": len(patterns),
            "repeats": repeats,
        }
        result["modes"][name] = entry
        del ws
        torch.cuda.empty_cache()
    # The non-deterministic outputs against the deterministic ones.
    reference, candidate = outputs["deterministic"], outputs["nondeterministic"]
    result["y_bitwise_equal"] = bool(torch.equal(reference[0], candidate[0]))
    result["dx_bitwise_equal"] = bool(torch.equal(reference[1], candidate[1]))
    result["dz_bitwise_equal"] = bool(torch.equal(reference[2], candidate[2]))
    result["dw_max_ulps_apart"] = int(
        (_ordered(reference[3]) - _ordered(candidate[3])).abs().max().item()
    )
    # deterministic GPU time over non-deterministic GPU time
    for stage in STAGES:
        result[f"{stage}_speedup"] = (
            result["modes"]["deterministic"][stage]["ms"]
            / result["modes"]["nondeterministic"][stage]["ms"]
        )
    return result


def _format(result: dict) -> list[str]:
    head = f"T={result['T']:>6} H={result['H']:>3}"
    lines = []
    for stage in STAGES:
        cells = []
        for name, _ in MODES:
            s = result["modes"][name][stage]
            cells.append(
                f"{name} {s['ms'] * 1e3:9.1f} us "
                f"[{s['min_ms'] * 1e3:8.1f}, {s['max_ms'] * 1e3:8.1f}] cv {s['cv'] * 100:4.1f}%"
            )
        lines.append(
            f"{head} {stage:4s} | "
            + " | ".join(cells)
            + f" | speedup x{result[f'{stage}_speedup']:5.2f}"
        )
    for name, _ in MODES:
        m = result["modes"][name]
        lines.append(
            f"{head} mem  | {name:16s} saved {m['saved_bytes'] / 2**20:8.1f} MiB  "
            f"peak {m['peak_bytes'] / 2**20:8.1f} MiB  temp {m['temp_bytes'] / 2**10:9.1f} KiB  "
            f"workspace {m['workspace_bytes'] / 2**10:9.1f} KiB  "
            f"dw patterns {m['dw_patterns']}/{m['repeats']}"
        )
    lines.append(
        f"{head} out  | y/dx/dz bitwise equal: {result['y_bitwise_equal']}/"
        f"{result['dx_bitwise_equal']}/{result['dz_bitwise_equal']}  "
        f"dw at most {result['dw_max_ulps_apart']} BF16 ulp apart"
    )
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--heads", type=int, nargs="+", default=[48])
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="backward calls per mode whose dw bit patterns are counted",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    device = torch.device("cuda", torch.cuda.current_device())
    if not is_cake_rmsnorm_silu_gate_train_supported(device, deterministic=False):
        raise SystemExit(
            "cake_rmsnorm_silu_gate_train(deterministic=False) has no exported route on this device"
        )
    torch.use_deterministic_algorithms(False)
    props = torch.cuda.get_device_properties(device)
    print(
        f"# {props.name}, sm_{props.major}{props.minor}, {props.multi_processor_count} SMs, "
        f"CUDA {torch.version.cuda}, torch {torch.__version__}"
    )
    results = []
    for H in args.heads:
        for T in args.tokens:
            result = bench_row(T, H, args.eps, args.seed, args.repeats)
            results.append(result)
            for line in _format(result):
                print(line, flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "device": props.name,
                    "capability": [props.major, props.minor],
                    "sm_count": props.multi_processor_count,
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
