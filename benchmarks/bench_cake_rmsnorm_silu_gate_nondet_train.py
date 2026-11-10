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
(``RMSNorm(x) * w * SiLU(z)`` over ``[T, H, 128]``) of the non-deterministic
package (``flashinfer.cake_rmsnorm_silu_gate_nondet_train``: a forward with
approximate FP32 intrinsics, and one backward kernel whose CTAs add the
weight-gradient partials exactly as 64-bit integer windows, the last CTA
rounding the sum once, or, for a BF16 weight gradient in its fused band, run
the deterministic package's fused backward's schedule and row arithmetic and
add FP32 runs of rows to FP64 accumulators with atomics, and past that band
run its copy of the deterministic package's backward) against the
deterministic package
(``flashinfer.cake_rmsnorm_silu_gate_det_train``: its one-kernel backward with
the exact fixed-point weight-gradient sum), each called through its own public
entry points.

Every row times, for both modes, on identical generated inputs:

- ``fwd``: one forward call (preallocated output);
- ``bwd``: one backward call into a BF16 weight gradient, the final weight
  gradient included (the last CTA of either backward), with a preallocated
  caller workspace;
- ``ckpt``: the activation-checkpointed step through autograd,
  ``torch.utils.checkpoint(cake_rmsnorm_silu_gate_nondet, ..., use_reentrant=False)``
  then the backward (which replays the forward): ``2 * fwd + bwd`` measured.

Each package picks its kernels from the rows and the SM count.  The
non-deterministic forward of packed rows (the benchmark's inputs) runs its tiny
forward up to 38 rows per SM on SM100, 36 on SM103 and 65 on SM107 and past
that its mid, long, extra-long or pipelined forward (its copy of the
deterministic forward) by bands of rows per SM (``cake_jit.FORWARD_BANDS``).
Its backward runs one cluster of 16 CTAs for few rows, else one kernel by bands
of rows per SM (``cake_jit.BACKWARD_BANDS``).  A BF16 weight gradient (the one
timed here) runs the short backward (one cluster, a fixed-order sum) up to 256
rows and, on SM100 and SM107, the exact tiny backward (one cluster, the bits of
the deterministic pair at FlashInfer 2e989556) up to 512 rows.  Past that, on
SM100 and SM103, it runs the fused backward (256-thread CTAs of 16-row blocks,
at most 4 per SM) up to 1000 rows per SM and the repeatable backward (its copy
of the deterministic package's backward, the same grid; its ``dw`` repeats its
bits) past it; on SM107 the repeatable backward up to 64 rows per SM, the
fused backward up to 256 and the repeatable backward past it.  From 2560 rows
per SM on SM100 and SM103 and 1536 on SM107 packed rows run the repeatable
backward's long program (its copy of the deterministic package's look-ahead
program, the same bits; ``cake_jit.LONG_BACKWARD_ROWS_PER_SM``).
An FP32 weight gradient runs the exact tiny backward up to 512 rows, the
static backward up to 256 rows per SM on SM100 and SM103 (64 on SM107) and the
claimed backward past that.  At ``--heads 48`` the default token counts reach
every forward kernel of packed rows and every BF16 backward kernel that a band
of a 148-SM B200 runs (2048 tokens: the extra-long forward).

Timing uses CUPTI kernel activity with a cold L2
(``flashinfer.testing.bench_gpu_time(enable_cupti=True)``); each stage reports
the median and, as its variation, the samples' minimum, maximum and coefficient
of variation; ``speedup`` is the deterministic median over the
non-deterministic one.  Memory columns report the bytes autograd keeps for the backward
(``saved``, retained between forward and backward), the peak allocation of one
autograd forward plus backward beyond the inputs (``peak``, outputs included),
the backward's temporary allocation with preallocated outputs and no caller
workspace (``temp``; the zero-filled ``torch.uint8`` workspaces that
non-deterministic calls without a caller workspace keep, one per device,
stream and kernel that takes one and a reserve per device, come from the calls before it) and
the caller workspace of each mode (``workspace``: the non-deterministic one
29192 bytes on SM100 and SM103 and 36360 on SM107, whatever the rows).  Output columns check the non-deterministic
results against the deterministic ones: ``y`` bitwise (the non-deterministic
forward's approximate intrinsics need not give the deterministic ``y`` bits),
``dx``, ``dz`` and ``dw`` as the largest distance in BF16 units in the last
place, and the number of distinct ``dw`` bit patterns over ``--repeats``
backward calls (1 for the deterministic package; 1 for the non-deterministic
package too, except in its fused band, where the BF16 ``dw``'s last bits may
vary from run to run, so the count may exceed 1).

Example::

    python benchmarks/bench_cake_rmsnorm_silu_gate_nondet_train.py --tokens 30193 32737 32388 --heads 48
    python benchmarks/bench_cake_rmsnorm_silu_gate_nondet_train.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable, Optional

import torch
import torch.utils.checkpoint

import flashinfer.cake_rmsnorm_silu_gate_det_train as det
import flashinfer.cake_rmsnorm_silu_gate_nondet_train as nd
from flashinfer.testing import bench_gpu_time

D = 128
# Each mode's entry points: the deterministic package's ``det`` names and the
# non-deterministic package's ``nondet`` names.
API = {
    "deterministic": {
        "forward": det.cake_rmsnorm_silu_gate_det_train_forward,
        "backward": det.cake_rmsnorm_silu_gate_det_train_backward,
        "workspace": det.cake_rmsnorm_silu_gate_det_train_backward_workspace,
        "workspace_bytes": det.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes,
        "autograd": det.cake_rmsnorm_silu_gate_det,
    },
    "nondeterministic": {
        "forward": nd.cake_rmsnorm_silu_gate_nondet_train_forward,
        "backward": nd.cake_rmsnorm_silu_gate_nondet_train_backward,
        "workspace": nd.cake_rmsnorm_silu_gate_nondet_train_backward_workspace,
        "workspace_bytes": nd.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes,
        "autograd": nd.cake_rmsnorm_silu_gate_nondet,
    },
}
MODES = tuple(API)
STAGES = ("fwd", "bwd", "ckpt")
# The issue's token counts and a spread around them, then small and irregular ones.
# The BF16 backward each reaches at 48 heads (``cake_jit.BACKWARD_BANDS``):
# - 148-SM B200 (sm_100a): 1 and 5 tokens the short backward; 7 the exact tiny
#   backward; 16 to 2048 the fused backward; 4097 the repeatable backward;
#   30000 and more its long program.
# - 152-SM GB300 (sm_103a): 1 and 5 tokens the short backward; 7 to 2048 the
#   fused backward; 4097 the repeatable backward; 30000 and more its long
#   program.
# - 212-SM R200 (sm_107a): 1 and 5 tokens the short backward; 7 the exact tiny
#   backward; 16 and 129 the repeatable backward; 1031 the fused backward;
#   2048 and 4097 the repeatable backward; 30000 and more its long program.
TOKENS = (30193, 32737, 32388, 30000, 33000, 1, 5, 7, 16, 129, 1031, 2048, 4097)


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
    for name in MODES:
        api = API[name]
        y = torch.empty_like(x)
        dx, dz = torch.empty_like(x), torch.empty_like(x)
        dw = torch.empty(D, dtype=torch.bfloat16, device=x.device)
        ws = api["workspace"](rows, x.device)

        def fwd(y=y, api=api):
            return api["forward"](x, z, w, eps, out=y)

        def bwd(dx=dx, dz=dz, dw=dw, ws=ws, api=api):
            return api["backward"](
                dy,
                x,
                z,
                w,
                eps,
                grad_x=dx,
                grad_z=dz,
                grad_weight=dw,
                workspace=ws,
            )

        def bwd_no_workspace(dx=dx, dz=dz, dw=dw, api=api):
            return api["backward"](
                dy,
                x,
                z,
                w,
                eps,
                grad_x=dx,
                grad_z=dz,
                grad_weight=dw,
            )

        def layer(xa, za, wa, api=api):
            return api["autograd"](xa, za, wa, eps)

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
            "workspace_bytes": api["workspace_bytes"](rows, x.device),
            "dw_patterns": len(patterns),
            "repeats": repeats,
        }
        result["modes"][name] = entry
        del ws
        torch.cuda.empty_cache()
    # The non-deterministic outputs against the deterministic ones.
    reference, candidate = outputs["deterministic"], outputs["nondeterministic"]
    result["y_bitwise_equal"] = bool(torch.equal(reference[0], candidate[0]))
    for i, grad in enumerate(("dx", "dz", "dw"), 1):
        result[f"{grad}_max_ulps_apart"] = int(
            (_ordered(reference[i]) - _ordered(candidate[i])).abs().max().item()
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
        for name in MODES:
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
    for name in MODES:
        m = result["modes"][name]
        lines.append(
            f"{head} mem  | {name:16s} saved {m['saved_bytes'] / 2**20:8.1f} MiB  "
            f"peak {m['peak_bytes'] / 2**20:8.1f} MiB  temp {m['temp_bytes'] / 2**10:9.1f} KiB  "
            f"workspace {m['workspace_bytes'] / 2**10:9.1f} KiB  "
            f"dw patterns {m['dw_patterns']}/{m['repeats']}"
        )
    lines.append(
        f"{head} out  | y bitwise equal: {result['y_bitwise_equal']}  "
        f"dx/dz/dw at most {result['dx_max_ulps_apart']}/"
        f"{result['dz_max_ulps_apart']}/{result['dw_max_ulps_apart']} BF16 ulp apart"
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
    if not nd.is_cake_rmsnorm_silu_gate_nondet_train_supported(
        device
    ) or not det.is_cake_rmsnorm_silu_gate_det_train_supported(device):
        raise SystemExit(
            "cake_rmsnorm_silu_gate_nondet_train or "
            "cake_rmsnorm_silu_gate_det_train has no exported route on this device"
        )
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
