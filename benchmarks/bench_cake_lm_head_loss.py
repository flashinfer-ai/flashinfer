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

Benchmark of the Cake chunked LM-head + loss training kernels in their two numerical modes (SM100 / SM103):
``deterministic=True`` (the deterministic package ``flashinfer.experimental.cake_lm_head_det_loss``) against
``deterministic=False`` (``flashinfer.experimental.cake.cake_lm_head_loss``: the dX GEMM's K-slice partials added
in an unordered FP32 reduction), both through the dispatching entries of ``flashinfer.chunked_lm_head``.

The rows are the deterministic benchmark's (``bench_cake_lm_head_det_loss.py``): the GLM-class geometry
(``H = 6144``, ``V = 154880``, BF16 ``X`` / ``W``, int64 labels with five percent ignored rows), the loss entry with
the cross-entropy and the policy objective and the log-probability entry at ``T`` = 16231, 16172, 4096, 32463 and
chunks of 2048 / 4096 / 8192 rows, plus the tail rows ``T`` = 1, 4095, 4097.  Per row and mode:

* ``fwd`` / ``bwd``: the package's explicit pair -- ``forward_loss`` / ``backward_loss`` (the loss forward runs the
  dX and dW GEMMs; the backward scales and casts, and runs the deferred last-chunk weight-gradient GEMM) or
  ``forward_logprob`` / ``backward_logprob`` (the recompute backward runs the GEMMs);
* ``step``: the public autograd entry, forward and backward;
* ``ckpt``: ``torch.utils.checkpoint(..., use_reentrant=False)`` around the public entry, then its backward -- the
  2 x forward + backward of activation checkpointing;
* ``*_us``: the GPU span of the phase's kernels (CUPTI activity, ``flashinfer.testing.bench_gpu_time``, cold L2);
* ``peak_temp_bytes`` / ``retained_bytes``: the allocator's peak above the bytes held before a ``step`` and the bytes
  a forward leaves alive for its backward.

The modes interleave: ``--rounds`` rounds, each timing every phase of both modes (det first in even rounds, nondet
first in odd ones); each value is the median over the rounds of the round's median, with its ``[min, max]`` over the
rounds.  ``speedup`` is ``det / nondet`` of the medians (above 1: the non-deterministic mode is faster),
``speedup_range`` the ``[min, max]`` of the per-round ratios.  A sanity record per row (not a gate): whether
``loss`` / ``dW`` are bitwise equal between the modes and the largest ``|dX_nondet - dX_det|``.

``--dx-slice-policy`` selects the non-deterministic package's dX slice rule (``cake_backend.set_dx_slice_policy``):
``det`` (the package default, the deterministic package's slice counts) or ``measured`` (the form-aware B200 rule).

``--facade`` adds the host cost of the dispatcher's default path on tiny log-probability rows (``T`` = 1, 128): the
median enqueue time of ``chunked_lm_head_logprob`` (default ``deterministic=True``) against calling
``chunked_lm_head_det_logprob`` directly, interleaved over rounds.

Example::

    python benchmarks/bench_cake_lm_head_loss.py --rows t16231_c4096_ce t16231_c4096_logprob --json out.json
    python benchmarks/bench_cake_lm_head_loss.py --facade --rounds 9
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer import chunked_lm_head as public  # noqa: E402
from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_lm_head_loss_reference import (  # noqa: E402
    DEFAULT_H,
    DEFAULT_V,
    make_inputs,
    make_weight,
)

# name -> (T, chunk, objective, entry): the deterministic benchmark's rows
ROWS: dict[str, tuple[int, int, str, str]] = {}
_SHAPES = (
    (16231, 4096),
    (16172, 4096),
    (16231, 2048),
    (16231, 8192),
    (4096, 4096),
    (32463, 4096),
)
for _T, _C in _SHAPES:
    ROWS[f"t{_T}_c{_C}_ce"] = (_T, _C, "ce", "loss")
for _T, _C in _SHAPES:
    ROWS[f"t{_T}_c{_C}_policy"] = (_T, _C, "policy", "loss")
for _T, _C in _SHAPES:
    ROWS[f"t{_T}_c{_C}_logprob"] = (_T, _C, "ce", "logprob")
for _T in (1, 4095, 4097):
    ROWS[f"t{_T}_c4096_ce"] = (_T, 4096, "ce", "loss")

MODES = {"det": True, "nondet": False}
PHASES = ("fwd", "bwd", "step", "ckpt")
SEED_BASE = 761
ROUNDS = 5
STEPS = 10
FACADE_ROWS = (1, 128)
FACADE_SAMPLES = 200


def _gpu_us(fn: Callable[[], object], steps: int) -> float:
    times = bench_gpu_time(
        fn, dry_run_iters=2, repeat_iters=steps, enable_cupti=True, cold_l2_cache=True
    )
    return 1e3 * float(statistics.median(times))


def _backends(deterministic: bool):
    """The package backend of a mode (the explicit forward / backward pair)."""
    if deterministic:
        from flashinfer.experimental.cake_lm_head_det_loss import cake_backend

        return cake_backend, {}
    from flashinfer.experimental.cake.cake_lm_head_loss import cake_backend

    return cake_backend, {"deterministic": False}


def _loss_kwargs(inp, chunk):
    return dict(
        objective=inp.objective,
        loss_div=inp.loss_div if inp.objective == "ce" else None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=chunk,
    )


def _phases(inp, chunk, entry, deterministic):
    """``phase -> callable`` of one row and mode."""
    be, mode = _backends(deterministic)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    if entry == "loss":
        kw = _loss_kwargs(inp, chunk)
        state = {"fwd": be.forward_loss(inp.X, inp.W, inp.labels, **kw, **mode)}

        def fwd():
            state["fwd"] = be.forward_loss(inp.X, inp.W, inp.labels, **kw, **mode)

        def bwd():
            return state["fwd"].backward()

        def call(x, w):
            return public.chunked_lm_head_loss(
                x, w, inp.labels, deterministic=deterministic, **kw
            )

        def step():
            call(X, W).backward()

        def ckpt():
            torch.utils.checkpoint.checkpoint(
                call, X, W, use_reentrant=False
            ).backward()

    else:
        fr = be.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=chunk, **mode)

        def fwd():
            be.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=chunk, **mode)

        def bwd():
            return be.backward_logprob(
                inp.X,
                inp.W,
                inp.labels,
                fr.lse,
                inp.dlogp,
                chunk_size=chunk,
                row_index=fr.row_index,
                **mode,
            )

        def call(x, w):
            return public.chunked_lm_head_logprob(
                x, w, inp.labels, chunk_size=chunk, deterministic=deterministic
            )

        def step():
            call(X, W).backward(inp.dlogp)

        def ckpt():
            torch.utils.checkpoint.checkpoint(call, X, W, use_reentrant=False).backward(
                inp.dlogp
            )

    def clear():
        X.grad = None
        W.grad = None

    phases = {"fwd": fwd, "bwd": bwd, "step": step, "ckpt": ckpt}
    return {name: (lambda f=f: (clear(), f())) for name, f in phases.items()}, (
        X,
        W,
        step,
    )


def _memory(inp, chunk, entry, deterministic) -> dict:
    """Peak bytes above the live tensors during one ``step`` and the bytes a forward keeps for its backward."""
    _, (X, W, step) = _phases(inp, chunk, entry, deterministic)
    step()  # warm: libraries, bindings, per-device state
    X.grad = W.grad = None
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    X.grad = W.grad = None
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    if entry == "loss":
        out = public.chunked_lm_head_loss(
            X, W, inp.labels, deterministic=deterministic, **_loss_kwargs(inp, chunk)
        )
    else:
        out = public.chunked_lm_head_logprob(
            X, W, inp.labels, chunk_size=chunk, deterministic=deterministic
        )
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - base - out.numel() * out.element_size()
    del out
    return dict(peak_temp_bytes=int(peak), retained_bytes=int(retained))


def _sanity(inp, chunk, entry) -> dict:
    """``loss`` / ``dW`` bit equality between the modes and the largest ``|dX_nondet - dX_det|`` (informational)."""
    out = {}
    for name, deterministic in MODES.items():
        _, (X, W, step) = _phases(inp, chunk, entry, deterministic)
        X.grad = W.grad = None
        if entry == "loss":
            loss = public.chunked_lm_head_loss(
                X,
                W,
                inp.labels,
                deterministic=deterministic,
                **_loss_kwargs(inp, chunk),
            )
            loss.backward()
        else:
            loss = public.chunked_lm_head_logprob(
                X, W, inp.labels, chunk_size=chunk, deterministic=deterministic
            )
            loss.backward(inp.dlogp)
        out[name] = (loss.detach().clone(), X.grad.clone(), W.grad.clone())
    (ld, xd, wd), (ln, xn, wn) = out["det"], out["nondet"]
    return dict(
        loss_bitwise=bool(torch.equal(ld.view(torch.int32), ln.view(torch.int32))),
        dW_bitwise=bool(torch.equal(wd.view(torch.int16), wn.view(torch.int16))),
        dX_bitwise=bool(torch.equal(xd.view(torch.int16), xn.view(torch.int16))),
        dX_max_abs_diff=float((xd.float() - xn.float()).abs().max())
        if xd.numel()
        else 0.0,
    )


def bench_row(name: str, W: torch.Tensor, args) -> dict:
    T, chunk, objective, entry = ROWS[name]
    inp = make_inputs(
        T,
        objective=objective,
        seed=SEED_BASE + T,
        device="cuda",
        W=W,
        ignore_frac=0.05 if T >= 16 else 0.0,
    )
    phases = {mode: _phases(inp, chunk, entry, det)[0] for mode, det in MODES.items()}
    samples = {mode: {phase: [] for phase in PHASES} for mode in MODES}
    for r in range(args.rounds):
        order = list(MODES) if r % 2 == 0 else list(MODES)[::-1]
        for phase in PHASES:
            for mode in order:
                samples[mode][phase].append(_gpu_us(phases[mode][phase], args.steps))
    result: dict = dict(row=name, T=T, chunk=chunk, objective=objective, entry=entry)
    for phase in PHASES:
        for mode in MODES:
            values = samples[mode][phase]
            result[f"{mode}_{phase}_us"] = statistics.median(values)
            result[f"{mode}_{phase}_us_range"] = [min(values), max(values)]
        ratios = [
            d / n
            for d, n in zip(
                samples["det"][phase], samples["nondet"][phase], strict=True
            )
        ]
        result[f"{phase}_speedup"] = (
            result[f"det_{phase}_us"] / result[f"nondet_{phase}_us"]
        )
        result[f"{phase}_speedup_range"] = [min(ratios), max(ratios)]
    for mode, det in MODES.items():
        for key, value in _memory(inp, chunk, entry, det).items():
            result[f"{mode}_{key}"] = value
    result["sanity"] = _sanity(inp, chunk, entry)
    return result


def bench_facade(W: torch.Tensor, args) -> list[dict]:
    """Median host enqueue time of the dispatcher's default path against the deterministic entry on tiny rows."""
    rows = []
    for T in FACADE_ROWS:
        inp = make_inputs(T, seed=SEED_BASE + T, device="cuda", W=W, ignore_frac=0.0)
        X = inp.X.detach().requires_grad_(True)
        calls = {
            "det": lambda: public.chunked_lm_head_det_logprob(X, W, inp.labels),
            "dispatch": lambda: public.chunked_lm_head_logprob(X, W, inp.labels),
        }
        per_round = {name: [] for name in calls}
        for fn in calls.values():
            for _ in range(10):
                fn()
        torch.cuda.synchronize()
        for r in range(args.rounds):
            order = list(calls) if r % 2 == 0 else list(calls)[::-1]
            for name in order:
                times = []
                for _ in range(FACADE_SAMPLES):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    calls[name]()
                    times.append(1e6 * (time.perf_counter() - t0))
                per_round[name].append(statistics.median(times))
        det, disp = per_round["det"], per_round["dispatch"]
        deltas = [b - a for a, b in zip(det, disp, strict=True)]
        rows.append(
            dict(
                T=T,
                det_host_us=statistics.median(det),
                dispatch_host_us=statistics.median(disp),
                delta_us=statistics.median(deltas),
                delta_us_range=[min(deltas), max(deltas)],
            )
        )
    return rows


def _format(result: dict) -> str:
    cells = [f"{result['row']:24s}"]
    for phase in PHASES:
        cells.append(
            f"{phase} {result[f'det_{phase}_us']:9.1f} / {result[f'nondet_{phase}_us']:9.1f} us "
            f"x{result[f'{phase}_speedup']:.3f} [{result[f'{phase}_speedup_range'][0]:.3f}, "
            f"{result[f'{phase}_speedup_range'][1]:.3f}]"
        )
    cells.append(
        f"peak {result['det_peak_temp_bytes'] / 2**20:.0f} / {result['nondet_peak_temp_bytes'] / 2**20:.0f} MiB"
    )
    s = result["sanity"]
    cells.append(
        f"loss/dW bitwise {s['loss_bitwise']}/{s['dW_bitwise']} dX max|diff| {s['dX_max_abs_diff']:.2e}"
    )
    return " | ".join(cells)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--rows", nargs="*", default=list(ROWS), choices=list(ROWS))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--facade", action="store_true", help="also time the dispatcher's default path"
    )
    parser.add_argument(
        "--dx-slice-policy",
        choices=("det", "measured"),
        default="det",
        help="the non-deterministic package's dX slice rule",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore", ExperimentalWarning)
    from flashinfer.experimental.cake.cake_lm_head_loss import (
        cake_backend as nd_backend,
    )

    nd_backend.set_dx_slice_policy(args.dx_slice_policy)
    torch.manual_seed(0)
    W = make_weight(DEFAULT_V, DEFAULT_H, seed=SEED_BASE + 1000003, device="cuda")
    props = torch.cuda.get_device_properties(0)
    report: dict = dict(
        env=dict(
            device=props.name,
            capability=list(torch.cuda.get_device_capability(0)),
            sms=props.multi_processor_count,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            dx_slice_policy=args.dx_slice_policy,
        ),
        rows=[],
    )
    print(
        f"{props.name} sm_{props.major}{props.minor} {props.multi_processor_count} SMs; det / nondet "
        f"(dX slice policy {args.dx_slice_policy})"
    )
    for name in args.rows:
        result = bench_row(name, W, args)
        report["rows"].append(result)
        print(_format(result), flush=True)
    if args.facade:
        report["facade"] = bench_facade(W, args)
        for row in report["facade"]:
            print(
                f"facade t{row['T']} logprob: det {row['det_host_us']:.2f} us, dispatch "
                f"{row['dispatch_host_us']:.2f} us, delta {row['delta_us']:+.2f} us "
                f"[{row['delta_us_range'][0]:+.2f}, {row['delta_us_range'][1]:+.2f}]"
            )
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
