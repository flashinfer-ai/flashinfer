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

Benchmark of the Cake chunked LM-head + loss training kernels in their two numerical modes (SM100 / SM103): the
deterministic module ``flashinfer.cake_lm_head_det_train`` (backend
``flashinfer.experimental.cake.cake_lm_head_det_train``) against the non-deterministic module
``flashinfer.cake_lm_head_nondet_train`` (backend ``flashinfer.experimental.cake.cake_lm_head_nondet_train``: the dX
GEMM's K-slice partials added in an unordered FP32 reduction), each through its own public entries.

The rows are the deterministic benchmark's (``bench_cake_lm_head_det_train.py``): the GLM-class geometry
(``H = 6144``, ``V = 154880``, BF16 ``X`` / ``W``, int64 labels with five percent ignored rows), the loss entry with
the cross-entropy and the policy objective and the log-probability entry at ``T`` = 16231, 16172, 4096, 32463 and
chunks of 2048 / 4096 / 8192 rows, plus the tail rows ``T`` = 1, 4095, 4097.  Per row and mode:

* ``fwd`` / ``bwd``: the package's explicit pair -- ``forward_loss`` / ``backward_loss`` (the loss forward runs the
  dX and dW GEMMs; the backward scales and casts, and runs the deferred last-chunk weight-gradient GEMM) or
  ``forward_logprob`` / ``backward_logprob`` (the recompute backward runs the GEMMs);
* ``step``: the mode's public autograd entry, forward and backward;
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

``--det-memory-mode`` selects the deterministic mode's memory form in its public entries (``step``, ``ckpt``, the
memory and sanity records): ``fast`` (the default; ``memory_mode="fast"``, the form of the non-deterministic mode's
default and of the explicit ``fwd`` / ``bwd`` pair) or ``auto`` (``memory_mode=None``, the deterministic default,
which switches a call of two or more chunks to a budget form, when one fits, where the fast form's predicted step peak
would exceed the logits-materializing head's bytes less a 2 % margin; a one-chunk call keeps the fast form).
The report records it (``env.det_memory_mode``).

Example::

    python benchmarks/bench_cake_lm_head_nondet_train.py --rows t16231_c4096_ce t16231_c4096_logprob --json out.json
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer import cake_lm_head_det_train as det_public  # noqa: E402
from flashinfer import cake_lm_head_nondet_train as public  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.experimental.test_helpers.cake_lm_head_loss_reference import (  # noqa: E402
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
# ``--det-memory-mode`` -> the deterministic public entries' ``memory_mode``
DET_MEMORY_MODES = {"fast": "fast", "auto": None}
_DET_MEMORY_MODE: dict[str, Optional[str]] = {"memory_mode": "fast"}
SEED_BASE = 761
ROUNDS = 5
STEPS = 10


def _gpu_us(fn: Callable[[], object], steps: int) -> float:
    times = bench_gpu_time(
        fn, dry_run_iters=2, repeat_iters=steps, enable_cupti=True, cold_l2_cache=True
    )
    return 1e3 * float(statistics.median(times))


def _backend(deterministic: bool):
    """The backend of a mode (the explicit forward / backward pair)."""
    if deterministic:
        from flashinfer.experimental.cake.cake_lm_head_det_train import cake_backend

        return cake_backend
    from flashinfer.experimental.cake.cake_lm_head_nondet_train import cake_backend

    return cake_backend


def _entries(deterministic: bool):
    """The public ``(loss, logprob)`` entries of a mode (the deterministic ones with ``--det-memory-mode``'s
    ``memory_mode``)."""
    if deterministic:
        memory = dict(memory_mode=_DET_MEMORY_MODE["memory_mode"])
        return (
            functools.partial(det_public.cake_lm_head_det_train_loss, **memory),
            functools.partial(det_public.cake_lm_head_det_train_logprob, **memory),
        )
    return (
        public.cake_lm_head_nondet_train_loss,
        public.cake_lm_head_nondet_train_logprob,
    )


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
    be = _backend(deterministic)
    loss_entry, logprob_entry = _entries(deterministic)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    if entry == "loss":
        kw = _loss_kwargs(inp, chunk)
        state = {"fwd": be.forward_loss(inp.X, inp.W, inp.labels, **kw)}

        def fwd():
            state["fwd"] = be.forward_loss(inp.X, inp.W, inp.labels, **kw)

        def bwd():
            return state["fwd"].backward()

        def call(x, w):
            return loss_entry(x, w, inp.labels, **kw)

        def step():
            call(X, W).backward()

        def ckpt():
            torch.utils.checkpoint.checkpoint(
                call, X, W, use_reentrant=False
            ).backward()

    else:
        fr = be.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=chunk)

        def fwd():
            be.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=chunk)

        def bwd():
            return be.backward_logprob(
                inp.X,
                inp.W,
                inp.labels,
                fr.lse,
                inp.dlogp,
                chunk_size=chunk,
                row_index=fr.row_index,
            )

        def call(x, w):
            return logprob_entry(x, w, inp.labels, chunk_size=chunk)

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
    loss_entry, logprob_entry = _entries(deterministic)
    if entry == "loss":
        out = loss_entry(X, W, inp.labels, **_loss_kwargs(inp, chunk))
    else:
        out = logprob_entry(X, W, inp.labels, chunk_size=chunk)
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
        loss_entry, logprob_entry = _entries(deterministic)
        if entry == "loss":
            loss = loss_entry(X, W, inp.labels, **_loss_kwargs(inp, chunk))
            loss.backward()
        else:
            loss = logprob_entry(X, W, inp.labels, chunk_size=chunk)
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
        "--dx-slice-policy",
        choices=("det", "measured"),
        default="det",
        help="the non-deterministic package's dX slice rule",
    )
    parser.add_argument(
        "--det-memory-mode",
        choices=tuple(DET_MEMORY_MODES),
        default="fast",
        help="the deterministic public entries' memory form: fast, or auto (the deterministic default)",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    _DET_MEMORY_MODE["memory_mode"] = DET_MEMORY_MODES[args.det_memory_mode]
    from flashinfer.experimental.cake.cake_lm_head_nondet_train import (
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
            det_memory_mode=args.det_memory_mode,
        ),
        rows=[],
    )
    print(
        f"{props.name} sm_{props.major}{props.minor} {props.multi_processor_count} SMs; det / nondet "
        f"(dX slice policy {args.dx_slice_policy}, det memory mode {args.det_memory_mode})"
    )
    for name in args.rows:
        result = bench_row(name, W, args)
        report["rows"].append(result)
        print(_format(result), flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
