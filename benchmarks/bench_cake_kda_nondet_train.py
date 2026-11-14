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

Benchmark of the Cake packed KDA chunked training kernels in both numerical
modes: the non-deterministic package
(``flashinfer.experimental.cake.cake_kda_nondet_train``)
against the deterministic package
(``flashinfer.experimental.cake.cake_kda_det_train``) and flash-linear-attention's
``chunk_kda`` (when it is installed), on the same inputs, on the boundaries of
a training step:

* ``fwd``: the forward with the saved set written;
* ``bwd``: the backward from the saved set;
* ``ckpt``: ``torch.utils.checkpoint`` forward + replay + backward (one span:
  two forwards and the backward);
* ``replay`` (optional, Cake arms only): the backward without a saved set
  (``saved=None``), which recomputes it.

Every row is one packed batch of ``--docs`` documents of ``--seqlen`` tokens
in total (unequal lengths with unaligned tails), or one ``--lengths`` packing
(explicit comma-separated document lengths), and ``--heads`` heads; each
Cake arm allocates its workspace once per row and reuses it, so the timed
closures allocate nothing.  Timing uses CUPTI kernel activity with a cold L2
(``flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)``):
the median over the timed calls of the summed GPU time of a boundary's
kernels, in milliseconds.  The arms run interleaved: ``--rounds`` rounds, each
timing every arm and boundary once, the arm order reversed on every other
round; a row reports the median of the round medians of each arm and boundary
with their spread (min, max), the ratios ``det / nd`` and ``fla / nd`` of those
medians (above 1: the non-deterministic arm is faster), the records each Cake
arm launches, and the non-deterministic results against the deterministic ones
(``out``, ``dq``, ``dk``, ``dv`` and ``dbeta`` are bitwise; ``dg`` differs by FP32
rounding on the rows whose backward runs the S1 pair).  The JSON records the
device, architecture, SM count, torch, CUDA and flash-linear-attention versions
(the distribution's and ``fla.__version__``).

``det2`` is the deterministic arm again, the row's A/A arm (as the
context-parallel benchmark's ``control``): the same programs and calls on the
deterministic arm's workspace, built after the two Cake arms (its saved set is
its own), timed in every round in the order ``det``, ``nd``, ``det2``, ``fla``,
reversed on every other round.  Per boundary the row reports ``det2 / det``
(``ratio_det2_over_det``), the A/A band ``[1 - w, 1 + w]`` (``aa_band``; ``w``
is the largest ``|det2 / det - 1|`` over the rounds' medians and the row's
medians) and whether ``det / nd`` lies outside it (``det_over_nd_beyond_aa``):
a ``det / nd`` difference inside the band is within the protocol's own noise.
With ``--build-order both`` the row's band covers both orders' bands and the
``det2 / det`` of the means.  With ``--arms-alone`` the arm runs alone like the
others, on a workspace of its own.

``--gate`` selects the forget gate's form per row: ``g`` (the default) the
activated FP32 gate the caller passes, ``raw`` the raw gate mode
(``use_gate_in_kernel=True``: the BF16 raw gate, FP32 ``A_log`` /
``dt_bias``, lower bound -5, activated in the kernels).  In the raw mode the
three arms run their raw forms on the same operands: the deterministic
package's and this package's raw mode (both workspaces of ``gate="raw"``;
the ``ckpt`` leaves include ``A_log`` and ``dt_bias``) and
flash-linear-attention's ``chunk_kda(use_gate_in_kernel=True, ...)``, whose
``out`` the row compares with the non-deterministic arm's
(``fla_raw_out_vs_nd``: the relative RMS difference; an adapter that drops a
gate term shows here).  This package's raw mode runs the deterministic
package's programs in the same order on every architecture (a tape backward
may run this package's w-only WY twin in place of the WY program, writing the
same ``w``), so every result (``dg`` of the raw gate, ``dA_log`` and
``ddt_bias`` included) is bitwise the deterministic one and no path runs an S1
pair.

``--build-order`` sets which Cake arm allocates first: its workspace, its first
results comparison call (where its first-use buffers are made), its saved set
and its closures.  The arena placement this decides moves a few kernels by
microseconds, so a det / nd ratio measured in one order carries a placement
bias.  ``det-first`` (the default)
is the order of the earlier runs; ``nd-first`` swaps it; ``both`` runs every
row in both orders, prints both, and reports per arm and boundary the mean of
the two orders' medians (``order_avg``), the ratios of those means
(``ratio_det_over_nd``), each order's ratio (``ratio_det_over_nd_by_order``)
and the boundaries where nd is slower in both orders
(``nd_slower_both_orders``).

``--l2n`` runs every row with the in-kernel Q/K L2 norm
(``use_qk_l2norm_in_kernel=True``), in each ``--gate`` mode: ``q`` / ``k``
are raw BF16 rows of a few magnitudes per token (drawn as the deterministic
package's in-kernel-norm tests draw them, :func:`raw_qk`; the other operands
are those of the row's gate mode), the Cake arms' workspaces are of that mode
(``l2n="fused"``) and their calls take the keyword, and flash-linear-attention's
arm runs ``chunk_kda(use_qk_l2norm_in_kernel=True, ...)`` on the same raw rows
in its layout: every arm's norm and the norm's gradient are inside its
boundaries.  The row compares flash-linear-attention's ``out`` with the
non-deterministic arm's (``fla_l2n_out_vs_nd``: the relative RMS difference;
an adapter that skips the norm shows here).  This package's in-kernel-norm mode
runs the deterministic package's programs in the same order (a tape backward
may run this package's w-only WY twin in place of the WY program, writing the
same ``w``), so every result (``dg`` included) is bitwise the deterministic one
and no path runs an S1 pair.

Memory: per row, arm and boundary one untimed call after the warm-up, between
``torch.cuda.reset_peak_memory_stats()`` and the allocator's counters, gives
``peak_bytes`` (the peak allocation of the call beyond what was allocated
before it) and ``retained_bytes`` (what is still allocated after it while its
results are held: after ``fwd`` the output and the saved set -- for FLA its
autograd graph --, after the other boundaries the gradients).  A Cake arm's
caller-owned workspace (``workspace_bytes``) is allocated before and counts in
neither.

The flash-linear-attention arm fails soft: an exception from its first call,
its warm-up, its memory call or its timing records ``fla_error`` (the exception
type and message) in the row, drops the arm, and the Cake arms go on (an error
that corrupts the CUDA context still stops the run).

``--side-stream`` adds, per row, Cake arm and boundary, the wall latency of the
call on its stream (CUDA events; the median of seven calls after three
warm-ups) alone and with a one-thread kernel (``torch.cuda._sleep``) resident
on another stream, alternating, and their ratio (the concurrent-launch
measurement of FlashInfer issue #7038, after the merged causal conv1d and
RMSNorm SiLU-gate benchmarks).  The side kernel is launched first, the call
after a short sleep on its own stream, and it spins for at least twice the
call's warm-up latency (``side_covered``: it was still running when every
call beside it ended).  Information, not a gate.

Example::

    python benchmarks/bench_cake_kda_nondet_train.py --seqlen 16384 65536 --docs 3 --heads 4 64
    python benchmarks/bench_cake_kda_nondet_train.py --seqlen 261393 --docs 4 --heads 4 --rounds 5 --json out.json
    python benchmarks/bench_cake_kda_nondet_train.py --seqlen 65536 --docs 3 --heads 16 --side-stream
    python benchmarks/bench_cake_kda_nondet_train.py --seqlen 16384 65536 --docs 3 --heads 4 64 --gate g raw
    python benchmarks/bench_cake_kda_nondet_train.py --seqlen 16384 --docs 1 3 --heads 4 64 --gate raw --build-order both
    python benchmarks/bench_cake_kda_nondet_train.py --seqlen 65536 --docs 3 --heads 4 64 --gate g raw --l2n --build-order both
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.cake.cake_kda_nondet_train import (  # noqa: E402
    cake_backend as nd_backend,
)
from flashinfer.experimental.cake.cake_kda_nondet_train import cake_jit  # noqa: E402
from flashinfer.experimental.cake.cake_kda_det_train import (  # noqa: E402
    cake_backend as det_backend,
)
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.experimental.test_helpers.cake_kda_det_train_reference import (  # noqa: E402
    D,
    LOWER_BOUND,
    make_inputs,
    make_raw_inputs,
)

BOUNDARIES = ("fwd", "bwd", "ckpt", "replay")
DEFAULT_BOUNDARIES = ("fwd", "bwd", "ckpt")
# the interleaved round order (reversed on every other round)
ARMS = ("det", "nd", "det2", "fla")
AA_ARM = "det2"  # the deterministic arm again: the A/A arm of every row
GRAD_ORDER = ("dq", "dk", "dv", "dbeta", "dg")  # the order the backward returns
RAW_GRAD_ORDER = GRAD_ORDER + ("dA_log", "ddt_bias")  # the raw gate mode's
GATES = ("g", "raw")
# --build-order: which Cake arm allocates its workspace, saved set and closures first ("both": every row in each)
ORDERS = ("det-first", "nd-first")
BUILD_ORDERS = ORDERS + ("both",)
# the Cake arms of one build order, the first built first
CAKE_ARMS = {"det-first": ("det", "nd"), "nd-first": ("nd", "det")}
# the key of a row's operands that marks the in-kernel Q/K L2 norm (raw q / k rows; see make_row_inputs)
L2N_KEY = "use_qk_l2norm_in_kernel"
# ``--side-stream``: a one-thread kernel resident on another stream while a Cake call runs.  It spins for at least
# SIDE_MIN_CYCLES (about 5 ms at 2 GHz) and SIDE_CYCLES_PER_MS per ms of the call's slowest warm-up (twice the call at
# 2 GHz); the call starts after PRE_SLEEP_CYCLES on its own stream, so the side kernel is resident first.
SIDE_MIN_CYCLES = 10_000_000
SIDE_CYCLES_PER_MS = 4_000_000
PRE_SLEEP_CYCLES = 1_000_000
SIDE_WARMUPS = 3
SIDE_SAMPLES = 7


def split_docs(seqlen: int, docs: int) -> tuple[int, ...]:
    """``docs`` unequal document lengths summing to ``seqlen`` whose tails are not chunk-aligned."""
    if docs < 1 or seqlen < docs:
        raise ValueError(f"cannot split {seqlen} tokens into {docs} documents")
    weights = [1.0 + 0.37 * i for i in range(docs)]
    total = sum(weights)
    lengths = [max(1, int(seqlen * w / total)) for w in weights]
    lengths[-1] += seqlen - sum(lengths)
    for i in range(docs - 1):
        if lengths[i] % 64 == 0 and lengths[i + 1] > 1:
            lengths[i] += 1
            lengths[i + 1] -= 1
    return tuple(lengths)


def _median_ms(samples) -> float:
    return float(statistics.median(list(samples)))


def _time(fn: Callable[[], object]) -> float:
    return _median_ms(bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True))


def _timer() -> str:
    """What ``bench_gpu_time(enable_cupti=True)`` measures here: CUPTI kernel time
    when ``cupti-python`` >= 13 imports, else its fallback, CUDA-event time."""
    try:
        from importlib.metadata import version

        from cupti import cupti  # noqa: F401

        cupti_version = version("cupti-python")
    except Exception:
        return "CUDA events (cupti-python unavailable)"
    if int(cupti_version.split(".")[0]) < 13:
        return f"CUDA events (cupti-python {cupti_version} < 13)"
    return f"CUPTI kernel time (cupti-python {cupti_version})"


def _fla_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("flash-linear-attention")
    except Exception:
        try:
            import fla

            return getattr(fla, "__version__", "unknown")
        except Exception:
            return None


def _fla_module_version() -> str | None:
    """``fla.__version__`` when the module imports (``None`` otherwise)."""
    try:
        import fla
    except Exception:
        return None
    return str(getattr(fla, "__version__", "unknown"))


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _gate_of(inputs: dict) -> str:
    """The gate mode of a row's operands: ``"raw"`` for :func:`make_raw_inputs`' (with ``A_log``)."""
    return "raw" if "A_log" in inputs else "g"


def _raw_kw(inputs: dict, params=None) -> dict:
    """The raw gate mode's keywords of a Cake call on ``inputs`` (``params``: the ``(A_log, dt_bias)`` leaves of an
    autograd call), empty in the explicit gate mode."""
    if _gate_of(inputs) == "g":
        return {}
    a_log, dt_bias = (inputs["A_log"], inputs["dt_bias"]) if params is None else params
    return {
        "use_gate_in_kernel": True,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": LOWER_BOUND,
    }


def _leaf_names(inputs: dict) -> tuple[str, ...]:
    """The autograd leaves of a row: ``q, k, v, g, beta`` and in the raw gate mode ``A_log, dt_bias``."""
    names = ("q", "k", "v", "g", "beta")
    return names + (("A_log", "dt_bias") if _gate_of(inputs) == "raw" else ())


def raw_qk(S: int, H: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (not normalized) BF16 ``q`` / ``k`` ``[S, H, 128]`` on the host, of a few magnitudes per token (the
    caller's projections), drawn as the deterministic package's in-kernel-norm tests draw them."""
    gen = torch.Generator().manual_seed(7817 + seed + 31 * S + H)
    mag = torch.exp(torch.randn(S, H, 1, generator=gen) * 1.5)
    q = (torch.randn(S, H, D, generator=gen) * mag).to(torch.bfloat16)
    k = (torch.randn(S, H, D, generator=gen) * mag.flip(0)).to(torch.bfloat16)
    return q, k


def make_row_inputs(
    lengths: tuple[int, ...],
    heads: int,
    device: torch.device,
    *,
    gate: str = "g",
    l2n: bool = False,
) -> dict:
    """The operands of a row: :func:`make_inputs` (``gate="g"``) or :func:`make_raw_inputs` (``"raw"``), and with
    ``l2n`` the raw ``q`` / ``k`` rows of :func:`raw_qk` in place of the normalized ones plus the :data:`L2N_KEY`
    mark (the in-kernel Q/K L2 norm of every arm on these operands)."""
    inputs = {"g": make_inputs, "raw": make_raw_inputs}[gate](lengths, heads, device)
    if l2n:
        q, k = raw_qk(int(sum(lengths)), heads)
        inputs["q"], inputs["k"] = q.to(device), k.to(device)
        inputs[L2N_KEY] = True
    return inputs


def _l2n_of(inputs: dict) -> bool:
    """Whether a row's operands are for the in-kernel Q/K L2 norm (raw ``q`` / ``k``; :func:`make_row_inputs`)."""
    return bool(inputs.get(L2N_KEY, False))


def _l2n_kw(inputs: dict) -> dict:
    """The in-kernel Q/K L2 norm keyword of a Cake call on ``inputs``, empty without it (the call as it always
    was)."""
    return {L2N_KEY: True} if _l2n_of(inputs) else {}


def cake_workspace(
    seqlen: int,
    docs: int,
    heads: int,
    device: torch.device,
    *,
    gate: str = "g",
    l2n: bool = False,
    det: bool = False,
):
    """A loaded workspace of one Cake arm (``det``: the deterministic package's, else this package's) for the row's
    gate mode and in-kernel Q/K L2 norm mode."""
    mode = {"gate": gate, "l2n": "fused" if l2n else "off"}
    if det:
        ws = det_backend.CakeKDADetWorkspace.for_problem(
            seqlen, docs, heads, device, **mode
        )
    else:
        ws = nd_backend.CakeKDANondetWorkspace.for_problem(
            seqlen, docs, heads, device, **mode
        )
    ws.load()
    return ws


def cake_boundaries(
    inputs: dict, ws, *, det: bool = False
) -> dict[str, Callable[[], object]]:
    """The boundaries of one Cake arm on its workspace (``det``: the deterministic package's entries, else this
    package's), in the gate mode and in-kernel Q/K L2 norm mode of ``inputs``.  Each closure returns its results
    (the memory call holds them)."""
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    if det:
        fwd_fn = det_backend.cake_kda_det_train_forward
        bwd_fn = det_backend.cake_kda_det_train_backward
        autograd_fn = det_backend.cake_kda_det
    else:
        fwd_fn = nd_backend.cake_kda_nondet_train_forward
        bwd_fn = nd_backend.cake_kda_nondet_train_backward
        autograd_fn = nd_backend.cake_kda_nondet
    raw = _raw_kw(inputs)
    l2n = _l2n_kw(inputs)

    def fwd() -> object:
        return fwd_fn(
            q,
            k,
            v,
            g,
            beta,
            cu,
            save_for_backward=True,
            workspace=ws,
            **raw,
            **l2n,
        )

    _, saved = fwd_fn(
        q,
        k,
        v,
        g,
        beta,
        cu,
        save_for_backward=True,
        workspace=ws,
        **raw,
        **l2n,
    )

    def bwd() -> object:
        return bwd_fn(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, **raw, **l2n
        )

    def replay() -> object:
        return bwd_fn(
            q, k, v, g, beta, cu, dout, saved=None, workspace=ws, **raw, **l2n
        )

    leaves = [
        inputs[n].detach().clone().requires_grad_(True) for n in _leaf_names(inputs)
    ]

    def step(*a):
        # the raw mode's A_log / dt_bias are leaves too (gradients to them)
        return autograd_fn(
            *a[:5], cu, workspace=ws, **_raw_kw(inputs, a[5:] or None), **l2n
        )

    def ckpt() -> object:
        o = checkpoint(step, *leaves, use_reentrant=False)
        return torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt, "replay": replay}


def fla_boundaries(inputs: dict) -> dict[str, Callable[[], object]] | None:
    """flash-linear-attention's ``chunk_kda`` on the same operands (no beta sigmoid; the preactivated gate, or in
    the raw gate mode its in-kernel gate with ``A_log``, ``dt_bias`` and the lower bound; its in-kernel L2 norm of
    the raw ``q`` / ``k`` rows on the in-kernel-norm operands of :func:`make_row_inputs`, else none), or ``None``
    when the package is not installed.  The first call runs here: an exception from it propagates to the caller
    (:func:`run_row` records it as ``fla_error``)."""
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda
    except Exception:
        return None
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )

    raw_mode = _gate_of(inputs) == "raw"
    l2n = _l2n_of(inputs)

    def call(q_, k_, v_, g_, beta_, *params):
        gate = (
            {
                "use_gate_in_kernel": True,
                "A_log": params[0],
                "dt_bias": params[1],
                "lower_bound": LOWER_BOUND,
            }
            if raw_mode
            else {"use_gate_in_kernel": False}
        )
        o, _ = fla_chunk_kda(
            q_.unsqueeze(0),
            k_.unsqueeze(0),
            v_.unsqueeze(0),
            g_.unsqueeze(0),
            beta_.unsqueeze(0),
            scale=nd_backend.SCALE,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=l2n,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=True,
            cu_seqlens=cu,
            **gate,
        )
        return o.squeeze(0)

    leaves = [
        inputs[n].detach().clone().requires_grad_(True) for n in _leaf_names(inputs)
    ]

    def fwd() -> object:
        return call(*leaves)

    retained = call(*leaves)

    def bwd() -> object:
        return torch.autograd.grad(retained, leaves, dout, retain_graph=True)

    def ckpt() -> object:
        o = checkpoint(call, *leaves, use_reentrant=False)
        return torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt}


def _results(inputs: dict, ws, *, det: bool, replay: bool) -> dict:
    """``out`` and the gradients of one Cake arm (``det``: the deterministic package's, else this package's; from
    the saved set, or the replay's), in the gate mode and in-kernel Q/K L2 norm mode of ``inputs``."""
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    raw = _raw_kw(inputs)
    l2n = _l2n_kw(inputs)
    if det:
        out, saved = det_backend.cake_kda_det_train_forward(
            q, k, v, g, beta, cu, workspace=ws, **raw, **l2n
        )
        grads = det_backend.cake_kda_det_train_backward(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=None if replay else saved,
            workspace=ws,
            **raw,
            **l2n,
        )
    else:
        out, saved = nd_backend.cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, **raw, **l2n
        )
        grads = nd_backend.cake_kda_nondet_train_backward(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=None if replay else saved,
            workspace=ws,
            **raw,
            **l2n,
        )
    order = RAW_GRAD_ORDER if raw else GRAD_ORDER
    return {
        "out": out.clone(),
        **{n: t.clone() for n, t in zip(order, grads, strict=True)},
    }


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _nd_vs_det(inputs: dict, det_ws, nd_ws, build_order: str = "det-first") -> dict:
    """The non-deterministic results against the deterministic ones on the same inputs, from the saved set and from
    the replay: the outputs that are bitwise (in the raw gate mode every one, ``dA_log`` and ``ddt_bias`` included;
    in the in-kernel Q/K L2 norm mode every one, ``dg`` included), and ``dg``'s differing elements and largest
    difference.  The arms run in ``build_order`` (:data:`CAKE_ARMS`): the first call on a workspace makes its
    first-use buffers."""
    report = {}
    for label, replay in (("saved", False), ("replay", True)):
        results = {
            arm: _results(
                inputs,
                det_ws if arm == "det" else nd_ws,
                det=arm == "det",
                replay=replay,
            )
            for arm in CAKE_ARMS[build_order]
        }
        det, nd = results["det"], results["nd"]
        diff = (nd["dg"].double() - det["dg"].double()).abs()
        report[label] = {
            "bitwise": [name for name in det if _bits_equal(nd[name], det[name])],
            "not_bitwise": [
                name for name in det if not _bits_equal(nd[name], det[name])
            ],
            "dg_differing_elements": int((diff > 0).sum().item()),
            "dg_max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
            "dg_max_abs": float(det["dg"].abs().max().item()) if diff.numel() else 0.0,
        }
    return report


def _nd_vs_det_or_error(
    inputs: dict, det_ws, nd_ws, build_order: str = "det-first"
) -> dict:
    """:func:`_nd_vs_det`, or its error when the two arms' results do not fit in device memory beside the inputs."""
    try:
        return _nd_vs_det(inputs, det_ws, nd_ws, build_order)
    except torch.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": _error(exc)}


def _s1_paths(ws) -> list[str]:
    """The backward paths of a workspace whose fused gradient runs an S1 record."""
    return [
        path
        for path, by_stage in ws.records().items()
        if by_stage.get("dqkg") in cake_jit.S1_RECORDS
    ]


def _memory(fn: Callable[[], object], device: torch.device) -> dict[str, int]:
    """``peak_bytes`` / ``retained_bytes`` of one untimed call of ``fn``: the peak allocation beyond what was
    allocated before the call, and what stays allocated after it while its results are held."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    results = fn()
    torch.cuda.synchronize(device)
    report = {
        "peak_bytes": torch.cuda.max_memory_allocated(device) - base,
        "retained_bytes": torch.cuda.memory_allocated(device) - base,
    }
    del results
    return report


def _latency_beside(fn: Callable[[], object], device: torch.device) -> dict:
    """The wall latency of ``fn`` on the current stream (CUDA events; median of :data:`SIDE_SAMPLES` calls after
    :data:`SIDE_WARMUPS`) alone and with a one-thread kernel resident on another stream, alternating.  The side
    kernel is launched first and spins for at least twice the slowest warm-up (at 2 GHz); ``side_covered`` tells
    whether it was still running when every call beside it ended."""
    side = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    side_end = torch.cuda.Event(enable_timing=True)

    def timed(cycles: int) -> tuple[float, bool]:
        torch.cuda.synchronize(device)
        if cycles:
            with torch.cuda.stream(side):
                torch.cuda._sleep(cycles)
            side_end.record(side)
        torch.cuda._sleep(
            PRE_SLEEP_CYCLES
        )  # the side kernel becomes resident before fn's launch
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)
        return ms, not cycles or start.elapsed_time(side_end) >= ms

    warm = [timed(0)[0] for _ in range(SIDE_WARMUPS)]
    cycles = max(SIDE_MIN_CYCLES, math.ceil(max(warm) * SIDE_CYCLES_PER_MS))
    alone, beside, covered = [], [], True
    for _ in range(SIDE_SAMPLES):
        alone.append(timed(0)[0])
        ms, ok = timed(cycles)
        beside.append(ms)
        covered = covered and ok
    alone_ms, beside_ms = _median_ms(alone), _median_ms(beside)
    return {
        "alone_ms": alone_ms,
        "beside_ms": beside_ms,
        "ratio_beside_over_alone": beside_ms / alone_ms,
        "side_cycles": cycles,
        "side_covered": covered,
    }


def run_row(
    seqlen: int,
    docs: int,
    heads: int,
    *,
    device: torch.device,
    baseline: bool,
    boundaries: tuple[str, ...],
    rounds: int,
    side_stream: bool = False,
    lengths: tuple[int, ...] | None = None,
    fla_alone: bool = False,
    arms_alone: bool = False,
    gate: str = "g",
    l2n: bool = False,
    build_order: str = "det-first",
) -> dict:
    if build_order not in ORDERS:
        raise ValueError(f"build_order must be one of {ORDERS}, got {build_order!r}")
    lengths = split_docs(seqlen, docs) if lengths is None else lengths
    inputs = make_row_inputs(lengths, heads, device, gate=gate, l2n=l2n)

    def det_workspace():
        return cake_workspace(seqlen, docs, heads, device, gate=gate, l2n=l2n, det=True)

    def nd_workspace():
        return cake_workspace(seqlen, docs, heads, device, gate=gate, l2n=l2n)

    if build_order == "det-first":
        det_ws, nd_ws = det_workspace(), nd_workspace()
    else:
        nd_ws, det_ws = nd_workspace(), det_workspace()
    row = {
        "seqlen": seqlen,
        "docs": docs,
        "lengths": list(lengths),
        "heads": heads,
        "gate": gate,
        "l2n": l2n,
        "build_order": build_order,
        "arch": nd_ws.arch,
        "recurrence": nd_ws.plan.recur,
        "segments_per_doc": nd_ws.plan.P,
        "records": {"det": det_ws.records(), "nd": nd_ws.records()},
        "s1_paths": _s1_paths(nd_ws),
        "workspace_bytes": {"det": det_ws.bytes, "nd": nd_ws.bytes},
        "saved_bytes": nd_backend.saved_bytes(
            seqlen,
            heads,
            docs,
            sm_count=nd_ws.plan.sm_count,
            l2n="fused" if l2n else "off",
        ),
        "nd_vs_det": {"skipped": "--arms-alone"}
        if arms_alone
        else _nd_vs_det_or_error(inputs, det_ws, nd_ws, build_order),
        "rounds": rounds,
        "arm_order": [],
        "ms_rounds": {},
        "ms": {},
        "ms_min": {},
        "ms_max": {},
        "ratio_det_over_nd": {},
        "ratio_fla_over_nd": {},
        "ratio_det2_over_det": {},
        "aa_band": {},
        "det_over_nd_beyond_aa": {},
        "memory": {},
    }
    # the closures (and the saved sets they hold) in the build order too
    arms: dict[str, dict[str, Callable[[], object]]] = {}
    if not arms_alone:
        for arm in ("det", "nd") if build_order == "det-first" else ("nd", "det"):
            arms[arm] = cake_boundaries(
                inputs, det_ws if arm == "det" else nd_ws, det=arm == "det"
            )
        # the A/A arm after the two Cake arms: the deterministic calls again on the deterministic workspace
        arms[AA_ARM] = cake_boundaries(inputs, det_ws, det=True)
    samples: dict[str, dict[str, list[float]]] = {}

    def fla_failed(exc: Exception) -> None:
        # the FLA arm fails soft: its error goes to the row, the arm (and its samples) away, the Cake arms go on
        row["fla_error"] = _error(exc)
        arms.pop("fla", None)
        samples.pop("fla", None)
        row["memory"].pop("fla", None)
        torch.cuda.synchronize(device)

    if baseline and not fla_alone and not arms_alone:
        try:
            fla = fla_boundaries(inputs)
        except Exception as exc:
            fla = None
            fla_failed(exc)
        if fla is not None:
            arms["fla"] = fla
    if (gate == "raw" or l2n) and "fla" in arms:
        # flash-linear-attention's raw or in-kernel-norm form against this package's out on the same operands
        key = "fla_l2n_out_vs_nd" if l2n else "fla_raw_out_vs_nd"
        try:
            row[key] = _fla_out_vs_nd(inputs, arms["fla"], nd_ws)
        except Exception as exc:
            fla_failed(exc)
    # warm every closure once (libraries loaded, the allocator primed) before the first timed round
    for arm in list(arms):
        try:
            for name in boundaries:
                if name in arms[arm]:
                    arms[arm][name]()
        except Exception as exc:
            if arm != "fla":
                raise
            fla_failed(exc)
    torch.cuda.synchronize(device)
    # peak and retained memory of one untimed call per arm and boundary
    for arm in list(arms):
        try:
            row["memory"][arm] = {
                name: _memory(arms[arm][name], device)
                for name in boundaries
                if name in arms[arm]
            }
        except Exception as exc:
            if arm != "fla":
                raise
            fla_failed(exc)
    samples.update(
        {arm: {name: [] for name in boundaries if name in arms[arm]} for arm in arms}
    )
    for r in range(rounds):
        order = [arm for arm in ARMS if arm in arms]
        round_order = order if r % 2 == 0 else order[::-1]
        row["arm_order"].append(round_order)
        for arm in round_order:
            if arm not in arms:
                continue
            try:
                for name in boundaries:
                    if name in arms[arm]:
                        samples[arm][name].append(_time(arms[arm][name]))
            except Exception as exc:
                if arm != "fla":
                    raise
                fla_failed(exc)
    for arm, by_name in samples.items():
        row["ms_rounds"][arm] = by_name
        row["ms"][arm] = {n: _median_ms(v) for n, v in by_name.items()}
        row["ms_min"][arm] = {n: min(v) for n, v in by_name.items()}
        row["ms_max"][arm] = {n: max(v) for n, v in by_name.items()}
    if arms_alone:
        # every arm on its own, one after another, its closures released before the next (rows whose Cake arms do
        # not fit in device memory together): the same inputs, ``rounds`` timed calls per boundary, not interleaved
        row["arms_alone"] = True
        # each Cake arm on a workspace of its own (the two workspaces of a large row need not fit together)
        det_ws = nd_ws = None
        gc.collect()
        torch.cuda.empty_cache()
        cake_builders = {
            "det": lambda: cake_boundaries(inputs, det_workspace(), det=True),
            "nd": lambda: cake_boundaries(inputs, nd_workspace()),
        }
        builders = {arm: cake_builders[arm] for arm in CAKE_ARMS[build_order]}
        builders[AA_ARM] = cake_builders["det"]
        if baseline:
            builders["fla"] = lambda: fla_boundaries(inputs)
        for arm, build in builders.items():
            fns = None
            try:
                fns = build()
                for name in boundaries:
                    if name in fns:
                        fns[name]()
                torch.cuda.synchronize(device)
                row["memory"][arm] = {
                    name: _memory(fns[name], device)
                    for name in boundaries
                    if name in fns
                }
                by_name = {
                    name: [_time(fns[name]) for _ in range(rounds)]
                    for name in boundaries
                    if name in fns
                }
                row["ms_rounds"][arm] = by_name
                row["ms"][arm] = {n: _median_ms(v) for n, v in by_name.items()}
                row["ms_min"][arm] = {n: min(v) for n, v in by_name.items()}
                row["ms_max"][arm] = {n: max(v) for n, v in by_name.items()}
            except Exception as exc:
                if arm != "fla":
                    raise
                row["fla_error"] = _error(exc)
                row["memory"].pop("fla", None)
            fns = None
            gc.collect()
            torch.cuda.empty_cache()
    if baseline and fla_alone and not arms_alone:
        # the FLA arm on its own, after the Cake arms and workspaces are released (rows whose three arms do not fit
        # in device memory at once): the same inputs, ``rounds`` timed calls per boundary, not interleaved
        row["fla_alone"] = True
        arms.clear()
        det_ws = nd_ws = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            fla = fla_boundaries(inputs)
            for name in boundaries:
                if name in fla:
                    fla[name]()
            torch.cuda.synchronize(device)
            row["memory"]["fla"] = {
                name: _memory(fla[name], device) for name in boundaries if name in fla
            }
            by_name = {
                name: [_time(fla[name]) for _ in range(rounds)]
                for name in boundaries
                if name in fla
            }
            row["ms_rounds"]["fla"] = by_name
            row["ms"]["fla"] = {n: _median_ms(v) for n, v in by_name.items()}
            row["ms_min"]["fla"] = {n: min(v) for n, v in by_name.items()}
            row["ms_max"]["fla"] = {n: max(v) for n, v in by_name.items()}
        except Exception as exc:
            row["fla_error"] = _error(exc)
            row["memory"].pop("fla", None)
            torch.cuda.synchronize(device)
        fla = None
        gc.collect()
        torch.cuda.empty_cache()
    for name in boundaries:
        nd_ms = row["ms"]["nd"].get(name)
        if nd_ms:
            row["ratio_det_over_nd"][name] = row["ms"]["det"][name] / nd_ms
            if "fla" in row["ms"] and name in row["ms"]["fla"]:
                row["ratio_fla_over_nd"][name] = row["ms"]["fla"][name] / nd_ms
        _aa_band(row, name)
    if side_stream:
        row["side_stream"] = {
            arm: {
                name: _latency_beside(arms[arm][name], device)
                for name in boundaries
                if name in arms[arm]
            }
            for arm in ("det", "nd")
        }
        row["side_stream_ratio_det_over_nd"] = {
            name: row["side_stream"]["det"][name]["beside_ms"]
            / row["side_stream"]["nd"][name]["beside_ms"]
            for name in row["side_stream"]["nd"]
        }
    return row


def combine_orders(by_order: dict[str, dict]) -> dict:
    """One row of ``--build-order both`` from its ``det-first`` and ``nd-first`` rows: per arm and boundary the mean
    of the two orders' medians (``order_avg``, also ``ms``), the ratios of those means, each order's ``det / nd``,
    and the boundaries where nd is slower than det in both orders."""
    first = by_order[ORDERS[0]]
    row = {
        key: first[key]
        for key in (
            "seqlen",
            "docs",
            "lengths",
            "heads",
            "gate",
            "l2n",
            "arch",
            "recurrence",
            "segments_per_doc",
            "s1_paths",
        )
    }
    row["build_order"] = "both"
    row["by_order"] = by_order
    ms: dict[str, dict[str, float]] = {}
    for arm in first["ms"]:
        if all(arm in r["ms"] for r in by_order.values()):
            ms[arm] = {
                name: statistics.fmean(r["ms"][arm][name] for r in by_order.values())
                for name in first["ms"][arm]
                if all(name in r["ms"][arm] for r in by_order.values())
            }
    row["ms"] = row["order_avg"] = ms
    row["ratio_det_over_nd"], row["ratio_fla_over_nd"] = {}, {}
    for name, nd_ms in ms.get("nd", {}).items():
        if "det" in ms and name in ms["det"]:
            row["ratio_det_over_nd"][name] = ms["det"][name] / nd_ms
        if "fla" in ms and name in ms["fla"]:
            row["ratio_fla_over_nd"][name] = ms["fla"][name] / nd_ms
    row["ratio_det_over_nd_by_order"] = {
        order: r["ratio_det_over_nd"] for order, r in by_order.items()
    }
    row["nd_slower_both_orders"] = [
        name
        for name in row["ratio_det_over_nd"]
        if all(r["ratio_det_over_nd"].get(name, 1.0) < 1.0 for r in by_order.values())
    ]
    # the A/A figures of the means: the band covers both orders' bands and the means' det2 / det
    row["ratio_det2_over_det"], row["aa_band"], row["det_over_nd_beyond_aa"] = (
        {},
        {},
        {},
    )
    for name, det_ms in ms.get("det", {}).items():
        bands = [r["aa_band"].get(name) for r in by_order.values()]
        if name not in ms.get(AA_ARM, {}) or None in bands:
            continue
        ratio = ms[AA_ARM][name] / det_ms
        w = max(abs(ratio - 1.0), *(band[1] - 1.0 for band in bands))
        row["ratio_det2_over_det"][name] = ratio
        row["aa_band"][name] = [1.0 - w, 1.0 + w]
        if name in row["ratio_det_over_nd"]:
            row["det_over_nd_beyond_aa"][name] = (
                abs(row["ratio_det_over_nd"][name] - 1.0) > w
            )
    errors = {o: r["fla_error"] for o, r in by_order.items() if "fla_error" in r}
    if errors:
        row["fla_error"] = "; ".join(f"{o}: {e}" for o, e in errors.items())
    return row


def _aa_band(row: dict, name: str) -> None:
    """The A/A figures of boundary ``name`` in ``row``: ``det2 / det`` of the medians, the band
    ``[1 - w, 1 + w]`` (``w`` the largest ``|det2 / det - 1|`` over the rounds' medians and the row's medians)
    and whether ``det / nd`` lies outside it."""
    det, aa = row["ms"].get("det", {}), row["ms"].get(AA_ARM, {})
    if name not in det or name not in aa:
        return
    ratio = aa[name] / det[name]
    det_rounds = row["ms_rounds"]["det"][name]
    aa_rounds = row["ms_rounds"][AA_ARM][name]
    per_round = [b / a for a, b in zip(det_rounds, aa_rounds, strict=True)]
    w = max(abs(x - 1.0) for x in (ratio, *per_round))
    row["ratio_det2_over_det"][name] = ratio
    row["aa_band"][name] = [1.0 - w, 1.0 + w]
    if name in row["ratio_det_over_nd"]:
        row["det_over_nd_beyond_aa"][name] = (
            abs(row["ratio_det_over_nd"][name] - 1.0) > w
        )


def _fla_out_vs_nd(inputs: dict, fla: dict, nd_ws) -> float:
    """The relative RMS difference of flash-linear-attention's ``out`` (its raw gate mode and / or its in-kernel
    L2 norm, as the row's operands take) from this package's on the same operands (an FLA adapter that drops a gate
    term or the norm differs by far more than the two kernels' roundings)."""
    out_fla = fla["fwd"]().detach().double()
    out_nd = cake_boundaries(inputs, nd_ws)["fwd"]()[0].double()
    return float((out_fla - out_nd).pow(2).mean().sqrt() / out_nd.pow(2).mean().sqrt())


def _print_row(row: dict, boundaries: tuple[str, ...]) -> None:
    cells = [
        f"S={row['seqlen']:>7} N={row['docs']} H={row['heads']:>2} gate={row['gate']}"
        f"{' l2n' if row['l2n'] else ''} {row['arch']} "
        f"{row['recurrence']}{'/P' + str(row['segments_per_doc']) if row['recurrence'] == 'seg' else ''} "
        f"S1[{','.join(row['s1_paths']) or '-'}] order={row['build_order']}"
    ]
    for name in boundaries:
        if name not in row["ms"]["nd"]:
            continue
        nd_ms, det_ms = row["ms"]["nd"][name], row["ms"]["det"][name]
        cell = (
            f"{name} det {det_ms:8.3f} nd {nd_ms:8.3f} ms det/nd x{det_ms / nd_ms:6.4f}"
        )
        if name in row["aa_band"]:
            w = row["aa_band"][name][1] - 1.0
            beyond = row["det_over_nd_beyond_aa"].get(name, False)
            cell += (
                f" det2/det x{row['ratio_det2_over_det'][name]:6.4f} A/A +-{100 * w:.2f}%"
                f"{' (det/nd beyond)' if beyond else ''}"
            )
        if name in row["ratio_fla_over_nd"]:
            cell += f" fla {row['ms']['fla'][name]:8.3f} fla/nd x{row['ratio_fla_over_nd'][name]:5.2f}"
        cells.append(cell)
    if "fla_raw_out_vs_nd" in row:
        cells.append(f"fla raw out vs nd rel RMS {row['fla_raw_out_vs_nd']:.2e}")
    if "fla_l2n_out_vs_nd" in row:
        cells.append(f"fla l2n out vs nd rel RMS {row['fla_l2n_out_vs_nd']:.2e}")
    if "fla_error" in row:
        cells.append(f"fla error: {row['fla_error'][:120]}")
    if row.get("nd_slower_both_orders"):
        cells.append(
            f"nd slower in both orders: {','.join(row['nd_slower_both_orders'])}"
        )
    print(" | ".join(cells))
    for name, ratio in row.get("side_stream_ratio_det_over_nd", {}).items():
        det, nd = row["side_stream"]["det"][name], row["side_stream"]["nd"][name]
        print(
            f"    {name} beside a resident kernel: det {det['alone_ms']:8.3f} -> {det['beside_ms']:8.3f} ms, "
            f"nd {nd['alone_ms']:8.3f} -> {nd['beside_ms']:8.3f} ms, det/nd x{ratio:6.4f}"
            + (
                ""
                if det["side_covered"] and nd["side_covered"]
                else " (side kernel ended early)"
            )
        )


def _write_json(
    path, rows, boundaries, device, arch, timer, fla_version, fla_module_version, args
) -> None:
    """The run's rows so far with its environment and timing method (rewritten after every row)."""
    with open(path, "w") as f:
        json.dump(
            {
                "device": torch.cuda.get_device_name(device),
                "arch": arch,
                "sm_count": cake_jit.sm_count(device.index),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "fla": fla_version,
                "fla_module_version": fla_module_version,
                "timing": f"{timer}, cold L2, median of the timed calls; per arm and boundary the "
                "median of the round medians",
                "memory": "per arm and boundary, one untimed call: peak_bytes = max_memory_allocated - "
                "the allocation before it, retained_bytes = memory_allocated after it while its results are "
                "held - the allocation before it",
                "side_stream": "CUDA-event wall latency, median of "
                f"{SIDE_SAMPLES} calls alone and beside a one-thread resident kernel"
                if args.side_stream
                else None,
                "boundaries": list(boundaries),
                "gates": list(args.gate),
                "build_order": args.build_order,
                "lower_bound": LOWER_BOUND if "raw" in args.gate else None,
                "l2n": args.l2n,
                "aa_arm": f"{AA_ARM}: the deterministic arm again on its workspace; "
                "aa_band = [1 - w, 1 + w], w the largest |det2 / det - 1| over the rounds' "
                "medians and the row's medians",
                "rows": rows,
            },
            f,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--docs", type=int, nargs="+", default=[3])
    parser.add_argument("--heads", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--gate",
        type=str,
        nargs="+",
        default=["g"],
        choices=GATES,
        help="the forget gate's forms per row: g (the activated FP32 gate) and / or raw (use_gate_in_kernel=True)",
    )
    parser.add_argument(
        "--l2n",
        action="store_true",
        help="every row with the in-kernel Q/K L2 norm (use_qk_l2norm_in_kernel=True on raw q / k, every arm)",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        nargs="+",
        default=None,
        help="explicit packings, each comma-separated document lengths (replaces --seqlen / --docs)",
    )
    parser.add_argument(
        "--boundaries",
        type=str,
        default=",".join(DEFAULT_BOUNDARIES),
        help=f"comma-separated subset of {','.join(BOUNDARIES)}",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="interleaved rounds per row (the arm order reversed every other round)",
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip flash-linear-attention"
    )
    parser.add_argument(
        "--side-stream",
        action="store_true",
        help="also time the Cake arms' wall latency alone and beside a kernel resident on another stream",
    )
    parser.add_argument("--json", type=str, default=None, help="write the rows here")
    parser.add_argument(
        "--arms-alone",
        action="store_true",
        help="time det, nd, det2 and flash-linear-attention one after another, each released before the next",
    )
    parser.add_argument(
        "--fla-alone",
        action="store_true",
        help="time the flash-linear-attention arm after releasing the Cake arms (rows whose three arms do not fit)",
    )
    parser.add_argument(
        "--build-order",
        type=str,
        default="det-first",
        choices=BUILD_ORDERS,
        help="which Cake arm allocates first (det-first, nd-first) or both orders per row (their mean)",
    )
    parser.add_argument(
        "--require-cupti",
        action="store_true",
        help="refuse to run when the timer would fall back to CUDA events",
    )
    args = parser.parse_args()
    timer = _timer()
    if (args.fla_alone or args.arms_alone) and args.side_stream:
        parser.error("--fla-alone releases the Cake arms the --side-stream rows time")
    if args.require_cupti and not timer.startswith("CUPTI"):
        parser.error(f"--require-cupti: the timer is {timer}")
    print(f"# timer: {timer}")
    boundaries = tuple(b.strip() for b in args.boundaries.split(",") if b.strip())
    unknown = [b for b in boundaries if b not in BOUNDARIES]
    if unknown or not boundaries:
        parser.error(f"unknown boundaries {unknown}; choose from {BOUNDARIES}")
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    device = torch.device("cuda", torch.cuda.current_device())
    arch = cake_jit.arch_for_capability(
        tuple(torch.cuda.get_device_capability(device.index))
    )
    if arch is None:
        raise SystemExit(
            f"{torch.cuda.get_device_name(device)} is not an exported target {cake_jit.ARCHES}"
        )
    fla_version = None if args.no_baseline else _fla_version()
    fla_module_version = None if args.no_baseline else _fla_module_version()
    if args.lengths:
        packs = [tuple(int(x) for x in p.split(",")) for p in args.lengths]
    else:
        packs = [split_docs(s, d) for s in args.seqlen for d in args.docs]
    rows = []
    orders = ORDERS if args.build_order == "both" else (args.build_order,)
    for lengths, heads, gate in itertools.product(packs, args.heads, args.gate):
        by_order = {}
        for order in orders:
            by_order[order] = run_row(
                sum(lengths),
                len(lengths),
                heads,
                device=device,
                baseline=not args.no_baseline,
                boundaries=boundaries,
                rounds=args.rounds,
                side_stream=args.side_stream,
                lengths=lengths,
                fla_alone=args.fla_alone,
                arms_alone=args.arms_alone,
                gate=gate,
                l2n=args.l2n,
                build_order=order,
            )
            _print_row(by_order[order], boundaries)
            gc.collect()
            torch.cuda.empty_cache()
        row = by_order[orders[0]] if len(orders) == 1 else combine_orders(by_order)
        if len(orders) > 1:
            _print_row(row, boundaries)
        rows.append(row)
        if args.json:
            _write_json(
                args.json,
                rows,
                boundaries,
                device,
                arch,
                timer,
                fla_version,
                fla_module_version,
                args,
            )
    for gate, name in itertools.product(args.gate, boundaries):
        ratios = [
            r["ratio_det_over_nd"][name]
            for r in rows
            if r["gate"] == gate and name in r["ratio_det_over_nd"]
        ]
        if ratios:
            geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            slower = sum(x < 1.0 for x in ratios)
            beyond = sum(
                r["ratio_det_over_nd"][name] < 1.0
                and r["det_over_nd_beyond_aa"].get(name, False)
                for r in rows
                if r["gate"] == gate and name in r["ratio_det_over_nd"]
            )
            print(
                f"# gate={gate}{' l2n' if args.l2n else ''} {name} ({args.build_order}): det / nd geomean "
                f"x{geo:.4f} over {len(ratios)} rows; nd slower on {slower} rows, beyond the A/A band on {beyond}"
            )
    if args.json:
        _write_json(
            args.json,
            rows,
            boundaries,
            device,
            arch,
            timer,
            fla_version,
            fla_module_version,
            args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
