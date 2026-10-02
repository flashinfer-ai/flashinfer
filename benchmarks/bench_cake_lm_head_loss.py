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
"""

"""Benchmark the experimental Cake chunked LM-head + loss training kernels (SM100 / SM103).

Times the forward, the backward and the whole training step (forward +
backward, wrapper-inclusive) of ``flashinfer.chunked_lm_head`` on the
GLM-class geometry (``H = 6144``, ``V = 154880``, BF16 ``X`` / ``W``, int64
labels with five percent ignored rows) over irregular token counts: the loss
entry with the cross-entropy and the policy objective at chunks of 2048 /
4096 / 8192 rows (``T`` = 16231, 16172, 4096, 32463), the log-probability
entry (recompute backward) at the same shapes, and the tail cases ``T`` = 1,
4095, 4097 (informational).  Baselines: the unchunked PyTorch autograd path
(``torch_unchunked``, the accuracy reference), a chunked PyTorch loop at the
same chunk and rounding boundaries (``torch_chunked``: forward-gradient form
for the loss entry, recompute form for the log-probability entry), and, when
importable, Liger's fused linear cross-entropy (``liger``, cross-entropy rows
only; its internal token chunk is recorded) and Cut Cross-Entropy (``cce``:
``impl="cce"`` with gradient filtering off, ``filter_eps=None``, the exact
form; the policy and log-probability rows composed from its per-token losses;
``cce_default`` = its default gradient filtering, approximate and
informational only, never a gate).

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and
a cold L2 between iterations (per-iteration GPU span); medians over
``--steps`` iterations.  Memory: the peak allocation above the live tensors
during the forward and during the backward, and the bytes the forward leaves
alive for the backward (saved state).  ``--accuracy`` adds the error report of
every arm against the chunked FP64 oracle and its ratio to the unchunked
PyTorch path (the 1.05x gate of the tests).  The JSON carries an ``env`` record
(torch / flashinfer / liger-kernel / cut-cross-entropy versions, device name
and compute capability).

Usage::

    python benchmarks/bench_cake_lm_head_loss.py [--rows t16231_c4096_ce ...]
        [--arms torch_unchunked,torch_chunked,liger,cce,cake] [--steps 20] [--accuracy] [--json out.json]
"""

import argparse
import importlib.metadata
import inspect
import json
import re
import statistics
import sys
import traceback
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashinfer  # noqa: E402
from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.chunked_lm_head import chunked_lm_head_logprob, chunked_lm_head_loss  # noqa: E402
from flashinfer.experimental.cake_lm_head_loss import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_lm_head_loss_reference import (  # noqa: E402
    DEFAULT_H,
    DEFAULT_LOSS_DIV,
    DEFAULT_V,
    IGNORE_INDEX,
    RATIO_CLIP,
    error_report,
    gate_ratios,
    make_inputs,
    make_weight,
    reference_fp64,
    reference_unchunked,
)

# name -> (T, chunk, objective, entry)
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

SEED_BASE = 761
H, V = DEFAULT_H, DEFAULT_V
LOSS_DIV = DEFAULT_LOSS_DIV


def median_ms(fn, steps):
    times = bench_gpu_time(
        fn, dry_run_iters=3, repeat_iters=steps, enable_cupti=True, cold_l2_cache=True
    )
    return float(statistics.median(times))


def gib(nbytes):
    return nbytes / 2**30


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def _mm_fp32(a, b):
    """``a @ b`` of BF16 operands with an FP32 result and no intermediate rounding."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.mm(a.float(), b.float())


def _addmm_fp32(acc, a, b):
    """``acc += a @ b`` of BF16 operands into the FP32 accumulator without intermediate rounding;
    returns the path taken (``addmm(out_dtype)``, ``mm(out_dtype)`` or FP32 operands)."""
    try:
        torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
        return "addmm_out_dtype"
    except (TypeError, RuntimeError):
        pass
    try:
        acc.add_(torch.mm(a, b, out_dtype=torch.float32))
        return "mm_out_dtype"
    except (TypeError, RuntimeError):
        acc.add_(torch.mm(a.float(), b.float()))
        return "float32_operands"


def _liger_internal_chunk(T, H, V):
    """Liger's internal token chunk, read from its source (``inc_factor = cdiv(V, k * H)``;
    ``chunk = min(next_pow2(cdiv(T, inc_factor)), T)``); ``"n/a"`` when the source does not match."""
    try:
        from liger_kernel.ops import fused_linear_cross_entropy as module

        src = inspect.getsource(module)
        lines = [
            line.strip()
            for line in src.splitlines()
            if "inc_factor" in line or "chunk_size" in line
        ][:8]
        match = re.search(r"inc_factor\s*=\s*triton\.cdiv\(\s*V\s*,\s*([^)]+)\)", src)
        expr = match.group(1).replace(" ", "") if match else ""
        if expr == "H":
            k = 1
        elif re.fullmatch(r"\d+\*H", expr):
            k = int(expr.split("*")[0])
        elif re.fullmatch(r"H\*\d+", expr):
            k = int(expr.split("*")[1])
        else:
            return dict(chunk="n/a", source=lines)
        inc_factor = -(-V // (k * H))
        n = max(1, -(-T // inc_factor))
        chunk = min(1 << (n - 1).bit_length(), T)
        return dict(
            chunk=chunk, num_chunks=-(-T // chunk), inc_factor=inc_factor, source=lines
        )
    except Exception as exc:
        return dict(chunk="n/a", error=f"{type(exc).__name__}: {exc}")


def _downstream_loss(logp, inp):
    """The log-probability entry's downstream loss ``(logp * dlogp)[valid].sum()``."""
    return (logp * inp.dlogp)[inp.valid].sum()


def _objective_loss(logp, inp):
    if inp.objective == "ce":
        return -logp[inp.valid].sum() / inp.loss_div
    ratio = torch.exp(logp - inp.infer_logp)
    return -(inp.loss_weights * torch.clamp_max(ratio, RATIO_CLIP))[inp.valid].sum()


# ---------------------------------------------------------------------------
# Arms: ``forward()`` returns the state (``loss`` FP32 scalar, ``logp`` FP32 [T] plus what the
# backward needs); ``backward(st)`` returns ``dX`` (BF16 [T, H]) and ``dW`` (BF16 [V, H]).
# ---------------------------------------------------------------------------


class ArmAutograd:
    """Base of the autograd arms: leaves, retained-graph backward."""

    def __init__(self, inp, *, entry, chunk):
        self.inp, self.entry, self.chunk = inp, entry, chunk
        self.X = inp.X.detach().requires_grad_()
        self.W = inp.W.detach().requires_grad_()

    def backward(self, st):
        dX, dW = torch.autograd.grad(st["loss"], (self.X, self.W), retain_graph=True)
        return dict(dX=dX, dW=dW)

    def outputs(self, st):
        return dict(loss=st["loss"].detach(), logp=st["logp"].detach())

    def _finish(self, logp):
        loss = (
            _downstream_loss(logp, self.inp)
            if self.entry == "logprob"
            else _objective_loss(logp, self.inp)
        )
        return dict(loss=loss, logp=logp)


class ArmTorchUnchunked(ArmAutograd):
    """The unchunked PyTorch path (accuracy reference): BF16 GEMM promoted to FP32, logsumexp, autograd."""

    name = "torch_unchunked"

    def versions(self):
        return dict(torch=torch.__version__)

    def forward(self):
        inp = self.inp
        z = (self.X @ self.W.t()).float()
        lse = torch.logsumexp(z, dim=-1)
        index = torch.where(inp.valid, inp.labels, torch.zeros_like(inp.labels))
        zy = z.gather(1, index[:, None]).squeeze(1)
        logp = torch.where(inp.valid, zy - lse, torch.zeros_like(lse))
        return self._finish(logp)


class ArmTorchChunked:
    """Chunked PyTorch loop at the same chunk and rounding boundaries, without autograd.

    Loss entry: the forward produces the FP32 gradient accumulators chunk by
    chunk (three GEMMs), the backward casts them.  Log-probability entry: the
    forward saves ``lse``; the backward recomputes each chunk's logits.
    """

    name = "torch_chunked"

    def __init__(self, inp, *, entry, chunk):
        self.inp, self.entry, self.chunk = inp, entry, chunk
        self.T = inp.T
        self.dx_acc = torch.empty((self.T, H), dtype=torch.float32, device=inp.X.device)
        self.dw_acc = torch.empty((V, H), dtype=torch.float32, device=inp.X.device)

    def versions(self):
        probe = torch.zeros((8, 8), dtype=torch.float32, device=self.inp.X.device)
        a = torch.zeros((8, 8), dtype=torch.bfloat16, device=probe.device)
        return dict(
            torch=torch.__version__,
            dw_accumulate=_addmm_fp32(probe, a.t(), a),
            dx_out=("out_dtype" if _mm_fp32(a, a).dtype == torch.float32 else "?"),
        )

    def _chunk_stats(self, r0, r1):
        inp = self.inp
        zf = torch.mm(
            inp.X[r0:r1], inp.W.t()
        ).float()  # BF16 GEMM output promoted to FP32
        lse = torch.logsumexp(zf, dim=-1)
        labels = inp.labels[r0:r1]
        valid = labels >= 0
        index = torch.where(valid, labels, torch.zeros_like(labels))
        zy = zf.gather(1, index[:, None]).squeeze(1)
        logp = torch.where(valid, zy - lse, torch.zeros_like(lse))
        return zf, lse, valid, index, logp

    @staticmethod
    def _dlogits(zf, lse, index, d):
        dz = torch.exp(zf - lse[:, None]).neg_()
        dz.scatter_add_(1, index[:, None], torch.ones_like(lse)[:, None])
        return dz.mul_(d[:, None]).to(torch.bfloat16)  # the BF16 dlogits boundary

    def _accumulate(self, r0, r1, dz, first):
        inp = self.inp
        self.dx_acc[r0:r1] = _mm_fp32(dz, inp.W)
        if first:
            self.dw_acc.zero_()
        _addmm_fp32(self.dw_acc, dz.t(), inp.X[r0:r1])  # dW_acc += dz^T @ X_c in FP32

    def forward(self):
        inp = self.inp
        logp_out = torch.empty((self.T,), dtype=torch.float32, device=inp.X.device)
        lse_out = torch.empty_like(logp_out)
        total = torch.zeros((), dtype=torch.float32, device=inp.X.device)
        for i, r0 in enumerate(range(0, self.T, self.chunk)):
            r1 = min(self.T, r0 + self.chunk)
            zf, lse, valid, index, logp = self._chunk_stats(r0, r1)
            logp_out[r0:r1], lse_out[r0:r1] = logp, lse
            zero = torch.zeros_like(lse)
            if self.entry == "logprob":
                total += (logp * inp.dlogp[r0:r1])[valid].sum()
                continue
            if inp.objective == "ce":
                d = torch.where(valid, torch.full_like(lse, -1.0 / inp.loss_div), zero)
                total += logp.sum()
            else:
                ratio = torch.exp(logp - inp.infer_logp[r0:r1])
                w = inp.loss_weights[r0:r1]
                d = torch.where(valid & (ratio <= RATIO_CLIP), -w * ratio, zero)
                total += torch.where(
                    valid, w * torch.clamp_max(ratio, RATIO_CLIP), zero
                ).sum()
            self._accumulate(r0, r1, self._dlogits(zf, lse, index, d), i == 0)
        if self.entry == "logprob":
            loss = total
        else:
            loss = -total / inp.loss_div if inp.objective == "ce" else -total
        return dict(loss=loss, logp=logp_out, lse=lse_out)

    def backward(self, st):
        inp = self.inp
        if self.entry == "logprob":
            for i, r0 in enumerate(range(0, self.T, self.chunk)):
                r1 = min(self.T, r0 + self.chunk)
                zf = torch.mm(inp.X[r0:r1], inp.W.t()).float()  # recompute
                labels = inp.labels[r0:r1]
                valid = labels >= 0
                index = torch.where(valid, labels, torch.zeros_like(labels))
                d = torch.where(
                    valid, inp.dlogp[r0:r1], torch.zeros_like(st["lse"][r0:r1])
                )
                self._accumulate(
                    r0, r1, self._dlogits(zf, st["lse"][r0:r1], index, d), i == 0
                )
        return dict(
            dX=self.dx_acc.to(torch.bfloat16), dW=self.dw_acc.to(torch.bfloat16)
        )

    def outputs(self, st):
        return dict(loss=st["loss"], logp=st["logp"])


class ArmLiger(ArmAutograd):
    """Liger fused linear cross-entropy (``LigerFusedLinearCrossEntropyFunction``, ``reduction="sum"``,
    ``loss = sum / loss_div``); cross-entropy loss rows only.  Its internal token chunk is recorded."""

    name = "liger"

    def __init__(self, inp, *, entry, chunk):
        super().__init__(inp, entry=entry, chunk=chunk)
        if entry != "loss" or inp.objective != "ce":
            raise NotImplementedError("liger: cross-entropy loss rows only")
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        self.function = LigerFusedLinearCrossEntropyFunction
        params = list(inspect.signature(self.function.forward).parameters.values())[
            1:
        ]  # drop ctx
        overrides = {
            params[0].name: self.X,
            params[1].name: self.W,
            params[2].name: inp.labels,
            "ignore_index": IGNORE_INDEX,
            "reduction": "sum",
        }
        if not {"ignore_index", "reduction"} <= {p.name for p in params}:
            raise RuntimeError(
                f"unexpected LigerFusedLinearCrossEntropyFunction.forward signature: {[p.name for p in params]}"
            )
        self.args = [overrides.get(p.name, p.default) for p in params]
        self.internal_chunk = _liger_internal_chunk(inp.T, H, V)

    def versions(self):
        return dict(
            liger_kernel=_version("liger-kernel"),
            internal_chunk=self.internal_chunk,
            torch=torch.__version__,
        )

    def forward(self):
        loss = self.function.apply(*self.args)
        if isinstance(loss, tuple):
            loss = loss[0]
        return dict(loss=loss / self.inp.loss_div, logp=None)

    def outputs(self, st):
        return dict(loss=st["loss"].detach(), logp=None)


class ArmCCE(ArmAutograd):
    """Cut Cross-Entropy in its exact form (``impl="cce"`` with gradient filtering off,
    ``filter_eps=None``): cross-entropy rows directly; policy and log-probability rows composed in
    PyTorch from its per-token losses (``reduction="none"``)."""

    name = "cce"
    filter_eps = None
    informational = False

    def __init__(self, inp, *, entry, chunk):
        super().__init__(inp, entry=entry, chunk=chunk)
        from cut_cross_entropy import linear_cross_entropy

        self.fn = linear_cross_entropy

    def versions(self):
        return dict(
            cut_cross_entropy=_version("cut-cross-entropy"),
            impl="cce",
            filter_eps=self.filter_eps,
            informational=self.informational,
            torch=torch.__version__,
        )

    def _cce(self, reduction):
        return self.fn(
            self.X,
            self.W,
            self.inp.labels,
            ignore_index=IGNORE_INDEX,
            reduction=reduction,
            impl="cce",
            filter_eps=self.filter_eps,
        )

    def forward(self):
        inp = self.inp
        if self.entry == "loss" and inp.objective == "ce":
            return dict(loss=self._cce("sum") / inp.loss_div, logp=None)
        nll = self._cce("none")
        logp = torch.where(inp.valid, -nll, torch.zeros_like(nll))
        return self._finish(logp)

    def outputs(self, st):
        return dict(
            loss=st["loss"].detach(),
            logp=None if st["logp"] is None else st["logp"].detach(),
        )


class ArmCCEDefault(ArmCCE):
    """Cut Cross-Entropy at its defaults (``impl="cce"``, ``filter_eps="auto"`` gradient filtering):
    approximate -- informational only, never a gate."""

    name = "cce_default"
    filter_eps = "auto"
    informational = True


class ArmCake(ArmAutograd):
    """The public experimental entry points (``flashinfer.chunked_lm_head``, ``backend="cake"``)."""

    name = "cake"

    def __init__(self, inp, *, entry, chunk):
        super().__init__(inp, entry=entry, chunk=chunk)
        device = inp.X.device
        if not cake_backend.generated_program_available(device, entry=entry):
            capability = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                f"generated chunked LM-head program ({entry} entry) not registered for compute capability "
                f"{capability[0]}.{capability[1]} in this checkout"
            )
        self.module_name, record = cake_backend.record_for(device)
        self.abi = cake_backend.record_abi(record)

    def versions(self):
        return dict(
            module=self.module_name,
            abi=self.abi,
            stages=list(cake_backend.stages_for_entry(self.entry)),
            compact_rows=cake_backend.compact_rows_default(),
        )  # valid-row compaction (the labels carry ignored rows)

    def forward(self):
        inp = self.inp
        if self.entry == "logprob":
            logp = chunked_lm_head_logprob(
                self.X, self.W, inp.labels, chunk_size=self.chunk, backend="cake"
            )
            return dict(loss=_downstream_loss(logp, inp), logp=logp)
        loss, logp = chunked_lm_head_loss(
            self.X,
            self.W,
            inp.labels,
            objective=inp.objective,
            loss_div=inp.loss_div if inp.objective == "ce" else None,
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=self.chunk,
            return_logp=True,
            backend="cake",
        )
        return dict(loss=loss, logp=logp)


ARMS = {
    cls.name: cls
    for cls in (
        ArmTorchUnchunked,
        ArmTorchChunked,
        ArmLiger,
        ArmCCE,
        ArmCCEDefault,
        ArmCake,
    )
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_perf(arm, steps):
    """Median forward / backward / step milliseconds, the peak allocation above the live tensors during
    the forward and during the backward, and the bytes the forward leaves alive for the backward."""
    saved = [
        arm.forward()
    ]  # the forward state the timed backward consumes; released before the memory probes
    torch.cuda.synchronize()
    fwd_ms = median_ms(arm.forward, steps)
    bwd_ms = median_ms(lambda: arm.backward(saved[0]), steps)

    def full():
        s = arm.forward()
        arm.backward(s)

    step_ms = median_ms(full, steps)
    saved.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    st = arm.forward()
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated() - base
    saved = torch.cuda.memory_allocated() - base
    torch.cuda.reset_peak_memory_stats()
    base_bwd = torch.cuda.memory_allocated()
    grads = arm.backward(st)
    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated() - base_bwd
    del grads, st
    return dict(
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        step_ms=step_ms,
        fwd_peak_gib=gib(fwd_peak),
        bwd_peak_gib=gib(bwd_peak),
        saved_gib=gib(saved),
    )


def measure_accuracy(arm, inp, oracle, b0_errors):
    st = arm.forward()
    grads = arm.backward(st)
    torch.cuda.synchronize()
    outs = arm.outputs(st)
    result = dict(loss=outs["loss"], logp=outs["logp"], dX=grads["dX"], dW=grads["dW"])
    if (
        result["logp"] is None
    ):  # arms without per-token output: the row metrics fall away
        result["logp"] = oracle["logp"].float()
    errors = error_report(result, oracle, inp.labels)
    if outs["logp"] is None:
        for key in ("logp_max_abs", "logp_mean_abs", "logp_ignored_zero"):
            errors[key] = None
    ratios = gate_ratios({k: v for k, v in errors.items() if v is not None}, b0_errors)
    informational = bool(getattr(arm, "informational", False))
    passes = (
        None
        if informational
        else all(r <= 1.05 or b0_errors.get(k, 0) == 0 for k, r in ratios.items())
    )
    return dict(
        errors=errors,
        vs_unchunked=ratios,
        passes_1p05x=passes,
        informational=informational,
    )


def _arm_or_error(cls, inp, entry, chunk):
    try:
        return cls(inp, entry=entry, chunk=chunk), None
    except (
        Exception
    ) as exc:  # optional dependency missing or arm unavailable on this device
        return None, f"{type(exc).__name__}: {exc}"


def _row_inputs(row, args, W):
    T, C, objective, entry = ROWS[row]
    return (
        make_inputs(
            T, objective=objective, seed=SEED_BASE + T, device=args.device, W=W
        ),
        C,
        entry,
    )


def run_perf(args, results, W):
    fmt = lambda v: "   n/a" if v is None else f"{v:8.3f}"
    for row in args.rows:
        inp, C, entry = _row_inputs(row, args, W)
        record = dict(
            row=row,
            T=inp.T,
            chunk=C,
            objective=inp.objective,
            entry=entry,
            inputs_gib=gib(inp.bytes_inputs()),
            arms={},
        )
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp, entry, C)
            if arm is None:
                record["arms"][name] = dict(unsupported=error)
                print(f"{row:22s} {name:16s} unsupported: {error}", flush=True)
                continue
            try:
                perf = measure_perf(arm, max(args.min_steps, args.steps))
                perf["versions"] = arm.versions()
            except Exception as exc:
                perf = dict(
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            record["arms"][name] = perf
            if "error" in perf:
                print(f"{row:22s} {name:16s} failed: {perf['error']}", flush=True)
            else:
                print(
                    f"{row:22s} {name:16s} fwd {fmt(perf['fwd_ms'])} ms  bwd {fmt(perf['bwd_ms'])} ms  step {fmt(perf['step_ms'])} ms  "
                    f"peak fwd {perf['fwd_peak_gib']:6.2f} GiB  bwd {perf['bwd_peak_gib']:6.2f} GiB  saved {perf['saved_gib']:6.2f} GiB",
                    flush=True,
                )
            del arm
            torch.cuda.empty_cache()
        results["rows"].append(record)
        del inp
        torch.cuda.empty_cache()


def run_accuracy(args, results, W):
    for row in args.rows:
        inp, C, entry = _row_inputs(row, args, W)
        oracle = reference_fp64(inp, entry=entry)
        b0 = reference_unchunked(inp, entry=entry)
        b0_errors = error_report(b0, oracle, inp.labels)
        del b0
        torch.cuda.empty_cache()
        record = dict(
            row=row,
            T=inp.T,
            chunk=C,
            objective=inp.objective,
            entry=entry,
            unchunked_errors=b0_errors,
            arms={},
        )
        print(
            f"{row:22s} {'unchunked (B0)':16s} {json.dumps(b0_errors, default=str)}",
            flush=True,
        )
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp, entry, C)
            if arm is None:
                record["arms"][name] = dict(unsupported=error)
                print(f"{row:22s} {name:16s} unsupported: {error}", flush=True)
                continue
            try:
                record["arms"][name] = measure_accuracy(arm, inp, oracle, b0_errors)
            except Exception as exc:
                record["arms"][name] = dict(
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            print(
                f"{row:22s} {name:16s} {json.dumps(record['arms'][name], default=str)}",
                flush=True,
            )
            del arm
            torch.cuda.empty_cache()
        results["accuracy"].append(record)
        del inp, oracle
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=list(ROWS), choices=list(ROWS))
    parser.add_argument(
        "--arms",
        default="torch_unchunked,torch_chunked,liger,cce,cake",
        help=f"comma-separated subset of {sorted(ARMS)}",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    args.device = device
    warnings.simplefilter(
        "ignore", ExperimentalWarning
    )  # the experimental banner of the public API
    env = dict(
        torch=torch.__version__,
        flashinfer=getattr(flashinfer, "__version__", None)
        or _version("flashinfer-python"),
        liger_kernel=_version("liger-kernel"),
        cut_cross_entropy=_version("cut-cross-entropy"),
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
    )
    results = dict(
        env=env,
        device=env["device"],
        capability=env["capability"],
        torch=torch.__version__,
        geometry=dict(H=H, V=V, loss_div=LOSS_DIV, seed_base=SEED_BASE),
        program_available=dict(
            loss=cake_backend.generated_program_available(device, entry="loss"),
            logprob=cake_backend.generated_program_available(device, entry="logprob"),
        ),
        rows=[],
        accuracy=[],
    )
    W = make_weight(
        V, H, seed=SEED_BASE + 1000003, device=device
    )  # shared by every row of the process
    if not args.no_perf:
        run_perf(args, results, W)
    if args.accuracy:
        run_accuracy(args, results, W)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
