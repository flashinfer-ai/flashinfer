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

Benchmark of the Cake fused BF16 RMSNorm training kernels against the
training-side comparators: ``torch.compile`` (inductor) of the FP32-promoted
formula, eager autograd of the same formula, ``torch.nn.functional.rms_norm``
autograd, Liger Kernel (when installed) and, for the forward only,
``flashinfer.norm.rmsnorm``.

Every row reports forward, backward (including the final dw reduction) and
their sum in microseconds plus the achieved GB/s over the logical byte volume
(forward ``4TH + 4T``, backward ``6TH + 4T + 4 * n_chunks * H``; the residual
routes add ``4TH`` / ``2TH``).  Timing uses CUPTI kernel activity with a cold
L2 (``flashinfer.testing.bench_gpu_time(enable_cupti=True)``); multi-kernel
comparators report the sum of their kernels' GPU time.

Example::

    python benchmarks/bench_cake_rmsnorm_train.py --hidden 6144 --rows 16231 65536
    python benchmarks/bench_cake_rmsnorm_train.py --residual --json out.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
from typing import Callable, Optional

import torch

import flashinfer
from flashinfer.cake_rmsnorm_train import (
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_forward,
    is_cake_rmsnorm_train_supported,
)
from flashinfer.jit import cake_rmsnorm_train as loader
from flashinfer.testing import bench_gpu_time

WIDTHS = ((6144, 1e-5), (2048, 1e-6), (512, 1e-6))
EPS = {hidden: eps for hidden, eps in WIDTHS}
ROWS = (256, 1024, 4096, 16172, 16231, 32768, 65536)


def _median_ms(samples) -> float:
    return float(statistics.median(list(samples)))


def _time(fn: Callable[[], object]) -> float:
    """Median GPU time (ms) of ``fn`` over its kernels, cold L2, CUPTI."""

    times = bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True)
    return _median_ms(times)


def _rmsnorm_fp32(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return ((xf * rstd) * w.float()).to(torch.bfloat16)


def _residual_rmsnorm_fp32(x, u, w, eps):
    h_new = (x.float() + u.float()).to(torch.bfloat16)
    return _rmsnorm_fp32(h_new, w, eps), h_new


class Comparator:
    name = "base"
    forward_only = False

    def available(self) -> tuple[bool, str]:
        return True, ""

    def make(self, x, w, eps, u=None):
        """Return ``(fwd, bwd)``: ``fwd()`` keeps the graph, ``bwd(g, g_h)`` runs it."""

        raise NotImplementedError


class _AutogradComparator(Comparator):
    def _forward_fn(self):
        raise NotImplementedError

    def make(self, x, w, eps, u=None):
        fn = self._forward_fn()
        xl = x.detach().clone().requires_grad_(True)
        wl = w.detach().float().requires_grad_(True)
        ul = u.detach().clone().requires_grad_(True) if u is not None else None
        state: dict = {}

        def fwd():
            xl.grad = wl.grad = None
            if ul is not None:
                ul.grad = None
                y, h_new = fn(xl, ul, wl, eps)
                state["outputs"] = (y, h_new)
                return y
            y = fn(xl, wl, eps)
            state["outputs"] = (y,)
            return y

        def bwd(g, g_h=None):
            xl.grad = wl.grad = None
            outputs = state["outputs"]
            grads = [g] if len(outputs) == 1 else [g, g_h]
            torch.autograd.backward(list(outputs), grads, retain_graph=True)
            return xl.grad, wl.grad

        return fwd, bwd


class TorchCompile(_AutogradComparator):
    name = "torch_compile"

    def available(self):
        try:
            import torch._dynamo  # noqa: F401
            import triton

            return True, f"triton {triton.__version__}"
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, repr(exc)

    def make(self, x, w, eps, u=None):
        self._residual = u is not None
        return super().make(x, w, eps, u)

    def _forward_fn(self):
        return torch.compile(
            _residual_rmsnorm_fp32 if self._residual else _rmsnorm_fp32, dynamic=False
        )


class EagerFp32(_AutogradComparator):
    name = "eager_fp32"

    def make(self, x, w, eps, u=None):
        self._residual = u is not None
        return super().make(x, w, eps, u)

    def _forward_fn(self):
        return _residual_rmsnorm_fp32 if self._residual else _rmsnorm_fp32


class EagerFRmsNorm(Comparator):
    name = "F.rms_norm"

    def make(self, x, w, eps, u=None):
        hidden = x.shape[-1]
        xl = x.detach().clone().requires_grad_(True)
        wl = w.detach().clone().requires_grad_(True)
        ul = u.detach().clone().requires_grad_(True) if u is not None else None
        state: dict = {}

        def fwd():
            xl.grad = wl.grad = None
            if ul is not None:
                ul.grad = None
                h_new = (xl.float() + ul.float()).to(torch.bfloat16)
                y = torch.nn.functional.rms_norm(h_new, (hidden,), wl, eps)
                state["outputs"] = (y, h_new)
                return y
            y = torch.nn.functional.rms_norm(xl, (hidden,), wl, eps)
            state["outputs"] = (y,)
            return y

        def bwd(g, g_h=None):
            xl.grad = wl.grad = None
            outputs = state["outputs"]
            grads = [g] if len(outputs) == 1 else [g, g_h]
            torch.autograd.backward(list(outputs), grads, retain_graph=True)
            return xl.grad, wl.grad.float()

        return fwd, bwd


class Liger(Comparator):
    name = "liger"

    def available(self):
        try:
            import liger_kernel
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # noqa: F401

            return True, f"liger-kernel {getattr(liger_kernel, '__version__', '?')}"
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, repr(exc)

    def make(self, x, w, eps, u=None):
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        nparams = len(inspect.signature(LigerRMSNormFunction.forward).parameters)
        xl = x.detach().clone().requires_grad_(True)
        wl = w.detach().clone().requires_grad_(True)
        ul = u.detach().clone().requires_grad_(True) if u is not None else None
        state: dict = {}

        def fwd():
            xl.grad = wl.grad = None
            source = xl
            if ul is not None:
                ul.grad = None
                source = (xl.float() + ul.float()).to(torch.bfloat16)
            # forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True[, row_mode])
            args = [source, wl, eps, 0.0, "llama", False]
            y = LigerRMSNormFunction.apply(*args[: max(3, nparams - 1)])
            state["outputs"] = (y,) if ul is None else (y, source)
            return y

        def bwd(g, g_h=None):
            xl.grad = wl.grad = None
            outputs = state["outputs"]
            grads = [g] if len(outputs) == 1 else [g, g_h]
            torch.autograd.backward(list(outputs), grads, retain_graph=True)
            return xl.grad, wl.grad.float()

        return fwd, bwd


class FlashInferForward(Comparator):
    name = "flashinfer.rmsnorm(fwd)"
    forward_only = True

    def make(self, x, w, eps, u=None):
        if u is not None:
            raise NotImplementedError("forward-only comparator without residual fusion")
        out = torch.empty_like(x)

        def fwd():
            flashinfer.norm.rmsnorm(x, w, eps, out=out)
            return out

        return fwd, None


class Cake(Comparator):
    name = "cake"

    def make(self, x, w, eps, u=None):
        rows, hidden = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        h_new = torch.empty_like(x) if u is not None else None
        dx = torch.empty_like(x)
        dw = torch.empty(hidden, dtype=torch.float32, device=x.device)
        workspace = cake_rmsnorm_train_backward_workspace(
            rows, hidden, x.device, residual=u is not None
        )
        normalized = h_new if u is not None else x

        def fwd():
            cake_rmsnorm_train_forward(
                x, w, eps, residual=u, out=y, rstd=rstd, residual_out=h_new
            )
            return y

        def bwd(g, g_h=None):
            cake_rmsnorm_train_backward(
                g,
                normalized,
                w,
                rstd,
                g_residual=g_h,
                workspace=workspace,
                dx=dx,
                dw=dw,
            )
            return dx, dw

        return fwd, bwd


COMPARATORS = (
    Cake(),
    TorchCompile(),
    EagerFp32(),
    EagerFRmsNorm(),
    Liger(),
    FlashInferForward(),
)


def _bytes(mode: str, rows: int, hidden: int, *, n_chunks: int, residual: bool) -> int:
    fwd = 4 * rows * hidden + 4 * rows + (4 * rows * hidden if residual else 0)
    bwd = (
        6 * rows * hidden
        + 4 * rows
        + 4 * n_chunks * hidden
        + (2 * rows * hidden if residual else 0)
    )
    return {"fwd": fwd, "bwd": bwd, "step": fwd + bwd}[mode]


def _n_chunks(
    arch: str, hidden: int, rows: int, residual: bool, *, device_index: int
) -> int:
    for name in loader.route_modules(arch, hidden, "bwd", residual):
        record = loader.MODULES[name]
        if record.get("workspace_rule") is not None:
            return loader.chunking(record, rows, device_index=device_index)[0]
    return 0


def bench_row(hidden: int, rows: int, *, residual: bool, comparators, device) -> dict:
    eps = EPS[hidden]
    generator = torch.Generator(device=device).manual_seed(760 + rows)
    x = torch.randn(
        rows, hidden, dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    w = (
        1.0
        + 0.1
        * torch.randn(hidden, dtype=torch.float32, device=device, generator=generator)
    ).to(torch.bfloat16)
    g = torch.randn(
        rows, hidden, dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    u = g_h = None
    if residual:
        u = torch.randn(
            rows, hidden, dtype=torch.float32, device=device, generator=generator
        ).to(torch.bfloat16)
        g_h = torch.randn(
            rows, hidden, dtype=torch.float32, device=device, generator=generator
        ).to(torch.bfloat16)
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    n_chunks = _n_chunks(arch, hidden, rows, residual, device_index=device.index)
    result = {
        "hidden": hidden,
        "rows": rows,
        "residual": residual,
        "eps": eps,
        "comparators": {},
    }
    for comparator in comparators:
        if comparator.forward_only and residual:
            continue
        entry: dict = {}
        try:
            fwd, bwd = comparator.make(x, w, eps, u)
            fwd()
            if bwd is not None:
                bwd(g, g_h)
            torch.cuda.synchronize()
            entry["fwd_ms"] = _time(fwd)
            if bwd is not None:
                fwd()
                torch.cuda.synchronize()
                entry["bwd_ms"] = _time(lambda: bwd(g, g_h))
                entry["step_ms"] = entry["fwd_ms"] + entry["bwd_ms"]
            for mode in ("fwd", "bwd", "step"):
                if f"{mode}_ms" in entry:
                    nbytes = _bytes(
                        mode, rows, hidden, n_chunks=n_chunks, residual=residual
                    )
                    entry[f"{mode}_gbps"] = nbytes / (entry[f"{mode}_ms"] * 1e-3) / 1e9
        except Exception as exc:  # keep the other comparators' rows
            entry["error"] = repr(exc)[:200]
        result["comparators"][comparator.name] = entry
        torch.cuda.empty_cache()
    return result


def _format(result: dict, names) -> list[str]:
    lines = []
    tag = " (+residual)" if result["residual"] else ""
    for mode in ("fwd", "bwd", "step"):
        cells = []
        for name in names:
            entry = result["comparators"].get(name, {})
            value = entry.get(f"{mode}_ms")
            cells.append("-" if value is None else f"{value * 1e3:9.2f}")
        line = (
            f"H={result['hidden']:5d} T={result['rows']:6d} {mode:4s}{tag:12s} | "
            + " | ".join(cells)
        )
        gbps = result["comparators"].get("cake", {}).get(f"{mode}_gbps")
        if gbps is not None:
            line += f" | cake {gbps:8.0f} GB/s"
        lines.append(line)
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hidden", type=int, nargs="*", default=[h for h, _ in WIDTHS])
    parser.add_argument("--rows", type=int, nargs="*", default=list(ROWS))
    parser.add_argument(
        "--residual",
        action="store_true",
        help="also run the fused residual-add rows (H=6144)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], help="comparator names to skip"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="write all rows to this JSON file"
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda", torch.cuda.current_device())
    if not is_cake_rmsnorm_train_supported(device):
        raise SystemExit("cake_rmsnorm_train has no exported route on this device")
    comparators = []
    for comparator in COMPARATORS:
        if comparator.name in args.skip:
            continue
        ok, note = comparator.available()
        print(f"{comparator.name}: {'available' if ok else 'unavailable'} {note}")
        if ok:
            comparators.append(comparator)
    names = [c.name for c in comparators]
    print("columns (us): " + " | ".join(names))
    results = []
    plans = [(h, r, False) for h in args.hidden for r in args.rows]
    if args.residual:
        plans += [(6144, r, True) for r in args.rows if 6144 in args.hidden]
    for hidden, rows, residual in plans:
        if not is_cake_rmsnorm_train_supported(device, hidden):
            print(f"H={hidden}: no exported route on this device, skipped")
            continue
        result = bench_row(
            hidden, rows, residual=residual, comparators=comparators, device=device
        )
        results.append(result)
        for line in _format(result, names):
            print(line, flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "torch": torch.__version__,
                    "rows": results,
                },
                handle,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
