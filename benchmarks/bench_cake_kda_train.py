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

Benchmark of the Cake packed KDA chunked training kernels
(``flashinfer.experimental.cake_kda_train``) on three boundaries of a
training step, against flash-linear-attention's ``chunk_kda`` when it is
installed:

* ``fwd``: the forward with the saved set written;
* ``bwd``: the backward from the saved set;
* ``ckpt``: ``torch.utils.checkpoint`` forward + replay + backward (one span).

Every row is one packed batch of ``--docs`` documents of ``--seqlen`` tokens
in total (unequal lengths with unaligned tails) and ``--heads`` heads; the
workspace is allocated once per row and reused, so the timed closures allocate
nothing.  Timing uses CUPTI kernel activity with a cold L2
(``flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)``);
the reported figure is the median over the timed calls of the summed GPU time
of a boundary's kernels, in milliseconds, with tokens per second beside it.

Example::

    python benchmarks/bench_cake_kda_train.py --seqlen 16384 65536 --docs 3 --heads 4 64
    python benchmarks/bench_cake_kda_train.py --seqlen 261393 --docs 4 --heads 4 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.cake_kda_train import cake_backend, cake_jit  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_kda_train_reference import make_inputs  # noqa: E402

BOUNDARIES = ("fwd", "bwd", "ckpt")


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


def cake_boundaries(
    inputs: dict, ws: cake_backend.KdaWorkspace
) -> dict[str, Callable[[], None]]:
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )

    def fwd() -> None:
        cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, save_for_backward=True, workspace=ws
        )

    _, saved = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, save_for_backward=True, workspace=ws
    )

    def bwd() -> None:
        cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )

    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def ckpt() -> None:
        o = checkpoint(
            lambda *a: cake_backend.chunk_kda(*a, cu, workspace=ws),
            *leaves,
            use_reentrant=False,
        )
        torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt}


def fla_boundaries(inputs: dict) -> dict[str, Callable[[], None]] | None:
    """flash-linear-attention's ``chunk_kda`` on the same preactivated operands (no in-kernel L2 norm, gate or
    sigmoid), or ``None`` when the package is not installed."""
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda
    except Exception:
        return None
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )

    def call(q_, k_, v_, g_, beta_):
        o, _ = fla_chunk_kda(
            q_.unsqueeze(0),
            k_.unsqueeze(0),
            v_.unsqueeze(0),
            g_.unsqueeze(0),
            beta_.unsqueeze(0),
            scale=cake_backend.SCALE,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=True,
            cu_seqlens=cu,
        )
        return o.squeeze(0)

    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def fwd() -> None:
        call(*leaves)

    retained = call(*leaves)

    def bwd() -> None:
        torch.autograd.grad(retained, leaves, dout, retain_graph=True)

    def ckpt() -> None:
        o = checkpoint(call, *leaves, use_reentrant=False)
        torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt}


def run_row(
    seqlen: int, docs: int, heads: int, *, device: torch.device, baseline: bool
) -> dict:
    lengths = split_docs(seqlen, docs)
    inputs = make_inputs(lengths, heads, device)
    ws = cake_backend.KdaWorkspace.for_problem(seqlen, docs, heads, device)
    ws.load()
    row = {
        "seqlen": seqlen,
        "docs": docs,
        "lengths": list(lengths),
        "heads": heads,
        "arch": ws.arch,
        "records": ws.records(),
        "recurrence": ws.plan.recur,
        "segments_per_doc": ws.plan.P,
        "workspace_bytes": ws.bytes,
        "saved_bytes": cake_backend.saved_bytes(
            seqlen, heads, docs, sm_count=ws.plan.sm_count
        ),
        "cake_ms": {},
        "fla_ms": {},
    }
    cake = cake_boundaries(inputs, ws)
    for name in BOUNDARIES:
        cake[name]()
    torch.cuda.synchronize(device)
    for name in BOUNDARIES:
        row["cake_ms"][name] = _time(cake[name])
    if baseline:
        fla = fla_boundaries(inputs)
        if fla is not None:
            for name in BOUNDARIES:
                fla[name]()
            torch.cuda.synchronize(device)
            for name in BOUNDARIES:
                row["fla_ms"][name] = _time(fla[name])
    return row


def _print_row(row: dict) -> None:
    cells = [
        f"S={row['seqlen']:>7} N={row['docs']} H={row['heads']:>2} {row['arch']} "
        f"{row['recurrence']}{'/P' + str(row['segments_per_doc']) if row['recurrence'] == 'seg' else ''}"
    ]
    for name in BOUNDARIES:
        ms = row["cake_ms"][name]
        cell = f"{name} {ms:8.3f} ms ({row['seqlen'] * row['heads'] / ms / 1e3:8.1f} Mtok-head/s)"
        if name in row["fla_ms"]:
            cell += (
                f" fla {row['fla_ms'][name]:8.3f} ms x{row['fla_ms'][name] / ms:5.2f}"
            )
        cells.append(cell)
    print(" | ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--docs", type=int, nargs="+", default=[3])
    parser.add_argument("--heads", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip flash-linear-attention"
    )
    parser.add_argument("--json", type=str, default=None, help="write the rows here")
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    arch = cake_jit.arch_for_capability(
        tuple(torch.cuda.get_device_capability(device.index))
    )
    if arch is None:
        raise SystemExit(
            f"{torch.cuda.get_device_name(device)} is not an exported target {cake_jit.ARCHES}"
        )
    rows = []
    for seqlen in args.seqlen:
        for docs in args.docs:
            for heads in args.heads:
                row = run_row(
                    seqlen, docs, heads, device=device, baseline=not args.no_baseline
                )
                _print_row(row)
                rows.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "arch": arch,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "rows": rows,
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
