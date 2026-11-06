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

"""Benchmark the experimental MLA value projection + sigmoid output gate (SM100 / SM103).

For every decode width ``B`` (``H = 12`` heads, one Kimi-K3 TP8 rank, unless
``--heads``) this reports the complete call ``runner.launch(latent, gate,
valid_lengths, out)`` replayed from a CUDA graph: CUPTI kernel time, median
over the measured iterations, with the L2 flushed before every iteration
unless ``--warm-l2``.  The flush needs CUPTI (``cupti-python`` >= 13); without
it the script refuses a cold-L2 run, and a ``--warm-l2`` run falls back to
CUDA-event timing of the graph replays.  The gate is the last ``H * 128``
columns of a ``[B, pitch]`` fused projection output (``--gate-pitch 3648``,
the default: ``projected[:, 2112:]`` at ``H = 12``) or a dense
``[B, H * 128]`` tensor (``--gate-pitch`` equal to ``H * 128``).  Every row is
valid.  The bandwidth column counts the bytes the call must move (``w_uv``,
``latent``, the gate and ``out``).  ``--baseline unfused`` adds the unfused
composition on the same tensors under the same protocol: ``torch.bmm`` into
the ``[B, H * 128]`` output layout, then the FP32 sigmoid gate, the BF16
rounding and the padded-row zeroing as elementwise torch operations.  The
``tile`` column is the row tile (16 or 8) of the program prepare chose for
that ``B``.

Usage::

    python benchmarks/bench_cake_mla_value_gate.py [--rows 1 2 ...] [--heads 12]
        [--gate-pitch 3648] [--no-pdl] [--warm-l2] [--baseline none|unfused]
        [--json out.json]
"""

import argparse
import json
import statistics

import torch

from flashinfer.cake_mla_value_gate import prepare_mla_value_gate
from flashinfer.testing import bench_gpu_time_with_cupti


def _cupti_available() -> bool:
    """True when CUPTI >= 13 is importable: the per-iteration L2 flush needs it."""
    try:
        from importlib.metadata import version

        from cupti import cupti  # noqa: F401

        return int(version("cupti-python").split(".")[0]) >= 13
    except (ImportError, ValueError):
        return False


# The owner's decode widths.
ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64)
LATENT, VALUE = 512, 128


def _inputs(rows, heads, pitch, seed=42):
    g = torch.Generator().manual_seed(seed)
    width = heads * VALUE
    w_uv = (
        (torch.randn((heads, VALUE, LATENT), generator=g) * 0.05)
        .to(torch.bfloat16)
        .cuda()
        .transpose(1, 2)
    )
    latent = torch.randn((rows, heads, LATENT), generator=g).to(torch.bfloat16).cuda()
    projected = (
        (torch.randn((rows, pitch), generator=g) * 4.0).to(torch.bfloat16).cuda()
    )
    gate = projected[:, pitch - width :]
    valid_lengths = torch.ones(rows, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
    return w_uv, latent, gate, valid_lengths, out


def _timed(fn, cold_l2):
    times = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=cold_l2)
    return float(statistics.median(times))


def _unfused(w_uv, latent, gate, valid_lengths, out):
    rows, heads, _ = latent.shape
    projection = torch.empty_like(out)

    def run():
        # [H, B, 512] x [H, 512, 128] -> the [B, H * 128] layout, BF16-rounded once.
        torch.bmm(
            latent.transpose(0, 1),
            w_uv,
            out=projection.view(rows, heads, VALUE).transpose(0, 1),
        )
        gated = projection.float() * torch.sigmoid(gate.float())
        gated.masked_fill_((valid_lengths <= 0)[:, None], 0.0)
        out.copy_(gated)

    return run


def bench_row(rows, heads, pitch, enable_pdl, cold_l2, baseline):
    w_uv, latent, gate, valid_lengths, out = _inputs(rows, heads, pitch)
    runner = prepare_mla_value_gate(
        w_uv, max_rows=64, sigmoid_precision="fp32", enable_pdl=enable_pdl
    )
    runner.launch(latent, gate, valid_lengths, out)  # load the program
    torch.cuda.synchronize()
    width = heads * VALUE
    nbytes = 2 * (heads * LATENT * VALUE + rows * heads * LATENT + 2 * rows * width)
    call_ms = _timed(lambda: runner.launch(latent, gate, valid_lengths, out), cold_l2)
    record = {
        "rows": rows,
        "heads": heads,
        "gate_pitch": pitch,
        "arch": runner.arch,
        "program": runner.program_for(rows),
        "tile_rows": runner.tile_rows_for(rows),
        "enable_pdl": enable_pdl,
        "l2": "cold" if cold_l2 else "warm",
        "call_us_median": call_ms * 1e3,
        "call_gbps": nbytes / (call_ms * 1e-3) / 1e9,
    }
    if baseline == "unfused":
        record["unfused_us_median"] = (
            _timed(_unfused(w_uv, latent, gate, valid_lengths, out), cold_l2) * 1e3
        )
        record["speedup_vs_unfused"] = (
            record["unfused_us_median"] / record["call_us_median"]
        )
    return record


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--gate-pitch", type=int, default=3648)
    parser.add_argument("--no-pdl", action="store_true")
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--baseline", default="none", choices=("none", "unfused"))
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    if args.gate_pitch < args.heads * VALUE or args.gate_pitch % 2:
        parser.error(f"--gate-pitch must be even and at least {args.heads * VALUE}")
    cupti = _cupti_available()
    if not args.warm_l2 and not cupti:
        parser.error(
            "a cold-L2 run flushes the L2 through CUPTI: install cupti-python >= 13 "
            "or pass --warm-l2"
        )
    timer = "CUPTI" if cupti else "CUDA events"
    records = [
        bench_row(
            rows,
            args.heads,
            args.gate_pitch,
            not args.no_pdl,
            not args.warm_l2,
            args.baseline,
        )
        for rows in args.rows
    ]
    first = records[0]
    print(
        f"{torch.cuda.get_device_name()}  {first['arch']}  CUDA {torch.version.cuda}  "
        f"H={args.heads}  gate pitch {args.gate_pitch}  L2 {first['l2']}  "
        f"PDL {'on' if first['enable_pdl'] else 'off'}  timer {timer}"
    )
    header = f"{'B':>4}{'tile':>6}{'call us':>10}{'GB/s':>8}"
    if args.baseline == "unfused":
        header += f"{'unfused us':>12}{'speedup':>9}"
    print(header)
    for r in records:
        line = f"{r['rows']:>4}{r['tile_rows']:>6}{r['call_us_median']:>10.2f}{r['call_gbps']:>8.0f}"
        if args.baseline == "unfused":
            line += f"{r['unfused_us_median']:>12.2f}{r['speedup_vs_unfused']:>9.2f}"
        print(line)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(),
                    "cuda": torch.version.cuda,
                    "timer": timer,
                    "rows": records,
                },
                handle,
                indent=1,
            )


if __name__ == "__main__":
    main()
