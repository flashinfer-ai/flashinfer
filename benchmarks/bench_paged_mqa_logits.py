# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark fp8_paged_mqa_logits / fp4_paged_mqa_logits through the public API.

This is the reproduction script for the performance numbers quoted on the
paged-MQA PRs (e.g. #4737): sweep a (kind, batch, seq_len, next_n) grid on the
current GPU and report two numbers per cell:

* ``graph_ms``  -- CUDA-graph replay time of one call (out= preallocated, the
  schedule computed inside the capture).  This isolates device work and is the
  methodology behind the quoted speedups.
* ``eager_ms``  -- wall-clock per eager call, everything included (host
  dispatch, reshapes, schedule computation).  These kernels run only a few
  microseconds, so eager calls are host-launch-bound and this number can be
  several times graph_ms; a serving stack that launches eagerly should look at
  this one.

The fp4 next_n=4 atom decomposition is a fixed internal rule (direct on Rubin,
two atoms of 2 on SM100/SM103) chosen from a forced-decomposition sweep run
with this script's timing methodology on B100 and Rubin -- see the policy
comment in flashinfer/attn_scores/attn_scores.py.  This script benchmarks what
ships; ablating other decompositions requires a locally patched tree.

Examples:
    # default grid on the current device
    python benchmarks/bench_paged_mqa_logits.py

    # the B200-vs-Rubin serving shape used in PR #4737
    python benchmarks/bench_paged_mqa_logits.py --batch 64 --seq-len 16384

    # a GLM-style fp8 indexer geometry (num_heads=32, six draft positions)
    python benchmarks/bench_paged_mqa_logits.py --kind fp8 --num-heads 32 --next-n 6
"""

import argparse
import time

import torch

from flashinfer import (
    fp4_paged_mqa_logits,
    fp8_paged_mqa_logits,
    min_block_table_width,
    padded_seq_len,
)

_FP4_HEADS = 64  # fp4 pins num_heads=64/head_dim=128; fp8 is parametric --
_HEAD_DIM = 128  # bench both at the fp4 (DeepSeek indexer) shape
_FP4_SF_BYTES_PER_TOKEN = _HEAD_DIM // 32  # one UE8M0 per 32-element group


def _make_inputs(kind, batch, seq_len, next_n, block_size, device, num_heads):
    """Random *finite* inputs in the fused layouts the public API requires.

    kv_fused is flat per block -- [all value bytes][all scale bytes] -- NOT
    per-token [value|scale] rows; the 4-D shape is only a size contract (see
    the kv_fused docs on fp8_paged_mqa_logits / fp4_paged_mqa_logits).  The
    two regions are therefore built separately and concatenated, exactly as
    the API docstring example does.  Timing is data-independent, but finite
    inputs let bench_one assert finite logits, which catches a wrong layout.
    """
    ntb_cols = min_block_table_width(seq_len, block_size)  # natural width
    num_blocks = max(batch * ntb_cols, 1)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(batch * ntb_cols, dtype=torch.int32, device=device)
        .reshape(batch, ntb_cols)
        .contiguous()
    )
    weights = torch.randn(batch * next_n, num_heads, dtype=torch.float32, device=device)
    if kind == "fp8":
        q = torch.randn(
            batch, next_n, num_heads, _HEAD_DIM, dtype=torch.float32, device=device
        ).to(torch.float8_e4m3fn)
        # Region 1: finite e4m3fn values (randn -> fp8 never produces the NaN
        # codes 0x7F/0xFF that random bytes would).  Region 2: one finite
        # positive float32 scale per token.
        kv_vals = torch.randn(
            num_blocks, block_size, _HEAD_DIM, dtype=torch.float32, device=device
        ).to(torch.float8_e4m3fn)
        kv_scales = (
            torch.rand(num_blocks, block_size, dtype=torch.float32, device=device) + 0.5
        )
        kv = torch.cat(
            [
                kv_vals.view(torch.uint8).flatten(1),
                kv_scales.view(torch.uint8).flatten(1),
            ],
            dim=1,
        ).view(num_blocks, block_size, 1, _HEAD_DIM + 4)
        return (q, kv, weights, block_tables, seq_lens, seq_len)
    # Every E2M1 nibble is finite, so random bytes are valid packed FP4 values.
    q = torch.randint(
        0,
        256,
        (batch, next_n, num_heads, _HEAD_DIM // 2),
        dtype=torch.uint8,
        device=device,
    )
    # UE8M0 exponent 0x7F == 2^(127-127) == 1.0 in every scale byte keeps the
    # block-scaled MMA finite (0xFF is NaN; >= 0xF5 overflows the fp32 acc).
    q_sf = torch.full(
        (batch, next_n, num_heads), 0x7F7F7F7F, dtype=torch.int32, device=device
    )
    kv_vals = torch.randint(
        0,
        256,
        (num_blocks, block_size * (_HEAD_DIM // 2)),
        dtype=torch.uint8,
        device=device,
    )
    kv_sf = torch.full(
        (num_blocks, block_size * _FP4_SF_BYTES_PER_TOKEN),
        0x7F,
        dtype=torch.uint8,
        device=device,
    )
    kv = torch.cat([kv_vals, kv_sf], dim=1).view(
        num_blocks, block_size, 1, _HEAD_DIM // 2 + _FP4_SF_BYTES_PER_TOKEN
    )
    return (q, q_sf, kv, weights, block_tables, seq_lens, seq_len)


def _call(kind, args, out):
    if kind == "fp8":
        return fp8_paged_mqa_logits(*args, out=out)
    return fp4_paged_mqa_logits(*args, out=out)


def bench_one(kind, batch, seq_len, next_n, block_size, iters, device, num_heads):
    args = _make_inputs(kind, batch, seq_len, next_n, block_size, device, num_heads)
    out = torch.empty(
        (batch * next_n, padded_seq_len(seq_len)),
        dtype=torch.float32 if kind == "fp8" else torch.bfloat16,
        device=device,
    )

    # Warm: JIT compile + schedule-bucket compile happen here, outside timing.
    _call(kind, args, out)
    torch.cuda.synchronize()
    # The kernels compute every column in [0, seq_len) (the caller masks the
    # causal tail), so finite inputs must give finite logits there; columns
    # past seq_len up to the padded pitch are scratch.  A non-finite result
    # means _make_inputs drifted from the API's fused layout -- abort loudly
    # (AssertionError is not swallowed by the sweep's skip handling).
    if not torch.isfinite(out[:, :seq_len]).all():
        raise AssertionError(
            f"{kind} bench inputs produced non-finite logits -- _make_inputs is "
            "no longer in the API's fused kv layout"
        )

    # Graph replay: device work only (the schedule recompute is captured too).
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _call(kind, args, out)
    g.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    reps = []
    for _ in range(5):
        start.record()
        for _ in range(iters):
            g.replay()
        stop.record()
        torch.cuda.synchronize()
        reps.append(start.elapsed_time(stop) / iters)
    graph_ms = sorted(reps)[len(reps) // 2]

    # Eager wall-clock: the full public path, host dispatch included.
    for _ in range(10):
        _call(kind, args, out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _call(kind, args, out)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / iters * 1e3
    return graph_ms, eager_ms


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=["fp8", "fp4"], nargs="+", default=["fp8", "fp4"])
    p.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--seq-len", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--next-n", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[_FP4_HEADS],
        help="fp8 is parametric in num_heads (e.g. 32 with --next-n 6 for a "
        "GLM-style indexer); fp4 requires 64 and other values are skipped for it.",
    )
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    dev = torch.device("cuda", torch.cuda.current_device())
    print(
        f"device: {torch.cuda.get_device_name(dev)} cc={torch.cuda.get_device_capability(dev)}"
    )
    print(
        f"{'kind':>4} {'heads':>5} {'batch':>5} {'seq_len':>7} {'next_n':>6} "
        f"{'graph_ms':>10} {'eager_ms':>10}"
    )
    for kind in args.kind:
        for heads in args.num_heads:
            if kind == "fp4" and heads != _FP4_HEADS:
                print(f"{kind:>4} {heads:>5}  skipped: fp4 requires num_heads == 64")
                continue
            for b in args.batch:
                for seq_len in args.seq_len:
                    for nn in args.next_n:
                        try:
                            g_ms, e_ms = bench_one(
                                kind,
                                b,
                                seq_len,
                                nn,
                                args.block_size,
                                args.iters,
                                dev,
                                heads,
                            )
                        except (ValueError, RuntimeError) as e:
                            print(
                                f"{kind:>4} {heads:>5} {b:>5} {seq_len:>7} {nn:>6}  "
                                f"skipped: {str(e).splitlines()[0][:60]}"
                            )
                            continue
                        print(
                            f"{kind:>4} {heads:>5} {b:>5} {seq_len:>7} {nn:>6} "
                            f"{g_ms:>10.4f} {e_ms:>10.4f}"
                        )


if __name__ == "__main__":
    main()
