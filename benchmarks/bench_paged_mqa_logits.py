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
"""

import argparse
import time

import torch

from flashinfer import (
    fp4_paged_mqa_logits,
    fp8_paged_mqa_logits,
    padded_seq_len,
)

_HEADS = 64  # fp4 pins num_heads=64/head_dim=128; fp8 is parametric --
_HEAD_DIM = 128  # bench both at the fp4 (DeepSeek indexer) shape


def _make_inputs(kind, batch, seq_len, next_n, block_size, device):
    """Random inputs in the exact layouts the public API requires."""
    ntb_cols = ((seq_len + 127) // 128 * 128) // block_size
    num_blocks = max(batch * ntb_cols, 1)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(batch * ntb_cols, dtype=torch.int32, device=device)
        .reshape(batch, ntb_cols)
        .contiguous()
    )
    weights = torch.randn(batch * next_n, _HEADS, dtype=torch.float32, device=device)
    if kind == "fp8":
        q = torch.randn(
            batch, next_n, _HEADS, _HEAD_DIM, dtype=torch.float32, device=device
        ).to(torch.float8_e4m3fn)
        kv = torch.randint(
            0,
            256,
            (num_blocks, block_size, 1, _HEAD_DIM + 4),
            dtype=torch.uint8,
            device=device,
        )
        return (q, kv, weights, block_tables, seq_lens, seq_len)
    q = torch.randint(
        0,
        256,
        (batch, next_n, _HEADS, _HEAD_DIM // 2),
        dtype=torch.uint8,
        device=device,
    )
    # UE8M0 exponents near 1.0 (bias 127) in each packed byte keep values finite.
    q_sf = torch.full(
        (batch, next_n, _HEADS), 0x7F7F7F7F, dtype=torch.int32, device=device
    )
    kv = torch.randint(
        0,
        256,
        (num_blocks, block_size, 1, _HEAD_DIM // 2 + 4),
        dtype=torch.uint8,
        device=device,
    )
    kv[:, :, :, _HEAD_DIM // 2 :] = 0x7F
    return (q, q_sf, kv, weights, block_tables, seq_lens, seq_len)


def _call(kind, args, out):
    if kind == "fp8":
        return fp8_paged_mqa_logits(*args, out=out)
    return fp4_paged_mqa_logits(*args, out=out)


def bench_one(kind, batch, seq_len, next_n, block_size, iters, device):
    args = _make_inputs(kind, batch, seq_len, next_n, block_size, device)
    out = torch.empty(
        (batch * next_n, padded_seq_len(seq_len)),
        dtype=torch.float32 if kind == "fp8" else torch.bfloat16,
        device=device,
    )

    # Warm: JIT compile + schedule-bucket compile happen here, outside timing.
    _call(kind, args, out)
    torch.cuda.synchronize()

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
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    dev = torch.device("cuda", torch.cuda.current_device())
    print(
        f"device: {torch.cuda.get_device_name(dev)} cc={torch.cuda.get_device_capability(dev)}"
    )
    print(
        f"{'kind':>4} {'batch':>5} {'seq_len':>7} {'next_n':>6} {'graph_ms':>10} {'eager_ms':>10}"
    )
    for kind in args.kind:
        for b in args.batch:
            for seq_len in args.seq_len:
                for nn in args.next_n:
                    try:
                        g_ms, e_ms = bench_one(
                            kind, b, seq_len, nn, args.block_size, args.iters, dev
                        )
                    except (ValueError, RuntimeError) as e:
                        print(
                            f"{kind:>4} {b:>5} {seq_len:>7} {nn:>6}  skipped: {str(e).splitlines()[0][:60]}"
                        )
                        continue
                    print(
                        f"{kind:>4} {b:>5} {seq_len:>7} {nn:>6} {g_ms:>10.4f} {e_ms:>10.4f}"
                    )


if __name__ == "__main__":
    main()
