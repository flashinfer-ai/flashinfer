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
"""Benchmark compute_paged_mqa_logits_schedule through the public API.

This is the reproduction script for the schedule-kernel numbers quoted on the
#6855 PR.  bench_paged_mqa_logits.py warms the schedule outside its timing and
measures it inside a whole logits call, so it shows neither cost; this script
times the schedule alone and reports three numbers per batch size:

* ``first_ms``  -- wall-clock of the first call, batch sizes in ascending
  order in one process.  A size that needs a new compiled kernel pays the
  compile here; one that reuses a kernel built earlier in the run does not.
  Both on-disk caches are disabled (FlashInfer's CuTe-DSL kernel cache and
  the DSL's own file cache), so this is always a real compile, never a reload
  from an earlier run.  The first call also pays one-time DSL initialization,
  so batch 1 is timed first and reported on its own.
* ``kernel_us`` -- median device time of one schedule kernel (CUPTI when
  installed, else CUDA events), cold L2.
* ``eager_us``  -- wall-clock per back-to-back eager call with ``out=``
  preallocated and a warm L2: host validation and launch, or the kernel itself
  once it outlasts them.  At large batches this can be below kernel_us, which
  re-reads context_lens from DRAM on every launch.

The script uses only the public API, so the same file runs on any revision:
run it on the base and on the change for before/after numbers.

Examples:
    # default sweep on the current device (the first-call sweep takes ~30 s
    # on a revision that compiles per batch size)
    python benchmarks/bench_paged_mqa_logits_schedule.py

    # steady-state kernel time only, decode-range batches
    python benchmarks/bench_paged_mqa_logits_schedule.py --skip-compile --batch 1 32 256
"""

import argparse
import os

# Both caches are read before anything compiles, so set them first.  The
# DSL's own compiled-IR cache (default /tmp/$USER/cutlass_python_cache) reads
# its variable once, at first use.
os.environ["FLASHINFER_CUTE_DSL_DISABLE_CACHE"] = "1"
os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"

import statistics
import time

import torch

from flashinfer import compute_paged_mqa_logits_schedule
from flashinfer.attn_scores.attn_scores import _cached_dsl_targets_device
from flashinfer.testing import bench_gpu_time


def _make_seq_lens(batch, device):
    """Mixed lengths with a run of zero-length rows.

    Zero-length rows repeat prefix-sum values, the tie case the schedule's
    search has to count, so the self-check covers it on every run.  Seeded by
    batch so a size gets the same lengths in every sweep and on every revision.
    """
    gen = torch.Generator().manual_seed(batch)
    seq_lens = torch.randint(1, 65536, (batch,), dtype=torch.int32, generator=gen)
    seq_lens[batch // 3 : batch // 3 + min(40, batch // 3)] = 0
    return seq_lens.to(device)


def _check(seq_lens, out):
    # The CPU path is the reference implementation the GPU kernel replaced;
    # the two must agree bit for bit.  AssertionError aborts the sweep.
    ref = compute_paged_mqa_logits_schedule(seq_lens, use_gpu_kernel=False)
    if not torch.equal(ref, out):
        raise AssertionError(
            f"batch {seq_lens.shape[0]}: GPU schedule differs from the CPU path"
        )


def bench_first_call(batch, device):
    seq_lens = _make_seq_lens(batch, device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = compute_paged_mqa_logits_schedule(seq_lens)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3
    _check(seq_lens, out)
    return first_ms


def bench_one(batch, iters, device):
    seq_lens = _make_seq_lens(batch, device)
    # Warm: any compile happens here, outside timing.
    out = compute_paged_mqa_logits_schedule(seq_lens)
    torch.cuda.synchronize()
    _check(seq_lens, out)

    times = bench_gpu_time(
        lambda: compute_paged_mqa_logits_schedule(seq_lens, out=out),
        enable_cupti=True,
    )
    kernel_us = statistics.median(times) * 1e3

    # Eager wall-clock: the full public path, host dispatch included.
    for _ in range(10):
        compute_paged_mqa_logits_schedule(seq_lens, out=out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        compute_paged_mqa_logits_schedule(seq_lens, out=out)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / iters * 1e6
    return kernel_us, eager_us


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 128, 256, 512, 1024, 4096, 8192, 16384, 32768],
    )
    # 8159 and 8160 share a 32-row bucket and 8193 opens the next one: the
    # sizes #6855 reported, which show whether compiles track the bucket.
    p.add_argument(
        "--compile-batch",
        type=int,
        nargs="+",
        default=[64, 256, 1024, 4096, 8159, 8160, 8192, 8193, 16384, 32768],
    )
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    dev = torch.device("cuda", torch.cuda.current_device())
    if not _cached_dsl_targets_device(dev.index):
        raise SystemExit(
            "the GPU schedule kernel cannot target this device: the API would "
            "fall back to CPU and this script would time that instead"
        )
    print(
        f"device: {torch.cuda.get_device_name(dev)} cc={torch.cuda.get_device_capability(dev)}"
    )
    # CUDA context and allocator setup, so the first timed call pays neither.
    torch.zeros(1, device=dev).add_(1)
    torch.cuda.synchronize()

    if not args.skip_compile:
        print(f"{'batch':>6} {'first_ms':>10}")
        print(f"{1:>6} {bench_first_call(1, dev):>10.1f}  (with DSL initialization)")
        for b in sorted(args.compile_batch):
            print(f"{b:>6} {bench_first_call(b, dev):>10.1f}")
        print()

    print(f"{'batch':>6} {'kernel_us':>10} {'eager_us':>10}")
    for b in args.batch:
        kernel_us, eager_us = bench_one(b, args.iters, dev)
        print(f"{b:>6} {kernel_us:>10.2f} {eager_us:>10.2f}")


if __name__ == "__main__":
    main()
