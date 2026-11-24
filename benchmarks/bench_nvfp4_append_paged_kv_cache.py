# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVFP4 KV-cache append benchmark (``nvfp4_quantize_append_paged_kv_cache_with_slot_mapping``).

One decode-step write per row: ``nnz`` new tokens, each with ``num_kv_heads``
K and V heads, quantized to NVFP4 and scattered to random slots of an NHD
paged cache.  Timings are medians of ``bench_gpu_time`` with CUDA graphs.  To
compare two builds, run this script once per build in separate processes and
alternate them a few times; each run prints one line per ``nnz``.

``--l2`` picks one L2 policy, applied the same way by both timers:

* ``cold`` (default): no measured iteration finds its data in L2.  CUPTI
  flushes L2 before each iteration; ``--no-cupti`` times CUDA-graph replays
  that rotate over clones of every input and cache tensor.  When the cache is
  too large to clone (``bench_gpu_time`` then skips rotation and the few
  touched slots would stay in L2), ``--no-cupti`` uses CUDA events with an L2
  flush before each launch instead.  This is the conservative bound.
* ``warm``: the same buffers every iteration, so they stay in L2.  In a decode
  step the K/V input was just written by the preceding kernels and is
  resident; only the cache lines being written are cold.

CUPTI (default) needs ``cupti-python>=13``; without it the script stops instead
of letting ``bench_gpu_time`` fall back to another timer.  The header line
names the L2 policy of the run and every row names the timer it used.
"""

from __future__ import annotations

import argparse
import importlib.metadata

import numpy as np
import torch

from flashinfer.page import nvfp4_quantize_append_paged_kv_cache_with_slot_mapping
from flashinfer.testing.utils import bench_gpu_time, calculate_rotation_count

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
_CUPTI_MIN_MAJOR = 13  # the version bench_gpu_time requires before it uses CUPTI


def _require_cupti() -> None:
    try:
        from cupti import cupti  # noqa: F401

        installed = importlib.metadata.version("cupti-python")
    except (ImportError, importlib.metadata.PackageNotFoundError) as e:
        raise SystemExit(
            f"CUPTI timer unavailable ({e}). Install it with "
            f"'pip install -U cupti-python>={_CUPTI_MIN_MAJOR}' or pass --no-cupti."
        ) from e
    if int(installed.split(".")[0]) < _CUPTI_MIN_MAJOR:
        raise SystemExit(
            f"cupti-python {installed} is too old; bench_gpu_time needs "
            f">={_CUPTI_MIN_MAJOR}. Upgrade it or pass --no-cupti."
        )


def _run(
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        k,
        v,
        slots,
        (k_cache, v_cache),
        (k_scales, v_scales),
        global_scale,
        global_scale,
        kv_layout="NHD",
    )


def _inputs(args: argparse.Namespace, nnz: int) -> tuple[torch.Tensor, ...]:
    dtype = _DTYPES[args.dtype]
    k = torch.randn((nnz, args.num_kv_heads, args.head_dim), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    num_slots = args.num_pages * args.page_size
    slots = torch.randperm(num_slots, device="cuda", dtype=torch.int64)[:nnz]
    cache_rows = (args.num_pages, args.page_size, args.num_kv_heads)
    k_cache = torch.zeros(
        (*cache_rows, args.head_dim // 2), device="cuda", dtype=torch.uint8
    )
    v_cache = torch.zeros_like(k_cache)
    k_scales = torch.zeros(
        (*cache_rows, args.head_dim // 16), device="cuda", dtype=torch.float8_e4m3fn
    )
    v_scales = torch.zeros_like(k_scales)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    return k, v, slots, k_cache, v_cache, k_scales, v_scales, global_scale


def _timer(args: argparse.Namespace, inputs: tuple[torch.Tensor, ...]) -> str:
    if not args.no_cupti:
        return "cupti"
    if args.l2 == "cold" and calculate_rotation_count(list(inputs)) == 1:
        # The tensors exceed 5x L2, so the CUDA-graph timer would not rotate
        # them, and an append touching only nnz slots would find those lines in
        # L2 on every replay.  Flush L2 before each launch instead.
        return "cuda-event"
    return "cuda-graph"


def _median_us(args: argparse.Namespace, nnz: int) -> tuple[str, float]:
    inputs = _inputs(args, nnz)
    _run(*inputs)  # JIT-compile and load the kernel outside the timed region
    torch.cuda.synchronize()
    timer = _timer(args, inputs)
    # The tensors go in as input_args so the CUDA-graph timer can rotate clones
    # of them for a cold L2; CUPTI and the CUDA-event timer flush L2 instead.
    measurements = bench_gpu_time(
        _run,
        dry_run_time_ms=args.warmup_ms,
        repeat_time_ms=args.measure_ms,
        enable_cupti=timer == "cupti",
        use_cuda_graph=timer != "cuda-event",
        input_args=inputs,
        cold_l2_cache=args.l2 == "cold",
    )
    return timer, float(np.median(measurements)) * 1e3


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("num_kv_heads", "head_dim", "page_size", "num_pages", "measure_ms"):
        if getattr(args, name) < 1:
            flag = "--" + name.replace("_", "-")
            parser.error(f"{flag} must be >= 1, got {getattr(args, name)}")
    if args.warmup_ms < 0:
        parser.error(f"--warmup-ms must be >= 0, got {args.warmup_ms}")
    if args.head_dim % 16:
        parser.error(f"--head-dim must be a multiple of 16, got {args.head_dim}")
    num_slots = args.num_pages * args.page_size
    bad = [nnz for nnz in args.nnz if not 1 <= nnz <= num_slots]
    if bad:
        parser.error(
            f"--nnz must be in [1, num_pages * page_size = {num_slots}], got {bad}"
        )
    if not torch.cuda.is_available():
        parser.error("needs a CUDA device")


def main() -> None:
    parser = argparse.ArgumentParser(description="NVFP4 KV-cache append benchmark")
    parser.add_argument(
        "--nnz", type=int, nargs="+", default=[16, 64, 128, 160, 256, 512]
    )
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-pages", type=int, default=4096)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measure-ms", type=int, default=1000)
    parser.add_argument(
        "--no-cupti",
        action="store_true",
        help="time CUDA-graph replays instead of CUPTI (CUDA events with an L2 "
        "flush when --l2 cold and the cache is too large to rotate)",
    )
    parser.add_argument(
        "--l2",
        choices=("cold", "warm"),
        default="cold",
        help="cold: no iteration finds its data in L2 (default); warm: buffers stay in L2",
    )
    parser.add_argument("--label", default="", help="free text echoed on every row")
    args = parser.parse_args()
    _check_args(parser, args)
    if not args.no_cupti:
        _require_cupti()

    torch.manual_seed(0)
    print(
        f"# {torch.cuda.get_device_name()} {args.dtype} heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} page_size={args.page_size} "
        f"num_pages={args.num_pages} NHD l2={args.l2}"
    )
    for nnz in args.nnz:
        timer, us = _median_us(args, nnz)
        print(
            f"label={args.label} nnz={nnz} timer={timer} median_us={us:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
