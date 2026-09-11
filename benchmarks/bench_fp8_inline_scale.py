"""
Copyright (c) 2024 by FlashInfer team.

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

"""Performance comparison of FA2 FP8 vs 16-bit KV-cache dtypes across all entry points.

For every (mode, mha/gqa, q/o dtype, head_dim, kv_len) this benchmarks a 3-way
comparison of KV-cache configurations (Q/O dtype is fp16 or bf16):

  * ``fp8_tensor`` - FP8 KV cache, per-tensor scale (the existing FP8 path).
  * ``fp8_inline`` - FP8 KV cache, per-(token, head) inline scale
                     (``use_inline_sf=True``; the feature under test).
  * ``kv_eq_qo``   - KV cache dtype == Q/O dtype (FP16 KV if Q/O is FP16, BF16 KV
                     if Q/O is BF16).

Two ratios are reported per row:

  * ``inl/tensor`` = fp8_inline / fp8_tensor -> overhead of the inline-scale
                     feature vs per-tensor FP8 (~1.0 means no regression).
  * ``inl/qo``     = fp8_inline / kv_eq_qo   -> how the FP8 inline-scale path
                     stacks up against the matching 16-bit KV cache
                     (<1 means FP8 is faster).

Modes (the 5 public entry points, covering prefill/decode x ragged/paged x
single/batch for every combination that has an API):

  * ``single_prefill_ragged``  - single_prefill_with_kv_cache
  * ``batch_prefill_ragged``   - BatchPrefillWithRaggedKVCacheWrapper
  * ``batch_prefill_paged``    - BatchPrefillWithPagedKVCacheWrapper
  * ``single_decode_ragged``   - single_decode_with_kv_cache
  * ``batch_decode_paged``     - BatchDecodeWithPagedKVCacheWrapper

Q/O dtype: FP16 on all hardware; BF16 only on SM80+ (skipped below SM80).

Notes on sm75 (e.g. T600):
  * No FP8 tensor cores -> FP8 KV is software-dequantized before the MMA; the
    dequant, not the inline scale, dominates the FP8 cost.

Timing: uses ``bench_gpu_time`` with CUPTI (pure GPU kernel time, most accurate).
Pass ``--no-cupti`` to force CUDA-event timing.

Usage:
    python benchmarks/bench_fp8_inline_scale.py
    python benchmarks/bench_fp8_inline_scale.py --head-dims 128 256 --kv-lens 512 2048
    python benchmarks/bench_fp8_inline_scale.py --modes single_decode_ragged batch_decode_paged
"""

import argparse
import subprocess
import sys

import torch

# Self-contained worker run in a subprocess (one cell). It builds the tensors for
# the requested (mode, gqa, q_dtype, config), benchmarks the config, and prints
# "RESULT <median_ms>". Running each cell in a fresh process isolates uncatchable
# CUDA crashes (SIGFPE / launch failure) so a crashing config cannot kill the
# whole benchmark.
_WORKER_CODE = r"""
import sys
import numpy as np
import torch
import flashinfer
from flashinfer.testing.utils import bench_gpu_time

WS = 128 * 1024 * 1024  # 128 MiB workspace for the batch wrappers


def make_slot(x_ref, dtype, head_dim):
    fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
    scale = (x_ref.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_ref / scale).to(dtype)
    slot_size = head_dim + 16
    slot = torch.zeros(*x_ref.shape[:-1], slot_size, dtype=torch.uint8, device=x_ref.device)
    slot[..., :head_dim] = x_fp8.view(torch.uint8)
    slot[..., head_dim : head_dim + 4] = scale.to(torch.float32).view(torch.uint8)
    return slot.view(dtype)


def main():
    (mode, gqa, q_dtype, config, head_dim, kv_len, batch_size, page_size,
     num_kv_heads, iters, cupti) = (
        sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4],
        int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]),
        int(sys.argv[9]), int(sys.argv[10]), sys.argv[11] == "1",
    )
    dev = "cuda"
    q_t = torch.float16 if q_dtype == "fp16" else torch.bfloat16
    fp8 = torch.float8_e4m3fn
    is_inline = config == "fp8_inline"
    is_fp8 = config in ("fp8_tensor", "fp8_inline")
    num_qo = num_kv_heads if not gqa else 2 * num_kv_heads

    def conv(x_ref):
        # Apply the KV config to a reference tensor.
        if is_inline:
            return make_slot(x_ref, fp8, head_dim)
        if is_fp8:
            return (x_ref / 10).to(fp8)
        return x_ref

    def ragged_kv(n):
        k = torch.randn(n, num_kv_heads, head_dim, dtype=q_t, device=dev)
        v = torch.randn(n, num_kv_heads, head_dim, dtype=q_t, device=dev)
        return conv(k), conv(v)

    def paged_kv():
        num_pages = batch_size * ((kv_len + page_size - 1) // page_size)
        k = torch.randn(num_pages, num_kv_heads, page_size, head_dim, dtype=q_t, device=dev)
        v = torch.randn(num_pages, num_kv_heads, page_size, head_dim, dtype=q_t, device=dev)
        kk, vv = conv(k), conv(v)
        return torch.cat([kk.unsqueeze(1), vv.unsqueeze(1)], dim=1)  # [pages,2,nk,ps,kv_last] HND

    def paged_idx():
        ind, idx, last = [0], [], []
        for _ in range(batch_size):
            npg = (kv_len + page_size - 1) // page_size
            ind.append(ind[-1] + npg)
            idx.extend(range(ind[-1] - npg, ind[-1]))
            last.append(kv_len % page_size or page_size)
        t = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
        return t(ind), t(idx), t(last)

    if mode == "single_prefill_ragged":
        qo_len = kv_len
        q = torch.randn(qo_len, num_qo, head_dim, dtype=q_t, device=dev)
        k, v = ragged_kv(kv_len)
        kw = dict(causal=False, backend="fa2")
        if is_inline:
            kw["use_inline_sf"] = True
        fn = lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, **kw)
    elif mode == "batch_prefill_ragged":
        qo_len = kv_len
        qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=dev) * qo_len
        kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=dev) * kv_len
        total_qo, total_kv = batch_size * qo_len, batch_size * kv_len
        q = torch.randn(total_qo, num_qo, head_dim, dtype=q_t, device=dev)
        k, v = ragged_kv(total_kv)
        ws = torch.empty(WS, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
        plan_kw = dict(causal=False, q_data_type=q_t, kv_data_type=fp8 if is_fp8 else q_t)
        if is_inline:
            plan_kw["use_inline_sf"] = True
        w.plan(qo_indptr, kv_indptr, num_qo, num_kv_heads, head_dim, **plan_kw)
        fn = lambda: w.run(q, k, v)
    elif mode == "batch_prefill_paged":
        qo_len = kv_len
        qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=dev) * qo_len
        total_qo = batch_size * qo_len
        q = torch.randn(total_qo, num_qo, head_dim, dtype=q_t, device=dev)
        pkv = paged_kv()
        pki, pks, pkl = paged_idx()
        ws = torch.empty(WS, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="HND", backend="fa2")
        plan_kw = dict(causal=False, q_data_type=q_t, kv_data_type=fp8 if is_fp8 else q_t)
        if is_inline:
            plan_kw["use_inline_sf"] = True
        w.plan(qo_indptr, pki, pks, pkl, num_qo, num_kv_heads, head_dim, page_size, **plan_kw)
        fn = lambda: w.run(q, pkv)
    elif mode == "single_decode_ragged":
        q = torch.randn(num_qo, head_dim, dtype=q_t, device=dev)
        k, v = ragged_kv(kv_len)
        kw = {}
        if is_inline:
            kw["use_inline_sf"] = True
        fn = lambda: flashinfer.single_decode_with_kv_cache(q, k, v, **kw)
    elif mode == "batch_decode_paged":
        q = torch.randn(batch_size, num_qo, head_dim, dtype=q_t, device=dev)
        pkv = paged_kv()
        pki, pks, pkl = paged_idx()
        ws = torch.empty(WS, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
        plan_kw = dict(q_data_type=q_t, kv_data_type=fp8 if is_fp8 else q_t)
        if is_inline:
            plan_kw["use_inline_sf"] = True
        w.plan(pki, pks, pkl, num_qo, num_kv_heads, head_dim, page_size, **plan_kw)
        fn = lambda: w.run(q, pkv)
    else:
        raise ValueError("unknown mode: " + mode)

    t = float(np.median(bench_gpu_time(fn, repeat_iters=iters, enable_cupti=cupti)))
    print("RESULT " + f"{t:.6f}", flush=True)


main()
"""

_MODES = [
    "single_prefill_ragged",
    "batch_prefill_ragged",
    "batch_prefill_paged",
    "single_decode_ragged",
    "batch_decode_paged",
]
_MODE_DISPLAY = {
    "single_prefill_ragged": "prefill-single-ragged",
    "batch_prefill_ragged": "prefill-batch-ragged",
    "batch_prefill_paged": "prefill-batch-paged",
    "single_decode_ragged": "decode-single-ragged",
    "batch_decode_paged": "decode-batch-paged",
}
_CONFIGS = ["fp8_tensor", "fp8_inline", "kv_eq_qo"]


def _bench_one(
    mode,
    gqa,
    q_dtype,
    config,
    head_dim,
    kv_len,
    batch_size,
    page_size,
    num_kv_heads,
    iters,
    cupti,
    timeout,
):
    """Benchmark one cell in a subprocess. Returns median_ms, or None if the
    config crashes / times out (a CUDA crash kills the child, not this process)."""
    argv = [
        sys.executable,
        "-c",
        _WORKER_CODE,
        mode,
        "1" if gqa else "0",
        q_dtype,
        config,
        str(head_dim),
        str(kv_len),
        str(batch_size),
        str(page_size),
        str(num_kv_heads),
        str(iters),
        "1" if cupti else "0",
    ]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _f_us(t):
    return f"{t * 1000:.1f}" if t is not None else "n/a"


def _f_ratio(a, b):
    if a is None or b is None or b == 0:
        return "n/a"
    return f"{a / b:.3f}x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dims", type=int, nargs="+", default=[128])
    ap.add_argument("--kv-lens", type=int, nargs="+", default=[512])
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-cell subprocess timeout in seconds (a cell that "
        "exceeds it is reported as n/a)",
    )
    ap.add_argument("--no-cupti", action="store_true", help="force CUDA-event timing")
    ap.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=_MODES,
        help="subset of modes to benchmark (see --help / docstring)",
    )
    args = ap.parse_args()

    cc = torch.cuda.get_device_capability(0)
    cupti = not args.no_cupti
    # BF16 Q/O dtype only on SM80+ (skipped below SM80).
    q_dtypes = ["fp16", "bf16"] if cc[0] >= 8 else ["fp16"]

    print(f"GPU: {torch.cuda.get_device_name(0)}  cc={cc}")
    print(f"iters={args.iters}  cupti={cupti}  q_dtypes={q_dtypes}")
    print(
        f"num_kv_heads={args.num_kv_heads} (mha: qo={args.num_kv_heads}, "
        f"gqa: qo={2 * args.num_kv_heads})  batch={args.batch_size}  page={args.page_size}"
    )
    print()

    for head_dim in args.head_dims:
        for kv_len in args.kv_lens:
            print(f"===== head_dim={head_dim} kv_len={kv_len} =====")
            print(
                f"{'mode':>22} {'gqa':>4} {'q_dtype':>7} | "
                f"{'fp8_tensor':>11} {'fp8_inline':>11} {'kv_eq_qo':>10} | "
                f"{'inl/tensor':>11} {'inl/qo':>9}"
            )
            print("-" * 92)
            for mode in args.modes:
                for gqa in [False, True]:
                    for q_dtype in q_dtypes:
                        times = {
                            config: _bench_one(
                                mode,
                                gqa,
                                q_dtype,
                                config,
                                head_dim,
                                kv_len,
                                args.batch_size,
                                args.page_size,
                                args.num_kv_heads,
                                args.iters,
                                cupti,
                                args.timeout,
                            )
                            for config in _CONFIGS
                        }
                        gqa_s = "gqa" if gqa else "mha"
                        print(
                            f"{_MODE_DISPLAY[mode]:>22} {gqa_s:>4} {q_dtype:>7} | "
                            f"{_f_us(times['fp8_tensor']):>11} "
                            f"{_f_us(times['fp8_inline']):>11} "
                            f"{_f_us(times['kv_eq_qo']):>10} | "
                            f"{_f_ratio(times['fp8_inline'], times['fp8_tensor']):>11} "
                            f"{_f_ratio(times['fp8_inline'], times['kv_eq_qo']):>9}"
                        )
            print()


if __name__ == "__main__":
    main()
