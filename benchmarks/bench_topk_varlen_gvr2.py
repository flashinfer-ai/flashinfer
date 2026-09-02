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
"""Benchmark the gvr_2 (self-sampling GVR V2) top_k_varlen backend.

Compares, on identical fp32 inputs (gvr_2 is fp32-only):

* ``fi_gvr2``          — the new self-sampling GVR V2 backend (this PR)
* ``fi_gvr``           — existing GVR, load-balance path (production default)
* ``fi_gvr_nolb``      — existing GVR, single-CTA path
* ``fi_radix``         — CuTe DSL single-pass multi-CTA radix
* ``fi_radix_cutlass`` — masked CUTLASS radix fallback
* ``trtllm_gvr2``      — upstream TRT-LLM run_varlen (port-parity check;
                         needs a local TRT-LLM checkout, see --trtllm-path)

Timing: ``bench_gpu_time(..., use_cuda_graph=True)`` — decode top-k kernels
are microsecond-scale, so graph replay removes launch-overhead noise (same
method as bench_topk_varlen_vs_trtllm.py). Every backend's output is
validated (tie-aware value multiset per row) before timing; a wrong backend
is reported as WRONG and excluded from win statistics.

Hints: ``pre_idx`` mixes a ``--hint-quality`` fraction of the true top-K with
random valid indices (hint quality shifts GVR-family perf but never
exactness). ``--oracle`` uses perfect hints (GVR-favorable upper bound).

Example:
    python benchmarks/bench_topk_varlen_gvr2.py \
        --batch-sizes 1,16,64,256 --n-vals 8192,32768,131072 --top-k 1024
"""

import argparse
import math
import os
import sys

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time

# Directory holding TRT-LLM's top_k CuTe-DSL kernels (a local checkout's
# tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k), used only by the
# optional trtllm_gvr2 port-parity twin. No path ships as a default — pass
# --trtllm-path or set TRTLLM_TOPK_DIR.
_DEFAULT_TRTLLM_TOPK = os.environ.get("TRTLLM_TOPK_DIR", "")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sizes", type=str, default="1,4,16,32,64,128,256")
    p.add_argument("--n-vals", type=str, default="4096,8192,16384,32768,65536,131072")
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument(
        "--scenarios",
        type=str,
        default="uniform,mixed",
        help="comma list of: uniform (all rows = N), mixed (ragged in "
        "[K+1, N]), short (75%% of rows <= N/8)",
    )
    p.add_argument("--next-n", type=int, default=1)
    p.add_argument("--compress-ratio", type=int, default=1, choices=[1, 4])
    p.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="logits dtype; gvr_2/trtllm_gvr2 are fp32-only and are dropped "
        "for bf16/fp16",
    )
    p.add_argument("--hint-quality", type=float, default=0.6)
    p.add_argument("--oracle", action="store_true", help="perfect pre_idx hints")
    p.add_argument(
        "--backends",
        type=str,
        default="fi_gvr2,fi_gvr,fi_gvr_nolb,fi_radix,fi_radix_cutlass",
        help="comma list; add trtllm_gvr2 for the upstream port-parity twin",
    )
    p.add_argument("--trtllm-path", type=str, default=_DEFAULT_TRTLLM_TOPK)
    p.add_argument("--repeat-ms", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-refcheck", action="store_true")
    return p.parse_args()


def make_inputs(scenario, batch, N, K, nn, cr, hint_quality, oracle, seed, dtype):
    """Logits [batch*nn, N] (compressed space), uncompressed seq_lens."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = batch * nn
    logits = torch.randn(rows, N, generator=gen, device="cuda", dtype=torch.float32).to(
        dtype
    )
    if scenario == "uniform":
        seq_lens = torch.full((batch,), N * cr, dtype=torch.int32, device="cuda")
    elif scenario in ("mixed", "short"):
        # Smallest uncompressed kv whose EVERY next_n row is a long row
        # (n_r = (kv - nn + j + 1) // cr >= K + 1 for all j): keeps the
        # documented domains ("mixed" = rows in [K+1, N]) so no row silently
        # bypasses refcheck's n <= K skip.
        lo = (K + 1) * cr + nn - 1
        if scenario == "mixed":
            hi = N * cr + 1
        else:
            hi = N * cr // 8
            if hi <= lo:
                print(
                    f"[NOTE] short scenario is degenerate for N={N}, K={K}: "
                    "N/8 <= K+1, so the 'short' rows are not short",
                    flush=True,
                )
        hi = max(hi, lo + 1)
        seq_lens = torch.randint(lo, hi, (batch,), generator=gen, device="cuda").int()
        if scenario == "short":
            n_long = max(batch // 4, 1)
            seq_lens[:n_long] = N * cr
    else:
        raise ValueError(scenario)

    # request-level hints: hint_quality fraction of the true top-K of the
    # request's first row, rest random valid indices
    pre_idx = torch.zeros(batch, K, dtype=torch.int32, device="cuda")
    sl = seq_lens.cpu().tolist()
    for b in range(batch):
        n0 = min(max((sl[b] - nn + 1) // cr, 1), N)
        k_eff = min(K, n0)
        true_top = torch.topk(logits[b * nn, :n0], k_eff).indices.int()
        if oracle:
            pre_idx[b, :k_eff] = true_top
        else:
            n_hit = int(k_eff * hint_quality)
            pre_idx[b] = torch.randint(
                0, n0, (K,), generator=gen, device="cuda", dtype=torch.int32
            )
            pre_idx[b, :n_hit] = true_top[:n_hit]
    return logits, seq_lens, pre_idx


def make_backend_fns(backend, logits, seq_lens, pre_idx, K, nn, cr, trt_host):
    """Return a zero-arg closure running the backend into preallocated outputs,
    or None if the backend cannot run this config."""
    rows = logits.shape[0]
    out_i = torch.empty(rows, K, dtype=torch.int32, device="cuda")

    if backend in ("fi_gvr2", "fi_gvr", "fi_gvr_nolb", "trtllm_gvr2") and K not in (
        512,
        1024,
        2048,
    ):
        return None, None  # outside the GVR-family top_k domain: n/a, not an error

    if backend == "fi_gvr2":

        def fn():
            flashinfer.top_k_varlen(
                logits,
                seq_lens,
                K,
                pre_idx=pre_idx,
                next_n=nn,
                compress_ratio=cr,
                out_indices=out_i,
                backend="gvr_2",
            )

        return fn, out_i
    if backend in ("fi_gvr", "fi_gvr_nolb"):
        lb = backend == "fi_gvr"
        ws = None
        if lb:
            if seq_lens.shape[0] > 1024:
                return None, None  # LB prepare kernel cap
            mbs = 64
            while mbs < seq_lens.shape[0]:
                mbs *= 2
            ws = {
                "gvr_order_row": torch.empty(mbs, dtype=torch.int32, device="cuda"),
                "gvr_counters": torch.empty(2, dtype=torch.int32, device="cuda"),
            }

        def fn():
            flashinfer.top_k_varlen(
                logits,
                seq_lens,
                K,
                pre_idx=pre_idx,
                next_n=nn,
                compress_ratio=cr,
                out_indices=out_i,
                backend="gvr",
                load_balance=lb,
                workspace=ws,
            )

        return fn, out_i
    if backend == "fi_radix":

        def fn():
            flashinfer.top_k_varlen(
                logits,
                seq_lens,
                K,
                next_n=nn,
                compress_ratio=cr,
                out_indices=out_i,
                backend="radix",
            )

        return fn, out_i
    if backend == "fi_radix_cutlass":

        def fn():
            flashinfer.top_k_varlen(
                logits,
                seq_lens,
                K,
                next_n=nn,
                compress_ratio=cr,
                out_indices=out_i,
                backend="radix_cutlass",
            )

        return fn, out_i
    if backend == "trtllm_gvr2":
        if trt_host is None:
            return None, None

        msl = logits.shape[1] * cr

        def fn():
            trt_host.run_varlen(
                logits,
                pre_idx,
                seq_lens,
                out_i,
                next_n=nn,
                compress_ratio=cr,
                max_seq_len=msl,
            )

        return fn, out_i
    raise ValueError(backend)


def refcheck(logits, seq_lens, out_i, K, nn, cr):
    """Tie-aware per-row check; rows with n<=K only checked for -1 padding
    conventions loosely (backends differ there). Returns (ok, msg)."""
    rows = logits.shape[0]
    sl = seq_lens.cpu().tolist()
    for r in range(rows):
        n = min(max((sl[r // nn] - nn + (r % nn) + 1) // cr, 0), logits.shape[1])
        if n <= K:
            continue  # short-row pad conventions differ across backends
        idx = out_i[r].to(torch.int64)
        if int(idx.min()) < 0 or int(idx.max()) >= n:
            return False, f"row {r}: index out of range"
        if int(torch.unique(idx).numel()) != K:
            return False, f"row {r}: duplicate indices"
        got = torch.sort(logits[r].gather(0, idx) + 0.0, descending=True).values
        ref = torch.sort(
            torch.topk(logits[r, :n], K).values + 0.0, descending=True
        ).values
        if not torch.equal(got, ref):
            nbad = int((got != ref).sum())
            return False, f"row {r}: value multiset mismatch ({nbad}/{K} slots)"
    return True, ""


def geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    n_vals = [int(x) for x in args.n_vals.split(",")]
    scenarios = args.scenarios.split(",")
    backends = args.backends.split(",")
    K, nn, cr = args.top_k, args.next_n, args.compress_ratio
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    if args.dtype != "fp32":
        dropped = [b for b in backends if b in ("fi_gvr2", "trtllm_gvr2")]
        if dropped:
            print(f"[WARN] {dropped} are fp32-only; dropping for {args.dtype}")
            backends = [b for b in backends if b not in dropped]

    trt_host = None
    if "trtllm_gvr2" in backends:
        if not args.trtllm_path:
            print("[WARN] trtllm_gvr2 needs --trtllm-path or TRTLLM_TOPK_DIR; dropping")
            backends = [b for b in backends if b != "trtllm_gvr2"]
        else:
            sys.path.insert(0, args.trtllm_path)
            try:
                import gvr_topk_decode_self_sampling_host as trt_host  # noqa: N813
            except ImportError as e:
                print(f"[WARN] trtllm_gvr2 unavailable ({e}); dropping")
                backends = [b for b in backends if b != "trtllm_gvr2"]

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"dtype={args.dtype} top_k={K} next_n={nn} cr={cr} hint_quality="
        f"{'oracle' if args.oracle else args.hint_quality} "
        f"timing=cuda-graph median, repeat_ms={args.repeat_ms}"
    )
    header = f"{'scenario':<8} {'B':>4} {'N':>7} " + "".join(
        f"{b:>17}" for b in backends
    )
    print(header)
    print("-" * len(header))

    results = []  # (scenario, B, N, {backend: (us, ok)})
    for scenario in scenarios:
        for B in batch_sizes:
            for N in n_vals:
                if N <= K:
                    continue
                logits, seq_lens, pre_idx = make_inputs(
                    scenario,
                    B,
                    N,
                    K,
                    nn,
                    cr,
                    args.hint_quality,
                    args.oracle,
                    seed=args.seed + B * 7 + N // 64,
                    dtype=dtype,
                )
                row = {}
                for backend in backends:
                    fn, out_i = make_backend_fns(
                        backend, logits, seq_lens, pre_idx, K, nn, cr, trt_host
                    )
                    if fn is None:
                        row[backend] = (float("nan"), False, "n/a")
                        continue
                    try:
                        fn()  # warmup / JIT outside timing
                        torch.cuda.synchronize()
                        ok, msg = (True, "")
                        if not args.no_refcheck:
                            ok, msg = refcheck(logits, seq_lens, out_i, K, nn, cr)
                        timings = bench_gpu_time(
                            fn,
                            dry_run_time_ms=50,
                            repeat_time_ms=args.repeat_ms,
                            use_cuda_graph=True,
                            num_iters_within_graph=10,
                        )
                        med_us = float(torch.tensor(timings).median()) * 1e3
                        row[backend] = (med_us, ok, msg)
                    except Exception as e:  # noqa: BLE001 — report and move on
                        row[backend] = (float("nan"), False, f"{type(e).__name__}: {e}")
                results.append((scenario, B, N, row))
                cells = ""
                for b in backends:
                    us, ok, msg = row[b]
                    if math.isnan(us):
                        cells += f"{'—':>17}"
                    else:
                        tag = "" if ok else "!WRONG"
                        cells += f"{us:>11.1f}us{tag:<5}"
                print(f"{scenario:<8} {B:>4} {N:>7} {cells}", flush=True)
                for b in backends:
                    us, ok, msg = row[b]
                    if msg and msg != "n/a":
                        print(f"    [{b}] {msg}", flush=True)

    # ---- summary (relative to gvr_2 when present, else the first backend) ---
    if not backends:
        return
    base = "fi_gvr2" if "fi_gvr2" in backends else backends[0]
    print(f"\n================ SUMMARY (ratios are other/{base}; >1 = {base} faster)")
    for other in backends:
        if other == base:
            continue
        ratios, wins, total = [], 0, 0
        for _, _, _, row in results:
            g2, g2_ok, _ = row[base]
            ot, ot_ok, _ = row.get(other, (float("nan"), False, ""))
            if math.isnan(g2) or math.isnan(ot) or not (g2_ok and ot_ok):
                continue
            ratios.append(ot / g2)
            wins += ot > g2
            total += 1
        if total:
            print(
                f"  vs {other:<17} geomean {geomean(ratios):5.2f}x   "
                f"{base} wins {wins}/{total}"
            )
    print("\nper-config winner:")
    for scenario, B, N, row in results:
        valid = {b: us for b, (us, ok, _) in row.items() if ok and not math.isnan(us)}
        if valid:
            w = min(valid, key=valid.get)
            print(f"  {scenario:<8} B={B:<4} N={N:<7} -> {w} ({valid[w]:.1f}us)")


if __name__ == "__main__":
    main()
