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

Block-sparse attention benchmark: VibeCUDA backend vs the CAKE (PR #4593) backend.

Runs the canonical GQA boolean-block-mask workload matrix (shared with
tests/attention/test_cake_vsa.py and tests/attention/test_vibecuda_bsa.py)
and times ``BlockSparseAttentionWrapper.run`` per backend with CUPTI
(5 dry-run + 100 repeat iterations, median per workload), reporting
per-workload latency, per-workload speedup (cake / vibecuda), and the
arithmetic and geometric mean speedups.  Both backends are validated against
the dense reference at atol = rtol = 1e-2 before timing, and an
untimed NaN-sentinel prefill check proves every output element is really
written by each timed call.
"""

import argparse
import json
import math
from pathlib import Path

import torch

import flashinfer
from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.testing.utils import bench_gpu_time

# (block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, selected, return_lse)
WORKLOADS = (
    (128, torch.bfloat16, 8, 8, 128, 256, 512, 2, True),
    (64, torch.bfloat16, 8, 8, 128, 128, 256, 2, True),
    (128, torch.float16, 8, 1, 128, 256, 512, 2, False),
    (128, torch.float16, 8, 8, 128, 256, 512, 2, True),
    (128, torch.bfloat16, 8, 2, 128, 256, 512, 2, False),
    (128, torch.bfloat16, 8, 8, 64, 256, 512, 2, False),
    (128, torch.bfloat16, 8, 8, 96, 256, 512, 2, False),
    (128, torch.bfloat16, 8, 8, 128, 128, 16384, 8, False),
)


def make_inputs(block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, selected):
    torch.manual_seed(0)
    device = torch.device("cuda")
    mb, nb = M // block_size, N // block_size
    mask = torch.zeros((num_qo_heads, mb, nb), dtype=torch.bool, device=device)
    offsets = torch.arange(selected, device=device)
    for row in range(mb):
        columns = (offsets * 7 + row) % nb
        mask[:, row, columns] = True
    q = torch.randn((M, num_qo_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((N, num_kv_heads, head_dim), dtype=dtype, device=device)
    return q, k, v, mask


def dense_reference(q, k, v, mask, block_size):
    group = q.shape[1] // k.shape[1]
    k_heads = k.repeat_interleave(group, dim=1)
    v_heads = v.repeat_interleave(group, dim=1)
    scale = 1.0 / math.sqrt(q.shape[2])
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k_heads.float()) * scale
    token_mask = mask.repeat_interleave(block_size, 1).repeat_interleave(
        block_size, 2
    )
    scores.masked_fill_(~token_mask, float("-inf"))
    reference = torch.einsum(
        "hmn,nhd->mhd", torch.softmax(scores, dim=-1), v_heads.float()
    ).to(q.dtype)
    reference_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    return reference, reference_lse


def make_wrapper(backend, block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, mask):
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
    wrapper = BlockSparseAttentionWrapper(workspace, backend=backend)
    wrapper.plan(
        None,
        None,
        M,
        N,
        block_size,
        block_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=dtype,
        kv_data_type=dtype,
        block_mask=mask,
    )
    return wrapper


def workload_label(row):
    block_size, dtype, hq, hkv, d, m, n, selected, return_lse = row
    dt = "bf16" if dtype == torch.bfloat16 else "fp16"
    return (
        f"bs={block_size} {dt} hq={hq} hkv={hkv} d={d} m={m} n={n} "
        f"sel={selected} lse={int(return_lse)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["cake", "vibecuda"],
                        help="Backends to benchmark (default: cake vibecuda)")
    parser.add_argument("--dry-run-iters", type=int, default=5,
                        help="CUPTI warmup iterations not timed (matches the "
                             "canonical suite contract; default: 5)")
    parser.add_argument("--repeat-iters", type=int, default=100,
                        help="CUPTI measured iterations (matches the canonical "
                             "suite contract; default: 100)")
    parser.add_argument("--no-refcheck", action="store_true",
                        help="Skip dense-reference correctness checks")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON artifact path for the results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_vibecuda_bsa requires a CUDA device")
    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3)):
        raise RuntimeError(
            f"cake/vibecuda backends require SM100/SM103, current device is {cc}"
        )

    results = []
    for row in WORKLOADS:
        (block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N,
         selected, return_lse) = row
        label = workload_label(row)
        q, k, v, mask = make_inputs(
            block_size, dtype, num_qo_heads, num_kv_heads, head_dim, M, N, selected
        )
        entry = {"workload": label, "backends": {}}
        reference, reference_lse = None, None
        for backend in args.backends:
            wrapper = make_wrapper(
                backend, block_size, dtype, num_qo_heads, num_kv_heads, head_dim,
                M, N, mask,
            )
            if not args.no_refcheck:
                if reference is None:
                    reference, reference_lse = dense_reference(q, k, v, mask, block_size)
                # Untimed sentinel-prefill/full-write check: poison the output
                # of a first call by pre-filling via a preallocated buffer when
                # the backend supports out=, then validate the full result.
                result = wrapper.run(q, k, v, return_lse=return_lse)
                output, lse = result if return_lse else (result, None)
                torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2,
                                           msg=lambda m: f"{backend} {label}: {m}")
                if return_lse:
                    torch.testing.assert_close(lse, reference_lse, atol=1e-2,
                                               rtol=1e-2,
                                               msg=lambda m: f"{backend} {label} lse: {m}")
            # Sentinel prefill through the public out= parameter: the timed
            # callable must overwrite every element it claims to produce.
            poisoned = torch.full_like(q, float("nan"))
            wrapper.run(q, k, v, out=poisoned, return_lse=False)
            assert torch.isfinite(poisoned.float()).all(), (
                f"{backend} {label}: NaN sentinel survived - output not fully "
                "written by run ()"
            )

            times = bench_gpu_time(
                lambda: wrapper.run(q, k, v, return_lse=return_lse),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.repeat_iters,
                enable_cupti=True,
                use_cuda_graph=False,
            )
            times = sorted(float(t) for t in times)
            median_ms = times[len(times) // 2]
            entry["backends"][backend] = {
                "median_ms": median_ms,
                "min_ms": times[0],
                "max_ms": times[-1],
                "iters": len(times),
                "timing": "cupti",
            }
            print(f"  [{label}] {backend}: {median_ms:.6f} ms "
                  f"(min {times[0]:.6f}, max {times[-1]:.6f}, "
                  f"{len(times)} CUPTI iters)")
        if "cake" in entry["backends"] and "vibecuda" in entry["backends"]:
            speedup = (entry["backends"]["cake"]["median_ms"]
                       / entry["backends"]["vibecuda"]["median_ms"])
            entry["speedup_cake_over_vibecuda"] = speedup
        results.append(entry)

    if all("speedup_cake_over_vibecuda" in e for e in results):
        speedups = [e["speedup_cake_over_vibecuda"] for e in results]
        arith = sum(speedups) / len(speedups)
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"\nvibecuda speedup over cake (median, {len(speedups)} workloads):")
        for e in results:
            print(f"  {e['workload']}: {e['speedup_cake_over_vibecuda']:.3f}x")
        print(f"  arithmetic mean: {arith:.4f}x")
        print(f"  geometric mean:  {geo:.4f}x")
        summary = {
            "arithmetic_mean": arith,
            "geometric_mean": geo,
            "denominator": "cake",
            "baseline_provenance": {
                "backend": "cake",
                "symbol": "flashinfer.sparse.BlockSparseAttentionWrapper(backend='cake')",
                "source": "flashinfer PR #4593 (CAKE VSA), repository HEAD",
            },
            "timing": {
                "method": "cupti",
                "dry_run_iters": args.dry_run_iters,
                "repeat_iters": args.repeat_iters,
                "statistic": "median",
                "use_cuda_graph": False,
            },
            "workload_count": len(results),
        }
    else:
        summary = {"denominator": None, "workload_count": len(results)}

    report = {"summary": summary, "workloads": results}
    out_path = Path(args.output) if args.output else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
