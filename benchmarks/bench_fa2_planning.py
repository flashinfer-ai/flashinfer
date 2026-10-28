# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Warm FA2 plan latency and completed plan + N runs; use on each revision."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

import flashinfer


def main():
    """Measure completed planning and preallocated-output runs against a CPU reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=1
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-per-plan", type=int, nargs="+", default=[0, 1, 8, 32])
    args = parser.parse_args()
    if (
        args.samples <= 0
        or args.output.exists()
        or any(n < 0 for n in args.runs_per_plan)
    ):
        parser.error("use positive samples and a fresh output filename")
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    dtype = getattr(torch, args.dtype)
    batch, nq, nk, heads, dim, page = args.batch_size, 256, 2048, 4, 128, 16
    q = torch.randn(batch * nq, heads, dim, device="cuda", dtype=dtype) / 5
    kv = (
        torch.randn(batch * nk // page, 2, page, heads, dim, device="cuda", dtype=dtype)
        / 5
    )
    metadata = dict(
        qo_indptr=torch.arange(batch + 1, device="cuda", dtype=torch.int32) * nq,
        paged_kv_indptr=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        * (nk // page),
        paged_kv_indices=torch.arange(
            batch * nk // page, device="cuda", dtype=torch.int32
        ),
        paged_kv_last_page_len=torch.full(
            (batch,), page, device="cuda", dtype=torch.int32
        ),
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim_qk=dim,
        page_size=page,
        causal=args.causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    # Snapshot independent FP32 reference on CPU before loading a native module.
    qcpu, kvcpu = q.cpu().float(), kv.cpu().float()
    refs, lses = [], []
    for request in range(batch):
        cache = kvcpu[request * nk // page : (request + 1) * nk // page]
        key, value = (cache[:, i].reshape(nk, heads, dim) for i in (0, 1))
        scores = torch.einsum(
            "qhd,khd->hqk", qcpu[request * nq : (request + 1) * nq], key
        ) / math.sqrt(dim)
        if args.causal:
            allowed = torch.arange(nk)[None, :] <= nk - nq + torch.arange(nq)[:, None]
            scores.masked_fill_(~allowed.unsqueeze(0), -torch.inf)
        refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
        lses.append(scores.logsumexp(-1).T / math.log(2))
    reference, reference_lse = torch.cat(refs), torch.cat(lses)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
    )
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")

    def run():
        """Run attention using the current plan and preallocated outputs."""
        wrapper.run(q, kv, out=out, lse=lse, return_lse=True)

    for _ in range(20):
        wrapper.plan(**metadata)
        run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu().float(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), reference_lse, rtol=1e-2, atol=1e-2)
    rows = []
    for runs in args.runs_per_plan:
        samples = []
        for _ in range(args.samples):
            run()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            wrapper.plan(**metadata)
            for _ in range(runs):
                run()
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1000)
        rows.append(
            dict(
                runs_per_plan=runs,
                median_us=statistics.median(samples),
                samples_us=samples,
            )
        )
    torch.testing.assert_close(out.cpu().float(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), reference_lse, rtol=1e-2, atol=1e-2)

    def tensor_sha(t):
        """Hash logical tensor bytes for paired input/reference/result audits."""
        return hashlib.sha256(
            t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    report = dict(
        completed=True,
        config={k: v for k, v in vars(args).items() if k != "output"},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        query_length=nq,
        kv_length=nk,
        heads=heads,
        dim=dim,
        page_size=page,
        input_sha256=dict(query=tensor_sha(q), kv=tensor_sha(kv)),
        reference_sha256=dict(
            output=tensor_sha(reference), lse=tensor_sha(reference_lse)
        ),
        output_sha256=dict(output=tensor_sha(out), lse=tensor_sha(lse)),
        independent_reference_passed=True,
        flashinfer_path=flashinfer.__file__,
        prefill_sha256=hashlib.sha256(
            Path(flashinfer.prefill.__file__).read_bytes()
        ).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "batch_prefill_with_kv_cache" in line and ".so" in line
            }
        },
        host_staging_used=hasattr(wrapper, "_fa2_host_metadata"),
        rows=rows,
        timing="Completed wall time in microseconds; fresh metadata planning and N preallocated-output runs, synchronized. Context priming, compilation, reference and validation excluded. N=0 measures plan only.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "batch": batch,
                "dtype": args.dtype,
                "rows": [
                    {"runs": x["runs_per_plan"], "us": x["median_us"]} for x in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
