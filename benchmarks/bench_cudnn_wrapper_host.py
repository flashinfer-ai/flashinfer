# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN wrapper host overhead, separate from graph GPU and cold plan costs.

BF16, Hq/Hkv=32/8, D=128, page size 16, fixed output buffers. Run the same
command on the baseline and candidate checkouts in separate processes:

    python benchmarks/bench_cudnn_wrapper_host.py --kind decode --batch 64 --kv 1024 --output decode.json
    python benchmarks/bench_cudnn_wrapper_host.py --kind ragged --batch 16 --q 512 --kv 512 --output ragged.json

Each result includes sampled independent math checks and bit-identical CUDA
graph replay after poisoning the output. Host enqueue excludes synchronization,
first-call compilation and output allocation; these are component measurements.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import cudnn
import flashinfer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["decode", "ragged"], required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--kv", type=int, required=True)
    p.add_argument("--layout", choices=["HND", "NHD"], default="NHD")
    p.add_argument("--backends", nargs="+", default=["cudnn"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.manual_seed(42)
    b, sq, sk, h, hk, d, page = a.batch, a.q, a.kv, 32, 8, 128, 16
    pages = (sk + page - 1) // page
    q = torch.randn(b * sq, h, d, dtype=torch.bfloat16, device="cuda")
    cache_nhd = torch.randn(b * pages, 2, page, hk, d, dtype=q.dtype, device=q.device)
    cache = (
        cache_nhd
        if a.layout == "NHD"
        else cache_nhd.permute(0, 1, 3, 2, 4).contiguous()
    )
    # Nonidentity page order catches code that drops page-table reconstruction.
    tables = torch.stack(
        [torch.randperm(pages, device="cuda") + i * pages for i in range(b)]
    ).int()
    indptr = torch.arange(b + 1, dtype=torch.int32) * pages
    indices = tables.flatten()
    last = torch.full((b,), sk - (pages - 1) * page, dtype=torch.int32)
    qo = torch.arange(b + 1, dtype=torch.int32) * sq
    if a.kind == "ragged":
        k_ragged = (
            cache_nhd[tables.long(), 0]
            .reshape(b, -1, hk, d)[:, :sk]
            .reshape(-1, hk, d)
            .contiguous()
        )
        v_ragged = (
            cache_nhd[tables.long(), 1]
            .reshape(b, -1, hk, d)[:, :sk]
            .reshape(-1, hk, d)
            .contiguous()
        )
    # Independent math reference for first/last requests and causal boundary rows.
    samples = [
        (bi, qi) for bi in sorted({0, b - 1}) for qi in sorted({0, sq // 2, sq - 1})
    ]
    references = []
    for bi, qi in samples:
        k = (
            cache_nhd[tables[bi].long(), 0]
            .reshape(-1, hk, d)[:sk]
            .float()
            .repeat_interleave(h // hk, 1)
        )
        v = (
            cache_nhd[tables[bi].long(), 1]
            .reshape(-1, hk, d)[:sk]
            .float()
            .repeat_interleave(h // hk, 1)
        )
        length = sk - sq + qi + 1
        scores = (
            torch.einsum("hd,shd->hs", q[bi * sq + qi].float(), k[:length]) / d**0.5
        )
        references.append(torch.einsum("hs,shd->hd", scores.softmax(-1), v[:length]))
    torch.cuda.synchronize()
    record = dict(
        args=vars(a),
        gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        cudnn=cudnn.backend_version(),
        fe_file=cudnn.__file__,
        fi_file=flashinfer.__file__,
        frost_env=os.getenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES"),
        results=[],
    )
    outputs = {}
    for backend in a.backends:
        row = dict(backend=backend, status="fail")
        try:
            ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            t0 = time.perf_counter_ns()
            if a.kind == "decode":
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    ws,
                    a.layout,
                    backend=backend,
                    use_tensor_cores=backend != "fa2" or sq > 1,
                )
                kw = dict(q_data_type=q.dtype, kv_data_type=q.dtype)
                if sq > 1:
                    kw["q_len_per_req"] = sq
                w.plan(indptr, indices, last, h, hk, d, page, **kw)
            else:
                w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    ws, "NHD", backend=backend
                )
                w.plan(
                    qo,
                    torch.arange(b + 1, dtype=torch.int32) * sk,
                    h,
                    hk,
                    d,
                    causal=True,
                    q_data_type=q.dtype,
                    kv_data_type=q.dtype,
                )
            row["plan_cpu_us"] = (time.perf_counter_ns() - t0) / 1e3
            out = torch.empty_like(q)
            run = (
                (lambda: w.run(q, k_ragged, v_ragged, out=out))
                if a.kind == "ragged"
                else (lambda: w.run(q, cache, out=out))
            )
            t0 = time.perf_counter_ns()
            run()
            torch.cuda.synchronize()
            row["first_run_wall_us"] = (time.perf_counter_ns() - t0) / 1e3
            for (bi, qi), ref in zip(samples, references, strict=True):
                torch.testing.assert_close(
                    out[bi * sq + qi].float(), ref, atol=0.01, rtol=0.01
                )
            if outputs:
                ref_name, ref = next(iter(outputs.items()))
                torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)
                row["full_reference_backend"] = ref_name
                row["full_max_abs"] = float((out.float() - ref.float()).abs().max())
            outputs[backend] = out.clone()
            prepared = getattr(w, "_cudnn_prepared", None)
            if prepared is not None:
                graph = prepared.graph
                idx = getattr(graph, "_plan_index", 0)
                row["fe_plan"] = graph.get_plan_name_at_index(idx)
            for _ in range(20):
                run()
            torch.cuda.synchronize()
            host = []
            for _ in range(100):
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                run()
                host.append((time.perf_counter_ns() - t0) / 1e3)
            torch.cuda.synchronize()
            row["host_enqueue_us"] = statistics.median(host)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    run()
            # Replay correctness is separate from eager, including poisoned output.
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, outputs[backend], atol=0, rtol=0)
            for _ in range(5):
                g.replay()
            ts = []
            for _ in range(20):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                g.replay()
                end.record()
                end.synchronize()
                ts.append(start.elapsed_time(end) * 1000 / 20)
            row.update(
                graph_gpu_us=statistics.median(ts),
                status="pass",
                replay="bit_identical",
                math_reference_rows=len(samples),
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            row["error"] = repr(exc)
        record["results"].append(row)
        print(json.dumps(row), flush=True)
        Path(a.output).write_text(json.dumps(record, indent=2) + "\n")
    if any(r["status"] != "pass" for r in record["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
