# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN wrapper host overhead, separate from graph GPU and cold plan costs.

Defaults: BF16, D=128, page size 16, fixed output buffers; Hq/Hkv defaults to 32/8. Run the same
command on the baseline and candidate checkouts in separate processes:

    python benchmarks/bench_cudnn_wrapper_host.py --kind decode --batch 64 --kv 1024 --output decode.json
    python benchmarks/bench_cudnn_wrapper_host.py --kind ragged --batch 16 --q 512 --kv 512 --output ragged.json
    python benchmarks/bench_cudnn_wrapper_host.py --kind paged --batch 16 --q 128 --kv 8192 --replan --output paged.json

For ragged planning, compare --replan --ragged-indptr gpu against
--replan --ragged-indptr gpu_with_cpu_mirrors with otherwise identical arguments.
Both bind the same kind of GPU prefixes; only the host readback differs.

Paged prefill uses pinned CPU prefixes and KV lengths, without caller-supplied
block tables or length maxima, matching the ordinary vLLM paged plan call.
Input pinning is outside the measured plan cost.
Prefill defaults to causal attention. Use --no-causal to compare noncausal
prefill, including its independent reference; decode remains causal.

Each result includes sampled independent math checks and one-ULP CUDA
graph replay after poisoning the output. Host enqueue excludes synchronization,
first-call compilation and output allocation; these are component measurements.
All backends are prepared before sampling. Each metric alternates forward/reverse
backend order per round; JSON records the actual order, including failed samples.
Workspaces and graph buffers for all requested backends remain live together.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import cudnn
import flashinfer


def _source_revision():
    """Identify the imported checkout, without mistaking a wheel's parent repo for it."""
    package = Path(flashinfer.__file__).resolve().parent
    try:

        def git(*args):
            return (
                subprocess.check_output(
                    ["git", "-C", str(package), *args], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

        root = Path(git("rev-parse", "--show-toplevel"))
        if root / "flashinfer" == package:
            return dict(
                commit=git("rev-parse", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--", ".")),
            )
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


def _interleave_measurements(measurements, executed_order):
    """Advance every backend to a metric barrier, then alternate sample order."""
    active = dict(measurements)
    while active:
        pending = {}
        for name, measurement in list(active.items()):
            try:
                pending[name] = next(measurement)
            except StopIteration:
                del active[name]
        if not pending:
            break
        phases = {(phase, count) for phase, count, _ in pending.values()}
        if len(phases) != 1:
            raise RuntimeError(f"Backend measurement phases differ: {phases}")
        phase, count = phases.pop()
        names = list(pending)
        for index in range(count):
            order = names if index % 2 == 0 else names[::-1]
            executed = dict(phase=phase, round=index, backends=[])
            executed_order.append(executed)
            for name in order:
                if name not in active:
                    continue
                executed["backends"].append(name)
                try:
                    pending[name][2]()
                except Exception as exc:
                    # Let the backend retain its traceback/error row, then stop
                    # timing it. Successful backends still finish every sample.
                    try:
                        active[name].throw(exc)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError("Failed backend did not stop") from exc
                    del active[name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["decode", "ragged", "paged"], required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--kv", type=int, required=True)
    p.add_argument("--qo-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--layout", choices=["HND", "NHD"], default="NHD")
    p.add_argument("--backends", nargs="+", default=["cudnn"])
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefill mask; --no-causal includes every KV token for each query",
    )
    p.add_argument("--expected-backend-version", type=int)
    p.add_argument("--expect-auto-backend", choices=["fa2", "fa3", "cudnn"])
    p.add_argument(
        "--ragged-indptr",
        choices=["cpu", "gpu", "gpu_with_cpu_mirrors"],
        default="cpu",
        help="compare GPU prefix readback with optional existing CPU mirrors",
    )
    p.add_argument(
        "--replan",
        action="store_true",
        help="also measure run after each plan and complete plan+run",
    )
    a = p.parse_args()
    if a.kind == "decode" and not a.causal:
        p.error("--no-causal applies only to prefill (--kind paged or ragged)")
    if a.kind != "ragged" and a.ragged_indptr != "cpu":
        p.error("--ragged-indptr applies only to --kind ragged")
    if a.kv_heads <= 0 or a.qo_heads <= 0 or a.qo_heads % a.kv_heads:
        p.error("head counts must be positive with an integral Q/KV ratio")
    if min(a.batch, a.q, a.kv, a.head_dim, a.page_size) <= 0 or a.q > a.kv:
        p.error(
            "batch, lengths, head dimension and page size must be positive; q <= kv"
        )
    if a.expect_auto_backend is not None and "auto" not in a.backends:
        p.error("--expect-auto-backend requires --backends auto")
    if (
        "auto" in a.backends
        and os.getenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES") is not None
    ):
        p.error(
            "unset CUDNN_FRONTEND_ENABLE_FROST_ENGINES when measuring auto dispatch"
        )
    if a.expected_backend_version is not None:
        assert cudnn.backend_version() == a.expected_backend_version
    if len(set(a.backends)) != len(a.backends):
        p.error("--backends must not contain duplicates")
    torch.manual_seed(42)
    b, sq, sk, h, hk, d, page = (
        a.batch,
        a.q,
        a.kv,
        a.qo_heads,
        a.kv_heads,
        a.head_dim,
        a.page_size,
    )
    pages = (sk + page - 1) // page
    q = torch.randn(b * sq, h, d, dtype=getattr(torch, a.dtype), device="cuda")
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
    kv = torch.arange(b + 1, dtype=torch.int32) * sk
    qo_plan, kv_plan = (qo, kv) if a.ragged_indptr == "cpu" else (qo.cuda(), kv.cuda())
    host_indptr = (
        dict(qo_indptr_cpu=qo, kv_indptr_cpu=kv)
        if a.ragged_indptr == "gpu_with_cpu_mirrors"
        else {}
    )
    if a.kind == "paged":
        qo, indptr, last = qo.pin_memory(), indptr.pin_memory(), last.pin_memory()
        seq_lens_cpu = torch.full((b,), sk, dtype=torch.int32, pin_memory=True)
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
    # Independent math reference for first/last requests and query boundary rows.
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
        length = sk - sq + qi + 1 if a.causal else sk
        scores = (
            torch.einsum("hd,shd->hs", q[bi * sq + qi].float(), k[:length]) / d**0.5
        )
        references.append(torch.einsum("hs,shd->hd", scores.softmax(-1), v[:length]))
    torch.cuda.synchronize()
    output_tol = 0.005 if q.dtype == torch.float16 else 0.01
    record = dict(
        args=vars(a),
        gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        cudnn=cudnn.backend_version(),
        frontend_version=cudnn.__version__,
        torch_version=torch.__version__,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fe_file=cudnn.__file__,
        fi_file=flashinfer.__file__,
        fi_version=flashinfer.__version__,
        fi_source=_source_revision(),
        frost_env=os.getenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES"),
        sampling_order=[],
        sampling_policy="alternating_forward_reverse_per_metric_round",
        workspace_bytes_per_backend=256 * 1024 * 1024,
        results=[],
    )
    outputs = {}

    def measure_backend(backend):
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
                plan = lambda: w.plan(indptr, indices, last, h, hk, d, page, **kw)
            elif a.kind == "paged":
                w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    ws, a.layout, backend=backend
                )
                plan = lambda: w.plan(
                    qo,
                    indptr,
                    indices,
                    last,
                    h,
                    hk,
                    d,
                    page,
                    causal=a.causal,
                    q_data_type=q.dtype,
                    kv_data_type=q.dtype,
                    seq_lens=seq_lens_cpu,
                )
            else:
                w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    ws, "NHD", backend=backend
                )
                plan = lambda: w.plan(
                    qo_plan,
                    kv_plan,
                    h,
                    hk,
                    d,
                    causal=a.causal,
                    q_data_type=q.dtype,
                    kv_data_type=q.dtype,
                    **host_indptr,
                )
            plan()
            row["plan_cpu_us"] = (time.perf_counter_ns() - t0) / 1e3
            # Keep asynchronous plan work out of the first execution timing.
            torch.cuda.synchronize()
            row["plan_completed_wall_us"] = (time.perf_counter_ns() - t0) / 1e3
            row["actual_backend"] = getattr(
                w, "resolved_backend", getattr(w, "_backend", backend)
            )
            if backend == "auto" and a.expect_auto_backend is not None:
                assert row["actual_backend"] == a.expect_auto_backend, row
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
                    out[bi * sq + qi].float(), ref, atol=output_tol, rtol=output_tol
                )
            if outputs:
                ref_name, ref = next(iter(outputs.items()))
                torch.testing.assert_close(out, ref, atol=output_tol, rtol=output_tol)
                row["full_reference_backend"] = ref_name
                row["full_max_abs"] = float((out.float() - ref.float()).abs().max())
            outputs[backend] = out.clone()
            prepared = getattr(w, "_cudnn_prepared", None)
            paged_plan = getattr(w, "_cudnn_plan", None)
            graph = None
            if paged_plan is not None:
                graph = paged_plan.bound_graph.graph
            elif prepared is not None:
                graph = prepared.graph
            if graph is not None:
                idx = getattr(graph, "_plan_index", 0)
                row["fe_plan"] = graph.get_plan_name_at_index(idx)
                engine, knobs = graph.get_engine_and_knobs_at_index(idx)
                row["fe_engine"] = engine
                row["fe_knobs"] = {str(k): v for k, v in knobs.items()}
            for _ in range(20):
                run()
            torch.cuda.synchronize()
            host = []

            def sample_host():
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                run()
                host.append((time.perf_counter_ns() - t0) / 1e3)

            yield "host_enqueue", 100, sample_host
            torch.cuda.synchronize()
            row["host_enqueue_us"] = statistics.median(host)
            row["host_enqueue_samples_us"] = host
            eager, submitted = [], []

            def sample_eager():
                torch.cuda.synchronize()
                begin = time.perf_counter_ns()
                for _ in range(20):
                    run()
                enqueued = time.perf_counter_ns()
                torch.cuda.synchronize()
                drained = time.perf_counter_ns()
                submitted.append((enqueued - begin) / 20 / 1000)
                eager.append((drained - begin) / 20 / 1000)

            yield "completed_eager", 20, sample_eager
            row["loop_submit_us"] = statistics.median(submitted)
            row["loop_submit_samples_us"] = submitted
            row["drained_eager_us"] = statistics.median(eager)
            row["drained_eager_samples_us"] = eager
            if a.replan:
                after_plan, complete, plans = [], [], []

                def sample_replan():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    plan()
                    plans.append((time.perf_counter_ns() - t0) / 1e3)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    run()
                    after_plan.append((time.perf_counter_ns() - t0) / 1e3)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    plan()
                    run()
                    complete.append((time.perf_counter_ns() - t0) / 1e3)

                yield "replan", 100, sample_replan
                torch.cuda.synchronize()
                row["plan_host_us"] = statistics.median(plans)
                row["plan_host_samples_us"] = plans
                row["run_after_plan_us"] = statistics.median(after_plan)
                row["plan_run_host_us"] = statistics.median(complete)
                row["run_after_plan_samples_us"] = after_plan
                row["plan_run_host_samples_us"] = complete
                for (bi, qi), ref in zip(samples, references, strict=True):
                    torch.testing.assert_close(
                        out[bi * sq + qi].float(), ref, atol=output_tol, rtol=output_tol
                    )
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    run()
            # Replay correctness is separate from eager, including poisoned output.
            # The ordinary paged wrapper has no graph-mode prefix buffers;
            # benchmark its plan-once replay independently of eager replanning.
            row["replan_before_replay"] = (
                a.replan and backend == "cudnn" and a.kind != "paged"
            )
            if row["replan_before_replay"]:
                plan()
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            expected = outputs[backend]
            upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
            lower = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
            ulp = torch.maximum(
                (upper.float() - expected.float()).abs(),
                (expected.float() - lower.float()).abs(),
            )
            assert torch.isfinite(out).all()
            assert ((out.float() - expected.float()).abs() <= ulp).all(), (
                "replay differs by more than one output ULP"
            )
            for _ in range(5):
                g.replay()
            ts = []

            def sample_graph():
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                g.replay()
                end.record()
                end.synchronize()
                ts.append(start.elapsed_time(end) * 1000 / 20)

            yield "graph_replay", 20, sample_graph
            # Replan can release storage still referenced by captured native
            # pointers. Allocations in the earlier checks may then recycle it:
            # validate the timed replays too, before accepting their timings.
            assert torch.isfinite(out).all()
            assert ((out.float() - expected.float()).abs() <= ulp).all(), (
                "timed replay differs by more than one output ULP"
            )
            row.update(
                graph_gpu_us=statistics.median(ts),
                graph_gpu_samples_us=ts,
                status="pass",
                replay="within_one_ulp",
                timed_replay="within_one_ulp",
                math_reference_rows=len(samples),
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            row["error"] = repr(exc)
        record["results"].append(row)
        print(json.dumps(row), flush=True)
        Path(a.output).write_text(json.dumps(record, indent=2) + "\n")

    allocated_before = torch.cuda.memory_allocated()
    _interleave_measurements(
        [(backend, measure_backend(backend)) for backend in a.backends],
        record["sampling_order"],
    )
    record["gpu_allocated_before_measurements_bytes"] = allocated_before
    record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    # Preserve CLI backend order even when a failed backend finishes early.
    record["results"].sort(key=lambda row: a.backends.index(row["backend"]))
    Path(a.output).write_text(json.dumps(record, indent=2) + "\n")
    if any(r["status"] != "pass" for r in record["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
