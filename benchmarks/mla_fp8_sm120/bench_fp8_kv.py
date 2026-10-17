"""Idle-GPU, CUDA-graph timed BF16 vs FP8 KV experiment for 20-head MLA.

Cache is already populated. Native-FP8 timing includes quantizing each new Q.
Historical KV quantization is outside decode timing. Every cold sample evicts L2
with a 256 MiB buffer outside the timed graph. No service is stopped or changed.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import threading
import time

import pynvml as nv
import torch
import triton
from flashinfer.mla import BatchMLAPagedAttentionWrapper
from kv_bench_kernels import ExperimentalMLA, quantize_rows, dequantize

ROOT = Path(__file__).resolve().parent
nv.nvmlInit()
GPU = nv.nvmlDeviceGetHandleByIndex(0)
OWN_PID = os.getpid()
torch.manual_seed(47)
torch.set_num_threads(4)
random.seed(47)
PROCESS_SAMPLES = []
MONITOR_ERRORS = []


def monitor_processes():
    last = int(time.time() * 1e6)
    while True:
        try:
            samples = nv.nvmlDeviceGetProcessUtilization(GPU, last)
            for s in samples:
                last = max(last, s.timeStamp)
                if s.pid != OWN_PID and (s.smUtil > 0 or s.memUtil > 0):
                    PROCESS_SAMPLES.append(
                        dict(
                            pid=s.pid,
                            sm=s.smUtil,
                            memory=s.memUtil,
                            timestamp=s.timeStamp,
                        )
                    )
        except nv.NVMLError_NotFound:
            pass
        except nv.NVMLError as exc:
            if str(exc) not in MONITOR_ERRORS:
                MONITOR_ERRORS.append(str(exc))
        time.sleep(0.25)


threading.Thread(target=monitor_processes, daemon=True).start()


def gpu_state():
    util = nv.nvmlDeviceGetUtilizationRates(GPU)
    processes = nv.nvmlDeviceGetComputeRunningProcesses(GPU)
    return dict(
        time=time.time(),
        util=util.gpu,
        memory_util=util.memory,
        free_bytes=nv.nvmlDeviceGetMemoryInfo(GPU).free,
        sm_clock_mhz=nv.nvmlDeviceGetClockInfo(GPU, nv.NVML_CLOCK_SM),
        mem_clock_mhz=nv.nvmlDeviceGetClockInfo(GPU, nv.NVML_CLOCK_MEM),
        power_w=nv.nvmlDeviceGetPowerUsage(GPU) / 1000,
        external_pids=[p.pid for p in processes if p.pid != OWN_PID],
        external_memory={
            str(p.pid): p.usedGpuMemory for p in processes if p.pid != OWN_PID
        },
    )


def wait_idle(min_free=0):
    count = 0
    last_log = 0
    states = []
    while count < 5:
        state = gpu_state()
        states.append(state)
        if (
            state["util"] <= 2
            and state["memory_util"] <= 2
            and state["free_bytes"] >= min_free
        ):
            count += 1
        else:
            count = 0
            if time.time() - last_log > 15:
                print("WAIT_IDLE", json.dumps(state), flush=True)
                last_log = time.time()
        time.sleep(1)
    return states[-5:]


def external_activity(since):
    monitored = [s for s in PROCESS_SAMPLES if s["timestamp"] >= int(since * 1e6)]
    try:
        samples = nv.nvmlDeviceGetProcessUtilization(GPU, int(since * 1e6))
        return monitored + [
            dict(pid=s.pid, sm=s.smUtil, memory=s.memUtil, timestamp=s.timeStamp)
            for s in samples
            if s.pid != OWN_PID and (s.smUtil > 0 or s.memUtil > 0)
        ]
    except nv.NVMLError_NotFound:
        return monitored
    except nv.NVMLError as exc:
        return monitored + [{"monitor_unavailable": str(exc)}]


class Captured:
    def __init__(self, fn):
        self.fn = fn
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(self.graph):
            self.start.record()
            self.output = fn()
            self.end.record()
        self.graph.replay()
        self.end.synchronize()

    def sample(self, flush):
        if flush is not None:
            flush.zero_()
        self.graph.replay()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000


def summary(xs):
    ys = sorted(xs)
    return {
        "median_us": statistics.median(ys),
        "mean_us": statistics.mean(ys),
        "p10_us": ys[int((len(ys) - 1) * 0.1)],
        "p90_us": ys[int((len(ys) - 1) * 0.9)],
        "samples_us": xs,
    }


def tune(q, kv, pages, lengths, mode, scales, flush, length):
    # A bounded local search, with failures recorded rather than hidden.
    layouts = [(32, 16, 8), (16, 32, 4)] if mode < 2 else [(32, 64, 8), (16, 64, 4)]
    records = []
    best = None
    batch = q.shape[0]
    for bm, bn, warps in layouts:
        target = triton.next_power_of_2(triton.cdiv(110, batch * triton.cdiv(20, bm)))
        max_splits = min(64, length // bn)
        split_options = sorted(
            {
                min(max_splits, max(1, target // 4)),
                min(max_splits, max(1, target // 2)),
                min(max_splits, target),
                min(max_splits, 2 * target),
            }
        )
        for splits in split_options:
            config = dict(mode=mode, bm=bm, bn=bn, warps=warps, splits=splits)
            try:
                runner = ExperimentalMLA(
                    q, kv, pages, lengths, mode, splits, bn, bm, warps
                )
                captured = Captured(lambda r=runner: r.run(scales))
                samples = [captured.sample(flush) for _ in range(5)]
                score = statistics.median(samples)
                row = dict(
                    **config,
                    tuning_us=score,
                    shared=runner.compiled_kernel.metadata.shared,
                    registers=runner.compiled_kernel.n_regs,
                )
                records.append(row)
                if best is None or score < best[0]:
                    best = (score, captured, runner, row)
                del captured, runner
            except triton.OutOfResources as exc:
                records.append(dict(**config, error=str(exc)))
    if best is None:
        raise RuntimeError(f"No valid config for mode={mode}")
    return best[1], best[2], best[3], records


def one_case(batch, length, workspace, flush, samples):
    page_size = 16
    n_pages = batch * length // page_size
    # Reserve space for BF16/FP8 caches, widening buffer, and graph scratch.
    required = n_pages * page_size * 576 * 5 + 256 * 1024**2
    idle_before = wait_idle(required)
    print("CASE_START", batch, length, json.dumps(idle_before[-1]), flush=True)
    torch.manual_seed(47 + batch * 100000 + length)
    q = torch.randn(batch, 20, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(n_pages, page_size, 576, device="cuda", dtype=torch.bfloat16)
    kv8, scales = quantize_rows(kv)
    pages = (
        torch.randperm(n_pages, device="cuda", dtype=torch.int32)
        .reshape(batch, length // page_size)
        .contiguous()
    )
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        torch.arange(batch + 1, dtype=torch.int32),
        torch.arange(batch + 1, dtype=torch.int32) * (length // page_size),
        pages.flatten(),
        lengths,
        20,
        512,
        64,
        page_size,
        False,
        1 / 16,
        torch.bfloat16,
        torch.bfloat16,
    )
    out = torch.empty(batch, 20, 512, device="cuda", dtype=torch.bfloat16)
    fi = Captured(
        lambda: wrapper.run(
            q[..., :512], q[..., 512:], kv[..., :512], kv[..., 512:], out=out
        )
    )
    fi.graph.replay()
    torch.cuda.synchronize()
    reference = out.clone()
    methods = {"flashinfer_bf16": fi}
    selected = {}
    tuning = {}
    errors = {}
    runners = []
    # Tune before the final paired measurements, then recheck an idle window.
    wait_idle()
    tune_started = time.time()
    for mode, name in [
        (0, "triton_bf16"),
        (1, "fp8_cache_bf16_mma"),
        (2, "fp8_cache_fp8_mma"),
    ]:
        captured, runner, config, records = tune(
            q, kv if mode == 0 else kv8, pages, lengths, mode, scales, flush, length
        )
        methods[name] = captured
        runners.append(runner)
        selected[name] = config
        tuning[name] = records
        torch.cuda.synchronize()
        actual = runner.out.float()
        ref = reference.float()
        err = ((actual - ref).norm() / ref.norm()).item()
        errors[name] = {
            "relative_l2_vs_flashinfer": err,
            "max_abs_vs_flashinfer": (actual - ref).abs().max().item(),
            "finite": bool(torch.isfinite(actual).all()),
        }
        assert errors[name]["finite"]
        if mode == 0:
            assert err < 0.01, errors[name]
        print(
            "SELECTED",
            batch,
            length,
            name,
            json.dumps(config),
            "error",
            err,
            flush=True,
        )
    # Hold tile shape/split count/warp count fixed to isolate cache storage.
    base_config = selected["triton_bf16"]
    same_tile_runner = ExperimentalMLA(
        q,
        kv8,
        pages,
        lengths,
        1,
        base_config["splits"],
        base_config["bn"],
        base_config["bm"],
        base_config["warps"],
    )
    methods["fp8_cache_same_tile_bf16_mma"] = Captured(
        lambda: same_tile_runner.run(scales)
    )
    runners.append(same_tile_runner)
    torch.cuda.synchronize()
    time.sleep(1.2)
    tune_interference = external_activity(tune_started)
    if any("pid" in x for x in tune_interference):
        print(
            "DISCARD_TUNING_EXTERNAL_ACTIVITY",
            json.dumps(tune_interference),
            flush=True,
        )
        return None
    # Measure a direct "store FP8, widen whole active cache, use existing FA2" approach.
    widened = torch.empty_like(kv)
    out_widened = torch.empty_like(out)

    def widen_then_fi():
        dequantize(kv8, scales, widened)
        return wrapper.run(
            q[..., :512],
            q[..., 512:],
            widened[..., :512],
            widened[..., 512:],
            out=out_widened,
        )

    methods["fp8_cache_external_dequant_flashinfer"] = Captured(widen_then_fi)
    # A new decode step adds one KV vector per request; measure row quantization
    # separately. This contiguous-write microbenchmark excludes page-allocation work.
    new_kv = torch.randn(batch, 576, device="cuda", dtype=torch.bfloat16)
    new_kv8 = torch.empty_like(new_kv, dtype=torch.float8_e4m3fn)
    new_scales = torch.empty(batch, device="cuda", dtype=torch.float32)
    quant = Captured(lambda: quantize_rows(new_kv, new_kv8, new_scales))
    idle_timing = wait_idle()
    started = time.time()
    cold = {name: [] for name in methods}
    hot = {name: [] for name in methods}
    # First replay after a pause is warmup, excluded from all statistics.
    for captured in methods.values():
        for _ in range(3):
            captured.sample(flush)
    for _ in range(samples):
        order = list(methods)
        random.shuffle(order)
        for name in order:
            cold[name].append(methods[name].sample(flush))
    for _ in range(samples):
        order = list(methods)
        random.shuffle(order)
        for name in order:
            # Warm this method's data after switching from another cache dtype.
            methods[name].sample(None)
            hot[name].append(methods[name].sample(None))
    qtimes = [quant.sample(None) for _ in range(samples)]
    torch.cuda.synchronize()
    after = gpu_state()
    time.sleep(1.2)
    interference = external_activity(started)
    if after["external_memory"] != idle_timing[-1]["external_memory"]:
        print("DISCARD_TIMING_EXTERNAL_MEMORY_CHANGED", flush=True)
        return None
    if any("pid" in x for x in interference):
        print("DISCARD_TIMING_EXTERNAL_ACTIVITY", json.dumps(interference), flush=True)
        return None
    stats = {
        name: {"cold": summary(cold[name]), "hot": summary(hot[name])}
        for name in methods
    }
    fi_us = stats["flashinfer_bf16"]["cold"]["median_us"]
    for name in stats:
        stats[name]["cold_speedup_vs_flashinfer"] = (
            fi_us / stats[name]["cold"]["median_us"]
        )
    return dict(
        batch=batch,
        kv_len=length,
        heads=20,
        page_size=page_size,
        cache_bytes_bf16=kv.numel() * 2,
        cache_bytes_fp8_including_scales=kv8.numel() + scales.numel() * 4,
        methods=stats,
        selected=selected,
        tuning=tuning,
        errors=errors,
        new_kv_quantization=summary(qtimes),
        idle_before=idle_before,
        idle_timing=idle_timing,
        gpu_after=after,
        external_activity=interference,
        tuning_external_activity=tune_interference,
        seed=47 + batch * 100000 + length,
        monitor_errors=list(MONITOR_ERRORS),
        torch_peak_allocated=torch.cuda.max_memory_allocated(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    shapes = (
        [(1, 1024), (4, 8192), (16, 32768)]
        if args.quick
        else [(b, l) for b in (1, 4, 16) for l in (1024, 8192, 32768)]
    )
    output = ROOT / ("kv_bench_quick.json" if args.quick else "kv_bench_results.json")
    report = {
        "environment": {
            "gpu": nv.nvmlDeviceGetName(GPU),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pid": OWN_PID,
            "start_state": gpu_state(),
            "kernel_sha256": hashlib.sha256(
                (ROOT / "kv_bench_kernels.py").read_bytes()
            ).hexdigest(),
        },
        "methodology": {
            "heads": 20,
            "qk_dim": 576,
            "latent_dim": 512,
            "softmax_scale": 1 / 16,
            "fp8": "E4M3, per-token FP32 KV scale, per-head FP32 Q scale",
            "timing": "CUDA graph with external CUDA events; randomized paired method order",
            "cold_l2_flush_bytes": 256 * 1024**2,
            "samples": args.samples,
            "scope": "Experimental fused decode operators; no model/server throughput claim",
            "q_quantization_included_in_native_fp8": True,
            "historical_kv_quantization_in_decode": False,
        },
        "cases": [],
    }
    wait_idle(600 * 1024**2)
    workspace = torch.empty(128 * 1024**2, dtype=torch.uint8, device="cuda")
    flush = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")
    for batch, length in shapes:
        while True:
            row = one_case(batch, length, workspace, flush, args.samples)
            gc.collect()
            torch.cuda.empty_cache()
            if row is not None:
                break
        report["cases"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            "CASE_DONE",
            batch,
            length,
            json.dumps(
                {k: round(v["cold"]["median_us"], 3) for k, v in row["methods"].items()}
            ),
            flush=True,
        )
    report["end_state"] = gpu_state()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", str(output), flush=True)


if __name__ == "__main__":
    main()
