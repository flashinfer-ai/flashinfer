"""Paired cold/hot timing of FlashInfer BF16, native FP8, and Triton FP8.

Preserves the service. Every tuning/measurement round starts with an idle GPU;
rounds contaminated by external activity are discarded. No whole-model claim.
"""

import argparse
import gc
import json
import itertools
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper
from bench_fp8_kv import (
    Captured,
    wait_idle,
    gpu_state,
    external_activity,
    summary,
    MONITOR_ERRORS,
)
from kv_bench_kernels import ExperimentalMLA, quantize_rows
from native_fp8.wrapper import NativeMLA, build

ROOT = Path(__file__).resolve().parent
CONFIGS = [
    (32, 32, 1, 2),
    (32, 64, 1, 2),
    (32, 32, 2, 2),
    (16, 64, 1, 2),
    (64, 32, 1, 2),
    (64, 64, 1, 2),
    (32, 64, 1, 4),
    (16, 64, 1, 4),
    (32, 32, 1, 4),
]


def contaminated(start, before):
    after = gpu_state()
    time.sleep(1.2)
    activity = external_activity(start)
    bad = bool(
        activity
        or MONITOR_ERRORS
        or before["external_memory"] != after["external_memory"]
    )
    return bad, dict(
        external_activity=activity,
        monitor_errors=list(MONITOR_ERRORS),
        before=before,
        after=after,
    )


def one_case(name, batch, qlen, length, workspace, flush, samples, quick):
    page = 16
    heads = 20
    causal = qlen > 1
    required = (
        batch * ((length + 15) // 16) * 16 * 576 * 3
        + batch * qlen * heads * (576 * 3 + 512 * 16)
        + 128 * 1024**2
    )
    idle = wait_idle(required)
    print("CASE_START", name, batch, qlen, length, flush=True)
    torch.manual_seed(47 + batch * 100000 + length + qlen)
    q = torch.randn(batch * qlen, heads, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(
        batch * ((length + 15) // 16), page, 576, device="cuda", dtype=torch.bfloat16
    )
    kv8, ks = quantize_rows(kv)
    indices = torch.randperm(kv.shape[0], device="cuda", dtype=torch.int32)
    qi = torch.arange(batch + 1, dtype=torch.int32) * qlen
    ki = torch.arange(batch + 1, dtype=torch.int32) * ((length + 15) // 16)
    lengths = torch.full((batch,), length, dtype=torch.int32)
    fi = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    fi.plan(
        qi,
        ki,
        indices,
        lengths,
        heads,
        512,
        64,
        page,
        causal,
        1 / 16,
        torch.bfloat16,
        torch.bfloat16,
    )
    fiout = torch.empty(batch * qlen, heads, 512, device="cuda", dtype=torch.bfloat16)
    original = Captured(
        lambda: fi.run(
            q[..., :512], q[..., 512:], kv[..., :512], kv[..., 512:], out=fiout
        )
    )
    original.sample(None)
    reference = fiout.float().clone()
    methods = {"flashinfer_bf16": original}
    keep = [fi]
    baseline_triton_config = None
    if qlen == 1:
        previous = json.loads((ROOT.parent / "kv_bench_results.json").read_text())
        found = [
            c
            for c in previous["cases"]
            if c["batch"] == batch and c["kv_len"] == length
        ]
        baseline_triton_config = (
            found[0]["selected"]["fp8_cache_fp8_mma"]
            if found
            else dict(bm=32, bn=64, splits=32, warps=8)
        )
        c = baseline_triton_config
        tr = ExperimentalMLA(
            q,
            kv8,
            indices.reshape(batch, -1),
            lengths.cuda(),
            2,
            c["splits"],
            c["bn"],
            c["bm"],
            c["warps"],
        )
        methods["triton_fp8"] = Captured(lambda: tr.run(ks))
        keep.append(tr)
    idle_tune = wait_idle()
    started = time.time()
    trials = []
    best = None
    configs = CONFIGS[:2] if quick else CONFIGS
    for bm, bn, stages, groups in configs:
        if qlen == 1 and bm == 64:
            continue
        for workers, fused in itertools.product(
            (110,) if quick else (55, 110, 220, 440),
            (True,) if quick else (True, False),
        ):
            try:
                runner = NativeMLA(
                    workspace,
                    qi,
                    ki,
                    indices,
                    lengths,
                    heads=heads,
                    page_size=page,
                    causal=causal,
                    bm=bm,
                    bn=bn,
                    stages=stages,
                    groups=groups,
                    workers=workers,
                    fused=fused,
                )
                cap = Captured(lambda r=runner: r.run(q, kv8, ks))
                out = runner.out.float()
                err = ((out - reference).norm() / reference.norm()).item()
                assert bool(torch.isfinite(out).all()) and err < 0.09, (
                    runner.config,
                    err,
                )
                times = [cap.sample(flush) for _ in range(7)]
                score = summary(times)["median_us"]
                trial = dict(
                    config=runner.config,
                    attributes=runner.attributes,
                    scheduling=runner.scheduling,
                    tuning_us=score,
                    relative_l2_vs_bf16=err,
                )
                trials.append(trial)
                if best is None or score < best[0]:
                    best = (score, runner, cap, trial)
                del runner, cap
            except ValueError as exc:
                trials.append(
                    dict(
                        config=dict(
                            bm=bm,
                            bn=bn,
                            stages=stages,
                            groups=groups,
                            workers=workers,
                            fused=fused,
                        ),
                        skipped=str(exc),
                    )
                )
    bad, tune_monitor = contaminated(started, idle_tune[-1])
    if bad:
        print("DISCARD_TUNING", name, json.dumps(tune_monitor), flush=True)
        return None
    _, runner, cap, selected = best
    methods["native_fp8"] = cap
    keep.append(runner)
    methods["native_fp8_prequantized_q"] = Captured(
        lambda: runner.run_prequantized(runner.q8, kv8, runner.qs, ks)
    )
    if qlen > 1:
        # An input-inclusive control: BF16 current KVs must also be quantized.
        # For extend, only the new rows are quantized, with contiguous writes.
        new_kv = torch.randn(batch * qlen, 576, device="cuda", dtype=torch.bfloat16)
        new_kv8 = torch.empty_like(new_kv, dtype=torch.float8_e4m3fn)
        new_ks = torch.empty(batch * qlen, device="cuda", dtype=torch.float32)
        methods["new_kv_quantization"] = Captured(
            lambda: quantize_rows(new_kv, new_kv8, new_ks)
        )
        if qlen == length:

            def full_prefill():
                quantize_rows(kv, kv8, ks)
                return runner.run(q, kv8, ks)

            methods["native_fp8_including_kv_quantization"] = Captured(full_prefill)
    print("SELECTED", name, json.dumps(selected), flush=True)
    idle_timing = wait_idle()
    started = time.time()
    cold = {key: [] for key in methods}
    hot = {key: [] for key in methods}
    for cap in methods.values():
        for _ in range(3):
            cap.sample(flush)
    for _ in range(samples):
        keys = list(methods)
        random.shuffle(keys)
        for key in keys:
            cold[key].append(methods[key].sample(flush))
    for _ in range(samples):
        keys = list(methods)
        random.shuffle(keys)
        for key in keys:
            methods[key].sample(None)
            hot[key].append(methods[key].sample(None))
    bad, monitor = contaminated(started, idle_timing[-1])
    if bad:
        print("DISCARD_TIMING", name, json.dumps(monitor), flush=True)
        return None
    stats = {
        key: dict(cold=summary(cold[key]), hot=summary(hot[key])) for key in methods
    }
    errors = {}
    for key in ("native_fp8", "triton_fp8"):
        if key in methods:
            cap = methods[key]
            cap.sample(None)
            out = cap.output.float()
            errors[key] = dict(
                relative_l2_vs_flashinfer=(
                    (out - reference).norm() / reference.norm()
                ).item(),
                max_abs_vs_flashinfer=(out - reference).abs().max().item(),
            )
    result = dict(
        name=name,
        batch=batch,
        qlen=qlen,
        kv_len=length,
        causal=causal,
        heads=heads,
        page_size=page,
        methods=stats,
        selected=selected,
        trials=trials,
        errors=errors,
        triton_config=baseline_triton_config,
        idle_before=idle,
        tuning_monitor=tune_monitor,
        timing_monitor=monitor,
        peak_allocated=torch.cuda.max_memory_allocated(),
        source=str(
            build(
                selected["config"]["bm"],
                selected["config"]["bn"],
                selected["config"]["stages"],
                selected["config"]["groups"],
            )
        ),
    )
    print(
        "RESULT",
        name,
        json.dumps({k: v["cold"]["median_us"] for k, v in stats.items()}),
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--output", default="/tmp/mla_fp8_sm120_results.json")
    p.add_argument("--cases", nargs="+")
    a = p.parse_args()
    for config in CONFIGS[:2] if a.quick else CONFIGS:
        print("BUILD", build(*config), flush=True)
    wait_idle(1100 * 1024**2)
    torch.set_num_threads(4)
    random.seed(47)
    workspace = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    scenarios = [
        ("decode_b1_1k", 1, 1, 1024),
        ("decode_b1_8k", 1, 1, 8192),
        ("decode_b1_32k", 1, 1, 32768),
        ("decode_b4_8k", 4, 1, 8192),
        ("decode_b4_32k", 4, 1, 32768),
        ("decode_b16_8k", 16, 1, 8192),
        ("decode_b16_32k", 16, 1, 32768),
        ("extend_16_8k", 1, 16, 8192),
        ("prefill_128", 1, 128, 128),
        ("prefill_512", 1, 512, 512),
        ("extend_128_8k", 1, 128, 8192),
        ("prefill_2048", 1, 2048, 2048),
        ("prefill_4096", 1, 4096, 4096),
    ]
    if a.quick:
        scenarios = [scenarios[1], scenarios[7], scenarios[8]]
    if a.cases:
        scenarios = [s for s in scenarios if s[0] in a.cases]
    report = dict(
        environment=dict(
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
        ),
        cases=[],
    )
    path = ROOT / a.output
    if path.exists():
        report = json.loads(path.read_text())
    done = {c["name"] for c in report["cases"]}
    for scenario in scenarios:
        if scenario[0] in done:
            continue
        result = None
        while result is None:
            result = one_case(*scenario, workspace, flush, a.samples, a.quick)
            gc.collect()
            torch.cuda.empty_cache()
        report["cases"].append(result)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", path, flush=True)


if __name__ == "__main__":
    main()
