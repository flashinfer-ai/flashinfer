"""Compare this checkout to a frozen commit, waiting for idle GPU windows."""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(STUDY))

import torch
import flashinfer
from flashinfer.mla import BatchMLAPagedAttentionWrapper
from bench_fp8_kv import Captured, wait_idle, summary
from kv_bench_kernels import quantize_rows
from native_fp8.bench import contaminated
from native_fp8.wrapper import NativeMLA, build

BASELINE_COMMIT = "8f02f40243e9b1f09712d72584a5103768f1cc2f"
CONFIGS = [
    (32, 32, 1, 2, False),
    (16, 32, 1, 2, True),
    (16, 64, 1, 2, True),
    (16, 64, 1, 4, True),
    (64, 32, 1, 4, True),
    (64, 32, 2, 4, True),
    (32, 32, 1, 2, True),
    (32, 32, 1, 4, True),
    (32, 64, 1, 2, True),
    (32, 64, 1, 4, True),
    (32, 64, 2, 2, True),
    (32, 64, 2, 4, True),
    (64, 32, 2, 2, True),
    (64, 64, 1, 2, True),
    (64, 64, 1, 4, True),
]


def baseline_module(commit):
    """Load only the old research backend; both builds use current core headers."""
    root = Path(tempfile.mkdtemp(prefix="mla-fp8-baseline-")) / "backend"
    root.mkdir()
    for name in (
        "__init__.py",
        "wrapper.py",
        "quantization.py",
        "mla_fp8.cu",
        "scheduler_fp8.cuh",
    ):
        source = subprocess.check_output(
            ["git", "show", f"{commit}:flashinfer/experimental/mla_fp8_sm120/{name}"],
            cwd=REPO,
        )
        (root / name).write_bytes(source)
    p = root / "wrapper.py"
    p.write_text(
        p.read_text().replace(
            "checkout = ROOT.parents[2]", f"checkout = Path({str(REPO)!r})"
        )
    )
    name = "_mla_fp8_frozen_baseline"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location(
        name + ".wrapper", root / "wrapper.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def case(name, batch, qlen, klen, old, ws, flush, args):
    page = 16
    heads = 20
    causal = qlen > 1
    wait_idle(1500 * 1024**2)
    torch.manual_seed(47 + batch * 100000 + klen + qlen)
    q = torch.randn(batch * qlen, heads, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(
        batch * ((klen + 15) // 16), page, 576, device="cuda", dtype=torch.bfloat16
    )
    kv8, ks = quantize_rows(kv)
    q8, qs = quantize_rows(q)
    idx = torch.randperm(kv.shape[0], device="cuda", dtype=torch.int32)
    qi = torch.arange(batch + 1, dtype=torch.int32) * qlen
    ki = torch.arange(batch + 1, dtype=torch.int32) * ((klen + 15) // 16)
    lengths = torch.full((batch,), klen, dtype=torch.int32)
    common = dict(heads=heads, page_size=page, causal=causal)
    inputs = (ws, qi, ki, idx, lengths)
    historical = json.loads((Path(__file__).parent / "results.json").read_text())
    base_config = next(
        c["selected"]["config"] for c in historical["cases"] if c["name"] == name
    )
    baseline = old.NativeMLA(*inputs, **common, **base_config)
    baseline.run_prequantized(q8, kv8, qs, ks)
    reference = baseline.out.float().clone()
    before = wait_idle()[-1]
    start = time.time()
    trials = []
    best = None
    configs = [dict(**base_config, share_p=False)]
    for bm, bn, stages, groups, share_p in CONFIGS:
        if qlen == 1 and bm == 64:
            continue
        workers_set = (
            (55, 110, 220, 440)
            if qlen <= 128 and klen > 512
            else ((55, 110, 220) if qlen <= 512 else (110, 220))
        )
        for workers in workers_set:
            for fused in (False, True) if qlen <= 128 else (False,):
                cfg = dict(
                    bm=bm,
                    bn=bn,
                    stages=stages,
                    groups=groups,
                    share_p=share_p,
                    workers=workers,
                    fused=fused,
                )
                if cfg not in configs:
                    configs.append(cfg)
    for cfg in configs:
        try:
            runner = NativeMLA(*inputs, **common, **cfg)
            cap = Captured(lambda r=runner: r.run(q, kv8, ks))
            error = ((runner.out.float() - reference).norm() / reference.norm()).item()
            assert torch.isfinite(runner.out).all() and error < 0.04, (cfg, error)
            times = [cap.sample(flush) for _ in range(5)]
            score = summary(times)["median_us"]
            trial = dict(
                config=cfg,
                attributes=runner.attributes,
                scheduling=runner.scheduling,
                median_us=score,
                relative_l2_vs_previous=error,
            )
            trials.append(trial)
            print("TRIAL", name, json.dumps(trial), flush=True)
            if best is None or score < best[0]:
                best = (score, runner, cap)
            del runner, cap
        except (ValueError, RuntimeError) as exc:
            trials.append(dict(config=cfg, skipped=str(exc)))
            print("SKIP", cfg, str(exc), flush=True)
    bad, monitor = contaminated(start, before)
    if bad:
        print("DISCARD_TUNING", name, json.dumps(monitor), flush=True)
        return None
    if best is None:
        raise RuntimeError("No candidate supports this shape.")
    _, runner, newcap = best
    # Compare the selected candidates in an independent, randomized paired round.
    fi = BatchMLAPagedAttentionWrapper(ws, backend="fa2")
    fi.plan(
        qi,
        ki,
        idx,
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
    fio = torch.empty_like(runner.out)
    methods = {
        "bf16": Captured(
            lambda: fi.run(
                q[..., :512], q[..., 512:], kv[..., :512], kv[..., 512:], out=fio
            )
        ),
        "previous": Captured(lambda: baseline.run(q, kv8, ks)),
        "optimized": newcap,
        "previous_prequantized": Captured(
            lambda: baseline.run_prequantized(q8, kv8, qs, ks)
        ),
        "optimized_prequantized": Captured(
            lambda: runner.run_prequantized(q8, kv8, qs, ks)
        ),
    }
    if qlen == klen:

        def inclusive(r):
            quantize_rows(kv, kv8, ks)
            return r.run(q, kv8, ks)

        methods["previous_inclusive"] = Captured(lambda: inclusive(baseline))
        methods["optimized_inclusive"] = Captured(lambda: inclusive(runner))
    before = wait_idle()[-1]
    start = time.time()
    cold = {k: [] for k in methods}
    hot = {k: [] for k in methods}
    for cap in methods.values():
        for _ in range(3):
            cap.sample(flush)
    for _ in range(args.samples):
        keys = list(methods)
        random.shuffle(keys)
        for key in keys:
            cold[key].append(methods[key].sample(flush))
    for _ in range(args.samples):
        keys = list(methods)
        random.shuffle(keys)
        for key in keys:
            methods[key].sample(None)
            hot[key].append(methods[key].sample(None))
    bad, timing_monitor = contaminated(start, before)
    if bad:
        print("DISCARD_TIMING", name, json.dumps(timing_monitor), flush=True)
        return None
    methods["bf16"].sample(None)
    methods["optimized"].sample(None)
    error = ((runner.out.float() - fio.float()).norm() / fio.float().norm()).item()
    result = dict(
        name=name,
        batch=batch,
        qlen=qlen,
        kv_len=klen,
        baseline_config=base_config,
        selected=dict(
            config=runner.config,
            attributes=runner.attributes,
            scheduling=runner.scheduling,
        ),
        relative_l2_vs_bf16=error,
        trials=trials,
        methods={
            key: dict(cold=summary(cold[key]), hot=summary(hot[key])) for key in methods
        },
        tuning_monitor=monitor,
        timing_monitor=timing_monitor,
    )
    print(
        "RESULT",
        name,
        json.dumps({k: summary(v)["median_us"] for k, v in cold.items()}),
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["prefill_512", "prefill_2048"])
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("/tmp/mla-fp8-revision.json"))
    args = p.parse_args()
    old = baseline_module(BASELINE_COMMIT)
    for cfg in CONFIGS:
        print("BUILD", build(*cfg), flush=True)
    torch.set_num_threads(4)
    random.seed(47)
    wait_idle(1700 * 1024**2)
    ws = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    report = dict(
        baseline_commit=BASELINE_COMMIT,
        flashinfer_baseline=flashinfer.__version__,
        core_headers_base="bfe80f7594c98bd8a67167ee891a7b44dd6e31ba",
        cases=[],
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        source_sha256=hashlib.sha256(
            (REPO / "flashinfer/experimental/mla_fp8_sm120/mla_fp8.cu").read_bytes()
        ).hexdigest(),
    )
    if args.samples < 5:
        p.error("Use at least five samples per method.")
    report["samples_per_method"] = args.samples
    if args.output.exists():
        previous_report = json.loads(args.output.read_text())
        for key in (
            "source_sha256",
            "baseline_commit",
            "samples_per_method",
            "flashinfer_baseline",
        ):
            if previous_report.get(key) != report[key]:
                raise ValueError(
                    f"Resume metadata mismatch for {key}; use a fresh output file."
                )
        report = previous_report
    scenarios = json.loads((Path(__file__).parent / "results.json").read_text())[
        "cases"
    ]
    unknown = set(args.cases) - {c["name"] for c in scenarios}
    if unknown:
        p.error(f"Unknown cases: {sorted(unknown)}")
    for c in scenarios:
        if c["name"] not in args.cases or c["name"] in {
            x["name"] for x in report["cases"]
        }:
            continue
        result = None
        while result is None:
            result = case(
                c["name"], c["batch"], c["qlen"], c["kv_len"], old, ws, flush, args
            )
            gc.collect()
            torch.cuda.empty_cache()
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
