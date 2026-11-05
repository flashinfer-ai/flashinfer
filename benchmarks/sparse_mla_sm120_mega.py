"""Equal-work prefill timing: fused, current unfused, and an archived unfused build.

Run through ``python -m benchmarks.bench_sparse_mla_sm120 --mega``.
The fixture/reference utilities require the repository's test dependencies.
"""

import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

from flashinfer.jit.mla import gen_sparse_mla_sm120_module
from flashinfer.mla._sparse_mla_sm120._execution import (
    _precision_code,
    resolve_model_type,
)
from flashinfer.testing.utils import bench_gpu_time


def add_arguments(parser):
    parser.add_argument(
        "--mega", action="store_true", help="Compare complete sparse prefill pipelines"
    )
    parser.add_argument(
        "--q-lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["v41_fp8", "nvfp4"],
        default=["v41_fp8", "nvfp4"],
    )
    parser.add_argument(
        "--old-module",
        type=Path,
        help="Archived pre-PR sparse MLA JIT .so (matching executor ABI)",
    )
    parser.add_argument(
        "--output", type=Path, help="Write measurements, samples and provenance as JSON"
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup-ms", type=int, default=40)
    parser.add_argument("--measure-ms", type=int, default=120)


def unfused_call(f, module):
    if f["cache_format"] == "nvfp4":
        from tests.attention.sparse_mla_prefill_utils import native_prefill_calls

        return native_prefill_calls(f, module)[1]
    q = f["q"]
    lse = torch.empty(q.shape[:2], device=q.device)
    model = resolve_model_type(512, f["wrapper"]._kv_scale_format)
    metadata = list(
        module.inspect_metadata(
            q,
            f["caches"][0],
            f["indices"][0],
            f["context"],
            f["lengths"][0],
            f["sink"],
            f["caches"][1],
            f["indices"][1],
            f["lengths"][1],
            lse,
            None,
            None,
            model,
            False,
            512,
        )
    )
    metadata[17] = 1  # SG prefill; also the dispatch entry for CuTe FP8.
    device = torch.cuda.get_device_properties(q.device)
    plan = module.resolve_attention(
        metadata,
        _precision_code("fp8"),
        1,
        device.multi_processor_count,
        device.shared_memory_per_block_optin,
    )

    def run():
        f["prologue"]()
        module.execute_attention(
            plan,
            f["rotated"],
            f["caches"][0],
            f["indices"][0],
            f["context"],
            lse,
            f["context"],
            lse,
            512**-0.5,
            f["lengths"][0],
            f["sink"],
            f["caches"][1],
            f["indices"][1],
            f["lengths"][1],
            math.log(2),
        )
        f["post"]()
        return lse

    return run


def run(args):
    from flashinfer.utils import is_sm120a_supported

    if not torch.cuda.is_available() or not is_sm120a_supported(torch.device("cuda")):
        raise SystemExit("Mega sparse prefill requires SM120/SM121")
    from tests.attention.sparse_mla_prefill_utils import (
        fused_call,
        native_prefill_calls,
        numerical_metrics,
        prefill_fixture,
    )

    if min(args.q_lengths) <= 64 or max(args.q_lengths) * 4 > 100000:
        raise ValueError(
            "Q lengths must be >64 and 4*Q must fit the 100001-row RoPE table"
        )
    if min(args.rounds, args.warmup_ms, args.measure_ms) < 1:
        raise ValueError("rounds and timing durations must be positive")
    current = gen_sparse_mla_sm120_module().build_and_load()
    old = None
    if args.old_module:
        import tvm_ffi

        old = tvm_ffi.load_module(str(args.old_module.resolve()))
    result = {
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "old_module": str(args.old_module.resolve()) if old else None,
        "old_module_sha256": hashlib.sha256(args.old_module.read_bytes()).hexdigest()
        if old
        else None,
        "timing": {
            "warm_l2": True,
            "cuda_graph": True,
            "rounds": args.rounds,
            "warmup_ms": args.warmup_ms,
            "measure_ms": args.measure_ms,
        },
        "rows": [],
    }
    print(
        "format,Q tokens,KV tokens,heads,mega ms,unfused ms,old unfused ms,old/mega speedup,old/unfused speedup",
        flush=True,
    )
    for fmt in args.formats:
        for qlen in args.q_lengths:
            f = prefill_fixture(
                1, fmt, heads=64, prefill_tokens=qlen, prefill_cp_size=4
            )
            mega = (
                native_prefill_calls(f)[0] if fmt == "nvfp4" else lambda: fused_call(f)
            )
            calls = {"mega": mega, "unfused": unfused_call(f, current)}
            if old:
                calls["old_unfused"] = unfused_call(f, old)
            rows = torch.linspace(0, qlen - 1, min(16, qlen), device="cuda").long()
            checks = {}
            for name, fn in calls.items():
                lse = fn()
                metrics = numerical_metrics(f, lse, rows)
                limit = 0.20 if fmt == "nvfp4" else 0.07
                if not metrics["finite_output"] or metrics["relative_l2"] >= limit:
                    raise AssertionError((fmt, qlen, name, metrics))
                checks[name] = metrics
            samples = {name: [] for name in calls}
            names = list(calls)
            for round_id in range(args.rounds):
                shift = round_id % len(names)
                for name in names[shift:] + names[:shift]:
                    values = bench_gpu_time(
                        calls[name],
                        use_cuda_graph=True,
                        cold_l2_cache=False,
                        dry_run_time_ms=args.warmup_ms,
                        repeat_time_ms=args.measure_ms,
                    )
                    samples[name].append(statistics.median(values))
            ms = {name: statistics.median(values) for name, values in samples.items()}
            row = {
                "format": fmt,
                "batch": 1,
                "q_tokens": qlen,
                "kv_tokens": 4 * qlen,
                "heads": 64,
                "head_dim": 512,
                "swa_candidates": 128,
                "sparse_candidates_max": 512,
                "latency_ms": ms,
                "samples_ms": samples,
                "reference_checks": checks,
            }
            if old:
                row["old_over_mega"] = ms["old_unfused"] / ms["mega"]
                row["old_over_unfused"] = ms["old_unfused"] / ms["unfused"]
            result["rows"].append(row)
            print(
                f"{fmt},{qlen},{4 * qlen},64,{ms['mega']:.6f},{ms['unfused']:.6f},"
                + (
                    f"{ms['old_unfused']:.6f},{row['old_over_mega']:.3f},{row['old_over_unfused']:.3f}"
                    if old
                    else "n/a,n/a,n/a"
                ),
                flush=True,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result
