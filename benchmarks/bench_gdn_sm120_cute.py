# SPDX-License-Identifier: Apache-2.0
"""Whole public-call comparison against FlashInfer's actual default route."""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
import warnings
import torch
from flashinfer import gdn_prefill as fi
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preallocated",
        action="store_true",
        help="include experimental output/state copy into caller buffers",
    )
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    warnings.filterwarnings("error", message=".*CUPTI.*[Ff]alling back.*")
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "flashinfer/gdn_prefill.py",
        *sorted((root / "flashinfer/experimental/gdn_sm120_cute").rglob("*.py")),
        *sorted((root / "flashinfer/experimental/gdn_sm120_cute").rglob("*.csv")),
    ]
    data = {
        "metadata": {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "gpu": torch.cuda.get_device_name(),
            "properties": str(torch.cuda.get_device_properties("cuda")),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tp_shape_only": args.tp,
            "distributed_tp": False,
            "preallocated": args.preallocated,
            "timing": "public API, CUPTI cold L2, 3 rotating rounds, warmup10, repeat"
            + str(args.repeats),
            "inputs": "same native FP32 log gates and CPU sequence metadata for both public calls; all materialization, normalization and buffer copies requested by the call are included; normalized BF16 QK prepared outside timing",
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
        },
        "cases": [],
    }

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(args.output)

    def relative(x, y):
        return (
            torch.linalg.vector_norm(x.double() - y.double())
            / torch.linalg.vector_norm(y.double()).clamp_min(1e-20)
        ).item()

    hq, hv = 16 // args.tp, 48 // args.tp
    cases = [
        [n] * b for b in (1, 2, 4, 8) for n in (2048, 4096, 8192, 16384, 32768)
    ] + [[2048, 4096], [8192, 2048, 4096, 1024], [2049, 4097]]
    from flashinfer.experimental.gdn_sm120_cute import _api

    for caseid, lengths in enumerate(cases):
        torch.manual_seed(634200 + args.tp * 1000 + sum(lengths) + len(lengths))
        total = sum(lengths)
        q, k = [
            torch.nn.functional.normalize(
                torch.randn(total, hq, 128, device="cuda"), dim=-1
            ).bfloat16()
            for _ in range(2)
        ]
        v = torch.randn(total, hv, 128, device="cuda", dtype=q.dtype)
        g = torch.nn.functional.logsigmoid(torch.randn(total, hv, device="cuda")) / 16
        g[:, 3 * hv // 4 :] = 0
        beta = torch.rand_like(g)
        initial = torch.randn(len(lengths), hv, 128, 128, device="cuda")
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        kwargs = dict(
            g=g,
            beta=beta,
            scale=128**-0.5,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
            cu_seqlens_cpu=offsets,
            gate_domain="log",
            use_cp="auto",
            max_seqlen=max(lengths),
        )
        if args.preallocated:
            kwargs.update(
                output=torch.empty_like(v), output_state=torch.empty_like(initial)
            )

        def official():
            return fi.chunk_gated_delta_rule(q, k, v, backend="flashinfer", **kwargs)

        def experimental():
            return fi.chunk_gated_delta_rule(q, k, v, backend="sm120_cute", **kwargs)

        native = []
        chosen = []
        cp, fused, cute = (
            fi.cp_delta_rule_dsl_sm120,
            fi.chunk_gated_delta_rule_sm120,
            _api.chunk_gated_delta_rule,
        )

        def cp_trace(*a, **kw):
            native.append("cp")
            return cp(*a, **kw)

        def fused_trace(*a, **kw):
            native.append("fused")
            return fused(*a, **kw)

        def cute_trace(*a, **kw):
            chosen.append("cute_cp" if kw["plan"].use_cp else "cute_fused")
            return cute(*a, **kw)

        fi.cp_delta_rule_dsl_sm120 = cp_trace
        fi.chunk_gated_delta_rule_sm120 = fused_trace
        _api.chunk_gated_delta_rule = cute_trace
        try:
            expected = tuple(x.clone() for x in official())
            official_route = list(native)
            native.clear()
            actual = experimental()
            torch.cuda.synchronize()
            errors = [relative(x, y) for x, y in zip(actual, expected, strict=True)]
            assert max(errors) < 0.02 and len(chosen) == 1 and not native, (
                lengths,
                errors,
                chosen,
                native,
            )
        finally:
            fi.cp_delta_rule_dsl_sm120 = cp
            fi.chunk_gated_delta_rule_sm120 = fused
            _api.chunk_gated_delta_rule = cute
        record = {
            "tp": args.tp,
            "batch": len(lengths),
            "lengths": lengths,
            "official_route": official_route,
            "experimental_route": chosen,
            "relative_l2": errors,
            "rounds": [],
        }
        data["cases"].append(record)
        save()
        fns = {"official_default": official, "cute_auto": experimental}
        for rnd in range(3):
            order = list(fns) if (rnd + caseid) % 2 == 0 else list(reversed(fns))
            samples = {
                n: [
                    float(x)
                    for x in bench_gpu_time(
                        fns[n],
                        dry_run_iters=10,
                        repeat_iters=args.repeats,
                        enable_cupti=True,
                        cold_l2_cache=True,
                    )
                ]
                for n in order
            }
            record["rounds"].append(samples)
            save()
        record["median_ms"] = {
            n: statistics.median(statistics.median(r[n]) for r in record["rounds"])
            for n in fns
        }
        record["speedup"] = (
            record["median_ms"]["official_default"] / record["median_ms"]["cute_auto"]
        )
        save()
        print(
            "DONE", caseid, lengths, record["median_ms"], record["speedup"], flush=True
        )
        del expected, actual
        gc.collect()
        torch.cuda.empty_cache()
    data["metadata"]["complete"] = True
    save()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
