# SPDX-License-Identifier: Apache-2.0
"""SM120 GDN forward: FLA Triton, official FI fuse/CP, and pipeline32.

All implementations receive the same inputs. Gate/state conversions remain in
FI timed callables. Imports are lazy so --help works without CUDA dependencies.
"""

import argparse
from contextlib import contextmanager
import statistics
import subprocess
import sys
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path

MODELS = {"27B TP1": (16, 48), "27B TP2": (8, 24), "Sym h32": (32, 32)}
PIPELINE_ENV = ("FLASHINFER_GDN_SM120_PIPELINE32",)


@contextmanager
def official_routes():
    """Disable the opt-in only for this benchmark, restoring embedded callers."""
    previous = {key: os.environ.get(key) for key in PIPELINE_ENV}
    try:
        for key in PIPELINE_ENV:
            os.environ[key] = "0"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["fwd"], default="fwd")
    p.add_argument("--model", choices=list(MODELS), default="27B TP1")
    p.add_argument(
        "--seqlen", default="1x", help="1x for 32K..2K; or 1x4096, 2x4096, 63+65"
    )
    p.add_argument(
        "--fi-path",
        choices=["all", "fuse", "cp", "p32", "cp32", "pipeline32"],
        default="all",
    )
    p.add_argument("--check-acc", action="store_true")
    p.add_argument(
        "--fla",
        action="store_true",
        help="also time FLA and use it as the accuracy reference; requires FLA",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=0.02,
        help="relative L2 threshold, not elementwise allclose",
    )
    p.add_argument(
        "--fla-chunk-size",
        type=int,
        choices=[32, 64],
        default=64,
        help="same FLA chunk size for accuracy and timing",
    )
    p.add_argument(
        "--backend", choices=["cupti", "cudagraph", "event"], default="cupti"
    )
    p.add_argument("--warmup", type=int, default=10, help="number of untimed calls")
    p.add_argument(
        "--repeats", type=int, default=100, help="number of timed calls/replays"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mma-load", choices=["ldmatrix", "generic"], default="ldmatrix")
    p.add_argument("--no-shared-padding", action="store_true")
    p.add_argument(
        "--json", type=Path, help="save measurements and source fingerprints"
    )
    return p


def cases(value):
    if value == "1x":
        return [[n] for n in (32768, 16384, 8192, 4096, 2048)]
    if "+" in value:
        result = [int(n) for n in value.split("+")]
    elif "x" in value:
        count, length = map(int, value.split("x"))
        if count <= 0:
            raise ValueError("sequence count must be positive")
        result = [length] * count
    else:
        result = [int(value)]
    if not result or min(result) <= 0:
        raise ValueError("benchmark sequence lengths must be positive")
    return [result]


def paths(value):
    return (
        ["fuse", "cp", "p32"]
        if value == "all"
        else ["p32" if value in ("cp32", "pipeline32") else value]
    )


def require_use_cp(fn):
    # Do not silently label a default dispatch as a forced CP/fuse baseline.
    if "use_cp" not in inspect.signature(fn).parameters:
        raise RuntimeError(
            "FlashInfer must expose use_cp explicitly to force official baselines"
        )


def relative_error(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    delta = torch.linalg.vector_norm(actual.double() - expected.double()).item()
    norm = torch.linalg.vector_norm(expected.double()).item()
    return delta / norm if norm else delta


def accepted(errors, threshold):
    return all(math.isfinite(e) and e <= threshold for e in errors)


def elapsed_ms(fn, args):
    """Measure independent cold-L2 iterations using the in-tree timing primitive."""
    from flashinfer.testing import bench_gpu_time

    if args.backend == "cupti":
        # Fail instead of silently publishing event timings under a CUPTI label.
        from cupti import cupti  # noqa: F401

        if int(importlib.metadata.version("cupti-python").split(".")[0]) < 13:
            raise RuntimeError("CUPTI >=13 is required")
    samples = bench_gpu_time(
        fn,
        dry_run_iters=args.warmup,
        repeat_iters=args.repeats,
        enable_cupti=args.backend == "cupti",
        cold_l2_cache=True,
        use_cuda_graph=args.backend == "cudagraph",
        num_iters_within_graph=1,
    )
    return statistics.median(samples)


def inputs(lengths, hq, hv):
    total = sum(lengths)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, total, hq, 128, device="cuda", dtype=torch.bfloat16).float(),
            p=2,
            dim=-1,
            eps=1e-6,
        ).to(torch.bfloat16)
        for _ in range(2)
    ]
    v = torch.randn(1, total, hv, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, total, hv, device="cuda")) / 16
    mask = torch.zeros(hv, dtype=torch.bool, device="cuda")
    mask[: math.ceil(0.75 * hv)] = True
    mask = mask[torch.randperm(hv, device="cuda")]
    g[:, :, ~mask] = 0
    beta = torch.randn(1, total, hv, device="cuda").sigmoid()
    h0 = torch.randn(len(lengths), hv, 128, 128, device="cuda")
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    return q, k, v, g, beta, h0, cu


def run(args):
    with official_routes():
        return _run(args)


def _run(args):
    global torch
    import torch
    import flashinfer

    fla_fwd = None
    if args.fla:
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_fwd
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_fwd
    from flashinfer.gdn_kernels.delta_rule_dsl import (
        delta_rule_sm120,
        delta_rule_cp_sm120,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 CUDA GPU")
    require_use_cp(fi_fwd)
    backend_kwargs = (
        {"backend": "flashinfer"}
        if "backend" in inspect.signature(fi_fwd).parameters
        else {}
    )
    p32 = None
    if "p32" in paths(args.fi_path):
        from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120_pipeline32 import (
            delta_rule_prefill_pipeline32,
        )

        p32 = delta_rule_prefill_pipeline32
    torch.manual_seed(args.seed)
    sources = [Path(delta_rule_sm120.__file__), Path(delta_rule_cp_sm120.__file__)]
    if p32:
        sources.append(Path(inspect.getfile(p32)))
        from flashinfer.gdn_kernels.delta_rule_dsl import sm120_pipeline32_layout

        sources.append(Path(sm120_pipeline32_layout.__file__))
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "source_version": (Path(flashinfer.__file__).parent.parent / "version.txt")
        .read_text()
        .strip(),
        "fla": importlib.metadata.version("flash-linear-attention")
        if args.fla
        else None,
        "cupti": importlib.metadata.version("cupti-python")
        if args.backend == "cupti"
        else None,
        "command": sys.argv,
        "base_pr_head": "38c70db5ecd10ca026d54cb8af5f423d6a6ce538",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "triton": importlib.metadata.version("triton"),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "timing": "median per-invocation GPU ms; cold L2; conversions included; CUPTI sums kernels, excludes CPU launch gaps; warmup/repeats are call counts",
    }
    print(json.dumps(metadata, indent=2))
    hq, hv = MODELS[args.model]
    rows, failures = [], 0
    for lengths in cases(args.seqlen):
        q, k, v, g, beta, h0, cu = inputs(lengths, hq, hv)
        scale = 128**-0.5

        def call_fla():
            result = fla_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu,
                chunk_size=args.fla_chunk_size,
            )
            return result[1].squeeze(0), result[3]

        reference = call_fla() if args.fla else None
        fla_ms = elapsed_ms(call_fla, args) if args.fla else None
        label = "+".join(map(str, lengths))
        print(
            f"{label}: FLA Triton {fla_ms:.3f} ms"
            if fla_ms is not None
            else f"{label}: FI-only (no FLA import)"
        )
        for path in paths(args.fi_path):

            def call_fi():
                # FLA log gate/KV state -> FI multiplicative gate/VK state.
                fq, fk, fv = q.squeeze(0), k.squeeze(0), v.squeeze(0)
                alpha = g.squeeze(0).exp().contiguous()
                fb = beta.squeeze(0).contiguous()
                initial = h0.transpose(-1, -2).contiguous()
                fi_cu = cu.to(torch.int64)
                if path == "p32":
                    output = torch.empty_like(fv)
                    final = torch.empty_like(initial)
                    p32(
                        output,
                        final,
                        fq,
                        fk,
                        fv,
                        initial,
                        alpha,
                        fb,
                        fi_cu,
                        scale,
                        mma_load=args.mma_load,
                        shared_padding=not args.no_shared_padding,
                    )
                else:
                    output, final = fi_fwd(
                        fq,
                        fk,
                        fv,
                        g=alpha,
                        beta=fb,
                        scale=scale,
                        initial_state=initial,
                        output_final_state=True,
                        cu_seqlens=fi_cu,
                        use_cp=(path == "cp"),
                        **backend_kwargs,
                    )
                return output, final.transpose(-1, -2).contiguous()

            row = {"seqlens": lengths, "path": path, "fla_ms": fla_ms}
            if args.check_acc:
                if reference is None:
                    ref_o, ref_s = fi_fwd(
                        q.squeeze(0),
                        k.squeeze(0),
                        v.squeeze(0),
                        g=g.squeeze(0).exp().contiguous(),
                        beta=beta.squeeze(0).contiguous(),
                        scale=scale,
                        initial_state=h0.transpose(-1, -2).contiguous(),
                        output_final_state=True,
                        cu_seqlens=cu.to(torch.int64),
                        use_cp=False,
                        **backend_kwargs,
                    )
                    reference = ref_o, ref_s.transpose(-1, -2).contiguous()
                actual = call_fi()
                errors = [
                    relative_error(a, r)
                    for a, r in zip(actual, reference, strict=False)
                ]
                ok = accepted(errors, args.rtol)
                row.update(output_rel_l2=errors[0], state_rel_l2=errors[1], passed=ok)
                failures += int(not ok)
                print(
                    f"[ACC {'OK' if ok else 'FAIL'}] {path}: o={errors[0]:.6f} final_state={errors[1]:.6f}"
                )
            row["fi_ms"] = elapsed_ms(call_fi, args)
            row["speedup_vs_fla"] = (
                fla_ms / row["fi_ms"] if fla_ms is not None else None
            )
            rows.append(row)
            print(f"  {path:5s}: {row['fi_ms']:.3f} ms", flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
        )
    if args.check_acc:
        print(f"[ACC] {failures} failed case(s), relative L2 threshold={args.rtol}")
    return 1 if failures else 0


if __name__ == "__main__":
    args = parser().parse_args()
    if (
        args.warmup < 1
        or args.repeats < 1
        or not math.isfinite(args.rtol)
        or args.rtol <= 0
    ):
        raise SystemExit("warmup/repeats and finite rtol must be positive")
    raise SystemExit(run(args))
