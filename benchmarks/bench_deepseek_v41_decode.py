# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Quantized-cache FP64 gate and paired CUDA Graph timing of DS4.1 decode.

python benchmarks/bench_deepseek_v41_decode.py --batches 1,4,16,32,64,128,256 \
    --context 32768 --flashmla --output results.json

FlashMLA is an optional external benchmark dependency, not a shipped decode
backend. Quantization, allocation, JIT and host overhead are excluded from GPU
timing. The timed CuTe invocation includes the public plan replay and merge.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import traceback

import torch

from flashinfer.deepseek_v41 import deepseek_v41_decode, deepseek_v41_quantize_cache


def selected(pool, indices, fp4):
    pages = pool.view(pool.shape[0], -1)
    ids = indices[:, 0].long()
    page, row = ids // 64, ids % 64
    column = torch.arange(512, device=ids.device)
    if fp4:
        data = pages[page[..., None], row[..., None] * 256 + column // 2]
        code = (data >> (4 * (column % 2))) & 15
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=ids.device,
            dtype=torch.float64,
        )
        data = lut[code.long()]
        scale = pages[page[..., None], 64 * 256 + row[..., None] * 32 + column // 16]
        scale = scale.contiguous().view(torch.float8_e4m3fn).double()
    else:
        data = pages[page[..., None], row[..., None] * 512 + column]
        data = data.contiguous().view(torch.float8_e4m3fn).double()
        scale = pages[page[..., None], 64 * 512 + row[..., None] * 16 + column // 32]
        scale = torch.exp2(scale.double() - 127)
    return (data * scale).bfloat16().double()


def fixture(batch, context, seed):
    torch.manual_seed(seed + batch)
    q = torch.randn(batch, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(batch * 128, 512, device="cuda", dtype=torch.bfloat16)
    swa = deepseek_v41_quantize_cache(w, format="swa_mxfp8")
    main = torch.empty(
        batch * context // 64, 64, 1, 288, device="cuda", dtype=torch.uint8
    )
    ci = torch.empty(batch, 1, 512, device="cuda", dtype=torch.int32)
    for b in range(batch):
        values = torch.randn(context, 512, device="cuda", dtype=torch.bfloat16)
        cache = deepseek_v41_quantize_cache(values, format="main_kv_fp4")
        main[b * (context // 64) : (b + 1) * (context // 64)].copy_(cache)
        ci[b, 0] = (
            torch.randperm(context, device="cuda")[:512].sort().values + b * context
        )
    wi = (
        (
            torch.arange(batch, device="cuda")[:, None, None] * 128
            + torch.arange(128, device="cuda")[None, None]
        )
        .int()
        .contiguous()
    )
    sink = torch.randn(64, device="cuda") * 0.1
    kv = torch.cat((selected(swa, wi, False), selected(main, ci, True)), dim=1)
    scores = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * 512**-0.5
    aug = torch.cat((scores, sink.double()[None, :, None].expand(batch, -1, 1)), dim=-1)
    expected = torch.einsum("bhk,bkd->bhd", aug.softmax(-1)[..., :-1], kv)
    return (q, swa, main, wi, ci, sink), expected, scores.logsumexp(-1)


def graph(fn):
    for _ in range(4):
        fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for _ in range(32):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    return g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,4,16,32,64,128,256")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=7003)
    parser.add_argument("--flashmla", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    if opts.output.exists():
        raise FileExistsError(opts.output)
    if opts.context < 512 or opts.context % 64:
        raise ValueError("context must be a multiple of64 >=512")
    from flashinfer.experimental.deepseek_v41 import (
        cache,
        decode,
        hca_v41,
        hca_v41_primitives,
    )

    report = {
        "status": "running",
        "device": str(torch.cuda.get_device_properties(0)),
        "torch": torch.__version__,
        "context_per_request": opts.context,
        "shape": "H64/Sq1/D512; SWA128 MXFP8 + selected512 FP4",
        "seed": opts.seed,
        "timing": "CUDA Graph/event, 32 calls/replay, symmetric order, 30 samples/arm",
        "scope": "decode+merge GPU latency; excludes preparation/JIT/capture/host time",
        "sha256": {
            Path(m.__file__).name: hashlib.sha256(
                Path(m.__file__).read_bytes()
            ).hexdigest()
            for m in (cache, decode, hca_v41, hca_v41_primitives)
        },
        "rows": [],
    }
    try:
        with torch.no_grad():
            for batch in map(int, opts.batches.split(",")):
                args, expected, expected_lse = fixture(batch, opts.context, opts.seed)
                out, lse, plan = deepseek_v41_decode(*args)
                fns = {
                    "cute": lambda args=args, plan=plan: deepseek_v41_decode(
                        *args, plan=plan
                    )[:2]
                }
                if opts.flashmla:
                    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

                    meta, _ = get_mla_metadata()

                    def flashmla(args=args, meta=meta):
                        q, w, c, wi, ci, sink = args
                        return flash_mla_with_kvcache(
                            q,
                            w,
                            None,
                            None,
                            512,
                            meta,
                            softmax_scale=512**-0.5,
                            is_fp8_kvcache=True,
                            indices=wi,
                            attn_sink=sink,
                            extra_k_cache=c,
                            extra_indices_in_kvcache=ci,
                        )

                    fns["flashmla"] = flashmla
                stats = {}
                for name, fn in fns.items():
                    o, l = fn()
                    difference = o[:, 0].double() - expected
                    stats[name] = {
                        "rel_l2": float(difference.norm() / expected.norm()),
                        "max_scaled": float(
                            difference.abs().max() / expected.abs().max()
                        ),
                        "lse_abs": float(
                            (l[..., 0].double() - expected_lse).abs().max()
                        ),
                    }
                    st = stats[name]
                    if not (
                        torch.isfinite(o).all()
                        and st["rel_l2"] < 0.005
                        and st["max_scaled"] < 0.01
                        and st["lse_abs"] < 1e-5
                    ):
                        torch.save(
                            {
                                "out": o.cpu(),
                                "lse": l.cpu(),
                                "expected": expected.cpu(),
                                "expected_lse": expected_lse.cpu(),
                            },
                            opts.output.with_suffix(".failure.pt"),
                        )
                        raise AssertionError((batch, name, st))
                graphs = {name: graph(fn) for name, fn in fns.items()}
                samples = {name: [] for name in fns}
                for _ in range(15):
                    for name in list(fns) + list(reversed(fns)):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000 / 32)
                row = {
                    "batch": batch,
                    "split": plan.split,
                    "correctness": stats,
                    "median_us": {k: statistics.median(v) for k, v in samples.items()},
                    "samples_us": samples,
                }
                report["rows"].append(row)
                opts.output.write_text(json.dumps(report, indent=2))
                print({k: row[k] for k in ("batch", "split", "median_us")}, flush=True)
                del graphs, fns, args, plan, expected, expected_lse, out, lse
            report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["error"] = traceback.format_exc()
        raise
    finally:
        opts.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
