# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Native sparse MLA: identical inputs, cold 4xL2 and CUDA Graph timing.

Default: the 480-case 32K-context model suite. Use --indices 38,39 for a
small subset, --compression-ratio 128 for HCA, or --no-swa for one source.
Metadata preparation is outside timing for both backends. Finishing kernels
and counter resets stay inside. A comparator that fails accuracy is N/A.
"""

import argparse
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import torch

from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from flashinfer.testing.sparse_mla import sparse_mla_reference
from flashinfer.testing.sparse_mla_metadata import (
    map_sparse_indices,
    prepare_sparse_mla_metadata,
)
from sparse_mla_bench_utils import ColdL2GraphBenchmark


def model_fixture(
    batch,
    heads,
    queries,
    topk,
    dtype,
    *,
    raw_tokens=32768,
    compression_ratio=1,
    seed=2026,
    device="cuda",
):
    score_chunk = 128
    if min(batch, heads, queries, topk, compression_ratio) <= 0:
        raise ValueError("dimensions and compression ratio must be positive")
    if raw_tokens < queries + 127:
        raise ValueError("the query suffix must have a full preceding SWA window")
    candidate_rows = raw_tokens // compression_ratio
    if candidate_rows == 0:
        raise ValueError("compression ratio leaves no candidate entries")
    g = torch.Generator(device=device).manual_seed(seed)

    def normal_native(shape, scale=1.0):
        result = torch.empty(shape, dtype=dtype, device=device)
        flat = result.view(-1, 512)
        # Avoid a full FP32 copy of a multi-GB native cache.
        for start in range(0, flat.shape[0], 32768):
            dst = flat[start : start + 32768]
            value = torch.randn(dst.shape, generator=g, device=device)
            value.mul_(scale).clamp_(-1, 1)
            dst.copy_(value)
        return result

    query = normal_native((batch, queries, heads, 512))
    sinks = torch.randn(heads, generator=g, device=device)
    sink_mask = torch.randn(heads, generator=g, device=device)
    sinks[sink_mask > 0.5] = torch.inf
    sinks[sink_mask < -0.5] = -torch.inf

    swa_page = 256
    swa_rows = math.ceil((queries + 127) / swa_page) * swa_page
    swa_table = (
        torch.randperm(batch * swa_rows // swa_page, generator=g, device=device)
        .to(torch.int32)
        .view(batch, -1)
    )
    swa = normal_native((batch * swa_rows // swa_page, swa_page, 512), 0.1)
    local_swa = (
        torch.arange(queries, device=device, dtype=torch.int32)[:, None]
        + torch.arange(128, device=device, dtype=torch.int32)[None, :]
    )
    si = (
        swa_table[:, local_swa // swa_page] * swa_page
        + local_swa[None, :, :] % swa_page
    ).contiguous()
    sl = torch.full((batch, queries), 128, dtype=torch.int32, device=device)

    primary_table = (
        torch.randperm(batch * candidate_rows, generator=g, device=device)
        .to(torch.int32)
        .view(batch, candidate_rows)
    )
    compressed = normal_native((batch * candidate_rows, 1, 512), 0.1)
    query_lengths = (
        raw_tokens
        - queries
        + torch.arange(queries, device=device, dtype=torch.int32)
        + 1
    )
    available = query_lengths // compression_ratio
    ci = torch.empty((batch, queries, topk), dtype=torch.int32, device=device)
    columns = max(candidate_rows, topk)
    positions = torch.arange(columns, device=device)
    for request in range(batch):
        for start in range(0, queries, score_chunk):
            end = min(queries, start + score_chunk)
            scores = torch.rand((end - start, columns), generator=g, device=device)
            scores.masked_fill_(
                positions[None, :] >= available[start:end, None], -torch.inf
            )
            logical = scores.topk(topk, dim=-1, sorted=True).indices
            valid = logical < available[start:end, None]
            physical = primary_table[request, logical.masked_fill(~valid, 0)]
            ci[request, start:end] = physical.masked_fill(~valid, -1)
    cl = available.clamp(max=topk)[None, :].expand(batch, queries).contiguous()

    # Poison unused physical rows, exactly accounting for all query selections.
    for cache, indices in ((swa, si), (compressed, ci)):
        flat = cache.view(-1, 512)
        unused = torch.ones(flat.shape[0], dtype=torch.bool, device=device)
        used = indices.reshape(-1)
        unused[used[used >= 0].long()] = False
        if dtype == torch.float8_e4m3fn:
            # Torch 2.8 lacks masked_fill for native FP8. E4M3FN 0x7f is
            # NaN; write the same native values through their byte view.
            flat.view(torch.uint8)[unused] = 0x7F
        else:
            flat[unused] = torch.nan

    fixture = dict(
        query=query,
        swa=swa,
        compressed=compressed,
        si=si,
        ci=ci,
        sl=sl,
        cl=cl,
        sinks=sinks,
        seq_lens=torch.full((batch,), raw_tokens, dtype=torch.int32, device=device),
        softmax_scale=512**-0.55,
    )
    return SimpleNamespace(**fixture)


def cases():
    """Stable IDs for 30 prefill and 450 decode configurations."""
    result = []
    for phase, batches, queries in (
        ("prefill", [1], [8192]),
        ("decode", [1, 4, 16, 64, 256], [1, 4, 8]),
    ):
        for b, q, k, h, dtype in itertools.product(
            batches, queries, [512, 1024, 2048], [8, 16, 32, 64, 128], ["bf16", "fp8"]
        ):
            result.append(
                dict(
                    id=len(result),
                    phase=phase,
                    batch=b,
                    queries=q,
                    heads=h,
                    topk=k,
                    dtype=dtype,
                )
            )
    return result


def make_backend(name, f, *, no_swa=False):
    q = f.query
    batch, queries, heads, _ = q.shape
    out = torch.empty_like(q, dtype=torch.bfloat16)
    if name == "ts-auto":
        w = BatchSparseMLADecodePagedTSWrapper()
        w.plan(
            q.device,
            batch,
            heads,
            max_seq_len_q=queries,
            q_data_type=q.dtype,
            max_topk=f.ci.shape[-1] if no_swa else 128,
            max_extra_topk=0 if no_swa else f.ci.shape[-1],
            has_sinks=True,
            assume_valid_prefix=True,
        )
        primary, extra = (f.compressed, None) if no_swa else (f.swa, f.compressed)
        metadata = prepare_sparse_mla_metadata(
            w,
            q,
            primary,
            f.ci if no_swa else f.si,
            f.cl if no_swa else f.sl,
            extra_kv_cache=extra,
            extra_indices=None if no_swa else f.ci,
            extra_lengths=None if no_swa else f.cl,
            sinks=f.sinks,
            softmax_scale=f.softmax_scale,
        )

        def run():
            return w.run(
                q,
                primary,
                metadata,
                extra,
                out=out,
                validate=False,
                sinks=f.sinks,
                softmax_scale=f.softmax_scale,
            )
    else:
        assert name == "trtllm-gen"
        workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
        si, _ = map_sparse_indices(
            f.si.view(batch * queries, -1),
            f.sl.view(-1),
            page_size=256,
            page_stride_rows=f.swa.stride(0) // 512,
        )
        ci, cl = map_sparse_indices(
            f.ci.view(batch * queries, -1),
            f.cl.view(-1),
            page_size=1,
            page_stride_rows=f.compressed.stride(0) // 512,
        )
        # TRT's fixed 128-entry SWA segment remains invalid for no-SWA runs.
        indices, lengths = torch.cat((si, ci), dim=-1), 128 + cl
        swa, compressed = f.swa.unsqueeze(1), f.compressed.unsqueeze(1)

        def run():
            return trtllm_batch_decode_sparse_mla_dsv4(
                q,
                swa,
                workspace,
                sparse_indices=indices,
                compressed_kv_cache=compressed,
                sparse_topk_lens=lengths,
                seq_lens=f.seq_lens,
                sinks=f.sinks,
                out=out,
                bmm1_scale=f.softmax_scale,
                bmm2_scale=1.0,
                kv_layout="HND",
                backend="trtllm-gen",
                sparse_indices_are_storage_offsets=True,
            )

    return run, out


def run_case(case, args):
    f = model_fixture(
        case["batch"],
        case["heads"],
        case["queries"],
        case["topk"],
        torch.bfloat16 if case["dtype"] == "bf16" else torch.float8_e4m3fn,
        raw_tokens=args.raw_kv_tokens,
        compression_ratio=args.compression_ratio,
        seed=args.seed,
    )
    if args.no_swa:
        f.si.fill_(-1)
        f.sl.zero_()
        if f.swa.dtype == torch.float8_e4m3fn:
            f.swa.view(torch.uint8).fill_(0x7F)
        else:
            f.swa.fill_(torch.nan)
    expected, _, bound = sparse_mla_reference(
        f.query,
        f.swa,
        f.compressed,
        f.si,
        f.ci,
        swa_topk_lens=f.sl,
        compressed_topk_lens=f.cl,
        sinks=f.sinks,
        softmax_scale=f.softmax_scale,
        return_fp8_error_bound=True,
    )
    if f.query.dtype == torch.bfloat16:
        bound = expected.abs() * 0.02 + 8e-4

    def check(out):
        if not ((out.double() - expected).abs() <= bound).all().item():
            raise AssertionError(
                "FP64 forward-error bound exceeded (or nonfinite output)"
            )

    runners, outputs, results = {}, {}, {}
    for name in ("ts-auto", "trtllm-gen"):
        try:
            fn, out = make_backend(name, f, no_swa=args.no_swa)
            fn()
            check(out)
            runner = ColdL2GraphBenchmark(
                fn, device=f.query.device, samples_per_replay=4
            )
            runner.sample()
            check(out)
            runners[name], outputs[name] = runner, out
        except (RuntimeError, NotImplementedError, AssertionError) as error:
            if name == "ts-auto":
                raise
            torch.cuda.synchronize()  # A poisoned CUDA context is fatal.
            results[name] = dict(status="failed", error=str(error))
    samples = {name: [] for name in runners}
    for replay in range(args.replays):
        order = list(runners) if replay % 2 == 0 else list(reversed(runners))
        for name in order:
            samples[name].extend(runners[name].sample())
    for name, times in samples.items():
        try:
            check(outputs[name])
        except AssertionError as error:
            if name == "ts-auto":
                raise
            results[name] = dict(status="failed", error=str(error))
        else:
            ordered = sorted(times)
            results[name] = dict(
                status="ok",
                median_us=statistics.median(times),
                p10_us=ordered[int(0.1 * (len(ordered) - 1))],
                p90_us=ordered[int(0.9 * (len(ordered) - 1))],
            )
    ts, trt = results["ts-auto"], results["trtllm-gen"]
    speedup = trt["median_us"] / ts["median_us"] if trt["status"] == "ok" else None
    return dict(**case, backends=results, speedup=speedup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices", help="Comma-separated IDs from the fixed model suite"
    )
    parser.add_argument("--raw-kv-tokens", type=int, default=32768)
    parser.add_argument("--compression-ratio", type=int, default=1)
    parser.add_argument("--no-swa", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    if min(args.replays, args.raw_kv_tokens, args.compression_ratio) < 1:
        parser.error("replays, context and compression ratio must be positive")
    matrix = cases()
    if args.indices:
        indices = {int(x) for x in args.indices.split(",")}
        if not indices <= set(range(len(matrix))):
            parser.error("case IDs must be in [0, 480)")
        matrix = [c for c in matrix if c["id"] in indices]
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        seed=args.seed,
        raw_kv_tokens=args.raw_kv_tokens,
        compression_ratio=args.compression_ratio,
        swa_window=0 if args.no_swa else 128,
        replays=args.replays,
        samples_per_replay=4,
        protocol="same-input/prepared-attention/cold-4xL2/CUDA-Graph",
        cases=[],
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in matrix:
        result = run_case(case, args)
        report["cases"].append(result)
        times = {
            n: round(r["median_us"], 3) if r["status"] == "ok" else "N/A: " + r["error"]
            for n, r in result["backends"].items()
        }
        print(f"{case}: {times}; speedup={result['speedup']}", flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
