# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Quantized-cache FP64 gates and paired CUDA Graph timing of DS4.1 backends.

python benchmarks/bench_deepseek_v41_decode.py --batches 1,4,16,32,64,128,256 \
    --backends cute-dsl prims-ts --context 32768 --output results.json

The default cute-dsl backend keeps BF16 Q/MXFP8 SWA, tuple returns and a
reusable plan. Its timed invocation includes the public plan replay and merge.
FlashMLA remains an optional external BF16-Q/MXFP8 benchmark dependency,
enabled only by --flashmla, not a shipped backend. Both legacy arms keep their
original FP64 reference and strict output/LSE gates.

Only explicit prims-ts uses E4M3 Q and typed E4M3 SWA, with E4M3 Q/K/V/P
attention for both sources. Its caches use common units and one runtime BMM
pair. Native main applies its original group16 E4M3 scales, then directly
rounds/saturates to E4M3, without amax, request descales or division. Values
outside +/-448 cache units intentionally saturate. No BF16 query expansion or
full converted cache is used. The native fixture uses common KV descale 1 and
explicitly passes 512**-0.5 for BMM1; its V4-style public default is 1.0.
Native and legacy arms have different represented inputs and arithmetic, not
identical precision. All derive from common seeded BF16 sources and share one
packed global FP4 pool/indices. The native gate uses its independent FP64
represented-input oracle, componentwise output bound and rtol=0/atol=1e-5 LSE.
Its caller-owned workspace/output/combined metadata are prepared outside
timing; replay includes inline live-metadata masking, selected-tile staging
and joint attention/finalization. Cache writing, allocation, JIT, capture and
host overhead are excluded for every arm.

Prims-TS requires CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler, split 1 and
a nonpersistent schedule. Historical GPU results do not qualify the native
joint/shared-scale/caller-workspace revision. This benchmark revision has not
run; native performance and backend crossovers are unmeasured.
See examples/deepseek_v41/README.md for scope and reproducer commands.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import traceback

import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_decode,
    deepseek_v41_decode_workspace_size,
    deepseek_v41_quantize_cache,
)


PRIMS_TS_SOURCES = (
    "experimental/deepseek_v41/prims_ts/_sparse_views.py",
    "experimental/deepseek_v41/prims_ts/_workspace.py",
    "experimental/deepseek_v41/hca_v41_primitives.py",
    "experimental/deepseek_v41/prims_ts/__init__.py",
    "experimental/deepseek_v41/prims_ts/kernel.py",
    "experimental/deepseek_v41/prims_ts/resources.py",
    "cute_dsl/fp4_common.py",
    "attention/prims_ts/kernels/mla_decode/helpers/ops.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/kernel.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/tasks.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/config.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/parallel_reduction.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/smem_resources.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_corr.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_p.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_s.py",
    "attention/prims_ts/kernels/mla_decode/helpers/sparse_indices.py",
    "attention/prims_ts/kernels/task_compat.py",
    "attention/prims_ts/kernels/separate_reduction.py",
)


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


def fixture(batch, context, seed, backends, flashmla=False):
    torch.manual_seed(seed + batch)
    q_source = torch.randn(batch, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    w_source = torch.randn(batch * 128, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.empty(
        batch * context // 64, 64, 1, 288, device="cuda", dtype=torch.uint8
    )
    ci = torch.empty(batch, 1, 512, device="cuda", dtype=torch.int32)
    for b in range(batch):
        values = torch.randn(context, 512, device="cuda", dtype=torch.bfloat16)
        cache = deepseek_v41_quantize_cache(
            values, format="main_kv_fp4", kv_layout="NHD"
        )
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
    # Q/SWA differ at rest; main arithmetic differs despite shared packed storage.
    # Do not allocate a second production-sized global pool or converted cache.
    shared = main, wi, ci, sink
    arms = {}
    if "cute-dsl" in backends or flashmla:
        legacy = (
            q_source,
            deepseek_v41_quantize_cache(w_source, format="swa_mxfp8", kv_layout="NHD"),
            *shared,
        )
        if "cute-dsl" in backends:
            arms["cute-dsl"] = legacy
        if flashmla:
            arms["flashmla"] = legacy
    if "prims-ts" in backends:
        arms["prims-ts"] = (
            q_source.to(torch.float8_e4m3fn),
            deepseek_v41_quantize_cache(w_source, format="swa_fp8", kv_layout="NHD"),
            *shared,
        )
    return arms


def legacy_reference(q, swa, main, wi, ci, sink):
    kv = torch.cat((selected(swa, wi, False), selected(main, ci, True)), dim=1)
    scores = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * 512**-0.5
    aug = torch.cat(
        (scores, sink.double()[None, :, None].expand(q.shape[0], -1, 1)), dim=-1
    )
    expected = torch.einsum("bhk,bkd->bhd", aug.softmax(-1)[..., :-1], kv)
    return expected, scores.logsumexp(-1)


def common_unit_main(decoded):
    """Independent common-unit E4M3 staging, without amax or a request descale."""
    return decoded.float().clamp(-448, 448).to(torch.float8_e4m3fn).double()


def selected_main_fp8(pool, safe, valid):
    """Gather/mask selected bytes before decoding; never convert the full cache."""
    pages = pool.view(pool.shape[0], -1)
    page, row = safe // 64, safe % 64
    column = torch.arange(512, device=safe.device)
    payload = pages[page[..., None], row[..., None] * 256 + column // 2]
    payload = torch.where(valid[..., None], payload, 0)
    code = (payload >> (4 * (column % 2))) & 15
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=safe.device,
        dtype=torch.float32,
    )
    data = lut[code.long()]
    scale = pages[page[..., None], 64 * 256 + row[..., None] * 32 + column // 16]
    scale = torch.where(valid[..., None], scale, 0)
    scale = scale.contiguous().view(torch.float8_e4m3fn).float()
    decoded = data * scale  # E2M1 * finite E4M3 is exact in float32.
    return common_unit_main(decoded)


def fp8_oracle(
    q, swa, global_cache, wi, ci, sink, *, bmm1_scale=512**-0.5, bmm2_scale=1.0
):
    """FP64 represented-input oracle and unchanged componentwise FP8 bound.

    Gather/mask selected main bytes before exact E2M1*group16 E4M3 decode,
    then independently clamp/round to E4M3 in common cache units. Shared BMM1
    scales logits and BMM2 scales output; there is no request amax or division.
    The attention/bound calculation matches the independent native GPU tests;
    no production conversion helper or source state is reused.
    """
    values, masks = [], []
    for pool, ids, fp4 in ((swa, wi, False), (global_cache, ci, True)):
        if pool is None:
            continue
        index = ids[:, 0].long()
        valid = (index >= 0) & (index < pool.shape[0] * 64) & (index != 0x7FFFFFFF)
        safe = index.clamp(0, pool.shape[0] * 64 - 1)
        if not fp4:
            gathered = pool.double().reshape(-1, 512)[safe]
        else:
            gathered = selected_main_fp8(pool, safe, valid)
        values.append(torch.where(valid[..., None], gathered, 0))
        masks.append(valid)
    kv, valid = torch.cat(values, 1), torch.cat(masks, 1)
    logits = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * bmm1_scale
    logits.masked_fill_(~valid[:, None], -math.inf)
    kv_lse = logits.logsumexp(-1)
    normalizer = torch.logaddexp(kv_lse, sink.double()[None, :])
    weights = torch.exp(logits - normalizer[..., None])
    weights = torch.where(torch.isfinite(normalizer)[..., None], weights, 0)
    expected = torch.einsum("bhk,bkd->bhd", weights, kv) * bmm2_scale
    sensitivity = torch.einsum("bhk,bkd->bhd", weights, kv.abs()) * abs(bmm2_scale)
    max_logit = logits.max(-1).values
    frame_mass = torch.exp(max_logit - normalizer)
    frame_mass = torch.where(torch.isfinite(kv_lse), frame_mass, 0)
    underflow = frame_mass[..., None] * kv.abs().sum(1)[:, None, :] * abs(bmm2_scale)
    # The same componentwise bound now covers one joint online-softmax pass.
    bound = (2**-4 + 3 * 2**-8 + 1e-5) * sensitivity + (2**-10 / 448) * underflow + 1e-6
    bound = torch.where(torch.isfinite(kv_lse)[..., None], bound, 0)
    return expected, kv_lse, bound


def check_fp8(out, lse, reference):
    expected, expected_lse, bound = reference
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert ((out[:, 0].double() - expected).abs() <= bound).all()
    empty = torch.isneginf(expected_lse)
    assert torch.equal(torch.isneginf(lse[..., 0]), empty)
    torch.testing.assert_close(
        lse[..., 0].double()[~empty], expected_lse[~empty], rtol=0, atol=1e-5
    )
    assert torch.equal(out[:, 0][empty], torch.zeros_like(out[:, 0][empty]))
    assert torch.equal(bound[empty], torch.zeros_like(bound[empty]))


def check_legacy(out, stats):
    assert (
        torch.isfinite(out).all()
        and stats["rel_l2"] < 0.005
        and stats["max_scaled"] < 0.01
        and stats["lse_abs"] < 1e-5
    )


def check_arm(name, args, fn, batch, output):
    native_fp8 = name == "prims-ts"
    reference = fp8_oracle(*args) if native_fp8 else legacy_reference(*args)
    expected, expected_lse = reference[:2]
    out, lse = fn()
    difference = out[:, 0].double() - expected
    if native_fp8:
        nonempty = ~torch.isneginf(expected_lse)
        stats = {
            # Ratios are diagnostic only; the native gate is componentwise.
            "rel_l2": float(difference.norm() / expected.norm().clamp_min(1e-30)),
            "max_scaled": float(
                difference.abs().max() / expected.abs().max().clamp_min(1e-30)
            ),
            "lse_abs": (
                float(
                    (lse[..., 0].double()[nonempty] - expected_lse[nonempty])
                    .abs()
                    .max()
                )
                if nonempty.any()
                else 0.0
            ),
            "max_bound_fraction": float(
                (difference.abs() / reference[2].clamp_min(1e-30)).max()
            ),
        }
    else:
        # Preserve the original legacy metric formulas and output limits.
        stats = {
            "rel_l2": float(difference.norm() / expected.norm()),
            "max_scaled": float(difference.abs().max() / expected.abs().max()),
            "lse_abs": float((lse[..., 0].double() - expected_lse).abs().max()),
        }
    try:
        if native_fp8:
            check_fp8(out, lse, reference)
        else:
            check_legacy(out, stats)
    except AssertionError as error:
        torch.save(
            {
                "out": out.cpu(),
                "lse": lse.cpu(),
                "expected": expected.cpu(),
                "expected_lse": expected_lse.cpu(),
                "bound": reference[2].cpu() if native_fp8 else None,
            },
            output.with_suffix(".failure.pt"),
        )
        raise AssertionError((batch, name, stats)) from error
    return stats


def arm_metadata(name, args):
    q, swa, main, _, _, _ = args
    native_fp8 = name == "prims-ts"
    contract = {"reference": "independent FP64 over this arm's represented Q/KV"}
    if native_fp8:
        arithmetic = (
            "native E4M3 Q/K/V/P for SWA and main; "
            "BK128/Keep-AB/M64/retained8/split1/nonpersistent; "
            "original E4M3 Q, no BF16 query scratch"
        )
        global_conversion = (
            "common cache units: valid selected E2M1 * signed finite group16 E4M3 "
            "-> E4M3 RN/satfinite SMEM; intentional +/-448 cache-unit saturation; "
            "no request amax, descale or division; shared runtime BMM1/BMM2"
        )
        contract.update(
            reference=(
                "independent FP64 over stored E4M3 Q/SWA and common-unit E4M3 main; "
                "mask selected packed bytes before exact decode, then independent "
                "FP32 clamp -> E4M3 RN -> FP64; shared BMM factors applied once"
            ),
            output="componentwise native FP8 P-rounding bound (not legacy relative gates)",
            componentwise_bound=(
                "(2^-4 + 3*2^-8 + 1e-5)*sensitivity + "
                "(2^-10/448)*underflow + 1e-6; exactly zero for empty rows"
            ),
            lse="sink-exclusive; rtol=0, atol=1e-5",
            empty="exact zero output, negative-infinity LSE and zero bound",
        )
    else:
        global_conversion = "scaled E2M1 -> BF16 (unchanged legacy represented values)"
        if name == "cute-dsl":
            arithmetic = "Frost BF16 Q/dequantized MXFP8 SWA and FP4 global"
        else:
            arithmetic = (
                "external FlashMLA legacy BF16-Q/MXFP8-SWA arm, not Prims-TS native FP8"
            )
        contract.update(
            rel_l2_lt=0.005,
            max_scaled_lt=0.01,
            sink_exclusive_lse_abs_lt=1e-5,
        )
    metadata = {
        "q_dtype": str(q.dtype),
        "swa_encoding": "swa_fp8" if native_fp8 else "swa_mxfp8",
        "global_encoding": "main_kv_fp4",
        "global_storage": "packed E2M1 + group16 E4M3 scales; width288 page-plane",
        "global_conversion": global_conversion,
        "cache_shape": {"swa": list(swa.shape), "global_fp4": list(main.shape)},
        "arithmetic": arithmetic,
        "accuracy_contract": contract,
        "kv_layout": "NHD",
    }
    if native_fp8:
        metadata.update(
            bmm1_scale=512**-0.5,
            bmm2_scale=1.0,
            global_precision="shared_pertensor_e4m3",
            main_scale="none; caller's common KV units, shared runtime BMM factors",
            compile_marker="dsv41_shared_pertensor_joint_e4m3_swa512_fp4_inline_metadata_v2",
        )
    return metadata


def prepare_decode(args, backend):
    """Reuse the legacy plan, or own the explicit native V4 replay buffers."""
    if backend == "cute-dsl":
        _, _, plan = deepseek_v41_decode(*args)

        def call():
            return deepseek_v41_decode(*args, plan=plan)[:2]

        return call, {"split": plan.split}
    if backend != "prims-ts":
        raise ValueError("backend must be cute-dsl or prims-ts")

    from flashinfer.experimental.deepseek_v41.prims_ts._workspace import (
        workspace_views,
    )

    q, swa, main, wi, ci, sink = args
    batch, compressed_k = q.shape[0], ci.shape[-1]
    indices = torch.cat((wi[:, 0], ci[:, 0]), dim=1)
    topk_lens = torch.full(
        (batch,), 128 + compressed_k, device=q.device, dtype=torch.int32
    )
    seq_lens = torch.full((batch,), 128, device=q.device, dtype=torch.int32)
    workspace = torch.empty(
        deepseek_v41_decode_workspace_size(batch, compressed_k, backend=backend),
        device=q.device,
        dtype=torch.uint8,
    )
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    lse = workspace_views(workspace, batch, compressed_k, backend=backend).lse

    def call():
        result = deepseek_v41_decode(
            q,
            swa,
            workspace,
            indices,
            main,
            topk_lens,
            seq_lens,
            out,
            bmm1_scale=512**-0.5,
            bmm2_scale=1.0,
            sinks=sink,
            kv_layout="NHD",
            backend=backend,
        )
        # Native LSE stays private scratch; the supplied output is returned as-is.
        assert result is out
        return result, lse

    return call, {"workspace_bytes": workspace.numel(), "split": 1}


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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,4,16,32,64,128,256")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=7003)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cute-dsl", "prims-ts"),
        default=["cute-dsl"],
        help="prims-ts uses E4M3 Q/SWA in common KV units; cute-dsl retains BF16 Q/MXFP8 SWA",
    )
    parser.add_argument("--flashmla", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    opts = build_parser().parse_args()
    if opts.output.exists():
        raise FileExistsError(opts.output)
    if opts.context < 512 or opts.context % 64:
        raise ValueError("context must be a multiple of64 >=512")
    from flashinfer import deepseek_v41

    # Locate sources from the lightweight public module, not a Frost import.
    package_root = Path(deepseek_v41.__file__).resolve().parent
    relative_sources = [
        "deepseek_v41.py",
        "experimental/deepseek_v41/cache.py",
        "experimental/deepseek_v41/_utils.py",
    ]
    if "cute-dsl" in opts.backends:
        relative_sources.extend(
            (
                "experimental/deepseek_v41/decode.py",
                "experimental/deepseek_v41/hca_v41.py",
                "experimental/deepseek_v41/hca_v41_primitives.py",
            )
        )
    if "prims-ts" in opts.backends:
        relative_sources.extend(PRIMS_TS_SOURCES)
    source_paths = {
        f"flashinfer.{relative[:-3].replace('/', '.')}": package_root / relative
        for relative in relative_sources
    }
    report = {
        "status": "running",
        "implementation": "deepseek_v41",
        "backends": opts.backends,
        "device": str(torch.cuda.get_device_properties(0)),
        "torch": torch.__version__,
        "context_per_request": opts.context,
        "shape": (
            "H64/Sq1/D512; SWA128 + selected512 packed FP4; encodings/shapes per arm"
        ),
        "comparison": (
            "common seeded BF16 source Q/SWA; one shared packed global FP4 pool "
            "and indices; native FP8 (when selected) directly stages common-unit "
            "E4M3 main with intentional saturation, without amax or division; "
            "legacy represented main values remain BF16 with their original "
            "scales; native runtime BMM pair (512**-0.5, 1.0); arm-specific "
            "arithmetic and accuracy contracts"
        ),
        "seed": opts.seed,
        "timing": "CUDA Graph/event, 32 calls/replay, symmetric order, 30 samples/arm",
        "scope": (
            "public replay GPU latency: legacy plan replay/merge; native inline "
            "live-metadata masking, selected-tile staging and joint finalization; "
            "excludes cache writing/preparation/JIT/allocation/capture/host time"
        ),
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_paths.items()
        },
        "rows": [],
    }
    if "prims-ts" in opts.backends:
        import cutlass

        report["toolchain"] = {
            "cutlass_dsl": cutlass.__version__,
            "cutlass_dsl_package": importlib.metadata.version("nvidia-cutlass-dsl"),
            "cutlass_module": str(Path(cutlass.__file__).resolve()),
            "cute_bundled_cuda": (
                f"{cutlass.CUDA_VERSION.major}.{cutlass.CUDA_VERSION.minor}"
            ),
            "torch_cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
        }
    try:
        with torch.no_grad():
            for batch in map(int, opts.batches.split(",")):
                arm_args = fixture(
                    batch,
                    opts.context,
                    opts.seed,
                    opts.backends,
                    opts.flashmla,
                )
                buffers, fns = {}, {}
                for backend in opts.backends:
                    fns[backend], buffers[backend] = prepare_decode(
                        arm_args[backend], backend
                    )
                # Replay the original legacy plan or the complete native call;
                # plans and native caller buffers are prepared outside timing.
                if opts.flashmla:
                    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

                    # FlashMLASchedMeta is initialized from this fixture by
                    # the eager correctness call below, before graph capture.
                    # The second return value is a None compatibility placeholder.
                    meta, _ = get_mla_metadata()

                    def flashmla(args=arm_args["flashmla"], meta=meta):
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
                    del flashmla
                stats = {
                    name: check_arm(name, arm_args[name], fn, batch, opts.output)
                    for name, fn in fns.items()
                }
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
                    "split": {name: info["split"] for name, info in buffers.items()},
                    "arms": {name: arm_metadata(name, arm_args[name]) for name in fns},
                    "correctness": stats,
                    "median_us": {k: statistics.median(v) for k, v in samples.items()},
                    "samples_us": samples,
                }
                if "prims-ts" in buffers:
                    row["workspace_bytes"] = {
                        "prims-ts": buffers["prims-ts"]["workspace_bytes"]
                    }
                report["rows"].append(row)
                opts.output.write_text(json.dumps(report, indent=2))
                print({k: row[k] for k in ("batch", "split", "median_us")}, flush=True)
                del graphs, fns, buffers, arm_args
                if opts.flashmla:
                    del meta
            report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["error"] = traceback.format_exc()
        raise
    finally:
        opts.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
