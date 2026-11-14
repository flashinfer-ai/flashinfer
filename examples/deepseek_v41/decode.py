# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DS4.1 decode with backend-specific GPU support.

The default cute-dsl example is SM100-only and keeps the original BF16 Q/MXFP8
SWA API and (out, lse, plan) return, reusing the plan after updating Q. Explicit
prims-ts accepts SM100/SM103/SM107 and requires CuTe-DSL >= 4.8 with its bundled
CUDA 13 compiler. It uses E4M3 Q and typed E4M3 SWA. Only that native path uses
the V4-style caller workspace, supplied output, runtime BMM factors and HND
caches shown below.

The native example, decode/cache regressions and fresh-process cache reuse passed
on B200 (SM100), GB300 (SM103) and R100 (SM107), with Frost tests skipped on the
two newer targets. See README.md for the tested toolchains and cold-compilation
limits. Performance is unmeasured.
"""

import argparse

import torch
from flashinfer import deepseek_v41 as ds


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cute-dsl", "prims-ts"),
        default="cute-dsl",
        help="explicit backend; prims-ts needs CuTe-DSL >= 4.8 with bundled CUDA 13",
    )
    return parser


def legacy_case():
    q = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    swa = ds.deepseek_v41_quantize_cache(x, format="swa_mxfp8")
    main = ds.deepseek_v41_quantize_cache(x, format="main_kv_fp4")
    ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    sink = torch.zeros(64, device="cuda")
    out, lse, plan = ds.deepseek_v41_decode(q, swa, main, ids, ids, sink)
    q.normal_()
    out, lse, _ = ds.deepseek_v41_decode(q, swa, main, ids, ids, sink, plan=plan)
    print("Frost (CuTe DSL primitives) decode:", out.shape, lse.shape)


def fp8_case():
    q_source = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    # The native caller owns Q quantization and common KV units. Normalize both
    # writers upstream and restore the common descale through one BMM pair.
    q = q_source.to(torch.float8_e4m3fn)
    kv_descale = 2.0
    cache_units = (x.float() / kv_descale).contiguous()
    swa = ds.deepseek_v41_quantize_cache(cache_units, format="swa_fp8", kv_layout="HND")
    main_cache = ds.deepseek_v41_quantize_cache(
        cache_units, format="main_kv_fp4", kv_layout="HND"
    )
    ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 128)
    indices = torch.cat((ids, ids), dim=1)
    topk_lens = torch.full((1,), 256, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((1,), 128, device="cuda", dtype=torch.int32)
    sink = torch.zeros(64, device="cuda")
    workspace = torch.empty(
        ds.deepseek_v41_decode_workspace_size(1, 128, backend="prims-ts"),
        device="cuda",
        dtype=torch.uint8,
    )
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)

    def mixed_decode():
        return ds.deepseek_v41_decode(
            q,
            swa,
            workspace,
            indices,
            main_cache,
            topk_lens,
            seq_lens,
            out,
            # The native V4 BMM1 default is 1.0; explicitly retain 1/sqrt(D).
            bmm1_scale=512**-0.5 * kv_descale,
            bmm2_scale=kv_descale,
            sinks=sink,
            kv_layout="HND",
            backend="prims-ts",
        )

    assert mixed_decode() is out
    q_source.normal_()
    q.copy_(q_source.to(q.dtype))
    assert mixed_decode() is out
    print("prims-ts mixed-cache decode:", out.shape)

    window_workspace = torch.empty(
        ds.deepseek_v41_decode_workspace_size(1, backend="prims-ts"),
        device="cuda",
        dtype=torch.uint8,
    )
    window_topk_lens = torch.full((1,), 128, device="cuda", dtype=torch.int32)
    window_out = torch.empty_like(out)
    result = ds.deepseek_v41_window_decode(
        q,
        swa,
        window_workspace,
        ids,
        window_topk_lens,
        seq_lens,
        out=window_out,
        bmm1_scale=512**-0.5 * kv_descale,
        bmm2_scale=kv_descale,
        sinks=sink,
        kv_layout="HND",
        backend="prims-ts",
    )
    assert result is window_out
    print("prims-ts window-only decode:", window_out.shape)


def main():
    opts = build_parser().parse_args()
    supported = (
        ((10, 0), (10, 3), (10, 7)) if opts.backend == "prims-ts" else ((10, 0),)
    )
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in supported
    ):
        required = "SM100, SM103 or SM107" if opts.backend == "prims-ts" else "SM100"
        raise RuntimeError(f"DS4.1 {opts.backend} decode requires {required}")
    if opts.backend == "prims-ts":
        fp8_case()
    else:
        legacy_case()


if __name__ == "__main__":
    main()
