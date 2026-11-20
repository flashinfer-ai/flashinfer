# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep DSv4.1 homogeneous FP8 decode split counts (128 + 512 candidates).

Run from the repository root:
    python benchmarks/bench_sparse_mla_sm120_fixed_splits.py

Compare the native decode resolver's auto choice, split-KV with the actual SM
count, and split-KV with fixed counts. Virtual resolver capacity only selects
the plan; the physical GPU is unchanged, and scratch is sized for that plan.
Check active_splits instead of treating the legacy chunks-per-block override
as a split count. Timing covers attention, including partial and merge
kernels, but excludes cache packing and planning. Eager runs flush L2;
graph runs use warm replay. Each line reports a per-arm timing record in JSON.
"""

import argparse
import json
import math

import numpy as np
import torch

from bench_sparse_mla_sm120 import _dsv41_pool
from flashinfer.mla._sparse_mla_sm120._execution import (
    get_sparse_mla_sm120_module,
    resolve_model_type,
)
from flashinfer.mla._sparse_mla_sm120._prepared import device_caps, validate_metadata
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm120a_supported


def gather_fp8(cache, indices, inline):
    """Independently decode just the selected packed E4M3/UE8M0 records."""
    page_size = cache.shape[1]
    ids = indices.clamp_min(0).long()
    page, slot = ids // page_size, ids % page_size
    if inline:
        rows = cache[page, slot]
        data = rows[..., :512].contiguous()
        exponents = rows[..., 512:]
    else:
        flat = cache.reshape(cache.shape[0], -1)
        data = flat[:, : page_size * 512].reshape(-1, page_size, 512)[page, slot]
        exponents = flat[:, page_size * 512 :].reshape(-1, page_size, 16)[page, slot]
    values = data.contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.pow(2.0, exponents.float() - 127)
    return (values.reshape(*indices.shape, 16, 32) * scales[..., None]).reshape(
        *indices.shape, 512
    )


def reference(q, main, extra, indices, extra_indices, lengths, extra_lengths, inline):
    kv = torch.cat(
        (gather_fp8(main, indices, inline), gather_fp8(extra, extra_indices, inline)),
        dim=1,
    ).double()
    valid = torch.cat(
        (
            (indices >= 0) & (torch.arange(128, device=q.device) < lengths[:, None]),
            (extra_indices >= 0)
            & (torch.arange(512, device=q.device) < extra_lengths[:, None]),
        ),
        dim=1,
    )
    scores = (q.double() @ kv.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.where(
        torch.isfinite(lse[..., None]), torch.exp(scores - lse[..., None]), 0
    )
    # Decode quantizes BF16 queries to FP8 with per-group power-of-two scales.
    # Independently check the arithmetic's LSE as well as the BF16 reference.
    groups = q.float().reshape(*q.shape[:2], 16, 32)
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    scale = ((scale.view(torch.int32) + 0x007FFFFF) & 0x7F800000).view(torch.float32)
    quantized_q = (
        ((groups / scale).to(torch.float8_e4m3fn).float() * scale)
        .reshape_as(q)
        .double()
    )
    quantized_scores = (quantized_q @ kv.transpose(1, 2)) * 512**-0.5
    quantized_scores.masked_fill_(~valid[:, None, :], -torch.inf)
    return weights @ kv, lse, torch.logsumexp(quantized_scores, dim=-1)


def check_output(out, lse, expected, expected_lse, expected_fp8_lse):
    finite = torch.isfinite(expected_lse)
    assert torch.equal(torch.isneginf(lse), ~finite)
    assert torch.isfinite(out).all() and not torch.isnan(lse).any()
    assert not torch.count_nonzero(out[~finite])
    rel_l2 = (
        (out.double() - expected).norm() / expected.norm().clamp_min(1e-30)
    ).item()
    lse_abs = (
        (lse[finite].double() - expected_lse[finite]).abs().max().item()
        if finite.any()
        else 0.0
    )
    # Match the DSv4.1 decode tests' BF16-reference and FP8-Q LSE gates.
    assert rel_l2 < 0.07 and lse_abs < 0.06, (rel_l2, lse_abs)
    torch.testing.assert_close(lse.double(), expected_fp8_lse, atol=2e-5, rtol=0)
    return rel_l2, lse_abs


def sweep_case(args, tokens, heads, inline):
    device = torch.device("cuda")
    main = _dsv41_pool(args.pool_pages, False, inline, device, args.seed)
    extra = _dsv41_pool(args.pool_pages, False, inline, device, args.seed + 1)
    rng = torch.Generator(device=device).manual_seed(args.seed + 2)
    q = (
        torch.randn(
            tokens, heads, 512, device=device, dtype=torch.bfloat16, generator=rng
        )
        / 10
    ).clamp(-1, 1)
    indices = torch.randint(
        args.pool_pages * 64,
        (tokens, 128),
        device=device,
        dtype=torch.int32,
        generator=rng,
    )
    extra_indices = torch.randint(
        args.pool_pages * 64,
        (tokens, 512),
        device=device,
        dtype=torch.int32,
        generator=rng,
    )
    lengths = torch.full((tokens,), 128, device=device, dtype=torch.int32)
    extra_lengths = torch.full((tokens,), 512, device=device, dtype=torch.int32)
    output = torch.empty_like(q)
    lse = torch.empty((tokens, heads), device=device, dtype=torch.float32)
    scale_format = "ue8m0_g32_inline" if inline else "ue8m0_g32"
    metadata = validate_metadata(
        q,
        main,
        indices,
        output,
        lengths,
        None,
        extra,
        extra_indices,
        extra_lengths,
        lse,
        None,
        None,
        model=resolve_model_type(512, scale_format),
        is_dsv4_nvfp4=False,
        extra_fp4=False,
        value_dim=512,
    )
    module = get_sparse_mla_sm120_module()
    sms, max_shared = device_caps(device)
    expected = reference(
        q, main, extra, indices, extra_indices, lengths, extra_lengths, inline
    )
    # H=16/32/64 has one resident CTA/SM in the split-KV selection policy.
    # Setting the resolver capacity to tasks*target_split produces exactly
    # that split count; the plan inspection below fails if the policy changes.
    tasks = tokens * ((heads + 15) // 16)
    modes = [("auto", 0, sms, None), ("split-auto", 1, sms, None)]
    modes.extend((f"split-{n}", 1, tasks * n, n) for n in (1, 2, 4, 8))
    arms = []
    verify_graph = (
        args.timing in ("graph", "both") and tokens == 4 and heads == min(args.heads)
    )
    for label, mode, capacity, target in modes:
        plan = module.resolve_decode_attention(metadata, mode, capacity, max_shared)
        facts = dict(plan.inspect())
        actual_splits = int(facts["active_splits"])
        if target is not None:
            assert facts["implementation"] == "dsv4_fp8_splitkv_decode"
            assert actual_splits == target, (target, facts)
        workspace = [
            torch.empty(tuple(shape), dtype=getattr(torch, str(dtype)), device=device)
            for shape, dtype, _, _ in list(plan.workspace())[:2]
        ]

        def run(plan=plan, workspace=workspace):
            module.execute_attention(
                plan,
                q,
                main,
                indices,
                workspace[0],
                workspace[1],
                output,
                lse,
                512**-0.5,
                lengths,
                None,
                extra,
                extra_indices,
                extra_lengths,
                math.log(2),
            )

        run()
        torch.cuda.synchronize()
        error, lse_error = check_output(output, lse, *expected)
        graph_checked = False
        if verify_graph:
            tensors = (q, indices, extra_indices, lengths, extra_lengths)
            originals = [t.clone() for t in tensors]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            q.mul_(0.75)
            lengths[0] = extra_lengths[0] = 0
            lengths[1] = 17
            extra_lengths[1] = 129
            indices[:, ::7] = -1
            extra_indices[:, ::7] = -1
            graph.replay()
            torch.cuda.synchronize()
            check_output(
                output,
                lse,
                *reference(
                    q,
                    main,
                    extra,
                    indices,
                    extra_indices,
                    lengths,
                    extra_lengths,
                    inline,
                ),
            )
            for tensor, original in zip(tensors, originals, strict=True):
                tensor.copy_(original)
            del graph
            run()
            torch.cuda.synchronize()
            graph_checked = True
        arms.append(
            (
                label,
                capacity,
                actual_splits,
                str(facts["implementation"]),
                run,
                error,
                lse_error,
                graph_checked,
            )
        )

    for timing in ("eager", "graph") if args.timing == "both" else (args.timing,):
        for trial in range(args.passes):
            for label, capacity, splits, impl, run, error, lse_error, graph_checked in (
                arms if trial % 2 == 0 else reversed(arms)
            ):
                times = (
                    np.asarray(
                        bench_gpu_time(
                            run,
                            dry_run_iters=10,
                            repeat_iters=args.iters,
                            enable_cupti=True,
                            use_cuda_graph=timing == "graph",
                            cold_l2_cache=timing == "eager",
                        )
                    )
                    * 1000
                )
                print(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "heads": heads,
                            "topk": 128,
                            "extra_topk": 512,
                            "layout": "inline" if inline else "footer",
                            "timing": timing,
                            "trial": trial,
                            "mode": label,
                            "actual_splits": splits,
                            "implementation": impl,
                            "real_sms": sms,
                            "resolver_capacity_sms": capacity,
                            "median_us": float(np.median(times)),
                            "p90_us": float(np.percentile(times, 90)),
                            "sample_count": len(times),
                            "relative_l2": error,
                            "lse_max_abs": lse_error,
                            "mutated_graph_checked": graph_checked,
                        }
                    ),
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    parser.add_argument(
        "--heads", type=int, nargs="+", choices=(16, 32, 64), default=[32, 64]
    )
    parser.add_argument(
        "--layout", choices=("footer", "inline", "both"), default="both"
    )
    parser.add_argument("--timing", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--pool-pages", type=int, default=4096)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not is_sm120a_supported(torch.device("cuda")):
        raise SystemExit("Sparse-MLA SM120 requires sm120a.")
    for inline in (
        (False, True) if args.layout == "both" else (args.layout == "inline",)
    ):
        for heads in args.heads:
            for tokens in args.tokens:
                sweep_case(args, tokens, heads, inline)


if __name__ == "__main__":
    main()
