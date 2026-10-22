"""Reproduce SM89 FP8 MLA decode against a fused dequantization + BF16 fallback.

Run from a FlashInfer checkout; see docs/design_docs/mla_sm89_fp8_design.md.
Copyright (c) 2026 by FlashInfer team. Licensed under the Apache License, Version 2.0.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import traceback

import torch
import flashinfer
from flashinfer.jit.core import gen_jit_spec
from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from flashinfer.testing import bench_gpu_time

ASSETS = Path(__file__).resolve().parent / "mla_sm89_fp8"
SOURCE = Path(flashinfer.__file__).resolve().parents[1]
torch.backends.cuda.matmul.allow_tf32 = False


def case_inputs(lengths, heads, page_size, scales):
    device = torch.device("cuda:0")
    batch = len(lengths)
    page_counts = [(n + page_size - 1) // page_size for n in lengths]
    num_pages = sum(page_counts)
    indices = torch.randperm(num_pages, device=device).to(torch.int32)
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    kv_indptr = (
        torch.tensor([0] + page_counts, dtype=torch.int32, device=device)
        .cumsum(0)
        .int()
    )
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    q = torch.randn(batch, heads, 512, device=device, dtype=torch.bfloat16)
    qpe = torch.randn(batch, heads, 64, device=device, dtype=torch.bfloat16)
    ckv8 = (torch.randn(num_pages, page_size, 512, device=device) / scales[0]).to(
        torch.float8_e4m3fn
    )
    kpe8 = (torch.randn(num_pages, page_size, 64, device=device) / scales[1]).to(
        torch.float8_e4m3fn
    )
    ckv16 = torch.empty_like(ckv8, dtype=torch.bfloat16)
    kpe16 = torch.empty_like(kpe8, dtype=torch.bfloat16)
    # Match upstream per-tensor repack: scalar is rounded to BF16 before hmul2.
    scale_c = float(torch.tensor(scales[0], dtype=torch.bfloat16))
    scale_k = float(torch.tensor(scales[1], dtype=torch.bfloat16))

    def convert():
        ckv16.copy_(ckv8)
        ckv16.mul_(scale_c)
        kpe16.copy_(kpe8)
        kpe16.mul_(scale_k)

    kwargs = dict(
        metadata=MLAPlanMetadata.csr(qo_indptr, kv_indptr, indices, kv_lens),
        num_heads=heads,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=page_size,
        causal=False,
        sm_scale=1 / math.sqrt(192),
        q_data_type=torch.bfloat16,
        output_dtype=torch.bfloat16,
        query_layout="split",
        kv_cache_layout="split",
        lse_mode="base2",
    )
    return q, qpe, ckv8, kpe8, ckv16, kpe16, convert, kwargs


def compare_reference(q, qpe, ckv, kpe, metadata, lengths, out, lse, sm_scale):
    errors = []
    for b, length in enumerate(lengths):
        start, end = metadata.kv_indptr[b : b + 2].cpu().tolist()
        pages = metadata.kv_indices[start:end].long()
        key_c = ckv[pages].reshape(-1, 512)[:length].float()
        key_p = kpe[pages].reshape(-1, 64)[:length].float()
        scores = (q[b].float() @ key_c.T + qpe[b].float() @ key_p.T) * sm_scale
        ref = torch.softmax(scores, dim=-1) @ key_c
        ref_lse = torch.logsumexp(scores, dim=-1) / math.log(2)
        torch.testing.assert_close(out[b].float(), ref, rtol=0.03, atol=0.004)
        torch.testing.assert_close(lse[b], ref_lse, rtol=0.001, atol=0.003)
        errors.append(
            {
                "output_max_abs": (out[b].float() - ref).abs().max().item(),
                "output_relative_l2": (
                    (out[b].float() - ref).norm() / ref.norm()
                ).item(),
                "lse_max_abs": (lse[b] - ref_lse).abs().max().item(),
            }
        )
    return errors


def write(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def fused_module():
    src = ASSETS / "fused_dequant.cu"
    key = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    return gen_jit_spec(
        "mla89_fused_pool_dequant_" + key,
        [src],
        extra_include_paths=[SOURCE / "csrc"],
        use_fast_math=False,
    ).build_and_load()


def measure(fn, graph, iters):
    return bench_gpu_time(
        fn,
        dry_run_iters=5,
        repeat_iters=iters,
        enable_cupti=False,
        use_cuda_graph=graph,
        num_iters_within_graph=5,
        cold_l2_cache=not graph,
    )


def stats(times):
    return dict(
        median_us=statistics.median(times) * 1000,
        min_us=min(times) * 1000,
        max_us=max(times) * 1000,
        samples_us=[x * 1000 for x in times],
    )


def reference_with_empty(q, qp, c, p, metadata, lengths, out, lse, sm_scale):
    errors = []
    for i, n in enumerate(lengths):
        start, end = metadata.kv_indptr[i : i + 2].cpu().tolist()
        if n == 0:
            torch.testing.assert_close(out[i], torch.zeros_like(out[i]), rtol=0, atol=0)
            assert bool(torch.isneginf(lse[i]).all())
            errors.append(dict(output_max_abs=0, output_relative_l2=0, lse_max_abs=0))
            continue
        md = MLAPlanMetadata.csr(
            torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            torch.tensor([0, end - start], device="cuda", dtype=torch.int32),
            metadata.kv_indices[start:end],
            torch.tensor([n], device="cuda", dtype=torch.int32),
        )
        err = compare_reference(
            q[i : i + 1],
            qp[i : i + 1],
            c,
            p,
            md,
            [n],
            out[i : i + 1],
            lse[i : i + 1],
            sm_scale,
        )[0]
        assert err["output_relative_l2"] < 0.006, err
        errors.append(err)
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--case-file", type=Path, default=ASSETS / "cases.json")
    ap.add_argument(
        "--group", choices=["development", "additional", "all"], default="development"
    )
    ap.add_argument("--cases", default="")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument(
        "--check-unsupported",
        action="store_true",
        help="With --baseline-only, assert upstream rejects the FP8 SM89 contract",
    )
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="NVTX native/baseline scopes once; no reference or timing",
    )
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--output", required=True)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--lse-mode", choices=["base2", "basee"], default="base2")
    args = ap.parse_args()
    path = Path(args.output)
    if args.check_unsupported and not args.baseline_only:
        ap.error("--check-unsupported requires --baseline-only")
    if torch.cuda.get_device_capability() != (8, 9):
        ap.error("this benchmark targets SM89")
    all_cases = json.loads(args.case_file.read_text())
    cases = (
        all_cases["development"] + all_cases["additional"]
        if args.group == "all"
        else all_cases[args.group]
    )
    if args.cases:
        selected = set(args.cases.split(","))
        cases = [x for x in cases if x["id"] in selected]
    assert cases, "empty selection"
    diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", ".", ":(exclude)3rdparty"], cwd=SOURCE, text=True
    )
    result = dict(
        label=args.label,
        source=str(SOURCE),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
        ).strip(),
        source_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        gpu=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        status="running",
        cases=[],
        method="interleaved cold-L2 CUDA events and repeated-address warm-L2 CUDA graphs",
        baseline="single fused full-pool FP8->BF16 conversion + FA2 BF16 MLA; preallocated outputs",
        caveats=[
            "plan/JIT/allocation excluded",
            "not end-to-end serving latency",
            "only allocated batch pages in pool",
            "timing order alternates",
        ],
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    result["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["case_file_sha256"] = hashlib.sha256(args.case_file.read_bytes()).hexdigest()
    result["compute_capability"] = list(torch.cuda.get_device_capability())
    result["driver"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()[0]
    write(result, path)
    module = fused_module()
    try:
        for ci, case in enumerate(cases):
            torch.manual_seed(
                20261022 + int(hashlib.sha256(case["id"].encode()).hexdigest()[:6], 16)
            )
            lengths = case.get(
                "lengths", [case.get("length", 1)] * case.get("batch", 1)
            )
            q, qp, c8, p8, c16, p16, torch_convert, kw = case_inputs(
                lengths, case["heads"], case["page"], case["scales"]
            )
            kw["causal"] = args.causal
            kw["lse_mode"] = args.lse_mode
            row = dict(
                case=case,
                lengths=lengths,
                correctness="pending",
                fallback_bf16_temporary_bytes=2 * (c8.numel() + p8.numel()),
                native_full_pool_temporary_bytes=0,
            )
            result["cases"].append(row)

            def convert():
                module.dequant(c8, p8, c16, p16, *case["scales"])

            torch_convert()
            ref_c, ref_p = c16.clone(), p16.clone()
            convert()
            torch.cuda.synchronize()
            torch.testing.assert_close(c16, ref_c, rtol=0, atol=0)
            torch.testing.assert_close(p16, ref_p, rtol=0, atol=0)
            del ref_c, ref_p
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            if args.check_unsupported:
                probe = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
                try:
                    probe.plan(
                        **kw,
                        kv_data_type=torch.float8_e4m3fn,
                        scale_mode="kv-per-tensor",
                    )
                except (ValueError, RuntimeError) as exc:
                    if "requires an SM90" not in str(exc):
                        raise
                    row["upstream_rejection"] = str(exc)
                else:
                    raise AssertionError(
                        "upstream accepted this contract; re-evaluate the support gap"
                    )
            wb = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
            wb.plan(**kw, kv_data_type=torch.bfloat16)
            ob = torch.empty_like(q)
            lb = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")

            def attention():
                return wb.run(
                    query=(q, qp),
                    kv_cache=(c16, p16),
                    out=ob,
                    lse=lb,
                    return_lse=True,
                    return_lse_base_on_e=args.lse_mode == "basee",
                )

            def baseline():
                convert()
                return attention()

            functions = {
                "conversion": convert,
                "bf16_attention": attention,
                "fused_baseline": baseline,
            }
            if not args.baseline_only:
                workspace_n = torch.empty_like(workspace)
                wn = BatchMLAPagedAttentionWrapper(workspace_n, backend="fa2")
                wn.plan(
                    **kw, kv_data_type=torch.float8_e4m3fn, scale_mode="kv-per-tensor"
                )
                on = torch.empty_like(q)
                ln = torch.empty_like(lb)

                def native():
                    return wn.run(
                        query=(q, qp),
                        kv_cache=(c8, p8),
                        out=on,
                        lse=ln,
                        return_lse=True,
                        ckv_scale=case["scales"][0],
                        kpe_scale=case["scales"][1],
                        return_lse_base_on_e=args.lse_mode == "basee",
                    )

                functions["native"] = native
            for fn in functions.values():
                fn()
            torch.cuda.synchronize()
            if args.profile:
                for name in ("fused_baseline", "native"):
                    if name not in functions:
                        continue
                    torch.cuda.nvtx.range_push(name)
                    functions[name]()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                row["correctness"] = "not_checked_profile_mode"
                write(result, path)
                continue
            ref_lse = lb if args.lse_mode == "base2" else lb / math.log(2)
            row["baseline_errors"] = reference_with_empty(
                q, qp, c16, p16, kw["metadata"], lengths, ob, ref_lse, kw["sm_scale"]
            )
            if not args.baseline_only:
                ref_ln = ln if args.lse_mode == "base2" else ln / math.log(2)
                row["native_errors"] = reference_with_empty(
                    q, qp, c16, p16, kw["metadata"], lengths, on, ref_ln, kw["sm_scale"]
                )
                row["native_vs_baseline_max_abs"] = (
                    (on.float() - ob.float()).abs().max().item()
                )
                torch.testing.assert_close(on, ob, rtol=0.03, atol=0.004)
                torch.testing.assert_close(ln, lb, rtol=0.001, atol=0.003)
            row["correctness"] = "passed"
            # Replay validation is a functional gate even when timing is disabled.
            graph_fn = functions.get("native", baseline)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                graph_fn()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            if not args.baseline_only:
                check_ln = ln if args.lse_mode == "base2" else ln / math.log(2)
                reference_with_empty(
                    q,
                    qp,
                    c16,
                    p16,
                    kw["metadata"],
                    lengths,
                    on,
                    check_ln,
                    kw["sm_scale"],
                )
            row["graph_replay"] = "passed"
            if not args.no_timing:
                for graph in (False, True):
                    mode = "warm_graph" if graph else "cold_events"
                    samples = {name: [] for name in functions}
                    round_medians = {name: [] for name in functions}
                    for rep in range(args.rounds):
                        order = list(functions)
                        if (rep + ci) % 2:
                            order.reverse()
                        for name in order:
                            values = measure(functions[name], graph, args.iters)
                            samples[name].extend(values)
                            round_medians[name].append(statistics.median(values) * 1000)
                    row[mode] = {
                        name: dict(**stats(values), round_median_us=round_medians[name])
                        for name, values in samples.items()
                    }
                    if not args.baseline_only:
                        row[mode]["speedup_vs_fused_baseline"] = statistics.median(
                            samples["fused_baseline"]
                        ) / statistics.median(samples["native"])
                        row[mode]["speedup_vs_bf16_attention_only"] = statistics.median(
                            samples["bf16_attention"]
                        ) / statistics.median(samples["native"])
            write(result, path)
            print(
                json.dumps(
                    {
                        "label": args.label,
                        "case": case["id"],
                        "correctness": row["correctness"],
                        "speedup": row.get("warm_graph", {}).get(
                            "speedup_vs_fused_baseline"
                        ),
                    }
                ),
                flush=True,
            )
            del functions, g, workspace
            if not args.baseline_only:
                del wn, workspace_n
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        write(result, path)


if __name__ == "__main__":
    main()
