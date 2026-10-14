# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVFP4/MXFP8 causal/noncausal attention timing and accuracy sweep.

Run in your activated environment from the repository root:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" python benchmarks/bench_expcast_configs.py

Run the explicit Q/KV cases below with both masks, both schedulers and three JIT variants:
blockscaled exp2, blockscaled expcast, and regular BF16 QK exp2.
All use FP8 E4M3 PV, BF16 output, LSE disabled; workload constants are below.
Accuracy reference: untimed FP16 QK/PV/output exp2 attention on dequantized
Q/K and decoded V, with FP32 accumulation. Report each path's error versus
FP16, its RMSE increase over matched blockscaled exp2 (percentage points),
and direct error versus blockscaled exp2. Regular Q/K are cast to BF16.

Compiler defaults: registers 192/96/32, threshold 2^-8, P offset +8,
full-tile P publication, dtype-derived KV stages, scale 1/sqrt(D), no PDL.
Timing: 5 warmups, 30 measured iterations, CUDA Graph, CUPTI/event fallback.
TFLOP/s counts QK and PV GEMMs only (4*B*Hq*D*attended_pairs), accounting
for bottom-right causal masking. Compilation/quantization/reference are excluded.
Edit constants below; results go to profiling/expcast-b300/expcast_configs.csv. See EXPCAST.md.
"""

import csv
from functools import partial
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from bench_blackwell_attention_cutedsl import dequantize_blockscaled

from flashinfer.attention.cute_dsl.fmha import cute_dsl_fmha_ragged_prefill
from flashinfer.attention.cute_dsl.fmha_blockscaled import (
    cute_dsl_fmha_blockscaled_prefill,
)
from flashinfer.cute_dsl.attention.fmha.compile import (
    compile_cute_dsl_fmha_blockscaled_kernel,
    compile_cute_dsl_fmha_kernel,
)
from flashinfer.cute_dsl.attention.fmha.fmha import (
    BlackwellFusedMultiHeadAttentionForward,
)
from flashinfer.cute_dsl.attention.fmha.fmha_blockscaled import (
    BlackwellFusedMultiHeadBlockScaledAttentionForward,
)
from flashinfer.cute_dsl.attention.fmha.quantize import quantize_blockscaled_qk
from flashinfer.testing.utils import bench_gpu_time

CASES = [
    (8192, 8192),
    (8192, 16384),
    (8192, 32768),
    (8192, 65536),
    (16384, 16384),
    (16384, 32768),
    (16384, 65536),
    (32768, 32768),
    (32768, 65536),
]

QK_MODES = ["nvfp4", "mxfp8"]
SCHEDULERS = [False, True]  # False = public-default nonpersistent scheduling.
CAUSAL_MODES = [False, True]
BATCH_SIZE = 1
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SEED = 0
WARMUP_ITERS = 5
REPEAT_ITERS = 30
OUTPUT = Path("profiling/expcast-b200/expcast_configs_gqa.csv")


def scheduling_override(original_init, persistent):
    def init(self, *args, **kwargs):
        kwargs["is_persistent"] = persistent
        original_init(self, *args, **kwargs)

    return init


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.executable}")
    print(
        f"GPU: {torch.cuda.get_device_name()}  CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}"
    )
    print(f"Torch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(
        f"NVFP4/MXFP8: exp2, expcast, regular exp2; Hq/Hkv={Q_HEADS}/{KV_HEADS}; FP8 PV. Accuracy ref: FP16 QK/PV."
    )
    rows = []
    dense_init = BlackwellFusedMultiHeadAttentionForward.__init__
    blockscaled_init = BlackwellFusedMultiHeadBlockScaledAttentionForward.__init__
    # An empty artifact directory forces both blockscaled exp2 baselines through JIT.
    with (
        TemporaryDirectory() as artifacts,
        patch.dict(os.environ, {"FLASHINFER_DSL_FMHA_LOCAL_DIR": artifacts}),
    ):
        for persistent in SCHEDULERS:
            with (
                patch.object(
                    BlackwellFusedMultiHeadAttentionForward,
                    "__init__",
                    scheduling_override(dense_init, persistent),
                ),
                patch.object(
                    BlackwellFusedMultiHeadBlockScaledAttentionForward,
                    "__init__",
                    scheduling_override(blockscaled_init, persistent),
                ),
            ):
                # Persistence is not part of the production compiler cache key.
                compile_cute_dsl_fmha_kernel.cache_clear()
                compile_cute_dsl_fmha_blockscaled_kernel.cache_clear()
                for q_len, kv_len in CASES:
                    for causal in CAUSAL_MODES:
                        torch.manual_seed(SEED)
                        q = torch.randn(
                            BATCH_SIZE,
                            q_len,
                            Q_HEADS,
                            HEAD_DIM,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        k = torch.randn(
                            BATCH_SIZE,
                            kv_len,
                            KV_HEADS,
                            HEAD_DIM,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        v = torch.randn_like(k).to(torch.float8_e4m3fn)
                        out = torch.empty_like(q)
                        qo_indptr = (
                            torch.arange(
                                BATCH_SIZE + 1, device="cuda", dtype=torch.int32
                            )
                            * q_len
                        )
                        kv_indptr = (
                            torch.arange(
                                BATCH_SIZE + 1, device="cuda", dtype=torch.int32
                            )
                            * kv_len
                        )
                        for qk_mode in QK_MODES:
                            qs, ks, qsf, ksf, qscale, kscale = (
                                quantize_blockscaled_qk(q, k, qk_mode)
                            )
                            qscale, kscale = float(qscale), float(kscale)
                            q_dequant = dequantize_blockscaled(
                                qs, qsf, qscale, qk_mode, HEAD_DIM
                            )
                            k_dequant = dequantize_blockscaled(
                                ks, ksf, kscale, qk_mode, HEAD_DIM
                            )
                            qr = q_dequant.to(torch.bfloat16)
                            kr = k_dequant.to(torch.bfloat16)
                            ref_out = torch.empty_like(q, dtype=torch.float16)
                            ref_kernel = compile_cute_dsl_fmha_kernel(
                                torch.float16,
                                torch.float16,
                                torch.float16,
                                Q_HEADS,
                                KV_HEADS,
                                HEAD_DIM,
                                HEAD_DIM,
                                causal,
                                False,
                                False,
                                False,
                                False,
                                q.device,
                            )
                            print(
                                f"Reference FP16 Q={q_len}, KV={kv_len}, QK={qk_mode}, persistent={persistent}",
                                flush=True,
                            )
                            cute_dsl_fmha_ragged_prefill(
                                q_dequant.to(torch.float16).flatten(0, 1),
                                k_dequant.to(torch.float16).flatten(0, 1),
                                v.to(torch.float16).flatten(0, 1),
                                ref_out.flatten(0, 1),
                                qo_indptr,
                                kv_indptr,
                                is_causal=causal,
                                max_qo_len=q_len,
                                max_kv_len=kv_len,
                                kernel_fn=ref_kernel,
                            )
                            for variant in ["exp2", "expcast", "regular"]:
                                exp_mode = (
                                    "expcast" if variant == "expcast" else "exp2"
                                )
                                if variant == "regular":
                                    kernel = compile_cute_dsl_fmha_kernel(
                                        torch.bfloat16,
                                        torch.float8_e4m3fn,
                                        torch.bfloat16,
                                        Q_HEADS,
                                        KV_HEADS,
                                        HEAD_DIM,
                                        HEAD_DIM,
                                        causal,
                                        False,
                                        False,
                                        False,
                                        False,
                                        q.device,
                                    )
                                    run = partial(
                                        cute_dsl_fmha_ragged_prefill,
                                        qr.flatten(0, 1),
                                        kr.flatten(0, 1),
                                        v.flatten(0, 1),
                                        out.flatten(0, 1),
                                        qo_indptr,
                                        kv_indptr,
                                        is_causal=causal,
                                        max_qo_len=q_len,
                                        max_kv_len=kv_len,
                                        kernel_fn=kernel,
                                    )
                                else:
                                    run = partial(
                                        cute_dsl_fmha_blockscaled_prefill,
                                        qs,
                                        ks,
                                        qsf,
                                        ksf,
                                        v,
                                        out,
                                        qk_mode=qk_mode,
                                        p_exp_mode=exp_mode,
                                        is_causal=causal,
                                        scale_q=qscale,
                                        scale_k=kscale,
                                    )
                                print(
                                    f"Starting Q={q_len}, KV={kv_len}, causal={causal}, QK={qk_mode}, variant={variant}, persistent={persistent}",
                                    flush=True,
                                )
                                run()  # Compile/warm before entering the timing helper.
                                torch.cuda.synchronize()
                                if variant == "exp2":
                                    exp2_out = out.clone()
                                diff = out.float() - ref_out.float()
                                rel_rmse = (
                                    diff.square().mean().sqrt()
                                    / ref_out.float()
                                    .square()
                                    .mean()
                                    .sqrt()
                                    .clamp_min(1e-12)
                                ).item()
                                out_max_abs = diff.abs().max().item()
                                if variant == "exp2":
                                    exp2_rel_rmse = rel_rmse
                                matched_diff = out.float() - exp2_out.float()
                                matched_rmse = (
                                    matched_diff.square().mean().sqrt()
                                    / exp2_out.float()
                                    .square()
                                    .mean()
                                    .sqrt()
                                    .clamp_min(1e-12)
                                ).item()
                                times = bench_gpu_time(
                                    run,
                                    dry_run_iters=WARMUP_ITERS,
                                    repeat_iters=REPEAT_ITERS,
                                    enable_cupti=True,
                                    use_cuda_graph=True,
                                )
                                median_ms = statistics.median(times)
                                pairs = q_len * kv_len
                                if causal:
                                    n = min(q_len, kv_len)
                                    pairs = n * (n + 1) // 2 + q_len * max(kv_len - q_len, 0)
                                flops = 4 * BATCH_SIZE * Q_HEADS * HEAD_DIM * pairs
                                row = {
                                    "q_len": q_len,
                                    "kv_len": kv_len,
                                    "batch_size": BATCH_SIZE,
                                    "q_heads": Q_HEADS,
                                    "kv_heads": KV_HEADS,
                                    "head_dim": HEAD_DIM,
                                    "causal": causal,
                                    "qk_mode": qk_mode,
                                    "kernel": "regular"
                                    if variant == "regular"
                                    else "blockscaled",
                                    "p_exp_mode": exp_mode,
                                    "persistent": persistent,
                                    "pv_dtype": "fp8_e4m3",
                                    "out_dtype": "bf16",
                                    "with_lse": False,
                                    "seed": SEED,
                                    "median_ms": median_ms,
                                    "tflops": flops / (median_ms * 1e9),
                                    "warmup_iters": WARMUP_ITERS,
                                    "repeat_iters": REPEAT_ITERS,
                                    "use_cuda_graph": True,
                                    "accuracy_ref": "fp16_qk_pv",
                                    "out_rel_rmse": rel_rmse,
                                    "out_max_abs": out_max_abs,
                                    "rmse_increase_vs_exp2_pp": 100
                                    * (rel_rmse - exp2_rel_rmse),
                                    "out_rel_rmse_vs_exp2": matched_rmse,
                                    "out_max_abs_vs_exp2": matched_diff.abs()
                                    .max()
                                    .item(),
                                }
                                rows.append(row)
                                print(row, flush=True)
                                with OUTPUT.open("w", newline="") as result_file:
                                    writer = csv.DictWriter(
                                        result_file, fieldnames=list(row)
                                    )
                                    writer.writeheader()
                                    writer.writerows(rows)
    print(f"Saved {len(rows)} configurations to {OUTPUT}")


if __name__ == "__main__":
    main()
