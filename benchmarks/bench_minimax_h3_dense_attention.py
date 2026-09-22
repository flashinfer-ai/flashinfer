# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark the MiniMax-H3 dense BF16 self-attention cut.

MiniMax-H3 (video DiT) runs one dense, non-causal, 56-head, head_dim-128 BF16
self-attention per block over a packed token stream of ``S`` tokens (batch 1),
with the query pre-scaled in BF16 before the softmax::

    heads(x) = x.view(S, 56, 128)
    Qs = (heads(q) * bf16(1 / sqrt(128))).to(bf16)     # 0.08837890625, rounded before SDPA
    y  = softmax(Qs @ heads(k)^T) @ heads(v)           # scale 1.0, no mask, no dropout
    y  = y.view(S, 7168)

Inputs and output are contiguous BF16 ``[S, 7168]`` rows; ``S`` is dynamic
(1 .. 131072).  Representative ``S`` values come from the model's request
types (T2VA / FL2VA / REF2VA at 124 and 345 frames, normal and SR-base
resolutions).  This script times the fused single-kernel route
``flashinfer.diffusion_ops.minimax_h3_dense_attention`` (``cake``: row<->head
layout + BF16 Q-scale + attention in one launch), FlashInfer
``single_prefill_with_kv_cache`` (``auto`` selects the ``fmha_v2`` route on
SM120, ``fa2`` is the generic FA2 path) and PyTorch SDPA at the same boundary,
i.e. including the BF16 Q-scale launch, and checks every backend against an
FP32 oracle.

Usage::

    python benchmarks/bench_minimax_h3_dense_attention.py --suite short
    python benchmarks/bench_minimax_h3_dense_attention.py --tokens 37804 --backends flashinfer_auto torch_efficient
    python benchmarks/bench_minimax_h3_dense_attention.py --suite all --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from flashinfer.prefill import single_prefill_with_kv_cache
from flashinfer.testing.utils import bench_gpu_time

NUM_HEADS = 56
HEAD_DIM = 128
WIDTH = NUM_HEADS * HEAD_DIM
ATOL, RTOL = 1e-2, 2e-2
Q_SCALE_BF16 = 0.08837890625

# label -> S (packed tokens). Same table as the MiniMax-H3 TensorRT-RTX request package.
SHAPES = {
    "sr_t2va_124f": 15493,
    "sr_fl2va_124f": 17147,
    "sr_ref2va_124f": 30224,
    "normal_t2va_124f": 37804,
    "normal_fl2va_124f": 41870,
    "sr_t2va_345f": 42554,
    "sr_fl2va_345f": 44208,
    "normal_ref2va_124f": 52535,
    "sr_ref2va_345f": 57287,
    "normal_t2va_345f": 104060,
    "normal_fl2va_345f": 108126,
    "normal_ref2va_345f": 118793,
}
SUITES = {
    "smoke": ["sr_t2va_124f"],
    "short": ["sr_t2va_124f", "normal_t2va_124f", "normal_ref2va_124f"],
    "all": list(SHAPES),
}
TORCH_BACKENDS = {
    "torch_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "torch_cudnn": SDPBackend.CUDNN_ATTENTION,
    "torch_flash": SDPBackend.FLASH_ATTENTION,
}
FLASHINFER_BACKENDS = {"flashinfer_auto": "auto", "flashinfer_fa2": "fa2"}
# Fused single-kernel route (row<->head layout + BF16 query scale + attention), SM120 target.
CAKE_BACKEND = "cake"
ALL_BACKENDS = [CAKE_BACKEND] + list(FLASHINFER_BACKENDS) + list(TORCH_BACKENDS)


def synthetic_inputs(tokens: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = []
    for _ in range(3):
        rows = torch.randn(tokens, WIDTH, generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        )
        tensors.append(rows.to(device))
    return tensors


def scaled_query(q: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(
        1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=q.device
    ).to(torch.bfloat16)
    assert float(scale) == Q_SCALE_BF16
    return q * scale  # BF16 product rounded before attention


def flops(tokens: int) -> float:
    return 4.0 * tokens * tokens * NUM_HEADS * HEAD_DIM


def run_flashinfer(q, k, v, backend: str) -> torch.Tensor:
    tokens = q.shape[0]
    qs = scaled_query(q).view(tokens, NUM_HEADS, HEAD_DIM)
    out = single_prefill_with_kv_cache(
        qs,
        k.view(tokens, NUM_HEADS, HEAD_DIM),
        v.view(tokens, NUM_HEADS, HEAD_DIM),
        causal=False,
        kv_layout="NHD",
        sm_scale=1.0,
        backend=backend,
    )
    return out.view(tokens, WIDTH)


def run_torch(q, k, v, backend: SDPBackend | None) -> torch.Tensor:
    tokens = q.shape[0]
    heads = lambda x: x.view(tokens, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)  # noqa: E731
    context = sdpa_kernel(backend) if backend is not None else nullcontext()
    with context:
        out = scaled_dot_product_attention(
            scaled_query(heads(q)),
            heads(k),
            heads(v),
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
    return out.squeeze(0).permute(1, 0, 2).reshape(tokens, WIDTH).contiguous()


@torch.inference_mode()
def fp32_oracle(q, k, v, query_chunk: int = 1024) -> torch.Tensor:
    """FP32 scores/softmax/PV over the BF16 operands, TF32 off, one final BF16 rounding."""

    tokens = q.shape[0]
    out = torch.empty_like(q)
    qs = scaled_query(q)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for head in range(NUM_HEADS):
            lo, hi = head * HEAD_DIM, (head + 1) * HEAD_DIM
            k_head = k[:, lo:hi].float()
            v_head = v[:, lo:hi].float()
            for start in range(0, tokens, query_chunk):
                stop = min(start + query_chunk, tokens)
                scores = qs[start:stop, lo:hi].float() @ k_head.T
                scores -= scores.amax(dim=-1, keepdim=True)
                probs = torch.exp(scores)
                out[start:stop, lo:hi] = (
                    (probs @ v_head) / probs.sum(dim=-1, keepdim=True)
                ).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return out


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual32, expected32 = actual.float(), expected.float()
    finite = bool(torch.isfinite(actual32).all())
    delta = (actual32 - expected32).abs()
    return {
        "allclose": finite
        and bool(torch.allclose(actual32, expected32, atol=ATOL, rtol=RTOL)),
        "finite": finite,
        "max_abs_err": float(delta.max()) if finite else None,
        "nrmse": float(
            delta.square().mean().sqrt()
            / expected32.square().mean().sqrt().clamp_min(1e-12)
        )
        if finite
        else None,
    }


def run_cake(q, k, v) -> torch.Tensor:
    from flashinfer.diffusion_ops import minimax_h3_dense_attention

    return minimax_h3_dense_attention(q, k, v)


def make_runner(name: str, q, k, v):
    if name == CAKE_BACKEND:
        return lambda: run_cake(q, k, v)
    if name in FLASHINFER_BACKENDS:
        return lambda: run_flashinfer(q, k, v, FLASHINFER_BACKENDS[name])
    if name in TORCH_BACKENDS:
        return lambda: run_torch(q, k, v, TORCH_BACKENDS[name])
    raise ValueError(f"unknown backend {name!r}")


def bench_shape(
    label: str,
    tokens: int,
    backends: list[str],
    *,
    seed: int,
    warmup: int,
    iters: int,
    check: bool,
    device,
):
    q, k, v = synthetic_inputs(tokens, seed, device)
    expected = fp32_oracle(q, k, v) if check else None
    rows = []
    for name in backends:
        runner = make_runner(name, q, k, v)
        row = {"shape": label, "tokens": tokens, "backend": name}
        try:
            out = runner()
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001 - report unavailable backends instead of hiding them
            row["error"] = repr(error)
            rows.append(row)
            print(f"[{label} S={tokens}] {name}: unavailable ({error})")
            continue
        if expected is not None:
            row.update(compare(out, expected))
        del out
        times = bench_gpu_time(
            runner,
            enable_cupti=True,
            dry_run_iters=warmup,
            repeat_iters=iters,
            cold_l2_cache=True,
        )
        median_ms = float(np.median(times))
        row.update(
            median_ms=median_ms,
            min_ms=float(np.min(times)),
            max_ms=float(np.max(times)),
            samples_ms=[float(t) for t in times],
            tflops=flops(tokens) / (median_ms * 1e-3) / 1e12,
        )
        status = "" if not check else (" ok" if row["allclose"] else " MISMATCH")
        print(
            f"[{label} S={tokens}] {name}: {median_ms:.3f} ms  {row['tflops']:.1f} TFLOP/s{status}"
        )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=sorted(SUITES), default="short")
    parser.add_argument(
        "--tokens", type=int, nargs="*", help="explicit S values instead of a suite"
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=["flashinfer_auto", "torch_efficient"],
        choices=ALL_BACKENDS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--no-check", action="store_true", help="skip the FP32 oracle comparison"
    )
    parser.add_argument("--json", type=str, help="write all rows to this JSON file")
    args = parser.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    header = {
        "device": props.name,
        "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "timing": "CUPTI kernel activity, cold L2; includes the BF16 Q-scale launch and any layout copies",
    }
    print(json.dumps(header))
    if args.tokens:
        shapes = [(f"s{tokens}", tokens) for tokens in args.tokens]
    else:
        shapes = [(label, SHAPES[label]) for label in SUITES[args.suite]]

    rows = []
    for label, tokens in shapes:
        rows.extend(
            bench_shape(
                label,
                tokens,
                args.backends,
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
                check=not args.no_check,
                device=device,
            )
        )
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump(
                {"header": header, "generated_at_unix": time.time(), "rows": rows},
                stream,
                indent=2,
            )
        print(f"wrote {args.json}")
    return 0 if all(row.get("allclose", True) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
