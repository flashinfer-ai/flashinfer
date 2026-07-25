"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Microbenchmark for mxfp8_attention_sm120_fwd (ragged per-tensor-FP8 prefill).

Example:
    python benchmarks/bench_mxfp8_attention_sm120.py --batch-size 8 --seq-len 4096
"""

import argparse

import torch


def _patch_cutlass_dsl_operand_major_mode() -> None:
    try:
        import cutlass.cute as cute
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
    except ImportError:
        return
    if not hasattr(cute.nvgpu, "OperandMajorMode"):
        cute.nvgpu.OperandMajorMode = OperandMajorMode


_patch_cutlass_dsl_operand_major_mode()

import flashinfer  # noqa: E402
from flashinfer.utils import is_sm120a_supported, is_sm121a_supported  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--seq-len", type=int, default=4096, help="per-request qo=kv len"
    )
    parser.add_argument("--num-qo-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def quantize_per_tensor(x: torch.Tensor):
    scale = (x.abs().amax().clamp(min=1e-6) / 448.0).item()
    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def main() -> None:
    args = parse_args()
    if not (
        is_sm120a_supported(torch.device("cuda"))
        or is_sm121a_supported(torch.device("cuda"))
    ):
        raise SystemExit("SM120/SM121 GPU is required")
    device = torch.device("cuda")
    B, S, Hq, Hkv, D = (
        args.batch_size,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
    )
    torch.manual_seed(0)

    q = torch.randn(B * S, Hq, D, device=device).to(torch.bfloat16)
    k = torch.randn(B * S, Hkv, D, device=device).to(torch.bfloat16)
    v = torch.randn(B * S, Hkv, D, device=device).to(torch.bfloat16)
    q8, qs = quantize_per_tensor(q)
    k8, ks = quantize_per_tensor(k)
    v8, vs = quantize_per_tensor(v)
    indptr = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=device)

    def run():
        flashinfer.mxfp8_attention_sm120_fwd(
            q8,
            k8,
            v8,
            indptr,
            indptr,
            sm_scale=D**-0.5,
            q_scale=qs,
            k_scale=ks,
            v_scale=vs,
            causal=args.causal,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeat):
        run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.repeat

    # Effective FLOPs: 2 GEMMs x 2 FLOP, causal halves the key coverage on average.
    flops = 4 * B * Hq * S * S * D * (0.5 if args.causal else 1.0)
    print(
        f"B={B} S={S} Hq={Hq} Hkv={Hkv} D={D} causal={args.causal}: "
        f"{ms:.3f} ms/iter, {flops / ms / 1e9:.1f} TFLOP/s"
    )


if __name__ == "__main__":
    main()
