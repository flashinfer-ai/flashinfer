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

# Run the experimental mHC pre big-fuse + RMSNorm op on an SM100 / SM103 GPU,
# eagerly and inside a CUDA graph, and compare it with its composable
# PyTorch implementation.

import argparse
import os

import torch

from flashinfer.mhc import (
    mhc_pre_big_fuse_with_norm,
    mhc_pre_big_fuse_with_norm_supported,
    mhc_pre_big_fuse_with_norm_warmup,
)

HIDDEN, HC, MIX = 4096, 4, 24
CONSTANTS = dict(
    rms_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    norm_eps=1e-5,
)


def make_inputs(tokens: int, num_splits: int) -> dict:
    # Split-K partials of the mix projection of x = residual.flatten(1).
    x = torch.randn(tokens, HC, HIDDEN, device="cuda")
    fn = torch.randn(MIX, HC, HIDDEN, device="cuda") * 0.03
    xs = x.view(tokens, HC, num_splits, HIDDEN // num_splits)
    fs = fn.view(MIX, HC, num_splits, HIDDEN // num_splits)
    return dict(
        gemm_out_mul=torch.einsum("tjsk,njsk->stn", xs, fs).contiguous(),
        gemm_out_sqrsum=xs.square().sum(dim=(1, 3)).t().contiguous(),
        hc_scale=torch.tensor([0.09, 0.16, 0.21], device="cuda"),
        hc_base=torch.randn(MIX, device="cuda"),
        residual=x.to(torch.bfloat16),
        norm_weight=torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--num-splits", type=int, default=8, choices=[8, 64])
    args = parser.parse_args()

    inputs = make_inputs(args.tokens, args.num_splits)
    print("cluster kernel usable:", mhc_pre_big_fuse_with_norm_warmup())
    print(
        "cluster kernel serves this call:",
        mhc_pre_big_fuse_with_norm_supported(**inputs, **CONSTANTS),
    )

    # Eager call.
    post_mix, comb_mix, layer_input = mhc_pre_big_fuse_with_norm(**inputs, **CONSTANTS)

    # CUDA-graph capture with caller-owned outputs (warmup above loaded the kernel).
    outs = (torch.empty_like(post_mix), torch.empty_like(comb_mix))
    outs += (torch.empty_like(layer_input),)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mhc_pre_big_fuse_with_norm(
            **inputs,
            **CONSTANTS,
            post_mix=outs[0],
            comb_mix=outs[1],
            layer_input=outs[2],
        )
    graph.replay()

    # Composable PyTorch implementation of the same contract.
    os.environ["FLASHINFER_SPECIALIZED_KERNEL_DISABLE"] = "1"
    ref = mhc_pre_big_fuse_with_norm(**inputs, **CONSTANTS)
    del os.environ["FLASHINFER_SPECIALIZED_KERNEL_DISABLE"]
    torch.cuda.synchronize()

    for name, got, want in zip(
        ("post_mix", "comb_mix", "layer_input"), outs, ref, strict=True
    ):
        err = (got.float() - want.float()).abs().max().item()
        print(f"{name}: shape={tuple(got.shape)} dtype={got.dtype} max|diff|={err:.3e}")


if __name__ == "__main__":
    main()
