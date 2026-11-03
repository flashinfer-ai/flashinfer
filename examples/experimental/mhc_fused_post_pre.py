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

# Run the experimental fused mHC post -> pre (+ input RMSNorm) on an SM103 GPU and
# compare it with the composable PyTorch implementation.

import argparse

import torch

from flashinfer.experimental.mhc_fused_post_pre import (
    mhc_fused_post_pre_reference,
    stats,
)
from flashinfer.mhc import mhc_fused_post_pre, mhc_fused_post_pre_warmup


def make_inputs(num_tokens: int, hidden: int = 4096, hc: int = 4) -> dict:
    mix = hc * (2 + hc)
    comb = torch.randn(num_tokens, hc, hc, device="cuda").softmax(dim=-1)
    for _ in range(20):
        comb = comb / comb.sum(dim=-2, keepdim=True)
        comb = comb / comb.sum(dim=-1, keepdim=True)
    return {
        "x": (0.5 * torch.randn(num_tokens, hidden, device="cuda")).bfloat16(),
        "residual": (
            0.5 * torch.randn(num_tokens, hc, hidden, device="cuda")
        ).bfloat16(),
        "post_layer_mix": 2.0
        * torch.sigmoid(torch.randn(num_tokens, hc, device="cuda")),
        "comb_res_mix": comb.contiguous(),
        "fn": torch.randn(mix, hc * hidden, device="cuda") / (hc * hidden) ** 0.5,
        "hc_scale": torch.tensor([0.75, -1.25, 1.5], device="cuda"),
        "hc_base": 0.5 * torch.randn(mix, device="cuda"),
        "norm_weight": (1.0 + 0.1 * torch.randn(hidden, device="cuda")).bfloat16(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=64)
    args = parser.parse_args()

    ready = mhc_fused_post_pre_warmup()
    print(f"specialized kernel ready: {ready}")
    inputs = make_inputs(args.num_tokens)
    outputs = mhc_fused_post_pre(**inputs)
    reference = mhc_fused_post_pre_reference(**inputs)
    torch.cuda.synchronize()
    names = ("residual_cur", "post_mix_cur", "comb_mix_cur", "layer_input_cur")
    for name, got, ref in zip(names, outputs, reference, strict=True):
        err = (got.float() - ref.float().view(got.shape)).abs().max().item()
        print(f"{name}: shape={tuple(got.shape)} dtype={got.dtype} max|err|={err:.3g}")
    print(f"specialized dispatches: {stats()['specialized_dispatches']}")


if __name__ == "__main__":
    main()
