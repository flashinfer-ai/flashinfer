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

# The fused mHC inference step of a four-route model (hidden 4096) with a
# quantized output through the experimental ``prepare_mega_mhc`` API on an
# SM100a, SM103a or SM107a GPU: beside the BF16 output the plan stores its quantization, what
# vLLM computes from that output -- ``--layout fp8_group128``: ``y_fp8`` /
# ``y_fp8_sf`` as ``per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=True)``
# (bitwise); ``--layout nvfp4``: ``y_fp4`` / ``y_fp4_sf`` as
# ``scaled_fp4_quant(y_bf16, fp4_global_scale, is_sf_swizzled_layout=False)``
# (the scale bytes bitwise, the codes within one E2M1 step) -- so the next
# layer's quantized GEMM reads them directly.
#
#   python examples/experimental/cake_mega_mhc_h4096_quant.py --tokens 64 --layout fp8_group128
#   python examples/experimental/cake_mega_mhc_h4096_quant.py --tokens 64 --layout nvfp4
#
# With vLLM installed the example compares the kernel's quantization against
# the op. ``--raw-scales`` keeps the exact ``absmax / 448`` FP8 scales
# (``fp8_sf_ue8m0=False``) instead of the power-of-two scales a UE8M0 GEMM
# reads. The NVFP4 global scale is the consumer's checkpoint constant; the
# example fits it to the output of a first BF16-output plan (``448 * 6 / amax``)
# unless ``--fp4-global-scale`` gives one.

from __future__ import annotations

import argparse

import torch
from flashinfer.mega_mhc import prepare_mega_mhc

MODEL_CONSTANTS = dict(
    hc_norm_eps=1e-5,  # the RMS statistic of the projection input
    hc_pre_eps=1e-6,
    hc_post_scale=2.0,
    sinkhorn_eps=1e-6,
    num_sinkhorn_iters=20,
    rmsnorm_eps=1e-5,  # the input RMSNorm's eps
    rmsnorm_scale=1.0,
)


def vllm_quant(y_bf16, layout, use_ue8m0, fp4_global_scale):
    """vLLM's quantization of the BF16 output, or None without vLLM."""
    try:
        if layout == "fp8_group128":
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8,
            )

            return per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=use_ue8m0)
        from vllm import _custom_ops as ops

        return ops.scaled_fp4_quant(
            y_bf16, fp4_global_scale, is_sf_swizzled_layout=False
        )
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--layout", choices=("fp8_group128", "nvfp4"), default="fp8_group128"
    )
    parser.add_argument("--raw-scales", action="store_true")
    parser.add_argument("--fp4-global-scale", type=float, default=None)
    args = parser.parse_args()
    tokens, hidden = args.tokens, 4096
    device = "cuda"
    use_ue8m0 = not args.raw_scales
    inputs = dict(
        x=torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device),
        residual=torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=device),
        post_mix=torch.full((tokens, 4, 1), 0.5, device=device),
        comb_res_mix=torch.eye(4, device=device).expand(tokens, 4, 4).contiguous(),
        shifted_prev_mix=None,  # normal mode: the step's own pre-mix feeds the collapse
        fn=torch.randn(24, 4 * hidden, device=device) * 0.01,
        mix_scales=torch.full((3,), 0.1, device=device),
        mix_bases=torch.zeros(24, device=device),
        rmsnorm_weight=torch.ones(hidden, dtype=torch.bfloat16, device=device),
    )
    quant = {"fp8_sf_ue8m0": use_ue8m0}
    if args.layout == "nvfp4":
        if args.fp4_global_scale is None:
            # The consumer's global scale is a checkpoint constant; fit one to this
            # output: 448 * 6 / amax of the BF16 output of a plain BF16-output plan.
            y_bf16 = prepare_mega_mhc(
                **inputs, **MODEL_CONSTANTS, sf_layout="bf16"
            ).run()["y_bf16"]
            gs = 448.0 * 6.0 / y_bf16.float().abs().amax()
        else:
            gs = torch.tensor(args.fp4_global_scale, device=device)
        quant = {"fp4_global_scale": gs.reshape(1).to(torch.float32)}
    plan = prepare_mega_mhc(**inputs, **MODEL_CONSTANTS, sf_layout=args.layout, **quant)
    outputs = plan.run()
    torch.cuda.synchronize()
    print(
        {
            name: (tuple(value.shape), str(value.dtype))
            for name, value in outputs.items()
        }
    )
    # fp8_group128: y_fp8 [T, 4096] float8_e4m3fn and y_fp8_sf [T, 32] float32, one
    # scale per 128-element group of a token (the layout vLLM's FP8 GEMMs read);
    # nvfp4: y_fp4 [T, 2048] packed E2M1 (element 2k in the low nibble of byte k)
    # and y_fp4_sf [T, 256] float8_e4m3fn, one scale per 16-element block.
    print("splits", plan.config["num_splits"], "ctas", plan.config["num_launch_sms"])
    reference = vllm_quant(
        outputs["y_bf16"], args.layout, use_ue8m0, quant.get("fp4_global_scale")
    )
    if reference is None:
        print("vLLM not installed: the op comparison is skipped")
        return
    codes, scales = reference
    if args.layout == "fp8_group128":
        print(
            "y_fp8 bitwise the op:",
            torch.equal(codes.view(torch.uint8), outputs["y_fp8"].view(torch.uint8)),
            "y_fp8_sf bitwise the op:",
            torch.equal(scales.reshape(tokens, -1), outputs["y_fp8_sf"]),
        )
    else:
        differing = (
            (codes.reshape(tokens, -1) != outputs["y_fp4"]).float().mean().item()
        )
        print(
            f"y_fp4 bytes differing from the op: {differing:.3e} (one E2M1 step at most);",
            "y_fp4_sf bitwise the op:",
            torch.equal(
                scales.reshape(tokens, -1).view(torch.uint8),
                outputs["y_fp4_sf"].view(torch.uint8),
            ),
        )


if __name__ == "__main__":
    main()
