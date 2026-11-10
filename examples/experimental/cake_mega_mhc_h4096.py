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

# The fused mHC inference step of a GLM-style four-route model (hidden 4096, BF16
# output) through the experimental ``prepare_mega_mhc`` API on an SM100a, SM103a or
# SM107a GPU.
#
# The caller's names (``residual_cur``, ``post_mix_cur``, ``comb_mix_cur``,
# ``layer_input_cur``) map onto the plan's outputs (``new_residual``,
# ``new_post_mix``, ``new_comb_res_mix``, ``y_bf16``); the model's constants are
# keyword arguments. ``fn`` is captured when the plan is prepared: a changed
# ``fn`` reaches the kernel through ``plan.update_fn()``.
#
#   python examples/experimental/cake_mega_mhc_h4096.py --tokens 64

from __future__ import annotations

import argparse

import torch
from flashinfer.mega_mhc import prepare_mega_mhc

GLM_CONSTANTS = dict(
    hc_norm_eps=1e-5,  # the RMS statistic of the projection input (rms_eps)
    hc_pre_eps=1e-6,
    hc_post_scale=2.0,  # hc_post_mult_value
    sinkhorn_eps=1e-6,  # hc_sinkhorn_eps
    num_sinkhorn_iters=20,
    rmsnorm_eps=1e-5,  # the input RMSNorm's eps (norm_eps)
    rmsnorm_scale=1.0,
)


def fused_post_pre(plan):
    """One step: the four outputs under the caller's names."""
    outputs = plan.run()
    return dict(
        residual_cur=outputs["new_residual"],
        post_mix_cur=outputs["new_post_mix"],
        comb_mix_cur=outputs["new_comb_res_mix"],
        layer_input_cur=outputs["y_bf16"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    tokens, hidden = args.tokens, 4096
    device = "cuda"
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
    plan = prepare_mega_mhc(**inputs, **GLM_CONSTANTS, sf_layout="bf16")
    outputs = fused_post_pre(plan)
    torch.cuda.synchronize()
    print(
        {
            name: (tuple(value.shape), str(value.dtype))
            for name, value in outputs.items()
        }
    )
    # A changed projection: rewrite fn in place, then re-split it for the kernel.
    inputs["fn"].mul_(0.5)
    plan.update_fn()
    fused_post_pre(plan)
    torch.cuda.synchronize()
    print("splits", plan.config["num_splits"], "ctas", plan.config["num_launch_sms"])


if __name__ == "__main__":
    main()
