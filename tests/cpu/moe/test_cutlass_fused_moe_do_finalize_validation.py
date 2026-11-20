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

``cutlass_fused_moe(do_finalize=False)`` rejects every option that the finalize step would
otherwise apply, before any device work, so these checks run on the CPU lane. GPU behavior is
covered by ``tests/moe/test_cutlass_fused_moe_do_finalize.py``.
"""

import pytest
import torch

from flashinfer.fused_moe import cutlass_fused_moe

NUM_TOKENS, HIDDEN, INTER, EXPERTS, TOP_K = 4, 128, 128, 8, 2


def _call(**kwargs):
    dtype = torch.bfloat16
    return cutlass_fused_moe(
        torch.zeros(NUM_TOKENS, HIDDEN, dtype=dtype),
        torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int32),
        torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.float32),
        torch.zeros(EXPERTS, 2 * INTER, HIDDEN, dtype=dtype),
        torch.zeros(EXPERTS, HIDDEN, INTER, dtype=dtype),
        dtype,
        quant_scales=None,
        do_finalize=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(use_fused_finalize=True), "use_fused_finalize"),
        (dict(output=torch.empty(NUM_TOKENS, HIDDEN)), "output"),
        (
            dict(
                fc1_expert_biases=torch.zeros(EXPERTS, 2 * INTER),
                fc2_expert_biases=torch.zeros(EXPERTS, HIDDEN),
            ),
            "fc2_expert_biases",
        ),
        (dict(min_latency_mode=True), "min_latency_mode"),
        (dict(use_deepseek_fp8_block_scale=True), "use_deepseek_fp8_block_scale"),
        (dict(enable_alltoall=True), "enable_alltoall"),
        (dict(backend="cake"), "cake"),
    ],
)
def test_do_finalize_false_rejects_finalize_only_options(kwargs, match):
    with pytest.raises(ValueError, match=f"do_finalize=False .*{match}"):
        _call(**kwargs)
