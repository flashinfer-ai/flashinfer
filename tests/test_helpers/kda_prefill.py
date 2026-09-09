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

"""Shared recurrent-KDA prefill test inputs.

Used by the stable lane (``tests/kda/``) and the experimental lane
(``tests/experimental/``), which cannot import each other.
"""

import torch


def cpu_route_tensors(token_count=2):
    shape = (1, token_count, 1, 128)
    return {
        "q": torch.empty(shape, dtype=torch.bfloat16),
        "k": torch.empty(shape, dtype=torch.bfloat16),
        "v": torch.empty(shape, dtype=torch.bfloat16),
        "g": torch.empty(shape, dtype=torch.bfloat16),
        "beta": torch.empty((1, token_count, 1), dtype=torch.bfloat16),
        "A_log": torch.empty(1, dtype=torch.float32),
        "dt_bias": torch.empty((1, 128), dtype=torch.float32),
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }


def packed_prefill_inputs(device, *, seq_lens, num_heads=2, seed=0):
    """Realistic packed multi-token prefill inputs on ``device``."""

    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype=torch.bfloat16, scale=1.0):
        out = torch.randn(
            shape, dtype=torch.float32, device=device, generator=generator
        )
        return (scale * out).to(dtype)

    total_tokens = sum(seq_lens)
    shape = (1, total_tokens, num_heads, 128)
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    return {
        "q": randn(shape),
        "k": randn(shape),
        "v": randn(shape),
        "g": randn(shape, scale=0.1),
        "beta": randn((1, total_tokens, num_heads)),
        "A_log": randn(num_heads, dtype=torch.float32, scale=0.1),
        "dt_bias": randn((num_heads, 128), dtype=torch.float32, scale=0.1),
        "cu_seqlens": torch.tensor(offsets, dtype=torch.int64, device=device),
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }
