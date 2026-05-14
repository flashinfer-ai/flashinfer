"""
Copyright (c) 2025 by FlashInfer team.

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

"""Inputs helper for CuteDSL MoE autotune profiling.

Provides a deterministic, balanced approx-max-load distribution for
``token_selected_experts`` during autotune profiling, using rejection
sampling so each token's top_k slots are independently filled.

Without this hook, fi's autotune profile uses random expert assignments
(via ``torch.randint`` in the tensor_initializers), which:

  1. Makes profile-time inputs non-deterministic across process invocations,
     causing run-to-run tactic-pick variance at marginal cells.

  2. Produces a uniform-random expert distribution that does not match
     real workload distributions; per-tactic profile times don't reflect
     production performance differences -- e.g. tile=256 2-CTA may
     profile competitively under uniform-random load but underperform
     at runtime.

Reference: ports trt-llm's GroupedGemmInputsHelper.generate_num_tokens_per_expert
and generate_token_selected_experts from
tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py (lines 31-167) and the
``inputs_pre_hook`` mechanism from CuteDslFusedMoENvfp4InputsHelper in
tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py.
"""

import math
from typing import List

import torch


class CuteDslMoEInputsHelper:
    """Generates a balanced approx-max-load distribution for autotune profiling.

    Used as ``TuningConfig(inputs_pre_hook=helper.inputs_pre_hook, ...)`` in
    the CuteDSL MoE autotune setup. The hook intercepts profile-time inputs
    after the autotuner synthesizes them and replaces ``token_selected_experts``
    with a deterministic balanced assignment.

    Args:
        num_experts: Total number of experts in the MoE layer.
        top_k: Number of experts each token routes to.
        num_local_experts: Number of experts processed by this rank
            (= num_experts when expert parallelism is disabled).
        local_expert_offset: Starting expert index for this rank
            (= 0 when expert parallelism is disabled).
        seed: Seed for reproducibility (default 515, matches trt-llm).
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        seed: int = 515,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.seed = seed

    def generate_num_tokens_per_expert(
        self, num_tokens: int, approx_max_load: bool = False
    ) -> List[int]:
        """Compute per-expert token counts under approx-max-load assumption.

        Models worst-case load skew using the balls-into-bins concentration
        bound (https://en.wikipedia.org/wiki/Balls_into_bins_problem); c=1.0
        matches trt-llm's reference implementation.
        """
        ep_size = self.num_experts // self.num_local_experts
        average_num_tokens_per_rank = num_tokens * self.top_k / ep_size

        if approx_max_load:
            c = 1.0
            extra = c * math.sqrt(average_num_tokens_per_rank * math.log(ep_size))
            num_tokens_on_curr_rank = math.ceil(average_num_tokens_per_rank + extra)
        else:
            num_tokens_on_curr_rank = math.ceil(average_num_tokens_per_rank)

        num_tokens_on_curr_rank = min(num_tokens * self.top_k, num_tokens_on_curr_rank)

        base, remainder = divmod(num_tokens_on_curr_rank, self.num_local_experts)
        per_expert = [base + 1] * remainder + [base] * (
            self.num_local_experts - remainder
        )
        assert len(per_expert) == self.num_local_experts
        assert sum(per_expert) == num_tokens_on_curr_rank
        return per_expert

    def generate_token_selected_experts(
        self,
        num_tokens: int,
        num_tokens_per_expert: List[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Balanced rejection sampling for the autotune profile.

        For each local expert j, pick num_tokens_per_expert[j] tokens at
        random (without replacement, using a per-expert random
        permutation seeded by self.seed) and assign expert (j+offset)
        to each picked token's next free top_k slot.

        Mirrors trt-llm's GroupedGemmInputsHelper.generate_token_selected_experts.
        """
        token_selected_experts = -torch.ones(num_tokens, self.top_k, dtype=torch.int32)
        num_selected_experts = torch.zeros(num_tokens, dtype=torch.int32)

        # Seeded random permutations on CPU for cross-process reproducibility.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            selection_orders = [
                torch.randperm(num_tokens) for _ in range(self.num_local_experts)
            ]

        for j, num_tokens_j in enumerate(num_tokens_per_expert):
            selection_order_j = selection_orders[j].tolist()
            # Prioritize tokens that risk not hitting top_k before all
            # experts are exhausted. (For typical MoE configs with
            # num_local_experts < num_experts the prioritized list is
            # empty; preserved for parity with trt's algorithm.)
            #
            # `j` is the local expert index. Using it (instead of the
            # global `j + self.local_expert_offset`) makes `prioritized`
            # always empty on any rank with `local_expert_offset > 0`,
            # so the prioritization is effectively dead code on multi-rank
            # EP. Faithful port of trt-llm's implementation; tracked
            # upstream at https://github.com/NVIDIA/TensorRT-LLM/issues/14146.
            limit = self.top_k - (self.num_experts - j)
            prioritized = (
                torch.nonzero(num_selected_experts <= limit).squeeze(-1).tolist()
            )
            if len(prioritized) > 0:
                p_set = set(prioritized)
                selection_order_j = prioritized + [
                    i for i in selection_order_j if i not in p_set
                ]
            # When `num_tokens_per_expert[j] == 0`, the inner loop still
            # enters and assigns one "ghost" token to expert `j` before
            # the `num_tokens_j <= 0` break fires. Faithful port of trt-
            # llm's behavior; tracked upstream at
            # https://github.com/NVIDIA/TensorRT-LLM/issues/14146.
            for i in selection_order_j:
                if num_selected_experts[i] < self.top_k:
                    slot = int(num_selected_experts[i])
                    token_selected_experts[i, slot] = j + self.local_expert_offset
                    num_selected_experts[i] += 1
                    num_tokens_j -= 1
                    if num_tokens_j <= 0:
                        break

        return token_selected_experts.to(device=device, dtype=dtype)

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Replace random ``token_selected_experts`` with rejection-sampled
        balanced approx-max-load assignment. All other inputs pass through
        unchanged.

        Input layout matches ``CuteDslMoEWrapper.run`` argument order:

            inputs[0]: x                       (passed through)
            inputs[1]: x_sf                    (passed through)
            inputs[2]: token_selected_experts  -> REPLACED
            inputs[3..]: token_final_scales, weights, scales, alphas, output
                                               (all passed through)
        """
        x, x_sf, token_selected_experts, *rest = inputs
        num_tokens = token_selected_experts.size(0)

        per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True
        )
        new_tse = self.generate_token_selected_experts(
            num_tokens=num_tokens,
            num_tokens_per_expert=per_expert,
            device=token_selected_experts.device,
            dtype=token_selected_experts.dtype,
        )
        return [x, x_sf, new_tse, *rest]
