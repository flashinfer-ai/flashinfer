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

"""Experimental prepared W4A8 routed MoE API."""

from ..api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_fp4_block_scale_routed_moe(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    topk_ids,
    topk_weights,
    *,
    local_expert_offset,
    output,
    backend="cake",
):
    """Prepare a reusable W4A8 MoE call over received token rows on GB300.

    The explicit ``cake`` backend supports H=3072, I=5120, 512 global experts,
    32 contiguous local experts, top-k 8 and 1..8192 received rows. Activations
    are E4M3 with linear uint8 E8M0 scales of shape ``[rows, H/32]``. Weights
    use packed E2M1 bytes with linear E8M0 block-32 scales, in logical gate/up
    order; preparation packs them once. IDs are int32 and routing weights are
    FP32. Every original route slot, including remote and duplicate IDs, must
    be retained without renormalizing its weight.

    ``output`` is caller-owned BF16 ``[rows, H]``. The returned object's
    ``run()`` reads live activation and routing contents at the prepared
    addresses and returns this output. It submits four kernels with PDL and
    allocates no device memory. Warm up once before external CUDA Graph
    capture. Recreate the object to change tensor addresses or dimensions.

    The output sums only this rank's expert contributions. Dispatch and
    cross-rank combination belong to the caller. Serialize calls sharing one
    prepared workspace; weights are immutable after preparation. Empty input
    is unsupported. This route requires a 152-SM GB300 and a compatible
    CUTLASS CuTe DSL installation.
    """
    if backend != "cake":
        raise ValueError("prepare_fp4_block_scale_routed_moe supports backend='cake'")
    from ..experimental.cake_w4a8_received_tokens import cake_backend

    return cake_backend.PreparedRoutedMoE(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        topk_ids,
        topk_weights,
        local_expert_offset=local_expert_offset,
        out=output,
    )
