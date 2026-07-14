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

import pytest
import torch
from torch.nn import functional as F

import flashinfer.fused_moe as fused_moe
from flashinfer.utils import is_sm90a_supported

pytestmark = pytest.mark.solo

BATCH_SIZES = [1, 4, 16]
HIDDEN_SIZES = [128, 2048]
NUM_EXPERTS = [2, 8]
TOP_K_VALUES = [2, 6]
INTERMEDIATE_SIZES = [128, 1024]


def compute_routing(router_logits, top_k):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.float(), selected_experts


def reference_finalize(gemm2_output, token_final_scales, expanded_idx):
    """Vectorized manual finalize: gather, scale, sum."""
    num_tokens, top_k = token_final_scales.shape
    hidden_size = gemm2_output.shape[1]
    gathered = gemm2_output[expanded_idx.long()].view(num_tokens, top_k, hidden_size)
    weights = token_final_scales.unsqueeze(-1)
    return (gathered.float() * weights.float()).sum(dim=1).to(gemm2_output.dtype)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda:0")),
    reason="Requires SM90a (H100/H200)",
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_do_finalize_false(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """``do_finalize=False`` skips the finalize kernel and returns raw GEMM2 output.

    Verifies that manual finalize of the returned data matches the pure-Python
    reference implementation, and that results are deterministic across calls.
    """
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) > num_experts ({num_experts})")

    torch.manual_seed(42)
    dtype = torch.bfloat16
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda") / 5
    router_logits = torch.randn(
        batch_size, num_experts, dtype=torch.float32, device="cuda"
    )
    w31_weight = (
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device="cuda"
        )
        / 5
    )
    w2_weight = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=dtype, device="cuda"
        )
        / 5
    )

    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    # Reference: do_finalize=True with use_fused_finalize=False (same autotuned runner)
    ref_raw = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w31_weight,
        w2_weight,
        dtype,
        quant_scales=None,
        use_fused_finalize=False,
        do_finalize=True,
    )
    ref_output = ref_raw[0] if isinstance(ref_raw, list) else ref_raw

    # do_finalize=False: get raw GEMM2 output + mapping
    result = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w31_weight,
        w2_weight,
        dtype,
        quant_scales=None,
        use_fused_finalize=False,
        do_finalize=False,
    )

    assert isinstance(result, list)
    assert len(result) == 3
    gemm2_output, scales_out, expanded_idx = result

    # Shape checks
    assert gemm2_output.shape == (batch_size * top_k, hidden_size)
    assert scales_out.shape == (batch_size, top_k)
    assert expanded_idx.shape == (batch_size * top_k,)
    assert expanded_idx.dtype == torch.int32

    # Scales passed through unchanged
    assert torch.equal(scales_out, routing_weights)

    # Permutation indices in valid range
    assert expanded_idx.min() >= 0
    assert expanded_idx.max() < batch_size * top_k

    # Manual finalize matches the kernel's own finalize (same runner, same tactics)
    manual_output = reference_finalize(gemm2_output, scales_out, expanded_idx)
    torch.testing.assert_close(ref_output, manual_output, rtol=1e-2, atol=1e-2)

    # Determinism: run again, must be bit-exact
    result2 = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w31_weight,
        w2_weight,
        dtype,
        quant_scales=None,
        use_fused_finalize=False,
        do_finalize=False,
    )
    gemm2_output2, _, expanded_idx2 = result2
    assert torch.equal(gemm2_output, gemm2_output2), "GEMM2 output not deterministic"
    assert torch.equal(expanded_idx, expanded_idx2), (
        "Permutation mapping not deterministic"
    )
