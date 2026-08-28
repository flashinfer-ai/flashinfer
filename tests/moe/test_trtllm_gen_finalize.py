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

Standalone tests for the trtllm-gen MoE finalize stage (trtllm_gen_moe_finalize).

Notes on test construction:
- The synthetic permutation maps the num_tokens * top_k expanded slots into a
  larger padded row space. Unmapped rows and padded hidden columns of
  gemm2_output are filled with NaN, so any out-of-contract read (a -1 slot not
  skipped, a padded row or column touched) poisons the output and fails the
  comparison rather than passing silently.
- The token-count sweep crosses the scalar -> vectorized-load dispatch boundary
  in moe::dev::finalize::run (numBlocksX * numBlocksY >= 1184 selects
  finalizeKernelVecLoad), so both kernels are exercised.
- The E2E test checks the op against the fused launcher itself: running
  trtllm_bf16_moe with do_finalize=True must equal do_finalize=False followed
  by this op, since both run the same finalize kernels on the same buffers.
"""

import zlib

import pytest
import torch

from flashinfer import shuffle_matrix_a
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_bf16_moe,
    trtllm_gen_moe_finalize,
)
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import device_support_pdl, get_compute_capability


@pytest.fixture(autouse=True)
def require_supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = get_compute_capability(torch.device("cuda"))
    if major not in (10, 12):
        pytest.skip("trtllm-gen finalize requires SM100/SM103/SM120/SM121")


def stable_seed(*parts):
    """Deterministic across processes (unlike hash(), which salts strings)."""
    return zlib.crc32(repr(parts).encode()) % (2**31)


def reference_finalize(
    gemm2_output,
    expert_weights,
    expanded_idx_to_permuted_idx,
    hidden_size,
    lora_delta=None,
    lora_delta_scale=1.0,
    lora_apply_expert_weights=False,
):
    """Float32 reference. Skips -1 slots in the base term (robust to NaN rows);
    accumulates the delta term unconditionally, per the op contract."""
    num_tokens, top_k = expert_weights.shape
    e2p = expanded_idx_to_permuted_idx.reshape(num_tokens * top_k).long()
    active = (e2p >= 0).unsqueeze(-1)
    gathered = gemm2_output[e2p.clamp(min=0), :hidden_size].float()
    gathered = torch.where(active, gathered, torch.zeros_like(gathered))
    out = (
        gathered.reshape(num_tokens, top_k, hidden_size)
        * expert_weights.float().unsqueeze(-1)
    ).sum(dim=1)
    if lora_delta is not None:
        delta = lora_delta.float()
        if delta.dim() == 2:
            delta = delta.unsqueeze(1)
        if lora_apply_expert_weights:
            delta = delta * expert_weights.float().unsqueeze(-1)
        out = out + lora_delta_scale * delta.sum(dim=1)
    return out


def make_case(
    num_tokens,
    top_k,
    hidden_size,
    *,
    hidden_pad=0,
    extra_rows=64,
    inactive_frac=0.0,
    dtype_out=torch.bfloat16,
    dtype_expw=torch.bfloat16,
    delta_kind="none",
    seed=0,
):
    """Synthetic unfinalized-MoE buffers with NaN poison outside the contract."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    num_expanded = num_tokens * top_k
    num_padded = num_expanded + extra_rows

    # Map each expanded slot to a distinct padded row; NaN-fill unmapped rows
    # and padded hidden columns so out-of-contract reads are caught.
    row_of_slot = torch.randperm(num_padded, device=device)[:num_expanded]
    gemm2_output = torch.full(
        (num_padded, hidden_size + hidden_pad),
        float("nan"),
        dtype=dtype_out,
        device=device,
    )
    gemm2_output[row_of_slot, :hidden_size] = torch.randn(
        num_expanded, hidden_size, dtype=dtype_out, device=device
    )

    expanded_idx_to_permuted_idx = row_of_slot.to(torch.int32).reshape(
        num_tokens, top_k
    )
    if inactive_frac > 0:
        inactive = torch.rand(num_tokens, top_k, device=device) < inactive_frac
        expanded_idx_to_permuted_idx = torch.where(
            inactive, -1, expanded_idx_to_permuted_idx
        )
        # NaN-poison the rows the now-inactive slots pointed to: the kernel
        # must not read them.
        gemm2_output[row_of_slot.reshape(num_tokens, top_k)[inactive].long()] = float(
            "nan"
        )
    else:
        inactive = torch.zeros(num_tokens, top_k, dtype=torch.bool, device=device)

    expert_weights = torch.rand(num_tokens, top_k, dtype=dtype_expw, device=device)

    if delta_kind == "none":
        lora_delta = None
    elif delta_kind == "2d":
        lora_delta = torch.randn(
            num_tokens, hidden_size, dtype=dtype_out, device=device
        )
    else:  # per-slot 3D; rows of inactive slots must be zero-filled
        lora_delta = torch.randn(
            num_tokens, top_k, hidden_size, dtype=dtype_out, device=device
        )
        lora_delta[inactive] = 0
    return gemm2_output, expert_weights, expanded_idx_to_permuted_idx, lora_delta


@pytest.mark.parametrize(
    "num_tokens",
    [1, 33, 256, 4096],  # crosses the scalar -> VecLoad dispatch boundary
)
@pytest.mark.parametrize("top_k", [1, 6, 8])  # TopKUnrollFactor 1, 2, 4
@pytest.mark.parametrize("hidden_size", [512, 2880])
@pytest.mark.parametrize("delta_kind", ["none", "2d", "3d", "3d_weighted"])
def test_finalize_matches_reference(num_tokens, top_k, hidden_size, delta_kind):
    apply_w = delta_kind == "3d_weighted"
    gemm2_output, expert_weights, e2p, lora_delta = make_case(
        num_tokens,
        top_k,
        hidden_size,
        hidden_pad=64 if hidden_size == 2880 else 0,
        inactive_frac=0.3,
        delta_kind="3d" if apply_w else delta_kind,
        seed=stable_seed(num_tokens, top_k, hidden_size, delta_kind),
    )
    lora_delta_scale = 0.5 if lora_delta is not None else 1.0

    out = trtllm_gen_moe_finalize(
        gemm2_output,
        expert_weights,
        e2p,
        lora_delta=lora_delta,
        lora_delta_scale=lora_delta_scale,
        lora_apply_expert_weights=apply_w,
        hidden_size=hidden_size,
    )
    ref = reference_finalize(
        gemm2_output,
        expert_weights,
        e2p,
        hidden_size,
        lora_delta=lora_delta,
        lora_delta_scale=lora_delta_scale,
        lora_apply_expert_weights=apply_w,
    )

    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == gemm2_output.dtype
    assert not out.isnan().any(), "output poisoned by an out-of-contract read"
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype_out", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dtype_expw", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [33, 256])  # scalar and VecLoad kernels
def test_finalize_dtypes(dtype_out, dtype_expw, num_tokens):
    gemm2_output, expert_weights, e2p, lora_delta = make_case(
        num_tokens,
        8,
        2880,
        hidden_pad=64,
        inactive_frac=0.3,
        dtype_out=dtype_out,
        dtype_expw=dtype_expw,
        delta_kind="3d",
        seed=stable_seed(str(dtype_out), str(dtype_expw), num_tokens),
    )
    out = trtllm_gen_moe_finalize(
        gemm2_output, expert_weights, e2p, lora_delta=lora_delta, hidden_size=2880
    )
    ref = reference_finalize(
        gemm2_output, expert_weights, e2p, 2880, lora_delta=lora_delta
    )
    assert out.dtype == dtype_out
    assert not out.isnan().any()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_finalize_rejects_bad_inputs():
    gemm2_output, expert_weights, e2p, _ = make_case(8, 4, 512, seed=0)
    with pytest.raises(Exception, match="lora_apply_expert_weights"):
        trtllm_gen_moe_finalize(
            gemm2_output,
            expert_weights,
            e2p,
            lora_delta=torch.zeros(8, 512, dtype=torch.bfloat16, device="cuda"),
            lora_apply_expert_weights=True,
        )
    with pytest.raises(Exception, match="dtype"):
        trtllm_gen_moe_finalize(
            gemm2_output,
            expert_weights,
            e2p,
            lora_delta=torch.zeros(8, 4, 512, dtype=torch.float32, device="cuda"),
        )


@pytest.mark.parametrize("num_tokens", [8, 300])  # scalar and VecLoad kernels
def test_finalize_matches_fused_moe(num_tokens):
    """E2E identity: trtllm_bf16_moe(do_finalize=True) must equal
    do_finalize=False + trtllm_gen_moe_finalize, which runs the same kernels."""
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm_bf16_moe is only supported on SM100/SM103")
    from flashinfer.autotuner import AutoTuner

    # The bitwise comparison requires both calls to pick the same GEMM tactic;
    # a profiling cache warmed by an earlier test could differentiate them
    # (do_finalize changes the cache key via the output placeholder).
    AutoTuner.get().clear_cache()
    torch.manual_seed(stable_seed("e2e", num_tokens))
    device = torch.device("cuda")
    enable_pdl = device_support_pdl(device)
    hidden_size = 1024
    intermediate_size = 1024
    num_experts = 16
    top_k = 4

    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.bfloat16)
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.bfloat16)
    block_k = 128
    gemm1_weights = torch.stack(
        [
            convert_to_block_layout(
                shuffle_matrix_a(gemm1_weights[i].view(torch.uint8), 64), block_k
            )
            for i in range(num_experts)
        ]
    ).view(torch.bfloat16)
    gemm2_weights = torch.stack(
        [
            convert_to_block_layout(
                shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), 64), block_k
            )
            for i in range(num_experts)
        ]
    ).view(torch.bfloat16)

    def run(do_finalize):
        return trtllm_bf16_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize.value,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
        )

    fused = run(do_finalize=True)
    gemm2_output, expert_weights, e2p = run(do_finalize=False)

    out = trtllm_gen_moe_finalize(
        gemm2_output,
        expert_weights,
        e2p,
        hidden_size=hidden_size,
        enable_pdl=enable_pdl,
    )
    torch.testing.assert_close(out, fused, atol=0.0, rtol=0.0)
