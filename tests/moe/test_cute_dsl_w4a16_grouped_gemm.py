# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Focused CuTe DSL W4A16 grouped-GEMM tests.

Run this file while iterating on the W4A16 CuTe kernels::

    pytest -q tests/moe/test_cute_dsl_w4a16_grouped_gemm.py

The full fused-MoE tests remain the integration gate for routing and
permutation.  This file instead constructs their post-``moe_sort`` contract
directly so kernel-only changes do not build the unrelated native TRT-LLM
routing module.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported
from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)


cute_dsl_required = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuTe DSL is not available"
)
sm100_family_required = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and is_cute_dsl_arch_supported(
            *torch.cuda.get_device_capability(0), native_only=True
        )
    ),
    reason="Requires a CuTe DSL MoE target in the SM100 family",
)

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _deinterleave_linear_and_gate(
    x: torch.Tensor, group_size: int = 64
) -> torch.Tensor:
    """Convert FC1 kernel rows back to logical ``[linear, gate]`` order."""
    num_experts, rows, k = x.shape
    intermediate_size = rows // 2
    assert rows % (2 * group_size) == 0
    return (
        x.view(num_experts, intermediate_size // group_size, 2, group_size, k)
        .transpose(1, 2)
        .contiguous()
        .view_as(x)
    )


def _make_nvfp4_weight(
    num_experts: int,
    rows: int,
    k: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create packed NVFP4 weights without invoking a CUDA JIT quantizer."""
    assert rows % 128 == 0
    assert k % 64 == 0
    device = torch.device("cuda")

    # Use small signed E2M1 values so the two GEMMs remain numerically stable.
    codebook = torch.tensor([1, 2, 9, 10], dtype=torch.uint8, device=device)
    logical_idx = torch.arange(
        num_experts * rows * k, dtype=torch.int64, device=device
    ).view(num_experts, rows, k)
    codes = codebook[(logical_idx * 5 + seed) % codebook.numel()]
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()

    # Logical E4M3 scales vary by expert, row, and K block.  Convert them to
    # the 128x4 physical swizzle and then expose the strided MMA view expected
    # by the kernel.  These are view/permute operations only.
    scale_codebook = torch.tensor(
        [0x20, 0x28, 0x30], dtype=torch.uint8, device=device
    )  # 0.125, 0.25, 0.5
    logical_scale_idx = torch.arange(
        num_experts * rows * (k // 16), dtype=torch.int64, device=device
    ).view(num_experts, rows, k // 16)
    logical_scale_codes = scale_codebook[
        (logical_scale_idx * 5 + seed) % scale_codebook.numel()
    ]
    physical_scale = (
        logical_scale_codes.view(num_experts, rows // 128, 4, 32, k // 64, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
    )
    mma_scale = physical_scale.permute(3, 4, 1, 5, 2, 0)

    magnitude = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
    decoded_codes = magnitude[(codes & 0x7).long()]
    decoded_codes = torch.where((codes & 0x8) != 0, -decoded_codes, decoded_codes)
    decoded_scales = logical_scale_codes.view(torch.float8_e4m3fn).float()
    reference = decoded_codes * decoded_scales.repeat_interleave(16, dim=-1)
    return packed, mma_scale, reference


def _make_post_sort_contract(
    x: torch.Tensor, *, expert_ids: tuple[int, ...], top_k: int, route_tile: int
) -> dict[str, torch.Tensor]:
    """Construct the exact routed-row metadata consumed by grouped GEMM."""
    assert top_k == len(expert_ids)
    num_tokens = x.size(0)
    tiles_per_expert = (num_tokens + route_tile - 1) // route_tile
    rows_per_expert = tiles_per_expert * route_tile
    route_slots = len(expert_ids) * rows_per_expert

    activations = torch.zeros((route_slots, x.size(1)), dtype=x.dtype, device=x.device)
    permuted_idx_to_expanded_idx = torch.full(
        (route_slots,), -1, dtype=torch.int32, device=x.device
    )
    token_ids = torch.arange(num_tokens, dtype=torch.int32, device=x.device)
    for topk_idx in range(len(expert_ids)):
        start = topk_idx * rows_per_expert
        activations[start : start + num_tokens] = x
        permuted_idx_to_expanded_idx[start : start + num_tokens] = (
            token_ids * top_k + topk_idx
        )

    tile_idx_to_expert_idx = torch.tensor(
        expert_ids, dtype=torch.int32, device=x.device
    ).repeat_interleave(tiles_per_expert)
    tile_idx_to_mn_limit = torch.tensor(
        [
            topk_idx * rows_per_expert + min((tile_idx + 1) * route_tile, num_tokens)
            for topk_idx in range(len(expert_ids))
            for tile_idx in range(tiles_per_expert)
        ],
        dtype=torch.int32,
        device=x.device,
    )
    return {
        "activations": activations,
        "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
        "tile_idx_to_expert_idx": tile_idx_to_expert_idx,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "num_non_exiting_tiles": torch.tensor(
            [tile_idx_to_expert_idx.numel()], dtype=torch.int32, device=x.device
        ),
    }


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_f = actual.float().reshape(-1)
    expected_f = expected.float().reshape(-1)
    expected_norm = torch.linalg.vector_norm(expected_f).clamp_min(1e-6)
    relative_l2 = torch.linalg.vector_norm(actual_f - expected_f) / expected_norm
    cosine = torch.nn.functional.cosine_similarity(actual_f, expected_f, dim=0)
    assert cosine.item() > 0.99, f"cosine similarity is {cosine.item():.6f}"
    assert relative_l2.item() < 0.08, f"relative L2 error is {relative_l2.item():.6f}"


@cute_dsl_required
@sm100_family_required
def test_w4a16_grouped_gemm_pipeline_without_native_routing(monkeypatch):
    """Cover GEMM1, GEMM2, and finalize without compiling ``moe_utils``."""
    from flashinfer.jit import core as jit_core

    def fail_if_native_jit_is_loaded(*_args, **_kwargs):
        raise AssertionError("focused W4A16 kernel test loaded a native JIT module")

    monkeypatch.setattr(
        jit_core.JitSpecNvcc, "build_and_load", fail_if_native_jit_is_loaded
    )
    from flashinfer.fused_moe.cute_dsl import moe_utils

    monkeypatch.setattr(
        moe_utils, "_get_moe_utils_module", fail_if_native_jit_is_loaded
    )
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import (
        _run_grouped_gemm,
    )

    torch.manual_seed(42)
    num_tokens, hidden_size, intermediate_size = 33, 256, 512
    num_experts, top_k, route_tile = 8, 2, 32
    expert_ids = (0, 1)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") / 8
    token_final_scales = (
        torch.tensor([0.4, 0.6], dtype=torch.float32, device="cuda")
        .expand(num_tokens, top_k)
        .contiguous()
    )
    routing = _make_post_sort_contract(
        x, expert_ids=expert_ids, top_k=top_k, route_tile=route_tile
    )

    w1_physical, w1_sf, w1_physical_ref = _make_nvfp4_weight(
        num_experts, 2 * intermediate_size, hidden_size, seed=1
    )
    w1_ref = _deinterleave_linear_and_gate(w1_physical_ref)
    w2, w2_sf, w2_ref = _make_nvfp4_weight(
        num_experts, hidden_size, intermediate_size, seed=2
    )
    w1_alpha = torch.linspace(
        0.75, 1.25, num_experts, dtype=torch.float32, device="cuda"
    )
    w2_alpha = torch.linspace(
        1.25, 0.75, num_experts, dtype=torch.float32, device="cuda"
    )
    gemm1_tactic = ((128, route_tile, 256), (2, 1), True)
    gemm2_tactic = ((256, route_tile, 256), (2, 1), True)

    intermediate = torch.empty(
        (routing["activations"].size(0), intermediate_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    common = {
        "tile_idx_to_expert_idx": routing["tile_idx_to_expert_idx"],
        "tile_idx_to_mn_limit": routing["tile_idx_to_mn_limit"],
        "num_non_exiting_tiles": routing["num_non_exiting_tiles"],
        "num_local_experts": num_experts,
        "swiglu_alpha": DEFAULT_SWIGLU_ALPHA,
        "swiglu_beta": DEFAULT_SWIGLU_BETA,
        "swiglu_limit": DEFAULT_SWIGLU_LIMIT,
        "situ_beta": None,
        "situ_linear_beta": None,
        "enable_pdl": False,
    }
    _run_grouped_gemm(
        weight=w1_physical,
        weight_sf=w1_sf,
        activations=routing["activations"],
        alpha=w1_alpha,
        output=intermediate,
        activation_type=ActivationType.Swiglu,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=None,
        token_final_scales=None,
        tactic=gemm1_tactic,
        **common,
    )

    expected_intermediate = torch.zeros_like(intermediate)
    rows_per_expert = intermediate.size(0) // top_k
    for topk_idx, expert_idx in enumerate(expert_ids):
        projected = torch.nn.functional.linear(x.float(), w1_ref[expert_idx])
        linear, gate = projected.chunk(2, dim=-1)
        linear = linear * w1_alpha[expert_idx]
        gate = gate * w1_alpha[expert_idx]
        expected_intermediate[
            topk_idx * rows_per_expert : topk_idx * rows_per_expert + num_tokens
        ] = (linear * torch.nn.functional.silu(gate)).bfloat16()
    valid_routes = routing["permuted_idx_to_expanded_idx"] >= 0
    _assert_close(intermediate[valid_routes], expected_intermediate[valid_routes])

    routed_output = torch.empty(
        (intermediate.size(0), hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    _run_grouped_gemm(
        weight=w2,
        weight_sf=w2_sf,
        activations=intermediate,
        alpha=w2_alpha,
        output=routed_output,
        activation_type=None,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=None,
        token_final_scales=None,
        tactic=gemm2_tactic,
        **common,
    )

    expected_routed_output = torch.zeros_like(routed_output)
    # Stage the reference on the already-validated BF16 GEMM1 output so this
    # assertion isolates GEMM2 weight decoding and accumulation.
    for topk_idx, expert_idx in enumerate(expert_ids):
        start = topk_idx * rows_per_expert
        expected_routed_output[start : start + num_tokens] = (
            torch.nn.functional.linear(
                intermediate[start : start + num_tokens].float(), w2_ref[expert_idx]
            )
            * w2_alpha[expert_idx]
        ).bfloat16()
    _assert_close(routed_output[valid_routes], expected_routed_output[valid_routes])

    fused_output = torch.zeros(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    _run_grouped_gemm(
        weight=w2,
        weight_sf=w2_sf,
        activations=intermediate,
        alpha=w2_alpha,
        output=fused_output,
        activation_type=None,
        use_fused_finalize=True,
        permuted_idx_to_expanded_idx=routing["permuted_idx_to_expanded_idx"],
        token_final_scales=token_final_scales,
        tactic=gemm2_tactic,
        **common,
    )
    # Likewise, isolate fused-finalize routing/scaling from GEMM2 numerics.
    expected_fused_output = sum(
        routed_output[
            topk_idx * rows_per_expert : topk_idx * rows_per_expert + num_tokens
        ].float()
        * token_final_scales[:, topk_idx, None]
        for topk_idx in range(top_k)
    ).bfloat16()
    _assert_close(fused_output, expected_fused_output)
