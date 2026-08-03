"""
Test file for MoE API support checks.

This file serves as a TODO list for support check implementations.
APIs with @pytest.mark.xfail need support checks to be implemented.
"""

import pytest

from flashinfer.fused_moe import (
    cutlass_fused_moe,
    fused_topk_deepseek,
    trtllm_bf16_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_mxint4_block_scale_moe,
)
from flashinfer.comm.trtllm_moe_alltoall import (
    moe_a2a_combine,
    moe_a2a_dispatch,
    moe_a2a_get_workspace_size_per_rank,
    moe_a2a_initialize,
    moe_a2a_sanitize_expert_ids,
    moe_a2a_wrap_payload_tensor_in_workspace,
)


def test_fused_topk_deepseek_support_checks():
    assert hasattr(fused_topk_deepseek, "is_compute_capability_supported")
    assert hasattr(fused_topk_deepseek, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for cutlass_fused_moe are not implemented"
)
def test_cutlass_fused_moe_support_checks():
    assert hasattr(cutlass_fused_moe, "is_compute_capability_supported")
    assert hasattr(cutlass_fused_moe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_bf16_moe are not implemented"
)
def test_trtllm_bf16_moe_support_checks():
    assert hasattr(trtllm_bf16_moe, "is_compute_capability_supported")
    assert hasattr(trtllm_bf16_moe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_fp8_per_tensor_scale_moe are not implemented"
)
def test_trtllm_fp8_per_tensor_scale_moe_support_checks():
    assert hasattr(trtllm_fp8_per_tensor_scale_moe, "is_compute_capability_supported")
    assert hasattr(trtllm_fp8_per_tensor_scale_moe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_fp8_block_scale_moe are not implemented"
)
def test_trtllm_fp8_block_scale_moe_support_checks():
    assert hasattr(trtllm_fp8_block_scale_moe, "is_compute_capability_supported")
    assert hasattr(trtllm_fp8_block_scale_moe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_fp4_block_scale_moe are not implemented"
)
def test_trtllm_fp4_block_scale_moe_support_checks():
    assert hasattr(trtllm_fp4_block_scale_moe, "is_compute_capability_supported")
    assert hasattr(trtllm_fp4_block_scale_moe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_fp4_block_scale_routed_moe are not implemented"
)
def test_trtllm_fp4_block_scale_routed_moe_support_checks():
    assert hasattr(trtllm_fp4_block_scale_routed_moe, "is_compute_capability_supported")
    assert hasattr(trtllm_fp4_block_scale_routed_moe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_mxint4_block_scale_moe are not implemented"
)
def test_trtllm_mxint4_block_scale_moe_support_checks():
    assert hasattr(trtllm_mxint4_block_scale_moe, "is_compute_capability_supported")
    assert hasattr(trtllm_mxint4_block_scale_moe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for moe_a2a_initialize are not implemented"
)
def test_moe_a2a_initialize_support_checks():
    assert hasattr(moe_a2a_initialize, "is_compute_capability_supported")
    assert hasattr(moe_a2a_initialize, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for moe_a2a_wrap_payload_tensor_in_workspace are not implemented"
)
def test_moe_a2a_wrap_payload_tensor_in_workspace_support_checks():
    assert hasattr(
        moe_a2a_wrap_payload_tensor_in_workspace, "is_compute_capability_supported"
    )
    assert hasattr(moe_a2a_wrap_payload_tensor_in_workspace, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for moe_a2a_dispatch are not implemented"
)
def test_moe_a2a_dispatch_support_checks():
    assert hasattr(moe_a2a_dispatch, "is_compute_capability_supported")
    assert hasattr(moe_a2a_dispatch, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for moe_a2a_combine are not implemented"
)
def test_moe_a2a_combine_support_checks():
    assert hasattr(moe_a2a_combine, "is_compute_capability_supported")
    assert hasattr(moe_a2a_combine, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for moe_a2a_sanitize_expert_ids are not implemented"
)
def test_moe_a2a_sanitize_expert_ids_support_checks():
    assert hasattr(moe_a2a_sanitize_expert_ids, "is_compute_capability_supported")
    assert hasattr(moe_a2a_sanitize_expert_ids, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for moe_a2a_get_workspace_size_per_rank are not implemented"
)
def test_moe_a2a_get_workspace_size_per_rank_support_checks():
    assert hasattr(
        moe_a2a_get_workspace_size_per_rank, "is_compute_capability_supported"
    )
    assert hasattr(moe_a2a_get_workspace_size_per_rank, "is_backend_supported")
