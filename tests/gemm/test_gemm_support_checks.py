"""
Test file for GEMM API support checks.

This file serves as a TODO list for support check implementations.
APIs with @pytest.mark.xfail need support checks to be implemented in gemm_base.py.
"""

import pytest

from flashinfer import (
    bmm_fp8,
    mm_fp4,
    mm_fp8,
    prepare_low_latency_gemm_weights,
    tgv_gemm_sm100,
)
from flashinfer.gemm import (
    SegmentGEMMWrapper,
    batch_deepgemm_fp8_nt_groupwise,
    gemm_fp8_nt_blockscaled,
    gemm_fp8_nt_groupwise,
    group_deepgemm_fp8_nt_groupwise,
    group_gemm_fp8_nt_groupwise,
    group_gemm_mxfp4_nt_groupwise,
)
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from flashinfer.cute_dsl.gemm_allreduce_two_shot import PersistentDenseGemmKernel
import flashinfer.triton.sm_constraint_gemm as sm_constraint_gemm


def test_mm_fp4_support_checks():
    assert hasattr(mm_fp4, "is_compute_capability_supported")
    assert hasattr(mm_fp4, "is_backend_supported")


def test_bmm_fp8_support_checks():
    assert hasattr(bmm_fp8, "is_compute_capability_supported")
    assert hasattr(bmm_fp8, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for tgv_gemm_sm100 are not implemented")
def test_tgv_gemm_sm100_support_checks():
    assert hasattr(tgv_gemm_sm100, "is_compute_capability_supported")
    assert hasattr(tgv_gemm_sm100, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for mm_fp8 are not implemented")
def test_mm_fp8_support_checks():
    assert hasattr(mm_fp8, "is_compute_capability_supported")
    assert hasattr(mm_fp8, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for gemm_fp8_nt_groupwise are not implemented"
)
def test_gemm_fp8_nt_groupwise_support_checks():
    assert hasattr(gemm_fp8_nt_groupwise, "is_compute_capability_supported")
    assert hasattr(gemm_fp8_nt_groupwise, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for gemm_fp8_nt_blockscaled are not implemented"
)
def test_gemm_fp8_nt_blockscaled_support_checks():
    assert hasattr(gemm_fp8_nt_blockscaled, "is_compute_capability_supported")
    assert hasattr(gemm_fp8_nt_blockscaled, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for group_gemm_fp8_nt_groupwise are not implemented"
)
def test_group_gemm_fp8_nt_groupwise_support_checks():
    assert hasattr(group_gemm_fp8_nt_groupwise, "is_compute_capability_supported")
    assert hasattr(group_gemm_fp8_nt_groupwise, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for group_gemm_mxfp4_nt_groupwise are not implemented"
)
def test_group_gemm_mxfp4_nt_groupwise_support_checks():
    assert hasattr(group_gemm_mxfp4_nt_groupwise, "is_compute_capability_supported")
    assert hasattr(group_gemm_mxfp4_nt_groupwise, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for group_deepgemm_fp8_nt_groupwise are not implemented"
)
def test_group_deepgemm_fp8_nt_groupwise_support_checks():
    assert hasattr(group_deepgemm_fp8_nt_groupwise, "is_compute_capability_supported")
    assert hasattr(group_deepgemm_fp8_nt_groupwise, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for batch_deepgemm_fp8_nt_groupwise are not implemented"
)
def test_batch_deepgemm_fp8_nt_groupwise_support_checks():
    assert hasattr(batch_deepgemm_fp8_nt_groupwise, "is_compute_capability_supported")
    assert hasattr(batch_deepgemm_fp8_nt_groupwise, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for SegmentGEMMWrapper are not implemented"
)
def test_segment_gemm_wrapper_support_checks():
    assert hasattr(SegmentGEMMWrapper.run, "is_compute_capability_supported")
    assert hasattr(SegmentGEMMWrapper.run, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for prepare_low_latency_gemm_weights are not implemented"
)
def test_prepare_low_latency_gemm_weights_support_checks():
    assert hasattr(prepare_low_latency_gemm_weights, "is_compute_capability_supported")
    assert hasattr(prepare_low_latency_gemm_weights, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for grouped_gemm_nt_masked are not implemented"
)
def test_grouped_gemm_nt_masked_support_checks():
    assert hasattr(grouped_gemm_nt_masked, "is_compute_capability_supported")
    assert hasattr(grouped_gemm_nt_masked, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for PersistentDenseGemmKernel are not implemented"
)
def test_persistent_dense_gemm_kernel_support_checks():
    assert hasattr(PersistentDenseGemmKernel, "is_compute_capability_supported")
    assert hasattr(PersistentDenseGemmKernel, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for sm_constraint_gemm.gemm are not implemented"
)
def test_sm_constraint_gemm_support_checks():
    assert hasattr(sm_constraint_gemm.gemm, "is_compute_capability_supported")
    assert hasattr(sm_constraint_gemm.gemm, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for sm_constraint_gemm.gemm_persistent are not implemented"
)
def test_sm_constraint_gemm_persistent_support_checks():
    assert hasattr(
        sm_constraint_gemm.gemm_persistent, "is_compute_capability_supported"
    )
    assert hasattr(sm_constraint_gemm.gemm_persistent, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for sm_constraint_gemm.gemm_descriptor_persistent are not implemented"
)
def test_sm_constraint_gemm_descriptor_persistent_support_checks():
    assert hasattr(
        sm_constraint_gemm.gemm_descriptor_persistent, "is_compute_capability_supported"
    )
    assert hasattr(
        sm_constraint_gemm.gemm_descriptor_persistent, "is_backend_supported"
    )
