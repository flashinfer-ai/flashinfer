"""
Test file for Comm API support checks.

This file serves as a TODO list for support check implementations.
APIs with @pytest.mark.xfail need support checks to be implemented.
"""

import pytest

from flashinfer.comm import (
    trtllm_allreduce_fusion,
    trtllm_custom_all_reduce,
    trtllm_moe_allreduce_fusion,
    trtllm_moe_finalize_allreduce_fusion,
    trtllm_create_ipc_workspace_for_all_reduce,
    trtllm_destroy_ipc_workspace_for_all_reduce,
    trtllm_create_ipc_workspace_for_all_reduce_fusion,
    trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
    trtllm_lamport_initialize,
    trtllm_lamport_initialize_all,
    vllm_all_reduce,
    vllm_init_custom_ar,
    vllm_dispose,
    vllm_register_buffer,
    vllm_register_graph_buffers,
    MoeAlltoAll,
)
from flashinfer.comm.nvshmem_allreduce import NVSHMEMAllReduce
from flashinfer.comm.mnnvl import MnnvlMemory
from flashinfer.comm.trtllm_alltoall import MnnvlMoe, MoEAlltoallInfo
from flashinfer.comm.trtllm_mnnvl_ar import (
    trtllm_mnnvl_allreduce,
    trtllm_mnnvl_fused_allreduce_add_rmsnorm,
    trtllm_mnnvl_fused_allreduce_rmsnorm,
    trtllm_mnnvl_all_reduce,
)


# TRTLLM AllReduce APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_allreduce_fusion are not implemented"
)
def test_trtllm_allreduce_fusion_support_checks():
    assert hasattr(trtllm_allreduce_fusion, "is_compute_capability_supported")
    assert hasattr(trtllm_allreduce_fusion, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_custom_all_reduce are not implemented"
)
def test_trtllm_custom_all_reduce_support_checks():
    assert hasattr(trtllm_custom_all_reduce, "is_compute_capability_supported")
    assert hasattr(trtllm_custom_all_reduce, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_moe_allreduce_fusion are not implemented"
)
def test_trtllm_moe_allreduce_fusion_support_checks():
    assert hasattr(trtllm_moe_allreduce_fusion, "is_compute_capability_supported")
    assert hasattr(trtllm_moe_allreduce_fusion, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_moe_finalize_allreduce_fusion are not implemented"
)
def test_trtllm_moe_finalize_allreduce_fusion_support_checks():
    assert hasattr(
        trtllm_moe_finalize_allreduce_fusion, "is_compute_capability_supported"
    )
    assert hasattr(trtllm_moe_finalize_allreduce_fusion, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_create_ipc_workspace_for_all_reduce are not implemented"
)
def test_trtllm_create_ipc_workspace_for_all_reduce_support_checks():
    assert hasattr(
        trtllm_create_ipc_workspace_for_all_reduce, "is_compute_capability_supported"
    )
    assert hasattr(trtllm_create_ipc_workspace_for_all_reduce, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_destroy_ipc_workspace_for_all_reduce are not implemented"
)
def test_trtllm_destroy_ipc_workspace_for_all_reduce_support_checks():
    assert hasattr(
        trtllm_destroy_ipc_workspace_for_all_reduce, "is_compute_capability_supported"
    )
    assert hasattr(trtllm_destroy_ipc_workspace_for_all_reduce, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_create_ipc_workspace_for_all_reduce_fusion are not implemented"
)
def test_trtllm_create_ipc_workspace_for_all_reduce_fusion_support_checks():
    assert hasattr(
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        "is_compute_capability_supported",
    )
    assert hasattr(
        trtllm_create_ipc_workspace_for_all_reduce_fusion, "is_backend_supported"
    )


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_destroy_ipc_workspace_for_all_reduce_fusion are not implemented"
)
def test_trtllm_destroy_ipc_workspace_for_all_reduce_fusion_support_checks():
    assert hasattr(
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        "is_compute_capability_supported",
    )
    assert hasattr(
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion, "is_backend_supported"
    )


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_lamport_initialize are not implemented"
)
def test_trtllm_lamport_initialize_support_checks():
    assert hasattr(trtllm_lamport_initialize, "is_compute_capability_supported")
    assert hasattr(trtllm_lamport_initialize, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_lamport_initialize_all are not implemented"
)
def test_trtllm_lamport_initialize_all_support_checks():
    assert hasattr(trtllm_lamport_initialize_all, "is_compute_capability_supported")
    assert hasattr(trtllm_lamport_initialize_all, "is_backend_supported")


# VLLM AllReduce APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for vllm_all_reduce are not implemented"
)
def test_vllm_all_reduce_support_checks():
    assert hasattr(vllm_all_reduce, "is_compute_capability_supported")
    assert hasattr(vllm_all_reduce, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for vllm_init_custom_ar are not implemented"
)
def test_vllm_init_custom_ar_support_checks():
    assert hasattr(vllm_init_custom_ar, "is_compute_capability_supported")
    assert hasattr(vllm_init_custom_ar, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for vllm_dispose are not implemented")
def test_vllm_dispose_support_checks():
    assert hasattr(vllm_dispose, "is_compute_capability_supported")
    assert hasattr(vllm_dispose, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for vllm_register_buffer are not implemented"
)
def test_vllm_register_buffer_support_checks():
    assert hasattr(vllm_register_buffer, "is_compute_capability_supported")
    assert hasattr(vllm_register_buffer, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for vllm_register_graph_buffers are not implemented"
)
def test_vllm_register_graph_buffers_support_checks():
    assert hasattr(vllm_register_graph_buffers, "is_compute_capability_supported")
    assert hasattr(vllm_register_graph_buffers, "is_backend_supported")


# NVSHMEM APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for NVSHMEMAllReduce are not implemented"
)
def test_nvshmem_allreduce_support_checks():
    assert hasattr(NVSHMEMAllReduce, "is_compute_capability_supported")
    assert hasattr(NVSHMEMAllReduce, "is_backend_supported")


# MNNVL APIs
@pytest.mark.xfail(reason="TODO: Support checks for MnnvlMemory are not implemented")
def test_mnnvl_memory_support_checks():
    assert hasattr(MnnvlMemory, "is_compute_capability_supported")
    assert hasattr(MnnvlMemory, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for MnnvlMoe are not implemented")
def test_mnnvl_moe_support_checks():
    assert hasattr(MnnvlMoe, "is_compute_capability_supported")
    assert hasattr(MnnvlMoe, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for MoEAlltoallInfo are not implemented"
)
def test_moe_alltoall_info_support_checks():
    assert hasattr(MoEAlltoallInfo, "is_compute_capability_supported")
    assert hasattr(MoEAlltoallInfo, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for MoeAlltoAll are not implemented")
def test_moe_alltoall_support_checks():
    assert hasattr(MoeAlltoAll.dispatch, "is_compute_capability_supported")
    assert hasattr(MoeAlltoAll.dispatch, "is_backend_supported")


# TRTLLM MNNVL AllReduce APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_mnnvl_allreduce are not implemented"
)
def test_trtllm_mnnvl_allreduce_support_checks():
    assert hasattr(trtllm_mnnvl_allreduce, "is_compute_capability_supported")
    assert hasattr(trtllm_mnnvl_allreduce, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_mnnvl_fused_allreduce_add_rmsnorm are not implemented"
)
def test_trtllm_mnnvl_fused_allreduce_add_rmsnorm_support_checks():
    assert hasattr(
        trtllm_mnnvl_fused_allreduce_add_rmsnorm, "is_compute_capability_supported"
    )
    assert hasattr(trtllm_mnnvl_fused_allreduce_add_rmsnorm, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_mnnvl_fused_allreduce_rmsnorm are not implemented"
)
def test_trtllm_mnnvl_fused_allreduce_rmsnorm_support_checks():
    assert hasattr(
        trtllm_mnnvl_fused_allreduce_rmsnorm, "is_compute_capability_supported"
    )
    assert hasattr(trtllm_mnnvl_fused_allreduce_rmsnorm, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_mnnvl_all_reduce are not implemented"
)
def test_trtllm_mnnvl_all_reduce_support_checks():
    assert hasattr(trtllm_mnnvl_all_reduce, "is_compute_capability_supported")
    assert hasattr(trtllm_mnnvl_all_reduce, "is_backend_supported")
