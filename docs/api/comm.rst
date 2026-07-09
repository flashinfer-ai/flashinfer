.. _apicomm:

flashinfer.comm
===============

.. currentmodule:: flashinfer.comm

This module provides communication primitives and utilities for distributed computing, including CUDA IPC, AllReduce operations, and memory management utilities.

CUDA IPC Utilities
------------------

.. autosummary::
    :toctree: ../generated

    CudaRTLibrary
    create_shared_buffer
    free_shared_buffer

DLPack Utilities
----------------

.. autosummary::
    :toctree: ../generated

    pack_strided_memory

Mapping Utilities
-----------------

.. autosummary::
    :toctree: ../generated

    Mapping

TensorRT-LLM AllReduce
----------------------

Types and Enums
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    AllReduceFusionOp
    AllReduceFusionPattern
    AllReduceStrategyConfig
    AllReduceStrategyType
    QuantizationSFLayout

Core Operations
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_allreduce_fusion
    trtllm_custom_all_reduce
    trtllm_moe_allreduce_fusion
    trtllm_moe_finalize_allreduce_fusion

Workspace Management
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_create_ipc_workspace_for_all_reduce
    trtllm_create_ipc_workspace_for_all_reduce_fusion
    trtllm_destroy_ipc_workspace_for_all_reduce
    trtllm_destroy_ipc_workspace_for_all_reduce_fusion

Initialization and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_lamport_initialize
    trtllm_lamport_initialize_all
    compute_fp4_swizzled_layout_sf_size

Unified AllReduce Fusion API
----------------------------

.. autosummary::
    :toctree: ../generated

    allreduce_fusion
    create_allreduce_fusion_workspace
    AllReduceFusionWorkspace
    TRTLLMAllReduceFusionWorkspace
    MNNVLAllReduceFusionWorkspace

vLLM AllReduce
--------------

.. autosummary::
    :toctree: ../generated

    vllm_all_reduce
    vllm_dispose
    vllm_init_custom_ar
    vllm_register_buffer
    vllm_register_graph_buffers
    vllm_get_graph_buffer_ipc_meta
    vllm_meta_size

MNNVL (Multi-Node NVLink)
-------------------------

.. currentmodule:: flashinfer.comm.mnnvl

Core Classes
~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    MnnvlMemory
    McastGPUBuffer

TensorRT-LLM MNNVL AllReduce
----------------------------

.. currentmodule:: flashinfer.comm.trtllm_mnnvl_ar

.. autosummary::
    :toctree: ../generated

    trtllm_mnnvl_all_reduce
    trtllm_mnnvl_allreduce
    trtllm_mnnvl_fused_allreduce_add_rmsnorm
    trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant
    trtllm_mnnvl_fused_allreduce_rmsnorm
    mpi_barrier

MNNVL A2A (Throughput Backend)
-------------------------------

.. currentmodule:: flashinfer.comm

.. autosummary::
    :toctree: ../generated

    moe_a2a_initialize
    moe_a2a_dispatch
    moe_a2a_combine
    moe_a2a_sanitize_expert_ids
    moe_a2a_get_workspace_size_per_rank
    moe_a2a_wrap_payload_tensor_in_workspace

.. autoclass:: MoeAlltoAll
    :members:
    :inherited-members:
    :show-inheritance:

    .. automethod:: __init__

``MoeAlltoAll`` preserves its CUDA virtual addresses across process
checkpoint/restore.  After quiescing all work, call ``checkpoint_prepare`` to
release the non-checkpointable physical MNNVL handles.  Then call
``checkpoint_restore`` with a fresh communication backend before replaying a
captured CUDA graph:

.. code-block:: python

    moe_alltoall.checkpoint_prepare()
    moe_alltoall.checkpoint_restore(comm_backend)

Both methods are collective.  Every rank must call them in the same order, and
``comm_backend`` must reproduce the original rank and world size.
Repeated calls are no-ops after the workspace reaches the requested state.
If an exception occurs after physical handle unmapping or remapping begins,
do not retry or reuse the workspace; restart the affected rank.

.. autosummary::
    :toctree: ../generated

    MoeAlltoAll.checkpoint_prepare
    MoeAlltoAll.checkpoint_restore

DCP All-to-All (Context-Parallel Attention Reduction)
-----------------------------------------------------

.. currentmodule:: flashinfer.comm

.. autosummary::
    :toctree: ../generated

    decode_cp_a2a_workspace_size
    decode_cp_a2a_allocate_mnnvl_workspace
    decode_cp_a2a_init_workspace
    decode_cp_a2a_alltoall

Mixed Communication
-------------------

.. currentmodule:: flashinfer.comm.mixed_comm

.. autosummary::
    :toctree: ../generated

    MixedCommOp
    MixedCommMode
    MixedCommHandler
    run_mixed_comm
