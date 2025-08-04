.. _apicomm:

flashinfer.comm
===============

.. currentmodule:: flashinfer.comm

This module provides communication primitives and utilities for distributed computing, including CUDA IPC, AllReduce operations, and memory management utilities.

CUDA IPC Utilities
------------------

.. autosummary::
    :toctree: ../generated

    cuda_ipc.create_shared_buffer
    cuda_ipc.free_shared_buffer


DLPack Utilities
----------------

.. autosummary::
    :toctree: ../generated

    dlpack_utils.pack_strided_memory

Mapping Utilities
-----------------

.. autosummary::
    :toctree: ../generated

    mapping.Mapping

TensorRT-LLM AllReduce
----------------------

Types and Enums
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_ar.AllReduceFusionOp
    trtllm_ar.AllReduceFusionPattern
    trtllm_ar.AllReduceStrategyConfig
    trtllm_ar.AllReduceStrategyType
    trtllm_ar.FP4QuantizationSFLayout

Core Operations
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_allreduce_fusion
    trtllm_custom_all_reduce
    trtllm_moe_allreduce_fusion
    trtllm_moe_finalize_allreduce_fusion

Workspace Management
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_create_ipc_workspace_for_all_reduce
    trtllm_create_ipc_workspace_for_all_reduce_fusion
    trtllm_destroy_ipc_workspace_for_all_reduce
    trtllm_destroy_ipc_workspace_for_all_reduce_fusion

Initialization and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_lamport_initialize
    trtllm_lamport_initialize_all
    compute_fp4_swizzled_layout_sf_size

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
~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    MnnvlMemory
    McastGPUBuffer

Utility Functions
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    create_tensor_from_cuda_memory
    alloc_and_copy_to_cuda

TensorRT-LLM MNNVL AllReduce
----------------------------

.. currentmodule:: flashinfer.comm.trtllm_mnnvl_ar

.. autosummary::
    :toctree: ../generated

    trtllm_mnnvl_all_reduce
    trtllm_mnnvl_fused_allreduce_rmsnorm
    mpi_barrier
