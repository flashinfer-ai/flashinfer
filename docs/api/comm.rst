.. _apicomm:

flashinfer.comm
===============

.. currentmodule:: flashinfer.comm

This module provides communication primitives and utilities for distributed computing, including CUDA IPC, AllReduce operations, and memory management utilities.

CUDA IPC Utilities
------------------

.. currentmodule:: flashinfer.comm.cuda_ipc

.. autosummary::
    :toctree: ../generated

    create_shared_buffer
    free_shared_buffer

DLPack Utilities
----------------

.. currentmodule:: flashinfer.comm.dlpack_utils

.. autosummary::
    :toctree: ../generated

    pack_strided_memory

Mapping Utilities
-----------------

.. currentmodule:: flashinfer.comm.mapping

.. autosummary::
    :toctree: ../generated

    Mapping

TensorRT-LLM AllReduce
----------------------

.. currentmodule:: flashinfer.comm.trtllm_ar

Types and Enums
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    AllReduceFusionOp
    AllReduceFusionPattern
    AllReduceStrategyConfig
    AllReduceStrategyType
    FP4QuantizationSFLayout

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

.. currentmodule:: flashinfer.comm.vllm_ar

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
