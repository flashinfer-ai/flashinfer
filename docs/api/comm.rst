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

CUDA graph-stable checkpoint detach/reattach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MNNVL workspaces used by MoE all-to-all kernels are CUDA VMM allocations.
For process checkpoint/restore, callers may release the non-checkpointable
physical/imported MNNVL handles while keeping the CUDA virtual address
reservations and Python workspace tensors alive.  Use
``detach_handles`` only after all kernels using the workspace have
quiesced.  After restore, use ``reattach_handles`` to recreate/export/
import handles and map them at the original virtual addresses before replaying
captured CUDA graphs.  The reattach path refreshes communicator/transport state
from the current ``MnnvlConfig`` when provided; it does not reuse stale
cross-process handles from the checkpoint.  ``reattach_handles`` maps fresh
physical handles back into the preserved graph-visible virtual addresses and
never reserves fresh graph-visible VA.

CUDA graphs remain valid only if every graph-visible workspace tensor pointer,
rank stride, rank count, rank id, and tensor layout validates unchanged.  The
MNNVL APIs fail closed when this metadata changes.  Non-workspace tensors
captured by a graph remain the caller's responsibility.

.. autosummary::
    :toctree: ../generated

    MnnvlMemory.detach_handles
    MnnvlMemory.reattach_handles
    MnnvlMemory.get_graph_visible_addresses
    MnnvlMemory.validate_graph_visible_addresses

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
