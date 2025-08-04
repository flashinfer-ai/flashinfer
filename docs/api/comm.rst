.. _apicomm:

Communication (``flashinfer.comm``)
====================================

The ``flashinfer.comm`` module provides utilities for distributed computing and inter-process communication in GPU environments. It includes optimized implementations for various deep learning frameworks and communication patterns.

CUDA IPC (Inter-Process Communication)
---------------------------------------

Low-level CUDA runtime bindings and shared buffer management.

.. currentmodule:: flashinfer.comm

.. autosummary::
   :toctree: generated/
   :nosignatures:

   create_shared_buffer
   free_shared_buffer

DLPack Utilities
----------------

Utilities for memory packing using the DLPack standard.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pack_strided_memory

Mapping and Parallelism
-----------------------

Configuration utilities for tensor, context, and pipeline parallelism.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Mapping

TensorRT-LLM AllReduce
----------------------

Optimized AllReduce operations and fusion patterns for TensorRT-LLM.

Strategy Types and Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AllReduceStrategyType
   AllReduceStrategyConfig
   AllReduceFusionOp
   AllReduceFusionPattern

Quantization Layouts
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FP4QuantizationSFLayout
   compute_fp4_swizzled_layout_sf_size

Core Functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   trtllm_custom_all_reduce
   trtllm_allreduce_fusion

Workspace Management
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   trtllm_create_ipc_workspace_for_all_reduce
   trtllm_create_ipc_workspace_for_all_reduce_fusion
   trtllm_destroy_ipc_workspace_for_all_reduce
   trtllm_destroy_ipc_workspace_for_all_reduce_fusion

Lamport Clock Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   trtllm_lamport_initialize
   trtllm_lamport_initialize_all

MoE (Mixture of Experts) Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   trtllm_moe_allreduce_fusion
   trtllm_moe_finalize_allreduce_fusion

vLLM AllReduce
--------------

Optimized AllReduce operations for vLLM framework.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   vllm_init_custom_ar
   vllm_all_reduce
   vllm_dispose

Buffer Management
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   vllm_register_buffer
   vllm_register_graph_buffers
   vllm_get_graph_buffer_ipc_meta
   vllm_meta_size
