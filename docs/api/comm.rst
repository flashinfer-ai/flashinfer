.. _apicomm:

flashinfer.comm
===============

.. currentmodule:: flashinfer.comm

This module provides communication primitives and utilities for distributed computing, including CUDA IPC, AllReduce operations, and memory management utilities.

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
