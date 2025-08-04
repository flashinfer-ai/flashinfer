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