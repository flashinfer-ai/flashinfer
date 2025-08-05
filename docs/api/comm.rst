.. _apicomm:

Communication (``flashinfer.comm``)
====================================

The ``flashinfer.comm`` module provides utilities for distributed computing and inter-process communication in GPU environments. It includes optimized implementations for various deep learning frameworks and communication patterns.

.. automodule:: flashinfer.comm

CUDA IPC Utilities
------------------

.. currentmodule:: flashinfer.comm.cuda_ipc

.. autosummary::
    :toctree: ../generated

    create_shared_buffer
    free_shared_buffer
