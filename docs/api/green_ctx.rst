.. _apigreenctx:

flashinfer.green_ctx
====================

.. currentmodule:: flashinfer.green_ctx

This module provides utilities for creating CUDA green contexts.

Green context creation
----------------------

.. autosummary::
    :toctree: ../generated

    split_device_green_ctx
    split_device_green_ctx_by_sm_count

uGPU localization (multi-die GPUs)
----------------------------------

On GPUs built from several dies (Rubin), each die is a locality domain with its own SMs
and memory. A kernel on a domain's green-context stream that reads memory placed in that
domain keeps its DRAM traffic off the die-to-die link.

.. autosummary::
    :toctree: ../generated

    get_locality_domain_count
    split_device_green_ctx_by_locality_domain
    empty_by_locality_domain
