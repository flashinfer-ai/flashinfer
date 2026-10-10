.. _apimamba:

flashinfer.mamba
================

Mamba / Mamba-2 state-space-model kernels. These wrap the selective scan and
state-update primitives used in SSM blocks.

.. currentmodule:: flashinfer.mamba

.. autosummary::
    :toctree: ../generated

    selective_state_update
    checkpointing_ssu
    ssd_combined_fwd
    replayssm_materialize

Prepared checkpointing dispatch
------------------------------

Serving runtimes with a validated, fixed tensor contract can resolve the JIT
kernel and batch policies after warmup/autotuning. The returned callable is a
low-level interface that bypasses Python argument validation; see the preparation
function's contract before calling it directly.

.. currentmodule:: flashinfer.mamba.checkpointing_ssu

.. autosummary::
    :toctree: ../generated

    prepare_checkpointing_ssu_runtime
    prepare_checkpointing_ssu_runtime_kernel
