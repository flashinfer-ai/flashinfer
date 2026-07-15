.. _apiparallel_attention:

flashinfer.parallel_attention
==============================

Context-parallel attention utilities. These wrap an attention backend with
Ulysses and/or Ring parallelism and provide the helpers to shard
variable-length inputs and build the required process groups.

.. currentmodule:: flashinfer.parallel_attention

Attention Wrapper
-----------------

.. autoclass:: ParallelAttention
    :members:

Parallelism Configs
-------------------

.. autoclass:: UnevenCPConfig
    :members:

.. autoclass:: VarlenCPConfig
    :members:

Sharding and Group Helpers
--------------------------

.. autosummary::
    :toctree: ../generated

    split_varlen_input
    ulysses_varlen_config
    ring_varlen_config
    uneven_cp_config
    get_parallel_groups
